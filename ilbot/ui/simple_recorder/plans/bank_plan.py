#!/usr/bin/env python3
"""
Generic Bank Plan
================

This plan provides a simple bank setup system that can be used by other plans.
It handles:
1. Traveling to a bank
2. Opening the bank
3. Checking for required items (fails if missing)
4. Setting up equipment based on skill levels
5. Setting up inventory with required items
6. Managing food and supplies

Note: This plan does NOT handle buying missing items from GE.
If items are missing, it will return MISSING_ITEMS status.

Return Status Codes:
- 0: SUCCESS - Bank setup completed successfully
- 1: TRAVELING - Currently traveling to bank
- 2: BANKING - Currently performing bank operations
- 3: EQUIPPING - Currently equipping items
- 4: INVENTORY_SETUP - Currently setting up inventory
- 5: MISSING_ITEMS - Required items not found in bank
- 6: ERROR - An error occurred during bank operations
- 7: WAITING - Waiting for an operation to complete
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..actions import bank, inventory, player, equipment, travel
from ..actions.equipment import get_best_weapon_for_level, get_best_armor_for_level, get_best_weapon_for_level_in_bank, get_best_armor_for_level_in_bank
from ..actions.timing import wait_until
from ..helpers.bank import near_any_bank
from ..helpers.inventory import inv_slots
from .base import Plan


class BankPlan(Plan):
    """Generic bank setup plan that can be used by other plans."""
    
    id = "BANK_PLAN"
    label = "Generic Bank Setup"
    
    # Return status codes
    SUCCESS = 0
    TRAVELING = 1
    BANKING = 2
    EQUIPPING = 3
    INVENTORY_SETUP = 4
    MISSING_ITEMS = 5
    ERROR = 6
    WAITING = 7
    
    def __init__(self, 
                 bank_area: str = None,
                 equipment_config: Dict = None,
                 inventory_config: Dict = None,
                 food_item: str = "Trout",
                 food_quantity: int = 20):
        """
        Initialize the bank plan.
        
        Args:
            bank_area: The area name for the bank (e.g., "FALADOR_BANK")
            equipment_config: Configuration for equipment setup
            inventory_config: Configuration for inventory setup
            food_item: Name of food item to withdraw
            food_quantity: Quantity of food to withdraw
        """
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration
        self.bank_area = bank_area or "FALADOR_BANK"
        self.food_item = food_item
        self.food_quantity = food_quantity
        
        # Equipment configuration
        self.equipment_config = equipment_config or {
            "weapon_tiers": [
                {"name": "Bronze scimitar", "attack_req": 1},
                {"name": "Iron scimitar", "attack_req": 1},
                {"name": "Steel scimitar", "attack_req": 5},
                {"name": "Mithril scimitar", "attack_req": 20},
                {"name": "Adamant scimitar", "attack_req": 30},
                {"name": "Rune scimitar", "attack_req": 40}
            ],
            "armor_tiers": {
                "helmet": [
                    {"name": "Bronze full helm", "defence_req": 1},
                    {"name": "Iron full helm", "defence_req": 1},
                    {"name": "Steel full helm", "defence_req": 5},
                    {"name": "Mithril full helm", "defence_req": 20},
                    {"name": "Adamant full helm", "defence_req": 30},
                    {"name": "Rune full helm", "defence_req": 40}
                ],
                "body": [
                    {"name": "Bronze platebody", "defence_req": 1},
                    {"name": "Iron platebody", "defence_req": 1},
                    {"name": "Steel platebody", "defence_req": 5},
                    {"name": "Mithril platebody", "defence_req": 20},
                    {"name": "Adamant platebody", "defence_req": 30},
                    {"name": "Rune platebody", "defence_req": 40}
                ],
                "legs": [
                    {"name": "Bronze platelegs", "defence_req": 1},
                    {"name": "Iron platelegs", "defence_req": 1},
                    {"name": "Steel platelegs", "defence_req": 5},
                    {"name": "Mithril platelegs", "defence_req": 20},
                    {"name": "Adamant platelegs", "defence_req": 30},
                    {"name": "Rune platelegs", "defence_req": 40}
                ],
                "shield": [
                    {"name": "Bronze kiteshield", "defence_req": 1},
                    {"name": "Iron kiteshield", "defence_req": 1},
                    {"name": "Steel kiteshield", "defence_req": 5},
                    {"name": "Mithril kiteshield", "defence_req": 20},
                    {"name": "Adamant kiteshield", "defence_req": 30},
                    {"name": "Rune kiteshield", "defence_req": 40}
                ]
            },
            "jewelry_tiers": {
                "amulet": [
                    {"name": "Amulet of strength", "defence_req": 1}
                ]
            }
        }
        
        # Inventory configuration
        self.inventory_config = inventory_config or {
            "required_items": [],
            "optional_items": [],
            "deposit_all": True
        }
        
        # State tracking
        self.setup_complete = False
        self.error_message = None
        
        # Set up camera immediately during initialization
        try:
            from ..helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Bank plan initialized")
        logging.info(f"[{self.id}] Bank area: {self.bank_area}")
        logging.info(f"[{self.id}] Food: {self.food_item} x{self.food_quantity}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "TRAVEL_TO_BANK")
        
        try:
            match(phase):
                case "TRAVEL_TO_BANK":
                    return self._handle_travel_to_bank()
                
                case "OPEN_BANK":
                    return self._handle_open_bank()
                
                case "CHECK_ITEMS":
                    return self._handle_check_items()
                
                case "DEPOSIT_INVENTORY":
                    return self._handle_deposit_inventory()
                
                case "EQUIP_ITEMS":
                    return self._handle_equip_items()
                
                case "WITHDRAW_ITEMS":
                    return self._handle_withdraw_items()
                
                case "SETUP_COMPLETE":
                    return self._handle_setup_complete()
                
                case "ERROR_STATE":
                    return self._handle_error_state()
                
                case _:
                    logging.warning(f"[{self.id}] Unknown phase: {phase}")
                    self.set_phase("ERROR_STATE")
                    self.error_message = f"Unknown phase: {phase}"
                    return self.ERROR
        
        except Exception as e:
            logging.error(f"[{self.id}] Error in phase {phase}: {e}")
            self.set_phase("ERROR_STATE")
            self.error_message = str(e)
            return self.ERROR
    
    def _handle_travel_to_bank(self) -> int:
        """Handle traveling to the bank."""
        if not near_any_bank():
            logging.info(f"[{self.id}] Traveling to bank...")
            travel.go_to_closest_bank()
            return self.TRAVELING
        else:
            self.set_phase("OPEN_BANK")
            return self.loop_interval_ms
    
    def _handle_open_bank(self) -> int:
        """Handle opening the bank."""
        if not bank.is_open():
            logging.info(f"[{self.id}] Opening bank...")
            bank.open_bank()
            return self.BANKING
        else:
            self.set_phase("CHECK_ITEMS")
            return self.loop_interval_ms
    
    def _handle_check_items(self) -> int:
        """Handle checking what items we have and need."""
        logging.info(f"[{self.id}] Checking bank, inventory, and equipped items...")
        
        # Determine what equipment we should have based on skill levels
        target_weapon = get_best_weapon_for_level(self.equipment_config["weapon_tiers"], self.id)
        target_armor_dict = get_best_armor_for_level(self.equipment_config["armor_tiers"], self.id)
        target_jewelry_dict = get_best_armor_for_level(self.equipment_config["jewelry_tiers"], self.id)
        
        # Helper function to check if we have an item anywhere
        def has_item_anywhere(item_name):
            return (bank.has_item(item_name) or 
                    inventory.has_item(item_name) or 
                    equipment.has_equipped(item_name))
        
        # Check what we have across all locations
        has_food = has_item_anywhere(self.food_item)
        has_weapon = target_weapon and has_item_anywhere(target_weapon["name"])
        has_armor = True
        has_jewelry = True
        
        # Check armor pieces
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                if not has_item_anywhere(armor_item["name"]):
                    has_armor = False
                    break
        else:
            has_armor = False
        
        # Check jewelry pieces
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                if not has_item_anywhere(jewelry_item["name"]):
                    has_jewelry = False
                    break
        else:
            has_jewelry = False
        
        # Determine what we need to buy
        items_needed = []
        if not has_food:
            items_needed.append(self.food_item)
        if not has_weapon and target_weapon:
            items_needed.append(target_weapon["name"])
        
        # Check each armor piece individually
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                if not has_item_anywhere(armor_item["name"]):
                    items_needed.append(armor_item["name"])
        
        # Check each jewelry piece individually
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                if not has_item_anywhere(jewelry_item["name"]):
                    items_needed.append(jewelry_item["name"])
        
        # Check required inventory items
        for item in self.inventory_config.get("required_items", []):
            if not has_item_anywhere(item):
                items_needed.append(item)
        
        logging.info(f"[{self.id}] Items needed: {items_needed}")
        
        if items_needed:
            logging.warning(f"[{self.id}] Missing required items: {items_needed}")
            bank.close_bank()
            self.set_phase("ERROR_STATE")
            self.error_message = f"Missing required items: {items_needed}"
            return self.MISSING_ITEMS
        else:
            logging.info(f"[{self.id}] All required items found!")
            self.set_phase("DEPOSIT_INVENTORY")
            return self.loop_interval_ms
    
    
    def _handle_deposit_inventory(self) -> int:
        """Handle depositing inventory items."""
        if self.inventory_config.get("deposit_all", True):
            logging.info(f"[{self.id}] Depositing all inventory items...")
            bank.deposit_inventory()
            time.sleep(0.5)
        
        self.set_phase("EQUIP_ITEMS")
        return self.INVENTORY_SETUP
    
    def _handle_equip_items(self) -> int:
        """Handle equipping items."""
        logging.info(f"[{self.id}] Equipping items...")
        
        # Determine what equipment we should have
        target_weapon = get_best_weapon_for_level(self.equipment_config["weapon_tiers"], self.id)
        target_armor_dict = get_best_armor_for_level(self.equipment_config["armor_tiers"], self.id)
        target_jewelry_dict = get_best_armor_for_level(self.equipment_config["jewelry_tiers"], self.id)
        
        # Equip weapon
        if target_weapon and not equipment.has_equipped(target_weapon["name"]):
            if bank.has_item(target_weapon["name"]):
                bank.withdraw_item(target_weapon["name"])
                time.sleep(0.5)
                equipment.equip_item(target_weapon["name"])
                time.sleep(0.5)
        
        # Equip armor
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                if not equipment.has_equipped(armor_item["name"]):
                    if bank.has_item(armor_item["name"]):
                        bank.withdraw_item(armor_item["name"])
                        time.sleep(0.5)
                        equipment.equip_item(armor_item["name"])
                        time.sleep(0.5)
        
        # Equip jewelry
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                if not equipment.has_equipped(jewelry_item["name"]):
                    if bank.has_item(jewelry_item["name"]):
                        bank.withdraw_item(jewelry_item["name"])
                        time.sleep(0.5)
                        equipment.equip_item(jewelry_item["name"])
                        time.sleep(0.5)
        
        self.set_phase("WITHDRAW_ITEMS")
        return self.EQUIPPING
    
    def _handle_withdraw_items(self) -> int:
        """Handle withdrawing items for inventory."""
        logging.info(f"[{self.id}] Withdrawing items...")
        
        # Withdraw food
        if bank.has_item(self.food_item):
            bank.withdraw_item(self.food_item, self.food_quantity)
            time.sleep(0.5)
        
        # Withdraw required inventory items
        for item in self.inventory_config.get("required_items", []):
            if bank.has_item(item):
                bank.withdraw_item(item)
                time.sleep(0.5)
        
        # Withdraw optional items if there's space
        for item in self.inventory_config.get("optional_items", []):
            if not inventory.is_full() and bank.has_item(item):
                bank.withdraw_item(item)
                time.sleep(0.5)
        
        bank.close_bank()
        self.set_phase("SETUP_COMPLETE")
        return self.INVENTORY_SETUP
    
    def _handle_setup_complete(self) -> int:
        """Handle setup completion."""
        logging.info(f"[{self.id}] Bank setup completed successfully!")
        self.setup_complete = True
        return self.SUCCESS
    
    def _handle_error_state(self) -> int:
        """Handle error state."""
        logging.error(f"[{self.id}] Error: {self.error_message}")
        return self.ERROR
    
    def is_setup_complete(self) -> bool:
        """Check if bank setup is complete."""
        return self.setup_complete
    
    def get_error_message(self) -> Optional[str]:
        """Get the current error message."""
        return self.error_message
    
    def reset(self):
        """Reset the plan to initial state."""
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.setup_complete = False
        self.error_message = None
        logging.info(f"[{self.id}] Plan reset to initial state")


# Example usage function
def create_bank_plan(bank_area: str = "FALADOR_BANK", 
                    food_item: str = "Trout", 
                    food_quantity: int = 20,
                    required_items: List[str] = None) -> BankPlan:
    """
    Create a bank plan with custom configuration.
    
    Args:
        bank_area: The bank area to use
        food_item: Food item to withdraw
        food_quantity: Quantity of food to withdraw
        required_items: List of required inventory items
    
    Returns:
        Configured BankPlan instance
    """
    inventory_config = {
        "required_items": required_items or [],
        "optional_items": [],
        "deposit_all": True
    }
    
    return BankPlan(
        bank_area=bank_area,
        food_item=food_item,
        food_quantity=food_quantity,
        inventory_config=inventory_config
    )
