#!/usr/bin/env python3
"""
Simplified Bank Plan
====================

A simplified bank plan that focuses on core banking operations without complex equipment logic.
The complex equipment, tool, and buying logic should be handled by the calling plans.

HOW IT WORKS:
1. You specify what inventory items you want
2. The plan goes to a bank (closest or specified)
3. It deposits your current inventory
4. It withdraws the items you specified
5. Your character is now set up exactly as requested

USE CASES:
- Simple inventory setup for any plan
- Withdraw specific items for crafting, combat, etc.
- Deposit all and withdraw specific items

EXAMPLE:
    bank_plan = BankPlanSimple(
        required_items=[{"name": "Gold bar", "quantity": 28}]
    )
    
    # Use in your plan loop
    status = bank_plan.loop(ui)
    if status == BankPlanSimple.SUCCESS:
        print("Bank setup completed!")
    elif status == BankPlanSimple.MISSING_ITEMS:
        missing = bank_plan.get_missing_items()
        print(f"Missing items: {missing}")

Return Status Codes:
- 0: SUCCESS - Bank setup completed successfully
- 1: TRAVELING - Currently traveling to bank
- 2: BANKING - Currently performing bank operations
- 3: MISSING_ITEMS - Required items not found in bank
- 4: ERROR - An error occurred during bank operations
- 5: WAITING - Waiting for an operation to complete
"""

import logging
from typing import Dict, List, Any
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from actions import close_ge
from helpers.utils import exponential_number, sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from actions import bank, inventory, player, ge, equipment
from actions import travel
from actions import wait_until
from helpers.bank import near_any_bank
from ..base import Plan


class BankPlanSimple(Plan):
    """Simplified bank setup plan focused on core banking operations."""
    
    id = "BANK_PLAN_SIMPLE"
    label = "Simplified Bank Setup"
    description = """Simplified bank utility focused on core banking operations. Deposits inventory and withdraws specified items. Perfect for simple inventory setup without complex equipment logic. Used by many plans for basic banking needs.

Starting Area: Any bank (travels to closest bank)
Required Items: Specified in plan initialization"""
    
    # Return status codes
    SUCCESS = 0
    TRAVELING = 1
    BANKING = 2
    MISSING_ITEMS = 3
    ERROR = 4
    WAITING = 5
    
    def __init__(self, 
                 bank_area: str = None,
                 required_items: List[Dict] = None,
                 deposit_all: bool = True,
                 equip_items: Dict[str, str | List[str]] = None):
        """
        Initialize the simplified bank plan.
        
        Args:
            bank_area: The area name for the bank (e.g., "FALADOR_BANK")
            required_items: List of items to withdraw [{"name": "Item", "quantity": 1}]
            deposit_all: Whether to deposit all inventory before withdrawing
            equip_items: Dictionary of items to equip {slot: "item_name" or ["item1", "item2"]} 
                        (e.g., {"weapon": ["Bronze sword", "Iron sword"]})
        """
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration
        self.bank_area = bank_area
        self.required_items = required_items or []
        self.deposit_all = deposit_all
        self.equip_items = equip_items or {}
        
        # State tracking
        self.missing_items = []
        self.error_message = None
        
        # Set up camera immediately during initialization
        try:
            from helpers import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Simplified bank plan initialized")
        logging.info(f"[{self.id}] Bank area: {self.bank_area or 'closest bank'}")
        logging.info(f"[{self.id}] Required items: {self.required_items}")
        logging.info(f"[{self.id}] Equip items: {self.equip_items}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from helpers import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "TRAVEL_TO_BANK")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            sleep_exponential(0.3, 0.8, 1.2)
            return exponential_number(300, 800, 1.2)
        if ge.is_open():
            close_ge()
            sleep_exponential(0.2, 0.5, 1.0)
            return exponential_number(200, 500, 1.0)
        if bank.is_open() and phase == "TRAVEL_TO_BANK":
            self.set_phase("CHECK_ITEMS")
            phase = "CHECK_ITEMS"

        if not bank.is_open() and not phase == "SETUP_COMPLETE" and not phase == "TRAVEL_TO_BANK" and not phase == "OPEN_BANK":
            bank.open_bank()
            sleep_exponential(0.3, 0.7, 1.2)
            return exponential_number(300, 700, 1.2)


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

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return exponential_number(300, 800, 1.2)
    
    def _handle_travel_to_bank(self) -> int:
        """Handle traveling to the bank."""
        # Check if we're near a bank in the destination area
        destination_area = self.bank_area if self.bank_area else None
        if not near_any_bank(destination_area):
            if self.bank_area:
                logging.info(f"[{self.id}] Traveling to {self.bank_area}...")
                travel.go_to(self.bank_area)
            else:
                logging.info(f"[{self.id}] Traveling to closest bank...")
                travel.go_to_closest_bank()
            sleep_exponential(0.5, 1.5, 1.3)
            return exponential_number(500, 1500, 1.3)
        
        logging.info(f"[{self.id}] Near bank, opening...")
        self.set_phase("OPEN_BANK")
        return exponential_number(200, 500, 1.0)
    
    def _handle_open_bank(self) -> int:
        """Handle opening the bank."""
        if not bank.is_open():
            bank.open_bank()
            sleep_exponential(0.3, 0.7, 1.2)
            return exponential_number(300, 700, 1.2)
        
        logging.info(f"[{self.id}] Bank opened, checking items...")
        self.set_phase("CHECK_ITEMS")
        return exponential_number(200, 500, 1.0)
    
    def _handle_check_items(self) -> int:
        """Handle checking what items we have and need."""
        logging.info(f"[{self.id}] Checking required items...")
        
        # Helper function to check if we have an item anywhere
        def has_item_anywhere(item_name):
            return (bank.has_item(item_name) or
                    inventory.has_item(item_name) or
                    equipment.has_equipped(item_name))
        
        # Check each required item
        missing_items = []
        for item in self.required_items:
            if isinstance(item, dict):
                item_name = item.get("name")
                required_quantity = item.get("quantity", 1)
                if required_quantity == -1:
                    required_quantity = 1
                
                # Check how many we have total across bank, inventory, and equipped
                total_quantity = 0
                total_quantity += bank.get_item_count(item_name)
                total_quantity += inventory.inv_count(item_name)

                if equipment.has_equipped(item_name):
                    total_quantity += 1  # Count equipped item as 1
                
                if total_quantity < required_quantity:
                    needed_qty = required_quantity - total_quantity
                    missing_items.append({"name": item_name, "quantity": needed_qty})
            else:
                # Simple string item (default quantity 1)
                if not has_item_anywhere(item):
                    missing_items.append({"name": item, "quantity": 1})
        
        # Check equipment items (only check the first item in each slot)
        for slot, items in self.equip_items.items():
            # Convert single item to list for consistent handling
            if isinstance(items, str):
                items = [items]
            
            # Only check the first item (index 0) in the list
            if items:  # Make sure the list is not empty
                first_item = items[0]
                if not has_item_anywhere(first_item):
                    missing_items.append({"name": first_item, "quantity": 1})
                    logging.warning(f"[{self.id}] Missing primary equipment for {slot} slot: {first_item}")
        
        if missing_items:
            logging.warning(f"[{self.id}] Missing required items: {missing_items}")
            self.missing_items = missing_items
            self.set_phase("MISSING_ITEMS")
            return self.MISSING_ITEMS
        
        logging.info(f"[{self.id}] All required items available, proceeding with setup...")
        self.set_phase("DEPOSIT_INVENTORY")
        return exponential_number(100, 300, 1.0)
    
    def _handle_deposit_inventory(self) -> int:
        """Handle depositing inventory."""
        if self.deposit_all and not inventory.is_empty():
            logging.info(f"[{self.id}] Depositing inventory...")
            bank.deposit_inventory()
            sleep_exponential(0.3, 0.7, 1.2)
            if not wait_until(inventory.is_empty, max_wait_ms=3000):
                logging.error(f"[{self.id}] Failed to deposit inventory")
                self.error_message = "Failed to deposit inventory"
                return self.ERROR
            logging.info(f"[{self.id}] Inventory deposited successfully")
        
        # Check if we need to equip items first
        if self.equip_items:
            self.set_phase("EQUIP_ITEMS")
        else:
            self.set_phase("WITHDRAW_ITEMS")
        return exponential_number(200, 500, 1.0)
    
    def _handle_withdraw_items(self) -> int:
        """Handle withdrawing required items."""
        logging.info(f"[{self.id}] Withdrawing required items...")

        if not inventory.is_empty():
            bank.deposit_inventory()
            sleep_exponential(0.3, 0.7, 1.2)
            if not wait_until(lambda: inventory.is_empty, min_wait_ms=exponential_number(0.3, 0.6, 1)):
                return self.ERROR
        
        for item in self.required_items:
            if isinstance(item, dict):
                item_name = item.get("name")
                quantity = item.get("quantity", 1)
                
                if quantity == -1:  # Withdraw all
                    bank.withdraw_item(item_name, withdraw_all=True)
                    sleep_exponential(0.3, 0.7, 1.2)
                    if not wait_until(lambda: inventory.has_item(item_name), min_wait_ms=exponential_number(0.3, 0.8, 1)):
                        return self.ERROR
                else:
                    bank.withdraw_item(item_name, quantity)
                    sleep_exponential(0.3, 0.7, 1.2)
                    if not wait_until(lambda: inventory.has_item(item_name, quantity), min_wait_ms=exponential_number(0.3, 0.8, 1)):
                        return self.ERROR
                
                logging.info(f"[{self.id}] Withdrew {quantity} {item_name}")
            else:
                # Simple string item (default quantity 1)
                bank.withdraw_item(item, 1)
                sleep_exponential(0.3, 0.7, 1.2)
                logging.info(f"[{self.id}] Withdrew 1 {item}")
        
        # Verify that we have the expected items in inventory
        logging.info(f"[{self.id}] Verifying inventory contains expected items...")
        missing_items = []
        
        for item in self.required_items:
            if isinstance(item, dict):
                item_name = item.get("name")
                expected_quantity = item.get("quantity", 1)
                
                if expected_quantity == -1:  # Withdraw all - just check if we have any
                    if not inventory.has_item(item_name):
                        missing_items.append(f"{item_name} (any quantity)")
                else:
                    actual_quantity = inventory.inv_count(item_name)
                    if actual_quantity < expected_quantity:
                        missing_items.append(f"{item_name} (expected {expected_quantity}, got {actual_quantity})")
            else:
                # Simple string item (default quantity 1)
                if not inventory.has_item(item):
                    missing_items.append(f"{item} (1)")
        
        if missing_items:
            logging.info(f"[{self.id}] Inventory verification: missing items {missing_items}, retrying withdrawal...")
            return self.BANKING
        
        logging.info(f"[{self.id}] Inventory verification successful - all required items present")
        self.set_phase("SETUP_COMPLETE")
        return self.SUCCESS
    
    def _handle_equip_items(self) -> int:
        """Handle equipping items."""
        logging.info(f"[{self.id}] Equipping items...")
        
        for slot, items in self.equip_items.items():
            # Convert single item to list for consistent handling
            if isinstance(items, str):
                items = [items]
            
            # Try each item in the list until we find one that's available
            equipped = False
            for item_name in items:
                if equipment.has_equipped(item_name):
                    equipped = True
                    break
                if inventory.has_item(item_name) or bank.has_item(item_name):
                    # If it's in bank but not inventory, withdraw it first
                    if bank.has_item(item_name) and not inventory.has_item(item_name):
                        logging.info(f"[{self.id}] Withdrawing {item_name} from bank...")
                        bank.withdraw_item(item_name, 1)
                        sleep_exponential(0.3, 0.7, 1.2)
                        if not wait_until(lambda: inventory.has_item(item_name), min_wait_ms=exponential_number(0.6, 1.2, 1)):
                            return self.ERROR
                        logging.info(f"[{self.id}] Equipping {item_name} to {slot} slot...")
                        bank.equip_item(item_name, slot)
                        sleep_exponential(0.3, 0.7, 1.2)
                        if not wait_until(lambda: equipment.has_equipped(item_name),
                                          min_wait_ms=exponential_number(0.3, 0.8, 1)):
                            return self.ERROR
                        equipped = True
                        break
                    
                    # Equip the item
                    elif inventory.has_item(item_name):
                        logging.info(f"[{self.id}] Equipping {item_name} to {slot} slot...")
                        bank.equip_item(item_name, slot)
                        sleep_exponential(0.3, 0.7, 1.2)
                        if not wait_until(lambda: equipment.has_equipped(item_name), min_wait_ms=exponential_number(0.3, 0.8, 1)):
                            return self.ERROR
                        equipped = True
                        break
                    else:
                        logging.info(f"[{self.id}] {item_name} already equipped in {slot} slot")
            
            if not equipped:
                logging.warning(f"[{self.id}] No available items for {slot} slot: {items}")

        bank.deposit_inventory()
        sleep_exponential(0.3, 0.7, 1.2)
        if not wait_until(inventory.is_empty, min_wait_ms=exponential_number(0.3, 0.8, 1.5)):
            return exponential_number(300, 800, 1.2)
        self.set_phase("WITHDRAW_ITEMS")
        return exponential_number(200, 500, 1.0)
    
    def _handle_setup_complete(self) -> int:
        """Handle setup completion."""
        logging.info(f"[{self.id}] Bank setup completed successfully!")
        if bank.is_open():
            bank.close_bank()
            sleep_exponential(0.2, 0.5, 1.0)
            if not wait_until(bank.is_closed, max_wait_ms=3000):
                return exponential_number(200, 500, 1.0)
        return self.SUCCESS
    
    def get_missing_items(self) -> List[Dict[str, Any]]:
        """Get the list of missing items with quantities."""
        return self.missing_items
    
    def get_error_message(self) -> str:
        """Get the error message if any."""
        return self.error_message or "Unknown error"
    
    def reset(self):
        """Reset the plan to start fresh."""
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.missing_items = []
        self.error_message = None
        logging.info(f"[{self.id}] Plan reset")



