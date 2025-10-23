#!/usr/bin/env python3
"""
Generic Bank Plan
================

This plan is a utility that sets up your character with a specific inventory and equipment loadout.

HOW IT WORKS:
1. You specify what inventory items and equipment you want
2. The plan goes to a bank (closest or specified)
3. It deposits your current inventory
4. It equips the best available equipment based on your skill levels
5. It withdraws the food and items you specified
6. Your character is now set up exactly as requested

USE CASES:
- Set up for combat: food + combat gear + teleports
- Set up for skilling: tools + materials + food
- Set up for questing: quest items + food + teleports
- Any custom loadout you want

EXAMPLE:
    bank_plan = setup_character_loadout(
        food_item="Trout",
        food_quantity=25,
        inventory_items=["Teleport to house", "Coins", "Rope"]
    )
    
    # Use in your plan loop
    status = bank_plan.loop(ui)
    if status == BankPlan.SUCCESS:
        print("Character ready!")

Note: This plan does NOT buy missing items from GE.
If items are missing, it returns MISSING_ITEMS status.

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

from ...actions.ge import close_ge

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ...actions import bank, inventory, player, equipment, travel, ge
from ...actions.equipment import get_best_weapon_for_level, get_best_armor_for_level, get_best_weapon_for_level_in_bank, get_best_armor_for_level_in_bank
from ...actions.timing import wait_until
from ...helpers.bank import near_any_bank
from ...helpers.inventory import inv_slots
from ..base import Plan


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
    ITEMS_TO_SELL = 8
    
    def __init__(self, 
                 bank_area: str = None,
                 equipment_config: Dict = None,
                 inventory_config: Dict = None,
                 food_item: str = None,
                 food_quantity: int = 0,
                 sellable_items: Dict = None):
        """
        Initialize the bank plan.
        
        Args:
            bank_area: The area name for the bank (e.g., "FALADOR_BANK")
            equipment_config: Configuration for equipment setup
            inventory_config: Configuration for inventory setup
            food_item: Name of food item to withdraw (None for no food)
            food_quantity: Quantity of food to withdraw (0 for no food)
            sellable_items: Dict of items to check for selling {item_name: min_quantity}
        """
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration
        self.bank_area = "CLOSEST_BANK"  # Keep as None to use closest bank
        self.food_item = food_item
        self.food_quantity = food_quantity
        
        # Equipment configuration
        self.equipment_config = equipment_config or {
            "weapon_tiers": [],
            "armor_tiers": {},
            "jewelry_tiers": {}
        }
        
        # Inventory configuration
        self.inventory_config = inventory_config or {
            "required_items": [],
            "optional_items": [],
            "deposit_all": True
        }
        
        # Sellable items configuration
        self.sellable_items = sellable_items or {}
        
        # State tracking
        self.setup_complete = False
        self.error_message = None
        self.fallback_items = {}  # Store fallback items found during check
        self.items_to_sell = {}  # Store items that can be sold
        self.should_sell_for_equipment = False  # Whether we need to sell for better equipment
        self.target_equipment = None  # The equipment we want to buy
        
        # Set up camera immediately during initialization
        try:
            from ...helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Bank plan initialized")
        logging.info(f"[{self.id}] Bank area: {self.bank_area or 'closest bank'}")
        logging.info(f"[{self.id}] Food: {self.food_item} x{self.food_quantity}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ...helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "TRAVEL_TO_BANK")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms
        if ge.is_open():
            close_ge()
            return

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

            case "MISSING_ITEMS":
                return self._handle_missing_items()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
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
        
        # Determine what tools we should have (e.g., axes for woodcutting)
        target_tool = None
        if "tool_tiers" in self.equipment_config and self.equipment_config["tool_tiers"]:
            try:
                from ...actions.equipment import get_best_tool_for_level
                target_tool, _, _, _ = get_best_tool_for_level(self.equipment_config["tool_tiers"], "woodcutting", self.id)
            except Exception as e:
                logging.warning(f"[{self.id}] Could not determine best tool: {e}")
                target_tool = None
        
        # Helper function to check if we have an item anywhere
        def has_item_anywhere(item_name):
            return (bank.has_item(item_name) or 
                    inventory.has_item(item_name) or 
                    equipment.has_equipped(item_name))
        
        # Check what we have across all locations
        has_food = has_item_anywhere(self.food_item)
        has_weapon = target_weapon and has_item_anywhere(target_weapon["name"])
        has_tool = target_tool and has_item_anywhere(target_tool)
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
        
        # Get available coins for affordability checking
        coin_count = 0
        try:
            coin_count = bank.get_item_count("Coins")
            logging.info(f"[{self.id}] Found {coin_count} coins in bank")
        except Exception as e:
            logging.warning(f"[{self.id}] Could not get coin count from bank: {e}")
            coin_count = 0
        
        # Determine what we need to buy (with fallback logic and coin checking)
        items_needed = []
        fallback_items = {}
        
        # Helper function to check if we can afford an item
        def can_afford_item(item_name, estimated_cost=1000):
            """Check if we can afford an item (rough estimation)."""
            return coin_count >= estimated_cost
        
        # Check food
        if not has_food and self.food_item:
            items_needed.append(self.food_item)
        
        # Check weapon
        if not has_weapon and target_weapon:
            items_needed.append(target_weapon["name"])
        
        # Check tool with fallback logic and coin checking
        if not has_tool and target_tool:
            fallback_tool = self._find_fallback_tool(target_tool)
            if fallback_tool:
                logging.info(f"[{self.id}] Found fallback tool: {fallback_tool} instead of {target_tool}")
                fallback_items[target_tool] = fallback_tool
                # Only add to missing items if we can afford the best tool
                if can_afford_item(target_tool, 5000):  # Estimated cost for tools
                    items_needed.append(target_tool)
                else:
                    logging.info(f"[{self.id}] Cannot afford {target_tool}, will use fallback {fallback_tool}")
            else:
                items_needed.append(target_tool)
        
        # Check armor pieces with fallback logic and coin checking
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                if not has_item_anywhere(armor_item["name"]):
                    fallback_armor = self._find_fallback_armor(armor_item["name"])
                    if fallback_armor:
                        logging.info(f"[{self.id}] Found fallback armor: {fallback_armor} instead of {armor_item['name']}")
                        fallback_items[armor_item["name"]] = fallback_armor
                        # Only add to missing items if we can afford the best armor
                        if can_afford_item(armor_item["name"], 10000):  # Estimated cost for armor
                            items_needed.append(armor_item["name"])
                        else:
                            logging.info(f"[{self.id}] Cannot afford {armor_item['name']}, will use fallback {fallback_armor}")
                    else:
                        items_needed.append(armor_item["name"])
        
        # Check jewelry pieces with fallback logic and coin checking
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                if not has_item_anywhere(jewelry_item["name"]):
                    fallback_jewelry = self._find_fallback_jewelry(jewelry_item["name"])
                    if fallback_jewelry:
                        logging.info(f"[{self.id}] Found fallback jewelry: {fallback_jewelry} instead of {jewelry_item['name']}")
                        fallback_items[jewelry_item["name"]] = fallback_jewelry
                        # Only add to missing items if we can afford the best jewelry
                        if can_afford_item(jewelry_item["name"], 5000):  # Estimated cost for jewelry
                            items_needed.append(jewelry_item["name"])
                        else:
                            logging.info(f"[{self.id}] Cannot afford {jewelry_item['name']}, will use fallback {fallback_jewelry}")
                    else:
                        items_needed.append(jewelry_item["name"])
        
        # Check required inventory items
        for item in self.inventory_config.get("required_items", []):
            if isinstance(item, dict):
                # Item with quantity specified
                item_name = item.get("name")
                required_quantity = item.get("quantity", 1)
                
                # Check how many we have total across all locations
                total_quantity = 0
                if bank.has_item(item_name):
                    total_quantity += bank.get_item_count(item_name)
                if inventory.has_item(item_name):
                    total_quantity += inventory.inv_count(item_name)
                if equipment.has_equipped(item_name):
                    total_quantity += 1  # Equipment counts as 1
                
                if total_quantity < required_quantity:
                    needed_qty = required_quantity - total_quantity
                    items_needed.append(f"{item_name}|{needed_qty}")
            else:
                # Simple string item (default quantity 1)
                if not has_item_anywhere(item):
                    items_needed.append(f"{item}|1")
        
        # Store fallback items for later use
        self.fallback_items = fallback_items
        
        logging.info(f"[{self.id}] Items needed: {items_needed}")
        if fallback_items:
            logging.info(f"[{self.id}] Fallback items: {fallback_items}")
        
        if items_needed:
            logging.warning(f"[{self.id}] Missing required items: {items_needed}")
            bank.close_bank()
            self.set_phase("MISSING_ITEMS")
            self.error_message = f"Missing required items: {items_needed}"
            return self.MISSING_ITEMS
        
        logging.info(f"[{self.id}] All required items found!")
        self.set_phase("DEPOSIT_INVENTORY")
        return self.loop_interval_ms
    
    def _handle_missing_items(self) -> int:
        """Handle missing items state."""
        logging.warning(f"[{self.id}] Missing required items: {self.error_message}")
        logging.warning(f"[{self.id}] Bank plan cannot continue without required items")
        logging.warning(f"[{self.id}] You need to obtain the missing items before banking can proceed")
        
        # Stay in this phase and keep returning MISSING_ITEMS status
        # The calling plan (falador_cows_2) will handle what to do next
        return self.MISSING_ITEMS
    
    def _find_fallback_tool(self, target_tool: str) -> str:
        """Find a fallback tool that we already have."""
        try:
            from ...actions import bank, inventory, equipment
            
            # Define tool tiers (from worst to best)
            tool_tiers = {
                "axe": ["Bronze axe", "Iron axe", "Steel axe", "Black axe", "Mithril axe", "Adamant axe", "Rune axe", "Dragon axe"],
                "pickaxe": ["Bronze pickaxe", "Iron pickaxe", "Steel pickaxe", "Black pickaxe", "Mithril pickaxe", "Adamant pickaxe", "Rune pickaxe", "Dragon pickaxe"]
            }
            
            # Find the category for this tool
            tool_lower = target_tool.lower()
            category = None
            for cat, tools in tool_tiers.items():
                if any(tool.lower() in tool_lower for tool in tools):
                    category = cat
                    break
            
            if not category:
                return None
            
            # Helper function to check if tool exists in any location
            def has_tool_anywhere(tool_name):
                return (bank.has_item(tool_name) or 
                        inventory.has_item(tool_name) or 
                        equipment.has_equipped(tool_name))
            
            # Find the best tool we already have in this category
            target_tools = tool_tiers[category]
            current_index = -1
            
            # Find current tool index
            for i, tier_tool in enumerate(target_tools):
                if tier_tool.lower() in tool_lower:
                    current_index = i
                    break
            
            if current_index == -1:
                return None
            
            # Check if we have any lower-tier tools in any location
            for i in range(current_index):
                fallback_tool = target_tools[i]
                if has_tool_anywhere(fallback_tool):
                    # Check which location has the tool
                    location = "unknown"
                    if bank.has_item(fallback_tool):
                        location = "bank"
                    elif inventory.has_item(fallback_tool):
                        location = "inventory"
                    elif equipment.has_equipped(fallback_tool):
                        location = "equipment"
                    
                    logging.info(f"[{self.id}] Found fallback tool {fallback_tool} in {location}")
                    return fallback_tool
            
            return None
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding fallback tool: {e}")
            return None
    
    def _find_fallback_armor(self, target_armor: str) -> str:
        """Find a fallback armor piece that we already have."""
        try:
            from ...actions import bank, inventory, equipment
            
            # Define armor tiers (from worst to best)
            armor_tiers = {
                "helmet": ["Bronze full helm", "Iron full helm", "Steel full helm", "Black full helm", "Mithril full helm", "Adamant full helm", "Rune full helm"],
                "platebody": ["Bronze platebody", "Iron platebody", "Steel platebody", "Black platebody", "Mithril platebody", "Adamant platebody", "Rune platebody"],
                "platelegs": ["Bronze platelegs", "Iron platelegs", "Steel platelegs", "Black platelegs", "Mithril platelegs", "Adamant platelegs", "Rune platelegs"],
                "kiteshield": ["Bronze kiteshield", "Iron kiteshield", "Steel kiteshield", "Black kiteshield", "Mithril kiteshield", "Adamant kiteshield", "Rune kiteshield"]
            }
            
            # Find the category for this armor
            armor_lower = target_armor.lower()
            category = None
            for cat, armors in armor_tiers.items():
                if any(armor.lower() in armor_lower for armor in armors):
                    category = cat
                    break
            
            if not category:
                return None
            
            # Helper function to check if armor exists in any location
            def has_armor_anywhere(armor_name):
                return (bank.has_item(armor_name) or 
                        inventory.has_item(armor_name) or 
                        equipment.has_equipped(armor_name))
            
            # Find the best armor we already have in this category
            target_armors = armor_tiers[category]
            current_index = -1
            
            # Find current armor index
            for i, tier_armor in enumerate(target_armors):
                if tier_armor.lower() in armor_lower:
                    current_index = i
                    break
            
            if current_index == -1:
                return None
            
            # Check if we have any lower-tier armor in any location
            for i in range(current_index):
                fallback_armor = target_armors[i]
                if has_armor_anywhere(fallback_armor):
                    # Check which location has the armor
                    location = "unknown"
                    if bank.has_item(fallback_armor):
                        location = "bank"
                    elif inventory.has_item(fallback_armor):
                        location = "inventory"
                    elif equipment.has_equipped(fallback_armor):
                        location = "equipment"
                    
                    logging.info(f"[{self.id}] Found fallback armor {fallback_armor} in {location}")
                    return fallback_armor
            
            return None
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding fallback armor: {e}")
            return None
    
    def _find_fallback_jewelry(self, target_jewelry: str) -> str:
        """Find a fallback jewelry piece that we already have."""
        try:
            from ...actions import bank, inventory, equipment
            
            # Define jewelry tiers (from worst to best)
            jewelry_tiers = {
                "amulet": ["Amulet of strength", "Amulet of power", "Amulet of glory"],
                "ring": ["Ring of life", "Ring of wealth", "Berserker ring"],
                "cape": ["Cape", "Team cape", "Obsidian cape"]
            }
            
            # Find the category for this jewelry
            jewelry_lower = target_jewelry.lower()
            category = None
            for cat, jewelries in jewelry_tiers.items():
                if any(jewelry.lower() in jewelry_lower for jewelry in jewelries):
                    category = cat
                    break
            
            if not category:
                return None
            
            # Helper function to check if jewelry exists in any location
            def has_jewelry_anywhere(jewelry_name):
                return (bank.has_item(jewelry_name) or 
                        inventory.has_item(jewelry_name) or 
                        equipment.has_equipped(jewelry_name))
            
            # Find the best jewelry we already have in this category
            target_jewelries = jewelry_tiers[category]
            current_index = -1
            
            # Find current jewelry index
            for i, tier_jewelry in enumerate(target_jewelries):
                if tier_jewelry.lower() in jewelry_lower:
                    current_index = i
                    break
            
            if current_index == -1:
                return None
            
            # Check if we have any lower-tier jewelry in any location
            for i in range(current_index):
                fallback_jewelry = target_jewelries[i]
                if has_jewelry_anywhere(fallback_jewelry):
                    # Check which location has the jewelry
                    location = "unknown"
                    if bank.has_item(fallback_jewelry):
                        location = "bank"
                    elif inventory.has_item(fallback_jewelry):
                        location = "inventory"
                    elif equipment.has_equipped(fallback_jewelry):
                        location = "equipment"
                    
                    logging.info(f"[{self.id}] Found fallback jewelry {fallback_jewelry} in {location}")
                    return fallback_jewelry
            
            return None
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding fallback jewelry: {e}")
            return None
    
    def _handle_deposit_inventory(self) -> int:
        """Handle depositing inventory items and unwanted equipped items."""
        if self.inventory_config.get("deposit_all", True):
            logging.info(f"[{self.id}] Depositing all inventory items...")
            bank.deposit_inventory()
            time.sleep(0.5)
        
        # Deposit equipped items that aren't in our loadout
        logging.info(f"[{self.id}] Checking for unwanted equipped items...")
        
        # Get what we should have equipped
        target_weapon = get_best_weapon_for_level(self.equipment_config["weapon_tiers"], self.id)
        target_armor_dict = get_best_armor_for_level(self.equipment_config["armor_tiers"], self.id)
        target_jewelry_dict = get_best_armor_for_level(self.equipment_config["jewelry_tiers"], self.id)
        
        # Get what tools we should have
        target_tool = None
        if "tool_tiers" in self.equipment_config and self.equipment_config["tool_tiers"]:
            try:
                from ...actions.equipment import get_best_tool_for_level
                target_tool, _, _, _ = get_best_tool_for_level(self.equipment_config["tool_tiers"], "woodcutting", self.id)
            except Exception as e:
                logging.warning(f"[{self.id}] Could not determine best tool: {e}")
                target_tool = None
        
        # Create set of items we should keep equipped
        keep_equipped = set()
        if target_weapon:
            keep_equipped.add(target_weapon["name"])
        if target_tool:
            keep_equipped.add(target_tool)
        if target_armor_dict:
            for armor_item in target_armor_dict.values():
                keep_equipped.add(armor_item["name"])
        if target_jewelry_dict:
            for jewelry_item in target_jewelry_dict.values():
                keep_equipped.add(jewelry_item["name"])
        
        # Check if we have any unwanted equipped items
        try:
            equipped_item_names = equipment.get_equipped_item_names()
            has_unwanted_equipment = False
            for item_name in equipped_item_names:
                if item_name and item_name not in keep_equipped:
                    has_unwanted_equipment = True
                    break
            
            # If we have unwanted equipment, deposit ALL equipment at once
            if has_unwanted_equipment:
                logging.info(f"[{self.id}] Found unwanted equipment, depositing all equipment")
                bank.deposit_equipment()
                time.sleep(0.5)  # Give time for equipment to be deposited
        except Exception as e:
            logging.warning(f"[{self.id}] Could not check/deposit equipped items: {e}")
        
        self.set_phase("EQUIP_ITEMS")
        return self.INVENTORY_SETUP
    
    def _handle_equip_items(self) -> int:
        """Handle equipping items."""
        logging.info(f"[{self.id}] Equipping items...")
        
        # Determine what equipment we should have
        target_weapon = get_best_weapon_for_level(self.equipment_config["weapon_tiers"], self.id)
        target_armor_dict = get_best_armor_for_level(self.equipment_config["armor_tiers"], self.id)
        target_jewelry_dict = get_best_armor_for_level(self.equipment_config["jewelry_tiers"], self.id)
        
        # Determine what tools we should have (use fallback if available)
        target_tool = None
        if "tool_tiers" in self.equipment_config and self.equipment_config["tool_tiers"]:
            try:
                from ...actions.equipment import get_best_tool_for_level
                best_tool, _, _, _ = get_best_tool_for_level(self.equipment_config["tool_tiers"], "woodcutting", self.id)
                # Use fallback if available, otherwise use the best tool
                target_tool = self.fallback_items.get(best_tool, best_tool)
            except Exception as e:
                logging.warning(f"[{self.id}] Could not determine best tool: {e}")
                target_tool = None
        
        # Equip weapon
        if target_weapon and not equipment.has_equipped(target_weapon["name"]):
            if bank.has_item(target_weapon["name"]):
                bank.withdraw_item(target_weapon["name"])
                if not wait_until(lambda: inventory.has_item(target_weapon["name"]), min_wait_ms=1200, max_wait_ms=3000):
                    return None
                bank.interact_inventory(target_weapon["name"], "Wield")
                if not wait_until(lambda: equipment.has_equipped(target_weapon["name"]), min_wait_ms=1200, max_wait_ms=3000):
                    return None
        
        # Equip tool (e.g., axe for woodcutting)
        if target_tool:
            # Check if we have a different tool equipped and need to swap
            try:
                equipped_item_names = equipment.get_equipped_item_names()
                has_different_tool = False
                for equipped_item_name in equipped_item_names:
                    if equipped_item_name and equipped_item_name != target_tool:
                        # Check if this equipped item is a tool from our tool_tiers
                        for tool_name, _, _, _ in self.equipment_config.get("tool_tiers", []):
                            if equipped_item_name == tool_name:
                                has_different_tool = True
                                break
                        if has_different_tool:
                            break
                
                # If we have a different tool equipped, deposit ALL equipment to clear it
                if has_different_tool:
                    logging.info(f"[{self.id}] Found different tool equipped, depositing all equipment to clear it")
                    bank.deposit_equipment()
                    time.sleep(0.5)  # Give time for equipment to be deposited
            except Exception as e:
                logging.warning(f"[{self.id}] Could not check/unequip existing tools: {e}")
            
            # Now equip the target tool if we don't already have it equipped
            if not equipment.has_equipped(target_tool) and bank.has_item(target_tool):
                bank.withdraw_item(target_tool)
                wait_until(lambda: inventory.has_item(target_tool), min_wait_ms=600, max_wait_ms=3000)
                
                # Check if we can equip this tool based on attack level
                try:
                    from ...actions.player import get_skill_level
                    attack_level = get_skill_level("attack")
                    if attack_level is None:
                        attack_level = 1
                    
                    # Find the attack requirement for this tool
                    tool_attack_req = 1  # Default
                    for tool_name, skill_req, att_req, def_req in self.equipment_config["tool_tiers"]:
                        if tool_name == target_tool:
                            tool_attack_req = att_req
                            break
                    
                    if attack_level >= tool_attack_req:
                        logging.info(f"[{self.id}] Equipping {target_tool} (attack level {attack_level} >= {tool_attack_req})")
                        bank.interact_inventory(target_tool, "Wield")
                        wait_until(lambda: equipment.has_equipped(target_tool), min_wait_ms=600, max_wait_ms=3000)
                    else:
                        logging.info(f"[{self.id}] Keeping {target_tool} in inventory (attack level {attack_level} < {tool_attack_req})")
                        # Tool stays in inventory, which is fine for woodcutting
                        
                except Exception as e:
                    logging.warning(f"[{self.id}] Could not check attack level for {target_tool}: {e}")
                    # Fallback: try to equip anyway
                    bank.interact_inventory(target_tool, "Wield")
                    wait_until(lambda: equipment.has_equipped(target_tool), min_wait_ms=600, max_wait_ms=3000)
        
        # Equip armor (use fallback if available)
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                # Use fallback if available, otherwise use the original armor
                armor_name = self.fallback_items.get(armor_item["name"], armor_item["name"])
                if not equipment.has_equipped(armor_name):
                    if bank.has_item(armor_name):
                        bank.withdraw_item(armor_name)
                        wait_until(lambda: inventory.has_item(armor_name), min_wait_ms=600, max_wait_ms=3000)
                        bank.interact_inventory(armor_name, "Wear")
                        wait_until(lambda: equipment.has_equipped(armor_name), min_wait_ms=600, max_wait_ms=3000)
        
        # Equip jewelry (use fallback if available)
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                # Use fallback if available, otherwise use the original jewelry
                jewelry_name = self.fallback_items.get(jewelry_item["name"], jewelry_item["name"])
                if not equipment.has_equipped(jewelry_name):
                    if bank.has_item(jewelry_name):
                        bank.withdraw_item(jewelry_name)
                        wait_until(lambda: inventory.has_item(jewelry_name), min_wait_ms=600, max_wait_ms=3000)
                        bank.interact_inventory(jewelry_name, "Wear")
                        wait_until(lambda: equipment.has_equipped(jewelry_name), min_wait_ms=600, max_wait_ms=3000)
        
        # Verify all items are actually equipped before moving to WITHDRAW_ITEMS phase
        all_equipped = True
        
        # Check weapon
        if target_weapon and not equipment.has_equipped(target_weapon["name"]):
            logging.warning(f"[{self.id}] Weapon {target_weapon['name']} not equipped")
            all_equipped = False
        
        # Check tool
        if target_tool and not equipment.has_equipped(target_tool):
            logging.warning(f"[{self.id}] Tool {target_tool} not equipped")
            all_equipped = False
        
        # Check armor
        if target_armor_dict:
            for armor_type, armor_item in target_armor_dict.items():
                armor_name = self.fallback_items.get(armor_item["name"], armor_item["name"])
                if not equipment.has_equipped(armor_name):
                    logging.warning(f"[{self.id}] Armor {armor_name} not equipped")
                    all_equipped = False
        
        # Check jewelry
        if target_jewelry_dict:
            for jewelry_type, jewelry_item in target_jewelry_dict.items():
                jewelry_name = self.fallback_items.get(jewelry_item["name"], jewelry_item["name"])
                if not equipment.has_equipped(jewelry_name):
                    logging.warning(f"[{self.id}] Jewelry {jewelry_name} not equipped")
                    all_equipped = False
        
        if not all_equipped:
            logging.warning(f"[{self.id}] Not all items are equipped, staying in EQUIP_ITEMS phase")
            return self.EQUIPPING
        
        self.set_phase("WITHDRAW_ITEMS")
        return self.EQUIPPING
    
    def _handle_withdraw_items(self) -> int:
        """Handle withdrawing items for inventory."""
        logging.info(f"[{self.id}] Withdrawing items...")
        
        # Don't deposit inventory here - we want to keep tools that should stay in inventory
        
        # Withdraw food
        if self.food_item and bank.has_item(self.food_item):
            bank.withdraw_item(self.food_item, self.food_quantity)
            time.sleep(0.5)
        
        # Withdraw required inventory items
        for item in self.inventory_config.get("required_items", []):
            if isinstance(item, dict):
                # Item with quantity specified
                item_name = item.get("name")
                required_quantity = item.get("quantity", 1)
                
                if bank.has_item(item_name):
                    # Check how many we already have in inventory/equipment
                    already_have = 0
                    if inventory.has_item(item_name):
                        already_have += inventory.get_item_count(item_name)
                    if equipment.has_equipped(item_name):
                        already_have += 1
                    
                    # Calculate how many we need to withdraw
                    need_to_withdraw = max(0, required_quantity - already_have)
                    available_in_bank = bank.get_item_count(item_name)
                    withdraw_quantity = min(need_to_withdraw, available_in_bank)
                    
                    if withdraw_quantity > 0:
                        logging.info(f"[{self.id}] Withdrawing {withdraw_quantity} {item_name} (need {required_quantity}, already have {already_have})")
                        bank.withdraw_item(item_name, withdraw_quantity)
                        time.sleep(0.5)
            else:
                # Simple string item (default quantity 1)
                if bank.has_item(item):
                    bank.withdraw_item(item)
                    time.sleep(0.5)
        
        # Withdraw tools that should stay in inventory (not equipped)
        if "tool_tiers" in self.equipment_config and self.equipment_config["tool_tiers"]:
            try:
                from ...actions.equipment import get_best_tool_for_level
                from ...actions.player import get_skill_level
                
                target_tool, _, _, _ = get_best_tool_for_level(self.equipment_config["tool_tiers"], "woodcutting", self.id)
                attack_level = get_skill_level("attack")
                if attack_level is None:
                    attack_level = 1
                
                # Find the attack requirement for this tool
                tool_attack_req = 1  # Default
                for tool_name, skill_req, att_req, def_req in self.equipment_config["tool_tiers"]:
                    if tool_name == target_tool:
                        tool_attack_req = att_req
                        break
                
                # If we can't equip the tool but can use it, keep it in inventory
                if target_tool and not equipment.has_equipped(target_tool) and attack_level < tool_attack_req:
                    if bank.has_item(target_tool):
                        logging.info(f"[{self.id}] Withdrawing {target_tool} to keep in inventory (can't equip due to attack level)")
                        bank.withdraw_item(target_tool)
                        time.sleep(0.5)
            except Exception as e:
                logging.warning(f"[{self.id}] Could not handle tool inventory setup: {e}")
        
        # Withdraw optional items if there's space
        for item in self.inventory_config.get("optional_items", []):
            if not inventory.is_full() and bank.has_item(item):
                bank.withdraw_item(item)
                time.sleep(0.5)
        
        # Check for sellable items before closing bank
        self._check_sellable_items()
        
        bank.close_bank()
        self.set_phase("SETUP_COMPLETE")
        return self.INVENTORY_SETUP
    
    def _check_sellable_items(self):
        """Check for sellable items in the bank and determine if we should sell them."""
        self.items_to_sell = {}
        
        for item_name, min_quantity in self.sellable_items.items():
            try:
                item_count = bank.get_item_count(item_name)
                if item_count >= min_quantity:
                    self.items_to_sell[item_name] = item_count
                    logging.info(f"[{self.id}] Found {item_count} {item_name} (min: {min_quantity}) - can be sold")
            except Exception as e:
                logging.warning(f"[{self.id}] Could not check {item_name} count: {e}")
        
        # If we have sellable items, check if we need to sell them for better equipment
        if self.items_to_sell:
            self._check_if_selling_needed()
    
    def _check_if_selling_needed(self):
        """Check if we need to sell items to afford better equipment."""
        try:
            from ...actions import player, equipment
            
            # Check if we have the best tool we can use
            if "tool_tiers" in self.equipment_config and self.equipment_config["tool_tiers"]:
                from ...actions.equipment import get_best_tool_for_level
                
                # Get current skill level (assuming woodcutting for tools)
                skill_level = player.get_skill_level("woodcutting")
                if skill_level is None:
                    skill_level = 1
                
                # Find the best tool we can use
                best_tool, _, _, _ = get_best_tool_for_level(self.equipment_config["tool_tiers"], "woodcutting", self.id)
                
                if best_tool:
                    # Check if we have the best tool
                    has_best_tool = (equipment.has_equipped(best_tool) or 
                                   bank.has_item(best_tool) or 
                                   inventory.has_item(best_tool))
                    
                    if not has_best_tool:
                        # We don't have the best tool, check if we have enough coins
                        try:
                            coin_count = bank.get_item_count("Coins")
                            # Estimate cost (this could be made configurable)
                            estimated_cost = 5000  # Default estimate
                            
                            if coin_count < estimated_cost:
                                logging.info(f"[{self.id}] Need to sell items to afford better tool: {best_tool}")
                                self.should_sell_for_equipment = True
                                self.target_equipment = best_tool
                                return
                        except Exception as e:
                            logging.warning(f"[{self.id}] Could not check coin count: {e}")
            
            # If we get here, we don't need to sell for equipment
            self.should_sell_for_equipment = False
            self.target_equipment = None
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking if selling needed: {e}")
            self.should_sell_for_equipment = False
            self.target_equipment = None
    
    def _handle_setup_complete(self) -> int:
        """Handle setup completion."""
        logging.info(f"[{self.id}] Bank setup completed successfully!")
        self.setup_complete = True
        
        # If we have items to sell and need to sell for equipment, return ITEMS_TO_SELL status
        if self.items_to_sell and self.should_sell_for_equipment:
            logging.info(f"[{self.id}] Items available to sell: {self.items_to_sell}")
            logging.info(f"[{self.id}] Need to sell for equipment: {self.target_equipment}")
            return self.ITEMS_TO_SELL
        
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
    
    def get_sellable_items(self) -> Dict[str, int]:
        """Get the items that can be sold."""
        return self.items_to_sell.copy()
    
    def get_target_equipment(self) -> Optional[str]:
        """Get the equipment we want to buy by selling items."""
        return self.target_equipment
    
    def should_sell_for_equipment(self) -> bool:
        """Check if we should sell items to buy better equipment."""
        return self.should_sell_for_equipment
    
    def reset(self):
        """Reset the plan to initial state."""
        self.state = {"phase": "TRAVEL_TO_BANK"}
        self.setup_complete = False
        self.error_message = None
        self.fallback_items = {}
        self.items_to_sell = {}
        self.should_sell_for_equipment = False
        self.target_equipment = None
        logging.info(f"[{self.id}] Plan reset to initial state")


# Helper functions for easy setup
def create_bank_plan(bank_area: str = None, 
                    food_item: str = "Trout", 
                    food_quantity: int = 20,
                    required_items: List[str] = None) -> BankPlan:
    """
    Create a bank plan with custom configuration.
    
    Args:
        bank_area: The bank area to use (or None for closest bank)
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


def setup_character_loadout(bank_area: str = None,
                          food_item: str = "Trout",
                          food_quantity: int = 20,
                          inventory_items: List[str] = None,
                          equipment_tiers: Dict = None) -> BankPlan:
    """
    Create a bank plan to set up your character with a specific loadout.
    
    This is the main function you'll use to configure your character's setup.
    
    Args:
        bank_area: Bank area to use (None = use closest bank)
        food_item: Food item to withdraw
        food_quantity: How much food to withdraw
        inventory_items: List of items to have in inventory
        equipment_tiers: Custom equipment configuration (optional)
    
    Returns:
        BankPlan instance ready to use
    
    Example:
        # Simple combat setup
        bank_plan = setup_character_loadout(
            bank_area="FALADOR_BANK",
            food_item="Trout",
            food_quantity=25,
            inventory_items=["Teleport to house", "Coins", "Rope"]
        )
        
        # Skilling setup
        bank_plan = setup_character_loadout(
            bank_area="VARROCK_WEST",
            food_item="Bread",
            food_quantity=10,
            inventory_items=["Knife", "Tinderbox", "Logs"]
        )
    """
    inventory_config = {
        "required_items": inventory_items or [],
        "optional_items": [],
        "deposit_all": True
    }
    
    return BankPlan(
        bank_area=bank_area,
        food_item=food_item,
        food_quantity=food_quantity,
        equipment_config=equipment_tiers,
        inventory_config=inventory_config
    )
