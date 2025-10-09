#!/usr/bin/env python3
"""
Falador Cows Plan
================

This plan handles the complete cow killing and hide collection cycle:
1. Start at Falador bank
2. Equip combat gear and withdraw food
3. Travel to cows area using long-distance pathfinding
4. Attack cows and collect cowhides
5. Bank when inventory is full
6. Repeat the cycle

Uses the new long-distance travel system and improved combat/inventory methods.
"""

import time
import random
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from ..actions import bank, inventory, player, combat, equipment, chat, travel, ge
from ..actions.equipment import get_best_weapon_for_level, get_best_armor_for_level, get_best_weapon_for_level_in_bank, \
    get_best_armor_for_level_in_bank
from ..actions.ge import check_and_buy_required_items, close_ge
from ..actions.ground_items import loot
from ..actions.timing import wait_until
from ..helpers.bank import near_any_bank

sys.path.insert(0, str(Path(__file__).parent.parent))

from ilbot.ui.simple_recorder.actions.travel import go_to, in_area
from ilbot.ui.simple_recorder.actions.bank import open_bank, deposit_inventory, withdraw_item, withdraw_items
from ilbot.ui.simple_recorder.constants import FALADOR_BANK, FALADOR_COWS
from .base import Plan


class FaladorCowsPlan(Plan):
    """Main plan class for Falador cows activity."""
    
    id = "FALADOR_COWS"
    label = "Falador Cows - Kill cows and collect hides"
    
    def __init__(self):
        self.state = {"phase": "GO_TO_CLOSEST_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
        setup_camera_optimal()
        
        # Equipment tiers (ordered by level requirement)
        self.weapon_tiers = [
            {"name": "Bronze scimitar", "attack_req": 1},
            {"name": "Iron scimitar", "attack_req": 1},
            {"name": "Steel scimitar", "attack_req": 5},
            {"name": "Mithril scimitar", "attack_req": 20},
            {"name": "Adamant scimitar", "attack_req": 30},
            {"name": "Rune scimitar", "attack_req": 40}
        ]
        
        self.armor_tiers = {
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
        }
        
        self.jewelry_tiers = {
            "amulet": [
                {"name": "Amulet of strength", "defence_req": 1}
            ]
        }
        
        self.food_item = "Trout"
        self.target_item = "Cowhide"
        
        # GE buying configuration - individual item pricing
        self.ge_config = {
            # Food items
            "Trout": {
                "quantity": 50,
                "bumps": 5,
                "set_price": 0  # 0 = use bumps, >0 = use set price
            },
            
            # Weapon items
            "Bronze scimitar": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000  # Fixed price
            },
            "Iron scimitar": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000  # Fixed price
            },
            "Steel scimitar": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000  # Fixed price
            },
            "Mithril scimitar": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 2000  # Fixed price
            },
            "Adamant scimitar": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 5000  # Fixed price
            },
            "Rune scimitar": {
                "quantity": 1,
                "bumps": 5,
                "set_price": 0  # Fixed price
            },
            
            # Armor items
            "Bronze full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Iron full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Steel full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Mithril full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 3000
            },
            "Adamant full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 5000
            },
            "Rune full helm": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 30000
            },
            
            "Bronze platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Iron platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Steel platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Mithril platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 10000
            },
            "Adamant platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 20000
            },
            "Rune platebody": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 50000
            },
            
            "Bronze platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Iron platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Steel platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Mithril platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 2000
            },
            "Adamant platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 5000
            },
            "Rune platelegs": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 40000
            },
            
            "Bronze kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Iron kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Steel kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 1000
            },
            "Mithril kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 2000
            },
            "Adamant kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 5000
            },
            "Rune kiteshield": {
                "quantity": 1,
                "bumps": 0,
                "set_price": 40000
            },
            
            # Jewelry items
            "Amulet of strength": {
                "quantity": 1,
                "bumps": 5,
                "set_price": 0  # 0 = use bumps, >0 = use set price
            }
        }
        
        # Areas
        self.bank_area = FALADOR_BANK
        self.cows_area = FALADOR_COWS
        
        # Required items for the plan
        self.required_items = {
            "food": self.food_item,
            "weapon": None,  # Will be determined by skill level
            "armor": None,   # Will be determined by skill level
            "jewelry": None, # Will be determined by skill level
        }
        
        # Track what we need to buy from GE
        self.items_to_buy = []
        
        print(f"[{self.id}] Plan initialized")
        print(f"[{self.id}] Weapon tiers: {len(self.weapon_tiers)} available")
        print(f"[{self.id}] Armor tiers: {len(self.armor_tiers)} types available")
        print(f"[{self.id}] Food: {self.food_item}")
        print(f"[{self.id}] Target loot: {self.target_item}")
    
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    
    def loop(self, ui):
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "GO_TO_CLOSEST_BANK")
        
        match(phase):
            case "GO_TO_CLOSEST_BANK":
                if not near_any_bank():
                    travel.go_to_closest_bank()
                else:
                    self.set_phase("CHECK_BANK_ITEMS", ui)
                return
            
            case "CHECK_BANK_ITEMS":
                if ge.is_open():
                    close_ge()
                    return
                if not bank.is_open():
                    print(f"[{self.id}] Opening bank...")
                    open_bank()
                    return
                else:
                    # Bank is open, check what items we have and what we need
                    print(f"[{self.id}] Checking bank, inventory, and equipped items...")
                    
                    # Determine what equipment we should have based on skill levels
                    target_weapon = get_best_weapon_for_level(self.weapon_tiers, self.id)
                    target_armor_dict = get_best_armor_for_level(self.armor_tiers, self.id)
                    target_jewelry_dict = get_best_armor_for_level(self.jewelry_tiers, self.id)
                    
                    # Helper function to check if we have an item anywhere (bank, inventory, or equipped)
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
                    
                    # Determine what we need to buy - check each item individually
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
                    
                    # Detailed logging of what we found where
                    print(f"[{self.id}] Item check complete:")
                    print(f"[{self.id}]   Food ({self.food_item}):")
                    print(f"[{self.id}]     Bank: {bank.has_item(self.food_item)}")
                    print(f"[{self.id}]     Inventory: {inventory.has_item(self.food_item)}")
                    print(f"[{self.id}]     Equipped: {equipment.has_equipped(self.food_item)}")
                    print(f"[{self.id}]     Total: {has_food}")
                    
                    if target_weapon:
                        print(f"[{self.id}]   Weapon ({target_weapon['name']}):")
                        print(f"[{self.id}]     Bank: {bank.has_item(target_weapon['name'])}")
                        print(f"[{self.id}]     Inventory: {inventory.has_item(target_weapon['name'])}")
                        print(f"[{self.id}]     Equipped: {equipment.has_equipped(target_weapon['name'])}")
                        print(f"[{self.id}]     Total: {has_weapon}")
                    
                    if target_armor_dict:
                        print(f"[{self.id}]   Armor pieces:")
                        for armor_type, armor_item in target_armor_dict.items():
                            print(f"[{self.id}]     {armor_item['name']}:")
                            print(f"[{self.id}]       Bank: {bank.has_item(armor_item['name'])}")
                            print(f"[{self.id}]       Inventory: {inventory.has_item(armor_item['name'])}")
                            print(f"[{self.id}]       Equipped: {equipment.has_equipped(armor_item['name'])}")
                            print(f"[{self.id}]       Total: {has_item_anywhere(armor_item['name'])}")
                    
                    if target_jewelry_dict:
                        print(f"[{self.id}]   Jewelry pieces:")
                        for jewelry_type, jewelry_item in target_jewelry_dict.items():
                            print(f"[{self.id}]     {jewelry_item['name']}:")
                            print(f"[{self.id}]       Bank: {bank.has_item(jewelry_item['name'])}")
                            print(f"[{self.id}]       Inventory: {inventory.has_item(jewelry_item['name'])}")
                            print(f"[{self.id}]       Equipped: {equipment.has_equipped(jewelry_item['name'])}")
                            print(f"[{self.id}]       Total: {has_item_anywhere(jewelry_item['name'])}")
                    
                    print(f"[{self.id}]   Items needed: {items_needed}")
                    
                    if items_needed:
                        self.items_to_buy = items_needed
                        bank.close_bank()
                        self.set_phase("GO_TO_GE")
                        return
                    else:
                        print(f"[{self.id}] All required items found across bank, inventory, and equipment!")
                        bank.close_bank()
                        self.set_phase("GO_TO_FALADOR_BANK")
                        return
            
            case "GO_TO_GE":
                # Go to Grand Exchange to buy missing items
                if not in_area("GE"):
                    print(f"[{self.id}] Traveling to Grand Exchange...")
                    go_to("GE")
                    return
                else:
                    self.set_phase("BUY_ITEMS_AT_GE")
                    return
            
            case "BUY_ITEMS_AT_GE":
                # Define required items and their requirements using individual item configuration
                required_items = []
                item_requirements = {}
                
                # Add food requirement using individual item config
                if self.food_item:
                    required_items.append(self.food_item)
                    if self.food_item in self.ge_config:
                        food_config = self.ge_config[self.food_item]
                        item_requirements[self.food_item] = (
                            food_config["quantity"], 
                            food_config["bumps"], 
                            food_config["set_price"]
                        )
                    else:
                        # Fallback if item not in config
                        item_requirements[self.food_item] = (50, 5, 0)
                
                # Add weapon requirement using individual item config
                target_weapon = get_best_weapon_for_level(self.weapon_tiers, self.id)
                if target_weapon:
                    required_items.append(target_weapon["name"])
                    if target_weapon["name"] in self.ge_config:
                        weapon_config = self.ge_config[target_weapon["name"]]
                        item_requirements[target_weapon["name"]] = (
                            weapon_config["quantity"], 
                            weapon_config["bumps"], 
                            weapon_config["set_price"]
                        )
                    else:
                        # Fallback if item not in config
                        item_requirements[target_weapon["name"]] = (1, 0, 1000)
                
                # Add armor requirements using individual item config
                target_armor_dict = get_best_armor_for_level(self.armor_tiers, self.id)
                if target_armor_dict:
                    for armor_type, armor_item in target_armor_dict.items():
                        required_items.append(armor_item["name"])
                        if armor_item["name"] in self.ge_config:
                            armor_config = self.ge_config[armor_item["name"]]
                            item_requirements[armor_item["name"]] = (
                                armor_config["quantity"], 
                                armor_config["bumps"], 
                                armor_config["set_price"]
                            )
                        else:
                            # Fallback if item not in config
                            item_requirements[armor_item["name"]] = (1, 0, 500)
                
                # Add jewelry requirements using individual item config
                target_jewelry_dict = get_best_armor_for_level(self.jewelry_tiers, self.id)
                if target_jewelry_dict:
                    for jewelry_type, jewelry_item in target_jewelry_dict.items():
                        required_items.append(jewelry_item["name"])
                        if jewelry_item["name"] in self.ge_config:
                            jewelry_config = self.ge_config[jewelry_item["name"]]
                            item_requirements[jewelry_item["name"]] = (
                                jewelry_config["quantity"], 
                                jewelry_config["bumps"], 
                                jewelry_config["set_price"]
                            )
                        else:
                            # Fallback if item not in config
                            item_requirements[jewelry_item["name"]] = (1, 5, 0)
                
                # Use the reusable method
                result = check_and_buy_required_items(required_items, item_requirements, self.id)
                
                if result["status"] == "complete":
                    print(f"[{self.id}] All items purchased! Returning to bank...")
                    self.set_phase("GO_TO_CLOSEST_BANK")
                    return
                elif result["status"] == "buying":
                    # Still working on buying items, continue in same phase
                    return
                elif result["status"] == "error":
                    print(f"[{self.id}] Error buying items: {result.get('error', 'Unknown error')}")
                    # Try again next loop
                    return
            
            case "GO_TO_FALADOR_BANK":
                if not in_area(self.bank_area):
                    print(f"[{self.id}] Traveling to bank...")
                    go_to(self.bank_area)
                    return
                else:
                    self.set_phase("PREPARE_AT_BANK")
                    return

            case "PREPARE_AT_BANK":
                # First check if we're in the right area
                if not in_area(self.bank_area):
                    print(f"[{self.id}] Traveling to bank...")
                    go_to(self.bank_area)
                    return

                # Open bank if not already open
                if not bank.is_open():
                    print(f"[{self.id}] Opening bank...")
                    open_bank()
                    return

                # Bank is open, now check what we need to do
                print(f"[{self.id}] Bank is open, checking inventory and equipment...")

                # Get target equipment based on skill levels - check bank, inventory, and equipped items
                target_weapon = get_best_weapon_for_level(self.weapon_tiers, self.id)
                target_armor_dict = get_best_armor_for_level(self.armor_tiers, self.id)
                target_jewelry_dict = get_best_armor_for_level(self.jewelry_tiers, self.id)

                # Check current inventory state
                has_sufficient_food = inventory.has_unnoted_item(self.food_item, 5)
                has_cowhides = inventory.has_item("Cowhide")

                # Check current equipment state
                has_correct_weapon = target_weapon and equipment.has_equipped(target_weapon["name"])

                has_correct_armor = True
                if target_armor_dict:
                    for armor_type, armor_item in target_armor_dict.items():
                        if not equipment.has_equipped(armor_item["name"]):
                            has_correct_armor = False
                            break
                else:
                    has_correct_armor = False

                has_correct_jewelry = True
                if target_jewelry_dict:
                    for jewelry_type, jewelry_item in target_jewelry_dict.items():
                        if not equipment.has_equipped(jewelry_item["name"]):
                            has_correct_jewelry = False
                            break
                else:
                    has_correct_jewelry = False

                # Log current state
                print(f"[{self.id}] Current state:")
                print(f"[{self.id}]   Food ({self.food_item}): {has_sufficient_food}")
                print(f"[{self.id}]   Cowhides in inventory: {has_cowhides}")
                print(f"[{self.id}]   Correct weapon equipped: {has_correct_weapon}")
                print(f"[{self.id}]   Correct armor equipped: {has_correct_armor}")
                print(f"[{self.id}]   Correct jewelry equipped: {has_correct_jewelry}")

                # If everything is ready, close bank and proceed
                if (has_sufficient_food and
                        has_correct_weapon and
                        has_correct_armor and
                        has_correct_jewelry and
                        not has_cowhides):
                    print(f"[{self.id}] All requirements met, closing bank...")
                    bank.close_bank()
                    self.set_phase("TRAVEL_TO_COWS")
                    return

                # We need to make changes - start with depositing inventory items
                print(f"[{self.id}] Preparing inventory and equipment...")

                # Deposit unwanted items first (cowhides, wrong food, etc.) but keep equipment we might need
                items_to_deposit = []
                
                # Always deposit cowhides
                if inventory.has_item("Cowhide"):
                    items_to_deposit.append("Cowhide")
                
                # Deposit wrong food if we have too much or wrong type
                if inventory.has_unnoted_item(self.food_item, 6):  # More than we need
                    items_to_deposit.append(self.food_item)
                
                # Deposit any other items that aren't our target equipment
                from ..helpers.inventory import inv_slots
                inventory_slots = inv_slots()
                for slot in inventory_slots:
                    item_name = slot.get("itemName", "").strip()
                    if not item_name or item_name in items_to_deposit:
                        continue
                        
                    # Check if this item is target equipment we might need
                    is_target_equipment = False
                    
                    # Check if it's our target weapon
                    if target_weapon and item_name == target_weapon["name"]:
                        is_target_equipment = True
                    
                    # Check if it's our target armor
                    if target_armor_dict:
                        for armor_type, armor_item in target_armor_dict.items():
                            if item_name == armor_item["name"]:
                                is_target_equipment = True
                                break
                    
                    # Check if it's our target jewelry
                    if target_jewelry_dict:
                        for jewelry_type, jewelry_item in target_jewelry_dict.items():
                            if item_name == jewelry_item["name"]:
                                is_target_equipment = True
                                break
                    
                    # If it's not target equipment and not our food, deposit it
                    if not is_target_equipment and item_name != self.food_item:
                        items_to_deposit.append(item_name)
                
                # Deposit the unwanted items
                if items_to_deposit:
                    print(f"[{self.id}] Depositing unwanted items: {items_to_deposit}")
                    for item in items_to_deposit:
                        bank.deposit_item(item)
                    return  # Wait for deposits to complete

                # Check weapon first
                if target_weapon and not has_correct_weapon:
                    print(f"[{self.id}] Handling weapon change...")
                    
                    # Check if target weapon is in inventory
                    if inventory.has_item(target_weapon["name"]):
                        print(f"[{self.id}] Target weapon in inventory, equipping directly...")
                        bank.interact_inventory(target_weapon["name"], "Wield")
                        return  # Wait for equipping
                    else:
                        # Need to withdraw from bank
                        print(f"[{self.id}] Withdrawing weapon from bank...")
                        withdraw_item(target_weapon["name"], 1)
                        return  # Wait for withdrawal

                # Check armor pieces
                if target_armor_dict:
                    for armor_type, armor_item in target_armor_dict.items():
                        if not equipment.has_equipped(armor_item["name"]):
                            print(f"[{self.id}] Handling {armor_type} change...")
                            
                            # Check if target armor is in inventory
                            if inventory.has_item(armor_item["name"]):
                                print(f"[{self.id}] Target {armor_type} in inventory, equipping directly...")
                                bank.interact_inventory(armor_item["name"], "Wear")
                                return  # Wait for equipping
                            else:
                                # Need to withdraw from bank
                                print(f"[{self.id}] Withdrawing {armor_type} from bank...")
                                withdraw_item(armor_item["name"], 1)
                                return  # Wait for withdrawal

                # Check jewelry pieces
                if target_jewelry_dict:
                    for jewelry_type, jewelry_item in target_jewelry_dict.items():
                        if not equipment.has_equipped(jewelry_item["name"]):
                            print(f"[{self.id}] Handling {jewelry_type} change...")
                            
                            # Check if target jewelry is in inventory
                            if inventory.has_item(jewelry_item["name"]):
                                print(f"[{self.id}] Target {jewelry_type} in inventory, equipping directly...")
                                bank.interact_inventory(jewelry_item["name"], "Wear")
                                return  # Wait for equipping
                            else:
                                # Need to withdraw from bank
                                print(f"[{self.id}] Withdrawing {jewelry_type} from bank...")
                                withdraw_item(jewelry_item["name"], 1)
                                return  # Wait for withdrawal

                # Handle food withdrawal
                if not has_sufficient_food:
                    print(f"[{self.id}] Withdrawing food ({self.food_item})...")
                    withdraw_item(self.food_item, 5)
                    return  # Wait for food withdrawal

                # If we get here, we should be ready
                print(f"[{self.id}] Preparation complete, checking final state...")
                return

            case "TRAVEL_TO_COWS":
                if not in_area(self.cows_area):
                    print(f"[{self.id}] Traveling to cows area...")
                    go_to(self.cows_area)
                    return
                else:
                    self.set_phase("KILL_COWS")
                    return
            
            case "KILL_COWS":
                # Select appropriate combat style for training
                from ilbot.ui.simple_recorder.actions.combat import select_combat_style_for_training
                select_combat_style_for_training()
                
                # Check if inventory is full
                empty_slots = inventory.get_empty_slots_count()
                
                if empty_slots == 0:
                    print(f"[{self.id}] Inventory full, returning to bank")
                    self.set_phase("PREPARE_AT_BANK")
                    return

                if player.get_health() <= player.get_skill_level("hitpoints") - 7:
                    inventory.interact("Trout", "Eat")
                    return

                if chat.can_continue():
                    chat.continue_dialogue()
                    return

                if player.is_in_combat():
                    wait_until(lambda: not player.is_in_combat())
                    return 3000

                hides_in_invent = inventory.inv_count("Cowhide")
                if loot("Cowhide", radius=4):
                    wait_until(lambda: inventory.inv_count("Cowhide") == hides_in_invent + 1, max_wait_ms=3000)
                    return

                if not player.is_in_combat():
                    combat.attack_closest(["Cow", "Cow calf"])
                    return

                time.sleep(1)
                return
            
            case "BANK_HIDES":
                if not in_area(self.bank_area):
                    print(f"[{self.id}] Traveling back to bank...")
                    go_to(self.bank_area)
                    return
                else:
                    self.set_phase("PREPARE_AT_BANK")
                    return
    
