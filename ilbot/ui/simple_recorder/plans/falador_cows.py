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

from ..actions import bank, inventory, player, combat, equipment, chat
from ..actions.ground_items import loot
from ..actions.timing import wait_until

sys.path.insert(0, str(Path(__file__).parent.parent))

from ilbot.ui.simple_recorder.actions.travel import go_to, in_area
from ilbot.ui.simple_recorder.actions.bank import open_bank, deposit_inventory, withdraw_item, withdraw_items
from ilbot.ui.simple_recorder.actions.inventory import interact, has_item
from ilbot.ui.simple_recorder.helpers.inventory import inv_count
from ilbot.ui.simple_recorder.actions.runtime import emit
from ilbot.ui.simple_recorder.helpers.context import get_payload, get_ui
from ilbot.ui.simple_recorder.constants import FALADOR_BANK, FALADOR_COWS
from .base import Plan


class FaladorCowsPlan(Plan):
    """Main plan class for Falador cows activity."""
    
    id = "FALADOR_COWS"
    label = "Falador Cows - Kill cows and collect hides"
    
    def __init__(self):
        self.state = {"phase": "KILL_COWS"}
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
            ]
        }
        
        self.food_item = "Trout"
        self.target_item = "Cowhide"
        
        # Areas
        self.bank_area = FALADOR_BANK
        self.cows_area = FALADOR_COWS
        
        print(f"[{self.id}] Plan initialized")
        print(f"[{self.id}] Weapon tiers: {len(self.weapon_tiers)} available")
        print(f"[{self.id}] Armor tiers: {len(self.armor_tiers)} types available")
        print(f"[{self.id}] Food: {self.food_item}")
        print(f"[{self.id}] Target loot: {self.target_item}")
    
    def compute_phase(self, payload, craft_recent):
        return self.state.get("phase", "GO_TO_BANK")
    
    def set_phase(self, phase: str, ui=None, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, ui, camera_setup)
    
    def get_best_weapon(self, payload):
        """Get the best weapon available based on attack level and bank contents."""
        from ilbot.ui.simple_recorder.actions.bank import get_bank_inventory, is_open
        from ilbot.ui.simple_recorder.actions.player import get_skill_level
        
        # Get current attack level
        attack_level = get_skill_level("attack")
        print(f"[{self.id}] Getting best weapon for attack level {attack_level}")
        
        # Ensure bank is open before getting inventory
        if not is_open():
            print(f"[{self.id}] Bank not open, cannot get bank inventory")
            return None
        
        # Get available items in bank
        bank_items = get_bank_inventory()
        available_items = [item.get("name", "") for item in bank_items if item.get("name")]
        print(f"[{self.id}] Available bank items: {available_items[:10]}...")  # Show first 10 items
        
        # Find the best weapon we can use (NO FALLBACKS - only weapons we can actually use)
        for weapon in reversed(self.weapon_tiers):  # Start from highest tier
            print(f"[{self.id}] Checking weapon: {weapon['name']} (req: {weapon['attack_req']})")
            if (weapon["attack_req"] <= attack_level and 
                weapon["name"] in available_items):
                print(f"[{self.id}] Selected weapon: {weapon['name']} (req: {weapon['attack_req']}, level: {attack_level})")
                return weapon
        
        print(f"[{self.id}] No suitable weapons found in bank for attack level {attack_level}! Available items: {available_items}")
        return None
    
    def get_best_armor(self, payload):
        """Get the best armor available based on defence level and bank contents."""
        from ilbot.ui.simple_recorder.actions.bank import get_bank_inventory, is_open
        from ilbot.ui.simple_recorder.actions.player import get_skill_level
        
        # Get current defence level
        defence_level = get_skill_level("defence")
        print(f"[{self.id}] Getting best armor for defence level {defence_level}")
        
        # Ensure bank is open before getting inventory
        if not is_open():
            print(f"[{self.id}] Bank not open, cannot get bank inventory")
            return None
        
        # Get available items in bank
        bank_items = get_bank_inventory()
        available_items = [item.get("name", "") for item in bank_items if item.get("name")]
        print(f"[{self.id}] Available bank items: {available_items[:10]}...")  # Show first 10 items
        
        # Find the best armor we can use for each type (NO FALLBACKS - only armor we can actually use)
        best_armor = {}
        for armour_type, armor_list in self.armor_tiers.items():
            print(f"[{self.id}] Checking {armour_type} armor...")
            for armor in reversed(armor_list):  # Start from highest tier
                print(f"[{self.id}] Checking armor: {armor['name']} (req: {armor['defence_req']})")
                if (armor["defence_req"] <= defence_level and 
                    armor["name"] in available_items):
                    print(f"[{self.id}] Selected {armour_type}: {armor['name']} (req: {armor['defence_req']}, level: {defence_level})")
                    best_armor[armour_type] = armor
                    break
        
        if not best_armor:
            print(f"[{self.id}] No suitable armor found in bank for defence level {defence_level}! Available items: {available_items}")
            return None
        
        print(f"[{self.id}] Selected armor: {list(best_armor.keys())}")
        return best_armor
    
    def needs_equipment_change(self, target_weapon, target_armor_dict, payload):
        """Check if we need to change equipment based on what's currently equipped vs what should be equipped."""
        from ilbot.ui.simple_recorder.actions.player import get_skill_level
        
        # Get current skill levels
        attack_level = get_skill_level("attack")
        defence_level = get_skill_level("defence")
        
        # Check if we have a weapon equipped
        current_weapon = None
        for weapon in self.weapon_tiers:
            if equipment.has_equipped(weapon["name"]):
                current_weapon = weapon
                break
        
        # Check if we have armor equipped
        current_armor = {}
        for armor_type, armor_list in self.armor_tiers.items():
            for armor in armor_list:
                if equipment.has_equipped(armor["name"]):
                    current_armor[armor_type] = armor
                    break
        
        print(f"[{self.id}] Current equipment - Weapon: {current_weapon['name'] if current_weapon else 'None'}")
        armor_list = [f"{k}: {v['name']}" for k, v in current_armor.items()]
        print(f"[{self.id}] Current armor: {armor_list}")
        
        # Check if we need to change weapon
        needs_weapon_change = False
        if target_weapon and current_weapon:
            if current_weapon["attack_req"] < target_weapon["attack_req"]:
                print(f"[{self.id}] Need weapon upgrade: {current_weapon['name']} -> {target_weapon['name']}")
                needs_weapon_change = True
            elif current_weapon["name"] != target_weapon["name"]:
                print(f"[{self.id}] Need weapon change: {current_weapon['name']} -> {target_weapon['name']}")
                needs_weapon_change = True
        elif target_weapon and not current_weapon:
            print(f"[{self.id}] Need to equip weapon: {target_weapon['name']}")
            needs_weapon_change = True
        
        # Check if we need to change armor
        needs_armor_change = False
        if target_armor_dict:
            for armor_type, target_armor in target_armor_dict.items():
                current_armor_item = current_armor.get(armor_type)
                if current_armor_item:
                    if current_armor_item["defence_req"] < target_armor["defence_req"]:
                        print(f"[{self.id}] Need {armor_type} upgrade: {current_armor_item['name']} -> {target_armor['name']}")
                        needs_armor_change = True
                    elif current_armor_item["name"] != target_armor["name"]:
                        print(f"[{self.id}] Need {armor_type} change: {current_armor_item['name']} -> {target_armor['name']}")
                        needs_armor_change = True
                else:
                    print(f"[{self.id}] Need to equip {armor_type}: {target_armor['name']}")
                    needs_armor_change = True
        
        needs_change = needs_weapon_change or needs_armor_change
        print(f"[{self.id}] Equipment change needed: {needs_change} (weapon: {needs_weapon_change}, armor: {needs_armor_change})")
        return needs_change
    
    def select_combat_style_for_training(self, payload):
        """Select combat style based on current skill levels and training goals."""
        from ..actions.combat_interface import select_combat_style, current_combat_style
        
        # Get current skill levels
        attack_level = player.get_skill_level("attack")
        defence_level = player.get_skill_level("defence")
        strength_level = player.get_skill_level("strength")
        
        print(f"[{self.id}] Current levels - Attack: {attack_level}, Defence: {defence_level}, Strength: {strength_level}")
        
        # Get current combat style
        current_style = current_combat_style(payload)
        
        # Check if we should switch away from current style
        should_switch = False
        current_skill_level = 0
        
        if current_style == 0:  # Attack style
            current_skill_level = attack_level
            should_switch = (attack_level % 5 == 0 or attack_level >= 40)
        elif current_style == 1:  # Strength style
            current_skill_level = strength_level
            should_switch = (strength_level % 5 == 0)
        elif current_style == 3:  # Defence style
            current_skill_level = defence_level
            should_switch = (defence_level % 5 == 0 or defence_level >= 10)
        
        if not should_switch:
            print(f"[{self.id}] Continuing with current combat style {current_style} (level {current_skill_level}, not at switching point)")
            return
        
        # Find the lowest skill level to train next
        # Check which skills are below their max levels
        skills_to_train = []
        
        if defence_level < 10:
            skills_to_train.append({"name": "Defence", "level": defence_level, "max": 10, "style": 3})
        
        if attack_level < 40:
            skills_to_train.append({"name": "Attack", "level": attack_level, "max": 40, "style": 0})
        
        # Strength has no max level, so always include it
        skills_to_train.append({"name": "Strength", "level": strength_level, "max": 999, "style": 1})
        
        # Find the skill with the lowest level
        if skills_to_train:
            lowest_skill = min(skills_to_train, key=lambda x: x["level"])
            target_style = lowest_skill["style"]
            reason = f"Switching to {lowest_skill['name']} training (level {lowest_skill['level']}, lowest level)"
        else:
            # Fallback to strength if no skills need training
            target_style = 1
            reason = f"Switching to Strength training (level {strength_level}, fallback)"
        
        print(f"[{self.id}] {reason}")
        select_combat_style(target_style, payload)
    
    def loop(self, ui, payload):
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "GO_TO_BANK")
        
        match(phase):
            case "GO_TO_BANK":
                if not in_area(self.bank_area):
                    print(f"[{self.id}] Traveling to bank...")
                    go_to(self.bank_area)
                    return
                else:
                    self.set_phase("PREPARE_AT_BANK")
                    return
            
            case "PREPARE_AT_BANK":
                has_food = inventory.has_item(self.food_item)
                
                if not bank.is_open():
                    if not in_area(self.bank_area):
                        print(f"[{self.id}] Traveling to bank...")
                        go_to(self.bank_area)
                        return
                    print(f"[{self.id}] Opening bank...")
                    open_bank()
                    return
                else:
                    # Bank is open, now we can check equipment requirements
                    # Get the specific weapon and armor that should be equipped based on skill levels
                    target_weapon = self.get_best_weapon(payload)
                    target_armor_dict = self.get_best_armor(payload)
                    
                    # Check if the correct weapon is equipped
                    has_weapon = target_weapon and equipment.has_equipped(target_weapon["name"])
                    
                    # Check if all required armor pieces are equipped
                    has_armor = True
                    if target_armor_dict:
                        for armor_type, armor_item in target_armor_dict.items():
                            if not equipment.has_equipped(armor_item["name"]):
                                has_armor = False
                                break
                    else:
                        has_armor = False
                    
                    # Check if we need to change equipment
                    needs_equipment_change = self.needs_equipment_change(target_weapon, target_armor_dict, payload)
                    
                    if has_food and not needs_equipment_change and not inventory.has_item("Cowhide"):
                        print(f"[{self.id}] Inventory and equipment check passed, closing bank...")
                        bank.close_bank()
                        self.set_phase("TRAVEL_TO_COWS")
                        return
                    else:
                        print(f"[{self.id}] Inventory/equipment not ready - food: {has_food}, weapon: {has_weapon}, armor: {has_armor}")
                        print(f"[{self.id}] Needs equipment change: {needs_equipment_change}")
                    
                    # Deposit all items first
                    print(f"[{self.id}] Depositing all items...")
                    deposit_inventory()
                    
                    # Only deposit equipment if we need to change it
                    if needs_equipment_change:
                        print(f"[{self.id}] Depositing equipment for upgrade...")
                        bank.deposit_equipment()
                    else:
                        print(f"[{self.id}] Equipment is already optimal, skipping deposit...")
                    
                    # Withdraw food
                    print(f"[{self.id}] Withdrawing food...")
                    withdraw_item(self.food_item, 5)
                    
                    # Only withdraw and equip equipment if we need to change it
                    if needs_equipment_change:
                        # Withdraw and equip best available gear
                        equipment_to_withdraw = []
                        equipment_actions = []

                        # Get best available equipment
                        best_weapon = self.get_best_weapon(payload)
                        best_armor_dict = self.get_best_armor(payload)
                        
                        if best_weapon:
                            equipment_to_withdraw.append(best_weapon["name"])
                            equipment_actions.append("Wield")
                        
                        if best_armor_dict:
                            for armor_type, armor_item in best_armor_dict.items():
                                equipment_to_withdraw.append(armor_item["name"])
                                equipment_actions.append("Wear")
                        
                        if equipment_to_withdraw:
                            print(f"[{self.id}] Withdrawing equipment: {equipment_to_withdraw}")
                            withdraw_items(equipment_to_withdraw)
                            bank.interact(equipment_to_withdraw, equipment_actions)
                        else:
                            print(f"[{self.id}] No suitable equipment found in bank!")
                    else:
                        print(f"[{self.id}] Equipment is already optimal, no changes needed!")
                    
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
                self.select_combat_style_for_training(payload)
                
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
    
