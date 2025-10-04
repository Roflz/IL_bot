#!/usr/bin/env python3
"""
Woodcutting Plan
================

This plan handles the complete woodcutting cycle:
1. Go to tree area
2. Cut logs until inventory is full
3. Go to bank and deposit logs
4. Repeat the cycle

Uses the new long-distance travel system and improved inventory methods.
"""

import time
import random
from typing import Dict, Any, Optional

from ..actions import bank, inventory, objects, travel as trav, npc as npc, player, equipment, chat
from ..actions.inventory import has_item
from ..actions.timing import wait_until
from ..helpers.navigation import player_in_rect
from ..helpers.context import get_payload, get_ui
from .base import Plan


class WoodcuttingPlan(Plan):
    """Main plan class for woodcutting activity."""
    
    id = "WOODCUTTING"
    label = "Woodcutting - Cut logs and bank them"
    
    def __init__(self):
        self.state = {"phase": "GO_TO_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        from ..helpers.camera import setup_camera_optimal
        setup_camera_optimal()
        
        # Configuration
        self.tree_area = "VARROCK_WEST_TREES"  # Change this to your preferred tree area
        self.bank_area = "VARROCK_WEST"   # Change this to your preferred bank
        self.tree_type = "Tree"             # Type of tree to cut
        self.log_name = "Logs"              # Name of logs to collect
        # Note: Axe is now selected dynamically based on woodcutting level
        
        # Axe options in order of preference (best to worst)
        # Format: (axe_name, woodcutting_level, attack_level, defence_level)
        self.axe_options = [
            ("Dragon axe", 61, 60, 1),
            ("Rune axe", 41, 40, 1),
            ("Adamant axe", 31, 30, 1),
            ("Mithril axe", 21, 20, 1),
            ("Steel axe", 6, 5, 1),
            ("Iron axe", 1, 1, 1),
            ("Bronze axe", 1, 1, 1)
        ]
        
        # State tracking
        self.cutting_start_time = 0
        self.banking_start_time = 0
        
        print(f"[{self.id}] Plan initialized")
        print(f"[{self.id}] Tree area: {self.tree_area}")
        print(f"[{self.id}] Bank area: {self.bank_area}")
        print(f"[{self.id}] Tree type: {self.tree_type}")
        print(f"[{self.id}] Collecting: {self.log_name}")
        
    def compute_phase(self, payload: Dict[str, Any], craft_recent: bool = False) -> str:
        """Determine current phase of the plan."""
        return self.state.get("phase", "GO_TO_TREES")

    def set_phase(self, phase: str, ui=None, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, ui, camera_setup)

    def loop(self, ui, payload: Dict[str, Any]) -> int:
        """Main loop function."""
        phase = self.state.get("phase", "GO_TO_TREES")
        
        ui.debug(f"[{self.id}] Phase: {phase}")

        try:
            match(phase):
                case "GO_TO_TREES":
                    return self._handle_go_to_trees(ui, payload)
                    
                case "FIND_TREE":
                    return self._handle_find_tree(ui, payload)
                    
                case "GO_TO_BANK":
                    return self._handle_go_to_bank(ui, payload)
                    
                case "BANKING":
                    return self._handle_banking(ui, payload)

                case _:
                    ui.debug(f"[{self.id}] Unknown phase: {phase}")
                    return self.loop_interval_ms

        except Exception as e:
            ui.debug(f"[{self.id}] Error in phase {phase}: {e}")
            return self.loop_interval_ms

    def _handle_go_to_trees(self, ui, payload: Dict[str, Any]) -> int:
        """Handle going to the tree area."""
        # Check if inventory is full
        if inventory.is_full():
            self.set_phase("GO_TO_BANK", ui)
            return self.loop_interval_ms
        
        # Check if we're already in the tree area
        if trav.in_area(self.tree_area):
            self.set_phase("FIND_TREE", ui)
            return self.loop_interval_ms
        
        ui.debug(f"[{self.id}] Going to {self.tree_area}")
        trav.go_to(self.tree_area)
        return self.loop_interval_ms

    def _handle_find_tree(self, ui, payload: Dict[str, Any]) -> int:
        """Handle finding a tree to cut."""
        # Check if inventory is full
        if inventory.is_full():
            self.set_phase("GO_TO_BANK", ui)
            return self.loop_interval_ms

        if chat.can_continue():
            chat.continue_dialogue()
            return self.loop_interval_ms
        
        ui.debug(f"[{self.id}] Looking for a tree to cut")
        if player.get_player_animation() == "CHOPPING":
            return self.loop_interval_ms
        
        # Look for trees in the area
        tree = objects.click(self.tree_type)
        wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=5000)
        return

    def _handle_go_to_bank(self, ui, payload: Dict[str, Any]) -> int:
        """Handle going to the bank."""
        # Check if we're already at the bank
        if trav.in_area(self.bank_area):
            self.set_phase("BANKING", ui)
            return self.loop_interval_ms
        else:
            trav.go_to(self.bank_area)
            return self.loop_interval_ms

    def _handle_banking(self, ui, payload: Dict[str, Any]) -> int:
        """Handle banking the logs."""
        ui.debug(f"[{self.id}] Banking logs")
        # best_axe = self._get_best_axe_for_level()

        if bank.is_closed():
            bank.open_bank()
            return self.loop_interval_ms
        else:
            best_axe, wc_lvl, att_lvl, def_lvl = self._get_best_axe_for_level()
            if bank.inv_has("logs"):
                bank.deposit_inventory()
                return self.loop_interval_ms
            elif not equipment.has_equipped(best_axe) and not bank.inv_has(best_axe):
                bank.deposit_equipment()
                time.sleep(0.5)
                bank.withdraw_item(best_axe)
                time.sleep(0.5)
                return self.loop_interval_ms
            elif bank.inv_has(best_axe) and self._can_equip_item(best_axe, required_attack=att_lvl) and not equipment.has_equipped(best_axe):
                bank.interact(best_axe, "Wield")
                return self.loop_interval_ms
            else:
                bank.close_bank()
                self.set_phase("GO_TO_TREES", ui)
                return self.loop_interval_ms

    def _get_best_axe_for_level(self):
        """Get the best axe name for current woodcutting level using IPC.
        Assumes bank is open and checks bank, inventory, and equipment for available axes."""
        try:
            # Get woodcutting level using IPC
            woodcutting_level = player.get_skill_level("woodcutting")
            if woodcutting_level is None:
                woodcutting_level = 1  # Fallback
        except:
            woodcutting_level = 1  # Fallback
        
        # Use class attribute for axe options
        
        # Get current bank contents, inventory, and equipment
        bank_items = bank.get_bank_contents()
        inventory_items = bank.get_bank_inventory()
        equipment_items = equipment.list_equipped_items()
        
        # Helper function to check if axe is available
        def is_axe_available(axe_name):
            # Check inventory
            if bank.inv_has(axe_name):
                return True
            # Check equipment
            if equipment.has_equipped(axe_name):
                return True
            # Check bank
            if bank.has_item(axe_name):
                return True
            return False
        
        # Find the best axe we can use that's actually available
        for axe_name, wc_level, att_level, def_level in self.axe_options:
            if (woodcutting_level >= wc_level and
                is_axe_available(axe_name)):
                return axe_name, wc_level, att_level, def_level
        
        raise Exception(f"No suitable axe found for woodcutting level {woodcutting_level}. Check bank, inventory, and equipment for any axe.")

    def _can_equip_item(self, item_name: str, required_attack: int = 0, required_defence: int = 0) -> bool:
        """Check if player has high enough attack or defence level to equip an item."""
        try:
            # Get current attack and defence levels using IPC
            attack_level = player.get_skill_level("attack")
            defence_level = player.get_skill_level("defence")
            
            if attack_level is None:
                attack_level = 1
            if defence_level is None:
                defence_level = 1
            
            # Check if we meet the requirements
            has_attack = attack_level >= required_attack
            has_defence = defence_level >= required_defence
            
            # Return True if we have either attack OR defence level (whichever is higher)
            # This covers cases where some items require attack, others defence
            can_equip = has_attack and has_defence
            
            print(f"[{self.id}] Can equip {item_name}? Attack: {attack_level}/{required_attack}, Defence: {defence_level}/{required_defence} -> {can_equip}")
            
            return can_equip
            
        except Exception as e:
            print(f"[{self.id}] Error checking equip requirements for {item_name}: {e}")
            return True  # Fallback to allowing equip if we can't check

    def _find_trees(self) -> list:
        """Find trees in the current area using IPC."""
        try:
            # Get objects using IPC
            objects_list = objects.get_objects()
            trees = []

            for obj in objects_list:
                if obj and obj.get("name", "").lower() == self.tree_type.lower():
                    trees.append(obj)

            # Sort by distance (closest first)
            trees.sort(key=lambda x: x.get("distance", 999))
            return trees
        except:
            return []

    def _move_randomly_in_area(self) -> None:
        """Move to a random spot in the tree area to find trees."""
        try:
            # Get current player position using IPC
            player_x = player.get_world_x()
            player_y = player.get_world_y()
            
            if player_x is None or player_y is None:
                return

            # Generate random offset
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-5, 5)

            # Create target position
            target_x = player_x + offset_x
            target_y = player_y + offset_y

            # Create small rect around target
            target_rect = (target_x - 1, target_x + 1, target_y - 1, target_y + 1)

            # Move to target
            trav.go_to(target_rect, center=True)
        except:
            pass  # Ignore errors in random movement