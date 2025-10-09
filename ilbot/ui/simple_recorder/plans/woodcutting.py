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
import logging
import time
from ..actions import bank, inventory, objects, travel as trav, player, equipment, chat
from ..actions.timing import wait_until
from .base import Plan


class WoodcuttingPlan(Plan):
    """Main plan class for woodcutting activity."""
    
    id = "WOODCUTTING"
    label = "Woodcutting - Cut logs and bank them"
    
    def __init__(self):
        from ..helpers.runtime_utils import ui, dispatch
        import logging
        self.ui = ui
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
            ("Black axe", 11, 10, 1),
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
        

    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        """Main loop function."""
        phase = self.state.get("phase", "GO_TO_TREES")
        
        logging.info(f"[{self.id}] Phase: {phase}")

        try:
            match(phase):
                case "GO_TO_TREES":
                    return self._handle_go_to_trees()
                    
                case "FIND_TREE":
                    return self._handle_find_tree()
                    
                case "GO_TO_BANK":
                    return self._handle_go_to_bank()
                    
                case "BANKING":
                    return self._handle_banking()

                case _:
                    logging.info(f"[{self.id}] Unknown phase: {phase}")
                    return self.loop_interval_ms

        except Exception as e:
            logging.info(f"[{self.id}] Error in phase {phase}: {e}")
            return self.loop_interval_ms

    def _handle_go_to_trees(self) -> int:
        """Handle going to the tree area."""
        # Check if inventory is full
        if inventory.is_full():
            self.set_phase("GO_TO_BANK")
            return self.loop_interval_ms
        
        # Check if we're already in the tree area
        if trav.in_area(self.tree_area):
            self.set_phase("FIND_TREE")
            return self.loop_interval_ms
        
        logging.info(f"[{self.id}] Going to {self.tree_area}")
        trav.go_to(self.tree_area)
        return self.loop_interval_ms

    def _handle_find_tree(self) -> int:
        """Handle finding a tree to cut."""
        # Check if inventory is full
        if inventory.is_full():
            self.set_phase("GO_TO_BANK")
            return self.loop_interval_ms

        if chat.can_continue():
            chat.continue_dialogue()
            return self.loop_interval_ms
        
        logging.info(f"[{self.id}] Looking for a tree to cut")
        if player.get_player_animation() == "CHOPPING":
            return self.loop_interval_ms
        
        # Look for trees in the area
        tree = objects.click(self.tree_type)
        wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=5000)
        return self.loop_interval_ms

    def _handle_go_to_bank(self) -> int:
        """Handle going to the bank."""
        # Check if we're already at the bank
        if trav.in_area(self.bank_area):
            self.set_phase("BANKING")
            return self.loop_interval_ms
        else:
            trav.go_to(self.bank_area)
            return self.loop_interval_ms

    def _handle_banking(self) -> int:
        """Handle banking the logs."""
        logging.info(f"[{self.id}] Banking logs")

        if bank.is_closed():
            bank.open_bank()
            return self.loop_interval_ms
        else:
            from ilbot.ui.simple_recorder.actions.equipment import get_best_tool_for_level, can_equip_item
            best_axe, wc_lvl, att_lvl, def_lvl = get_best_tool_for_level(self.axe_options, "woodcutting", self.id)
            if inventory.inv_has("logs"):
                bank.deposit_inventory()
                return self.loop_interval_ms
            elif not equipment.has_equipped(best_axe) and not inventory.inv_has(best_axe):
                bank.deposit_equipment()
                time.sleep(0.5)
                bank.withdraw_item(best_axe)
                time.sleep(0.5)
                return self.loop_interval_ms
            elif inventory.inv_has(best_axe) and can_equip_item(best_axe, required_attack=att_lvl) and not equipment.has_equipped(best_axe):
                bank.interact(best_axe, "Wield")
                return self.loop_interval_ms
            else:
                bank.close_bank()
                self.set_phase("GO_TO_TREES")
                return self.loop_interval_ms


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