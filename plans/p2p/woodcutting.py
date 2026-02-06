#!/usr/bin/env python3
"""
Woodcutting Plan (P2P)
======================

Simple woodcutting plan that cuts normal trees at the Grand Exchange and banks them.
"""

import logging
import time
from pathlib import Path
import sys

from actions import player, bank, inventory, objects, chat, equipment, npc
from actions import wait_until
from actions.travel import travel_to_bank, go_to, in_area, go_to_tile
from actions import objects_legacy
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.utils import exponential_number, sleep_exponential
from helpers.widgets import widget_exists
from helpers.keyboard import press_spacebar
from helpers.inventory import inv_count

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan


class WoodcuttingPlan(Plan):
    """Simple woodcutting plan for P2P - cuts trees at Grand Exchange and banks them."""
    
    id = "WOODCUTTING"
    label = "Woodcutting: Grand Exchange"
    description = """Cuts normal trees at the Grand Exchange and banks logs for woodcutting XP. Simple and efficient loop with automatic banking support.

Starting Area: Grand Exchange
Required Items: Axe"""
    
    def __init__(self):
        self.state = {"phase": "BANK", "last_chopping_ts": None, "available_axes_in_bank": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration - Regular trees
        self.tree_area = (3140, 3193, 3483, 3515)  # Grand Exchange bounds
        self.tree_type = "Tree"
        self.log_name = "Logs"
        
        # Configuration - Oak trees (level 15+)
        self.oak_tree_tile = (2719, 3480)  # Specific oak tree tile
        self.oak_tree_type = "Oak"
        self.oak_log_name = "Oak logs"
        
        # Configuration - Willow trees (level 30+)
        self.willow_tree_area = (3081, 3094, 3225, 3239)  # Willow tree area
        self.willow_tree_type = "Willow"
        self.willow_log_name = "Willow logs"
        
        # Configuration - Maple trees (level 45+)
        # Check if "CAMELOT_MAPLES" exists in REGIONS, otherwise use coordinates
        from constants import REGIONS
        if "CAMELOT_MAPLES" in REGIONS:
            self.maple_tree_area = REGIONS["CAMELOT_MAPLES"]
        else:
            # Default maple tree area near Camelot/Seers (typical location)
            self.maple_tree_area = (2720, 2730, 3500, 3510)
        self.maple_tree_type = "Maple"
        self.maple_log_name = "Maple logs"
        
        self.bank_area = "GE"  # Will use Grand Exchange bank booths
        
        # Axe options in order of preference (best to worst)
        # Format: (axe_name, woodcutting_level_req, attack_level_req)
        self.axe_options = [
            ("Dragon axe", 61, 60),
            ("Rune axe", 41, 40),
            ("Adamant axe", 31, 30),
            ("Mithril axe", 21, 20),
            ("Black axe", 11, 10),
            ("Steel axe", 6, 5),
            ("Iron axe", 1, 1),
            ("Bronze axe", 1, 1)
        ]
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Tree area: {self.tree_area}")
        logging.info(f"[{self.id}] Tree type: {self.tree_type}")
        logging.info(f"[{self.id}] Collecting: {self.log_name}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK")
        logged_in = player.logged_in()
        if not logged_in and not phase == "DONE":
            player.login()
            return exponential_number(600, 1600, 1.2)

        match(phase):
            case "BANK":
                return self._handle_bank()
            case "WOODCUTTING":
                return self._handle_woodcutting()
            case "OAK_WOODCUTTING":
                return self._handle_oak_woodcutting()
            case "WILLOW_WOODCUTTING":
                return self._handle_willow_woodcutting()
            case "MAPLE_WOODCUTTING":
                return self._handle_maple_woodcutting()
            # Forestry event phases
            case "FORESTRY_FLOWERING_TREE":
                return self._handle_flowering_tree_event()
            case "FORESTRY_LEPRECHAUN":
                return self._handle_leprechaun_event()
            case "FORESTRY_BEEHIVE":
                return self._handle_beehive_event()
            case "FORESTRY_FRIENDLY_ENT":
                return self._handle_friendly_ent_event()
            case "FORESTRY_POACHERS":
                return self._handle_poachers_event()
            case "FORESTRY_ENCHANTMENT_RITUAL":
                return self._handle_enchantment_ritual_event()
            case "FORESTRY_PHEASANT_CONTROL":
                return self._handle_pheasant_control_event()
            case "FORESTRY_RISING_ROOTS":
                return self._handle_rising_roots_event()
            case "FORESTRY_STRUGGLING_SAPLING":
                return self._handle_struggling_sapling_event()
            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return exponential_number(400, 1100, 1.2)
    
    def _handle_bank(self) -> int:
        """Handle banking phase - intelligent axe management and inventory preparation."""
        # Get woodcutting level to determine bank area and best axe
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        fletching_level = player.get_skill_level("fletching") or 1

        # Determine bank area based on level (SEERS_BANK for maple, DRAYNOR_BANK for willow, SEERS_BANK for oak, GE for regular)
        if woodcutting_level >= 45 and fletching_level >= 45:
            bank_area = "SEERS_BANK"  # Camelot bank
        elif woodcutting_level >= 30 and fletching_level >= 30:
            bank_area = "DRAYNOR_BANK"
        elif woodcutting_level >= 15 and fletching_level >= 15:
            bank_area = "SEERS_BANK"
        else:
            bank_area = "GE"
        
        # Travel to bank if not already there
        if not travel_to_bank(bank_area):
            return exponential_number(400, 1100, 1.2)
        
        # Open bank
        if not bank.is_open():
            bank.open_bank()
            wait_until(bank.is_open, max_wait_ms=5000)
            return exponential_number(400, 1100, 1.2)
        
        # Save/refresh available axes in bank (save on first bank, refresh on subsequent banks)
        self._save_available_axes_in_bank()
        
        # Determine best axe
        best_axe_for_level = self._determine_best_axe_for_level(woodcutting_level)
        
        if not best_axe_for_level:
            logging.warning(f"[{self.id}] No suitable axe found for woodcutting level {woodcutting_level}")
            self.set_phase("DONE")
            return exponential_number(1000, 3000, 1.2)
        
        # Find best available axe (check bank, inventory, equipment)
        best_available_axe = self._find_best_available_axe(woodcutting_level)
        
        if not best_available_axe:
            logging.info(f"[{self.id}] No axe available (bank/inventory/equipment), plan complete")
            self.set_phase("DONE")
            return exponential_number(1000, 3000, 1.2)
        
        logging.info(f"[{self.id}] Using axe: {best_available_axe} (best for level: {best_axe_for_level})")
        
        # Check if we can equip this axe
        can_equip = self._can_equip_axe(best_available_axe)
        
        # Determine where the axe currently is
        axe_in_inventory = inventory.has_item(best_available_axe)
        axe_equipped = equipment.has_equipped(best_available_axe)
        axe_in_bank = bank.has_item(best_available_axe)
        
        # Smart deposit: deposit everything except the axe (if not equipped), knife, and feathers
        required_items_for_deposit = []
        if not axe_equipped:
            required_items_for_deposit = [best_available_axe]
        # Always keep knife and feathers in inventory
        required_items_for_deposit.extend(["Knife", "Feather", "Forestry kit", "Arrow shaft", "Headless arrow"])
        
        deposit_result = bank.deposit_unwanted_items(required_items_for_deposit, max_unique_for_bulk=3)
        if deposit_result is not None:
            # Wait for deposits to complete
            return exponential_number(200, 600, 1.2)
        
        # Handle axe withdrawal/equipment
        if can_equip:
            # We can equip the axe
            if not axe_equipped:
                # Need to get the axe and equip it
                if not axe_in_inventory:
                    # Withdraw from bank if not in inventory
                    if axe_in_bank:
                        bank.withdraw_item(best_available_axe, withdraw_x=1)
                        wait_until(lambda: inventory.has_item(best_available_axe), max_wait_ms=2000)
                        sleep_exponential(0, 0.6, 1.2)
                    else:
                        logging.warning(f"[{self.id}] Axe {best_available_axe} not found in bank or inventory")
                        return exponential_number(400, 1100, 1.2)
                
                # Equip the axe
                if inventory.has_item(best_available_axe):
                    # Equip the axe using inventory interact
                    bank.interact(best_available_axe, "Wield")
                    wait_until(lambda: equipment.has_equipped(best_available_axe), max_wait_ms=2000)
                    sleep_exponential(0, 0.6, 1.2)
            
            # Ensure inventory only has knife and feathers (axe should be equipped)
            # Deposit everything except knife and feathers
            required_items_for_deposit = ["Knife", "Feather"]
            deposit_result = bank.deposit_unwanted_items(required_items_for_deposit, max_unique_for_bulk=3)
            if deposit_result is not None:
                return exponential_number(200, 600, 1.2)
        else:
            # Cannot equip - keep axe in inventory
            if not axe_in_inventory:
                # Withdraw from bank if not in inventory
                if axe_in_bank:
                    bank.withdraw_item(best_available_axe, withdraw_x=1)
                    wait_until(lambda: inventory.has_item(best_available_axe), max_wait_ms=2000)
                    sleep_exponential(0, 0.6, 1.2)
                else:
                    logging.warning(f"[{self.id}] Axe {best_available_axe} not found in bank")
                    return exponential_number(400, 1100, 1.2)
            
            # Ensure inventory only has axe, knife, and feathers
            required_items_for_deposit = [best_available_axe, "Knife", "Feather"]
            deposit_result = bank.deposit_unwanted_items(required_items_for_deposit, max_unique_for_bulk=3)
            if deposit_result is not None:
                return exponential_number(200, 600, 1.2)
        
        # Withdraw knife and feathers if not already in inventory
        if not inventory.has_item("Knife"):
            if bank.has_item("Knife"):
                bank.withdraw_item("Knife", withdraw_x=1)
                wait_until(lambda: inventory.has_item("Knife"), max_wait_ms=2000)
                sleep_exponential(0, 0.6, 1.2)
            else:
                logging.warning(f"[{self.id}] Knife not found in bank")
                return exponential_number(400, 1100, 1.2)
        
        # Withdraw feathers (withdraw all available, or a large amount)
        if not inventory.has_item("Feather"):
            if bank.has_item("Feather"):
                bank.withdraw_item("Feather", withdraw_all=True)
                wait_until(lambda: inventory.has_item("Feather"), max_wait_ms=2000)
                sleep_exponential(0, 0.6, 1.2)
            else:
                logging.warning(f"[{self.id}] Feathers not found in bank")
                return exponential_number(400, 1100, 1.2)

        # Withdraw feathers (withdraw all available, or a large amount)
        if not inventory.has_item("Forestry kit"):
            if bank.has_item("Forestry kit"):
                bank.withdraw_item("Forestry kit", withdraw_x=1)
                wait_until(lambda: inventory.has_item("Forestry kit"), max_wait_ms=2000)
                sleep_exponential(0, 0.6, 1.2)
            else:
                logging.warning(f"[{self.id}] Forestry kit not found in bank")
        
        # Close bank and transition to appropriate woodcutting phase based on level
        if bank.is_open():
            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=3000)
        
        # Determine which woodcutting phase to use based on level (already retrieved earlier)
        if woodcutting_level >= 45 and fletching_level >= 45:
            logging.info(f"[{self.id}] Banking complete, starting maple woodcutting (level {woodcutting_level})")
            self.set_phase("MAPLE_WOODCUTTING")
        elif woodcutting_level >= 30 and fletching_level >= 30:
            logging.info(f"[{self.id}] Banking complete, starting willow woodcutting (level {woodcutting_level})")
            self.set_phase("WILLOW_WOODCUTTING")
        elif woodcutting_level >= 15 and fletching_level >= 15:
            logging.info(f"[{self.id}] Banking complete, starting oak woodcutting (level {woodcutting_level})")
            self.set_phase("OAK_WOODCUTTING")
        else:
            logging.info(f"[{self.id}] Banking complete, starting regular woodcutting (level {woodcutting_level})")
            self.set_phase("WOODCUTTING")
        return exponential_number(400, 1100, 1.2)
    
    def _determine_best_axe_for_level(self, woodcutting_level: int) -> str:
        """Determine the best axe for the given woodcutting level."""
        try:
            for axe_info in self.axe_options:
                axe_name, woodcutting_req, attack_req = axe_info
                if woodcutting_level >= woodcutting_req:
                    return axe_name
            return ""
        except Exception as e:
            logging.warning(f"[{self.id}] Error determining best axe for level: {e}")
            return ""
    
    def _save_available_axes_in_bank(self) -> None:
        """Save which axes are available in the bank for future reference."""
        try:
            available_axes = []
            for axe_info in self.axe_options:
                axe_name, _, _ = axe_info
                if bank.has_item(axe_name):
                    available_axes.append(axe_name)
            
            self.state["available_axes_in_bank"] = available_axes
            logging.info(f"[{self.id}] Saved available axes in bank: {available_axes}")
        except Exception as e:
            logging.warning(f"[{self.id}] Error saving available axes in bank: {e}")
    
    def _find_best_available_axe(self, woodcutting_level: int) -> str:
        """Find the best available axe we actually have (bank, inventory, or equipment)."""
        try:
            # Check each axe from best to worst that we can use
            for axe_info in self.axe_options:
                axe_name, woodcutting_req, attack_req = axe_info
                if woodcutting_level >= woodcutting_req:
                    # Check if we have this axe
                    if (bank.has_item(axe_name) or 
                        inventory.has_item(axe_name) or 
                        equipment.has_equipped(axe_name)):
                        logging.info(f"[{self.id}] Found available axe: {axe_name}")
                        return axe_name
            return ""
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding available axe: {e}")
            return ""
    
    def _get_current_axe(self) -> str | None:
        """Get the currently equipped or held axe."""
        try:
            # Check equipment first
            for axe_info in self.axe_options:
                axe_name = axe_info[0]
                if equipment.has_equipped(axe_name):
                    return axe_name
            
            # Check inventory
            for axe_info in self.axe_options:
                axe_name = axe_info[0]
                if inventory.has_item(axe_name):
                    return axe_name
            
            return None
        except Exception as e:
            logging.warning(f"[{self.id}] Error getting current axe: {e}")
            return None
    
    def _check_for_better_axe_available(self, current_woodcutting_level: int, current_axe: str) -> bool:
        """
        Check if we've leveled up enough to use a better axe that we have in the bank.
        
        Args:
            current_woodcutting_level: Current woodcutting level
            current_axe: Name of the currently equipped/used axe
            
        Returns:
            bool: True if a better axe is available that we can now use
        """
        try:
            # Get available axes in bank (from saved state)
            available_axes_in_bank = self.state.get("available_axes_in_bank", [])
            if not available_axes_in_bank:
                return False
            
            # Find the index of current axe in axe_options
            current_axe_index = -1
            for i, axe_info in enumerate(self.axe_options):
                if axe_info[0] == current_axe:
                    current_axe_index = i
                    break
            
            if current_axe_index == -1:
                return False
            
            # Check if any better axe (earlier in list) is available in bank and we can now use it
            for i in range(current_axe_index):
                axe_name, woodcutting_req, attack_req = self.axe_options[i]
                
                # Check if this axe is in our saved bank list
                if axe_name in available_axes_in_bank:
                    # Check if we meet the requirements now
                    if current_woodcutting_level >= woodcutting_req:
                        attack_level = player.get_skill_level("attack") or 1
                        if attack_level >= attack_req:
                            logging.info(f"[{self.id}] Better axe available: {axe_name} (WC: {woodcutting_req}, Attack: {attack_req})")
                            return True
            
            return False
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking for better axe: {e}")
            return False
    
    def _process_logs_into_headless_arrows(self) -> bool:
        """
        Process logs into headless arrows by:
        1. Using knife on logs to make arrow shafts
        2. Using arrow shafts on feathers to make headless arrows
        
        Uses widget-based approach with proper animation waiting.
        
        Returns:
            bool: True if processing was successful or no logs to process, False on error
        """
        # Check if we have knife
        if not inventory.has_item("Knife"):
            logging.warning(f"[{self.id}] No knife in inventory, cannot process logs")
            return False
        
        # Check if we have feathers
        if not inventory.has_item("Feather"):
            logging.warning(f"[{self.id}] No feathers in inventory, cannot make headless arrows")
            return False

        # Check if we have logs to process
        log_types = ["Logs", "Oak logs", "Willow logs", "Maple logs"]
        has_logs = False
        log_name = None

        for log_type in log_types:
            if inventory.has_item(log_type):
                has_logs = True
                log_name = log_type
                break

        while has_logs:
            logging.info(f"[{self.id}] Using knife on {log_name} to make arrow shafts")

            result = inventory.use_item_on_item("Knife", log_name)
            if not result:
                logging.warning(f"[{self.id}] Failed to use knife on {log_name}")
                return False

            # Wait for widget to appear
            if not wait_until(lambda: widget_exists(17694720), max_wait_ms=5000):
                logging.warning(f"[{self.id}] Fletching widget did not appear")
                return False

            sleep_exponential(0.1, 0.8, 1.2)
            press_spacebar()
            sleep_exponential(0.3, 0.8, 1.2)

            # Wait for fletching animation to start (animation ID 1248 for fletching)
            # Check if animation is active (not None and not 0/-1)
            if not wait_until(lambda: player.get_player_animation() == "FLETCHING_ARROWSHAFTS",
                              max_wait_ms=2000):
                logging.warning(f"[{self.id}] Fletching animation did not start")
                return False

            # Wait until animation is done OR we have less logs than before OR we're out of logs
            def arrow_shafts_done():
                current_log_count = inv_count(log_name) or 0

                return current_log_count == 0 or chat.can_continue()

            wait_until(arrow_shafts_done, max_wait_ms=60000)  # Max 60 seconds for a full inventory

            for log_type in log_types:
                if inventory.has_item(log_type):
                    has_logs = True
                    log_name = log_type
                    break
                else:
                    has_logs = False

        # Step 2: Use arrow shafts on feathers to make headless arrows
        while inventory.has_item("Arrow shaft") and inventory.has_item("Feather"):
            logging.info(f"[{self.id}] Using arrow shafts on feathers to make headless arrows")

            # Get initial arrow shaft count
            initial_arrow_shaft_count = inv_count("Arrow shaft") or 0

            result = inventory.use_item_on_item("Arrow shaft", "Feather")
            if not result:
                logging.warning(f"[{self.id}] Failed to use arrow shafts on feathers")
                return False

            # Wait for widget to appear
            if not wait_until(lambda: widget_exists(17694720), max_wait_ms=5000):
                logging.warning(f"[{self.id}] Fletching widget did not appear")
                return False

            sleep_exponential(0.1, 0.8, 1.2)
            press_spacebar()
            sleep_exponential(0.3, 0.8, 1.2)

            # Wait for fletching animation to start (animation ID 1248 for fletching)
            # Check if animation is active (not None and not 0/-1)
            if not wait_until(lambda:
                              player.get_player_animation() == "FLETCHING_HEADLESS_ARROWS",
                              max_wait_ms=2000):
                logging.warning(f"[{self.id}] Fletching animation did not start")
                return False

            # Wait until animation is done OR we have 150 less arrow shafts than before OR we're out of arrow shafts
            def headless_arrows_done():
                current_arrow_shaft_count = inv_count("Arrow shaft") or 0
                return (
                        (initial_arrow_shaft_count - current_arrow_shaft_count) >= 150 or
                        current_arrow_shaft_count == 0 or chat.can_continue())

            wait_until(headless_arrows_done, max_wait_ms=60000)  # Max 60 seconds for a full inventory

        return True
    
    def _can_equip_axe(self, axe_name: str) -> bool:
        """Check if we can equip the given axe based on attack level."""
        try:
            # Find attack requirement for this axe
            axe_attack_req = 0
            for axe_info in self.axe_options:
                if axe_info[0] == axe_name:
                    axe_attack_req = axe_info[2]  # attack_req is at index 2
                    break
            
            attack_level = player.get_skill_level("attack") or 1
            can_equip = attack_level >= axe_attack_req
            
            logging.info(f"[{self.id}] Can equip {axe_name}: {attack_level} >= {axe_attack_req} = {can_equip}")
            return can_equip
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking if can equip axe: {e}")
            return False
    
    def _handle_woodcutting(self) -> int:
        """Handle woodcutting phase - cut regular trees until inventory is full."""
        # Close bank if it's open
        if bank.is_open():
            bank.close_bank()
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've leveled up enough to use a better axe
        fletching_level = player.get_skill_level("fletching") or 1
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        current_axe = self._get_current_axe()
        if current_axe and self._check_for_better_axe_available(woodcutting_level, current_axe):
            logging.info(f"[{self.id}] Better axe available after leveling up, going to bank")
            self.set_phase("BANK")
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've reached level 15+ and should switch to oak trees
        if woodcutting_level >= 15 and fletching_level >= 15:
            logging.info(f"[{self.id}] Reached level {woodcutting_level}, switching to oak trees")
            self.set_phase("OAK_WOODCUTTING")
            return exponential_number(400, 1100, 1.2)
        
        # Check if inventory is full - process logs into headless arrows instead of banking
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, processing logs into headless arrows")
            if self._process_logs_into_headless_arrows():
                # Processing successful, continue woodcutting
                return exponential_number(400, 1100, 1.2)
            else:
                # Processing failed, try again next loop
                return exponential_number(400, 1100, 1.2)
        
        # Check if we're already chopping
        current_tree_tile = self.state.get("current_tree_tile")
        if current_tree_tile and player.get_player_animation() == "CHOPPING":
            if objects.object_at_tile_has_action(
                current_tree_tile.get('x'), 
                current_tree_tile.get('y'), 
                current_tree_tile.get('p'), 
                "Tree", 
                "Chop down",
                types=["GAME"],
                exact_match_object=False
            ):
                return 0
        
        # Check if we're in the tree area
        if not in_area(self.tree_area):
            logging.info(f"[{self.id}] Traveling to tree area...")
            go_to(self.tree_area)
            return exponential_number(400, 1100, 1.2)
        
        # Look for trees to cut
        logging.info(f"[{self.id}] Looking for a {self.tree_type} to cut")
        
        # Find the tree object first to get its coordinates
        from helpers.runtime_utils import ipc
        from actions.objects import _find_closest_object_in_area, click_object_no_camera
        from helpers.rects import unwrap_rect
        from helpers.utils import rect_beta_xy
        from actions.player import get_player_plane
        
        tree_obj = _find_closest_object_in_area(
            "Tree",
            self.tree_area,
            types=["GAME"],
            required_action="Chop down"
        )
        
        if tree_obj:
            # Get the tile coordinates from the tree object
            tree_world = tree_obj.get("world", {})
            tree_x = tree_world.get("x")
            tree_y = tree_world.get("y")
            tree_plane = tree_world.get("p", get_player_plane() or 0)
            
            # Get click coordinates from the object
            tree_rect = unwrap_rect(tree_obj.get("clickbox")) or unwrap_rect(tree_obj.get("bounds"))
            tree_click_coords = None
            if tree_rect:
                cx, cy = rect_beta_xy(
                    (tree_rect.get("x", 0), tree_rect.get("x", 0) + tree_rect.get("width", 0),
                     tree_rect.get("y", 0), tree_rect.get("y", 0) + tree_rect.get("height", 0)),
                    alpha=2.0,
                    beta=2.0,
                )
                tree_click_coords = {"x": cx, "y": cy}
            elif isinstance(tree_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(tree_obj.get("canvas", {}).get("y"), (int, float)):
                tree_click_coords = {"x": int(tree_obj["canvas"]["x"]), "y": int(tree_obj["canvas"]["y"])}
            
            # Click the specific tree at the specific tile
            if isinstance(tree_x, int) and isinstance(tree_y, int) and tree_click_coords:
                world_coords = {"x": tree_x, "y": tree_y, "p": tree_plane}
                tree = click_object_no_camera(
                    object_name="Tree",
                    action="Chop down",
                    world_coords=world_coords,
                    click_coords=tree_click_coords
                )
                
                # Save the tree tile coordinates to state
                self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                logging.info(f"[{self.id}] Saved tree tile: ({tree_x}, {tree_y}, {tree_plane})")
                
                if tree:
                    # Wait a bit for chopping to start
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
            else:
                # Fallback: use legacy method if we can't get coordinates
                tree = objects_legacy.click_object_in_area_prefer_no_camera("Tree", self.tree_area, "Chop down")
                if tree:
                    # Try to save tile from the result if available
                    if isinstance(tree, dict) and tree.get("world"):
                        tree_world = tree.get("world", {})
                        tree_x = tree_world.get("x")
                        tree_y = tree_world.get("y")
                        tree_plane = tree_world.get("p", get_player_plane() or 0)
                        if isinstance(tree_x, int) and isinstance(tree_y, int):
                            self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
        else:
            # No tree found
            tree = None
        
        return exponential_number(400, 1100, 1.2)
    
    def _handle_oak_woodcutting(self) -> int:
        """Handle oak woodcutting phase - cut oak trees until inventory is full."""
        # Close bank if it's open
        if bank.is_open():
            bank.close_bank()
            return exponential_number(400, 1100, 1.2)
        
        # Check for forestry events
        event_phase = self._check_for_forestry_event()
        if event_phase:
            logging.info(f"[{self.id}] Forestry event detected: {event_phase}")
            self.set_phase(event_phase)
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've leveled up enough to use a better axe
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        fletching_level = player.get_skill_level("fletching") or 1
        current_axe = self._get_current_axe()
        if current_axe and self._check_for_better_axe_available(woodcutting_level, current_axe):
            logging.info(f"[{self.id}] Better axe available after leveling up, going to bank")
            self.set_phase("BANK")
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've reached level 30+ and should switch to willow trees
        if woodcutting_level >= 30 and fletching_level >= 30:
            logging.info(f"[{self.id}] Reached level {woodcutting_level}, switching to willow trees")
            self.set_phase("WILLOW_WOODCUTTING")
            return exponential_number(400, 1100, 1.2)
        
        # Check if inventory is full - process logs into headless arrows instead of banking
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, processing logs into headless arrows")
            if self._process_logs_into_headless_arrows():
                # Processing successful, continue woodcutting
                return exponential_number(400, 1100, 1.2)
            else:
                # Processing failed, try again next loop
                return exponential_number(400, 1100, 1.2)
        
        # Check if we're already chopping
        current_tree_tile = self.state.get("current_tree_tile")
        if current_tree_tile and player.get_player_animation() == "CHOPPING":
            if objects.object_at_tile_has_action(
                current_tree_tile.get('x'), 
                current_tree_tile.get('y'), 
                current_tree_tile.get('p'), 
                "Oak", 
                "Chop down",
                types=["GAME"],
                exact_match_object=False
            ):
                return 0
        
        # Check if we're on the oak tree tile
        current_x = player.get_x()
        current_y = player.get_y()
        oak_tile_x, oak_tile_y = self.oak_tree_tile
        
        if current_x != oak_tile_x or current_y != oak_tile_y:
            logging.info(f"[{self.id}] Traveling to oak tree tile ({oak_tile_x}, {oak_tile_y})...")
            # Use precision mode for exact tile targeting
            from actions.travel import go_to_tile_precise
            go_to_tile(oak_tile_x, oak_tile_y, arrive_radius=0)
            return exponential_number(400, 1100, 1.2)
        
        # Look for oak trees to cut
        logging.info(f"[{self.id}] Looking for an {self.oak_tree_type} to cut")
        
        # Find the tree object first to get its coordinates
        from helpers.runtime_utils import ipc
        from actions.objects import _find_closest_object_by_distance, click_object_no_camera
        from helpers.rects import unwrap_rect
        from helpers.utils import rect_beta_xy
        from actions.player import get_player_plane
        
        tree_obj = _find_closest_object_by_distance(
            "Oak tree",
            types=["GAME"],
            required_action="Chop down"
        )
        
        if tree_obj:
            # Get the tile coordinates from the tree object
            tree_world = tree_obj.get("world", {})
            tree_x = tree_world.get("x")
            tree_y = tree_world.get("y")
            tree_plane = tree_world.get("p", get_player_plane() or 0)
            
            # Get click coordinates from the object
            tree_rect = unwrap_rect(tree_obj.get("clickbox")) or unwrap_rect(tree_obj.get("bounds"))
            tree_click_coords = None
            if tree_rect:
                cx, cy = rect_beta_xy(
                    (tree_rect.get("x", 0), tree_rect.get("x", 0) + tree_rect.get("width", 0),
                     tree_rect.get("y", 0), tree_rect.get("y", 0) + tree_rect.get("height", 0)),
                    alpha=2.0,
                    beta=2.0,
                )
                tree_click_coords = {"x": cx, "y": cy}
            elif isinstance(tree_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(tree_obj.get("canvas", {}).get("y"), (int, float)):
                tree_click_coords = {"x": int(tree_obj["canvas"]["x"]), "y": int(tree_obj["canvas"]["y"])}
            
            # Click the specific tree at the specific tile
            if isinstance(tree_x, int) and isinstance(tree_y, int) and tree_click_coords:
                world_coords = {"x": tree_x, "y": tree_y, "p": tree_plane}
                tree = click_object_no_camera(
                    object_name="Oak tree",
                    action="Chop down",
                    world_coords=world_coords,
                    click_coords=tree_click_coords
                )
                
                # Save the tree tile coordinates to state
                self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                logging.info(f"[{self.id}] Saved tree tile: ({tree_x}, {tree_y}, {tree_plane})")
                
                if tree:
                    # Wait a bit for chopping to start
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
            else:
                # Fallback: use legacy method if we can't get coordinates
                tree = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Oak", "Chop down")
                if tree:
                    # Try to save tile from the result if available
                    if isinstance(tree, dict) and tree.get("world"):
                        tree_world = tree.get("world", {})
                        tree_x = tree_world.get("x")
                        tree_y = tree_world.get("y")
                        tree_plane = tree_world.get("p", get_player_plane() or 0)
                        if isinstance(tree_x, int) and isinstance(tree_y, int):
                            self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
        else:
            # No tree found
            tree = None
        
        return exponential_number(400, 1100, 1.2)
    
    def _handle_willow_woodcutting(self) -> int:
        """Handle willow woodcutting phase - cut willow trees until inventory is full."""
        # Close bank if it's open
        if bank.is_open():
            bank.close_bank()
            return exponential_number(400, 1100, 1.2)
        
        # Check for forestry events
        event_phase = self._check_for_forestry_event()
        if event_phase:
            logging.info(f"[{self.id}] Forestry event detected: {event_phase}")
            self.set_phase(event_phase)
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've leveled up enough to use a better axe
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        fletching_level = player.get_skill_level("fletching") or 1
        current_axe = self._get_current_axe()
        if current_axe and self._check_for_better_axe_available(woodcutting_level, current_axe):
            logging.info(f"[{self.id}] Better axe available after leveling up, going to bank")
            self.set_phase("BANK")
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've reached level 45+ and should switch to maple trees
        if woodcutting_level >= 45 and fletching_level >= 45:
            logging.info(f"[{self.id}] Reached level {woodcutting_level}, switching to maple trees")
            self.set_phase("MAPLE_WOODCUTTING")
            return exponential_number(400, 1100, 1.2)
        
        # Check if inventory is full - process logs into headless arrows instead of banking
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, processing logs into headless arrows")
            if self._process_logs_into_headless_arrows():
                # Processing successful, continue woodcutting
                return exponential_number(400, 1100, 1.2)
            else:
                # Processing failed, try again next loop
                return exponential_number(400, 1100, 1.2)
        
        # Check if we're already chopping
        current_tree_tile = self.state.get("current_tree_tile")
        if current_tree_tile and player.get_player_animation() == "CHOPPING":
            if objects.object_at_tile_has_action(
                current_tree_tile.get('x'), 
                current_tree_tile.get('y'), 
                current_tree_tile.get('p'), 
                "Willow tree", 
                "Chop down",
                types=["GAME"],
                exact_match_object=False
            ):
                return 0
        
        # Check if we're in the willow tree area
        if not in_area(self.willow_tree_area):
            logging.info(f"[{self.id}] Traveling to willow tree area...")
            go_to(self.willow_tree_area)
            return exponential_number(400, 1100, 1.2)
        
        # Look for willow trees to cut
        logging.info(f"[{self.id}] Looking for a {self.willow_tree_type} to cut")
        
        # Find the tree object first to get its coordinates
        from helpers.runtime_utils import ipc
        from actions.objects import _find_closest_object_in_area, click_object_no_camera
        from helpers.rects import unwrap_rect
        from helpers.utils import rect_beta_xy
        from actions.player import get_player_plane
        
        tree_obj = _find_closest_object_in_area(
            "Willow tree",
            self.willow_tree_area,
            types=["GAME"],
            required_action="Chop down"
        )
        
        if tree_obj:
            # Get the tile coordinates from the tree object
            tree_world = tree_obj.get("world", {})
            tree_x = tree_world.get("x")
            tree_y = tree_world.get("y")
            tree_plane = tree_world.get("p", get_player_plane() or 0)
            
            # Get click coordinates from the object
            tree_rect = unwrap_rect(tree_obj.get("clickbox")) or unwrap_rect(tree_obj.get("bounds"))
            tree_click_coords = None
            if tree_rect:
                cx, cy = rect_beta_xy(
                    (tree_rect.get("x", 0), tree_rect.get("x", 0) + tree_rect.get("width", 0),
                     tree_rect.get("y", 0), tree_rect.get("y", 0) + tree_rect.get("height", 0)),
                    alpha=2.0,
                    beta=2.0,
                )
                tree_click_coords = {"x": cx, "y": cy}
            elif isinstance(tree_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(tree_obj.get("canvas", {}).get("y"), (int, float)):
                tree_click_coords = {"x": int(tree_obj["canvas"]["x"]), "y": int(tree_obj["canvas"]["y"])}
            
            # Click the specific tree at the specific tile
            if isinstance(tree_x, int) and isinstance(tree_y, int) and tree_click_coords:
                world_coords = {"x": tree_x, "y": tree_y, "p": tree_plane}
                tree = click_object_no_camera(
                    object_name="Willow tree",
                    action="Chop down",
                    world_coords=world_coords,
                    click_coords=tree_click_coords
                )
                
                # Save the tree tile coordinates to state
                self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                logging.info(f"[{self.id}] Saved tree tile: ({tree_x}, {tree_y}, {tree_plane})")
                
                if tree:
                    # Wait a bit for chopping to start
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
            else:
                # Fallback: use legacy method if we can't get coordinates
                tree = objects_legacy.click_object_in_area_prefer_no_camera("Willow tree", self.willow_tree_area, "Chop down")
                if tree:
                    # Try to save tile from the result if available
                    if isinstance(tree, dict) and tree.get("world"):
                        tree_world = tree.get("world", {})
                        tree_x = tree_world.get("x")
                        tree_y = tree_world.get("y")
                        tree_plane = tree_world.get("p", get_player_plane() or 0)
                        if isinstance(tree_x, int) and isinstance(tree_y, int):
                            self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
        else:
            # No tree found
            tree = None
        
        return exponential_number(400, 1100, 1.2)
    
    def _handle_maple_woodcutting(self) -> int:
        """Handle maple woodcutting phase - cut maple trees until inventory is full."""
        # Close bank if it's open
        if bank.is_open():
            bank.close_bank()
            return exponential_number(400, 1100, 1.2)
        
        # Check for forestry events
        event_phase = self._check_for_forestry_event()
        if event_phase:
            logging.info(f"[{self.id}] Forestry event detected: {event_phase}")
            self.set_phase(event_phase)
            return exponential_number(400, 1100, 1.2)
        
        # Check if we've leveled up enough to use a better axe
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        current_axe = self._get_current_axe()
        if current_axe and self._check_for_better_axe_available(woodcutting_level, current_axe):
            logging.info(f"[{self.id}] Better axe available after leveling up, going to bank")
            self.set_phase("BANK")
            return exponential_number(400, 1100, 1.2)
        
        # Check if inventory is full - process logs into headless arrows instead of banking
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, processing logs into headless arrows")
            if self._process_logs_into_headless_arrows():
                # Processing successful, continue woodcutting
                return exponential_number(400, 1100, 1.2)
            else:
                # Processing failed, try again next loop
                return exponential_number(400, 1100, 1.2)
        
        # Check if we're already chopping
        current_tree_tile = self.state.get("current_tree_tile")
        if current_tree_tile and player.get_player_animation() == "CHOPPING":
            if objects.object_at_tile_has_action(
                current_tree_tile.get('x'), 
                current_tree_tile.get('y'), 
                current_tree_tile.get('p'), 
                "Maple", 
                "Chop down",
                types=["GAME"],
                exact_match_object=False
            ):
                return 0
        
        # Check if we're in the maple tree area
        if not in_area(self.maple_tree_area):
            logging.info(f"[{self.id}] Traveling to maple tree area...")
            go_to(self.maple_tree_area)
            return exponential_number(400, 1100, 1.2)
        
        # Look for maple trees to cut
        logging.info(f"[{self.id}] Looking for a {self.maple_tree_type} to cut")
        
        # Find the tree object first to get its coordinates
        from helpers.runtime_utils import ipc
        from actions.objects import _find_closest_object_in_area, click_object_no_camera
        from helpers.rects import unwrap_rect
        from helpers.utils import rect_beta_xy
        from actions.player import get_player_plane
        
        tree_obj = _find_closest_object_in_area(
            "Maple",
            self.maple_tree_area,
            types=["GAME"],
            required_action="Chop down"
        )
        
        if tree_obj:
            # Get the tile coordinates from the tree object
            tree_world = tree_obj.get("world", {})
            tree_x = tree_world.get("x")
            tree_y = tree_world.get("y")
            tree_plane = tree_world.get("p", get_player_plane() or 0)
            
            # Get click coordinates from the object
            tree_rect = unwrap_rect(tree_obj.get("clickbox")) or unwrap_rect(tree_obj.get("bounds"))
            tree_click_coords = None
            if tree_rect:
                cx, cy = rect_beta_xy(
                    (tree_rect.get("x", 0), tree_rect.get("x", 0) + tree_rect.get("width", 0),
                     tree_rect.get("y", 0), tree_rect.get("y", 0) + tree_rect.get("height", 0)),
                    alpha=2.0,
                    beta=2.0,
                )
                tree_click_coords = {"x": cx, "y": cy}
            elif isinstance(tree_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(tree_obj.get("canvas", {}).get("y"), (int, float)):
                tree_click_coords = {"x": int(tree_obj["canvas"]["x"]), "y": int(tree_obj["canvas"]["y"])}
            
            # Click the specific tree at the specific tile
            if isinstance(tree_x, int) and isinstance(tree_y, int) and tree_click_coords:
                world_coords = {"x": tree_x, "y": tree_y, "p": tree_plane}
                tree = click_object_no_camera(
                    object_name="Maple",
                    action="Chop down",
                    world_coords=world_coords,
                    click_coords=tree_click_coords
                )
                
                # Save the tree tile coordinates to state
                self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                logging.info(f"[{self.id}] Saved tree tile: ({tree_x}, {tree_y}, {tree_plane})")
                
                if tree:
                    # Wait a bit for chopping to start
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
            else:
                # Fallback: use legacy method if we can't get coordinates
                tree = objects_legacy.click_object_in_area_prefer_no_camera("Maple", self.maple_tree_area, "Chop down")
                if tree:
                    # Try to save tile from the result if available
                    if isinstance(tree, dict) and tree.get("world"):
                        tree_world = tree.get("world", {})
                        tree_x = tree_world.get("x")
                        tree_y = tree_world.get("y")
                        tree_plane = tree_world.get("p", get_player_plane() or 0)
                        if isinstance(tree_x, int) and isinstance(tree_y, int):
                            self.state["current_tree_tile"] = {"x": tree_x, "y": tree_y, "p": tree_plane}
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or chat.can_continue(), max_wait_ms=8000)
        else:
            # No tree found
            tree = None
        
        return exponential_number(400, 1100, 1.2)
    
    # ============================================================================
    # Forestry Event Detection and Handlers
    # ============================================================================
    
    def _check_for_forestry_event(self) -> str | None:
        """
        Check if a forestry event has spawned.
        Returns the event phase name if detected, None otherwise.
        """
        # Check for event objects/spawns
        # Flowering Tree: Look for "Flowering bush" or "Pollinated bush" / "Unpollinated bush"
        if objects.object_exists("Flowering bush") or objects.object_exists("Pollinated bush") or objects.object_exists("Unpollinated bush"):
            return "FORESTRY_FLOWERING_TREE"
        
        # Leprechaun: Look for "Woodcutting Leprechaun" or "End of rainbow"
        if objects.object_exists("Woodcutting Leprechaun") or objects.object_exists("End of rainbow"):
            return "FORESTRY_LEPRECHAUN"
        
        # Beehive: Look for "Unfinished beehive"
        if objects.object_exists("Unfinished beehive"):
            return "FORESTRY_BEEHIVE"
        
        # Friendly Ent: Look for "Entling"
        if objects.object_exists("Entling"):
            return "FORESTRY_FRIENDLY_ENT"
        
        # Poachers: Look for "Fox trap" or "Poacher"
        if objects.object_exists("Fox trap") or objects.object_exists("Poacher"):
            return "FORESTRY_POACHERS"
        
        # Enchantment Ritual: Look for "Dryad" or "Ritual circle"
        if objects.object_exists("Dryad") or objects.object_exists("Ritual circle"):
            return "FORESTRY_ENCHANTMENT_RITUAL"
        
        # Pheasant Control: Look for "Pheasant nest" or "Pheasant" or "Freaky Forester"
        if objects.object_exists("Pheasant nest") or objects.object_exists("Pheasant") or objects.object_exists("Freaky Forester"):
            return "FORESTRY_PHEASANT_CONTROL"
        
        # Rising Roots: Look for "Root" or "Anima root" or "Imbued root"
        if objects.object_exists("Root") or objects.object_exists("Anima root") or objects.object_exists("Imbued root"):
            return "FORESTRY_RISING_ROOTS"
        
        # Struggling Sapling: Look for "Struggling sapling" or "Resource pile"
        if objects.object_exists("Struggling sapling") or objects.object_exists("Resource pile"):
            return "FORESTRY_STRUGGLING_SAPLING"
        
        return None
    
    def _get_previous_woodcutting_phase(self) -> str:
        """Get the previous woodcutting phase to return to after event."""
        woodcutting_level = player.get_skill_level("woodcutting") or 1
        if woodcutting_level >= 45:
            return "MAPLE_WOODCUTTING"
        elif woodcutting_level >= 30:
            return "WILLOW_WOODCUTTING"
        elif woodcutting_level >= 15:
            return "OAK_WOODCUTTING"
        else:
            return "WOODCUTTING"
    
    def _handle_flowering_tree_event(self) -> int:
        """
        Flowering Tree Event:
        - Find pollinated or unpollinated bush (part of active matching pair)
        - Click Tend to collect Strange pollen
        - Find matching counterpart (opposite state)
        - Tend that bush to apply pollen, extract again
        - Repeat until pair changes/event ends
        """
        # Check if event is still active
        if not (objects.object_exists("Flowering bush") or objects.object_exists("Pollinated bush") or objects.object_exists("Unpollinated bush")):
            logging.info(f"[{self.id}] Flowering Tree event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Find a bush (pollinated or unpollinated)
        bush = objects_legacy.click_object_closest_by_distance_prefer_no_camera(
            ["Flowering bush", "Pollinated bush", "Unpollinated bush"],
            "Tend"
        )
        
        if bush:
            # Wait for action to complete
            wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=3000)
            time.sleep(0.3)
            
            # Now find the matching counterpart (opposite state)
            # If we clicked pollinated, find unpollinated, and vice versa
            # This is a simplified approach - may need refinement based on actual object states
            counterpart = objects_legacy.click_object_closest_by_distance_prefer_no_camera(
                ["Flowering bush", "Pollinated bush", "Unpollinated bush"],
                "Tend"
            )
            
            if counterpart:
                wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=3000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_leprechaun_event(self) -> int:
        """
        Woodcutting Leprechaun Event:
        - Watch for "end of rainbow" tiles appearing
        - Stand on each end-of-rainbow as it appears to gain Leprechaun's luck stacks
        - (Optional) Use leprechaun as deposit box
        - After event, continue woodcutting (luck converts to XP + anima-infused bark)
        """
        # Check if event is still active
        if not (objects.object_exists("Woodcutting Leprechaun") or objects.object_exists("End of rainbow")):
            logging.info(f"[{self.id}] Leprechaun event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Look for "End of rainbow" tiles and stand on them
        rainbow = objects_legacy.click_object_closest_by_distance_prefer_no_camera("End of rainbow", None)
        
        if rainbow:
            # Wait to stand on the tile
            wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
        
        # Optional: Use leprechaun as deposit box if inventory is getting full
        # Check if inventory is getting full (less than 8 empty slots)
        if inventory.get_empty_slots_count() < 8:
            leprechaun = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Woodcutting Leprechaun", "Deposit")
            if leprechaun:
                wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_beehive_event(self) -> int:
        """
        Beehive Event:
        - Keep logs in inventory (or log basket)
        - When Unfinished beehives appear, use logs on unfinished beehive
        - Keep adding logs until beehive is fully built
        - Move to next unfinished beehive and repeat until event ends
        """
        # Check if event is still active
        if not objects.object_exists("Unfinished beehive"):
            logging.info(f"[{self.id}] Beehive event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Check if we have logs in inventory
        if not inventory.has_item("Logs") and not inventory.has_item("Oak logs") and not inventory.has_item("Willow logs") and not inventory.has_item("Maple logs"):
            logging.warning(f"[{self.id}] No logs available for beehive event")
            # Could try to get logs from log basket if available
            return exponential_number(400, 1100, 1.2)
        
        # Find unfinished beehive
        beehive = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Unfinished beehive", None)
        
        if beehive:
            # Use logs on beehive (this would need to be implemented as a use-item-on-object action)
            # For now, this is a scaffold - would need to implement use_item_on_object
            logging.info(f"[{self.id}] Using logs on beehive (scaffold - needs use-item-on-object implementation)")
            wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=3000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_friendly_ent_event(self) -> int:
        """
        Friendly Ent Event:
        - When entlings appear, click one and read what it asks for
        - Right-click entling and choose matching prune option:
          * "Breezy at the back!" → Prune-back
          * "A leafy mullet!" → Prune-top + Prune-sides
          * "Short back and sides!" → Prune-back + Prune-sides
          * "Short on top!" → Prune-top
        - Prune each entling multiple times until satisfied
        - Avoid wrong pruning (can stun you)
        """
        # Check if event is still active
        if not npc.closest_npc_by_name("Entling"):
            logging.info(f"[{self.id}] Friendly Ent event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Find an entling
        entling = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Entling", None)
        
        if entling:
            # Read the entling's request (would need to check chat/interface)
            # For now, scaffold - would need to implement chat reading or interface checking
            logging.info(f"[{self.id}] Found entling (scaffold - needs chat/interface reading for request)")
            
            # Try different prune options based on request
            # This is simplified - actual implementation would need to read the request first
            prune_options = ["Prune-back", "Prune-top", "Prune-sides"]
            for option in prune_options:
                pruned = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Entling", option)
                if pruned:
                    wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
                    break
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_poachers_event(self) -> int:
        """
        Poachers Event:
        - When poachers start placing fox traps, watch for newly-laid traps
        - Disarm each fox trap as soon as it appears
        - Repeat until event ends
        """
        # Check if event is still active
        if not (objects.object_exists("Fox trap") or objects.object_exists("Poacher")):
            logging.info(f"[{self.id}] Poachers event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Find and disarm fox traps
        trap = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Fox trap", "Disarm")
        
        if trap:
            wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_enchantment_ritual_event(self) -> int:
        """
        Enchantment Ritual Event:
        - When Dryad spawns ritual circles, look at 5 markings on ground
        - Identify the odd one out (only circle whose shape+color combination doesn't match)
        - Stand on the odd one out until it charges/completes
        - Repeat each round until ritual finishes
        """
        # Check if event is still active
        if not (objects.object_exists("Dryad") or objects.object_exists("Ritual circle")):
            logging.info(f"[{self.id}] Enchantment Ritual event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Find ritual circles and identify the odd one out
        # This is a scaffold - would need to implement visual analysis of circle shapes/colors
        logging.info(f"[{self.id}] Enchantment Ritual active (scaffold - needs visual analysis of circles)")
        
        # For now, try to click on ritual circles
        circle = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Ritual circle", None)
        
        if circle:
            # Stand on the circle
            wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_pheasant_control_event(self) -> int:
        """
        Pheasant Control Event:
        - When nests + pheasants spawn, find unoccupied nest (no pheasant sitting on it)
        - Take pheasant egg from empty nest (can only carry one at a time)
        - Bring egg to Freaky Forester to turn in
        - Empty nest will change - find new unoccupied nest and repeat
        - Misclicking/too slow can stun you
        """
        # Check if event is still active
        if not (objects.object_exists("Pheasant nest") or objects.object_exists("Pheasant") or objects.object_exists("Freaky Forester")):
            logging.info(f"[{self.id}] Pheasant Control event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Check if we're already carrying an egg
        if inventory.has_item("Pheasant egg"):
            # Bring egg to Freaky Forester
            forester = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Freaky Forester", None)
            if forester:
                wait_until(lambda: not inventory.has_item("Pheasant egg"), max_wait_ms=3000)
        else:
            # Find unoccupied nest (no pheasant on it)
            # This is a scaffold - would need to detect if nest is occupied
            nest = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Pheasant nest", "Take")
            
            if nest:
                wait_until(lambda: inventory.has_item("Pheasant egg"), max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_rising_roots_event(self) -> int:
        """
        Rising Roots Event:
        - When roots erupt around the tree, start chopping any roots for bark/XP
        - Prioritize anima/imbued root (green trim/green veins) when it appears
        - If roots shift position, move and keep chopping until event ends
        """
        # Check if event is still active
        if not (objects.object_exists("Root") or objects.object_exists("Anima root") or objects.object_exists("Imbued root")):
            logging.info(f"[{self.id}] Rising Roots event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Prioritize anima/imbued root first
        from helpers.runtime_utils import ipc
        from actions.player import get_player_plane
        from actions.objects import click_object_no_camera
        from helpers.rects import unwrap_rect
        from helpers.utils import rect_beta_xy
        
        # Find the root object first to get its coordinates
        anima_root_obj = None
        anima_root_resp = ipc.find_object("Tree roots", types=["GAME"], exact_match=False)
        if anima_root_resp and anima_root_resp.get("ok") and anima_root_resp.get("found"):
            anima_root_obj = anima_root_resp.get("object")
            # Verify it has the "Chop" action
            if anima_root_obj:
                obj_actions = [str(a).strip().lower() for a in (anima_root_obj.get("actions") or []) if a]
                if "chop down" not in obj_actions:
                    anima_root_obj = None
        
        if anima_root_obj:
            # Get the tile coordinates of the root
            root_world = anima_root_obj.get("world", {})
            root_x = root_world.get("x")
            root_y = root_world.get("y")
            root_plane = root_world.get("p", get_player_plane() or 0)
            
            # Get click coordinates from the object
            root_rect = unwrap_rect(anima_root_obj.get("clickbox")) or unwrap_rect(anima_root_obj.get("bounds"))
            root_click_coords = None
            if root_rect:
                cx, cy = rect_beta_xy(
                    (root_rect.get("x", 0), root_rect.get("x", 0) + root_rect.get("width", 0),
                     root_rect.get("y", 0), root_rect.get("y", 0) + root_rect.get("height", 0)),
                    alpha=2.0,
                    beta=2.0,
                )
                root_click_coords = {"x": cx, "y": cy}
            elif isinstance(anima_root_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(anima_root_obj.get("canvas", {}).get("y"), (int, float)):
                root_click_coords = {"x": int(anima_root_obj["canvas"]["x"]), "y": int(anima_root_obj["canvas"]["y"])}
            
            # Click the specific root at the specific tile
            if isinstance(root_x, int) and isinstance(root_y, int) and root_click_coords:
                world_coords = {"x": root_x, "y": root_y, "p": root_plane}
                anima_root = click_object_no_camera(
                    object_name="Tree roots",
                    action="Chop",
                    world_coords=world_coords,
                    click_coords=root_click_coords
                )
                
                if anima_root:
                    # Wait until that specific tile no longer has tree roots with "Chop" action
                    wait_until(
                        lambda: not objects.object_at_tile_has_action(
                            root_x, root_y, root_plane, 
                            ["Tree roots", "Anima root"], 
                            "Chop",
                            types=["GAME"],
                            exact_match_object=False
                        ),
                        max_wait_ms=20000
                    )
                else:
                    # Fallback: wait for animation or object to disappear
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or not objects.object_exists("Anima root"), max_wait_ms=20000)
            else:
                # Fallback: use legacy method if we can't get coordinates
                anima_root = objects_legacy.click_object_closest_by_distance_prefer_no_camera(
                    "Tree roots",
                    "Chop"
                )
                if anima_root:
                    wait_until(lambda: player.get_player_animation() == "CHOPPING" or not objects.object_exists("Anima root"), max_wait_ms=20000)
        else:
            # Chop regular roots
            # Find the root object first to get its coordinates
            root_obj = None
            root_resp = ipc.find_object("Root", types=["GAME"], exact_match=False)
            if root_resp and root_resp.get("ok") and root_resp.get("found"):
                root_obj = root_resp.get("object")
                # Verify it has the "Chop" action
                if root_obj:
                    obj_actions = [str(a).strip().lower() for a in (root_obj.get("actions") or []) if a]
                    if "chop" not in obj_actions:
                        root_obj = None
            
            if root_obj:
                # Get the tile coordinates of the root
                root_world = root_obj.get("world", {})
                root_x = root_world.get("x")
                root_y = root_world.get("y")
                root_plane = root_world.get("p", get_player_plane() or 0)
                
                # Get click coordinates from the object
                root_rect = unwrap_rect(root_obj.get("clickbox")) or unwrap_rect(root_obj.get("bounds"))
                root_click_coords = None
                if root_rect:
                    cx, cy = rect_beta_xy(
                        (root_rect.get("x", 0), root_rect.get("x", 0) + root_rect.get("width", 0),
                         root_rect.get("y", 0), root_rect.get("y", 0) + root_rect.get("height", 0)),
                        alpha=2.0,
                        beta=2.0,
                    )
                    root_click_coords = {"x": cx, "y": cy}
                elif isinstance(root_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(root_obj.get("canvas", {}).get("y"), (int, float)):
                    root_click_coords = {"x": int(root_obj["canvas"]["x"]), "y": int(root_obj["canvas"]["y"])}
                
                # Click the specific root at the specific tile
                if isinstance(root_x, int) and isinstance(root_y, int) and root_click_coords:
                    world_coords = {"x": root_x, "y": root_y, "p": root_plane}
                    root = click_object_no_camera(
                        object_name="Root",
                        action="Chop",
                        world_coords=world_coords,
                        click_coords=root_click_coords
                    )
                    
                    if root:
                        # Wait until that specific tile no longer has roots with "Chop" action
                        wait_until(
                            lambda: not objects.object_at_tile_has_action(
                                root_x, root_y, root_plane,
                                "Root",
                                "Chop",
                                types=["GAME"],
                                exact_match_object=False
                            ),
                            max_wait_ms=20000
                        )
                    else:
                        # Fallback: wait for animation
                        wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=2000)
                else:
                    # Fallback: use legacy method if we can't get coordinates
                    root = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Root", "Chop")
                    if root:
                        wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_struggling_sapling_event(self) -> int:
        """
        Struggling Sapling Event:
        - When sapling appears with resource piles, collect from piles to make mulch (gather 3 times per mulch)
        - Feed mulch to sapling; game tells which ingredient position was correct (1st/2nd/3rd)
        - Use feedback to build correct 3-ingredient order
        - Keep feeding correct mulch until sapling is fully helped
        """
        # Check if event is still active
        if not (objects.object_exists("Struggling sapling") or objects.object_exists("Resource pile")):
            logging.info(f"[{self.id}] Struggling Sapling event ended, returning to woodcutting")
            self.set_phase(self._get_previous_woodcutting_phase())
            return exponential_number(400, 1100, 1.2)
        
        # Check if we have mulch ready
        if inventory.has_item("Packed Mulch"):
            # Feed mulch to sapling
            sapling = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Struggling sapling", "Feed")
            if sapling:
                # Wait and check for feedback on which position was correct
                # This is a scaffold - would need to read chat/interface for feedback
                wait_until(lambda: not inventory.has_item("Mulch") or chat.can_continue(), max_wait_ms=3000)
                # Read feedback from chat if available
                logging.info(f"[{self.id}] Fed mulch to sapling (scaffold - needs chat reading for feedback)")
        else:
            # Collect from resource piles (gather 3 times per mulch)
            # Track how many times we've gathered
            if "mulch_gather_count" not in self.state:
                self.state["mulch_gather_count"] = 0
            
            pile = objects_legacy.click_object_closest_by_distance_prefer_no_camera("Resource pile", "Gather")
            if pile:
                self.state["mulch_gather_count"] = (self.state.get("mulch_gather_count", 0) + 1) % 3
                wait_until(lambda: player.get_player_animation() != "CHOPPING", max_wait_ms=2000)
                if self.state["mulch_gather_count"] == 0:
                    # Should have mulch now
                    wait_until(lambda: inventory.has_item("Mulch"), max_wait_ms=2000)
        
        return exponential_number(200, 500, 1.2)
    
    def _handle_done(self) -> int:
        """Handle done phase - plan is complete."""
        logging.info(f"[{self.id}] Plan complete")
        return exponential_number(1000, 3000, 1.2)

