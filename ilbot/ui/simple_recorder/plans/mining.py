#!/usr/bin/env python3
"""
Mining Plan
===========

This plan handles mining with a single initial bank phase.
- Banks once at the start to withdraw all available pickaxes
- Mines copper/tin ore until level 15, dropping ores when inventory is full
- Then mines iron ore, dropping ores when inventory is full
- No banking after initial setup
"""

import logging
from pathlib import Path
import sys

from ..actions import objects, player, inventory, bank, equipment
from ..actions.inventory import interact as inventory_interact
from ..actions.timing import wait_until
from ..actions.travel import in_area, go_to, travel_to_bank
from ..helpers.utils import sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from ..helpers.camera import setup_camera_optimal
from ..helpers.phase_utils import set_phase_with_camera


class MiningPlan(Plan):
    """Mining plan with single initial bank phase."""
    
    id = "MINING"
    label = "Mining Plan"
    
    def __init__(self):
        self.state = {"phase": "MINING_COPPER_TIN"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Configuration
        self.copper_tin_area = "AL_KHARID_MINE"  # TODO: Set actual copper/tin location
        self.iron_area = "AL_KHARID_MINE"  # TODO: Set actual iron location
        self.target_level = 15  # Level to reach before switching to iron
        
        # Pickaxe options in order of preference (best to worst)
        # Format: (pickaxe_name, mining_level, attack_level, defence_level)
        self.pickaxe_options = [
            ("Dragon pickaxe", 61, 60, 1),
            ("Rune pickaxe", 41, 40, 1),
            ("Adamant pickaxe", 31, 30, 1),
            ("Mithril pickaxe", 21, 20, 1),
            ("Black pickaxe", 11, 10, 1),
            ("Steel pickaxe", 6, 5, 1),
            ("Iron pickaxe", 1, 1, 1),
            ("Bronze pickaxe", 1, 1, 1)
        ]
        
        # Create simple bank plan - will be configured in _handle_bank
        self.bank_plan = BankPlanSimple(
            bank_area=None,  # Use closest bank
            required_items=[],  # Will be set dynamically
            deposit_all=True,
            equip_items={}  # Will be set dynamically
        )
        
        # Flag to track if bank plan has been configured (only once)
        self.bank_plan_configured = False
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Copper/Tin area: {self.copper_tin_area}")
        logging.info(f"[{self.id}] Iron area: {self.iron_area}")
        logging.info(f"[{self.id}] Target level: {self.target_level}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK")
        logged_in = player.logged_in()
        if not logged_in:
            logging.info(f"player not logged in")
            player.login()
            return self.loop_interval_ms

        match(phase):
            case "BANK":
                return self._handle_bank(ui)

            case "MINING_COPPER_TIN":
                return self._handle_mining_copper_tin()

            case "MINING_IRON":
                return self._handle_mining_iron()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    
    def _handle_bank(self, ui) -> int:
        """Handle initial banking phase - withdraw all available pickaxes."""
        if not travel_to_bank():
            return self.loop_interval_ms
        if not bank.is_open():
            bank.open_bank()
            return self.loop_interval_ms
        
        # Configure bank plan only once
        if not self.bank_plan_configured:
            # Get mining level to determine which pickaxes we can use
            mining_level = player.get_skill_level("mining") or 1
            attack_level = player.get_skill_level("attack") or 1
            
            logging.info(f"[{self.id}] Mining level: {mining_level}, Attack level: {attack_level}")
            
            # Find all pickaxes available in bank
            available_pickaxes = []
            for pickaxe_info in self.pickaxe_options:
                pickaxe_name, mining_req, attack_req, defence_req = pickaxe_info
                if bank.has_item(pickaxe_name):
                    available_pickaxes.append((pickaxe_name, mining_req, attack_req))
                    logging.info(f"[{self.id}] Found {pickaxe_name} in bank")
                    if mining_level >= mining_req:
                        break
            
            if not available_pickaxes:
                logging.warning(f"[{self.id}] No pickaxes found in bank!")
                bank.close_bank()
                self.set_phase("MINING_COPPER_TIN")
                return self.loop_interval_ms
            
            # Withdraw all available pickaxes
            required_items = []
            equip_items = {}
            best_pickaxe_for_equip = None
            
            for pickaxe_name, mining_req, attack_req in available_pickaxes:
                # If we can use it and it's the best one we can equip, mark it for equipping
                if mining_level >= mining_req and attack_level >= attack_req:
                    if best_pickaxe_for_equip is None:
                        best_pickaxe_for_equip = pickaxe_name
                        break

                required_items.append({"name": pickaxe_name, "quantity": 1})
            
            # Configure bank plan
            self.bank_plan.required_items = required_items
            if best_pickaxe_for_equip:
                self.bank_plan.equip_items = {"weapon": [best_pickaxe_for_equip]}
                logging.info(f"[{self.id}] Will equip {best_pickaxe_for_equip}")
            else:
                self.bank_plan.equip_items = {}
                logging.info(f"[{self.id}] Cannot equip any pickaxes, will keep in inventory")
            
            # Mark as configured so this doesn't run again
            self.bank_plan_configured = True
        
        # Execute bank plan
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            if bank.is_open():
                bank.close_bank()
                if not wait_until(bank.is_closed, max_wait_ms=3000):
                    return self.loop_interval_ms
            self.set_phase("MINING_COPPER_TIN")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on banking (TRAVELING, BANKING, EQUIPPING, etc.)
            return bank_status
    
    def _handle_mining_copper_tin(self) -> int:
        """Handle mining copper/tin ore until level 15."""
        # Check if we've reached level 15
        mining_level = player.get_skill_level("mining") or 1
        if mining_level >= self.target_level:
            logging.info(f"[{self.id}] Reached level {mining_level}, switching to iron mining")
            self.set_phase("MINING_IRON")
            return self.loop_interval_ms
        
        # Toggle run if needed
        if player.get_run_energy() > 2000 and not player.is_run_on():
            player.toggle_run()
        
        # Drop ores if inventory is full
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, dropping ores...")
            self._drop_all_ores()
            return self.loop_interval_ms
        
        # Check if we're already mining
        if player.get_player_animation() == "MINING":
            return self.loop_interval_ms

        # Check if we're in the copper/tin area
        if not in_area(self.copper_tin_area):
            logging.info(f"[{self.id}] Traveling to {self.copper_tin_area}...")
            go_to(self.copper_tin_area)
            return self.loop_interval_ms

        # Look for copper or tin rocks to mine
        logging.info(f"[{self.id}] Looking for copper/tin rocks to mine")
        rock = objects.click_object_in_area("Rocks", self.copper_tin_area, "Mine")
        if not rock:
            # Try "Copper ore" or "Tin ore" as object names
            rock = objects.click_object_in_area("Copper ore", self.copper_tin_area, "Mine")
            if not rock:
                rock = objects.click_object_in_area("Tin ore", self.copper_tin_area, "Mine")
        
        if rock:
            wait_until(lambda: player.get_player_animation() == "MINING", max_wait_ms=5000)
        
        return self.loop_interval_ms
    
    def _handle_mining_iron(self) -> int:
        """Handle mining iron ore."""
        # Toggle run if needed
        if player.get_run_energy() > 2000 and not player.is_run_on():
            player.toggle_run()
        
        # Drop ores if inventory is full
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, dropping ores...")
            self._drop_all_ores()
            return self.loop_interval_ms
        
        # Check if we're already mining
        if player.get_player_animation() == "MINING":
            return self.loop_interval_ms

        # Check if we're in the iron area
        if not in_area(self.iron_area):
            logging.info(f"[{self.id}] Traveling to {self.iron_area}...")
            go_to(self.iron_area)
            return self.loop_interval_ms

        # Look for iron rocks to mine
        logging.info(f"[{self.id}] Looking for iron rocks to mine")
        rock = objects.click_object_in_area("Rocks", self.iron_area, "Mine")
        if not rock:
            # Try "Iron ore" as object name
            rock = objects.click_object_in_area("Iron ore", self.iron_area, "Mine")
        
        if rock:
            wait_until(lambda: player.get_player_animation() == "MINING", max_wait_ms=5000)
        
        return self.loop_interval_ms
    
    def _drop_all_ores(self):
        """Drop all ores from inventory."""
        ore_names = ["Copper ore", "Tin ore", "Iron ore"]
        for ore_name in ore_names:
            if inventory.has_item(ore_name):
                count = inventory.inv_count(ore_name)
                logging.info(f"[{self.id}] Dropping {count} {ore_name}")
                # Drop all of this ore type
                while inventory.has_item(ore_name):
                    # Try "Drop-All" first, fallback to "Drop" if that doesn't work
                    result = inventory_interact(ore_name, "Drop-All")
                    if not result or not result.get("ok"):
                        # Try single drop
                        result = inventory_interact(ore_name, "Drop")
                    if result and result.get("ok"):
                        sleep_exponential(0.1, 0.3, 1.2)
                    else:
                        break  # Failed to drop, break out of loop

