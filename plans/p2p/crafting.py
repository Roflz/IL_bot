#!/usr/bin/env python3
"""
Crafting Plan (P2P)
===================

Simple crafting plan that cycles between Edgeville bank and crafting.
Configure jewelry type via self.jewelry_config in __init__.
"""

import logging
import time
from pathlib import Path
import sys

from actions import player, bank, inventory, objects, chat, widgets, tab
from actions import wait_until
from actions.travel import travel_to_bank, go_to
from actions.tab import is_tab_open
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.runtime_utils import ipc
from helpers.utils import exponential_number, sleep_exponential
from helpers.widgets import widget_exists
from constants import JEWELRY_CRAFTING_WIDGETS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan


class CraftingPlan(Plan):
    """Simple crafting plan for P2P."""
    
    id = "CRAFTING"
    label = "Crafting: Edgeville Bank"
    
    def __init__(self):
        self.state = {"phase": "BANK", "next_tab_switch_ts": None, "last_smelting_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Jewelry configuration - change this to craft different jewelry
        # Valid options:
        #   - Gold bracelets: {"mould": "bracelet", "gem": None}
        #   - Sapphire/Emerald/Ruby bracelets: {"mould": "bracelet", "gem": "sapphire"/"emerald"/"ruby"}
        #   - Opal/Jade/Topaz jewelry: {"mould": "ring"/"necklace"/"bracelet", "gem": "opal"/"jade"/"topaz"}
        self.jewelry_config = {
            "mould": "necklace",  # Options: "ring", "necklace", "bracelet"
            "gem": "opal"            # Options: None (gold), "sapphire", "emerald", "ruby", "opal", "jade", "topaz"
        }
        
        # Calculate required items based on jewelry config
        self._update_required_items()
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Jewelry config: {self.jewelry_config}")
        logging.info(f"[{self.id}] Required items: {self.required_items}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def _maybe_tab_switch(self) -> None:
        """
        Occasionally switch to either SKILLS or INVENTORY (human-like tab switching).

        Frequency is re-sampled each time using an exponential distribution:
        ~1 minute up to ~30 minutes between peeks.
        """
        # Avoid tab peeks while bank is open (can be visually noisy / interfere with some flows)
        if bank.is_open():
            return

        now = time.time()
        nxt = self.state.get("next_tab_switch_ts")
        if not isinstance(nxt, (int, float)):
            # Seed initial schedule
            self.state["next_tab_switch_ts"] = now + exponential_number(60.0, 1800.0, 0.5, output_type="float")
            return

        if now < float(nxt):
            return

        # Toggle behavior:
        # - If Inventory is open -> open Skills
        # - If Skills is open -> open Inventory
        # - Else -> open Inventory
        if is_tab_open("INVENTORY"):
            tab.open_tab("SKILLS")
        elif is_tab_open("SKILLS"):
            tab.open_tab("INVENTORY")
        else:
            tab.open_tab("INVENTORY")

        # Schedule next switch
        self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))
    
    def _update_required_items(self):
        """Update required_items based on jewelry_config."""
        mould_name = f"{self.jewelry_config['mould']} mould"
        
        # Determine bar type: jade, topaz, and opal use silver bars; everything else uses gold bars
        gem = self.jewelry_config.get("gem")
        if gem and gem.lower() in ["jade", "topaz", "opal"]:
            bar_type = "Silver bar"
        else:
            bar_type = "Gold bar"
        
        # When crafting with gems, use 13 bars and 13 gems; otherwise withdraw all bars
        if gem:
            bar_quantity = 13
            gem_quantity = 13
        else:
            bar_quantity = -1  # -1 means withdraw all
        
        self.required_items = [
            {"name": mould_name, "quantity": 1},
            {"name": bar_type, "quantity": bar_quantity}
        ]
        
        if gem:
            self.required_items.append({"name": gem.capitalize(), "quantity": gem_quantity})
    
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
            case "CRAFTING":
                return self._handle_crafting()
            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return exponential_number(400, 1100, 1.2)
    
    def _handle_bank(self) -> int:
        """Simple banking - deposit all, withdraw required items."""
        # Travel to bank
        if not travel_to_bank():
            return exponential_number(300, 800, 1.2)
        
        # Open bank (with randomization between 2 closest booths, prefer no camera)
        if not bank.is_open():
            bank.open_bank(randomize_closest=2, prefer_no_camera=True)
            wait_until(bank.is_open, max_wait_ms=5000)
            return exponential_number(300, 800, 1.2)
        
        # Check if we have required materials in bank OR inventory - if not, go to DONE phase
        required_item_names = [item["name"] for item in self.required_items]
        materials_available = bank.has_materials_available(required_item_names)
        
        for item_name, is_available in materials_available.items():
            if not is_available:
                # Out of materials - go to DONE phase
                logging.info(f"[{self.id}] Out of materials (bank and inventory): {item_name}, plan complete")
                self.set_phase("DONE")
                return exponential_number(1000, 3000, 1.2)
        
        # Smart deposit logic - only deposit unwanted items
        required_item_names = [item["name"] for item in self.required_items]
        deposit_result = bank.deposit_unwanted_items(required_item_names, max_unique_for_bulk=3)
        
        if deposit_result is not None:
            # Wait for deposits to complete
            return exponential_number(200, 600, 1.2)
        
        # Withdraw required items (with randomized order)
        from helpers.bank import randomize_item_order
        items_to_withdraw = randomize_item_order(self.required_items, reverse_probability_range=(0.2834, 0.4167))
        
        for item in items_to_withdraw:
            item_name = item["name"]
            quantity = item["quantity"]
            
            # Get current inventory count
            from helpers.inventory import inv_count
            current_count = inv_count(item_name)
            
            # If we have the exact required quantity, skip
            if quantity == -1:
                # For -1 (withdraw all), just check if we have any
                if current_count > 0:
                    continue
            else:
                # For specific quantities, check if we have exactly the right amount
                if current_count == quantity:
                    continue
            
            # If quantity doesn't match, deposit all of this item first (if we have any)
            if current_count > 0:
                bank.deposit_item(item_name, deposit_all=True)
                wait_until(lambda name=item_name: not inventory.has_item(name), max_wait_ms=2000)
                sleep_exponential(0, 0.6, 1.2)
            
            # Check if bank has it
            bank_count = bank.get_item_count(item_name)
            if bank_count is None or int(bank_count) <= 0:
                logging.warning(f"[{self.id}] Missing required item: {item_name}")
                return exponential_number(400, 1100, 1.2)
            
            # Withdraw the required amount
            if quantity == -1:
                bank.withdraw_item(item_name, withdraw_all=True)
            else:
                bank.withdraw_item(item_name, withdraw_x=quantity)
            
            wait_until(lambda name=item_name: inventory.has_item(name), max_wait_ms=2000)
            sleep_exponential(0, 0.6, 1.2)
        
        # Close bank and move to crafting (sometimes click furnace directly instead)
        result = bank.close_bank_or_click_object(
            object_name="Furnace",
            action="Smelt",
            click_probability_range=(0.2347, 0.3891),
            prefer_no_camera=True
        )
        
        if result:
            # If we clicked the furnace, wait for bank to close and crafting interface to open
            if wait_until(lambda: bank.is_closed() and (widget_exists(29229056) or widget_exists(393216)), max_wait_ms=7000):
                self.set_phase("CRAFTING")
                return exponential_number(0, 800, 1.2)
        
        # Fallback: ensure bank is closed and transition to crafting
        if bank.is_open():
            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=3000)
        self.set_phase("CRAFTING")
        return exponential_number(0, 800, 1.2)
    
    def _handle_crafting(self) -> int:
        """Handle crafting phase."""
        # Occasionally switch tabs (human-like behavior)
        self._maybe_tab_switch()
        
        # Check if we have required materials - if not, go to BANK phase
        # If out of materials, disregard timer and immediately go to BANK
        for item in self.required_items:
            item_name = item["name"]
            if not inventory.has_unnoted_item(item_name):
                # Out of materials - clear crafting timer and go to BANK
                self.state["last_smelting_ts"] = None
                self.set_phase("BANK")
                return exponential_number(0, 10000, 3)

        if chat.can_continue():
            self.state["last_smelting_ts"] = None

        # Get mould and gem types from config
        mould_type = self.jewelry_config["mould"]
        gem_type = self.jewelry_config.get("gem")

        if widget_exists(29229056) or widget_exists(393216):
            if not self.craft_jewelry(mould_type, gem_type):
                return exponential_number(300, 800, 1.2)
            if wait_until(lambda: player.get_player_animation() == "SMELTING", max_wait_ms=1000):
                # Update timestamp when we see SMELTING animation
                self.state["last_smelting_ts"] = time.time()
                return 0
            return exponential_number(300, 800, 1.2)

        # Check if already crafting - hybrid timer + state approach
        if player.is_activity_active("SMELTING", "last_smelting_ts", 1.0, self.state):
            # Currently crafting - return early
            return 0
        
        # Click furnace
        objects.click_object_closest_by_distance_simple_prefer_no_camera("Furnace", "Smelt")
        if not wait_until(lambda: widget_exists(29229056) or widget_exists(393216)):  # Crafting interface
            return exponential_number(300, 800, 1.2)
        
        sleep_exponential(0.3, 0.8, 1.2)
        
        # Craft jewelry using widget
        if not self.craft_jewelry(mould_type, gem_type):
            return exponential_number(300, 800, 1.2)
        if wait_until(lambda: player.get_player_animation() == "SMELTING", max_wait_ms=1000):
            # Update timestamp when we see SMELTING animation
            self.state["last_smelting_ts"] = time.time()
            return 0
        return exponential_number(300, 800, 1.2)
    
    def craft_jewelry(self, mould_type: str, gem_type: str = None) -> bool:
        """
        Craft jewelry using the appropriate widget based on mould and gem type.
        
        Args:
            mould_type: "ring", "necklace", or "bracelet"
            gem_type: "sapphire", "emerald", "ruby", "opal", "jade", "topaz", or None for gold jewelry
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine the widget key based on mould and gem type
            if gem_type:
                widget_key = f"{gem_type.upper()}_{mould_type.upper()}"
            else:
                # Gold jewelry (no gem)
                widget_key = f"GOLD_{mould_type.upper()}"
            
            # Get the widget ID from constants
            widget_id = JEWELRY_CRAFTING_WIDGETS.get(widget_key)
            
            if not widget_id:
                logging.error(f"[{self.id}] Unknown jewelry combination: {mould_type} + {gem_type or 'gold'}")
                return False
            
            # Click widget or press spacebar based on highlight state
            success = widgets.click_widget_or_spacebar(
                widget_id=widget_id,
                highlighted_count=4,
                spacebar_probability_range=(0.80, 0.90)
            )
            
            if success:
                logging.info(f"[{self.id}] Successfully interacted with {widget_key} (ID: {widget_id})")
            else:
                logging.error(f"[{self.id}] Failed to interact with {widget_key} (ID: {widget_id})")
            
            return success
            
        except Exception as e:
            logging.error(f"[{self.id}] Error in craft_jewelry: {e}")
            return False
    
    def _handle_done(self) -> int:
        """Handle DONE phase - out of materials."""
        logging.info(f"[{self.id}] Out of materials, plan complete")
        return exponential_number(1000, 3000, 1.2)
