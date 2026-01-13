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
from actions.player import hop_world, random_world
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.runtime_utils import ipc
from helpers.utils import exponential_number, sleep_exponential
from helpers.vars import get_var
from helpers.widgets import widget_exists
from constants import JEWELRY_CRAFTING_WIDGETS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan


class CraftingPlan(Plan):
    """Simple crafting plan for P2P."""
    
    id = "CRAFTING"
    label = "Crafting: Edgeville Bank"
    
    def __init__(self):
        self.state = {"phase": "BANK", "next_tab_switch_ts": None, "last_smelting_ts": None, "last_world_hop_ts": None}
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

        # RuneLite Var Inspector: VarbitID.STAMINA_ACTIVE = 25
        self.STAMINA_ACTIVE_VARBIT = 25
        # RuneLite Var Inspector: VarbitID.STAMINA_DURATION = 24
        self.STAMINA_DURATION_VARBIT = 24
        
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
    
    def _maybe_world_hop(self) -> bool:
        """
        Occasionally hop to a random members world (human-like behavior).
        
        Frequency: every 60-80 minutes (randomized using exponential distribution).
        
        Returns:
            bool: True if a world hop was attempted, False otherwise
        """
        now = time.time()
        last_hop = self.state.get("last_world_hop_ts")
        
        # Initialize timestamp if not set
        if not isinstance(last_hop, (int, float)):
            # Schedule first hop between 60-80 minutes from now
            hop_interval_seconds = exponential_number(1800, 3000, 1, output_type="float")  # 60-80 minutes
            self.state["last_world_hop_ts"] = now + hop_interval_seconds
            return False
        
        # Check if it's time to hop (60-80 minutes have passed)
        if now < float(last_hop):
            return False
        
        # Only hop during BANK phase (safest time, not during active crafting)
        phase = self.state.get("phase", "BANK")
        if phase != "BANK":
            # If not in BANK phase, reschedule for later
            hop_interval_seconds = exponential_number(3600.0, 4800.0, 0.5, output_type="float")
            self.state["last_world_hop_ts"] = now + hop_interval_seconds
            return False
        
        # Get a random members world (excluding current world)
        target_world = random_world(kind="p2p", exclude_current=True)
        if not target_world:
            logging.warning(f"[{self.id}] Failed to get random members world for hopping")
            # Reschedule for later
            hop_interval_seconds = exponential_number(3600.0, 4800.0, 0.5, output_type="float")
            self.state["last_world_hop_ts"] = now + hop_interval_seconds
            return False
        
        # Attempt world hop
        logging.info(f"[{self.id}] Hopping to world {target_world}")
        success = hop_world(target_world)
        
        if success:
            logging.info(f"[{self.id}] Successfully hopped to world {target_world}")
            # Schedule next hop (60-80 minutes from now)
            hop_interval_seconds = exponential_number(3600.0, 4800.0, 0.5, output_type="float")
            self.state["last_world_hop_ts"] = now + hop_interval_seconds
        else:
            logging.warning(f"[{self.id}] Failed to hop to world {target_world}")
            # Reschedule for later (try again in 5-10 minutes)
            self.state["last_world_hop_ts"] = now + exponential_number(300.0, 600.0, 0.5, output_type="float")
        
        return True
    
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
        
        # Check if it's time to world hop (only when logged in and not in DONE phase)
        if logged_in and phase != "DONE":
            if self._maybe_world_hop():
                # World hop was attempted, return early to allow login process to complete
                return exponential_number(2000, 5000, 1.2)

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

        stamina_delay = self._maybe_drink_stamina_at_bank()
        if stamina_delay is not None:
            return stamina_delay
        
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
        result = bank.close_bank(
            # object_name="Furnace",
            # action="Smelt",
            # click_probability_range=(0.2347, 0.3891),
            # prefer_no_camera=True
        )
        
        if result:
            # If we clicked the furnace, wait for bank to close and crafting interface to open
            # if wait_until(lambda: bank.is_closed() and (widget_exists(29229056) or widget_exists(393216)), max_wait_ms=7000):
            if wait_until(lambda: bank.is_closed(), max_wait_ms=7000):
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
        if player.is_activity_active("SMELTING", "last_smelting_ts", 2, self.state):
            # Currently crafting - return early
            return 0
        
        # Click furnace
        objects.click_object_closest_by_distance_simple_no_camera("Furnace", "Smelt")
        if not wait_until(lambda: widget_exists(29229056) or widget_exists(393216), max_wait_ms=5000):  # Crafting interface
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

    def _maybe_drink_stamina_at_bank(self) -> int | None:
        """
        If stamina is not active, drink 1 dose while the bank is open:
        - Withdraw a stamina potion
        - Drink it (from player inventory while bank is open)
        - Deposit it back
        """
        # Only drink when run energy is low
        run_energy = player.get_run_energy()
        # RuneLite runEnergy is typically 0..10000 (100.00%); treat < 30% as < 3000
        if (run_energy is None) or (int(run_energy) >= 3000):
            return None

        try:
            stamina_active = int(get_var(self.STAMINA_ACTIVE_VARBIT, timeout=0.35) or 0)
        except Exception:
            stamina_active = 0

        # Stamina duration counts down from 20 -> 0 (about 2 minutes total).
        # That implies ~6 seconds per unit. Drink when <= 10 seconds remain.
        try:
            stamina_dur_units = int(get_var(self.STAMINA_DURATION_VARBIT, timeout=0.35) or 0)
        except Exception:
            stamina_dur_units = 0
        dur_unit_seconds = 6
        drink_when_le_seconds = 10
        drink_when_le_units = max(0, (drink_when_le_seconds + dur_unit_seconds - 1) // dur_unit_seconds)  # ceil

        should_drink = (stamina_active != 1) or (stamina_dur_units <= int(drink_when_le_units))
        if not should_drink:
            return None

        if not bank.is_open():
            return None

        # Prefer lowest doses first
        pot_names = [
            "Stamina potion(1)",
            "Stamina potion(2)",
            "Stamina potion(3)",
            "Stamina potion(4)",
        ]

        # Ensure we have a potion in inventory
        inv_pot = None
        for nm in pot_names:
            if inventory.has_item(nm):
                inv_pot = nm
                break

        if inv_pot is None:
            # Withdraw 1 potion from bank
            for nm in pot_names:
                try:
                    if int(bank.get_item_count(nm) or 0) > 0:
                        bank.withdraw_item(nm, withdraw_x=1)
                        wait_until(lambda: inventory.has_any_items(pot_names), max_wait_ms=1200)
                        break
                except Exception:
                    continue

            for nm in pot_names:
                if inventory.has_item(nm):
                    inv_pot = nm
                    break

            if inv_pot is None:
                logging.warning(f"[{self.id}] Stamina not active but no stamina potions found in bank.")
                return None

            return exponential_number(200, 650, 1.2)

        # Drink a dose (menu option should exist even while bank is open)
        if not bank.interact_inventory(inv_pot, "Drink"):
            return exponential_number(200, 650, 1.2)

        # Wait briefly for varbit to flip
        wait_until(lambda: int(get_var(self.STAMINA_ACTIVE_VARBIT, timeout=0.2) or 0) == 1, max_wait_ms=1500)
        sleep_exponential(0.12, 0.35, 1.2)

        # Deposit whatever stamina dose remains
        for nm in pot_names:
            if inventory.has_item(nm):
                bank.deposit_item(nm, deposit_all=True)
                break

        return exponential_number(200, 650, 1.2)