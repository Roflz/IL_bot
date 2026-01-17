#!/usr/bin/env python3
"""
Nightmare Zone (NMZ) Plan
==========================

Step 13: Prayer flicking
- Flick 'Rapid Heal' prayer every 1-30 seconds to maintain 1hp
"""

import logging
import time
from pathlib import Path
import sys
from typing import Optional

from actions import player
from actions import inventory
from actions import tab
from actions import widgets
from actions import wait_until
from actions import prayer
from helpers.npc import get_npcs_by_name
from helpers.runtime_utils import ipc
from helpers.tab import is_tab_open
from helpers.utils import sleep_exponential, random_number, exponential_number
from helpers.vars import get_var
from constants import PRAYER_WIDGETS

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera


class NMZPlan(Plan):
    id = "NMZ"
    label = "Nightmare Zone"
    DONE = 0

    def __init__(self):
        self.state = {
            "phase": "TRAIN",
            "next_prayer_flick_ts": None,
            "last_overload_drink_ts": None,  # Track when we last drank overload potion
            "absorption_low_threshold": None,  # Random threshold between 100-500
            "absorption_target": None  # Random target between 501-950 when refilling
        }
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # Configuration
        self.prayer_flick_interval_min = 1.0  # Minimum seconds between prayer flicks
        self.prayer_flick_interval_max = 30.0  # Maximum seconds between prayer flicks
        self.rapid_heal_prayer_name = "Rapid Heal"
        self.rapid_heal_widget_id = PRAYER_WIDGETS.get(self.rapid_heal_prayer_name)
        self.overload_hp_threshold = 50  # Drink overload if HP is above this
        self.overload_cooldown_seconds = 10  # Don't guzzle rock cake within this many seconds after drinking overload
        self.rock_cake_name = "Dwarven rock cake"
        # Varbit IDs
        self.absorption_varbit_id = 3956  # NZONE_ABSORB_POTION_EFFECTS
        self.overload_varbit_id = 3955  # Overload potion effect varbit

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "TRAIN")

        # Check if player is logged out - if so, we're done
        if not player.logged_in():
            self.set_phase("DONE")
            logging.info(f"[{self.id}] Player logged out, setting phase to DONE")
            return self.loop_interval_ms

        match phase:
            case "TRAIN":
                return self._handle_train()
            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_train(self) -> int:
        """
        Step 13: Prayer flick 'Rapid Heal' every 1-30 seconds
        """
        # Check if Dominic Onion exists (we're back at the start/outside NMZ)
        self._check_done_conditions()
        
        # Check and refill absorption potions if needed (first priority)
        self._maybe_drink_absorption()
        self._check_done_conditions()
        
        # Check HP and drink Overload potion if HP > 50
        self._maybe_drink_overload()
        self._check_done_conditions()
        
        self._maybe_prayer_flick()
        self._check_done_conditions()
        
        # Guzzle rock cake if HP > 1 (but not within 10 seconds of drinking overload)
        self._maybe_guzzle_rock_cake()
        self._check_done_conditions()
        
        # Flick combat boost prayer (Ultimate Strength) every time
        # self._maybe_flick_combat_boost()
        # self._check_done_conditions()
        
        return 0

    def _check_done_conditions(self):
        dominic_npcs = get_npcs_by_name("Dominic Onion")
        if dominic_npcs and len(dominic_npcs) > 0:
            logging.info(f"[{self.id}] Dominic Onion detected - NMZ session complete")
            self.set_phase("DONE")
            return self.loop_interval_ms
        return False

    def _maybe_drink_absorption(self) -> None:
        """
        Drink absorption potions when absorption points drop to or below low threshold.
        Sets a random low threshold (100-500) that persists until refill.
        When at or below threshold, drinks until reaching a random target (501-950).
        """
        try:
            # Get current absorption points from varbit
            if self.absorption_varbit_id is None:
                logging.warning(f"[{self.id}] Absorption varbit ID not set, skipping absorption check")
                return
            
            absorption_value = get_var(self.absorption_varbit_id, timeout=0.35)
            if absorption_value is None:
                logging.warning(f"[{self.id}] Could not read absorption varbit")
                return
            
            current_absorption = int(absorption_value) if isinstance(absorption_value, (int, dict)) else None
            if current_absorption is None:
                # Try to extract from dict if it's a dict response
                if isinstance(absorption_value, dict):
                    current_absorption = absorption_value.get("value") or absorption_value.get("varbit")
                if current_absorption is None:
                    logging.warning(f"[{self.id}] Could not parse absorption value: {absorption_value}")
                    return
                current_absorption = int(current_absorption)
            
            # Initialize thresholds if not set
            low_threshold = self.state.get("absorption_low_threshold")
            if not isinstance(low_threshold, int):
                # Set random low threshold between 100-500
                low_threshold = int(random_number(100, 500, output_type="float"))
                self.state["absorption_low_threshold"] = low_threshold
                logging.info(f"[{self.id}] Set absorption low threshold to {low_threshold}")
            
            # Check if we're at or below the low threshold - if not, we don't need to drink
            if current_absorption > low_threshold:
                # We're above threshold, reset target if we were refilling
                if self.state.get("absorption_target") is not None:
                    self.state["absorption_target"] = None
                return  # No need to drink
            
            # We're at or below threshold - need to refill
            # Set target if not already set
            target = self.state.get("absorption_target")
            if not isinstance(target, int):
                # Set random target between 501-950
                target = int(random_number(501, 950, output_type="float"))
                self.state["absorption_target"] = target
                logging.info(f"[{self.id}] Absorption is {current_absorption} (at/below threshold {low_threshold}), setting target to {target}")
            
            # Check if we've reached the target
            if current_absorption >= target:
                # We've reached our target, reset thresholds for next cycle
                logging.info(f"[{self.id}] Reached absorption target {target}, resetting thresholds")
                self.state["absorption_low_threshold"] = None
                self.state["absorption_target"] = None
                return
            
            # We're at or below threshold and haven't reached target - loop drinking absorption potions
            max_drinks = 50  # Safety limit to prevent infinite loops
            drink_count = 0
            
            while drink_count < max_drinks:
                # Re-check absorption after each drink
                absorption_value = get_var(self.absorption_varbit_id, timeout=0.35)
                if absorption_value is None:
                    logging.warning(f"[{self.id}] Could not read absorption varbit during drinking loop")
                    break
                
                current_absorption = int(absorption_value) if isinstance(absorption_value, (int, dict)) else None
                if current_absorption is None:
                    if isinstance(absorption_value, dict):
                        current_absorption = absorption_value.get("value") or absorption_value.get("varbit")
                    if current_absorption is None:
                        logging.warning(f"[{self.id}] Could not parse absorption value during loop")
                        break
                    current_absorption = int(current_absorption)
                
                # Check if we've reached the target
                if current_absorption >= target:
                    # We've reached our target, reset thresholds for next cycle
                    if drink_count > 0:
                        logging.info(f"[{self.id}] Reached absorption target {target} after {drink_count} drink(s)")
                    self.state["absorption_low_threshold"] = None
                    self.state["absorption_target"] = None
                    break
                
                # Find all Absorption potions in inventory (1-4 doses)
                absorption_potions = []
                for dose in range(1, 5):
                    potion_name = f"Absorption ({dose})"
                    if inventory.has_item(potion_name):
                        absorption_potions.append((dose, potion_name))
                
                if not absorption_potions:
                    logging.warning(f"[{self.id}] Absorption is {current_absorption} but no Absorption potions found in inventory - continuing without refill")
                    # Reset thresholds so we don't keep trying
                    self.state["absorption_low_threshold"] = None
                    self.state["absorption_target"] = None
                    break
                
                # Sort by dose count (ascending) to get the one with least doses first
                absorption_potions.sort(key=lambda x: x[0])
                dose_count, potion_name = absorption_potions[0]
                
                # Drink the potion
                result = inventory.interact(potion_name, "Drink", exact_match=False)
                if result:
                    drink_count += 1
                    # Wait between drinks
                    sleep_exponential(0.5, 1.0, 1.2)
                else:
                    logging.warning(f"[{self.id}] Failed to drink {potion_name}")
                    # Wait a bit before retrying
                    sleep_exponential(0.3, 0.5, 1.2)
            
            if drink_count >= max_drinks:
                logging.warning(f"[{self.id}] Reached max drinks ({max_drinks}) but absorption may still be below target")
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking/drinking absorption potion: {e}")

    def _maybe_drink_overload(self) -> None:
        """
        Drink an Overload potion if HP is above the threshold.
        Prefers potions with fewer doses (1 dose over 4 dose).
        """
        try:
            hp = player.get_health()
            if not isinstance(hp, int) or hp <= self.overload_hp_threshold:
                return  # HP is not above threshold, no need to drink
            
            # Find all Overload potions in inventory (1-4 doses)
            overload_potions = []
            for dose in range(1, 5):
                potion_name = f"Overload ({dose})"
                if inventory.has_item(potion_name):
                    overload_potions.append((dose, potion_name))
            
            if not overload_potions:
                logging.warning(f"[{self.id}] HP is {hp} but no Overload potions found in inventory - continuing without overload")
                return
            
            # Sort by dose count (ascending) to get the one with least doses first
            overload_potions.sort(key=lambda x: x[0])
            dose_count, potion_name = overload_potions[0]
            
            logging.info(f"[{self.id}] HP is {hp}, drinking {potion_name} (least doses available)")
            result = inventory.interact(potion_name, "Drink", exact_match=False)
            if result:
                # Track when we drank the overload potion
                self.state["last_overload_drink_ts"] = time.time()
                sleep_exponential(3, 6, 1)
            else:
                logging.warning(f"[{self.id}] Failed to drink {potion_name}")
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking/drinking overload potion: {e}")

    def _maybe_guzzle_rock_cake(self) -> None:
        """
        Guzzle Dwarven rock cake until HP is at 1.
        Skips if we drank an overload potion within the last 10 seconds.
        Skips if overload potion varbit is below 2 (overload effect running out).
        If no overload potions are in inventory, ignores the varbit check.
        """
        try:
            # Check if we have any overload potions in inventory
            has_overload_potions = False
            for dose in range(1, 5):
                potion_name = f"Overload ({dose})"
                if inventory.has_item(potion_name):
                    has_overload_potions = True
                    break
            
            # Only check overload varbit if we have overload potions in inventory
            if has_overload_potions:
                # Check overload potion varbit - don't guzzle if below 2
                overload_value = get_var(self.overload_varbit_id, timeout=0.35)
                if overload_value is not None:
                    overload_level = int(overload_value) if isinstance(overload_value, (int, dict)) else None
                    if overload_level is None:
                        if isinstance(overload_value, dict):
                            overload_level = overload_value.get("value") or overload_value.get("varbit")
                        if overload_level is not None:
                            overload_level = int(overload_level)
                    
                    if isinstance(overload_level, int) and overload_level < 2:
                        # Overload effect is running out (below 2), don't guzzle
                        return
            
            # Check if we drank overload recently
            last_overload_ts = self.state.get("last_overload_drink_ts")
            if isinstance(last_overload_ts, (int, float)):
                time_since_overload = time.time() - float(last_overload_ts)
                if time_since_overload < self.overload_cooldown_seconds:
                    # Too soon after drinking overload, skip
                    return
            
            # Check current HP
            hp = player.get_health()
            if not isinstance(hp, int) or hp <= 1:
                return  # HP is already at 1 or below, no need to guzzle
            
            # Check if we have rock cake
            if not inventory.has_item(self.rock_cake_name):
                logging.warning(f"[{self.id}] HP is {hp} but no {self.rock_cake_name} found in inventory - continuing without guzzling")
                return
            
            # Loop guzzling rock cake until we're at 1hp
            max_guzzles = 50  # Safety limit to prevent infinite loops
            guzzle_count = 0
            if self._check_done_conditions():
                return
            while guzzle_count < max_guzzles:
                if self._check_done_conditions():
                    break
                hp = player.get_health()
                if not isinstance(hp, int) or hp <= 1:
                    # We're at 1hp or below, we're done
                    if guzzle_count > 0:
                        logging.info(f"[{self.id}] Reached 1hp after {guzzle_count} guzzle(s)")
                    break
                
                # Guzzle the rock cake
                result = inventory.interact(self.rock_cake_name, "Guzzle", exact_match=False)
                if result:
                    guzzle_count += 1
                    # Wait between guzzles (0.6-1 second)
                    sleep_exponential(0.6, 1.0, 1.2)
                else:
                    logging.warning(f"[{self.id}] Failed to guzzle {self.rock_cake_name}")
                    # Wait a bit before retrying
                    sleep_exponential(0.3, 0.5, 1.2)
            
            if guzzle_count >= max_guzzles:
                logging.warning(f"[{self.id}] Reached max guzzles ({max_guzzles}) but HP may still be above 1")
        except Exception as e:
            logging.warning(f"[{self.id}] Error guzzling rock cake: {e}")

    def _get_prayer_points(self) -> Optional[int]:
        """
        Get the player's current prayer points.
        
        Returns:
            Current prayer points (int) if successful, None otherwise
        """
        try:
            resp = ipc.get_player()
            if not resp or not resp.get("ok"):
                return None
            
            player_data = resp.get("player")
            if not player_data:
                return None
            
            # Get prayer skill data
            skills = player_data.get("skills", {})
            prayer_data = skills.get("prayer")
            if prayer_data:
                return prayer_data.get("boostedLevel")  # Current prayer points (boosted level)
            
            return None
        except Exception as e:
            logging.warning(f"[{self.id}] Error getting prayer points: {e}")
            return None

    def _maybe_prayer_flick(self) -> None:
        """
        Flick 'Rapid Heal' prayer every 1-30 seconds to maintain 1hp.
        Only flicks if prayer points > 0.
        """
        now = time.time()
        next_flick = self.state.get("next_prayer_flick_ts")
        
        # Initialize next flick time if not set
        if not isinstance(next_flick, (int, float)):
            interval = random_number(
                self.prayer_flick_interval_min,
                self.prayer_flick_interval_max,
                output_type="float"
            )
            self.state["next_prayer_flick_ts"] = now + float(interval)
            return

        # Check if it's time to flick
        if now < float(next_flick):
            return

        # Check if we have prayer points
        prayer_points = self._get_prayer_points()
        if prayer_points is None or prayer_points <= 0:
            logging.warning(f"[{self.id}] No prayer points available ({prayer_points}), skipping prayer flick")
            # Schedule next flick even if we have no prayer points
            interval = random_number(
                self.prayer_flick_interval_min,
                self.prayer_flick_interval_max,
                output_type="float"
            )
            self.state["next_prayer_flick_ts"] = now + float(interval)
            return

        # Time to flick - use prayer action methods
        try:
            # Flick Rapid Heal prayer with verification
            prayer.flick_prayer_with_verification(
                self.rapid_heal_prayer_name,
                delay_between_clicks=(0.3, 10, 10),
                verify_inactive=True
            )

            # Schedule next flick
            interval = random_number(
                self.prayer_flick_interval_min,
                self.prayer_flick_interval_max,
                output_type="float"
            )
            self.state["next_prayer_flick_ts"] = now + float(interval)
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error during prayer flick: {e}")
            # Schedule next flick even on error
            interval = random_number(
                self.prayer_flick_interval_min,
                self.prayer_flick_interval_max,
                output_type="float"
            )
            self.state["next_prayer_flick_ts"] = now + float(interval)

    def _maybe_flick_combat_boost(self) -> None:
        """
        Flick 'Ultimate Strength' prayer every time (no timer check).
        Waits 2-2.5 seconds between clicks when flicking it back off.
        Only flicks if prayer points > 0.
        """
        # Check if we have prayer points
        prayer_points = self._get_prayer_points()
        if prayer_points is None or prayer_points <= 0:
            logging.warning(f"[{self.id}] No prayer points available ({prayer_points}), skipping combat boost prayer flick")
            return
        
        try:
            # Flick Ultimate Strength prayer with longer delay (2-2.5 seconds)
            prayer.flick_prayer_with_verification(
                "Ultimate Strength",
                delay_between_clicks=(2.0, 2.5, 1.2),
                verify_inactive=True
            )
        except Exception as e:
            logging.warning(f"[{self.id}] Error during combat boost prayer flick: {e}")

    def _handle_done(self) -> int:
        """
        NMZ session is complete (Dominic Onion detected or player logged out).
        Return DONE status every 30 seconds.
        """
        sleep_exponential(2.0, 5.0, 1.2)
        return 30000  # Return every 30 seconds
