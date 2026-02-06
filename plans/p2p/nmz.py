#!/usr/bin/env python3
"""
Nightmare Zone (NMZ) Plan
==========================

Step 13: Prayer flicking
- Flick 'Rapid Heal' prayer every 1-30 seconds to maintain 1hp
"""

import logging
import random
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
    description = """Nightmare Zone training with prayer flicking. Flicks 'Rapid Heal' prayer every 1-30 seconds to maintain 1hp for maximum XP rates. Optimized for AFK combat training.

Starting Area: Nightmare Zone
Required Items: Combat equipment, Prayer points"""
    DONE = 0

    def __init__(self):
        self.state = {
            "phase": "TRAIN",
            "next_prayer_flick_ts": None,
            "last_overload_drink_ts": None,  # Track when we last drank overload potion
            "absorption_low_threshold": None,  # Random threshold between 100-500
            "absorption_target": None,  # Random target between 501-950 when refilling
            "skip_rapid_heal_rock_cake_until": None,  # Shared skip timer for Rapid Heal, rock cake, and combat boost
            "skip_overload_until": None,  # Timestamp until which overload drinking should be skipped
            "rapid_heal_call_count": 0,  # Number of times Rapid Heal has been called
            "rock_cake_call_count": 0,  # Number of times rock cake has been called
            "overload_call_count": 0,  # Number of times overload has been called
            "combat_boost_call_count": 0,  # Number of times combat boost has been called
            "absorption_no_potions_logged": False  # Track if we've already logged that there are no potions
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
        
        # Lazy behavior configuration
        self.lazy_enabled = True  # Enable lazy behavior (sometimes forgets to perform actions)
        self.lazy_probability = 0.15  # 15% chance to skip each action when lazy mode is enabled (default)
        self.lazy_probability_rapid_heal = 0.005  # 2% chance to skip Rapid Heal prayer
        self.lazy_probability_rock_cake = 0.15  # 15% chance to skip rock cake guzzling
        self.lazy_probability_overload = 0.05  # 15% chance to skip overload drinking
        self.lazy_probability_combat_boost = 0.005  # 0.5% chance to skip combat boost prayer
        # Skip timer ranges (in seconds)
        self.skip_rapid_heal_min = 2.0  # Minimum seconds to skip Rapid Heal
        self.skip_rapid_heal_max = 240.0  # Maximum seconds to skip Rapid Heal (4 minutes)
        self.skip_rock_cake_min = 2.0  # Minimum seconds to skip rock cake guzzling
        self.skip_rock_cake_max = 240.0  # Maximum seconds to skip rock cake guzzling (4 minutes)
        self.skip_overload_min = 2.0  # Minimum seconds to skip overload drinking
        self.skip_overload_max = 30.0  # Maximum seconds to skip overload drinking
        # Minimum call counts before lazy behavior can skip actions
        self.rapid_heal_min_calls = 15  # Don't skip Rapid Heal for first 15 calls
        self.rock_cake_min_calls = 15  # Don't skip rock cake for first 15 calls
        self.overload_min_calls = 5  # Don't skip overload for first 5 calls
        self.combat_boost_min_calls = 15  # Don't skip combat boost for first 15 calls

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

    def _reset_all_skip_timers(self):
        """Reset all skip timers when any timer expires."""
        self.state["skip_rapid_heal_rock_cake_until"] = None
        self.state["skip_overload_until"] = None
        
        # Log player stats when timers reset
        try:
            hp = player.get_health()
            prayer_points = self._get_prayer_points()
            hp_str = str(hp) if isinstance(hp, int) else "unknown"
            prayer_str = str(prayer_points) if isinstance(prayer_points, int) else "unknown"
            logging.info(f"[{self.id}] Lazy: All skip timers reset (HP: {hp_str}, Prayer: {prayer_str})")
        except Exception:
            logging.info(f"[{self.id}] Lazy: All skip timers reset")

    def _should_skip_lazy(self, action_name: str = None, should_perform: bool = False) -> bool:
        """
        Check if we should skip this action due to lazy behavior.
        Only sets skip timer if should_perform is True (action was supposed to be performed).
        When any skip timer expires, resets all skip timers.
        
        Args:
            action_name: Optional action name ("rapid_heal", "rock_cake", "overload", "absorption")
            should_perform: True if the action was supposed to be performed (conditions met)
        
        Returns:
            True if we should skip the action, False otherwise
        """
        if not self.lazy_enabled:
            return False
        
        now = time.time()
        
        # Check if any skip timer has expired - if so, reset all timers
        skip_rapid_heal_rock_cake_until = self.state.get("skip_rapid_heal_rock_cake_until")
        skip_overload_until = self.state.get("skip_overload_until")
        
        any_timer_expired = False
        if isinstance(skip_rapid_heal_rock_cake_until, (int, float)) and now >= float(skip_rapid_heal_rock_cake_until):
            any_timer_expired = True
        if isinstance(skip_overload_until, (int, float)) and now >= float(skip_overload_until):
            any_timer_expired = True
        
        if any_timer_expired:
            self._reset_all_skip_timers()
        
        # Check skip timers for specific actions
        if action_name == "rapid_heal":
            # Check if we've reached minimum call count
            call_count = self.state.get("rapid_heal_call_count", 0)
            if call_count < self.rapid_heal_min_calls:
                return False  # Don't skip until minimum calls reached
            
            # Check shared skip timer for rapid_heal and rock_cake
            skip_until = self.state.get("skip_rapid_heal_rock_cake_until")
            if isinstance(skip_until, (int, float)) and now < float(skip_until):
                return True  # Still in skip period
            
            # Timer expired or not set - only set new timer if action was supposed to be performed
            if should_perform and random.random() < self.lazy_probability_rapid_heal:
                # Set shared skip timer (used by rapid_heal, rock_cake, and combat_boost)
                skip_duration = random.uniform(self.skip_rapid_heal_min, self.skip_rapid_heal_max)
                self.state["skip_rapid_heal_rock_cake_until"] = now + skip_duration
                logging.info(f"[{self.id}] Lazy: Skipping Rapid Heal, rock cake, and combat boost for {skip_duration:.1f} seconds")
                return True
            return False
        
        elif action_name == "rock_cake":
            # Check if we've reached minimum call count
            call_count = self.state.get("rock_cake_call_count", 0)
            if call_count < self.rock_cake_min_calls:
                return False  # Don't skip until minimum calls reached
            
            # Check shared skip timer for rapid_heal and rock_cake
            skip_until = self.state.get("skip_rapid_heal_rock_cake_until")
            if isinstance(skip_until, (int, float)) and now < float(skip_until):
                return True  # Still in skip period
            
            # Timer expired or not set - only set new timer if action was supposed to be performed
            if should_perform and random.random() < self.lazy_probability_rock_cake:
                # Set shared skip timer (used by rapid_heal, rock_cake, and combat_boost)
                skip_duration = random.uniform(self.skip_rock_cake_min, self.skip_rock_cake_max)
                self.state["skip_rapid_heal_rock_cake_until"] = now + skip_duration
                logging.info(f"[{self.id}] Lazy: Skipping rock cake, Rapid Heal, and combat boost for {skip_duration:.1f} seconds")
                return True
            return False
        
        elif action_name == "combat_boost":
            # Check if we've reached minimum call count
            call_count = self.state.get("combat_boost_call_count", 0)
            if call_count < self.combat_boost_min_calls:
                return False  # Don't skip until minimum calls reached
            
            # Check shared skip timer for rapid_heal, rock_cake, and combat_boost
            skip_until = self.state.get("skip_rapid_heal_rock_cake_until")
            if isinstance(skip_until, (int, float)) and now < float(skip_until):
                return True  # Still in skip period
            
            # Timer expired or not set - only set new timer if action was supposed to be performed
            if should_perform and random.random() < self.lazy_probability_combat_boost:
                # Set shared skip timer (used by rapid_heal, rock_cake, and combat_boost)
                skip_duration = random.uniform(self.skip_rapid_heal_min, self.skip_rapid_heal_max)
                self.state["skip_rapid_heal_rock_cake_until"] = now + skip_duration
                logging.info(f"[{self.id}] Lazy: Skipping combat boost, Rapid Heal, and rock cake for {skip_duration:.1f} seconds")
                return True
            return False
        
        elif action_name == "overload":
            # Check if we've reached minimum call count
            call_count = self.state.get("overload_call_count", 0)
            if call_count < self.overload_min_calls:
                return False  # Don't skip until minimum calls reached
            
            skip_until = self.state.get("skip_overload_until")
            if isinstance(skip_until, (int, float)) and now < float(skip_until):
                return True  # Still in skip period
            
            # Timer expired or not set - only set new timer if action was supposed to be performed
            if should_perform and random.random() < self.lazy_probability_overload:
                # Set new skip timer
                skip_duration = random.uniform(self.skip_overload_min, self.skip_overload_max)
                self.state["skip_overload_until"] = now + skip_duration
                logging.info(f"[{self.id}] Lazy: Skipping overload drinking for {skip_duration:.1f} seconds")
                return True
            return False
        
        # For absorption or other actions, use simple probability check (no timer)
        return random.random() < self.lazy_probability

    def _handle_train(self) -> int:
        """
        Step 13: Prayer flick 'Rapid Heal' every 1-30 seconds
        """
        # Check if Dominic Onion exists (we're back at the start/outside NMZ)
        self._check_done_conditions()

        # Check HP and drink Overload potion if HP > 50
        # Check if overload should be performed first
        should_drink_overload = self._should_perform_overload()
        if not self._should_skip_lazy("overload", should_perform=should_drink_overload):
            self._maybe_drink_overload()
        self._check_done_conditions()
        
        # Check and refill absorption potions if needed (first priority)
        # Absorption should never be skipped
        self._maybe_drink_absorption()
        self._check_done_conditions()
        
        # Check if prayer flick should be performed
        should_flick_prayer = self._should_perform_prayer_flick()
        if not self._should_skip_lazy("rapid_heal", should_perform=should_flick_prayer):
            self._maybe_prayer_flick()
        self._check_done_conditions()
        
        # Guzzle rock cake if HP > 1 (but not within 10 seconds of drinking overload)
        # Check if rock cake should be performed
        should_guzzle_rock_cake = self._should_perform_rock_cake()
        if not self._should_skip_lazy("rock_cake", should_perform=should_guzzle_rock_cake):
            self._maybe_guzzle_rock_cake()
        self._check_done_conditions()
        
        # Flick combat boost prayer (Ultimate Strength) every 1-30 seconds
        # Check if combat boost should be performed
        # should_flick_combat_boost = self._should_perform_combat_boost()
        # if not self._should_skip_lazy("combat_boost", should_perform=should_flick_combat_boost):
        #     self._maybe_flick_combat_boost()
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
            
            # Check if we have absorption potions BEFORE setting thresholds
            has_absorption_potions = False
            for dose in range(1, 5):
                potion_name = f"Absorption ({dose})"
                if inventory.has_item(potion_name):
                    has_absorption_potions = True
                    break
            
            # If no potions, only log once and reset thresholds
            if not has_absorption_potions:
                if not self.state.get("absorption_no_potions_logged", False):
                    logging.warning(f"[{self.id}] No Absorption potions found in inventory - skipping absorption checks")
                    self.state["absorption_no_potions_logged"] = True
                # Reset thresholds to prevent spam logging
                self.state["absorption_low_threshold"] = None
                self.state["absorption_target"] = None
                return
            
            # We have potions, reset the "no potions logged" flag
            if self.state.get("absorption_no_potions_logged", False):
                self.state["absorption_no_potions_logged"] = False
            
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
                    # No potions found - log once and reset thresholds
                    if not self.state.get("absorption_no_potions_logged", False):
                        logging.warning(f"[{self.id}] No Absorption potions found in inventory - skipping absorption checks")
                        self.state["absorption_no_potions_logged"] = True
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

    def _should_perform_overload(self) -> bool:
        """
        Check if overload should be performed (conditions met).
        
        Returns:
            True if overload should be performed, False otherwise
        """
        try:
            hp = player.get_health()
            if not isinstance(hp, int) or hp <= self.overload_hp_threshold:
                return False  # HP is not above threshold
            
            # Check if we have overload potions
            for dose in range(1, 5):
                potion_name = f"Overload ({dose})"
                if inventory.has_item(potion_name):
                    return True  # Should perform - we have potions and HP is high enough
            
            return False  # No potions available
        except Exception:
            return False

    def _should_perform_prayer_flick(self) -> bool:
        """
        Check if prayer flick should be performed (conditions met).
        
        Returns:
            True if prayer flick should be performed, False otherwise
        """
        now = time.time()
        next_flick = self.state.get("next_prayer_flick_ts")
        
        # Check if it's time to flick
        if not isinstance(next_flick, (int, float)) or now < float(next_flick):
            return False  # Not time to flick yet
        
        # Check if we have prayer points
        prayer_points = self._get_prayer_points()
        if prayer_points is None or prayer_points <= 0:
            return False  # No prayer points
        
        return True  # Should perform - time to flick and have prayer points

    def _should_perform_combat_boost(self) -> bool:
        """
        Check if combat boost prayer flick should be performed (conditions met).
        Combat boost flicks every time it's called (no timer), so this just checks prayer points.
        
        Returns:
            True if combat boost should be performed, False otherwise
        """
        # Check if we have prayer points
        prayer_points = self._get_prayer_points()
        if prayer_points is None or prayer_points <= 0:
            return False  # No prayer points
        
        return True  # Should perform - have prayer points

    def _should_perform_rock_cake(self) -> bool:
        """
        Check if rock cake guzzling should be performed (conditions met).
        
        Returns:
            True if rock cake should be performed, False otherwise
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
                        return False
            
            # Check if we drank overload recently
            last_overload_ts = self.state.get("last_overload_drink_ts")
            if isinstance(last_overload_ts, (int, float)):
                time_since_overload = time.time() - float(last_overload_ts)
                if time_since_overload < self.overload_cooldown_seconds:
                    # Too soon after drinking overload
                    return False
            
            # Check current HP
            hp = player.get_health()
            if not isinstance(hp, int) or hp <= 1:
                return False  # HP is already at 1 or below
            
            # Check if we have rock cake
            if not inventory.has_item(self.rock_cake_name):
                return False  # No rock cake available
            
            return True  # Should perform - all conditions met
        except Exception:
            return False

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
                # Increment call count (only when actually called, not skipped)
                self.state["overload_call_count"] = self.state.get("overload_call_count", 0) + 1
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
            
            # Increment call count (only when actually called, not skipped)
            self.state["rock_cake_call_count"] = self.state.get("rock_cake_call_count", 0) + 1
            
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
                    sleep_exponential(0.2, 0.6, 1.2)
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
            # Increment call count (only when actually called, not skipped)
            self.state["rapid_heal_call_count"] = self.state.get("rapid_heal_call_count", 0) + 1
            
            # Flick Rapid Heal prayer with verification
            prayer.flick_prayer_with_verification(
                self.rapid_heal_prayer_name,
                delay_between_clicks=(0.3, 5, 10),
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
        Flick 'Ultimate Strength' prayer every time this method is called.
        Holds the prayer on for 2-2.5 seconds between clicks.
        Only flicks if prayer points > 0.
        """
        # Check if we have prayer points
        prayer_points = self._get_prayer_points()
        if prayer_points is None or prayer_points <= 0:
            logging.warning(f"[{self.id}] No prayer points available ({prayer_points}), skipping combat boost prayer flick")
            return

        # Flick every time - use prayer action methods
        try:
            # Increment call count (only when actually called, not skipped)
            self.state["combat_boost_call_count"] = self.state.get("combat_boost_call_count", 0) + 1
            
            # Flick Ultimate Strength prayer with verification
            # Hold prayer on for 2-2.5 seconds between clicks
            prayer.flick_prayer_with_verification(
                "Ultimate Strength",
                delay_between_clicks=(1.0, 1.5, 1.2),
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
