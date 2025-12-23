#!/usr/bin/env python3
"""
Cooking Plan
============

Phases:
- BANK: Withdraw raw fish to cook
- GO_TO_RANGE: Travel to a cooking range
- COOK: Cook the fish
- GO_TO_BANK: Travel back to a bank (then repeats)

This is a simple loop: bank -> range -> cook -> bank -> ...
"""

import logging
import time
from pathlib import Path
import sys

from actions import bank, inventory, player
from actions import wait_until
from actions import objects
from actions import tab
from actions.travel import in_area, go_to, travel_to_bank
from helpers.tab import is_tab_open
from helpers.keyboard import press_spacebar
from helpers.utils import sleep_exponential, exponential_number
from helpers.widgets import widget_exists

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera


class CookingPlan(Plan):
    id = "COOKING"
    label = "Cooking Plan"
    DONE = 0

    def __init__(self):
        self.state = {"phase": "BANK", "next_tab_switch_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        # ---- Configuration ----
        # Where the range is. You can set this to a region key (constants.REGIONS) or a rect tuple.
        # TODO: set this to your preferred range location.
        self.range_area = (3206, 3214, 3217, 3226)

        # The object name(s) to use the fish on.
        self.range_object_names = ["Range", "Cooking range"]

        # What we withdraw/cook, with cooking level requirements.
        # Ordered by preference (best fish first that you can cook + have in bank).
        # Format: (raw_fish_name, cooking_level_required)
        self.raw_fish = [
            ("Raw swordfish", 45),
            ("Raw lobster", 40),
            ("Raw tuna", 30),
            ("Raw salmon", 25),
            ("Raw trout", 15),
            ("Raw herring", 5),
            ("Raw sardine", 1),
            ("Raw anchovies", 1),
            ("Raw shrimp", 1),
        ]

        # Withdraw 28 of the first raw fish we can find in bank (best-effort).
        # If you want strict behavior per fish type, you can customize required_items.
        self.bank_plan = BankPlanSimple(
            bank_area="ALKHARID_BANK",
            required_items=[],  # built dynamically in BANK phase
            deposit_all=True,
            equip_items={},
        )

        logging.info(f"[{self.id}] Plan initialized")

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

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "BANK")

        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        self._maybe_tab_switch()

        match phase:
            case "BANK":
                return self._handle_bank(ui)

            case "COOK":
                return self._handle_cook()

            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _pick_raw_fish_in_inventory(self) -> str | None:
        for nm, _lvl in self.raw_fish:
            if inventory.has_item(nm):
                return nm
        return None

    def _configure_bank_plan_for_fish(self):
        """
        Pick a fish type to withdraw based on:
        - what's available in the bank
        - what we have the cooking level to cook

        Withdraw quantity is -1 (withdraw all).
        """
        cooking_level = player.get_skill_level("cooking") or 1

        chosen = None
        chosen_req = None
        for nm, req in self.raw_fish:
            if cooking_level < int(req):
                continue
            if bank.has_item(nm):
                chosen = nm
                chosen_req = int(req)
                break

        if chosen is None:
            # Nothing cookable found in bank for our level
            self.bank_plan.required_items = []
            logging.warning(f"[{self.id}] No cookable raw fish found in bank for cooking level {cooking_level}")
            return False

        self.bank_plan.required_items = [{"name": chosen, "quantity": -1}]
        logging.info(f"[{self.id}] Withdrawing all: {chosen} (req cooking {chosen_req}, have {cooking_level})")
        return True

    def _handle_bank(self, ui) -> int:
        if not travel_to_bank("ALKHARID_BANK"):
            return self.loop_interval_ms

        if not bank.is_open():
            bank.open_bank()
            return self.loop_interval_ms

        ok = self._configure_bank_plan_for_fish()
        if not ok:
            # No fish left to cook -> transition to DONE.
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            self.bank_plan.reset()
            self.set_phase("DONE")
            return self.loop_interval_ms

        status = self.bank_plan.loop(ui)

        if status == BankPlanSimple.SUCCESS:
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)

            self.bank_plan.reset()
            self.set_phase("COOK")
            return self.loop_interval_ms

        if status == BankPlanSimple.ERROR:
            logging.error(f"[{self.id}] Banking error: {self.bank_plan.get_error_message()}")
            return self.loop_interval_ms

        return status

    def _handle_done(self) -> int:
        """
        Cooking is complete (no cookable raw fish found in bank).
        Idle here so the GUI clearly shows DONE.
        """
        sleep_exponential(2.0, 5.0, 1.2)
        return 1500

    def _handle_cook(self) -> int:
        # If we ran out of raw fish, go back to bank.
        raw = self._pick_raw_fish_in_inventory()
        if not raw:
            self.set_phase("BANK")
            return self.loop_interval_ms

        if player.get_player_animation() == "COOKING":
            return exponential_number(0.15, 5.0, 1)

        before = inventory.inv_count(raw)

        # 2) click the range (object)
        res = objects.click_object_closest_by_distance_simple_prefer_no_camera("Range", "Cook")
        if not res:
            return self.loop_interval_ms

        wait_until(lambda: widget_exists(17694720), max_wait_ms=15000)

        # 3) often space starts "Cook All" (varies by interface)
        sleep_exponential(0.15, 2, 1)
        press_spacebar()

        # Wait a bit for progress (either fish count decreases, or you start animating)
        wait_until(lambda: inventory.inv_count(raw) < before, max_wait_ms=5000)
        return self.loop_interval_ms


