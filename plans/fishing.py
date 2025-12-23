#!/usr/bin/env python3
"""
Fishing Plan
============

Phases:
- BANK
- GO_TO_OCEAN_FISHING_AREA
- FISH_OCEAN_FISH
- GO_TO_RIVER_FISHING_AREA
- FISH_RIVER_FISH

This is intentionally structured like `plans/mining.py`, but for fishing.
You will likely want to tweak the area rectangles and fishing actions for your preferred spots.
"""

import logging
import time
from pathlib import Path
import sys

from actions import player, inventory, bank
from actions import wait_until
from actions import npc
from actions import tab
from actions.travel import in_area, go_to, travel_to_bank
from helpers.tab import is_tab_open
from helpers.utils import sleep_exponential, exponential_number

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera


class FishingPlan(Plan):
    id = "FISHING"
    label = "Fishing Plan"

    def __init__(self):
        self.state = {"phase": "BANK", "after_bank": "GO_TO_OCEAN_FISHING_AREA", "next_tab_switch_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        # ---- Configuration ----
        # NOTE: These are placeholders. Replace with your actual rectangles / region keys.
        # You can use either:
        #   - a region key from constants.REGIONS / constants.BANK_REGIONS, or
        #   - a raw rect tuple: (min_x, max_x, min_y, max_y)
        self.ocean_fishing_area = "LUMBRIDGE_OCEAN_FISHING_AREA"
        self.river_fishing_area = "LUMBRIDGE_RIVER_FISHING_AREA"

        # Fishing spot configuration
        self.fishing_spot_name = "Fishing spot"
        self.ocean_action = "Net"    # e.g. "Net", "Harpoon"
        self.river_action = "Lure"   # e.g. "Bait", "Lure"

        # Bank plan: deposit everything and withdraw basic fishing items if present.
        # (If you want strict requirements, make these required in your bank plan.)
        self.bank_plan = BankPlanSimple(
            bank_area=None,  # closest bank
            required_items=[
                {"name": "Small fishing net", "quantity": 1},
                {"name": "Fishing rod", "quantity": 1},
                {"name": "Fishing bait", "quantity": -1},
                {"name": "Fly fishing rod", "quantity": 1},
                {"name": "Feather", "quantity": -1},
            ],
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
        # Avoid tab peeks while bank is open
        if bank.is_open():
            return

        now = time.time()
        nxt = self.state.get("next_tab_switch_ts")
        if not isinstance(nxt, (int, float)):
            self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))
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

        self.state["next_tab_switch_ts"] = now + float(exponential_number(60.0, 1800.0, 0.5, output_type="float"))

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "BANK")

        if not player.logged_in():
            player.login()
            return self.loop_interval_ms
        if player.get_skill_level("Fishing") >= 5 and player.get_skill_level("Fishing") < 20:
            self.ocean_action = "Bait"

        self._maybe_tab_switch()

        match phase:
            case "BANK":
                return self._handle_bank(ui)

            case "GO_TO_OCEAN_FISHING_AREA":
                if not in_area(self.ocean_fishing_area):
                    go_to(self.ocean_fishing_area)
                    return self.loop_interval_ms
                self.set_phase("FISH_OCEAN_FISH")
                return self.loop_interval_ms

            case "FISH_OCEAN_FISH":
                return self._handle_fish(action=self.ocean_action, next_phase="GO_TO_RIVER_FISHING_AREA")

            case "GO_TO_RIVER_FISHING_AREA":
                if not in_area(self.river_fishing_area):
                    go_to(self.river_fishing_area)
                    return self.loop_interval_ms
                self.set_phase("FISH_RIVER_FISH")
                return self.loop_interval_ms

            case "FISH_RIVER_FISH":
                return self._handle_fish(action=self.river_action, next_phase="GO_TO_OCEAN_FISHING_AREA")

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_bank(self, ui) -> int:
        if not travel_to_bank():
            return self.loop_interval_ms

        if not bank.is_open():
            bank.open_bank()
            return self.loop_interval_ms

        status = self.bank_plan.loop(ui)

        if status == BankPlanSimple.SUCCESS:
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            if player.get_skill_level("FISHING") < 20:
                next_phase = "GO_TO_OCEAN_FISHING_AREA"
            else:
                next_phase = "GO_TO_RIVER_FISHING_AREA"
            self.state["after_bank"] = None
            self.set_phase(str(next_phase))
            return self.loop_interval_ms

        if status == BankPlanSimple.ERROR:
            logging.error(f"[{self.id}] Banking error: {self.bank_plan.get_error_message()}")
            return self.loop_interval_ms

        return status

    def _handle_fish(self, *, action: str, next_phase: str) -> int:
        # Drop fish if inventory is full
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, dropping fish...")
            inventory.drop_all(["Shrimp", "Anchovies", "Sardine", "Herring", "Trout", "Salmon"])
            return self.loop_interval_ms

        if player.get_skill_level("Fishing") >= 20 and not in_area("LUMBRIDGE_RIVER_FISHING_AREA"):
            self.set_phase("GO_TO_RIVER_FISHING_AREA")
            return self.loop_interval_ms

        # If we're already netting, just idle a bit.
        if player.get_player_animation() == "NETTING" or player.get_player_animation() == "BAITING":
            return sleep_exponential(0.15, 5.0, 1)

        # Click a nearby fishing spot.
        res = npc.click_npc_action_simple(self.fishing_spot_name, action)
        if res:
            # For netting, we can wait for the animation to start.
            if action.lower() == "net":
                wait_until(lambda: player.get_player_animation() == "NETTING", max_wait_ms=5000)
            if action.lower() == "bait" or action.lower() == "lure":
                wait_until(lambda: player.get_player_animation() == "BAITING", max_wait_ms=5000)
        return self.loop_interval_ms


