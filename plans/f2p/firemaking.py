#!/usr/bin/env python3
"""
Firemaking Plan
===============

Two phases (like the updated cooking plan pattern):
- BANK: at Varrock West bank; deposit inventory, withdraw tinderbox + logs
- FIREMAKING: move to a start tile east of the bank, ensure a clear runway (15 tiles west),
              then light logs one-by-one while moving west each fire.
"""

import logging
from pathlib import Path
import sys

from actions import bank, inventory, player, wait_until
from actions.travel import go_to_tile, travel_to_bank
from helpers.utils import sleep_exponential, random_number, exponential_number
from helpers.runtime_utils import ipc

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan
from plans.utilities.bank_plan_simple import BankPlanSimple
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera


class FiremakingPlan(Plan):
    id = "FIREMAKING"
    label = "Firemaking Plan"
    description = """Lights logs for firemaking XP at Varrock West Bank. Creates a clear runway and lights logs one-by-one while moving west for efficient training.

Starting Area: Varrock West Bank
Required Items: Logs, Tinderbox"""
    DONE = 0

    def __init__(self):
        self.state = {
            "phase": "BANK",
            "start_target": None,  # (x, y) target we path to before committing (chosen once per run)
            "start_tile": None,  # (x, y)
            "start_y": None,     # int: y row we commit to for this run
            "offset": 0,         # how far west we've progressed from start_tile
            "log_name": None,    # which log we're burning this run
        }
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        # ---- Configuration ----
        # Start at Varrock West bank.
        self.bank_area = "VARROCK_WEST"

        # Choose a random start tile in this 3x3 area (east of Varrock West bank).
        # TODO: tweak these coords to your preferred line.
        self.start_area_3x3 = (3204, 3209, 3428, 3430)  # (min_x, max_x, min_y, max_y)

        # How far west we require to be fire-free before starting a run.
        self.runway_west_tiles = 15

        # Firemaking animation name (from constants.PLAYER_ANIMATIONS mapping).
        self.firemaking_anim = "FIREMAKING"

        # Logs with required firemaking levels (best first).
        # Format: (log_name, firemaking_level_required)
        self.logs_by_level = [
            ("Redwood logs", 90),
            ("Magic logs", 75),
            ("Yew logs", 60),
            ("Mahogany logs", 50),
            ("Maple logs", 45),
            ("Teak logs", 35),
            ("Willow logs", 30),
            ("Oak logs", 15),
            ("Logs", 1),
        ]

        self.tinderbox_name = "Tinderbox"

        # Bank plan is configured dynamically once we pick which logs to burn.
        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[],
            deposit_all=True,
            equip_items={},
        )

        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "BANK")

        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        match phase:
            case "BANK":
                return self._handle_bank(ui)
            case "FIREMAKING":
                return self._handle_firemaking()
            case "DONE":
                # Plan is finished (no more logs in bank).
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _pick_best_log_for_level(self) -> str | None:
        lvl = player.get_skill_level("firemaking") or 1
        for name, req in self.logs_by_level:
            if lvl < int(req):
                continue
            if bank.has_item(name):
                return name
        return None

    def _configure_bank_plan(self) -> bool:
        """
        Choose logs based on firemaking level + bank availability.
        Withdraw tinderbox (1) + all of the chosen log type (-1).
        """
        log_name = self._pick_best_log_for_level()
        if log_name is None:
            logging.warning(f"[{self.id}] No burnable logs found in bank for current firemaking level")
            self.bank_plan.required_items = [{"name": self.tinderbox_name, "quantity": 1}]
            self.state["log_name"] = None
            return False

        self.state["log_name"] = log_name
        self.bank_plan.required_items = [
            {"name": self.tinderbox_name, "quantity": 1},
            {"name": log_name, "quantity": -1},
        ]
        logging.info(f"[{self.id}] Banking for: {log_name} (withdraw all) + Tinderbox")
        return True

    def _handle_bank(self, ui) -> int:
        if not travel_to_bank(self.bank_area):
            return self.loop_interval_ms

        if not bank.is_open():
            bank.open_bank()
            return self.loop_interval_ms

        ok = self._configure_bank_plan()
        if not ok:
            # No logs left to burn -> transition to DONE.
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            self.bank_plan.reset()
            self.state["start_target"] = None
            self.state["start_tile"] = None
            self.state["start_y"] = None
            self.state["offset"] = 0
            self.set_phase("DONE")
            return self.loop_interval_ms

        status = self.bank_plan.loop(ui)
        if status == BankPlanSimple.SUCCESS:
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)

            self.bank_plan.reset()
            # reset our runway progress when re-banking
            self.state["start_target"] = None
            self.state["start_tile"] = None
            self.state["start_y"] = None
            self.state["offset"] = 0
            self.set_phase("FIREMAKING")
            return self.loop_interval_ms

        if status == BankPlanSimple.ERROR:
            logging.error(f"[{self.id}] Banking error: {self.bank_plan.get_error_message()}")
            return self.loop_interval_ms

        return status

    def _handle_done(self) -> int:
        """
        Firemaking is complete for this profile (no more logs found in bank).
        We just idle here so the GUI shows DONE.
        """
        # Slow down loop while done.
        sleep_exponential(2.0, 5.0, 1.2)
        return 1500

    def _rand_int(self, lo: int, hi: int) -> int:
        return int(random_number(float(lo), float(hi) + 1, output_type="int"))

    def _choose_start_tile(self) -> tuple[int, int]:
        min_x, max_x, min_y, max_y = self.start_area_3x3
        return self._rand_int(min_x, max_x), self._rand_int(min_y, max_y)

    def _in_start_strip(self, x: int, y: int) -> bool:
        """
        True if the player is within the 3x3 staging area.
        """
        min_x, max_x, min_y, max_y = self.start_area_3x3
        return int(min_x) <= int(x) <= int(max_x) and int(min_y) <= int(y) <= int(max_y)

    def _in_start_row_any_x(self, x: int, y: int, start_y: int) -> bool:
        """
        True if the player is on the correct chosen Y row, and X is within the start strip X bounds.
        """
        min_x, max_x, _min_y, _max_y = self.start_area_3x3
        return int(y) == int(start_y) and int(min_x) <= int(x) <= int(max_x)

    def _has_fire_on_tile(self, x: int, y: int, p: int = 0) -> bool:
        """
        Best-effort fire detection using IPC tile object query.
        """
        try:
            resp = ipc.get_object_at_tile(x=x, y=y, plane=p, name="Fire")
        except Exception:
            resp = None

        if not resp or not resp.get("ok"):
            return False

        objs = resp.get("objects") or []
        for obj in objs:
            nm = (obj.get("name") or "").lower()
            if "fire" in nm:
                return True
        return False

    def _runway_is_clear(self, start_x: int, start_y: int, p: int = 0) -> bool:
        """
        Check there are no fires on any tiles from (start_x, start_y) west for `runway_west_tiles`.
        """
        for dx in range(0, int(self.runway_west_tiles) + 1):
            if self._has_fire_on_tile(start_x - dx, start_y, p):
                return False
        return True

    def _handle_firemaking(self) -> int:
        log_name = self.state.get("log_name")
        if not log_name or not inventory.has_item(log_name):
            self.state["start_target"] = None
            self.state["start_tile"] = None
            self.state["start_y"] = None
            self.state["offset"] = 0
            self.set_phase("BANK")
            return self.loop_interval_ms
        if not inventory.has_item(self.tinderbox_name):
            self.state["start_target"] = None
            self.state["start_tile"] = None
            self.state["start_y"] = None
            self.state["offset"] = 0
            self.set_phase("BANK")
            return self.loop_interval_ms

        # If already firemaking, wait for it to finish before continuing.
        if player.get_player_animation() == self.firemaking_anim:
            sleep_exponential(0.25, 1.25, 1.2)
            return self.loop_interval_ms

        plane = player.get_player_plane(0)

        px, py = player.get_x(), player.get_y()
        if px is None or py is None:
            return self.loop_interval_ms

        # Pick the Y row for this run (Y matters; X can float in the start strip).
        if self.state.get("start_y") is None:
            _sx, sy = self._choose_start_tile()
            self.state["start_y"] = int(sy)

        start_y = int(self.state["start_y"])

        # Pick an approach target tile ONCE per run (so we don't re-roll every loop).
        if self.state.get("start_target") is None:
            min_x, max_x, _min_y, _max_y = self.start_area_3x3
            target_x = self._rand_int(int(min_x), int(max_x))
            self.state["start_target"] = (int(target_x), int(start_y))

        # If we don't have a committed start_tile yet, move to the start strip on the chosen Y row.
        if self.state.get("start_tile") is None:
            # If we're already on the correct Y row within the start strip X bounds, don't re-path.
            if self._in_start_row_any_x(px, py, start_y):
                # We've arrived in the staging strip; pause a bit, re-check, then lock start_tile to current tile.
                sleep_exponential(0.75, 1.35, 1.2)
                px2, py2 = player.get_x(), player.get_y()
                if px2 is None or py2 is None:
                    return self.loop_interval_ms
                if not self._in_start_row_any_x(px2, py2, start_y):
                    return self.loop_interval_ms

                # We reached the designated area. Wait ~1s, then re-check we're still in it,
                # then lock the run's start_tile to our current tile (X is flexible; Y must match).
                #
                # Validate runway from our current tile on this Y row before committing.
                if not self._runway_is_clear(int(px2), int(py2), p=plane):
                    # Try a new row / new start next tick.
                    self.state["start_target"] = None
                    self.state["start_tile"] = None
                    self.state["start_y"] = None
                    self.state["offset"] = 0
                    return self.loop_interval_ms

                self.state["start_tile"] = (int(px2), int(py2))
                self.state["offset"] = 0
                # Clear target after committing so we don't accidentally reuse it.
                self.state["start_target"] = None
            else:
                # Not yet in the correct staging strip row; walk to our chosen approach target.
                tx0, ty0 = self.state["start_target"]
                go_to_tile(int(tx0), int(ty0))
                return self.loop_interval_ms

        sx, sy = self.state["start_tile"]
        offset = int(self.state.get("offset") or 0)

        # Target tile for this log is start - offset (moving west each fire).
        tx = int(sx) - offset
        ty = int(sy)

        # If current target tile already has fire, step to the next tile west.
        if self._has_fire_on_tile(tx, ty, p=plane):
            self.state["offset"] = offset + 1
            return self.loop_interval_ms

        # Move to the tile if not already there.
        if (px, py) != (tx, ty):
            go_to_tile(tx, ty)
            return self.loop_interval_ms

        # Light a fire: use tinderbox on logs.
        before = inventory.inv_count(log_name)
        if not inventory.use_item_on_item(self.tinderbox_name, log_name):
            return self.loop_interval_ms

        # Wait until the tile we're lighting now has a fire on it.
        # This is more reliable than animation gating when animations are flaky/missed.
        wait_until(lambda: self._has_fire_on_tile(tx, ty, p=plane), max_wait_ms=20000)

        # If we consumed a log (or we finished the cycle), move west for next fire.
        after = inventory.inv_count(log_name)
        if after < before:
            self.state["offset"] = offset + 1
        else:
            # If no change, still advance to avoid getting stuck on a tile.
            self.state["offset"] = offset + 1

        return exponential_number(0.15, 2.0, 2)


