#!/usr/bin/env python3
"""
Thieving Cake Stall Plan (template)
==================================

Very simple loop:
- BANK: go to a bank and deposit inventory (starting cleanup)
  - NOTE: this will travel to the closest bank by default; set bank_area if you want a specific bank.
- GO_TO_TILE: walk to a specific world tile (stand tile)
- STEAL: click the Cake stall to "Steal-from"
  - If inventory is full, drop all Cakes + Chocolate cakes (configurable)
"""

import logging
from pathlib import Path
import sys
import time

from actions import bank, inventory, objects, player, wait_until
from actions.travel import go_to_tile
from helpers.utils import exponential_number, sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple


class ThievingCakeStallPlan(Plan):
    id = "THIEVING_CAKE_STALL"
    label = "Thieving: Cake Stall"

    def __init__(self):
        self.state = {"phase": "STEAL", "last_hop_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # ---- Config you should set ----
        # Tile the character should stand on while stealing
        self.stand_tile_x = 2669
        self.stand_tile_y = 3310
        self.stand_tile_plane = 0

        # Object + action
        self.stall_names = ["Cake stall"]
        self.steal_action = "Steal-from"

        # Drops when inventory is full
        self.drop_items = ["Cake", "Chocolate slice", "Bread"]

        # Optional: set to a specific bank area name if you want.
        # If None, BankPlanSimple travels to the closest bank.
        self.bank_area = "ARDOUGNE_EAST_SOUTH_BANK"

        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[],
            deposit_all=True,
            equip_items={},
        )

        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase

    def _at_stand_tile(self) -> bool:
        x, y = player.get_player_position() or (None, None)
        return (x == self.stand_tile_x) and (y == self.stand_tile_y)

    def loop(self, ui) -> int:
        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        phase = self.state.get("phase", "BANK")

        match phase:
            case "BANK":
                return self._handle_bank(ui)
            case "GO_TO_TILE":
                return self._handle_go_to_tile()
            case "STEAL":
                return self._handle_steal()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _handle_bank(self, ui) -> int:
        status = self.bank_plan.loop(ui)
        if status == BankPlanSimple.SUCCESS:
            if bank.is_open():
                bank.close_bank()
                wait_until(bank.is_closed, max_wait_ms=3000)
            self.bank_plan.reset()
            self.set_phase("GO_TO_TILE")
            return self.loop_interval_ms

        return status

    def _handle_go_to_tile(self) -> int:
        if self._at_stand_tile():
            self.set_phase("STEAL")
            return self.loop_interval_ms

        go_to_tile(
            self.stand_tile_x,
            self.stand_tile_y,
            plane=self.stand_tile_plane,
            arrive_radius=0,
            aim_ms=700,
        )
        return exponential_number(0.8, 4.0, 0.8)

    def _handle_steal(self) -> int:
        # Stay on the stand tile (if we got moved)
        if not self._at_stand_tile():
            self.set_phase("GO_TO_TILE")
            return exponential_number(0.8, 4.0, 0.8)

        # If inventory full: drop food and continue stealing
        if inventory.is_full():
            inventory.drop_all(self.drop_items)
            return exponential_number(0.4, 1.2, 1.2)

        # Make sure we aren't stuck in bank UI
        if bank.is_open():
            bank.close_bank()
            return self.loop_interval_ms

        # Hop if another player is on our tile (avoid competition / reports).
        # Cooldown prevents spamming hop attempts if something fails.
        other = player.find_other_player_on_tile(self.stand_tile_x, self.stand_tile_y, plane=self.stand_tile_plane)
        if other:
            now = time.time()
            last = self.state.get("last_hop_ts")
            if not isinstance(last, (int, float)) or (now - float(last)) > 20.0:
                self.state["last_hop_ts"] = now
                wid = player.random_world(kind="p2p", exclude_current=True)
                if wid is not None:
                    logging.info(f"[{self.id}] Player on tile ({other.get('name')}); hopping to p2p world {wid}")
                    ok = player.hop_world(int(wid))
                    # After a hop we may get moved; re-walk to the stand tile.
                    self.set_phase("GO_TO_TILE")
                    return exponential_number(3.0, 6.0, 1.2) if ok else exponential_number(1.0, 2.5, 1.2)
            # If we're in cooldown, just wait a bit and retry.
            return exponential_number(0.8, 1.8, 1.2)

        # Keep the target constrained to a small area around our stand tile so we don't
        # accidentally click a similar stall elsewhere.
        stall_area = (
            int(self.stand_tile_x) - 5,
            int(self.stand_tile_x) + 5,
            int(self.stand_tile_y) - 5,
            int(self.stand_tile_y) + 5,
        )
        res = objects.click_object_in_area_action_auto_no_camera(
            "Baker's stall",
            area=stall_area,
            prefer_action=self.steal_action,
            exact_match_object=True,
            exact_match_target_and_action=False,
        )
        if not res:
            # If we can't find/click it, nudge a bit and retry.
            return exponential_number(0.6, 1.6, 1.1)

        # Small human-ish delay after stealing
        sleep_exponential(0.15, 6.0, 2)
        return self.loop_interval_ms


