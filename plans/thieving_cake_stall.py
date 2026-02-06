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
    description = """Steals from cake stalls for thieving XP. Automatically drops inventory when full and manages positioning for optimal thieving rates.

Starting Area: Ardougne (Cake Stall location)
Required Items: None"""

    def __init__(self):
        self.state = {"phase": "STEAL", "last_hop_ts": None}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # ---- Config you should set ----
        # Tile the character should stand on while stealing
        self.stand_tile_x = 2669
        self.stand_tile_y = 3310
        self.stand_tile_plane = 0

        # Object + action (will be configured based on level)
        self.stall_name = "Cake stall"
        self.steal_action = "Steal-from"
        self.using_fruit_stall = False

        # Drops when inventory is full (will be configured based on level)
        self.drop_items = ["Cake", "Chocolate slice", "Bread"]

        # Level threshold for switching to fruit stall
        self.fruit_stall_level = 25

        # Optional: set to a specific bank area name if you want.
        # If None, BankPlanSimple travels to the closest bank.
        self.bank_area = "ARDOUGNE_EAST_SOUTH_BANK"

        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[],
            deposit_all=True,
            equip_items={},
        )

        # Configure based on initial thieving level
        self._configure_for_level()

        logging.info(f"[{self.id}] Plan initialized")

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase

    def _get_thieving_level(self) -> int:
        """Get current thieving level."""
        try:
            level = player.get_skill_level("thieving")
            return level if level else 1
        except Exception:
            return 1

    def _configure_for_level(self) -> None:
        """Configure stall name and drop items based on thieving level."""
        thieving_level = self._get_thieving_level()
        
        if thieving_level >= self.fruit_stall_level:
            self.stall_name = "Fruit Stall"
            self.drop_items = ["Banana", "Strawberry", "Lemon", "Lime", "Orange", "Pineapple", "Cooking apple", "Redberries", "Jangerberries", "Golovanova fruit top", "Strange fruit", "Papaya fruit"]
            self.using_fruit_stall = True
        else:
            self.stall_name = "Cake stall"
            self.drop_items = ["Cake", "Chocolate slice", "Bread"]
            self.using_fruit_stall = False

    def _at_stand_tile(self) -> bool:
        x, y = player.get_player_position() or (None, None)
        return (x == self.stand_tile_x) and (y == self.stand_tile_y)

    def loop(self, ui) -> int:
        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        # Reconfigure if level changed
        self._configure_for_level()

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
            # Skip GO_TO_TILE phase if using fruit stall
            if self.using_fruit_stall:
                self.set_phase("STEAL")
            else:
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
        # Only check for stand tile if using cake stall
        if not self.using_fruit_stall:
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

        # Only check for other players on tile if using cake stall
        if not self.using_fruit_stall:
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

        # Keep the target constrained to a small area so we don't
        # accidentally click a similar stall elsewhere.
        if self.using_fruit_stall:
            # For fruit stall, use player's current position +/- 2 tiles
            player_pos = player.get_player_position()
            if player_pos:
                player_x, player_y = player_pos
                stall_area = (
                    int(player_x) - 2,
                    int(player_x) + 2,
                    int(player_y) - 2,
                    int(player_y) + 2,
                )
            else:
                # Fallback if we can't get player position
                stall_area = None
        else:
            # For cake stall, use the hardcoded stand tile
            stall_area = (
                int(self.stand_tile_x) - 2,
                int(self.stand_tile_x) + 2,
                int(self.stand_tile_y) - 2,
                int(self.stand_tile_y) + 2,
            )

        res = objects.click_object_closest_by_distance_simple_no_camera(
            self.stall_name,
            prefer_action=self.steal_action,
        )
        
        if not res:
            # If we can't find/click it, nudge a bit and retry.
            return exponential_number(0.6, 1.6, 1.1)

        # Small human-ish delay after stealing
        sleep_exponential(0.15, 6.0, 2)
        return self.loop_interval_ms


