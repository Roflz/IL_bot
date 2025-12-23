#!/usr/bin/env python3
"""
Gnome Agility Course Plan (template)
===================================

Structure:
- GO_TO_COURSE: travel to the course area (rect or named region)
- One phase per obstacle (OBSTACLE_1, OBSTACLE_2, ...):
    - click the obstacle object with the configured action
    - wait briefly for movement to start / position to change
    - advance to the next obstacle phase (loops back to OBSTACLE_1 after the last)

You must fill in:
- course_area: either a region key (string) or a (minX,maxX,minY,maxY) tuple
- obstacles: list of dicts: {"phase": "...", "name": "...", "action": "..."}
"""

import logging
import time
from pathlib import Path
import sys

from actions import objects, player, wait_until
from actions import ground_items
from actions.travel import go_to, in_area
from helpers.utils import exponential_number, sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan


class GnomeAgilityCoursePlan(Plan):
    id = "GNOME_AGILITY_COURSE"
    label = "Agility: Gnome Course"

    def __init__(self):
        # ---- Config ----
        # Set this to either:
        # - a region key in constants.REGIONS (string), or
        # - a tuple: (min_x, max_x, min_y, max_y)
        self.course_area = "GNOME_AGILITY_COURSE"

        # Obstacle loop order. Replace names/actions with the exact object/action in your scene.
        # Tip: use RuneLite object inspector or IPC "objects" command to confirm names/actions.
        self.obstacles = [
            # NOTE: Log balance is a GROUND object (not GAME) in RuneLite, so we must search types=["GROUND"].
            {"phase": "OBSTACLE_1", "name": "Log balance", "action": "Walk-across", "types": ["GROUND"]},
            {"phase": "OBSTACLE_2", "name": "Obstacle net", "action": "Climb-over"},
            {"phase": "OBSTACLE_3", "name": "Tree branch", "action": "Climb"},
            {"phase": "OBSTACLE_4", "name": "Balancing rope", "action": "Walk-on", "types": ["GROUND"]},
            {"phase": "OBSTACLE_5", "name": "Tree branch", "action": "Climb-down"},
            {"phase": "OBSTACLE_6", "name": "Obstacle net", "action": "Climb-over"},
            {"phase": "OBSTACLE_7", "name": "Obstacle pipe", "action": "Squeeze-through"}
        ]

        # Ground loot configuration (e.g. "Mark of grace" on rooftop courses)
        # You can override per obstacle with keys:
        #   - loot_item_name: str
        #   - loot_radius: int
        self.loot_item_name = "Mark of grace"
        self.loot_radius = 5

        self.state = {"phase": "GO_TO_COURSE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        logging.info(f"[{self.id}] Plan initialized with {len(self.obstacles)} obstacles")

    def set_phase(self, phase: str, camera_setup: bool = True) -> str:
        self.state["phase"] = str(phase)
        return self.state["phase"]

    def loop(self, ui) -> int:
        if not player.logged_in():
            player.login()
            return self.loop_interval_ms

        phase = self.state.get("phase", "GO_TO_COURSE")

        if phase == "GO_TO_COURSE":
            return self._handle_go_to_course()

        # obstacle phases
        for idx, ob in enumerate(self.obstacles):
            if phase == ob["phase"]:
                return self._handle_obstacle(idx)

        logging.warning(f"[{self.id}] Unknown phase: {phase}; resetting to GO_TO_COURSE")
        self.set_phase("GO_TO_COURSE")
        return self.loop_interval_ms

    def _handle_go_to_course(self) -> int:
        # If course_area is a string region name, we can use in_area().
        if isinstance(self.course_area, str):
            if not in_area(self.course_area):
                go_to(self.course_area)
                return exponential_number(0.8, 4.0, 0.8)
        else:
            # Tuple rect: treat as "in area" when within bounds.
            try:
                min_x, max_x, min_y, max_y = self.course_area
                x, y = player.get_player_position() or (None, None)
                if not (isinstance(x, int) and isinstance(y, int) and min_x <= x <= max_x and min_y <= y <= max_y):
                    go_to(self.course_area)
                    return exponential_number(0.8, 4.0, 0.8)
            except Exception:
                # Bad config; keep trying but warn.
                logging.warning(f"[{self.id}] Invalid course_area config: {self.course_area}")

        # Arrived: start obstacle loop
        if self.obstacles:
            self.set_phase(self.obstacles[0]["phase"])
        return self.loop_interval_ms

    def _handle_obstacle(self, idx: int) -> int:
        if idx < 0 or idx >= len(self.obstacles):
            self.set_phase("GO_TO_COURSE")
            return self.loop_interval_ms

        ob = self.obstacles[idx]
        name = ob.get("name")
        action = ob.get("action")
        types = ob.get("types")  # optional; e.g. ["GROUND"] for Log balance

        # Opportunistic loot at each obstacle (before clicking the next obstacle).
        # If an item is found and picked up, stay on the same obstacle phase and try again next loop.
        loot_name = (ob.get("loot_item_name") or self.loot_item_name or "").strip()
        loot_radius = ob.get("loot_radius", self.loot_radius)
        try:
            loot_radius = int(loot_radius)
        except Exception:
            loot_radius = self.loot_radius
        if loot_name:
            loot_res = ground_items.loot(loot_name, radius=max(1, loot_radius))
            if loot_res:
                return exponential_number(0.25, 0.9, 1.4)

        # Snapshot position so we can detect progress
        before = player.get_player_position()

        # Use the existing object click helpers, but allow overriding the RuneLite object type.
        # For Log balance, set types=["GROUND"] in the obstacle config.
        res = objects.click_object_closest_by_distance_prefer_no_camera(
            name,
            action=action,
            types=types,
            exact_match_object=False,
            exact_match_target_and_action=False,
        )

        if not res:
            # Could not click the obstacle; wait a bit and retry same phase
            return exponential_number(0.4, 1.3, 1.2)

        # Wait for traversal to COMPLETE (end-tile based per obstacle).
        # Each obstacle should land on a specific unique tile; fill these in once and it's very stable.
        #
        # Tip: to get the end tile for an obstacle, traverse it once manually and read
        # player.get_player_position() after you land.
        match ob.get("phase"):
            case "OBSTACLE_1":  # Log balance
                end_tile = (2474, 3429)
                wait_until(lambda: player.get_player_position() == end_tile, max_wait_ms=15000)
            case "OBSTACLE_2":  # Obstacle net
                end_area = (2471, 2476, 3423, 3424)
                wait_until(lambda: in_area(end_area), max_wait_ms=15000)
            case "OBSTACLE_3":  # Tree branch (up)
                end_tile = (2473, 3420)
                wait_until(lambda: player.get_player_position() == end_tile, max_wait_ms=15000)
            case "OBSTACLE_4":  # Balancing rope
                end_tile = (2483, 3420)
                wait_until(lambda: player.get_player_position() == end_tile, max_wait_ms=15000)
            case "OBSTACLE_5":  # Tree branch (down)
                end_tile = (2487, 3420)
                wait_until(lambda: player.get_player_position() == end_tile, max_wait_ms=15000)
            case "OBSTACLE_6":  # Obstacle net
                end_area = (2483, 2488, 3427, 3428)
                wait_until(lambda: in_area(end_area), max_wait_ms=15000)
            case "OBSTACLE_7":
                end_area = (2483, 2488, 3437, 3438)
                wait_until(lambda: in_area(end_area), max_wait_ms=15000)
                time.sleep(2)

        # Advance to next obstacle (loop)
        nxt = (idx + 1) % len(self.obstacles)
        self.set_phase(self.obstacles[nxt]["phase"])
        return exponential_number(450, 1500, 2)


