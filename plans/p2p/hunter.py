#!/usr/bin/env python3
"""
Hunter Plan
===========

A plan for hunting in the Piscatoris Hunter area.
- Levels < 25: Hunts Ruby Harvest butterflies
- Levels 25-34: Hunts Sapphire Glacialis butterflies
- Levels 35-42: Hunts both Snowy Knight and Sapphire Glacialis butterflies (prioritizes Snowy Knight)
- Levels >= 43: Falconry phase - catches Spotted kebbits with falcons
Clicks 'Catch' on target NPCs within the specified area.
Also manages bird snares for additional hunter XP (butterfly phases only).
"""

import logging
import random
from pathlib import Path
import sys
from typing import Optional, List

from actions import player
from actions.player import get_player_plane
from actions import wait_until
from actions import inventory
from actions import objects
from actions.ground_items import interact_ground_item
from helpers.npc import get_npcs_by_name
from actions.travel import go_to, in_area, go_to_tile
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.utils import sleep_exponential, exponential_number
from services.click_with_camera import click_npc_with_camera
from helpers.runtime_utils import ipc, dispatch
from constants import REGIONS

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hunter level thresholds
SAPPHIRE_GLACIALIS_LEVEL = 25
SNOWY_KNIGHT_LEVEL = 35
FALCONRY_LEVEL = 43

# Ruby Harvest configuration (levels < 25)
RUBY_HARVEST_NPC = "Ruby Harvest"
RUBY_HARVEST_AREA = "RUBY_HARVEST_AREA"
RUBY_HARVEST_BIRD_SNARE_TILES = [
    {"x": 2315, "y": 3597, "p": 0},
    {"x": 2314, "y": 3597, "p": 0}
]

# Sapphire Glacialis configuration (levels >= 25)
SAPPHIRE_GLACIALIS_NPC = "Sapphire Glacialis"
SAPPHIRE_GLACIALIS_AREA = "SAPPHIRE_GLACIALIS_AREA"
# Bird snare area: southeast 2723,3774 to northwest 2730,3780
# Using center tiles for the area
SAPPHIRE_GLACIALIS_BIRD_SNARE_TILES = [
    {"x": 2726, "y": 3777, "p": 0},  # Center of the bird snare area
    {"x": 2727, "y": 3777, "p": 0}   # Second center tile
]

# Snowy Knight configuration (levels >= 35, caught in same area as Sapphire Glacialis)
SNOWY_KNIGHT_NPC = "Snowy Knight"
SNOWY_KNIGHT_AREA = "SAPPHIRE_GLACIALIS_AREA"  # Same area as Sapphire Glacialis
SNOWY_KNIGHT_BIRD_SNARE_TILES = SAPPHIRE_GLACIALIS_BIRD_SNARE_TILES  # Same bird snare area

# Falconry configuration (levels >= 43)
FALCONRY_TARGET_NPC = "Spotted kebbit"  # TODO: Fill in actual NPC name
FALCONRY_CATCH_NPC = "Falcon"  # TODO: Fill in actual NPC name that appears after clicking
FALCONRY_AREA = "FALCONRY_AREA"  # TODO: Add area to constants.py

# ============================================================================
# End of configuration
# ============================================================================


class HunterPlan(Plan):
    id = "HUNTER"
    label = "Hunter"
    description = """Hunter training at Piscatoris Hunter area. Automatically selects targets based on level: Ruby Harvest (<25), Sapphire Glacialis (25-34), Snowy Knight (35-42), and Falconry (43+). Manages bird snares for additional XP.

Starting Area: Piscatoris Hunter Area
Required Items: Bird snares (for butterfly phases), Falcon (for falconry phase)"""
    DONE = 0

    def __init__(self):
        self.state = {"phase": "HUNT", "navigating_to_area": False}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # Get hunter level and configure based on level
        hunter_level = self._get_hunter_level()
        self._configure_for_level(hunter_level)
        
        self.bird_snare_item_name = "Bird snare"
        self.drop_items = ["Raw bird meat", "Bones"]
        self.drop_min_count = 5  # Minimum count of each item before dropping
        self.drop_probability = 0.4  # 40% chance to drop when conditions are met
        
        # Only set up bird snare areas if not in falconry phase
        if hasattr(self, 'bird_snare_center_tiles'):
            # Calculate 4x4 areas around each center tile
            self.bird_snare_areas = []
            for center in self.bird_snare_center_tiles:
                x, y = center["x"], center["y"]
                # 4x4 area: 2 tiles in each direction from center
                area = (x - 2, x + 2, y - 2, y + 2)
                self.bird_snare_areas.append({
                    "area": area,
                    "center": center
                })
        
        # Calculate area center (for hunt phase) or falconry area (for falconry phase)
        if hasattr(self, 'target_area'):
            self.area_center = self._calculate_area_center(self.target_area)
        elif hasattr(self, 'falconry_area'):
            self.area_center = self._calculate_area_center(self.falconry_area)

        # Set up camera
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Hunter level: {hunter_level}")
        if hasattr(self, 'phase') and self.phase == "FALCONRY":
            logging.info(f"[{self.id}] Phase: FALCONRY")
            logging.info(f"[{self.id}] Target NPC: {self.falconry_target_npc}")
            logging.info(f"[{self.id}] Catch NPC: {self.falconry_catch_npc}")
            logging.info(f"[{self.id}] Area: {self.falconry_area}")
        else:
            logging.info(f"[{self.id}] Target NPCs: {self.target_npcs}")
            logging.info(f"[{self.id}] Target Area: {self.target_area}")
        if self.area_center:
            logging.info(f"[{self.id}] Area center: ({self.area_center['x']}, {self.area_center['y']})")

    def _get_hunter_level(self) -> int:
        """Get current hunter level."""
        try:
            level = player.get_skill_level("hunter")
            return level if level is not None else 1
        except Exception:
            return 1

    def _configure_for_level(self, hunter_level: int):
        """Configure target NPC(s), area, and bird snare tiles based on hunter level."""
        if hunter_level >= FALCONRY_LEVEL:
            # Level 43+: Falconry phase
            self.phase = "FALCONRY"
            self.falconry_target_npc = FALCONRY_TARGET_NPC
            self.falconry_catch_npc = FALCONRY_CATCH_NPC
            self.falconry_area = FALCONRY_AREA
            logging.info(f"[{self.id}] Using Falconry configuration (level {hunter_level} >= {FALCONRY_LEVEL})")
        elif hunter_level >= SNOWY_KNIGHT_LEVEL:
            # Level 35-42: Catch both Snowy Knight and Sapphire Glacialis
            self.phase = "HUNT"
            self.target_npcs = [SNOWY_KNIGHT_NPC, SAPPHIRE_GLACIALIS_NPC]  # Prioritize Snowy Knight
            self.target_area = SNOWY_KNIGHT_AREA
            self.bird_snare_center_tiles = SNOWY_KNIGHT_BIRD_SNARE_TILES
            logging.info(f"[{self.id}] Using Snowy Knight + Sapphire Glacialis configuration (level {hunter_level} >= {SNOWY_KNIGHT_LEVEL})")
        elif hunter_level >= SAPPHIRE_GLACIALIS_LEVEL:
            # Level 25-34: Catch Sapphire Glacialis
            self.phase = "HUNT"
            self.target_npcs = [SAPPHIRE_GLACIALIS_NPC]
            self.target_area = SAPPHIRE_GLACIALIS_AREA
            self.bird_snare_center_tiles = SAPPHIRE_GLACIALIS_BIRD_SNARE_TILES
            logging.info(f"[{self.id}] Using Sapphire Glacialis configuration (level {hunter_level} >= {SAPPHIRE_GLACIALIS_LEVEL})")
        else:
            # Level < 25: Catch Ruby Harvest
            self.phase = "HUNT"
            self.target_npcs = [RUBY_HARVEST_NPC]
            self.target_area = RUBY_HARVEST_AREA
            self.bird_snare_center_tiles = RUBY_HARVEST_BIRD_SNARE_TILES
            logging.info(f"[{self.id}] Using Ruby Harvest configuration (level {hunter_level} < {SAPPHIRE_GLACIALIS_LEVEL})")
        
        # For backward compatibility, set target_npc to first in list (if not falconry)
        if hasattr(self, 'target_npcs'):
            self.target_npc = self.target_npcs[0]

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "HUNT")

        # Check if player is logged out
        if not player.logged_in():
            self.set_phase("DONE")
            logging.info(f"[{self.id}] Player logged out, setting phase to DONE")
            return self.loop_interval_ms

        # Check hunter level and switch phase if needed
        hunter_level = self._get_hunter_level()
        if hunter_level >= FALCONRY_LEVEL and phase != "FALCONRY":
            logging.info(f"[{self.id}] Hunter level {hunter_level} reached, switching to Falconry phase")
            self._configure_for_level(hunter_level)
            self.set_phase("FALCONRY")
            phase = "FALCONRY"
        elif hunter_level < FALCONRY_LEVEL and phase == "FALCONRY":
            logging.info(f"[{self.id}] Hunter level {hunter_level} below falconry threshold, switching back to Hunt phase")
            self._configure_for_level(hunter_level)
            self.set_phase("HUNT")
            phase = "HUNT"

        match phase:
            case "HUNT":
                return self._handle_hunt()
            case "FALCONRY":
                return self._handle_falconry()
            case "NAVIGATE_TO_AREA":
                return self._handle_navigate_to_area()
            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    def _calculate_area_center(self, area: str) -> Optional[dict]:
        """
        Calculate the center coordinates of an area.
        
        Args:
            area: Area name from constants.py
            
        Returns:
            Dict with center coordinates {"x": int, "y": int, "p": int} or None
        """
        if area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            return {"x": center_x, "y": center_y, "p": 0}
        return None
    
    def _is_in_area(self) -> bool:
        """
        Check if player is within the target area.
        
        Returns:
            True if player is within area bounds, False otherwise
        """
        if self.target_area not in REGIONS:
            return False
        
        min_x, max_x, min_y, max_y = REGIONS[self.target_area]
        player_x = player.get_x()
        player_y = player.get_y()
        
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return False
        
        return min_x <= player_x <= max_x and min_y <= player_y <= max_y

    def _find_closest_npc_from_list_in_area(self, npc_names: List[str], area: str) -> Optional[dict]:
        """
        Find the closest NPC from a list of names within a specific area,
        prioritizing the one closest to the area center and by name priority.
        
        Args:
            npc_names: List of exact NPC names to match, in order of priority
            area: Area name from constants.py
            
        Returns:
            Closest NPC with exact name match within the area (closest to area center),
            or None if not found.
        """
        # Resolve area coordinates
        if area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            logging.warning(f"[{self.id}] Unknown area: {area}")
            return None
        
        # Get all NPCs
        npcs_resp = ipc.get_npcs()
        if not npcs_resp or not npcs_resp.get("ok"):
            return None
        
        npcs = npcs_resp.get("npcs", [])
        if not npcs:
            return None
        
        # Calculate area center for distance comparison
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Try each NPC name in priority order
        for target_name in npc_names:
            matching_npcs = []
            exact_name_lower = target_name.lower().strip()
            for npc in npcs:
                npc_name = (npc.get("name") or "").strip()
                # Exact name match (case-insensitive, whitespace trimmed)
                if npc_name.lower() == exact_name_lower:
                    # Check if NPC is within the area bounds
                    world_x = npc.get("world").get('x')
                    world_y = npc.get("world").get('y')
                    
                    if (isinstance(world_x, int) and isinstance(world_y, int) and
                        min_x <= world_x <= max_x and min_y <= world_y <= max_y):
                        matching_npcs.append(npc)
            
            if matching_npcs:
                # Return the NPC closest to the area center from the current target_name list
                closest_npc = min(matching_npcs, key=lambda npc: abs(npc.get("world").get('x') - center_x) + abs(npc.get("world").get('y') - center_y))
                return closest_npc
        
        return None

    def _handle_hunt(self) -> int:
        """Catch target NPCs in the target area, prioritizing bird snares."""
        # Check hunter level and reconfigure if needed (in case player leveled up)
        hunter_level = self._get_hunter_level()
        needs_reconfig = False
        
        if hunter_level >= SNOWY_KNIGHT_LEVEL:
            if not hasattr(self, 'target_npcs') or SNOWY_KNIGHT_NPC not in self.target_npcs:
                needs_reconfig = True
        elif hunter_level >= SAPPHIRE_GLACIALIS_LEVEL:
            if not hasattr(self, 'target_npcs') or self.target_npcs != [SAPPHIRE_GLACIALIS_NPC]:
                needs_reconfig = True
        else:
            if not hasattr(self, 'target_npcs') or self.target_npcs != [RUBY_HARVEST_NPC]:
                needs_reconfig = True
        
        if needs_reconfig:
            logging.info(f"[{self.id}] Hunter level {hunter_level} changed, reconfiguring...")
            self._configure_for_level(hunter_level)
            # Recalculate bird snare areas
            self.bird_snare_areas = []
            for center in self.bird_snare_center_tiles:
                x, y = center["x"], center["y"]
                area = (x - 2, x + 2, y - 2, y + 2)
                self.bird_snare_areas.append({
                    "area": area,
                    "center": center
                })
            self.area_center = self._calculate_area_center(self.target_area)
        
        # Check if we're in the target area
        if not self._is_in_area():
            logging.info(f"[{self.id}] Not in {self.target_area}, navigating to area...")
            self.set_phase("NAVIGATE_TO_AREA")
            return self.loop_interval_ms

        # Drop raw bird meat and bones periodically
        self._maybe_drop_items()

        # Prioritize bird snares over butterfly catching
        bird_snare_result = self._handle_bird_snares()
        if bird_snare_result is not None:
            return exponential_number(1000, 2000, 1)
        
        # Find closest NPC from target list
        npc = self._find_closest_npc_from_list_in_area(self.target_npcs, self.target_area)
        
        if not npc:
            npc_names_str = ", ".join(self.target_npcs)
            logging.debug(f"[{self.id}] No {npc_names_str} found in area {self.target_area}, waiting...")
            sleep_exponential(1.0, 2.0, 1.2)
            return self.loop_interval_ms
        
        # Get hunter XP before catch
        hunter_xp_before = player.get_skill_xp("hunter")
        
        npc_name = npc.get('name', self.target_npcs[0])
        world_coords = {
            "x": npc.get("world").get('x'),
            "y": npc.get("world").get('y'),
            "p": npc.get("world").get('p')
        }
        
        result = click_npc_with_camera(
            npc_name=npc_name,
            world_coords=world_coords,
            aim_ms=420,
            action="Catch",
            disable_pitch=True,  # Disable pitch movement
            area=self.target_area  # Only click NPCs within the target area
        )
        
        if result:
            logging.debug(f"[{self.id}] Catch initiated successfully")
            # Wait until animation starts and hunter XP increases
            wait_until(
                lambda: player.get_player_animation() == 6606 and 
                        player.get_skill_xp("hunter") is not None and
                        player.get_skill_xp("hunter") != hunter_xp_before,
                max_wait_ms=10000
            )
            
            # Get hunter XP after catch
            hunter_xp_after = player.get_skill_xp("hunter")
            if hunter_xp_before is not None and hunter_xp_after is not None:
                xp_gained = hunter_xp_after - hunter_xp_before
                logging.debug(f"[{self.id}] Hunter XP: {hunter_xp_before} -> {hunter_xp_after} (+{xp_gained})")
            
            sleep_exponential(0.3, 0.6, 1.2)
        else:
            logging.warning(f"[{self.id}] Failed to catch NPC")
            sleep_exponential(0.5, 1.0, 1.2)

        return self.loop_interval_ms

    def _handle_navigate_to_area(self) -> int:
        """Navigate to the target area."""
        if not self.area_center:
            logging.warning(f"[{self.id}] Cannot navigate - area center not calculated")
            self.set_phase("HUNT")
            return self.loop_interval_ms
        
        # Check if we're already in the area
        if self._is_in_area():
            logging.info(f"[{self.id}] Arrived at {self.target_area}")
            self.set_phase("HUNT")
            return self.loop_interval_ms
        
        # Navigate to area center
        center_x = self.area_center["x"]
        center_y = self.area_center["y"]
        center_p = self.area_center.get("p", 0)
        
        logging.info(f"[{self.id}] Navigating to {self.target_area} center: ({center_x}, {center_y})")
        result = go_to_tile(center_x, center_y, plane=center_p, arrive_radius=3, aim_ms=700)
        
        if result:
            logging.debug(f"[{self.id}] Navigation command sent")
            sleep_exponential(0.5, 1.0, 1.2)
        else:
            logging.warning(f"[{self.id}] Navigation failed, retrying...")
            sleep_exponential(1.0, 2.0, 1.2)
        
        return self.loop_interval_ms

    def _handle_falconry(self) -> int:
        """
        Handle falconry phase: Click on Spotted kebbit, wait for Gyr Falcon, click it, verify success.
        """
        # Drop items if needed
        self._maybe_drop_falconry_items()
        
        # Get bones count before action
        bones_before = inventory.inv_count("Bones")
        npc_name = "Spotted kebbit"
        
        logging.info(f"[{self.id}] Clicking {npc_name} for falconry")
        result = click_npc_with_camera(
            npc_name=npc_name,
            aim_ms=420,
            action="Catch",
            disable_pitch=True,
        )
        
        if not result:
            logging.warning(f"[{self.id}] Failed to click {npc_name}")
            return self.loop_interval_ms
        
        if wait_until(lambda: len(get_npcs_by_name("Gyr Falcon")) > 0, max_wait_ms=2000):
            # Get the Gyr Falcon NPC
            result = click_npc_with_camera(
                npc_name="Gyr Falcon",
                aim_ms=420,
                action="Retrieve",
                disable_pitch=True,
            )
            if not result:
                logging.info(f"[{self.id}] Failed to click Gyr Falcon")
        else:
            return

        if wait_until(lambda: inventory.inv_count("Bones") > bones_before, max_wait_ms=3000):
            return
        else:
            return

    def _handle_done(self) -> int:
        """Plan is done."""
        sleep_exponential(2.0, 5.0, 1.2)
        return 30000  # Return every 30 seconds when done

    def _maybe_drop_items(self) -> None:
        """
        Periodically drop raw bird meat and bones from inventory.
        - Only drops if there are at least drop_min_count of each item
        - Has a probability (drop_probability) of dropping when conditions are met
        - Always drops if inventory is full
        """
        # Check if inventory is full - if so, always drop
        inventory_full = inventory.is_full()
        
        # Count items
        meat_count = inventory.count_unnoted_item("Raw bird meat")
        bones_count = inventory.count_unnoted_item("Bones")
        
        # Check if we have at least the minimum count of each
        has_enough_meat = meat_count >= self.drop_min_count
        has_enough_bones = bones_count >= self.drop_min_count
        
        # Decide whether to drop
        should_drop = False
        drop_reason = ""
        
        if inventory_full:
            # Always drop if inventory is full
            should_drop = True
            drop_reason = "inventory full"
        elif has_enough_meat and has_enough_bones:
            # Check probability if we have enough of both
            if random.random() < self.drop_probability:
                should_drop = True
                drop_reason = f"probability check (meat: {meat_count}, bones: {bones_count})"
        
        if should_drop:
            items_to_drop = []
            if meat_count > 0:
                items_to_drop.append("Raw bird meat")
            if bones_count > 0:
                items_to_drop.append("Bones")
            
            if items_to_drop:
                logging.info(f"[{self.id}] Dropping {', '.join(items_to_drop)} ({drop_reason})")
                inventory.drop_all(items_to_drop)

    def _maybe_drop_falconry_items(self) -> None:
        """
        Periodically drop bones and Spotted kebbit fur from inventory during falconry.
        - Only drops if there are at least drop_min_count of each item
        - Has a probability (drop_probability) of dropping when conditions are met
        - Always drops if inventory is full
        """
        # Check if inventory is full - if so, always drop
        inventory_full = inventory.is_full()
        
        # Count items
        bones_count = inventory.count_unnoted_item("Bones")
        fur_count = inventory.count_unnoted_item("Spotted kebbit fur")
        
        # Check if we have at least the minimum count of each
        has_enough_bones = bones_count >= self.drop_min_count
        has_enough_fur = fur_count >= self.drop_min_count
        
        # Decide whether to drop
        should_drop = False
        drop_reason = ""
        
        if inventory_full:
            # Always drop if inventory is full
            should_drop = True
            drop_reason = "inventory full"
        elif has_enough_bones and has_enough_fur:
            # Check probability if we have enough of both
            if random.random() < self.drop_probability:
                should_drop = True
                drop_reason = f"probability check (bones: {bones_count}, fur: {fur_count})"
        
        if should_drop:
            items_to_drop = []
            if bones_count > 0:
                items_to_drop.append("Bones")
            if fur_count > 0:
                items_to_drop.append("Spotted kebbit fur")
            
            if items_to_drop:
                logging.info(f"[{self.id}] Dropping {', '.join(items_to_drop)} ({drop_reason})")
                inventory.drop_all(items_to_drop)

    def _maybe_drop_falconry_items(self) -> None:
        """
        Periodically drop bones and Spotted kebbit fur from inventory during falconry.
        - Only drops if there are at least drop_min_count of each item
        - Has a probability (drop_probability) of dropping when conditions are met
        - Always drops if inventory is full
        """
        # Check if inventory is full - if so, always drop
        inventory_full = inventory.is_full()
        
        # Count items
        bones_count = inventory.count_unnoted_item("Bones")
        fur_count = inventory.count_unnoted_item("Spotted kebbit fur")
        
        # Check if we have at least the minimum count of each
        has_enough_bones = bones_count >= self.drop_min_count
        has_enough_fur = fur_count >= self.drop_min_count
        
        # Decide whether to drop
        should_drop = False
        drop_reason = ""
        
        if inventory_full:
            # Always drop if inventory is full
            should_drop = True
            drop_reason = "inventory full"
        elif has_enough_bones and has_enough_fur:
            # Check probability if we have enough of both
            if random.random() < self.drop_probability:
                should_drop = True
                drop_reason = f"probability check (bones: {bones_count}, fur: {fur_count})"
        
        if should_drop:
            items_to_drop = []
            if bones_count > 0:
                items_to_drop.append("Bones")
            if fur_count > 0:
                items_to_drop.append("Spotted kebbit fur")
            
            if items_to_drop:
                logging.info(f"[{self.id}] Dropping {', '.join(items_to_drop)} ({drop_reason})")
                inventory.drop_all(items_to_drop)

    def _get_all_bird_snares_in_area(self, area: tuple, area_info: dict = None) -> List[dict]:
        """
        Get all bird snares within a lenient area around the center point.
        Uses a large radius from the center to catch snares placed slightly outside the intended area.
        
        Args:
            area: Tuple (min_x, max_x, min_y, max_y) defining the intended area (for reference)
            area_info: Optional dict with center coordinates for lenient checking
            
        Returns:
            List of dicts with state info: [{"type": "object"|"ground", "object": {...}, "ground_item": {...}, "actions": [...], "tile": {...}}, ...]
            Empty list if no snares found
        """
        # Calculate center if not provided
        min_x, max_x, min_y, max_y = area
        if area_info and "center" in area_info:
            center_x = area_info["center"]["x"]
            center_y = area_info["center"]["y"]
        else:
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
        
        # Use a very lenient radius from center (10 tiles) - snares placed anywhere near the center will be found
        # This is much larger than the 4x4 placement area to catch snares placed slightly outside
        check_radius = 10
        
        snares = []
        
        # Get all objects near the center (use large radius)
        obj_resp = ipc.get_objects("Bird snare", types=["GAME"], radius=15)
        if obj_resp and obj_resp.get("ok"):
            objects_list = obj_resp.get("objects", [])
            for obj in objects_list:
                world_coords = obj.get("world", {})
                obj_x = world_coords.get("x")
                obj_y = world_coords.get("y")
                obj_p = world_coords.get("p", 0)
                
                # Lenient check - if it's within the radius from center, count it
                if isinstance(obj_x, int) and isinstance(obj_y, int):
                    dx = abs(obj_x - center_x)
                    dy = abs(obj_y - center_y)
                    if dx <= check_radius and dy <= check_radius:
                        actions = obj.get("actions", [])
                        snares.append({
                            "type": "object",
                            "object": obj,
                            "actions": actions,
                            "tile": {"x": obj_x, "y": obj_y, "p": obj_p}
                        })
        
        # Check for ground items near the center (use large radius)
        ground_resp = ipc.get_ground_items("Bird snare", radius=15)
        if ground_resp and ground_resp.get("ok"):
            items = ground_resp.get("items", [])
            for item in items:
                item_world = item.get("world", {})
                item_x = item_world.get("x")
                item_y = item_world.get("y")
                item_p = item_world.get("p", 0)
                
                # Lenient check - if it's within the radius from center, count it
                if isinstance(item_x, int) and isinstance(item_y, int):
                    dx = abs(item_x - center_x)
                    dy = abs(item_y - center_y)
                    if dx <= check_radius and dy <= check_radius:
                        actions = item.get("actions", [])
                        snares.append({
                            "type": "ground",
                            "ground_item": item,
                            "actions": actions,
                            "tile": {"x": item_x, "y": item_y, "p": item_p}
                        })
        
        return snares

    def _has_bird_snare_on_tile(self, x: int, y: int, p: int = 0) -> bool:
        """
        Check if there's a bird snare (object or ground item) on a specific tile.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            p: Plane (default: 0)
            
        Returns:
            True if a bird snare exists on the tile, False otherwise
        """
        # Check for objects on this tile
        obj_resp = ipc.get_objects("Bird snare", types=["GAME"], radius=1)
        if obj_resp and obj_resp.get("ok"):
            for obj in obj_resp.get("objects", []):
                world_coords = obj.get("world", {})
                obj_x = world_coords.get("x")
                obj_y = world_coords.get("y")
                obj_p = world_coords.get("p", 0)
                if obj_x == x and obj_y == y and obj_p == p:
                    return True
        
        # Check for ground items on this tile
        ground_resp = ipc.get_ground_items("Bird snare", radius=1)
        if ground_resp and ground_resp.get("ok"):
            for item in ground_resp.get("items", []):
                item_world = item.get("world", {})
                item_x = item_world.get("x")
                item_y = item_world.get("y")
                item_p = item_world.get("p", 0)
                if item_x == x and item_y == y and item_p == p:
                    return True
        
        return False

    def _find_empty_tile_in_area(self, area_info: dict) -> Optional[dict]:
        """
        Find an empty tile (no bird snare) within the area.
        
        Args:
            area_info: Dict with {"area": (min_x, max_x, min_y, max_y), "center": {"x": int, "y": int, "p": int}}
            
        Returns:
            Dict with {"x": int, "y": int, "p": int} for an empty tile, or None if no empty tile found
        """
        area = area_info["area"]
        min_x, max_x, min_y, max_y = area
        center = area_info["center"]
        center_p = center.get("p", 0)
        
        # Get all existing snares in the area to know which tiles are occupied
        existing_snares = self._get_all_bird_snares_in_area(area, area_info)
        occupied_tiles = set()
        for snare in existing_snares:
            tile = snare.get("tile", {})
            tile_x = tile.get("x")
            tile_y = tile.get("y")
            if isinstance(tile_x, int) and isinstance(tile_y, int):
                occupied_tiles.add((tile_x, tile_y))
        
        # Try tiles starting from center and expanding outward
        center_x, center_y = center["x"], center["y"]
        
        # Check center first
        if (center_x, center_y) not in occupied_tiles:
            if not self._has_bird_snare_on_tile(center_x, center_y, center_p):
                return {"x": center_x, "y": center_y, "p": center_p}
        
        # Try tiles in a spiral pattern from center
        for radius in range(1, 5):  # Check up to 4 tiles away from center
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check tiles on the perimeter of current radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    check_x = center_x + dx
                    check_y = center_y + dy
                    
                    # Check if tile is within area bounds
                    if not (min_x <= check_x <= max_x and min_y <= check_y <= max_y):
                        continue
                    
                    # Check if tile is occupied
                    if (check_x, check_y) in occupied_tiles:
                        continue
                    
                    # Check if there's actually a snare on this tile
                    if not self._has_bird_snare_on_tile(check_x, check_y, center_p):
                        return {"x": check_x, "y": check_y, "p": center_p}
        
        # No empty tile found
        return None

    def _lay_bird_snare_in_area(self, area_info: dict) -> bool:
        """
        Lay a bird snare from inventory within a 4x4 area.
        
        Args:
            area_info: Dict with {"area": (min_x, max_x, min_y, max_y), "center": {"x": int, "y": int, "p": int}}
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we have bird snare in inventory
        if not inventory.has_item(self.bird_snare_item_name):
            logging.warning(f"[{self.id}] No {self.bird_snare_item_name} in inventory")
            return False
        
        area = area_info["area"]
        min_x, max_x, min_y, max_y = area
        center = area_info["center"]
        center_x, center_y = center["x"], center["y"]
        
        # Try to get into the intended area for placement (strict for placement)
        # But we'll be lenient about checking if we succeed
        logging.info(f"[{self.id}] Navigating to area ({min_x}, {min_y}) to ({max_x}, {max_y}) to lay bird snare")
        max_attempts = 10
        for attempt in range(max_attempts):
            if in_area(area):
                break  # We're in the area, proceed
            
            result = go_to(area)
            if not result:
                logging.warning(f"[{self.id}] Failed to navigate to area on attempt {attempt + 1}")
                sleep_exponential(0.5, 1.0, 1.2)
                continue
            
            wait_until(lambda: in_area(area), max_wait_ms=3000)
        
        # If we're not in the exact area, check if we're at least close (within 3 tiles of center)
        if not in_area(area):
            player_x = player.get_x()
            player_y = player.get_y()
            if isinstance(player_x, int) and isinstance(player_y, int):
                dx = abs(player_x - center_x)
                dy = abs(player_y - center_y)
                if dx <= 3 and dy <= 3:
                    logging.debug(f"[{self.id}] Close to center ({dx}, {dy} tiles away), proceeding...")
                else:
                    logging.debug(f"[{self.id}] Not in exact area but proceeding anyway to attempt placement")
            # Proceed anyway - the checking logic will find the snare even if placed slightly outside
        
        # Check if there's already a bird snare on the current tile
        player_x = player.get_x()
        player_y = player.get_y()
        player_p = get_player_plane(0)  # Get current plane, default to 0
        
        if isinstance(player_x, int) and isinstance(player_y, int):
            if self._has_bird_snare_on_tile(player_x, player_y, player_p):
                logging.info(f"[{self.id}] Bird snare already exists on current tile ({player_x}, {player_y}), finding empty tile...")
                # Find an empty tile in the area
                empty_tile = self._find_empty_tile_in_area(area_info)
                if empty_tile:
                    logging.info(f"[{self.id}] Moving to empty tile ({empty_tile['x']}, {empty_tile['y']}) to lay bird snare")
                    result = go_to_tile(empty_tile["x"], empty_tile["y"], plane=empty_tile.get("p", 0), arrive_radius=0, aim_ms=700)
                    if result:
                        # Wait until we're on the target tile
                        wait_until(
                            lambda: player.get_x() == empty_tile["x"] and player.get_y() == empty_tile["y"],
                            max_wait_ms=5000
                        )
                        sleep_exponential(0.5, 1.0, 1.2)
                    else:
                        logging.warning(f"[{self.id}] Failed to navigate to empty tile, proceeding anyway")
                else:
                    logging.warning(f"[{self.id}] No empty tile found in area, proceeding anyway")
        
        # Use "Lay" action on bird snare from inventory
        sleep_exponential(0.5, 1.0, 1.2)
        logging.info(f"[{self.id}] Laying {self.bird_snare_item_name} in area ({min_x}, {min_y}) to ({max_x}, {max_y})")
        result = inventory.interact(self.bird_snare_item_name, "Lay")
        if not result:
            logging.warning(f"[{self.id}] Failed to use 'Lay' on {self.bird_snare_item_name}")
            return False
        
        sleep_exponential(1.0, 2.0, 1.2)

        # Wait until bird snare is placed (has "dismantle" action) - use lenient area check
        def check_snare_laid():
            snares = self._get_all_bird_snares_in_area(area, area_info)
            for snare in snares:
                actions = snare.get("actions", [])
                action_names = [a.lower() if isinstance(a, str) else "" for a in actions]
                if "dismantle" in action_names:
                    return True
            return False

        if wait_until(check_snare_laid, max_wait_ms=5000):
            snares = self._get_all_bird_snares_in_area(area, area_info)
            if snares:
                snare_tile = snares[0].get("tile", {})
                logging.debug(f"[{self.id}] Successfully laid bird snare near area at tile ({snare_tile.get('x')}, {snare_tile.get('y')})")
            sleep_exponential(0.5, 1.0, 1.2)
            return True
        else:
            # Don't fail - the snare might have been laid slightly outside the area, which is fine
            # The lenient checking will find it on the next loop
            logging.debug(f"[{self.id}] Snare placement check timed out, but may have been laid successfully")
            sleep_exponential(0.5, 1.0, 1.2)
            return True  # Return True anyway - don't break the script

    def _handle_bird_snare_in_area(self, area_info: dict) -> Optional[int]:
        """
        Handle bird snares within a 4x4 area. Maintains 2 snares per area.
        
        Args:
            area_info: Dict with {"area": (min_x, max_x, min_y, max_y), "center": {"x": int, "y": int, "p": int}}
        
        Returns:
            Loop interval if action was taken, None if no action needed (2 snares present and none need action)
        """
        area = area_info["area"]
        snares = self._get_all_bird_snares_in_area(area, area_info)
        snare_count = len(snares)
        
        # Handle snares that need action (check, lay, dismantle)
        # Priority: check > ground lay > dismantle only
        for state in snares:
            actions = state.get("actions", [])
            action_names = [a.lower() if isinstance(a, str) else "" for a in actions]
            snare_tile = state.get("tile", {})
            x, y, p = snare_tile.get("x"), snare_tile.get("y"), snare_tile.get("p", 0)
            snare_type = state.get("type", "")
            
            logging.debug(f"[{self.id}] Checking snare at ({x}, {y}), type: {snare_type}, actions: {action_names}")
            
            # Priority 1: Check if bird is caught (has "check" action)
            if "check" in action_names:
                logging.info(f"[{self.id}] Bird caught in snare at tile ({x}, {y}), checking...")
                result = objects.click_object_no_camera(
                    object_name="Bird snare",
                    action="Check",
                    world_coords={"x": x, "y": y, "p": p}
                )
                
                if result:
                    # Wait until this specific snare no longer exists
                    def check_snare_gone():
                        current_snares = self._get_all_bird_snares_in_area(area, area_info)
                        for snare in current_snares:
                            snare_t = snare.get("tile", {})
                            if snare_t.get("x") == x and snare_t.get("y") == y:
                                return False
                        return True
                    wait_until(check_snare_gone, max_wait_ms=5000)
                    logging.debug(f"[{self.id}] Checked bird snare successfully")
                    sleep_exponential(0.5, 1.0, 1.2)
                    return self.loop_interval_ms
                continue  # Try next snare if this one failed
            
            # Priority 2: Check if snare expired/collapsed (ground item with "lay" action)
            # Handle if it's a ground item type OR if it has "lay" action (collapsed trap)
            if snare_type == "ground" or "lay" in action_names:
                logging.info(f"[{self.id}] Bird snare expired/collapsed at tile ({x}, {y}), type: {snare_type}, laying again...")
                
                # For ground items, use ground item interaction method
                if snare_type == "ground":
                    ground_item = state.get("ground_item")
                    if ground_item:
                        result = interact_ground_item(ground_item, "Lay")
                    else:
                        logging.warning(f"[{self.id}] Ground item data missing for snare at ({x}, {y})")
                        continue
                else:
                    # For objects with "lay" action, use object method
                    result = objects.click_object_no_camera(
                        object_name="Bird snare",
                        action="Lay",
                        world_coords={"x": x, "y": y, "p": p}
                    )
                
                if result:
                    # Wait until this specific snare no longer exists
                    def check_snare_gone():
                        current_snares = self._get_all_bird_snares_in_area(area, area_info)
                        for snare in current_snares:
                            snare_t = snare.get("tile", {})
                            if snare_t.get("x") == x and snare_t.get("y") == y:
                                return False
                        return True
                    wait_until(check_snare_gone, max_wait_ms=5000)
                    logging.debug(f"[{self.id}] Laid bird snare on ground successfully")
                    sleep_exponential(0.5, 1.0, 1.2)
                    return self.loop_interval_ms
                continue  # Try next snare if this one failed
            
            # Priority 3: Check if snare failed (only "dismantle" action, no "investigate" or "check")
            # A failed snare will have ONLY "dismantle" as an action (no "investigate", no "check")
            if "dismantle" in action_names:
                # Filter out empty strings and check if only dismantle remains
                non_empty_actions = [a for a in action_names if a and a != ""]
                if len(non_empty_actions) == 1 and non_empty_actions[0] == "dismantle":
                    logging.info(f"[{self.id}] Bird snare failed at tile ({x}, {y}), dismantling and relaying...")
                    result = objects.click_object_no_camera(
                        object_name="Bird snare",
                        action="Dismantle",
                        world_coords={"x": x, "y": y, "p": p}
                    )
                    if result:
                        # Wait until this specific snare no longer exists
                        def check_snare_gone():
                            current_snares = self._get_all_bird_snares_in_area(area, area_info)
                            for snare in current_snares:
                                snare_t = snare.get("tile", {})
                                if snare_t.get("x") == x and snare_t.get("y") == y:
                                    return False
                            return True
                        wait_until(check_snare_gone, max_wait_ms=5000)
                        sleep_exponential(0.5, 1.0, 1.2)
                        # Then lay from inventory
                        if self._lay_bird_snare_in_area(area_info):
                            return self.loop_interval_ms
                    continue  # Try next snare if this one failed
        
        # If we have 2 snares and none needed action, we're done
        if snare_count >= 2:
            return None
        
        # If we have fewer than 2 snares, lay more
        snares_needed = 2 - snare_count
        if snares_needed > 0 and inventory.has_item(self.bird_snare_item_name):
            logging.info(f"[{self.id}] Only {snare_count} bird snare(s) in area, need {snares_needed} more, laying...")
            if self._lay_bird_snare_in_area(area_info):
                return self.loop_interval_ms
        
        return None

    def _handle_bird_snares(self) -> Optional[int]:
        """
        Handle all bird snares - check each area and take appropriate action.
        
        Returns:
            Loop interval if action was taken, None if no action needed
        """
        for area_info in self.bird_snare_areas:
            result = self._handle_bird_snare_in_area(area_info)
            if result is not None:
                return result
        return None
