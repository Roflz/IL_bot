#!/usr/bin/env python3
"""
Simple Attack Plan
==================

A simple plan that attacks NPCs within a specified area.
Easy to configure - just change the NPC name and area at the top of the file.

Configuration:
    - TARGET_NPC: Name of the NPC to attack (partial match allowed)
    - TARGET_AREA: Area name from constants.py (e.g., "FALADOR_COWS") or tuple (min_x, max_x, min_y, max_y)
"""

import logging
import time
from pathlib import Path
import sys
from typing import Optional, List, Union

from actions import player
from actions import combat
from actions import wait_until
from actions.travel import go_to_tile
from actions.combat import _is_npc_known_dead
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from helpers.utils import sleep_exponential
from services.click_with_camera import click_npc_with_camera
from helpers.runtime_utils import ipc
from constants import REGIONS

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan

# ============================================================================
# CONFIGURATION - Change these values to attack different NPCs/areas
# ============================================================================

# NPC names to attack (exact match required, tried in order)
# If first NPC is not found, will try the next one, etc.
TARGET_NPCS = ["Goblin", "Hobgoblin"]

# Area to attack in - can be:
#   - String: Area name from constants.py (e.g., "FALADOR_COWS", "LUMBRIDGE_GOBLINS", "GWD_BANDOS")
#   - Tuple: (min_x, max_x, min_y, max_y) for custom coordinates
TARGET_AREA = "GWD_BANDOS"

# ============================================================================
# End of configuration
# ============================================================================


class SimpleAttackPlan(Plan):
    id = "SIMPLE_ATTACK"
    label = "Simple Attack NPCs"
    description = """Simple combat plan for attacking NPCs in a specified area. Easy to configure by changing NPC names and target area. Attacks NPCs within the configured area for combat XP training.

Starting Area: Configurable (default: GWD_BANDOS)
Required Items: Combat equipment"""
    DONE = 0

    def __init__(self):
        self.state = {"phase": "ATTACK", "navigating_to_center": False}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # Configuration from module-level constants
        self.target_npcs = TARGET_NPCS  # List of NPC names to try in order
        self.target_area = TARGET_AREA
        
        # Calculate area center
        self.area_center = self._calculate_area_center(self.target_area)

        # Set up camera
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")

        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Target NPCs: {', '.join(self.target_npcs)} (tried in order)")
        logging.info(f"[{self.id}] Target Area: {self.target_area}")
        if self.area_center:
            logging.info(f"[{self.id}] Area center: ({self.area_center['x']}, {self.area_center['y']})")

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui) -> int:
        phase = self.state.get("phase", "ATTACK")

        # Check if player is logged out
        if not player.logged_in():
            self.set_phase("DONE")
            logging.info(f"[{self.id}] Player logged out, setting phase to DONE")
            return self.loop_interval_ms

        match phase:
            case "ATTACK":
                return self._handle_attack()
            case "NAVIGATE_TO_CENTER":
                return self._handle_navigate_to_center()
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
    
    def _is_near_center(self, radius: int = 5) -> bool:
        """
        Check if player is near the area center.
        
        Args:
            radius: Maximum distance in tiles to be considered "near" (default: 5)
            
        Returns:
            True if player is within radius of center, False otherwise
        """
        if not self.area_center:
            return False
        
        player_x = player.get_x()
        player_y = player.get_y()
        
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return False
        
        center_x = self.area_center["x"]
        center_y = self.area_center["y"]
        
        dx = abs(player_x - center_x)
        dy = abs(player_y - center_y)
        distance = dx + dy  # Manhattan distance
        
        return distance <= radius

    def _find_closest_npc_from_list_in_area(self, npc_names: List[str], area: str) -> Optional[dict]:
        """
        Find the NPC closest to area center from a list of NPC names (tried in order).
        
        Args:
            npc_names: List of NPC names to try in order (exact match required)
            area: Area name from constants.py (e.g., "GWD_BANDOS")
            
        Returns:
            NPC closest to area center from the first available name in the list, or None if not found
        """
        for npc_name in npc_names:
            npc = self._find_closest_npc_exact_in_area(npc_name, area)
            if npc:
                logging.debug(f"[{self.id}] Found {npc_name}, skipping remaining NPCs in list")
                return npc
        return None

    def _find_closest_npc_exact_in_area(self, exact_name: str, area: str) -> Optional[dict]:
        """
        Find the NPC with exact name match closest to the area center.
        
        Args:
            exact_name: Exact NPC name to match
            area: Area name from constants.py (e.g., "GWD_BANDOS")
            
        Returns:
            NPC with exact name match closest to the area center, or None if not found
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
        
        # Filter NPCs by exact name match, area, and health
        matching_npcs = []
        for npc in npcs:
            npc_name = npc.get("name", "")
            # Exact name match (case-insensitive)
            if npc_name.lower() == exact_name.lower():
                # Check if NPC is within the area bounds
                world_x = npc.get("world").get('x')
                world_y = npc.get("world").get('y')
                
                if (isinstance(world_x, int) and isinstance(world_y, int) and
                    min_x <= world_x <= max_x and min_y <= world_y <= max_y):
                    # Check NPC health - skip if NPC is known to be dead (health = 0)
                    if _is_npc_known_dead(npc):
                        logging.debug(f"[{self.id}] Skipping {npc_name} - health is 0 (dead)")
                        continue
                    
                    # NPC is alive (or health unknown), add to matching list
                    matching_npcs.append(npc)
        
        if not matching_npcs:
            return None
        
        # Calculate distance from each NPC to the area center
        if not self.area_center:
            # Fallback to closest to player if center not available
            closest_npc = min(matching_npcs, key=lambda npc: npc.get("distance", 999))
            return closest_npc
        
        center_x = self.area_center["x"]
        center_y = self.area_center["y"]
        
        # Find NPC closest to area center (Manhattan distance)
        def distance_to_center(npc):
            world_x = npc.get("world", {}).get('x')
            world_y = npc.get("world", {}).get('y')
            if isinstance(world_x, int) and isinstance(world_y, int):
                dx = abs(world_x - center_x)
                dy = abs(world_y - center_y)
                return dx + dy  # Manhattan distance
            return 999  # Invalid NPC, put at end
        
        closest_npc = min(matching_npcs, key=distance_to_center)
        center_distance = distance_to_center(closest_npc)
        logging.debug(f"[{self.id}] Selected {closest_npc.get('name')} - distance to center: {center_distance} tiles")
        return closest_npc

    def _handle_attack(self) -> int:
        """Attack NPCs in the target area."""
        # If already in combat, wait for it to finish
        if player.is_in_combat():
            logging.debug(f"[{self.id}] In combat, waiting...")
            wait_until(lambda: not player.is_in_combat(), max_wait_ms=30000)
            return self.loop_interval_ms

        # Find closest NPC from target list (tried in order) in the target area
        npc = self._find_closest_npc_from_list_in_area(self.target_npcs, self.target_area)
        
        if not npc:
            # No NPCs found - check if we need to navigate to center
            if not self._is_near_center(radius=5):
                npc_names_str = ", ".join(self.target_npcs)
                logging.info(f"[{self.id}] No {npc_names_str} found and not near center, navigating to area center...")
                self.set_phase("NAVIGATE_TO_CENTER")
                return self.loop_interval_ms
            else:
                npc_names_str = ", ".join(self.target_npcs)
                logging.debug(f"[{self.id}] No {npc_names_str} (exact match) found in area {self.target_area}, waiting...")
                sleep_exponential(1.0, 2.0, 1.2)
                return self.loop_interval_ms

        # Attack the NPC using click_npc_with_camera but with pitch disabled
        npc_name = npc.get('name', self.target_npcs[0] if self.target_npcs else "Unknown")
        world_coords = {
            "x": npc.get("world").get('x'),
            "y": npc.get("world").get('y'),
            "p": npc.get("world").get('p')
        }
        
        logging.info(f"[{self.id}] Attacking {npc_name} at distance {npc.get('distance')} (pitch disabled)")
        result = click_npc_with_camera(
            npc_name=npc_name,
            world_coords=world_coords,
            aim_ms=420,
            action="Attack",
            disable_pitch=True,  # Disable pitch movement
            area=self.target_area  # Only click NPCs within the target area
        )
        
        if result:
            logging.debug(f"[{self.id}] Attack initiated successfully")
            sleep_exponential(0.3, 0.6, 1.2)
        else:
            logging.warning(f"[{self.id}] Failed to attack NPC")
            sleep_exponential(0.5, 1.0, 1.2)

        return self.loop_interval_ms

    def _handle_navigate_to_center(self) -> int:
        """Navigate to the center of the target area."""
        if not self.area_center:
            logging.warning(f"[{self.id}] Cannot navigate - area center not calculated")
            self.set_phase("ATTACK")
            return self.loop_interval_ms
        
        # Check if we're already near the center
        if self._is_near_center(radius=5):
            logging.info(f"[{self.id}] Arrived at area center")
            self.set_phase("ATTACK")
            return self.loop_interval_ms
        
        # Check if we're in combat - wait for it to finish
        if player.is_in_combat():
            logging.debug(f"[{self.id}] In combat while navigating, waiting...")
            wait_until(lambda: not player.is_in_combat(), max_wait_ms=30000)
            return self.loop_interval_ms
        
        # Navigate to center
        center_x = self.area_center["x"]
        center_y = self.area_center["y"]
        center_p = self.area_center.get("p", 0)
        
        logging.info(f"[{self.id}] Navigating to area center: ({center_x}, {center_y})")
        result = go_to_tile(center_x, center_y, plane=center_p, arrive_radius=3, aim_ms=700)
        
        if result:
            logging.debug(f"[{self.id}] Navigation command sent")
            sleep_exponential(0.5, 1.0, 1.2)
        else:
            logging.warning(f"[{self.id}] Navigation failed, retrying...")
            sleep_exponential(1.0, 2.0, 1.2)
        
        return self.loop_interval_ms

    def _handle_done(self) -> int:
        """Plan is done."""
        sleep_exponential(2.0, 5.0, 1.2)
        return 30000  # Return every 30 seconds when done
