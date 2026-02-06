"""
Click with Camera System V2 - Rewritten click system with autonomous camera integration.

This module provides a clean interface for clicking objects, NPCs, and ground tiles
using the autonomous camera system that runs in the background.
"""

import logging
import time
import random
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from helpers.runtime_utils import ipc, dispatch
from helpers.utils import clean_rs, sleep_exponential, rect_beta_xy
from helpers.ipc import get_last_interaction
from services.camera_v2 import (
    get_camera_controller,
    set_current_interaction_target,
    clear_interaction_target
)


# ============================================================================
# Types and Data Structures
# ============================================================================

@dataclass
class ClickResult:
    """Result of a click operation."""
    success: bool
    screen_x: Optional[float] = None
    screen_y: Optional[float] = None
    interaction_verified: bool = False
    error_message: Optional[str] = None


# ============================================================================
# Screen Coordinate Management
# ============================================================================

class ScreenCoordinateManager:
    """Manages screen coordinate acquisition and validation."""
    
    def __init__(self):
        pass
    
    def get_object_screen_coords(
        self,
        world_x: int,
        world_y: int,
        world_z: int = 0,
        object_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get screen coordinates for an object at a world position.
        
        Returns:
            Dict with screen coordinates, bounds, and object data, or None if not found
        """
        try:
            objects_resp = ipc.get_object_at_tile(
                x=world_x,
                y=world_y,
                plane=world_z,
                name=None  # Don't filter by name here
            )
            
            if not objects_resp.get("ok") or not objects_resp.get("objects"):
                return None
            
            # Find matching object by name if provided
            objects = objects_resp.get("objects", [])
            if object_name:
                want_name = clean_rs(object_name).strip().lower()
                matching_object = None
                soft_match = None
                
                for obj in objects:
                    nm = clean_rs(obj.get("name", "")).strip().lower()
                    if nm == want_name:
                        matching_object = obj
                        break
                    if (want_name in nm) or (nm in want_name):
                        if soft_match is None:
                            soft_match = obj
                
                matching_object = matching_object or soft_match
            else:
                matching_object = objects[0] if objects else None
            
            if not matching_object:
                return None
            
            return {
                "object": matching_object,
                "bounds": matching_object.get("bounds", {}),
                "canvas": matching_object.get("canvas", {}),
                "name": matching_object.get("name", "")
            }
        except Exception as e:
            logging.debug(f"[CLICK_V2] Error getting object coords: {e}")
            return None
    
    def get_npc_screen_coords(
        self,
        npc_name: str,
        world_x: Optional[int] = None,
        world_y: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get screen coordinates for an NPC.
        
        Returns:
            Dict with screen coordinates, bounds, and NPC data, or None if not found
        """
        try:
            npc_resp = ipc.find_npc(npc_name)
            
            if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
                return None
            
            npc = npc_resp.get("npc")
            if not npc:
                return None
            
            # Verify world coordinates if provided
            if world_x is not None and world_y is not None:
                npc_world = npc.get("world", {})
                npc_x = npc_world.get("x")
                npc_y = npc_world.get("y")
                if npc_x != world_x or npc_y != world_y:
                    return None  # Wrong NPC
            
            return {
                "npc": npc,
                "bounds": npc.get("bounds", {}),
                "clickbox": npc.get("clickbox", {}),
                "canvas": npc.get("canvas", {}),
                "name": npc.get("name", npc_name)
            }
        except Exception as e:
            logging.debug(f"[CLICK_V2] Error getting NPC coords: {e}")
            return None
    
    def get_ground_screen_coords(
        self,
        world_x: int,
        world_y: int
    ) -> Optional[Dict]:
        """
        Get screen coordinates for a ground tile.
        
        Returns:
            Dict with screen coordinates and bounds, or None if off-screen
        """
        try:
            proj = ipc.project_world_tile(world_x, world_y)
            if not proj or not proj.get("ok"):
                return None
            
            return {
                "canvas": proj.get("canvas", {}),
                "bounds": proj.get("bounds", {}),
                "onscreen": proj.get("onscreen", False)
            }
        except Exception as e:
            logging.debug(f"[CLICK_V2] Error getting ground coords: {e}")
            return None
    
    def calculate_click_point(
        self,
        coords_data: Dict,
        randomize: bool = True
    ) -> Tuple[int, int]:
        """
        Calculate optimal click point from coordinate data.
        
        Args:
            coords_data: Coordinate data from get_*_screen_coords
            randomize: Whether to add randomization
        
        Returns:
            (x, y) click coordinates
        """
        # Try bounds first (more accurate)
        bounds = coords_data.get("bounds", {})
        if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
            cx, cy = rect_beta_xy((
                bounds.get("x", 0),
                bounds.get("x", 0) + bounds.get("width", 0),
                bounds.get("y", 0),
                bounds.get("y", 0) + bounds.get("height", 0)
            ), alpha=2.0, beta=2.0)
        else:
            # Fallback to canvas coordinates
            canvas = coords_data.get("canvas", {})
            cx = int(canvas.get("x", 0))
            cy = int(canvas.get("y", 0))
        
        # Add randomization if requested
        if randomize:
            cx += random.randint(-3, 3)
            cy += random.randint(-3, 3)
        
        return (cx, cy)
    
    def is_point_blocked(self, x: int, y: int) -> bool:
        """Check if a screen point is blocked by UI."""
        try:
            from helpers.widgets import is_point_in_blocking_widget
            return is_point_in_blocking_widget(x, y)
        except Exception:
            return False
    
    def find_unblocked_point(
        self,
        x: int,
        y: int,
        max_offset: int = 100
    ) -> Optional[Tuple[int, int]]:
        """
        Find an unblocked point near the given coordinates.
        
        Returns:
            (x, y) unblocked coordinates, or None if not found
        """
        # Get screen dimensions
        where = ipc.where() or {}
        screen_width = int(where.get("w", 1920))
        screen_height = int(where.get("h", 1080))
        margin = 10
        
        # Try different offsets
        offsets = [
            (0, -50), (0, 50), (-50, 0), (50, 0),  # Up, down, left, right
            (-30, -30), (30, -30), (-30, 30), (30, 30),  # Diagonals
            (0, -100), (0, 100), (-100, 0), (100, 0),  # Further cardinal
        ]
        
        for offset_x, offset_y in offsets:
            test_x = x + offset_x
            test_y = y + offset_y
            
            # Check bounds
            if not (margin <= test_x <= screen_width - margin and
                    margin <= test_y <= screen_height - margin):
                continue
            
            # Check if unblocked
            if not self.is_point_blocked(test_x, test_y):
                return (test_x, test_y)
        
        return None


# ============================================================================
# Interaction Verification
# ============================================================================

class InteractionVerifier:
    """Verifies that interactions were successful."""
    
    def __init__(self):
        pass
    
    def verify_object_interaction(
        self,
        object_name: str,
        action: Optional[str] = None,
        exact_match: bool = False
    ) -> bool:
        """Verify that an object interaction succeeded."""
        try:
            last_interaction = get_last_interaction()
            if not last_interaction:
                return False
            
            # Check action
            if action:
                clean_action = clean_rs(last_interaction.get("action", ""))
                want_action = action.lower()
                if exact_match:
                    action_match = clean_action.lower() == want_action
                else:
                    action_match = (want_action in clean_action.lower() or
                                  clean_action.lower() in want_action)
                if not action_match:
                    return False
            
            # Check target
            want = clean_rs(object_name).lower()
            tgt = clean_rs(last_interaction.get("target_name") or 
                          last_interaction.get("target", "")).lower()
            
            if exact_match:
                target_match = tgt == want
            else:
                target_match = (want in tgt) or (tgt and (tgt in want))
            
            return target_match
        except Exception:
            return False
    
    def verify_npc_interaction(
        self,
        npc_name: str,
        action: Optional[str] = None,
        exact_match: bool = False
    ) -> bool:
        """Verify that an NPC interaction succeeded."""
        try:
            last_interaction = get_last_interaction()
            if not last_interaction:
                return False
            
            # Check action
            if action:
                clean_action = clean_rs(last_interaction.get("action", ""))
                want_action = action.lower()
                if exact_match:
                    action_match = clean_action.lower() == want_action
                else:
                    action_match = (want_action in clean_action.lower() or
                                  clean_action.lower() in want_action)
                if not action_match:
                    return False
            
            # Check target
            want = clean_rs(npc_name).lower()
            tgt = clean_rs(last_interaction.get("target_name") or 
                          last_interaction.get("target", "")).lower()
            
            if exact_match:
                target_match = tgt == want
            else:
                target_match = (want in tgt) or (tgt and (tgt in want))
            
            return target_match
        except Exception:
            return False
    
    def verify_ground_interaction(
        self,
        world_x: int,
        world_y: int,
        tolerance: int = 2
    ) -> bool:
        """Verify that a ground click succeeded."""
        try:
            last_interaction = get_last_interaction()
            if not last_interaction or last_interaction.get("action") != "Walk here":
                return False
            
            # Check selected tile
            selected_tile = ipc.get_selected_tile()
            if not selected_tile:
                return False
            
            actual_x = selected_tile.get("x")
            actual_y = selected_tile.get("y")
            
            if actual_x is None or actual_y is None:
                return False
            
            # Calculate distance
            distance = abs(actual_x - world_x) + abs(actual_y - world_y)
            return distance <= tolerance
        except Exception:
            return False


# ============================================================================
# Main Click Handler
# ============================================================================

class ClickWithCameraHandler:
    """Main handler for clicking with camera integration."""
    
    def __init__(self):
        self.camera = get_camera_controller()
        self.coords_manager = ScreenCoordinateManager()
        self.verifier = InteractionVerifier()
    
    def _wait_for_camera_stable(self, timeout: float = 2.0) -> bool:
        """
        Wait for camera to stabilize after setting target.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if camera stabilized, False if timeout
        """
        start_time = time.time()
        check_interval = 0.05  # Check every 50ms
        stable_count_required = 3
        
        last_yaw = None
        last_pitch = None
        last_zoom = None
        yaw_stable_count = 0
        pitch_stable_count = 0
        zoom_stable_count = 0
        
        while (time.time() - start_time) < timeout:
            camera_data = ipc.get_camera()
            if not camera_data:
                time.sleep(check_interval)
                continue
            
            current_yaw = camera_data.get("yaw", 0)
            current_pitch = camera_data.get("pitch", 256)
            current_zoom = camera_data.get("scale", 512)
            
            # Check stability
            if last_yaw is not None:
                if current_yaw == last_yaw:
                    yaw_stable_count += 1
                    if yaw_stable_count >= stable_count_required:
                        yaw_stable = True
                    else:
                        yaw_stable = False
                else:
                    yaw_stable_count = 0
                    yaw_stable = False
            else:
                yaw_stable = False
            
            if last_pitch is not None:
                if current_pitch == last_pitch:
                    pitch_stable_count += 1
                    if pitch_stable_count >= stable_count_required:
                        pitch_stable = True
                    else:
                        pitch_stable = False
                else:
                    pitch_stable_count = 0
                    pitch_stable = False
            else:
                pitch_stable = False
            
            if last_zoom is not None:
                if current_zoom == last_zoom:
                    zoom_stable_count += 1
                    if zoom_stable_count >= stable_count_required:
                        zoom_stable = True
                    else:
                        zoom_stable = False
                else:
                    zoom_stable_count = 0
                    zoom_stable = False
            else:
                zoom_stable = False
            
            if yaw_stable and pitch_stable and zoom_stable:
                time.sleep(0.1)  # Small additional wait
                return True
            
            last_yaw = current_yaw
            last_pitch = current_pitch
            last_zoom = current_zoom
            
            time.sleep(check_interval)
        
        return False
    
    def click_object(
        self,
        object_name: str,
        world_x: int,
        world_y: int,
        world_z: int = 0,
        action: Optional[str] = None,
        exact_match: bool = False,
        max_attempts: int = 3,
        camera_wait_timeout: float = 2.0
    ) -> ClickResult:
        """
        Click an object with camera movement.
        
        Args:
            object_name: Name of the object
            world_x: World X coordinate
            world_y: World Y coordinate
            world_z: World Z coordinate (plane)
            action: Action to perform (None = left click)
            exact_match: Whether to use exact name matching
            max_attempts: Maximum number of click attempts
            camera_wait_timeout: Time to wait for camera to stabilize
        
        Returns:
            ClickResult with success status
        """
        for attempt in range(max_attempts):
            try:
                # Set interaction target - camera will automatically adjust
                set_current_interaction_target(world_x, world_y, world_z, target_type="object")
                
                # Wait for camera to stabilize
                if not self._wait_for_camera_stable(timeout=camera_wait_timeout):
                    logging.debug(f"[CLICK_V2] Camera didn't stabilize within {camera_wait_timeout}s")
                
                # Small delay for camera to settle
                time.sleep(0.1)
                
                # Get fresh screen coordinates
                coords_data = self.coords_manager.get_object_screen_coords(
                    world_x, world_y, world_z, object_name
                )
                
                if not coords_data:
                    logging.debug(f"[CLICK_V2] Object not found at ({world_x}, {world_y})")
                    continue
                
                # Calculate click point
                cx, cy = self.coords_manager.calculate_click_point(coords_data, randomize=True)
                
                # Check if point is blocked and find alternative
                if self.coords_manager.is_point_blocked(cx, cy):
                    unblocked = self.coords_manager.find_unblocked_point(cx, cy)
                    if unblocked:
                        cx, cy = unblocked
                        logging.debug(f"[CLICK_V2] Click point blocked, using ({cx}, {cy})")
                    else:
                        logging.warning(f"[CLICK_V2] Click point blocked and no alternative found")
                
                # Small delay before clicking
                sleep_exponential(0.05, 0.15, 1.5)
                
                # Perform click
                step = {
                    "action": "click-object-context",
                    "option": action,
                    "click": {
                        "type": "context-select",
                        "x": cx,
                        "y": cy,
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match
                    },
                    "target": {
                        "domain": "object",
                        "name": object_name,
                        "world": {"x": world_x, "y": world_y, "p": world_z}
                    },
                    "anchor": {"x": cx, "y": cy}
                }
                
                result = dispatch(step)
                
                if result:
                    # Verify interaction
                    if self.verifier.verify_object_interaction(object_name, action, exact_match):
                        # Clear target after successful click
                        clear_interaction_target()
                        return ClickResult(
                            success=True,
                            screen_x=cx,
                            screen_y=cy,
                            interaction_verified=True
                        )
                    else:
                        logging.debug(f"[CLICK_V2] Interaction verification failed, retrying...")
                        continue
                
            except Exception as e:
                logging.warning(f"[CLICK_V2] Error clicking object: {e}")
                continue
        
        # Clear target on failure
        clear_interaction_target()
        return ClickResult(
            success=False,
            error_message=f"Failed to click object after {max_attempts} attempts"
        )
    
    def click_npc(
        self,
        npc_name: str,
        world_x: Optional[int] = None,
        world_y: Optional[int] = None,
        action: Optional[str] = None,
        exact_match: bool = False,
        max_attempts: int = 3,
        camera_wait_timeout: float = 2.0
    ) -> ClickResult:
        """
        Click an NPC with camera movement.
        
        Args:
            npc_name: Name of the NPC
            world_x: Optional world X coordinate (for targeting)
            world_y: Optional world Y coordinate (for targeting)
            action: Action to perform (None = left click)
            exact_match: Whether to use exact name matching
            max_attempts: Maximum number of click attempts
            camera_wait_timeout: Time to wait for camera to stabilize
        
        Returns:
            ClickResult with success status
        """
        for attempt in range(max_attempts):
            try:
                # Try to find NPC first to get world coordinates if not provided
                if world_x is None or world_y is None:
                    npc_resp = ipc.find_npc(npc_name)
                    if npc_resp and npc_resp.get("ok") and npc_resp.get("found"):
                        npc = npc_resp.get("npc")
                        if npc:
                            npc_world = npc.get("world", {})
                            world_x = npc_world.get("x")
                            world_y = npc_world.get("y")
                            world_z = npc_world.get("plane", 0)
                        else:
                            continue
                    else:
                        continue
                else:
                    world_z = 0
                
                # Set interaction target - camera will automatically adjust
                set_current_interaction_target(world_x, world_y, world_z, target_type="npc")
                
                # Wait for camera to stabilize
                if not self._wait_for_camera_stable(timeout=camera_wait_timeout):
                    logging.debug(f"[CLICK_V2] Camera didn't stabilize within {camera_wait_timeout}s")
                
                # Small delay for camera to settle
                time.sleep(0.1)
                
                # Get fresh screen coordinates
                coords_data = self.coords_manager.get_npc_screen_coords(
                    npc_name, world_x, world_y
                )
                
                if not coords_data:
                    logging.debug(f"[CLICK_V2] NPC not found: {npc_name}")
                    continue
                
                # Calculate click point (use clickbox if available, else bounds, else canvas)
                npc = coords_data.get("npc", {})
                clickbox = coords_data.get("clickbox", {})
                bounds = coords_data.get("bounds", {})
                canvas = coords_data.get("canvas", {})
                
                if clickbox and clickbox.get("width", 0) > 0:
                    cx, cy = rect_beta_xy((
                        clickbox.get("x", 0),
                        clickbox.get("x", 0) + clickbox.get("width", 0),
                        clickbox.get("y", 0),
                        clickbox.get("y", 0) + clickbox.get("height", 0)
                    ), alpha=2.0, beta=2.0)
                elif bounds and bounds.get("width", 0) > 0:
                    cx, cy = rect_beta_xy((
                        bounds.get("x", 0),
                        bounds.get("x", 0) + bounds.get("width", 0),
                        bounds.get("y", 0),
                        bounds.get("y", 0) + bounds.get("height", 0)
                    ), alpha=2.0, beta=2.0)
                elif canvas.get("x") is not None and canvas.get("y") is not None:
                    cx = int(canvas["x"])
                    cy = int(canvas["y"])
                else:
                    continue
                
                # Add randomization
                cx += random.randint(-3, 3)
                cy += random.randint(-3, 3)
                
                # Check if point is blocked
                if self.coords_manager.is_point_blocked(cx, cy):
                    unblocked = self.coords_manager.find_unblocked_point(cx, cy)
                    if unblocked:
                        cx, cy = unblocked
                    else:
                        logging.warning(f"[CLICK_V2] Click point blocked and no alternative found")
                
                # Small delay before clicking
                sleep_exponential(0.05, 0.15, 1.5)
                
                # Perform click
                step = {
                    "action": "click-npc-context",
                    "option": action,
                    "click": {
                        "type": "context-select",
                        "x": cx,
                        "y": cy,
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match
                    },
                    "target": {
                        "domain": "npc",
                        "name": npc_name,
                        "world": {"x": world_x, "y": world_y, "p": world_z}
                    },
                    "anchor": {"x": cx, "y": cy}
                }
                
                result = dispatch(step)
                
                if result:
                    # Verify interaction
                    if self.verifier.verify_npc_interaction(npc_name, action, exact_match):
                        # Clear target after successful click
                        clear_interaction_target()
                        return ClickResult(
                            success=True,
                            screen_x=cx,
                            screen_y=cy,
                            interaction_verified=True
                        )
                    else:
                        logging.debug(f"[CLICK_V2] Interaction verification failed, retrying...")
                        continue
                
            except Exception as e:
                logging.warning(f"[CLICK_V2] Error clicking NPC: {e}")
                continue
        
        # Clear target on failure
        clear_interaction_target()
        return ClickResult(
            success=False,
            error_message=f"Failed to click NPC after {max_attempts} attempts"
        )
    
    def click_ground(
        self,
        world_x: int,
        world_y: int,
        max_attempts: int = 3,
        verify_tile: bool = True,
        camera_wait_timeout: float = 2.0
    ) -> ClickResult:
        """
        Click ground tile with camera movement.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            max_attempts: Maximum number of click attempts
            verify_tile: Whether to verify the clicked tile
            camera_wait_timeout: Time to wait for camera to stabilize
        
        Returns:
            ClickResult with success status
        """
        for attempt in range(max_attempts):
            try:
                # Set interaction target - camera will automatically adjust
                set_current_interaction_target(world_x, world_y, 0, target_type="ground")
                
                # Wait for camera to stabilize
                if not self._wait_for_camera_stable(timeout=camera_wait_timeout):
                    logging.debug(f"[CLICK_V2] Camera didn't stabilize within {camera_wait_timeout}s")
                
                # Small delay for camera to settle
                time.sleep(0.1)
                
                # Get fresh screen coordinates
                coords_data = self.coords_manager.get_ground_screen_coords(world_x, world_y)
                
                if not coords_data or not coords_data.get("onscreen"):
                    logging.debug(f"[CLICK_V2] Ground tile off-screen at ({world_x}, {world_y})")
                    continue
                
                # Calculate click point
                bounds = coords_data.get("bounds", {})
                canvas = coords_data.get("canvas", {})
                
                if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                    base_x = bounds.get("x", 0) + bounds.get("width", 0) // 2
                    base_y = bounds.get("y", 0) + bounds.get("height", 0) // 2
                    cx = base_x + random.randint(-bounds.get("width", 0) // 4, bounds.get("width", 0) // 4)
                    cy = base_y + random.randint(-bounds.get("height", 0) // 4, bounds.get("height", 0) // 4)
                else:
                    base_x = int(canvas.get("x", 0))
                    base_y = int(canvas.get("y", 0))
                    cx = base_x + random.randint(-5, 5)
                    cy = base_y + random.randint(-5, 5)
                
                # Check if point is blocked
                if self.coords_manager.is_point_blocked(cx, cy):
                    unblocked = self.coords_manager.find_unblocked_point(cx, cy)
                    if unblocked:
                        cx, cy = unblocked
                    else:
                        logging.warning(f"[CLICK_V2] Click point blocked and no alternative found")
                
                # Small delay before clicking
                sleep_exponential(0.05, 0.15, 1.5)
                
                # Press shift for "Walk here" action
                try:
                    ipc.key_press("SHIFT")
                except Exception:
                    pass
                
                # Perform click
                result = ipc.click(cx, cy, button=1)
                
                # Release shift
                try:
                    ipc.key_release("SHIFT")
                except Exception:
                    pass
                
                if not result:
                    continue
                
                # Wait for game to process click
                time.sleep(0.05)
                
                # Verify interaction if requested
                if verify_tile:
                    if self.verifier.verify_ground_interaction(world_x, world_y):
                        # Clear target after successful click
                        clear_interaction_target()
                        return ClickResult(
                            success=True,
                            screen_x=cx,
                            screen_y=cy,
                            interaction_verified=True
                        )
                    else:
                        logging.debug(f"[CLICK_V2] Ground interaction verification failed, retrying...")
                        continue
                else:
                    # No verification - assume success
                    clear_interaction_target()
                    return ClickResult(
                        success=True,
                        screen_x=cx,
                        screen_y=cy,
                        interaction_verified=False
                    )
                
            except Exception as e:
                logging.warning(f"[CLICK_V2] Error clicking ground: {e}")
                continue
        
        # Clear target on failure
        clear_interaction_target()
        return ClickResult(
            success=False,
            error_message=f"Failed to click ground after {max_attempts} attempts"
        )


# ============================================================================
# Convenience Functions
# ============================================================================

_click_handler: Optional[ClickWithCameraHandler] = None


def get_click_handler() -> ClickWithCameraHandler:
    """Get or create the global click handler instance."""
    global _click_handler
    if _click_handler is None:
        _click_handler = ClickWithCameraHandler()
    return _click_handler


def click_object_with_camera(
    object_name: str,
    world_x: int,
    world_y: int,
    world_z: int = 0,
    action: Optional[str] = None,
    exact_match: bool = False,
    **kwargs
) -> Optional[Dict]:
    """
    Convenience function for clicking objects with camera.
    
    Returns:
        Dispatch result dict if successful, None otherwise
    """
    handler = get_click_handler()
    result = handler.click_object(
        object_name, world_x, world_y, world_z, action, exact_match, **kwargs
    )
    return {"ok": result.success} if result.success else None


def click_npc_with_camera(
    npc_name: str,
    world_x: Optional[int] = None,
    world_y: Optional[int] = None,
    action: Optional[str] = None,
    exact_match: bool = False,
    **kwargs
) -> Optional[Dict]:
    """
    Convenience function for clicking NPCs with camera.
    
    Returns:
        Dispatch result dict if successful, None otherwise
    """
    handler = get_click_handler()
    result = handler.click_npc(npc_name, world_x, world_y, action, exact_match, **kwargs)
    return {"ok": result.success} if result.success else None


def click_ground_with_camera(
    world_x: int,
    world_y: int,
    verify_tile: bool = True,
    **kwargs
) -> Optional[Dict]:
    """
    Convenience function for clicking ground with camera.
    
    Returns:
        Dispatch result dict if successful, None otherwise
    """
    handler = get_click_handler()
    result = handler.click_ground(world_x, world_y, verify_tile=verify_tile, **kwargs)
    return {"ok": result.success} if result.success else None
