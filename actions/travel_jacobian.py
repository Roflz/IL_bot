"""
Travel functions using Jacobian-based camera system.

This module provides travel functions that use the new Jacobian-based camera
positioning system. It's a wrapper around the existing travel logic but uses
the new camera system for precise positioning.
"""

import time
import logging
import random

from helpers.runtime_utils import ipc
from actions import player
from actions import travel
from actions.travel import (
    _calculate_distance,
    _calculate_click_location,
    _verify_click_path,
    _get_player_movement_state,
    _is_moving_toward_path,
    _first_blocking_door_from_waypoints,
    _handle_door_opening,
    _clear_movement_state,
    _movement_state,
    _calculate_area_target_tile,
    _is_within_area,
    _get_next_long_distance_waypoints,
    get_long_distance_waypoints
)
from helpers.navigation import get_nav_rect, closest_bank_key
from services.travel_camera_jacobian import click_ground_with_camera_jacobian


def go_to_jacobian(
    destination: str | tuple | list | dict,
    *,
    center: bool = False,
    destination_method: str = "center",
    arrive_radius: int = 0,
    calibration_data_path: str = None,
    target_screen_position: str = "center_top",
    aim_ms: int = 700
) -> bool | dict | None:
    """
    Unified movement system for both tile-based and area-based navigation using Jacobian camera.
    
    Args:
        destination: Can be:
            - Single tile: (x, y) tuple or {"x": x, "y": y} dict
            - Area: (min_x, max_x, min_y, max_y) tuple or area key string
        center: Deprecated, use destination_method="center"
        destination_method: For areas only - "center", "random", "center_weighted"
        arrive_radius: Consider "arrived" if within this many tiles (Manhattan distance)
        calibration_data_path: Path to calibration JSONL file (required for Jacobian)
        target_screen_position: Preset name for target screen position (default: "center_top")
        aim_ms: Maximum time for camera aiming (kept for compatibility, not used with Jacobian)
    
    Returns:
        - True if already at destination or if we're on a valid path
        - dict (click result) if a click was attempted
        - None on failure
    """
    global _movement_state
    
    try:
        if not calibration_data_path:
            logging.error("calibration_data_path is required for go_to_jacobian()")
            return None
        
        # Get player position
        player_x, player_y = player.get_x(), player.get_y()
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return None
        
        # Toggle run if needed
        run_energy = player.get_run_energy()
        if run_energy is not None and run_energy > 2000 and not player.is_run_on():
            player.toggle_run()
        
        # STEP 1: Normalize input to target tile
        target_x = None
        target_y = None
        rect = None
        rect_key = None
        is_single_tile = False
        
        if isinstance(destination, (tuple, list)):
            if len(destination) == 2:
                target_x, target_y = int(destination[0]), int(destination[1])
                rect = (target_x, target_x, target_y, target_y)
                rect_key = "tile"
                is_single_tile = True
            elif len(destination) == 4:
                rect = tuple(destination)
                rect_key = "custom"
        elif isinstance(destination, dict):
            if "x" in destination and "y" in destination:
                target_x, target_y = int(destination["x"]), int(destination["y"])
                rect = (target_x, target_x, target_y, target_y)
                rect_key = "tile"
                is_single_tile = True
        else:
            rect_key = str(destination)
            if rect_key == "CLOSEST_BANK":
                actual_bank_key = closest_bank_key()
                rect = get_nav_rect(actual_bank_key)
                rect_key = actual_bank_key
            else:
                rect = get_nav_rect(rect_key)
        
        if not rect or len(rect) != 4:
            return None
        
        if center:
            destination_method = "center"
        
        if target_x is None or target_y is None:
            target_tile = _calculate_area_target_tile(rect, rect_key, destination_method)
            target_x = target_tile["x"]
            target_y = target_tile["y"]
        
        # STEP 2: Check if already at destination
        if is_single_tile:
            if _calculate_distance(player_x, player_y, target_x, target_y) <= max(0, int(arrive_radius)):
                _clear_movement_state()
                return True
        else:
            if _is_within_area(player_x, player_y, rect):
                _clear_movement_state()
                return True
        
        # STEP 3: Check if destination changed - if so, clear state
        final_dest = {"x": target_x, "y": target_y}
        if is_single_tile:
            if _movement_state.get("final_destination") != final_dest:
                _clear_movement_state()
                _movement_state["final_destination"] = final_dest
                _movement_state["final_destination_type"] = "tile"
        else:
            area_dest = {"min_x": rect[0], "max_x": rect[1], "min_y": rect[2], "max_y": rect[3]}
            current_area = _movement_state.get("final_destination_area")
            current_type = _movement_state.get("final_destination_type")
            
            if current_type != "area" or current_area != area_dest:
                _clear_movement_state()
                _movement_state["final_destination_area"] = area_dest
                _movement_state["final_destination_type"] = "area"
                _movement_state["final_destination"] = final_dest
                _movement_state["area_target_tile"] = {"x": target_x, "y": target_y}
                _movement_state["area_rect_key"] = rect_key
            else:
                _movement_state["final_destination"] = final_dest
        
        # STEP 4: Simple check if already moving - just check path length
        movement_state = _get_player_movement_state()
        if movement_state and movement_state.get("is_moving"):
            last_clicked = _movement_state.get("last_clicked_tile")
            if last_clicked:
                clicked_x = last_clicked.get("x")
                clicked_y = last_clicked.get("y")
                if clicked_x is not None and clicked_y is not None:
                    path_to_clicked, _ = ipc.path(rect=(clicked_x - 1, clicked_x + 1, clicked_y - 1, clicked_y + 1), visualize=False)
                    if path_to_clicked and len(path_to_clicked) > 0:
                        threshold = random.randint(8, 10) if movement_state.get("is_running", False) else random.randint(3, 4)
                        if len(path_to_clicked) > threshold:
                            return True  # Still far from last clicked tile, don't click again
        
        # STEP 5: Get waypoints (long-distance OR short-distance) - ONE path call
        use_long_distance = False
        target_waypoint = None
        
        # Check if destination is in scene using IPC path response
        wps, test_resp = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
        
        goal_in_scene = False
        if test_resp and test_resp.get("ok"):
            goal_in_scene = test_resp.get("debug", {}).get("goalInScene", False)
        
        # Use long-distance if destination is NOT in scene
        if not goal_in_scene:
            waypoint_batch = _get_next_long_distance_waypoints(rect_key, rect, destination_method)
            if waypoint_batch:
                use_long_distance = True
                target_waypoint = waypoint_batch[0]
                # Get path to waypoint (not final destination)
                wps, _ = ipc.path(rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1), visualize=True)
                if not wps:
                    # Fallback to final destination path
                    wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
                    click_target_x = target_x
                    click_target_y = target_y
                else:
                    click_target_x = target_waypoint[0]
                    click_target_y = target_waypoint[1]
            else:
                # Fallback to short distance
                if not wps:
                    if is_single_tile:
                        wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
                    else:
                        wps, _ = ipc.path(rect=tuple(rect), visualize=True)
                click_target_x = target_x
                click_target_y = target_y
        else:
            # Short distance - wps already fetched above
            if not wps:
                if is_single_tile:
                    wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
                else:
                    wps, _ = ipc.path(rect=tuple(rect), visualize=True)
            click_target_x = target_x
            click_target_y = target_y
        
        if not wps:
            return None
        
        _movement_state["intended_path"] = wps
        
        # STEP 6: Handle doors before clicking
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            if not _handle_door_opening(door_plan, wps):
                return None
            return True
        
        # STEP 7: Calculate click location
        click_location = _calculate_click_location(player_x, player_y, click_target_x, click_target_y, is_precise=None)
        
        # STEP 8: Compensate for movement
        if movement_state and movement_state.get("is_moving"):
            movement_dir = movement_state.get("movement_direction")
            if movement_dir:
                dx = movement_dir.get("dx", 0)
                dy = movement_dir.get("dy", 0)
                compensation = 0.4 if movement_state.get("is_running", False) else 0.2
                if dx != 0:
                    click_location["x"] = int(click_location["x"] - (dx / abs(dx)) * compensation)
                if dy != 0:
                    click_location["y"] = int(click_location["y"] - (dy / abs(dy)) * compensation)
        
        # STEP 9: Click with Jacobian camera
        result = click_ground_with_camera_jacobian(
            world_coords=click_location,
            description=f"Move toward {rect_key if rect_key else 'destination'}",
            path_waypoints=wps,
            movement_state=movement_state,
            calibration_data_path=calibration_data_path,
            target_screen_position=target_screen_position,
            aim_ms=aim_ms,
            verify_tile=False,
            expected_tile=None,
            verify_path=False
        )
        
        if not result:
            return None
        
        # STEP 10: Store state (ONE path call)
        time.sleep(0.05)  # Small delay for game to process
        clicked_path, _ = ipc.path(rect=tuple(rect), visualize=False)
        if clicked_path:
            selected_tile = ipc.get_selected_tile()
            if selected_tile:
                _movement_state["last_clicked_tile"] = selected_tile
            _movement_state["current_path"] = clicked_path
            _movement_state["is_moving"] = True
            _movement_state["last_movement_check_ts"] = time.time()
        
        return result
    
    except Exception as e:
        logging.error(f"[GO_TO_JACOBIAN] Error: {e}")
        return None


def go_to_tile_jacobian(
    tile_x: int, 
    tile_y: int, 
    *, 
    plane: int | None = None, 
    arrive_radius: int = 0,
    calibration_data_path: str = None,
    target_screen_position: str = "center_top",
    aim_ms: int = 700
) -> dict | bool | None:
    """
    Wrapper around unified go_to_jacobian() for single tile movement.
    
    This function simply calls go_to_jacobian((tile_x, tile_y)) with the same parameters.
    All logic is now in the unified go_to_jacobian() function.
    
    Args:
        tile_x: World X coordinate
        tile_y: World Y coordinate
        plane: Optional plane. If None, uses current plane.
        arrive_radius: Consider "arrived" if within this many tiles (Manhattan distance).
        calibration_data_path: Path to calibration JSONL file (required for Jacobian)
        target_screen_position: Preset name for target screen position (default: "center_top")
        aim_ms: Maximum time for camera aiming (kept for compatibility, not used with Jacobian)
    
    Returns:
        - True if already within arrive_radius or if we're on a valid path
        - dict (click result) if a click was attempted
        - None on failure
    """
    return go_to_jacobian(
        (tile_x, tile_y),
        arrive_radius=arrive_radius,
        calibration_data_path=calibration_data_path,
        target_screen_position=target_screen_position,
        aim_ms=aim_ms
    )

