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
    Click once, then monitor path until it ends or deviates.
    
    This function:
    1. Normalizes input (tile or area) to target tile
    2. Checks if already at destination
    3. Checks if we have a valid current path and don't need to click again
    4. Gets waypoints (long-distance collision map OR short-distance IPC path)
    5. Clicks toward next waypoint (not final destination)
    6. Follows waypoints step by step
    7. Arrives when close to final destination
    
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
        # Get calibration data path (required)
        if not calibration_data_path:
            logging.error("calibration_data_path is required for go_to_jacobian()")
            return None
        
        # Get player position first
        player_x, player_y = player.get_x(), player.get_y()
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return None
        
        # LOG: Initial destination input
        logging.info(f"[GO_TO_JACOBIAN] ===== STARTING go_to_jacobian() =====")
        logging.info(f"[GO_TO_JACOBIAN] Input destination: {destination}")
        logging.info(f"[GO_TO_JACOBIAN] Player position: ({player_x}, {player_y})")
        
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
        
        # Check if input is a single tile
        if isinstance(destination, (tuple, list)):
            if len(destination) == 2:
                # Single tile: (x, y)
                target_x, target_y = int(destination[0]), int(destination[1])
                rect = (target_x, target_x, target_y, target_y)  # 1x1 area
                rect_key = "tile"
                is_single_tile = True
            elif len(destination) == 4:
                # Area: (min_x, max_x, min_y, max_y)
                rect = tuple(destination)
                rect_key = "custom"
        elif isinstance(destination, dict):
            if "x" in destination and "y" in destination:
                # Single tile: {"x": x, "y": y}
                target_x, target_y = int(destination["x"]), int(destination["y"])
                rect = (target_x, target_x, target_y, target_y)  # 1x1 area
                rect_key = "tile"
                is_single_tile = True
        else:
            # Area key string
            rect_key = str(destination)
            # Handle special case for CLOSEST_BANK
            if rect_key == "CLOSEST_BANK":
                actual_bank_key = closest_bank_key()
                rect = get_nav_rect(actual_bank_key)
                rect_key = actual_bank_key
            else:
                rect = get_nav_rect(rect_key)
        
        if not rect or len(rect) != 4:
            return None
        
        # Handle deprecated center parameter
        if center:
            destination_method = "center"
        
        # Calculate target tile if not already set (for areas)
        if target_x is None or target_y is None:
            target_tile = _calculate_area_target_tile(rect, rect_key, destination_method)
            target_x = target_tile["x"]
            target_y = target_tile["y"]
        
        # LOG: Normalized destination
        logging.info(f"[GO_TO_JACOBIAN] Normalized destination: tile=({target_x}, {target_y}), rect={rect}, rect_key={rect_key}, is_single_tile={is_single_tile}")
        
        # STEP 2: Check if already at destination
        distance = _calculate_distance(player_x, player_y, target_x, target_y)
        
        # LOG: Scene area check
        # Estimate scene area (RuneScape scenes are typically 104x104 tiles, centered roughly on player)
        # Scene base is typically at multiples of 64, but we'll estimate from player position
        scene_size = 104
        scene_half = scene_size // 2
        scene_min_x = player_x - scene_half
        scene_max_x = player_x + scene_half
        scene_min_y = player_y - scene_half
        scene_max_y = player_y + scene_half
        scene_area = (scene_min_x, scene_max_x, scene_min_y, scene_max_y)
        
        # Check if destination is in scene
        dest_in_scene = (scene_min_x <= target_x <= scene_max_x and scene_min_y <= target_y <= scene_max_y)
        
        logging.info(f"[GO_TO_JACOBIAN] Scene area: ({scene_min_x}, {scene_max_x}, {scene_min_y}, {scene_max_y})")
        logging.info(f"[GO_TO_JACOBIAN] Destination in scene: {dest_in_scene}")
        logging.info(f"[GO_TO_JACOBIAN] Distance to destination: {distance} tiles")
        if is_single_tile:
            # For single tile, check exact distance
            if distance <= max(0, int(arrive_radius)):
                _clear_movement_state()
                return True
        else:
            # For area, check if within area bounds
            if _is_within_area(player_x, player_y, rect):
                _clear_movement_state()
                return True
        
        # STEP 3: Check if destination changed - if so, clear state
        final_dest = {"x": target_x, "y": target_y}
        if is_single_tile:
            # For single tile, store as final_destination
            if _movement_state.get("final_destination") != final_dest:
                _clear_movement_state()
                _movement_state["final_destination"] = final_dest
                _movement_state["final_destination_type"] = "tile"
        else:
            # For area, store as area
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
        
        # STEP 4: Check if already moving toward destination using PATH COMPARISON
        movement_state = _get_player_movement_state()
        if movement_state and movement_state.get("is_moving"):
            # Check if we're close to last clicked tile (path distance) - if so, click ahead again
            last_clicked = _movement_state.get("last_clicked_tile")
            if last_clicked:
                clicked_x = last_clicked.get("x")
                clicked_y = last_clicked.get("y")
                if clicked_x is not None and clicked_y is not None:
                    # Get path from current position to last clicked tile
                    rect_clicked = (clicked_x - 1, clicked_x + 1, clicked_y - 1, clicked_y + 1)
                    path_to_clicked, _ = ipc.path(rect=rect_clicked, visualize=False)
                    
                    if path_to_clicked and len(path_to_clicked) > 0:
                        path_length = len(path_to_clicked)
                        is_running = movement_state.get("is_running", False)
                        
                        # Dynamic thresholds based on movement speed
                        if is_running:
                            # Running: click again when path is ≤8-10 waypoints (with variation)
                            threshold = random.randint(8, 10)
                        else:
                            # Walking: click again when path is ≤3-4 waypoints (with variation)
                            threshold = random.randint(3, 4)
                        
                        if path_length <= threshold:
                            # Close to last clicked tile - click ahead again
                            logging.debug(f"[GO_TO_JACOBIAN] Close to last clicked tile (path length: {path_length} ≤ {threshold}), clicking ahead again")
                            # Fall through to click logic below
                        else:
                            # Not close enough - check if path converges with destination
                            if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest):
                                # Paths converge - we're moving correctly, don't click
                                return True
            else:
                # No last clicked tile - check path convergence
                if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest):
                    # Paths converge - we're moving correctly, don't click
                    return True
        
        # STEP 5: Need to click - get waypoints (long-distance OR short-distance)
        # Check if destination is in scene using IPC path response
        use_long_distance = False
        target_waypoint = None
        
        # LOG: Before pathing decision
        logging.info(f"[GO_TO_JACOBIAN] Checking pathing method...")
        
        # Get test path to check if destination is in scene
        test_path, test_resp = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=False, max_wps=10000)
        
        # Extract goalInScene from debug response
        goal_in_scene = False
        if test_resp and test_resp.get("ok"):
            debug_info = test_resp.get("debug", {})
            goal_in_scene = debug_info.get("goalInScene", False)
        
        logging.info(f"[GO_TO_JACOBIAN] Destination in scene (from IPC): {goal_in_scene}")
        logging.info(f"[GO_TO_JACOBIAN] Estimated scene area: ({scene_min_x}, {scene_max_x}, {scene_min_y}, {scene_max_y})")
        
        # Use long-distance if destination is NOT in scene
        if not goal_in_scene:
            # Try long-distance waypoint batching
            waypoint_batch = _get_next_long_distance_waypoints(rect_key, rect, destination_method)
            waypoints_remaining = len(travel._long_distance_path_cache) - travel._long_distance_waypoint_index if travel._long_distance_path_cache else 0
            
            if waypoints_remaining > 0 and waypoint_batch:
                use_long_distance = True
                target_waypoint = waypoint_batch[0]
                logging.info(f"[GO_TO_JACOBIAN] Pathing method: LONG DISTANCE ({waypoints_remaining} waypoints remaining)")
                logging.info(f"[GO_TO_JACOBIAN] Target waypoint: {target_waypoint}")
            else:
                logging.info(f"[GO_TO_JACOBIAN] Pathing method: SHORT DISTANCE (long-distance path not available, falling back)")
        else:
            logging.info(f"[GO_TO_JACOBIAN] Pathing method: SHORT DISTANCE (destination in scene)")
        
        if use_long_distance and target_waypoint:
            # LONG DISTANCE: Get path to waypoint (not final destination)
            wps, _ = ipc.path(rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1), visualize=True)
            if not wps:
                # Fallback to final destination path
                wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
            
            # For long distance (>20 tiles), randomly select from last 5 waypoints in path
            if wps and len(wps) > 0:
                click_target_distance = _calculate_distance(player_x, player_y, target_waypoint[0], target_waypoint[1])
                if click_target_distance > 20:
                    # Select random waypoint from last 5 in path
                    last_5_count = min(5, len(wps))
                    if last_5_count > 0:
                        last_5_wps = wps[-last_5_count:]  # Get last 5 waypoints
                        selected_wp = random.choice(last_5_wps)
                        click_target_x = selected_wp.get("x")
                        click_target_y = selected_wp.get("y")
                        logging.info(f"[GO_TO_JACOBIAN] Long distance: randomly selected waypoint from last {last_5_count}: ({click_target_x}, {click_target_y})")
                    else:
                        # Fallback to original waypoint
                        click_target_x = target_waypoint[0]
                        click_target_y = target_waypoint[1]
                else:
                    # Not long distance, use original waypoint
                    click_target_x = target_waypoint[0]
                    click_target_y = target_waypoint[1]
            else:
                # No waypoints, use original waypoint
                click_target_x = target_waypoint[0]
                click_target_y = target_waypoint[1]
        else:
            # SHORT DISTANCE: Get path to final destination
            logging.info(f"[GO_TO_JACOBIAN] Using short distance path to final destination")
            if is_single_tile:
                wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
            else:
                wps, _ = ipc.path(rect=tuple(rect), visualize=True)
            # Click target is the FINAL DESTINATION
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
        
        # STEP 7: Calculate where to click (toward waypoint OR final destination)
        # Calculate distance to click target (waypoint or final destination)
        click_target_distance = _calculate_distance(player_x, player_y, click_target_x, click_target_y)
        
        # LOG: Click target
        logging.info(f"[GO_TO_JACOBIAN] Click target tile: ({click_target_x}, {click_target_y})")
        logging.info(f"[GO_TO_JACOBIAN] Distance to click target: {click_target_distance} tiles")
        
        # Use distance-based precision (calculated automatically in _calculate_click_location)
        click_location = _calculate_click_location(player_x, player_y, click_target_x, click_target_y, is_precise=None)
        
        # LOG: Calculated click location
        logging.info(f"[GO_TO_JACOBIAN] Calculated click location: ({click_location['x']}, {click_location['y']})")
        
        # STEP 8: Compensate for movement - adjust click position opposite to movement direction
        if movement_state and movement_state.get("is_moving"):
            movement_dir = movement_state.get("movement_direction")
            is_running = movement_state.get("is_running", False)
            
            if movement_dir:
                dx = movement_dir.get("dx", 0)
                dy = movement_dir.get("dy", 0)
                
                # Calculate compensation based on movement speed
                if is_running:
                    compensation = 0.4
                else:
                    compensation = 0.2
                
                # Adjust click position opposite to movement direction
                if dx != 0:
                    click_location["x"] = int(click_location["x"] - (dx / abs(dx)) * compensation)
                if dy != 0:
                    click_location["y"] = int(click_location["y"] - (dy / abs(dy)) * compensation)
        
        # Get movement state for camera movement compensation
        movement_state = _get_player_movement_state()
        
        # LOG: Before camera movement - get camera state
        camera_data_before = ipc.get_camera()
        if camera_data_before:
            yaw_before = camera_data_before.get("yaw", 0)
            pitch_before = camera_data_before.get("pitch", 0)
            zoom_before = camera_data_before.get("zoom", 0)
            logging.info(f"[GO_TO_JACOBIAN] Camera state BEFORE: yaw={yaw_before}, pitch={pitch_before}, zoom={zoom_before}")
        else:
            logging.warning(f"[GO_TO_JACOBIAN] Could not get camera state before movement")
        
        # LOG: Camera target (the tile the camera will point at)
        # The camera targets the click_location tile (not the click target or final destination)
        camera_target_x = click_location["x"]
        camera_target_y = click_location["y"]
        logging.info(f"[GO_TO_JACOBIAN] Camera target tile: ({camera_target_x}, {camera_target_y})")
        
        # STEP 9: Click at calculated location with Jacobian-based camera
        logging.info(f"[GO_TO_JACOBIAN] Executing click with Jacobian camera...")
        result = click_ground_with_camera_jacobian(
            world_coords=click_location,
            description=f"Move toward {rect_key if rect_key else 'destination'}",
            path_waypoints=wps if wps else None,
            movement_state=movement_state,
            calibration_data_path=calibration_data_path,
            target_screen_position=target_screen_position,
            aim_ms=aim_ms,
            verify_tile=False,  # Will verify below
            expected_tile=None,
            verify_path=False
        )
        
        if not result:
            logging.warning(f"[GO_TO_JACOBIAN] Click failed!")
            return None
        
        # LOG: After camera movement - get camera state
        time.sleep(0.05)  # 50ms delay for camera to move
        camera_data_after = ipc.get_camera()
        if camera_data_after:
            yaw_after = camera_data_after.get("yaw", 0)
            pitch_after = camera_data_after.get("pitch", 0)
            zoom_after = camera_data_after.get("zoom", 0)
            logging.info(f"[GO_TO_JACOBIAN] Camera state AFTER: yaw={yaw_after}, pitch={pitch_after}, zoom={zoom_after}")
            if camera_data_before:
                yaw_delta = yaw_after - yaw_before
                pitch_delta = pitch_after - pitch_before
                zoom_delta = zoom_after - zoom_before
                logging.info(f"[GO_TO_JACOBIAN] Camera movement deltas: yaw={yaw_delta:+.1f}, pitch={pitch_delta:+.1f}, zoom={zoom_delta:+.1f}")
        else:
            logging.warning(f"[GO_TO_JACOBIAN] Could not get camera state after movement")
        
        logging.info(f"[GO_TO_JACOBIAN] ===== COMPLETED go_to_jacobian() =====")
        
        # STEP 10: Get the ACTUAL selected tile from the game state
        selected_tile = ipc.get_selected_tile()
        
        if not selected_tile:
            # Can't verify clicked tile - still try to verify path if possible
            clicked_path, _ = ipc.path(rect=tuple(rect), visualize=False)
            if clicked_path:
                _movement_state["current_path"] = clicked_path
                _movement_state["is_moving"] = True
                _movement_state["last_movement_check_ts"] = time.time()
            return result
        
        # STEP 11: Verify the clicked tile generates a valid path
        if _verify_click_path(selected_tile, final_dest, wps):
            # Get the actual path from clicked tile to destination
            clicked_path, _ = ipc.path(rect=tuple(rect), visualize=False)
            if clicked_path:
                # Store state for monitoring
                _movement_state["last_clicked_tile"] = selected_tile
                _movement_state["current_path"] = clicked_path
                _movement_state["is_moving"] = True
                _movement_state["last_movement_check_ts"] = time.time()
                return result
            else:
                # Couldn't get path, but click succeeded
                return result
        else:
            # Path verification failed, but click succeeded - might still work
            logging.warning(f"[GO_TO_JACOBIAN] Path verification failed for click at ({selected_tile.get('x')}, {selected_tile.get('y')})")
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

