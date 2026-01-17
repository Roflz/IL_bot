"""
Jacobian-based camera system for travel/movement.

This module provides camera positioning for ground clicks and path following
using the Jacobian-based system. It positions path look-ahead points at
specific screen coordinates for optimal viewing.
"""

import random
import logging
import time

from helpers.runtime_utils import ipc
from helpers.utils import sleep_exponential
from services.camera_jacobian import (
    calculate_camera_movement_to_screen_position,
    execute_jacobian_camera_movement,
    get_screen_position_preset
)


def calculate_path_lookahead_for_camera(
    path_waypoints: list,
    player_x: int,
    player_y: int,
    final_dest_x: int,
    final_dest_y: int,
    movement_state: dict = None
) -> dict:
    """
    Calculate optimal look-ahead point for camera positioning along the path.
    
    This finds a point ahead of the player along the path waypoints to position
    the camera for better visibility during travel.
    
    Args:
        path_waypoints: List of waypoint dicts from pathfinding (path from player to click location)
        player_x, player_y: Current player position
        final_dest_x, final_dest_y: Click location (where we're clicking, not final destination)
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
        
    Returns:
        {
            "x": int,
            "y": int,
            "distance_ahead": int,  # How many tiles ahead
            "path_distance": float  # Total path distance
        }
    """
    if not path_waypoints or len(path_waypoints) == 0:
        # No path - just aim at click location
        total_distance = abs(final_dest_x - player_x) + abs(final_dest_y - player_y)
        return {
            "x": final_dest_x,
            "y": final_dest_y,
            "distance_ahead": 0,
            "path_distance": total_distance
        }
    
    # Calculate total path distance (Manhattan) from player to click location
    total_distance = abs(final_dest_x - player_x) + abs(final_dest_y - player_y)
    
    # Determine look-ahead distance based on total distance
    if total_distance > 20:
        # Long distance: look 5-10 tiles ahead (with variation)
        look_ahead_tiles = random.randint(5, 10)
    elif total_distance > 5:
        # Medium distance: look 2-5 tiles ahead (with variation)
        look_ahead_tiles = random.randint(2, 5)
    else:
        # Short distance: look at destination or 1-2 tiles ahead
        look_ahead_tiles = random.randint(0, 2)
    
    # Adjust for movement speed if available
    if movement_state:
        is_running = movement_state.get("is_running", False)
        if is_running:
            # Running: look further ahead
            look_ahead_tiles = int(look_ahead_tiles * 1.3)
    
    # Find waypoint that's approximately look_ahead_tiles along the path
    accumulated_distance = 0
    target_distance = look_ahead_tiles
    
    for i, wp in enumerate(path_waypoints):
        # Extract waypoint coordinates
        wp_x = wp.get("x") or wp.get("world", {}).get("x")
        wp_y = wp.get("y") or wp.get("world", {}).get("y")
        
        if wp_x is None or wp_y is None:
            continue
        
        # Calculate distance from player to this waypoint
        if i == 0:
            # First waypoint - distance from player
            dist = abs(wp_x - player_x) + abs(wp_y - player_y)
        else:
            # Distance from previous waypoint
            prev_wp = path_waypoints[i - 1]
            prev_x = prev_wp.get("x") or prev_wp.get("world", {}).get("x")
            prev_y = prev_wp.get("y") or prev_wp.get("world", {}).get("y")
            if prev_x is None or prev_y is None:
                continue
            dist = abs(wp_x - prev_x) + abs(wp_y - prev_y)
        
        accumulated_distance += dist
        
        # If we've reached our target distance, use this waypoint
        if accumulated_distance >= target_distance:
            # Apply movement direction compensation if available
            if movement_state and movement_state.get("movement_direction"):
                movement_dir = movement_state.get("movement_direction")
                dx = movement_dir.get("dx", 0)
                dy = movement_dir.get("dy", 0)
                
                # Look slightly ahead in movement direction (0.5-1 tile)
                if dx != 0 or dy != 0:
                    compensation = random.uniform(0.5, 1.0)
                    wp_x = int(wp_x + (dx / max(abs(dx), abs(dy))) * compensation)
                    wp_y = int(wp_y + (dy / max(abs(dx), abs(dy))) * compensation)
            
            return {
                "x": wp_x,
                "y": wp_y,
                "distance_ahead": look_ahead_tiles,
                "path_distance": total_distance
            }
    
    # If we didn't find a waypoint within look-ahead distance, use final destination
    return {
        "x": final_dest_x,
        "y": final_dest_y,
        "distance_ahead": 0,
        "path_distance": total_distance
    }


def click_ground_with_camera_jacobian(
    world_coords: dict,
    description: str = "Move",
    path_waypoints: list = None,
    movement_state: dict = None,
    calibration_data_path: str = None,
    target_screen_position: str = "center_top",  # Preset name
    aim_ms: int = 700,
    verify_tile: bool = False,
    expected_tile: dict = None,
    verify_path: bool = False
) -> dict | None:
    """
    Click ground with Jacobian-based camera positioning.
    
    This is the NEW version that uses Jacobian system.
    Old version: services/click_with_camera.py::click_ground_with_camera()
    
    Flow:
    1. Calculate path look-ahead point (if path_waypoints provided)
    2. Use Jacobian to position look-ahead point at target screen position
    3. Wait for camera movement
    4. Project click tile to screen
    5. Click with shift+left-click
    6. Verify clicked tile (if verify_tile=True)
    
    Args:
        world_coords: {"x": int, "y": int} - World coordinates to click
        description: Description for logging
        path_waypoints: Optional list of waypoint dicts from pathfinding
        movement_state: Optional movement state dict with is_running, orientation, etc.
        calibration_data_path: Path to calibration JSONL file (required)
        target_screen_position: Preset name for target screen position (default: "center_top")
        aim_ms: Maximum time for camera aiming (not used with Jacobian, but kept for compatibility)
        verify_tile: Whether to verify the clicked tile matches expected
        expected_tile: Expected tile coordinates for verification
        verify_path: Whether to verify clicked tile is on path
    
    Returns:
        Click result dict or None on failure
    """
    try:
        if not world_coords:
            return None
        
        from actions import player
        
        # Get player position
        player_x = player.get_x()
        player_y = player.get_y()
        
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return None
        
        click_x = world_coords.get('x')
        click_y = world_coords.get('y')
        
        if not isinstance(click_x, int) or not isinstance(click_y, int):
            return None
        
        # Get calibration data path (required)
        if not calibration_data_path:
            logging.error("calibration_data_path is required for Jacobian camera system")
            return None
        
        # Get movement state if not provided
        if movement_state is None:
            try:
                resp = ipc.get_player()
                if resp and resp.get("ok") and resp.get("player"):
                    player_data = resp.get("player", {})
                    is_running = player_data.get("isRunning", False)
                    orientation = player_data.get("orientation", 0)
                    movement_state = {
                        "is_running": is_running,
                        "orientation": orientation
                    }
            except Exception:
                movement_state = None
        
        # Get screen dimensions
        where = ipc.where() or {}
        screen_width = int(where.get("w", 0))
        screen_height = int(where.get("h", 0))
        if screen_width == 0 or screen_height == 0:
            logging.error("Could not get screen dimensions")
            return None
        
        # Import color codes and formatting functions
        from services.camera_jacobian import (
            COLOR_RESET, COLOR_BOLD, COLOR_CYAN, COLOR_BLUE, COLOR_RED,
            COLOR_GREEN, COLOR_YELLOW, COLOR_MAGENTA,
            _format_table_row, _print_table_header, _print_table_separator,
            _get_error_color
        )
        
        # Camera targets THE CLICK TILE - nothing else, no look-ahead, no path calculations
        camera_target = {"x": click_x, "y": click_y}
        
        # Get target screen position preset
        target_screen = get_screen_position_preset(target_screen_position, screen_width, screen_height)
        
        # Print formatted header
        _print_table_header("CLICK GROUND WITH CAMERA JACOBIAN")
        print(_format_table_row("Click Tile", f"({click_x}, {click_y})"))
        print(_format_table_row("Player Position", f"({player_x}, {player_y})"))
        print(_format_table_row("Camera Target", f"({camera_target['x']}, {camera_target['y']}) {COLOR_RED}[THE CLICK TILE]{COLOR_RESET}"))
        print(_format_table_row("Target Screen", f"{target_screen_position} -> ({target_screen['x']}, {target_screen['y']})"))
        print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}\n")
        
        movement_result = calculate_camera_movement_to_screen_position(
            object_world_coords=camera_target,  # THE CLICK TILE
            target_screen_coords=target_screen,
            calibration_data_path=calibration_data_path,
            use_global_model=True,
            use_zoom=False,
            step_scale=1.0,  # No damping - move full distance
            max_yaw_step=9999.0,  # No clamping - allow full yaw movement
            max_pitch_step=9999.0,  # No clamping - allow full pitch movement
            pitch_min=280,  # Minimum pitch for travel (keep camera high)
            jacobian_method="finite_diff",  # Use finite-difference method (default)
            finite_diff_dyaw=0.0,  # Auto-adaptive based on error
            finite_diff_dpitch=0.0  # Auto-adaptive based on error
        )
        
        if not movement_result.get("success"):
            logging.warning(f"[TRAVEL_CAMERA] Camera calculation failed: {movement_result.get('message')}")
            # Continue anyway - might still be able to click
        
        # Project target tile BEFORE camera movement
        proj_before, _ = ipc.project_many([{"x": camera_target['x'], "y": camera_target['y']}])
        screen_before = None
        screen_before_x = None
        screen_before_y = None
        if proj_before and len(proj_before) > 0:
            screen_before = proj_before[0].get("canvas")
            if screen_before:
                screen_before_x = screen_before.get("x")
                screen_before_y = screen_before.get("y")
        
        # Execute camera movement
        if movement_result.get("success"):
            execute_jacobian_camera_movement(movement_result, wait_for_stable=True, max_wait_time=3.0)
        
        # Project target tile AFTER camera movement
        proj_after, _ = ipc.project_many([{"x": camera_target['x'], "y": camera_target['y']}])
        screen_after = None
        screen_after_x = None
        screen_after_y = None
        if proj_after and len(proj_after) > 0:
            screen_after = proj_after[0].get("canvas")
            if screen_after:
                screen_after_x = screen_after.get("x")
                screen_after_y = screen_after.get("y")
        
        # Print formatted before/after comparison
        target_x = target_screen.get("x")
        target_y = target_screen.get("y")
        
        _print_table_header("SCREEN POSITION COMPARISON")
        print(_format_table_row("Target Tile", f"({camera_target['x']}, {camera_target['y']})"))
        _print_table_separator()
        
        if screen_before_x is not None:
            print(_format_table_row("BEFORE Screen", f"({screen_before_x}, {screen_before_y})"))
        else:
            print(_format_table_row("BEFORE Screen", "Off-screen", COLOR_RED))
        
        if screen_after_x is not None:
            print(_format_table_row("AFTER Screen", f"({screen_after_x}, {screen_after_y})"))
        else:
            print(_format_table_row("AFTER Screen", "Off-screen", COLOR_RED))
        
        if target_x is not None and target_y is not None:
            print(_format_table_row("Target Screen", f"({target_x}, {target_y})"))
            
            if screen_after_x is not None:
                error_x = screen_after_x - target_x
                error_y = screen_after_y - target_y
                error_distance = (error_x ** 2 + error_y ** 2) ** 0.5
                error_color = _get_error_color(error_distance)
                
                _print_table_separator()
                print(_format_table_row("Error X", f"{error_x:+.1f} px", error_color))
                print(_format_table_row("Error Y", f"{error_y:+.1f} px", error_color))
                print(_format_table_row("Error Distance", f"{error_distance:.2f} px", error_color))
        
        print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}\n")
        
        # Project the click tile to get screen coordinates (for compatibility with existing code)
        logging.info(f"[TRAVEL_CAMERA] Projecting click tile ({click_x}, {click_y}) to screen...")
        proj, _ = ipc.project_many([{"x": click_x, "y": click_y}])
        
        if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
            logging.warning(f"[TRAVEL_CAMERA] Could not project click tile to screen")
            return None
        
        proj_data = proj[0]
        bounds = proj_data.get("bounds", {})
        
        # Get click coordinates with some randomization
        if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
            # Use bounds with randomization
            base_x = bounds.get("x", 0) + bounds.get("width", 0) // 2
            base_y = bounds.get("y", 0) + bounds.get("height", 0) // 2
            cx = base_x + random.randint(-bounds.get("width", 0) // 4, bounds.get("width", 0) // 4)
            cy = base_y + random.randint(-bounds.get("height", 0) // 4, bounds.get("height", 0) // 4)
        else:
            # Fallback to canvas coordinates with randomization
            fresh_coords = proj_data["canvas"]
            base_x = int(fresh_coords["x"])
            base_y = int(fresh_coords["y"])
            cx = base_x + random.randint(-5, 5)
            cy = base_y + random.randint(-5, 5)
        
        # Canvas dimensions
        canvas_width = screen_width
        canvas_height = screen_height
        
        # Check if target is off-screen and adjust click to canvas edge in that direction
        if base_x < 0 or base_x >= canvas_width or base_y < 0 or base_y >= canvas_height:
            # Target is off-screen - click at canvas edge in the direction of the target
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # Calculate direction vector from canvas center to target
            dx = base_x - canvas_center_x
            dy = base_y - canvas_center_y
            
            # Calculate which edge we'll hit first
            edge_distance_from_center = random.uniform(30, 80)  # Human-like: 30-80px from edge
            
            if base_y < 0:
                # Target is above canvas - click at top edge
                cy = int(edge_distance_from_center)
                if abs(dx) > 0:
                    edge_x = canvas_center_x + int(dx / abs(dx) * min(abs(dx), canvas_width // 2 - 50))
                    cx = edge_x + random.randint(-30, 30)
                else:
                    cx = canvas_center_x + random.randint(-50, 50)
                cx = max(50, min(canvas_width - 50, cx))
            elif base_y >= canvas_height:
                # Target is below canvas - click at bottom edge
                cy = int(canvas_height - edge_distance_from_center)
                if abs(dx) > 0:
                    edge_x = canvas_center_x + int(dx / abs(dx) * min(abs(dx), canvas_width // 2 - 50))
                    cx = edge_x + random.randint(-30, 30)
                else:
                    cx = canvas_center_x + random.randint(-50, 50)
                cx = max(50, min(canvas_width - 50, cx))
            elif base_x < 0:
                # Target is left of canvas - click at left edge
                cx = int(edge_distance_from_center)
                if abs(dy) > 0:
                    edge_y = canvas_center_y + int(dy / abs(dy) * min(abs(dy), canvas_height // 2 - 50))
                    cy = edge_y + random.randint(-30, 30)
                else:
                    cy = canvas_center_y + random.randint(-50, 50)
                cy = max(50, min(canvas_height - 50, cy))
            elif base_x >= canvas_width:
                # Target is right of canvas - click at right edge
                cx = int(canvas_width - edge_distance_from_center)
                if abs(dy) > 0:
                    edge_y = canvas_center_y + int(dy / abs(dy) * min(abs(dy), canvas_height // 2 - 50))
                    cy = edge_y + random.randint(-30, 30)
                else:
                    cy = canvas_center_y + random.randint(-50, 50)
                cy = max(50, min(canvas_height - 50, cy))
            
            logging.debug(f"[TRAVEL_CAMERA] Target off-screen (base: {base_x}, {base_y}), clicking at canvas edge ({cx}, {cy})")
        else:
            # Target is on-screen - use coordinates with minor clamping for safety
            cx = base_x + random.randint(-3, 3)
            cy = base_y + random.randint(-3, 3)
            
            # Clamp to canvas bounds with safety margin
            margin = random.randint(10, 20)  # Human-like: 10-20px margin from edge
            cx = max(margin, min(canvas_width - margin, cx))
            cy = max(margin, min(canvas_height - margin, cy))
        
        # Check if click point is within any blocking UI widget
        from helpers.widgets import is_point_in_blocking_widget
        if is_point_in_blocking_widget(cx, cy):
            # Click point is blocked by UI - adjust to nearest point outside widget bounds
            adjusted = False
            for offset_x, offset_y in [
                (0, -50), (0, 50), (-50, 0), (50, 0),  # Up, down, left, right
                (-30, -30), (30, -30), (-30, 30), (30, 30),  # Diagonals
                (0, -100), (0, 100), (-100, 0), (100, 0),  # Further in cardinal directions
            ]:
                test_x = cx + offset_x
                test_y = cy + offset_y
                
                # Ensure test point is still within canvas bounds
                if (10 <= test_x <= canvas_width - 10 and 
                    10 <= test_y <= canvas_height - 10 and
                    not is_point_in_blocking_widget(test_x, test_y)):
                    cx = test_x
                    cy = test_y
                    adjusted = True
                    logging.debug(f"[TRAVEL_CAMERA] Click point blocked by UI widget, adjusted to ({cx}, {cy})")
                    break
            
            if not adjusted:
                logging.warning(f"[TRAVEL_CAMERA] Click point ({cx}, {cy}) is blocked by UI widget and couldn't be adjusted")
        
        logging.info(f"[TRAVEL_CAMERA] Final click coordinates: ({cx}, {cy})")
        logging.info(f"[TRAVEL_CAMERA] Clicking at screen position ({cx}, {cy}) for world tile ({click_x}, {click_y})")
        
        sleep_exponential(0.05, 0.15, 1.5)
        
        # Press shift before clicking (RuneLite plugin makes "Walk here" top action when shift is held)
        try:
            ipc.key_press("SHIFT")
        except Exception:
            pass  # Don't fail if key press fails
        
        # Perform a simple left click (shift is held, so "Walk here" will be the action)
        result = ipc.click(cx, cy, button=1)
        logging.info(f"[TRAVEL_CAMERA] Click executed, result: {result}")
        
        # Release shift after clicking
        try:
            ipc.key_release("SHIFT")
        except Exception:
            pass  # Don't fail if key release fails
        
        if not result:
            return None
        
        # Wait a short delay for the game to process the click and update selected tile
        time.sleep(0.05)  # 50ms delay
        
        # Get the ACTUAL selected tile from the game state (not from IPC response)
        selected_tile = ipc.get_selected_tile()
        
        # Check if the correct interaction was performed
        from helpers.ipc import get_last_interaction
        last_interaction = get_last_interaction()
        if not last_interaction or last_interaction.get("action") != "Walk here":
            logging.warning(f"[TRAVEL_CAMERA] Incorrect interaction: {last_interaction}")
            return None
        
        # Verify the clicked tile matches the intended target (if requested)
        if verify_tile and expected_tile:
            if selected_tile:
                actual_x = selected_tile.get("x")
                actual_y = selected_tile.get("y")
                intended_x = expected_tile.get("x", click_x)
                intended_y = expected_tile.get("y", click_y)
                
                if actual_x is not None and actual_y is not None:
                    # Calculate Manhattan distance between intended and actual tiles
                    distance = abs(actual_x - intended_x) + abs(actual_y - intended_y)
                    
                    # Determine tolerance based on distance from player
                    player_distance = abs(player_x - intended_x) + abs(player_y - intended_y)
                    if player_distance <= 5:
                        max_tolerance = 1  # Close range: exact or 1 tile
                    elif player_distance <= 20:
                        max_tolerance = 2  # Medium range: 1-2 tiles
                    else:
                        max_tolerance = 3  # Long range: 2-3 tiles
                    
                    if distance > max_tolerance:
                        logging.warning(f"[TRAVEL_CAMERA] Tile mismatch: intended ({intended_x}, {intended_y}), actual ({actual_x}, {actual_y}), distance: {distance}, tolerance: {max_tolerance}")
                        return None
        
        return result
        
    except Exception as e:
        logging.error(f"[TRAVEL_CAMERA] Error in click_ground_with_camera_jacobian: {e}")
        return None

