# ilbot/ui/simple_recorder/actions/travel.py
import random
import time
import logging

from . import player
from .player import get_player_position
from helpers.runtime_utils import ipc
from helpers.navigation import get_nav_rect, closest_bank_key, _merge_door_into_projection
from helpers.utils import sleep_exponential, get_random_walkable_tile, get_center_weighted_walkable_tile, exponential_number, normal_number
from services.click_with_camera import click_object_with_camera
from services.camera_integration import aim_midtop_at_world
from constants import BANK_REGIONS, REGIONS
from actions.movement import get_movement_direction, clear_movement_state

# Timing instrumentation constants
_GO_TO_TIMING_ENABLED = True

# Global variable to track the most recently traversed door
_most_recently_traversed_door = None

# Global variables for long-distance path caching
_long_distance_path_cache = None
_long_distance_destination = None
_long_distance_waypoint_index = 0

# Global variables for sophisticated movement tracking
_movement_state = {
    "last_clicked_tile": None,  # {"x": int, "y": int}
    "current_path": None,  # List of waypoints from last click to final destination
    "final_destination": None,  # {"x": int, "y": int} - For precise tile movement
    "intended_path": None,  # List of waypoints from player to final destination
    "is_moving": False,  # Whether we're currently on a valid path
    "last_movement_check_ts": None,  # Timestamp of last movement check
    "last_player_pos": None,  # {"x": int, "y": int} - Last known player position
    "last_player_animation": None,  # Last known animation ID
    "last_player_orientation": None,  # Last known orientation
    # Area movement fields
    "final_destination_area": None,  # {"min_x": int, "max_x": int, "min_y": int, "max_y": int}
    "final_destination_type": None,  # "tile" or "area"
    "area_target_tile": None,  # {"x": int, "y": int} - Calculated target within area
    "area_rect_key": None,  # str - Area key for destination selection
    # Path call throttling (to prevent lag when close to objects)
    "last_path_call_ts": None,  # Timestamp of last path call
    "cached_path_to_clicked": None,  # Cached path to last clicked tile
    "cached_path_to_clicked_ts": None,  # Timestamp when path was cached
    "cached_path_to_dest": None,  # Cached path to destination
    "cached_path_to_dest_ts": None,  # Timestamp when path was cached
}

def _calculate_path_distance_to_waypoint(waypoint_index: int, player_x: int, player_y: int) -> float:
    """
    Calculate the total path distance from player position to a waypoint.
    This is the sum of distances between consecutive waypoints in the path.
    """
    if waypoint_index < 0 or waypoint_index >= len(_long_distance_path_cache):
        return float('inf')
    
    total_distance = 0.0
    
    # Distance from player to first waypoint in the path segment
    first_waypoint = _long_distance_path_cache[waypoint_index]
    total_distance += _calculate_distance(player_x, player_y, first_waypoint[0], first_waypoint[1])
    
    # Add distances between consecutive waypoints from start to target
    for i in range(waypoint_index + 1, len(_long_distance_path_cache)):
        prev_waypoint = _long_distance_path_cache[i - 1]
        curr_waypoint = _long_distance_path_cache[i]
        total_distance += _calculate_distance(prev_waypoint[0], prev_waypoint[1], curr_waypoint[0], curr_waypoint[1])
    
    return total_distance


def _get_next_long_distance_waypoints(destination_key: str, custom_dest_rect: tuple = None, destination: str = "center") -> list[tuple[int, int]] | None:
    """
    Get the next waypoint from the cached long-distance path.
    Generate path ONCE at the start, then follow it step by step.
    Uses path distance (not straight-line distance) to select waypoints.
    
    Args:
        destination_key: Navigation key for destination
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
    """
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index

    
    # Generate path ONCE if we don't have one for this destination
    if (_long_distance_path_cache is None or 
        _long_distance_destination != destination_key):
        
        waypoints = get_long_distance_waypoints(destination_key, custom_dest_rect, destination)
        _long_distance_path_cache = waypoints
        _long_distance_destination = destination_key
        _long_distance_waypoint_index = 0
        if not waypoints:
            return None
        

    
    # Check if we've reached the end of the path
    if _long_distance_waypoint_index >= len(_long_distance_path_cache):
        print(f"[TRAVEL] Reached end of path")
        
        # Check if we're close to destination
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            min_x, max_x, min_y, max_y = custom_dest_rect
            dest_center_x = (min_x + max_x) // 2
            dest_center_y = (min_y + max_y) // 2
        else:
            dest_rect = get_nav_rect(destination_key)
            if dest_rect and len(dest_rect) == 4:
                min_x, max_x, min_y, max_y = dest_rect
                dest_center_x = (min_x + max_x) // 2
                dest_center_y = (min_y + max_y) // 2
            else:
                return None
        
        player_x, player_y = player.get_x(), player.get_y()
        if isinstance(player_x, int) and isinstance(player_y, int):
            distance_to_dest = _calculate_distance(player_x, player_y, dest_center_x, dest_center_y)
            if distance_to_dest <= 15:  # Close to destination
                print(f"[TRAVEL] Close to destination, path complete!")
                clear_long_distance_cache()
            else:
                print(f"[TRAVEL] Still far from destination, path may be incomplete")
        
        return None
    
    # Get player position
    player_x, player_y = player.get_x(), player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        # Fallback to next waypoint if we can't get player position
        current_waypoint = _long_distance_path_cache[_long_distance_waypoint_index]
        _long_distance_waypoint_index += 1
        return [current_waypoint]
    
    # Find the closest waypoint to player's current position
    # Only check waypoints from current index forward (cannot go backwards)
    # Consider all waypoints within 25 tile radius of player
    start_index = _long_distance_waypoint_index  # Cannot go behind current index
    end_index = len(_long_distance_path_cache)  # Check all remaining waypoints
    
    closest_distance = float('inf')
    closest_index = _long_distance_waypoint_index  # Default to current index
    radius_limit = 25  # 25 tile radius limit
    
    for i in range(start_index, end_index):
        waypoint = _long_distance_path_cache[i]
        distance = _calculate_distance(player_x, player_y, waypoint[0], waypoint[1])
        # Only consider waypoints within 25 tile radius
        if distance <= radius_limit and distance < closest_distance:
            closest_distance = distance
            closest_index = i
    
    # Update our cached position in the path to the closest waypoint
    _long_distance_waypoint_index = closest_index
    
    # Set target waypoint to the furthest waypoint ahead in the path that is within 25 tile radius
    target_index = closest_index  # Start from closest waypoint
    target_waypoint = _long_distance_path_cache[target_index]
    
    # Find the furthest waypoint from closest_index forward that is within 25 tiles
    for i in range(closest_index, len(_long_distance_path_cache)):
        waypoint = _long_distance_path_cache[i]
        distance = _calculate_distance(player_x, player_y, waypoint[0], waypoint[1])
        if distance <= radius_limit:
            target_index = i
            target_waypoint = waypoint
        else:
            # Once we exceed the radius, stop (waypoints are ordered, so we won't find more)
            break
    
    distance_to_target = _calculate_distance(player_x, player_y, target_waypoint[0], target_waypoint[1])
    
    # Progress indicator showing current and target waypoints
    print(f"[TRAVEL] Current waypoint: {closest_index + 1}/{len(_long_distance_path_cache)}, Target: {target_index + 1}/{len(_long_distance_path_cache)}")
    
    return [target_waypoint]


def clear_long_distance_cache():
    """Clear the long-distance path cache."""
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index
    _long_distance_path_cache = None
    _long_distance_destination = None
    _long_distance_waypoint_index = 0
    # Cache cleared

def _is_same_door(door_plan: dict) -> bool:
    """
    Check if the door_plan is the same as the most recently traversed door.
    """
    global _most_recently_traversed_door
    
    if not door_plan or not _most_recently_traversed_door:
        return False
    
    # Compare door IDs and coordinates
    current_door = door_plan.get("door", {})
    current_id = current_door.get("id")
    current_x = door_plan.get("x")
    current_y = door_plan.get("y")
    current_p = door_plan.get("p", 0)
    
    recent_door = _most_recently_traversed_door.get("door", {})
    recent_id = recent_door.get("id")
    recent_x = _most_recently_traversed_door.get("x")
    recent_y = _most_recently_traversed_door.get("y")
    recent_p = _most_recently_traversed_door.get("p", 0)
    
    return (current_id == recent_id and 
            current_x == recent_x and 
            current_y == recent_y and 
            current_p == recent_p)


def _is_door_blocking_path(door_plan: dict, path_waypoints: list) -> bool:
    """
    Check if a door is actually blocking the path based on its orientation and path direction.
    
    Args:
        door_plan: Door plan containing door information including orientation
        path_waypoints: List of waypoints in the path (FULL path from go_to)
        
    Returns:
        True if door is blocking the path, False if door can be passed through
    """
    if not door_plan or not path_waypoints or len(path_waypoints) < 2:
        return True  # Default to blocking if we can't determine
    
    door_x = door_plan.get("x")
    door_y = door_plan.get("y")
    door_orientation_a = door_plan['door'].get("orientationA", 0)
    door_orientation_b = door_plan['door'].get("orientationB", 0)
    
    if not isinstance(door_x, int) or not isinstance(door_y, int):
        return True  # Default to blocking if no valid coordinates
    
    # Find waypoints that cross the door tile
    for i in range(len(path_waypoints) - 1):
        current_wp = path_waypoints[i]
        next_wp = path_waypoints[i + 1]
        
        # Check if this path segment crosses the door tile
        if _path_segment_crosses_door_tile(current_wp, next_wp, door_x, door_y):
            # Determine which side(s) of the door tile the path crosses
            door_sides = _get_door_side_crossed(current_wp, next_wp, door_x, door_y)
            
            if door_sides is None:
                continue  # Path doesn't cross a door side
            
            # Check if the door has a wall on any of the sides being crossed
            if _door_has_wall_on_side(door_orientation_a, door_orientation_b, door_sides):
                return True

    return False

def _path_segment_crosses_door_tile(wp1: dict, wp2: dict, door_x: int, door_y: int) -> bool:
    """Check if a path segment crosses the door tile."""
    # Extract coordinates from waypoints (handle both regular waypoints and door objects)
    x1, y1 = _extract_waypoint_coords(wp1)
    x2, y2 = _extract_waypoint_coords(wp2)
    
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return False
    
    # Check if the line segment from (x1,y1) to (x2,y2) passes through tile (door_x, door_y)
    # This happens if:
    # 1. One waypoint is on the door tile and the other is adjacent
    # 2. The line segment passes through the door tile
    
    # Case 1: One waypoint is on the door tile
    if (x1 == door_x and y1 == door_y) or (x2 == door_x and y2 == door_y):
        return True
    
    # Case 2: Line segment passes through the door tile
    # Check if the door tile is between the two waypoints
    if min(x1, x2) < door_x < max(x1, x2) and min(y1, y2) < door_y < max(y1, y2):
        return True
    
    return False

def _extract_waypoint_coords(wp: dict) -> tuple:
    """Extract x,y coordinates from a waypoint (handles both regular waypoints and door objects)."""
    return wp.get('x'), wp.get('y')

def _get_door_side_crossed(wp1: dict, wp2: dict, door_x: int, door_y: int) -> str:
    """Determine which side of the door tile the path crosses."""
    # Extract coordinates from waypoints (handle both regular waypoints and door objects)
    x1, y1 = _extract_waypoint_coords(wp1)
    x2, y2 = _extract_waypoint_coords(wp2)
    
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    
    # Check which waypoint is on the door tile
    wp1_on_door = (x1 == door_x and y1 == door_y)
    wp2_on_door = (x2 == door_x and y2 == door_y)
    
    if not wp1_on_door and not wp2_on_door:
        return None  # Neither waypoint is on the door tile
    
    # Calculate movement direction
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine which side(s) of the door tile is being crossed
    # The side being crossed depends on whether we're entering or leaving the door tile
    sides = []
    
    if wp1_on_door and not wp2_on_door:
        # Leaving the door tile - the side we're crossing is the direction we're moving
        if dx < 0:  # Moving West from door tile
            sides.append("west")
        elif dx > 0:  # Moving East from door tile
            sides.append("east")
        
        if dy < 0:  # Moving South from door tile
            sides.append("south")
        elif dy > 0:  # Moving North from door tile
            sides.append("north")
    
    elif wp2_on_door and not wp1_on_door:
        # Entering the door tile - the side we're crossing is opposite to the direction we're moving
        if dx < 0:  # Moving West to door tile
            sides.append("east")  # Crossing the East side of the door tile
        elif dx > 0:  # Moving East to door tile
            sides.append("west")  # Crossing the West side of the door tile
        
        if dy < 0:  # Moving South to door tile
            sides.append("north")  # Crossing the North side of the door tile
        elif dy > 0:  # Moving North to door tile
            sides.append("south")  # Crossing the South side of the door tile
    
    elif wp1_on_door and wp2_on_door:
        # Both waypoints are on the door tile - this shouldn't happen in normal pathfinding
        # but if it does, we can't determine which side is being crossed
        return None
    
    # Return the sides being crossed (could be 1 for cardinal, 2 for diagonal)
    return sides if sides else None

def _door_has_wall_on_side(orient_a: int, orient_b: int, sides) -> bool:
    """Check if the door has a wall on any of the specified sides."""
    side_flags = {
        "north": 2,
        "east": 4,
        "south": 8,
        "west": 1
    }
    
    # Handle both single side and list of sides
    if isinstance(sides, str):
        sides = [sides]
    
    for side in sides:
        if side not in side_flags:
            continue
        
        flag = side_flags[side]
        if (orient_a & flag) != 0 or (orient_b & flag) != 0:
            return True  # Door has a wall on at least one of the sides being crossed
    
    return False  # Door has no walls on any of the sides being crossed


def _handle_door_opening(door_plan: dict, full_path_waypoints: list) -> bool:
    """
    Handle door opening logic with retry mechanism and recently traversed door tracking.
    
    Args:
        door_plan: Door plan containing door information
        full_path_waypoints: Full path waypoints to check if door is actually blocking
        
    Returns:
        True if door was opened successfully or skipped, False if door opening failed
    """
    global _most_recently_traversed_door
    
    if not door_plan:
        return True  # No door to handle
    
    # Check if this is the same door we just traversed
    if _is_same_door(door_plan):
        # Skipping recently traversed door
        # Reset the recently traversed door so we don't skip it forever
        _most_recently_traversed_door = None
        return True  # Continue to normal ground move without door handling
    
    # Get initial door coordinates for pathfinding
    door_x = door_plan.get("x")
    door_y = door_plan.get("y")
    door_p = door_plan.get("p", 0)
    
    if not isinstance(door_x, int) or not isinstance(door_y, int):
        return True  # Invalid door coordinates
    
    # Check if the door is actually blocking the path before trying to open it
    if full_path_waypoints and not _is_door_blocking_path(door_plan, full_path_waypoints):
        _most_recently_traversed_door = door_plan
        return True  # Door is not blocking, continue with normal movement
    
    # Door is closed, try to open it with retry logic
    door_opened = not door_plan["door"].get("closed")
    door_retry_count = 0
    max_door_retries = 3

    while not door_opened and door_retry_count < max_door_retries:
        # Get fresh door data for each retry attempt
        wps, dbg_path = ipc.path(goal=(full_path_waypoints[-1].get('x'), full_path_waypoints[-1].get('y')), visualize=False)
        fresh_door_plan = _first_blocking_door_from_waypoints(wps)
        
        if not fresh_door_plan:
            # No door found, it might be open already
            door_opened = True
            break
            
        # Use fresh door data for clicking
        d = (fresh_door_plan.get("door") or {})
        door_name = d.get("name") or "Door"
        
        # Get world coordinates for the door
        world_coords = {}
        world_coords['x'] = door_plan.get("x", {})
        world_coords['y'] = door_plan.get("y", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            # No valid world coordinates for door, skipping
            break
        
        # Use click_object_with_camera to open the door
        door_result = click_object_with_camera(
            object_name=door_name,
            action="Open",  # Default door action
            world_coords=world_coords,
            door_plan=fresh_door_plan,  # Pass the door plan for specific door handling
            aim_ms=420
        )
        if door_result:
            # Wait up to 5 seconds for door to open
            door_wait_start = time.time()
            while (time.time() - door_wait_start) * 1000 < 5000:
                if _door_is_open(fresh_door_plan):
                    # Door opened successfully, mark it as recently traversed
                    _most_recently_traversed_door = fresh_door_plan
                    # Door opened successfully
                    door_opened = True
                    break
                sleep_exponential(0.05, 0.15, 1.5)

        if not door_opened:
            door_retry_count += 1
            if door_retry_count < max_door_retries:
                # Door opening attempt failed, retrying...
                sleep_exponential(0.3, 0.8, 1.2)

    if not door_opened:
        # Door opening failed after retries
        print(f"[TRAVEL] Failed to open door after {max_door_retries} attempts")
        return False
    
    return True


def _door_is_open(door_plan: dict) -> bool:
    """
    Check if a door is open by using IPC to get current door state.
    Uses the door_state IPC command to check if the door still exists.
    """
    if not door_plan or not door_plan.get("door"):
        return True
    
    # World coordinates are at the top level of door_plan, not inside door
    door_x = door_plan.get("x")
    door_y = door_plan.get("y")
    door_p = door_plan.get("p", 0)  # Default to plane 0 if not specified
    
    if not isinstance(door_x, int) or not isinstance(door_y, int):
        return True
    
    # Use IPC to get current door state
    resp = ipc._send({
        "cmd": "door_state",
        "door_x": door_x,
        "door_y": door_y,
        "door_p": door_p
    })
    
    if not resp or not resp.get("ok"):
        return True  # If IPC fails, assume door is open
    
    # Check if wall_object exists in the response
    if 'wall_object' not in resp or resp['wall_object'] is None:
        return True  # No wall_object means door doesn't exist (is open)
    
    # If wall_object exists, door is still there (is closed)
    og_door_id = door_plan['door'].get('id')
    new_door_id = resp['wall_object'].get('id')
    if not og_door_id == new_door_id:
        return True  # Door ID changed, it's open
    
    return False  # Door still exists with same ID, it's closed


def go_to(destination: str | tuple | list | dict, center: bool = False, destination_method: str = "center", arrive_radius: int = 0) -> bool | dict | None:
    """
    Unified movement system for both tile-based and area-based navigation.
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
    
    Returns:
        - True if already at destination or if we're on a valid path
        - dict (click result) if a click was attempted
        - None on failure
    """
    global _movement_state
    
    try:
        # Toggle run if needed
        run_energy = player.get_run_energy()
        if run_energy is not None and run_energy > 2000 and not player.is_run_on():
            player.toggle_run()

        # Get player position
        player_x, player_y = player.get_x(), player.get_y()
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return None
        
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
        
        # STEP 2: Check if already at destination
        distance = _calculate_distance(player_x, player_y, target_x, target_y)
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
        
        # STEP 5: Check if already moving toward area using PATH COMPARISON
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
                            print(f"[GO_TO] Close to last clicked tile (path length: {path_length} ≤ {threshold}), clicking ahead again")
                            # Fall through to click logic below
                        else:
                            # Not close enough - check if path converges with destination
                            final_dest = {"x": target_x, "y": target_y}
                            if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest):
                                # Paths converge - we're moving correctly, don't click
                                return True
            else:
                # No last clicked tile - check path convergence
                final_dest = {"x": target_x, "y": target_y}
                if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest):
                    # Paths converge - we're moving correctly, don't click
                    return True
        
        # STEP 6: Need to click - get waypoints (long-distance OR short-distance)
        # Check if destination is outside scene (heuristic: try IPC path first, if fails or very far, use long-distance)
        use_long_distance = False
        target_waypoint = None
        waypoint_batch = None
        
        # Try IPC path first to check if destination is reachable
        test_path, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=False)
        
        # Use long-distance if: path fails OR distance is very large (>100 tiles)
        if not test_path or distance > 100:
            # Try long-distance waypoint batching
            waypoint_batch = _get_next_long_distance_waypoints(rect_key, rect, destination_method)
        
        waypoints_remaining = len(_long_distance_path_cache) - _long_distance_waypoint_index if _long_distance_path_cache else 0
        
        if waypoints_remaining > 0 and waypoint_batch:
            use_long_distance = True
            target_waypoint = waypoint_batch[0]
            logging.info(f"[GO_TO] Using long distance path ({waypoints_remaining} waypoints remaining), target waypoint: {target_waypoint}")
        
        if use_long_distance and target_waypoint:
            # LONG DISTANCE: Get path to waypoint (not final destination)
            wps, _ = ipc.path(rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1), visualize=True)
            if not wps:
                # Fallback to final destination path
                wps, _ = ipc.path(rect=(target_x - 1, target_x + 1, target_y - 1, target_y + 1), visualize=True)
            # Click target is the WAYPOINT, not final destination
            click_target_x = target_waypoint[0]
            click_target_y = target_waypoint[1]
        else:
            # SHORT DISTANCE: Get path to final destination
            logging.info(f"[GO_TO] Using short distance path to final destination")
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
        
        # STEP 7: Handle doors before clicking
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            if not _handle_door_opening(door_plan, wps):
                return None
            return True
        
        # STEP 8: Calculate where to click (toward waypoint OR final destination)
        # Calculate distance to click target (waypoint or final destination)
        click_target_distance = _calculate_distance(player_x, player_y, click_target_x, click_target_y)
        # Use distance-based precision (calculated automatically in _calculate_click_location)
        click_location = _calculate_click_location(player_x, player_y, click_target_x, click_target_y, is_precise=None)
        
        # STEP 9: Compensate for movement - adjust click position opposite to movement direction
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
        
        # STEP 10: Click at calculated location with old camera logic
        result = _click_ground_old_camera(
            world_coords=click_location,
            aim_ms=700,
            path_waypoints=wps if wps else None
        )
        
        if not result:
            return None
        
        # Wait a short delay for the game to process the click and update selected tile
        time.sleep(0.05)  # 50ms delay
        
        # STEP 11: Get the ACTUAL selected tile from the game state
        selected_tile = ipc.get_selected_tile()
        
        if not selected_tile:
            # Can't verify clicked tile - still try to verify path if possible
            clicked_path, _ = ipc.path(rect=tuple(rect), visualize=False)
            if clicked_path:
                _movement_state["current_path"] = clicked_path
                _movement_state["is_moving"] = True
                _movement_state["last_movement_check_ts"] = time.time()
            return result
        
        # STEP 12: Verify the clicked tile generates a valid path
        final_dest = {"x": target_x, "y": target_y}
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
            print(f"[GO_TO] Path verification failed for click at ({selected_tile.get('x')}, {selected_tile.get('y')})")
            return result
    
    except Exception as e:
        logging.error(f"[GO_TO] Error: {e}")
        return None


def go_to_tile(tile_x: int, tile_y: int, *, plane: int | None = None, arrive_radius: int = 0, aim_ms: int = 700) -> dict | bool | None:
    """
    DEPRECATED: This function is now a wrapper around unified go_to().
    Use go_to((tile_x, tile_y)) instead.
    
    This function will be removed in the future. All logic is now in go_to().
    """
    # Simply call unified go_to() with the tile
    return go_to((tile_x, tile_y), arrive_radius=arrive_radius)


def go_to_tile_precise(tile_x: int, tile_y: int, *, plane: int | None = None, arrive_radius: int = 0, aim_ms: int = 700, max_attempts: int = 5) -> dict | bool | None:
    """
    Precision walking to a specific tile using area-based clicking with path verification.
    
    This function:
    - Clicks in an area ahead in the path direction (not exact waypoints)
    - Verifies the clicked tile is on/near the intended path
    - Uses smoother, more human-like navigation
    
    Args:
        tile_x: World X of destination
        tile_y: World Y of destination
        plane: Optional plane. If None, uses current plane.
        arrive_radius: Consider "arrived" if within this many tiles
        aim_ms: Camera aim time
        max_attempts: Maximum number of precision attempts (currently unused, kept for compatibility)
        
    Returns:
        - True if already at destination
        - dict (click result) if successful
        - None on failure
    """
    if not isinstance(tile_x, int) or not isinstance(tile_y, int):
        return None
    
    px, py = player.get_x(), player.get_y()
    if isinstance(px, int) and isinstance(py, int):
        distance = _calculate_distance(px, py, tile_x, tile_y)
        if distance <= max(0, int(arrive_radius)):
            clear_movement_state()  # Clear movement state when arrived
            return True
    
    try:
        if plane is None:
            plane = player.get_plane()
    except Exception:
        plane = None
    
    # Get path from player to destination for visualization and verification
    wps, _ = ipc.path(rect=(tile_x - 1, tile_x + 1, tile_y - 1, tile_y + 1), visualize=True)
    
    if not wps:
        # Can't get path - fall back to regular pathfinding
        print(f"[PRECISION] Cannot get path to ({tile_x}, {tile_y}), using regular pathfinding")
        return go_to_tile(tile_x, tile_y, plane=plane, arrive_radius=arrive_radius, aim_ms=aim_ms)
    
    # Use area-based clicking with old camera logic
    world_coords = {"x": tile_x, "y": tile_y, "p": plane or 0}
    
    result = _click_ground_old_camera(
        world_coords=world_coords,
        aim_ms=aim_ms,
        path_waypoints=wps
    )
    
    return result
    
    # All attempts failed - fall back to regular pathfinding
    print(f"[PRECISION] All precision attempts failed, falling back to regular pathfinding")
    return go_to_tile(tile_x, tile_y, plane=plane, arrive_radius=arrive_radius, aim_ms=aim_ms)

def travel_to_bank(bank_area= "CLOSEST_BANK"):
    """Handle traveling to the bank."""
    # Check if we're near a bank in the destination area
    destination_area = bank_area
    from helpers.bank import near_any_bank
    if not near_any_bank(destination_area):
        if bank_area:
            go_to(bank_area)
        else:
            go_to_closest_bank()
        return False
    else:
        return True

def go_to_bank(
    bank_area: str = "CLOSEST_BANK",
    prefer: str | None = None,
    randomize_closest: int | None = None,
    prefer_no_camera: bool = False,
    max_travel_attempts: int = 50
) -> dict | bool | None:
    """
    Travel to a bank area and open it, stopping early if an interactable bank is detected.
    
    This function will:
    1. Check if bank is already open (returns early)
    2. Check if there's an interactable bank in the target area (regardless of player position)
    3. If found, opens it immediately without traveling
    4. Otherwise, travels toward the bank area, checking after each movement for interactable banks
    5. Opens the bank as soon as one becomes interactable
    
    Args:
        bank_area: Bank area key (e.g., "FALADOR_BANK", "CLOSEST_BANK")
        prefer: Preferred bank type ("bank booth", "banker", "bank chest", etc.) - passed to open_bank()
        randomize_closest: If set to an integer X, randomly selects between the X closest banks - passed to open_bank()
        prefer_no_camera: If True, attempts to open bank without moving camera - passed to open_bank()
        max_travel_attempts: Maximum number of go_to() calls before giving up (default: 50)
    
    Returns:
        - True if bank is already open
        - dict (result from open_bank()) if bank was opened successfully
        - None if travel/opening failed after max attempts
    """
    from actions import bank
    from helpers.bank import _has_interactable_bank_in_area
    
    # Early exit: bank already open
    if bank.is_open():
        return True
    
    # Handle CLOSEST_BANK
    if bank_area == "CLOSEST_BANK" or not bank_area:
        from helpers.navigation import closest_bank_key
        bank_area = closest_bank_key()
    
    # Check if we can interact with a bank in the target area right now
    if _has_interactable_bank_in_area(bank_area):
        result = bank.open_bank(
            prefer=prefer,
            randomize_closest=randomize_closest,
            prefer_no_camera=prefer_no_camera
        )
        if result and bank.is_open():
            return result
    
    # Travel loop: move toward bank area, checking for interactable banks after each movement
    for attempt in range(max_travel_attempts):
        # Travel one step toward the bank area
        travel_result = go_to(bank_area)
        
        # After each movement, check if we can now interact with a bank in the target area
        if _has_interactable_bank_in_area(bank_area):
            result = bank.open_bank(
                prefer=prefer,
                randomize_closest=randomize_closest,
                prefer_no_camera=prefer_no_camera
            )
            if result and bank.is_open():
                return result
        
        # If travel failed completely, break
        if travel_result is None:
            continue
        
        # Small delay between attempts
        from helpers.utils import sleep_exponential
        sleep_exponential(0.1, 0.3, 1.2)
    
    # Final check: try to open bank one more time
    if _has_interactable_bank_in_area(bank_area):
        result = bank.open_bank(
            prefer=prefer,
            randomize_closest=randomize_closest,
            prefer_no_camera=prefer_no_camera
        )
        if result and bank.is_open():
            return result
    
    # Failed to open bank after max attempts
    return None


def _click_ground_old_camera(world_coords: dict, aim_ms: int = 700, path_waypoints: list = None) -> dict | None:
    """
    Old-style ground click using simple aim_midtop_at_world camera logic.
    This is the original camera logic before jacobian was introduced.
    """
    try:
        if not world_coords:
            return None
        
        click_x = world_coords.get('x')
        click_y = world_coords.get('y')
        
        if not isinstance(click_x, int) or not isinstance(click_y, int):
            return None
        
        # Use old camera logic: simple aim_midtop_at_world
        aim_midtop_at_world(click_x, click_y, max_ms=aim_ms, path_waypoints=path_waypoints)
        
        # Project the click tile to get screen coordinates
        proj, _ = ipc.project_many([{"x": click_x, "y": click_y}])
        
        if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
            return None
        
        proj_data = proj[0]
        canvas_coords = proj_data.get("canvas", {})
        cx = int(canvas_coords.get("x", 0))
        cy = int(canvas_coords.get("y", 0))
        
        if cx == 0 and cy == 0:
            return None
        
        # Press shift before clicking (RuneLite plugin makes "Walk here" top action when shift is held)
        try:
            ipc.key_press("SHIFT")
        except Exception:
            pass
        
        # Perform a simple left click
        result = ipc.click(cx, cy, button=1)
        
        # Release shift after clicking
        try:
            ipc.key_release("SHIFT")
        except Exception:
            pass
        
        return result
    
    except Exception as e:
        logging.error(f"[CLICK_GROUND_OLD] Error: {e}")
        return None


def _calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


def _calculate_area_target_tile(rect: tuple, rect_key: str, destination: str = "center") -> dict:
    """
    Calculate target tile within area for movement.
    
    Args:
        rect: (min_x, max_x, min_y, max_y) tuple
        rect_key: Area key for destination selection methods
        destination: Destination selection method ("center", "random", "center_weighted")
        
    Returns:
        {"x": int, "y": int} - Target tile within area
    """
    min_x, max_x, min_y, max_y = rect
    
    if destination == "center":
        target_x = (min_x + max_x) // 2
        target_y = (min_y + max_y) // 2
        return {"x": target_x, "y": target_y}
    elif destination == "random":
        if rect_key != "custom":
            selected_tile = get_random_walkable_tile(rect_key)
            if selected_tile:
                return {"x": selected_tile[0], "y": selected_tile[1]}
        # Fallback to center
        target_x = (min_x + max_x) // 2
        target_y = (min_y + max_y) // 2
        return {"x": target_x, "y": target_y}
    elif destination == "center_weighted":
        if rect_key != "custom":
            selected_tile = get_center_weighted_walkable_tile(rect_key)
            if selected_tile:
                return {"x": selected_tile[0], "y": selected_tile[1]}
        # Fallback to center
        target_x = (min_x + max_x) // 2
        target_y = (min_y + max_y) // 2
        return {"x": target_x, "y": target_y}
    else:
        # Default to center
        target_x = (min_x + max_x) // 2
        target_y = (min_y + max_y) // 2
        return {"x": target_x, "y": target_y}


def _is_within_area(player_x: int, player_y: int, rect: tuple) -> bool:
    """
    Check if player is within area bounds.
    
    Args:
        player_x, player_y: Player position
        rect: (min_x, max_x, min_y, max_y) tuple
        
    Returns:
        True if player is within area bounds, False otherwise
    """
    min_x, max_x, min_y, max_y = rect
    return min_x <= player_x <= max_x and min_y <= player_y <= max_y


def _clear_movement_state():
    """Clear the movement state (call when destination changes or arrived)."""
    global _movement_state
    _movement_state = {
        "last_clicked_tile": None,
        "current_path": None,
        "final_destination": None,
        "intended_path": None,
        "is_moving": False,
        "last_movement_check_ts": None,
        "last_player_pos": None,
        "last_player_animation": None,
        "last_player_orientation": None,
        "final_destination_area": None,
        "final_destination_type": None,
        "area_target_tile": None,
        "area_rect_key": None,
    }


def _calculate_click_location(player_x: int, player_y: int, dest_x: int, dest_y: int, is_precise: bool = None) -> dict:
    """
    Calculate where to click based on distance. Clicks AHEAD OF THE DESTINATION TILE 
    in the direction we're going, not ahead of the player.
    
    Uses distance-based precision (is_precise is calculated from distance if not provided).
    
    Args:
        player_x, player_y: Player's current position
        dest_x, dest_y: Target destination (waypoint or final destination)
        is_precise: Optional override. If None, calculated from distance:
            - Long distance (>20 tiles): False (lax: 5-10 tiles ahead)
            - Medium distance (5-20 tiles): False (moderate: 1-3 tiles ahead)
            - Short distance (≤5 tiles): True (precise: exact tile)
        
    Returns:
        {"x": int, "y": int} - World coordinates to click (ahead of destination)
    """
    dx = dest_x - player_x
    dy = dest_y - player_y
    distance = abs(dx) + abs(dy)
    
    if distance == 0:
        return {"x": dest_x, "y": dest_y}
    
    # Calculate is_precise from distance if not provided
    if is_precise is None:
        if distance > 20:
            is_precise = False  # Long distance: lax clicking
        elif distance > 5:
            is_precise = False  # Medium distance: moderate clicking
        else:
            is_precise = True  # Short distance: precise clicking
    
    # Calculate click distance based on total distance (how far ahead of destination to click)
    if distance > 20:
        # Long distance: click 5-10 tiles ahead of destination
        click_distance = random.randint(5, 10)
    elif distance > 5:
        # Medium distance: click 1-3 tiles ahead of destination
        click_distance = random.randint(1, 3)
    else:
        # Within 5 tiles: precise clicking (exact tile)
        if is_precise:
            return {"x": dest_x, "y": dest_y}  # Click the destination tile itself
        else:
            click_distance = random.randint(2, 4)  # Click a few tiles ahead
    
    # Calculate direction vector from player to destination
    # Then extend that direction beyond the destination by click_distance
    # Normalize the direction vector and extend it beyond destination
    if distance == 0:
        return {"x": dest_x, "y": dest_y}
    
    # Normalize direction (Manhattan distance)
    # Calculate unit direction components
    if abs(dx) > abs(dy):
        # More horizontal - normalize by dx
        unit_dx = 1 if dx > 0 else -1
        unit_dy = int(dy / abs(dx)) if dx != 0 else 0
    else:
        # More vertical - normalize by dy
        unit_dx = int(dx / abs(dy)) if dy != 0 else 0
        unit_dy = 1 if dy > 0 else -1
    
    # Click click_distance tiles ahead of destination in the direction we're going
    click_x = dest_x + (unit_dx * click_distance)
    click_y = dest_y + (unit_dy * click_distance)
    
    # Enforce maximum distance of 25 tiles from player
    final_dx = click_x - player_x
    final_dy = click_y - player_y
    final_distance = abs(final_dx) + abs(final_dy)
    
    if final_distance > 25:
        # Clamp click location to be within 25 tiles of player
        # Calculate how much to reduce by
        excess = final_distance - 25
        
        # Reduce the click_distance to bring it within 25 tiles
        if abs(unit_dx) + abs(unit_dy) > 0:
            # Calculate reduction needed
            reduction = min(excess, click_distance)
            click_x = dest_x + (unit_dx * (click_distance - reduction))
            click_y = dest_y + (unit_dy * (click_distance - reduction))
            
            # If still too far, clamp to exactly 25 tiles from player
            final_dx = click_x - player_x
            final_dy = click_y - player_y
            final_distance = abs(final_dx) + abs(final_dy)
            
            if final_distance > 25:
                # Move click location back toward player to be exactly 25 tiles away
                if final_distance > 0:
                    scale = 25.0 / final_distance
                    click_x = player_x + int(final_dx * scale)
                    click_y = player_y + int(final_dy * scale)
    
    return {"x": click_x, "y": click_y}


def _verify_click_path(
    clicked_tile: dict, 
    final_dest: dict, 
    intended_path: list,
    clicked_path: list = None
) -> bool:
    """
    Verify that clicking at clicked_tile generates a valid path to final_dest.
    
    Args:
        clicked_tile: {"x": int, "y": int} - The tile that was clicked
        final_dest: {"x": int, "y": int} - Final destination
        intended_path: List of waypoints from player to final destination
        clicked_path: Optional pre-calculated path from clicked tile to destination (avoids recalculation)
    
    Returns:
        True if path is valid, False otherwise
    """
    if not clicked_tile or not final_dest or not intended_path:
        return False
    
    clicked_x = clicked_tile.get("x")
    clicked_y = clicked_tile.get("y")
    dest_x = final_dest.get("x")
    dest_y = final_dest.get("y")
    
    if clicked_x is None or clicked_y is None or dest_x is None or dest_y is None:
        return False
    
    # Get path from clicked tile to final destination (only if not provided)
    if clicked_path is None:
        rect = (dest_x - 1, dest_x + 1, dest_y - 1, dest_y + 1)
        clicked_path, _ = ipc.path(rect=rect, visualize=False)
    
    if not clicked_path:
        return False
    
    # Check if clicked tile is on/near the intended path
    if _is_tile_on_path(clicked_x, clicked_y, intended_path, max_distance_from_path=3):
        return True
    
    # Check if the path from clicked tile converges with intended path
    # Simple check: see if any waypoints from clicked_path are on intended_path
    for wp in clicked_path[:min(10, len(clicked_path))]:  # Check first 10 waypoints
        wp_x = wp.get("x") or wp.get("world", {}).get("x")
        wp_y = wp.get("y") or wp.get("world", {}).get("y")
        if wp_x is not None and wp_y is not None:
            if _is_tile_on_path(wp_x, wp_y, intended_path, max_distance_from_path=2):
                return True
    
    # If clicked path doesn't converge, check if it at least heads toward destination
    # Get first waypoint of clicked path
    first_wp = clicked_path[0] if clicked_path else None
    if first_wp:
        wp_x = first_wp.get("x") or first_wp.get("world", {}).get("x")
        wp_y = first_wp.get("y") or first_wp.get("world", {}).get("y")
        if wp_x is not None and wp_y is not None:
            # Check if first waypoint is closer to destination than clicked tile
            clicked_dist = _calculate_distance(clicked_x, clicked_y, dest_x, dest_y)
            wp_dist = _calculate_distance(wp_x, wp_y, dest_x, dest_y)
            if wp_dist < clicked_dist:
                return True  # Path is heading toward destination
    
    return False


def _get_player_movement_state() -> dict:
    """
    Get sophisticated player movement state using IPC get_player().
    
    Returns:
        dict with:
        - "position": {"x": int, "y": int}
        - "is_moving": bool (position changed since last check)
        - "is_running": bool
        - "is_walking": bool
        - "is_idle": bool
        - "animation": int
        - "orientation": int
        - "movement_direction": {"dx": int, "dy": int} or None
    """
    global _movement_state
    
    resp = ipc.get_player()
    if not resp or not resp.get("ok") or not resp.get("player"):
        return None
    
    player_data = resp.get("player", {})
    current_x = player_data.get("worldX")
    current_y = player_data.get("worldY")
    animation = player_data.get("animation", -1)
    pose_animation = player_data.get("poseAnimation", -1)
    orientation = player_data.get("orientation", 0)
    
    # Use pose animation to determine movement state
    # 824 = running, 819 = walking, 808 = standing still
    is_running = player_data.get("isRunning", False)  # From IPC (poseAnimation == 824)
    is_walking_pose = player_data.get("isWalking", False)  # From IPC (poseAnimation == 819)
    is_standing = player_data.get("isStanding", False)  # From IPC (poseAnimation == 808)
    
    if current_x is None or current_y is None:
        return None
    
    current_pos = {"x": current_x, "y": current_y}
    last_pos = _movement_state.get("last_player_pos")
    
    # Determine if moving based on position change
    is_moving = False
    movement_direction = None
    if last_pos:
        dx = current_x - last_pos.get("x", 0)
        dy = current_y - last_pos.get("y", 0)
        if dx != 0 or dy != 0:
            is_moving = True
            movement_direction = {"dx": dx, "dy": dy}
    
    # Determine movement state from pose animation and position change
    # Pose animation: 824 = running, 819 = walking, 808 = standing
    # If position changed, we're moving (regardless of pose animation)
    is_walking = is_walking_pose or (is_moving and not is_running)
    is_idle = is_standing and not is_moving
    
    # Update state
    _movement_state["last_player_pos"] = current_pos
    _movement_state["last_player_animation"] = animation
    _movement_state["last_player_orientation"] = orientation
    
    return {
        "position": current_pos,
        "is_moving": is_moving,
        "is_running": is_running,
        "is_walking": is_walking,
        "is_idle": is_idle,
        "animation": animation,
        "orientation": orientation,
        "movement_direction": movement_direction
    }


def _is_moving_toward_path(
    movement_state: dict, 
    current_path: list, 
    final_dest: dict,
    path_to_clicked: list = None,
    path_to_dest: list = None
) -> bool:
    """
    Check if player is currently moving toward the path/destination by comparing paths.
    
    Gets the path from player's current position to the last clicked tile and compares it
    to the path from player's current position to the final destination.
    
    Args:
        movement_state: Result from _get_player_movement_state()
        current_path: List of waypoints from last click to final destination (may be outdated)
        final_dest: {"x": int, "y": int} - Final destination
        path_to_clicked: Optional pre-calculated path from player to last clicked tile (avoids recalculation)
        path_to_dest: Optional pre-calculated path from player to final destination (avoids recalculation)
    
    Returns:
        True if player is moving toward the path/destination, False otherwise
    """
    global _movement_state
    
    if not movement_state or not movement_state.get("is_moving"):
        return False  # Not moving
    
    player_pos = movement_state["position"]
    player_x = player_pos["x"]
    player_y = player_pos["y"]
    dest_x = final_dest.get("x")
    dest_y = final_dest.get("y")
    
    if dest_x is None or dest_y is None:
        return False
    
    # Get the last clicked tile
    last_clicked = _movement_state.get("last_clicked_tile")
    if not last_clicked:
        return False  # No last clicked tile to compare
    
    clicked_x = last_clicked.get("x")
    clicked_y = last_clicked.get("y")
    
    if clicked_x is None or clicked_y is None:
        return False
    
    # Get path from current player position to last clicked tile (only if not provided)
    if path_to_clicked is None:
        rect_clicked = (clicked_x - 1, clicked_x + 1, clicked_y - 1, clicked_y + 1)
        path_to_clicked, _ = ipc.path(rect=rect_clicked, visualize=False)
    
    # Get path from current player position to final destination (only if not provided)
    if path_to_dest is None:
        rect_dest = (dest_x - 1, dest_x + 1, dest_y - 1, dest_y + 1)
        path_to_dest, _ = ipc.path(rect=rect_dest, visualize=False)
    
    if not path_to_clicked or not path_to_dest:
        return False  # Can't get paths, can't verify
    
    # Compare the two paths - check if they converge or head in similar direction
    # Check if the first few waypoints of both paths are similar
    max_check = min(5, len(path_to_clicked), len(path_to_dest))
    if max_check == 0:
        return False
    
    similar_waypoints = 0
    for i in range(max_check):
        wp_clicked = path_to_clicked[i]
        wp_dest = path_to_dest[i]
        
        wp_clicked_x = wp_clicked.get("x") or wp_clicked.get("world", {}).get("x")
        wp_clicked_y = wp_clicked.get("y") or wp_clicked.get("world", {}).get("y")
        wp_dest_x = wp_dest.get("x") or wp_dest.get("world", {}).get("x")
        wp_dest_y = wp_dest.get("y") or wp_dest.get("world", {}).get("y")
        
        if wp_clicked_x is None or wp_clicked_y is None or wp_dest_x is None or wp_dest_y is None:
            continue
        
        # Check if waypoints are the same or very close (within 2 tiles)
        distance = _calculate_distance(wp_clicked_x, wp_clicked_y, wp_dest_x, wp_dest_y)
        if distance <= 2:
            similar_waypoints += 1
    
    # If at least 60% of the first waypoints are similar, paths are converging
    similarity_ratio = similar_waypoints / max_check if max_check > 0 else 0
    if similarity_ratio >= 0.6:
        return True  # Paths are similar, we're moving in the right direction
    
    # Also check if the path to clicked tile converges with the path to destination
    # by checking if any waypoint from path_to_clicked is on/near path_to_dest
    for wp in path_to_clicked[:min(10, len(path_to_clicked))]:
        wp_x = wp.get("x") or wp.get("world", {}).get("x")
        wp_y = wp.get("y") or wp.get("world", {}).get("y")
        if wp_x is not None and wp_y is not None:
            if _is_tile_on_path(wp_x, wp_y, path_to_dest, max_distance_from_path=2):
                return True  # Path to clicked converges with path to destination
    
    return False  # Paths don't converge, we're not moving toward destination


def _should_path_toward_target_with_movement_state(
    path_distance: int,
    manhattan_distance: float,
    movement_state: dict = None,
    running_min_path: int = 5,
    running_mid_path: int = 10,
    running_max_path: int = 20,
    running_min_click_prob: float = 0.1,
    running_mid_click_prob: float = 0.8,
    running_max_click_prob: float = 1.0,
    walking_min_path: int = 3,
    walking_mid_path: int = 12,
    walking_max_path: int = 15,
    walking_min_click_prob: float = 0.2,
    walking_mid_click_prob: float = 0.2,
    walking_max_click_prob: float = 1.0,
    stationary_min_path: int = 5,
    stationary_mid_path: int = 10,
    stationary_max_path: int = 20,
    stationary_min_click_prob: float = 0.1,
    stationary_mid_click_prob: float = 0.8,
    stationary_max_click_prob: float = 1.0,
    manhattan_cutoff_min: float = 20.0,
    manhattan_cutoff_max: float = 25.0,
    probability_variance: float = 0.05
) -> bool:
    """
    Determine if we should path toward a target (object/NPC/ground item) instead of clicking directly.
    
    Uses movement-state-aware probability with path distance (waypoint count) and Manhattan distance guard.
    
    Args:
        path_distance: Path distance (number of waypoints) to target
        manhattan_distance: Manhattan distance to target (for guard check)
        movement_state: Movement state dict from _get_player_movement_state() (None = assume stationary)
        running_min_path: Below this path distance, always click directly (running)
        running_mid_path: Midpoint for probability curve (running)
        running_max_path: Above this, use min_click_prob (running)
        running_min_click_prob: Minimum click-directly probability at max_path (with variance)
        running_mid_click_prob: Click-directly probability at mid_path (with variance)
        running_max_click_prob: Maximum click-directly probability at min_path (with variance)
        walking_min_path: Below this path distance, always click directly (walking)
        walking_mid_path: Midpoint for probability curve (walking)
        walking_max_path: Above this, use min_click_prob (walking)
        walking_min_click_prob: Minimum click-directly probability at max_path (with variance)
        walking_mid_click_prob: Click-directly probability at mid_path (with variance)
        walking_max_click_prob: Maximum click-directly probability at min_path (with variance)
        stationary_min_path: Below this path distance, always click directly (stationary)
        stationary_mid_path: Midpoint for probability curve (stationary)
        stationary_max_path: Above this, use min_click_prob (stationary)
        stationary_min_click_prob: Minimum click-directly probability at max_path (with variance)
        stationary_mid_click_prob: Click-directly probability at mid_path (with variance)
        stationary_max_click_prob: Maximum click-directly probability at min_path (with variance)
        manhattan_cutoff_min: Minimum Manhattan distance for always-path guard
        manhattan_cutoff_max: Maximum Manhattan distance for always-path guard
        probability_variance: Variance to add to probability values (e.g., 0.05 = ±5%)
    
    Returns:
        True if we should path toward target, False if we should click directly
    """
    # Manhattan distance guard: if very far (20-25 tiles, randomized), always path
    manhattan_cutoff = random.uniform(manhattan_cutoff_min, manhattan_cutoff_max)
    if manhattan_distance >= manhattan_cutoff:
        return True  # Always path when very far (Manhattan distance)
    
    # Determine movement state
    is_running = False
    is_walking = False
    is_stationary = True  # Default to stationary
    
    if movement_state:
        is_running = movement_state.get("is_running", False)
        is_walking = movement_state.get("is_walking", False)
        is_stationary = movement_state.get("is_idle", True)
    
    # Select parameters based on movement state
    if is_running:
        min_path = running_min_path
        mid_path = running_mid_path
        max_path = running_max_path
        min_click_prob = running_min_click_prob
        mid_click_prob = running_mid_click_prob
        max_click_prob = running_max_click_prob
    elif is_walking:
        min_path = walking_min_path
        mid_path = walking_mid_path
        max_path = walking_max_path
        min_click_prob = walking_min_click_prob
        mid_click_prob = walking_mid_click_prob
        max_click_prob = walking_max_click_prob
    else:  # Stationary
        min_path = stationary_min_path
        mid_path = stationary_mid_path
        max_path = stationary_max_path
        min_click_prob = stationary_min_click_prob
        mid_click_prob = stationary_mid_click_prob
        max_click_prob = stationary_max_click_prob
    
    # Add variance to probability values
    def add_variance(prob: float) -> float:
        variance = random.uniform(-probability_variance, probability_variance)
        return max(0.0, min(1.0, prob + variance))
    
    min_click_prob = add_variance(min_click_prob)
    mid_click_prob = add_variance(mid_click_prob)
    max_click_prob = add_variance(max_click_prob)
    
    # Calculate click-directly probability based on path distance
    if path_distance <= min_path:
        # Always click directly when very close
        click_directly_prob = max_click_prob
    elif path_distance <= mid_path:
        # Linear interpolation between min_path and mid_path
        ratio = (path_distance - min_path) / (mid_path - min_path)
        click_directly_prob = max_click_prob + (mid_click_prob - max_click_prob) * ratio
    elif path_distance <= max_path:
        # Linear interpolation between mid_path and max_path
        ratio = (path_distance - mid_path) / (max_path - mid_path)
        click_directly_prob = mid_click_prob + (min_click_prob - mid_click_prob) * ratio
    else:
        # Above max_path, use min_click_prob
        click_directly_prob = min_click_prob
    
    # Clamp probability to [0, 1]
    click_directly_prob = max(0.0, min(1.0, click_directly_prob))
    
    # Return True if we should path (inverse of click_directly_prob)
    return random.random() >= click_directly_prob


def _should_path_toward_target(distance: float, min_distance: float = 3.0, max_distance: float = 15.0, base_probability: float = 0.3) -> bool:
    """
    DEPRECATED: Use _should_path_toward_target_with_movement_state() instead.
    Kept for backward compatibility.
    """
    if distance <= min_distance:
        return False
    if distance >= max_distance:
        return True
    ratio = (distance - min_distance) / (max_distance - min_distance)
    probability = base_probability + (1.0 - base_probability) * ratio
    return random.random() < probability


def _calculate_path_click_location_for_target(
    player_x: int, 
    player_y: int, 
    target_x: int, 
    target_y: int, 
    radius: int = 5,
    center_bias: float = 0.7
) -> dict:
    """
    Calculate where to click when pathing toward a target (object/NPC/ground item).
    Uses normal distribution to click in an area AROUND the target, not always ahead.
    
    This is different from _calculate_click_location which always clicks ahead.
    For targets, we want to click in a general area around them.
    
    Args:
        player_x, player_y: Player's current position
        target_x, target_y: Target's position
        radius: Radius around target to consider for clicking (default 5 tiles)
        center_bias: How much to bias toward center (0.0 = uniform, 1.0 = very centered)
    
    Returns:
        {"x": int, "y": int} - World coordinates to click (in area around target)
    """
    from helpers.utils import normal_number
    
    # Create a rectangle around the target
    min_x = target_x - radius
    max_x = target_x + radius
    min_y = target_y - radius
    max_y = target_y + radius
    
    # Use normal distribution to pick a point in this rectangle
    # Higher center_bias = more likely to click near the target
    click_x = normal_number(min_x, max_x, center_bias, "int")
    click_y = normal_number(min_y, max_y, center_bias, "int")
    
    return {"x": click_x, "y": click_y}


def _handle_target_interaction_with_movement(
    target_world_coords: dict,
    click_directly_callback: callable,
    path_click_callback: callable = None,
    min_distance: float = 3.0,
    max_distance: float = 15.0,
    base_probability: float = 0.3,
    path_radius: int = 5,
    path_center_bias: float = 0.7
) -> dict | bool | None:
    """
    Reusable function to handle target interaction (object/NPC/ground item) with sophisticated movement.
    
    Decides whether to click the target directly or path toward it based on distance-based probability.
    Uses the full sophisticated movement system when pathing.
    
    Args:
        target_world_coords: {"x": int, "y": int, "p": int} - Target's world coordinates
        click_directly_callback: Function to call when clicking target directly.
                                 Should accept (target_world_coords, click_coords) and return result.
        path_click_callback: Optional function to call when pathing toward target.
                            If None, uses go_to_tile(). Should accept (click_location) and return result.
        min_distance: Below this distance, always click directly
        max_distance: Above this distance, always path
        base_probability: Base probability at min_distance
        path_radius: Radius around target when pathing (for normal distribution)
        path_center_bias: Center bias for normal distribution when pathing
    
    Returns:
        Result from click_directly_callback or path_click_callback, or None on failure
    """
    from actions import player
    
    target_x = target_world_coords.get("x")
    target_y = target_world_coords.get("y")
    
    if not isinstance(target_x, int) or not isinstance(target_y, int):
        return None
    
    player_x = player.get_x()
    player_y = player.get_y()
    
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return None
    
    # Calculate distance to target
    dx = target_x - player_x
    dy = target_y - player_y
    distance = abs(dx) + abs(dy)
    
    # Decide whether to path toward target or click directly
    should_path = _should_path_toward_target(distance, min_distance, max_distance, base_probability)
    
    if should_path:
        # Path toward target using sophisticated movement
        # Calculate click location using normal distribution around target
        click_location = _calculate_path_click_location_for_target(
            player_x, player_y, target_x, target_y, path_radius, path_center_bias
        )
        
        # Get intended path for verification
        rect = (target_x - 1, target_x + 1, target_y - 1, target_y + 1)
        intended_path, _ = ipc.path(rect=rect, visualize=False)
        
        if not intended_path:
            # Can't get path - fall back to direct click
            should_path = False
        else:
            # Verify the click location paths correctly
            final_dest = {"x": target_x, "y": target_y}
            if _verify_click_path(click_location, final_dest, intended_path):
                # Click location is valid - use path_click_callback or go_to_tile
                if path_click_callback:
                    return path_click_callback(click_location)
                else:
                    # Default: use go_to_tile with sophisticated movement
                    return go_to_tile(click_location["x"], click_location["y"], arrive_radius=0)
            else:
                # Click location doesn't path correctly - fall back to direct click
                should_path = False
    
    if not should_path:
        # Click target directly
        # Apply movement compensation to click coordinates
        movement_state = _get_player_movement_state()
        click_coords = None  # Will be calculated by callback if needed
        
        if movement_state and movement_state.get("is_moving"):
            # Get canvas coordinates for movement compensation
            proj = ipc.project_world_tile(target_x, target_y)
            if proj and proj.get("ok") and proj.get("canvas"):
                canvas_x = proj["canvas"].get("x")
                canvas_y = proj["canvas"].get("y")
                
                if isinstance(canvas_x, (int, float)) and isinstance(canvas_y, (int, float)):
                    movement_dir = movement_state.get("movement_direction")
                    is_running = movement_state.get("is_running", False)
                    
                    if movement_dir:
                        dx_dir = movement_dir.get("dx", 0)
                        dy_dir = movement_dir.get("dy", 0)
                        
                        # Calculate compensation based on movement speed
                        if is_running:
                            compensation = 0.4
                        else:
                            compensation = 0.2
                        
                        # Adjust canvas coordinates opposite to movement direction
                        if dx_dir != 0:
                            canvas_x = int(canvas_x - (dx_dir / abs(dx_dir)) * compensation * 50)
                        if dy_dir != 0:
                            canvas_y = int(canvas_y - (dy_dir / abs(dy_dir)) * compensation * 50)
                        
                        click_coords = {"x": canvas_x, "y": canvas_y}
        
        # Call the direct click callback
        return click_directly_callback(target_world_coords, click_coords)
    
    return None


def _should_click_again(final_dest: dict, current_path: list, player_pos: dict, is_running: bool) -> bool:
    """
    Determine if we need to click again based on sophisticated movement state detection.
    
    Uses IPC get_player() to determine:
    - If player is actually moving (position changed)
    - If player is walking/running/idle
    - Movement direction
    - If player has deviated from path or reached path end
    
    Args:
        final_dest: {"x": int, "y": int} - Final destination
        current_path: List of waypoints from last click to final destination
        player_pos: {"x": int, "y": int} - Current player position (from player.get_x/y)
        is_running: Whether player is running (legacy param, now using IPC data)
        
    Returns:
        True if we should click again, False if current path is still valid
    """
    global _movement_state
    
    if not current_path or not final_dest:
        return True  # No path, need to click
    
    # Get sophisticated movement state from IPC
    movement_state = _get_player_movement_state()
    if not movement_state:
        # Fallback to basic check if IPC fails
        if not player_pos:
            return True
        player_x = player_pos.get("x")
        player_y = player_pos.get("y")
        if player_x is None or player_y is None:
            return True
    else:
        player_x = movement_state["position"]["x"]
        player_y = movement_state["position"]["y"]
        is_moving = movement_state["is_moving"]
        is_idle = movement_state["is_idle"]
        movement_dir = movement_state.get("movement_direction")
    
    dest_x = final_dest.get("x")
    dest_y = final_dest.get("y")
    
    if player_x is None or player_y is None or dest_x is None or dest_y is None:
        return True
    
    # Check if we've reached the final destination
    distance_to_dest = _calculate_distance(player_x, player_y, dest_x, dest_y)
    if distance_to_dest <= 1:  # Within 1 tile
        _clear_movement_state()
        return False
    
    # If player is idle and not moving, check if we should click again
    if movement_state and is_idle and not is_moving:
        # Player has stopped - check if we're on path or need to continue
        if not _is_tile_on_path(player_x, player_y, current_path, max_distance_from_path=2):
            # Not on path and idle - need to click again
            return True
    
    # Check if player is still on/near the current path
    if not _is_tile_on_path(player_x, player_y, current_path, max_distance_from_path=2):
        # Player has deviated from path
        return True
    
    # Check if we've reached the end of current path (clicked destination)
    if len(current_path) > 0:
        last_wp = current_path[-1]
        last_x = last_wp.get("x") or last_wp.get("world", {}).get("x")
        last_y = last_wp.get("y") or last_wp.get("world", {}).get("y")
        if last_x is not None and last_y is not None:
            distance_to_end = _calculate_distance(player_x, player_y, last_x, last_y)
            if distance_to_end <= 1:
                # Reached end of current path, need to click again
                return True
    
    # Check if current path still heads toward final destination
    # Get a waypoint from the middle/end of current path
    if len(current_path) > 5:
        check_wp = current_path[min(5, len(current_path) - 1)]
        wp_x = check_wp.get("x") or check_wp.get("world", {}).get("x")
        wp_y = check_wp.get("y") or check_wp.get("world", {}).get("y")
        if wp_x is not None and wp_y is not None:
            # Check if this waypoint is still heading toward destination
            wp_to_dest = _calculate_distance(wp_x, wp_y, dest_x, dest_y)
            player_to_dest = _calculate_distance(player_x, player_y, dest_x, dest_y)
            if wp_to_dest >= player_to_dest:
                # Path is not heading toward destination
                return True
    
    # Current path is still valid
    return False


def _is_tile_on_path(tile_x: int, tile_y: int, path_waypoints: list, max_distance_from_path: int = 2) -> bool:
    """
    Check if a tile is on or near the intended path.
    
    Args:
        tile_x: X coordinate of the tile to check
        tile_y: Y coordinate of the tile to check
        path_waypoints: List of waypoint dicts from pathfinding (each with 'x', 'y', 'p' or 'world' dict)
        max_distance_from_path: Maximum Manhattan distance from path to be considered "on path" (default: 2)
        
    Returns:
        True if the tile is on or near the path, False otherwise
    """
    if not path_waypoints or len(path_waypoints) == 0:
        return False
    
    # Helper to extract coordinates from a waypoint
    def get_wp_coords(wp):
        if isinstance(wp, dict):
            world = wp.get("world")
            if world and isinstance(world, dict):
                return world.get("x"), world.get("y")
            return wp.get("x"), wp.get("y")
        return None, None
    
    # Check distance to each waypoint
    for wp in path_waypoints:
        wp_x, wp_y = get_wp_coords(wp)
        if wp_x is None or wp_y is None:
            continue
        
        # Check distance to this waypoint
        dist = _calculate_distance(tile_x, tile_y, wp_x, wp_y)
        if dist <= max_distance_from_path:
            return True
    
    # Check distance to path segments (between consecutive waypoints)
    for i in range(len(path_waypoints) - 1):
        wp1 = path_waypoints[i]
        wp2 = path_waypoints[i + 1]
        
        x1, y1 = get_wp_coords(wp1)
        x2, y2 = get_wp_coords(wp2)
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        
        # Check if tile is near the line segment between waypoints
        # For Manhattan distance, we check if the tile is within max_distance of the segment
        # by checking if it's closer to the segment than max_distance
        
        # Calculate distances to both endpoints
        dist_to_start = _calculate_distance(tile_x, tile_y, x1, y1)
        dist_to_end = _calculate_distance(tile_x, tile_y, x2, y2)
        
        # If close to either endpoint, it's on the path
        if dist_to_start <= max_distance_from_path or dist_to_end <= max_distance_from_path:
            return True
        
        # Check if tile is "between" the waypoints (rough approximation for Manhattan)
        # For a tile to be on the segment, it should be closer to the segment than to either endpoint
        # We use a simple check: if the tile is within the bounding box of the segment + max_distance
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        # Expand bounding box by max_distance
        if (min_x - max_distance_from_path <= tile_x <= max_x + max_distance_from_path and
            min_y - max_distance_from_path <= tile_y <= max_y + max_distance_from_path):
            # Tile is in the expanded bounding box - check if it's actually near the segment
            # Simple check: if the sum of distances to both endpoints is close to the segment length
            segment_length = _calculate_distance(x1, y1, x2, y2)
            total_dist = dist_to_start + dist_to_end
            # If total distance is close to segment length, tile is near the segment
            if abs(total_dist - segment_length) <= max_distance_from_path * 2:
                return True
    
    return False


def _first_blocking_door_from_waypoints(waypoints: list[dict]) -> dict | None:
    """
    Find the first blocking door from a list of waypoints.
    """
    if not waypoints:
        return None

    for i, wp in enumerate(waypoints):
        door_info = wp.get("door", {})
        if not door_info.get("present"):
            continue

        closed = door_info.get("closed")
        if closed is False:
            continue  # already open

        # Check if door has clickable geometry
        bounds = door_info.get("bounds")
        canvas = door_info.get("canvas")

        if not isinstance(bounds, dict) and not isinstance(canvas, dict):
            continue  # no clickable geometry

        # Create door plan
        wx = wp.get("x")
        wy = wp.get("y")
        wp_plane = wp.get("p", 0)

        if not isinstance(wx, int) or not isinstance(wy, int):
            continue

        door_plan = {
            "x": wx,
            "y": wy,
            "p": wp_plane,
            "door": door_info
        }

        # Check if this door is actually blocking the path
        if _is_door_blocking_path(door_plan, waypoints):
            return door_plan
        # If door is not blocking, continue to next door

    return None

def go_to_closest_bank() -> dict | None:
    """Go to the closest bank."""
    bank_key = closest_bank_key()
    if bank_key:
        return go_to(bank_key)

def in_area(rect_or_key: str | tuple | list) -> bool:
    """Check if player is in the specified area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
    """
    
    # Get player position
    player_x, player_y = player.get_x(), player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Get area rectangle - handle both tuple/rect and string key
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
    else:
        # Treat as area key and get rectangle
        rect = get_nav_rect(str(rect_or_key))
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return False
    
    min_x, max_x, min_y, max_y = rect
    return min_x <= player_x <= max_x and min_y <= player_y <= max_y


def get_long_distance_waypoints(destination_key: str, custom_dest_rect: tuple = None, destination: str = "center") -> list[tuple[int, int]] | None:
    """
    Generate waypoints for long-distance travel using the collision map pathfinder.
    This calls the pathfinder.py directly to get a complete path.
    
    Args:
        destination_key: Navigation key for destination, or "tile" for single tile
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
                         OR single tile as (x, x, y, y) for 1x1 area
        destination: Destination selection method ("center", "random", "center_weighted") - ignored for single tiles
    """
    import time
    total_start = time.time()
    print(f"\n{'='*80}")
    print(f"[LONG_DISTANCE_PATHFINDING] Starting pathfinding for destination: {destination_key}")
    print(f"{'='*80}")
    
    try:
        # Import pathfinder functions
        from collision_cache.pathfinder import (
            load_collision_data, 
            get_walkable_tiles, 
            astar_pathfinding
        )
        
        # Get current player position
        step_start = time.time()
        current_pos = get_player_position()
        step_time = time.time() - step_start
        if not current_pos:
            print(f"[LONG_DISTANCE_PATHFINDING] ERROR: Could not get player position (took {step_time:.3f}s)")
            return None
        print(f"[LONG_DISTANCE_PATHFINDING] Step 1: Got player position {current_pos} (took {step_time:.3f}s)")
        
        # Get destination coordinates
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            dest_rect = custom_dest_rect
            min_x, max_x, min_y, max_y = dest_rect
            # Check if this is a single tile (1x1 area)
            is_single_tile = (min_x == max_x and min_y == max_y)
        else:
            if destination_key == "tile":
                # Single tile case - should have custom_dest_rect
                return None
            dest_rect = get_nav_rect(destination_key)
            if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
                return None
            min_x, max_x, min_y, max_y = dest_rect
            is_single_tile = False
        
        # Handle destination selection (skip for single tiles)
        if is_single_tile:
            # Single tile - use the tile coordinates directly
            destination = (min_x, min_y)
        else:
            # Area - select destination tile within area
            if destination == "random":
                selected_tile = get_random_walkable_tile(destination_key, dest_rect)
            elif destination == "center_weighted":
                selected_tile = get_center_weighted_walkable_tile(destination_key, dest_rect)
            else:  # "center" or default
                selected_tile = None
        
        if selected_tile:
            destination = selected_tile
        else:
            # Fallback to center if no tile found or using center mode
            dest_center_x = (min_x + max_x) // 2
            dest_center_y = (min_y + max_y) // 2
            destination = (dest_center_x, dest_center_y)
        
        # Load collision data filtered by start and destination coordinates
        step_start = time.time()
        print(f"[LONG_DISTANCE_PATHFINDING] Step 2: Loading collision data from cache...")
        print(f"[LONG_DISTANCE_PATHFINDING]   Start: {current_pos}, Destination: {destination}")
        collision_data = load_collision_data(current_pos, destination)
        step_time = time.time() - step_start
        if not collision_data:
            print(f"[LONG_DISTANCE_PATHFINDING] ERROR: Failed to load collision data (took {step_time:.3f}s)")
            return None
        print(f"[LONG_DISTANCE_PATHFINDING] Step 2: Loaded {len(collision_data)} collision tiles (took {step_time:.3f}s)")
        
        # Get walkable tiles
        step_start = time.time()
        print(f"[LONG_DISTANCE_PATHFINDING] Step 3: Processing walkable tiles from {len(collision_data)} collision tiles...")
        walkable_tiles, blocked_tiles, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
        step_time = time.time() - step_start
        print(f"[LONG_DISTANCE_PATHFINDING] Step 3: Processed {len(walkable_tiles)} walkable, {len(blocked_tiles)} blocked tiles (took {step_time:.3f}s)")
        
        # Find closest walkable tiles to start and goal
        def find_closest_walkable(target, candidates):
            min_dist = float('inf')
            closest = None
            for candidate in candidates:
                dist = ((target[0] - candidate[0])**2 + (target[1] - candidate[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest = candidate
            return closest
        
        # Find walkable tiles near current position
        step_start = time.time()
        current_walkable = []
        search_radius = 10
        search_area = (2 * search_radius + 1) ** 2
        print(f"[LONG_DISTANCE_PATHFINDING] Step 4: Finding walkable tiles near start (searching {search_area} tiles)...")
        for x in range(current_pos[0] - 10, current_pos[0] + 11):
            for y in range(current_pos[1] - 10, current_pos[1] + 11):
                if (x, y) in walkable_tiles:
                    current_walkable.append((x, y))
        step_time = time.time() - step_start
        print(f"[LONG_DISTANCE_PATHFINDING] Step 4: Found {len(current_walkable)} walkable tiles near start (took {step_time:.3f}s)")
        
        # Find walkable tiles near destination
        step_start = time.time()
        dest_walkable = []
        dest_center_x = destination[0]
        dest_center_y = destination[1]
        print(f"[LONG_DISTANCE_PATHFINDING] Step 5: Finding walkable tiles near destination (searching {search_area} tiles)...")
        for x in range(dest_center_x - 10, dest_center_x + 11):
            for y in range(dest_center_y - 10, dest_center_y + 11):
                # Only include tiles that are both walkable AND within the target area bounds
                if ((x, y) in walkable_tiles and 
                    min_x <= x <= max_x and min_y <= y <= max_y):
                    dest_walkable.append((x, y))
        step_time = time.time() - step_start
        print(f"[LONG_DISTANCE_PATHFINDING] Step 5: Found {len(dest_walkable)} walkable tiles near destination (took {step_time:.3f}s)")
        
        if not current_walkable or not dest_walkable:
            return None
        
        # Find closest walkable tiles
        step_start = time.time()
        start = find_closest_walkable(current_pos, current_walkable)
        goal = find_closest_walkable(destination, dest_walkable)
        step_time = time.time() - step_start
        print(f"[LONG_DISTANCE_PATHFINDING] Step 6: Found closest walkable tiles - start: {start}, goal: {goal} (took {step_time:.3f}s)")
        
        # Use A* pathfinding directly - pass wall_masks to avoid redundant loading
        step_start = time.time()
        print(f"[LONG_DISTANCE_PATHFINDING] Step 7: Starting A* pathfinding from {start} to {goal}...")
        print(f"[LONG_DISTANCE_PATHFINDING]   Walkable tiles available: {len(walkable_tiles)}")
        print(f"[LONG_DISTANCE_PATHFINDING]   Passing wall_masks to avoid redundant loading...")
        path = astar_pathfinding(start, goal, walkable_tiles, wall_masks=wall_masks, orientation_blockers=orientation_blockers)
        step_time = time.time() - step_start
        if not path:
            print(f"[LONG_DISTANCE_PATHFINDING] ERROR: A* pathfinding failed (took {step_time:.3f}s)")
            return None
        
        total_time = time.time() - total_start
        print(f"[LONG_DISTANCE_PATHFINDING] Step 7: A* pathfinding completed - found path with {len(path)} waypoints (took {step_time:.3f}s)")
        print(f"[LONG_DISTANCE_PATHFINDING] {'='*80}")
        print(f"[LONG_DISTANCE_PATHFINDING] TOTAL TIME: {total_time:.3f}s")
        print(f"[LONG_DISTANCE_PATHFINDING] {'='*80}\n")
        
        return path
        
    except Exception as e:
        total_time = time.time() - total_start
        print(f"[LONG_DISTANCE_PATHFINDING] ERROR: Exception after {total_time:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def go_to_and_find_npc(rect_or_key: str | tuple | list, npc_name: str, 
                                                                                                                                                                                                                                                                                                                                center: bool = False, search_radius: int = 10, 
                      max_search_attempts: int = 20) -> dict | None:
    """
    Go to an area and then search for an NPC by traversing around the area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        npc_name: Name of the NPC to find (partial match, case-insensitive)
        center: If True, go to center of region instead of stopping at boundary
        search_radius: Radius around the destination to search for NPC
        max_search_attempts: Maximum number of search attempts before giving up
        
    Returns:
        NPC data if found, None otherwise
    """
    print(f"[TRAVEL] Going to {rect_or_key} to find {npc_name}")

    from actions.npc import closest_npc_by_name
    if closest_npc_by_name(npc_name):
        return True
    
    # First, go to the destination area
    if not in_area(rect_or_key) and not closest_npc_by_name(npc_name):
        go_to_result = go_to(rect_or_key, center=center)
        if go_to_result is None:
            print(f"[TRAVEL] Failed to go to {rect_or_key}")
            return None
        return False

    if in_area(rect_or_key) and not closest_npc_by_name(npc_name):
        move_to_random_tile_in_area(rect_or_key)
        return False

    return False



def move_to_random_tile_near_player(max_distance: int = 3) -> bool:
    """
    Move to a random walkable tile within the specified distance of the player.
    
    Args:
        max_distance: Maximum distance in tiles from player (default: 3)
    
    Returns:
        bool: True if movement was successful, False otherwise
    """
    try:
        # Get current player position
        player_pos = get_player_position()
        if not player_pos:
            # Could not get player position
            return False
        
        player_x, player_y = player_pos
        print(f"[TRAVEL] Finding walkable tiles within {max_distance} tiles")
        
        # Load collision data to find walkable tiles
        from .long_distance_travel import load_collision_data, get_walkable_tiles
        
        collision_data = load_collision_data()
        if not collision_data:
            print(f"[TRAVEL] Could not load collision data")
            return False
        
        walkable_tiles, _ = get_walkable_tiles(collision_data)
        
        # Find all walkable tiles within the specified distance
        nearby_walkable = []
        for tile_x, tile_y in walkable_tiles:
            distance = max(abs(tile_x - player_x), abs(tile_y - player_y))
            if distance <= max_distance and distance > 0:  # Don't include current position
                nearby_walkable.append((tile_x, tile_y))
        
        if not nearby_walkable:
            print(f"[TRAVEL] No walkable tiles found within {max_distance} tiles of player")
            return False
        
        # Select a random walkable tile
        import random
        target_tile = random.choice(nearby_walkable)
        target_x, target_y = target_tile
        
        print(f"[TRAVEL] Moving to random tile ({target_x}, {target_y})")
        
        # Use old camera logic to move to the random tile
        world_coords = {"x": target_x, "y": target_y}
        result = _click_ground_old_camera(
            world_coords=world_coords,
            aim_ms=700
        )
        
        if result and result.get("ok"):
            print(f"[RANDOM_MOVE] Successfully moved to ({target_x}, {target_y})")
            return True
        else:
            print(f"[RANDOM_MOVE] Failed to move to ({target_x}, {target_y}): {result}")
            return False
            
    except Exception as e:
        print(f"[RANDOM_MOVE] Error moving to random tile: {e}")
        return False

def move_to_random_tile_in_area(area_key: str) -> bool:
    """
    Move to a random tile within the specified area using old camera logic.
    
    Args:
        area_key: The area key from BANK_REGIONS or REGIONS (e.g., "VARROCK_WEST_TREES", "FALADOR_COWS")
    
    Returns:
        bool: True if movement was successful, False otherwise
    """
    # Get area bounds
    area_bounds = None
    if area_key in BANK_REGIONS:
        area_bounds = BANK_REGIONS[area_key]
    elif area_key in REGIONS:
        area_bounds = REGIONS[area_key]
    else:
        print(f"[MOVE_TO_RANDOM_TILE] Unknown area: {area_key}")
        return False
    
    if not area_bounds or len(area_bounds) != 4:
        print(f"[MOVE_TO_RANDOM_TILE] Invalid area bounds for {area_key}: {area_bounds}")
        return False
    
    min_x, max_x, min_y, max_y = area_bounds
    
    # Generate random coordinates within the area bounds
    random_x = random.randint(min_x, max_x)
    random_y = random.randint(min_y, max_y)
    
    print(f"[MOVE_TO_RANDOM_TILE] Moving to random tile in {area_key}: ({random_x}, {random_y})")
    
    # Use old camera logic to move to the random tile
    world_coords = {"x": random_x, "y": random_y, "p": 0}
    result = _click_ground_old_camera(
        world_coords=world_coords,
        aim_ms=700
    )
    
    if result:
        print(f"[MOVE_TO_RANDOM_TILE] Successfully moved to random tile in {area_key}")
        return True
    else:
        print(f"[MOVE_TO_RANDOM_TILE] Failed to move to random tile in {area_key}")
        return False
