# ilbot/ui/simple_recorder/actions/travel.py
import random
import time

from . import player
from .player import get_player_position
from helpers.runtime_utils import ipc
from helpers.navigation import get_nav_rect, closest_bank_key, _merge_door_into_projection
from helpers.utils import sleep_exponential, get_random_walkable_tile, get_center_weighted_walkable_tile
from services.click_with_camera import click_ground_with_camera, click_object_with_camera

# Timing instrumentation constants
_GO_TO_TIMING_ENABLED = True

# Global variable to track the most recently traversed door
_most_recently_traversed_door = None

# Global variables for long-distance path caching
_long_distance_path_cache = None
_long_distance_destination = None
_long_distance_waypoint_index = 0

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
    start_index = _long_distance_waypoint_index  # Cannot go behind current index
    end_index = min(len(_long_distance_path_cache), _long_distance_waypoint_index + 15)  # Look ahead max 15 waypoints
    
    closest_distance = float('inf')
    closest_index = _long_distance_waypoint_index  # Default to current index
    
    for i in range(start_index, end_index):
        waypoint = _long_distance_path_cache[i]
        distance = _calculate_distance(player_x, player_y, waypoint[0], waypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i
    
    # Constraint: Don't jump more than 15 waypoints ahead of where we were
    max_allowed_index = _long_distance_waypoint_index + 15
    if closest_index > max_allowed_index:
        closest_index = max_allowed_index
    
    # Update our cached position in the path to the closest waypoint
    _long_distance_waypoint_index = closest_index
    
    # Set target waypoint 15-20 waypoints ahead of current position
    waypoints_ahead = min(20, len(_long_distance_path_cache) - closest_index - 1)
    if waypoints_ahead < 15:
        # If we're near the end, use the last waypoint
        target_index = len(_long_distance_path_cache) - 1
        target_waypoint = _long_distance_path_cache[target_index]
    else:
        # Look ahead 15-20 waypoints from current position
        target_index = closest_index + 15
        target_waypoint = _long_distance_path_cache[target_index]
    
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


def go_to(rect_or_key: str | tuple | list, center: bool = False, destination: str = "center") -> dict | None:
    """
    One movement click toward an area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        center: If True, go to center of region instead of stopping at boundary
        destination: Destination selection method ("center", "random", "center_weighted")
    """
    try:
        run_energy = player.get_run_energy()
        if run_energy is not None and run_energy > 2000 and not player.is_run_on():
            player.toggle_run()

        if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
            rect = tuple(rect_or_key)
            rect_key = "custom"
        else:
            rect_key = str(rect_or_key)
            # Handle special case for CLOSEST_BANK
            if rect_key == "CLOSEST_BANK":
                actual_bank_key = closest_bank_key()
                rect = get_nav_rect(actual_bank_key)
                rect_key = actual_bank_key  # Use the actual bank key for consistency
            else:
                rect = get_nav_rect(rect_key)
        
        # Handle destination selection
        if destination != "center" and rect_key != "custom":
            if destination == "random":
                selected_tile = get_random_walkable_tile(rect_key)
            elif destination == "center_weighted":
                selected_tile = get_center_weighted_walkable_tile(rect_key)
            else:
                selected_tile = None
            
            if selected_tile:
                # Use the selected tile as a custom destination
                rect = (selected_tile[0], selected_tile[0], selected_tile[1], selected_tile[1])
                rect_key = "custom"

        player_x, player_y = player.get_x(), player.get_y()
    
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            # Could not get player position
            return None
        
        # Check how many waypoints are left in the long-distance path
        waypoint_batch = _get_next_long_distance_waypoints(rect_key, rect, destination)
        waypoints_remaining = len(_long_distance_path_cache) - _long_distance_waypoint_index if _long_distance_path_cache else 0
    
        if waypoints_remaining > 0:
            print(f"[TRAVEL] Using long distance path ({waypoints_remaining} waypoints remaining)")
            # LONG DISTANCE: Get cached path and click ~25 tiles away
            if not waypoint_batch:
                # Fallback to short-distance pathfinding by treating as short distance
                distance = 25  # Force short-distance pathfinding
                # No waypoint batch, falling back to short distance
            else:
                # Use the single waypoint from the path
                target_waypoint = waypoint_batch[0]
                # Using target waypoint
                
                # Project waypoint to screen coordinates
                wps, _ = ipc.path(rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1))
            
                if wps:
                    waypoint_proj, _ = ipc.project_many(wps)
                    waypoint_proj = _merge_door_into_projection(wps, waypoint_proj)
                    waypoint_usable = [p for p in waypoint_proj if isinstance(p, dict) and p.get("canvas")]
                
                    if waypoint_usable:
                        chosen = waypoint_usable[-1]
                        world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
                        
                        # Check for doors on the long-distance path and handle them
                        gx = chosen["world"].get("x")
                        gy = chosen["world"].get("y")
                        
                        if isinstance(gx, int) and isinstance(gy, int):
                            # wps, dbg_path = ipc.path(goal=(gx, gy), visualize=False)
                            door_plan = _first_blocking_door_from_waypoints(wps)
                            
                            if door_plan:
                                door_result = _handle_door_opening(door_plan, wps)
                                if not door_result:
                                    return None  # Door opening failed
                                else:
                                    return True
                    else:
                        world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
                else:
                    world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
    
        else:
            print(f"[TRAVEL] Using short distance path")
            # SHORT DISTANCE: Use standard local pathing
            wps, dbg_path = ipc.path(rect=tuple(rect))
        
            if not wps:
                return None

            proj, dbg_proj = ipc.project_many(wps)
            proj = _merge_door_into_projection(wps, proj)
        
            usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
            if not usable:
                return None
            
            # Check for doors on the path and handle them
            if usable:
                last_waypoint = usable[-1]
                gx = last_waypoint["world"].get("x")
                gy = last_waypoint["world"].get("y")
                
                if isinstance(gx, int) and isinstance(gy, int):
                    # wps, dbg_path = ipc.path(goal=(gx, gy), visualize=False)
                    door_plan = _first_blocking_door_from_waypoints(wps)
                    
                    if door_plan:
                        door_result = _handle_door_opening(door_plan, wps)
                        if not door_result:
                            return None  # Door opening failed
                        else:
                            return True
        
            # Pick from the last 3 waypoints (or all if less than 3)
            if len(usable) <= 3:
                chosen = random.choice(usable)
            else:
                chosen = random.choice(usable[-3:])
            
            world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
        
        from ..services.click_with_camera import click_ground_with_camera

        result = click_ground_with_camera(
            world_coords=world_coords,
            description=f"Move toward {rect_key}",
            aim_ms=700,
            waypoint_path=wps
        )
        return result
    
    except Exception as e:
        raise e

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


def _calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


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
        destination_key: Navigation key for destination
        custom_dest_rect: Optional custom destination rectangle as (min_x, max_x, min_y, max_y)
        destination: Destination selection method ("center", "random", "center_weighted")
    """
    try:
        # Import pathfinder functions
        from collision_cache.pathfinder import (
            load_collision_data, 
            get_walkable_tiles, 
            astar_pathfinding
        )
        
        # Get current player position
        current_pos = get_player_position()
        if not current_pos:
            return None
        
        # Get destination coordinates
        if custom_dest_rect and isinstance(custom_dest_rect, (tuple, list)) and len(custom_dest_rect) == 4:
            dest_rect = custom_dest_rect
        else:
            dest_rect = get_nav_rect(destination_key)
            if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
                return None
        
        min_x, max_x, min_y, max_y = dest_rect
        
        # Handle destination selection
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
        collision_data = load_collision_data(current_pos, destination)
        if not collision_data:
            return None
        
        # Get walkable tiles
        walkable_tiles, blocked_tiles, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
        
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
        current_walkable = []
        for x in range(current_pos[0] - 10, current_pos[0] + 11):
            for y in range(current_pos[1] - 10, current_pos[1] + 11):
                if (x, y) in walkable_tiles:
                    current_walkable.append((x, y))
        
        # Find walkable tiles near destination
        dest_walkable = []
        for x in range(dest_center_x - 10, dest_center_x + 11):
            for y in range(dest_center_y - 10, dest_center_y + 11):
                # Only include tiles that are both walkable AND within the target area bounds
                if ((x, y) in walkable_tiles and 
                    min_x <= x <= max_x and min_y <= y <= max_y):
                    dest_walkable.append((x, y))
        
        if not current_walkable or not dest_walkable:
            return None
        
        # Find closest walkable tiles
        start = find_closest_walkable(current_pos, current_walkable)
        goal = find_closest_walkable(destination, dest_walkable)
        
        # Use A* pathfinding directly
        path = astar_pathfinding(start, goal, walkable_tiles)
        
        if not path:
            return None
        
        return path
        
    except Exception as e:
        print(f"[TRAVEL] Error generating waypoints: {e}")
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

    from ..actions.npc import closest_npc_by_name
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
        
        # Use click_ground_with_camera to move to the random tile
        world_coords = {"x": target_x, "y": target_y}
        result = click_ground_with_camera(
            world_coords=world_coords,
            description="Move to random tile"
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
    Move to a random tile within the specified area using click_ground_with_camera.
    
    Args:
        area_key: The area key from BANK_REGIONS or REGIONS (e.g., "VARROCK_WEST_TREES", "FALADOR_COWS")
    
    Returns:
        bool: True if movement was successful, False otherwise
    """
    # Import area definitions
    from ..constants import BANK_REGIONS, REGIONS
    
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
    
    # Use click_ground_with_camera to move to the random tile
    world_coords = {"x": random_x, "y": random_y, "p": 0}
    result = click_ground_with_camera(
        world_coords=world_coords,
        description=f"Random tile in {area_key}",
        aim_ms=700
    )
    
    if result:
        print(f"[MOVE_TO_RANDOM_TILE] Successfully moved to random tile in {area_key}")
        return True
    else:
        print(f"[MOVE_TO_RANDOM_TILE] Failed to move to random tile in {area_key}")
        return False
