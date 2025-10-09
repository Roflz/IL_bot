# ilbot/ui/simple_recorder/actions/travel.py
import random
import time

from . import player
from .player import get_player_position
from ..helpers.runtime_utils import ipc
from ..helpers.navigation import get_nav_rect, closest_bank_key, _merge_door_into_projection
from ..services.click_with_camera import click_ground_with_camera, click_object_with_camera

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


def _get_next_long_distance_waypoints(destination_key: str) -> list[tuple[int, int]] | None:
    """
    Get the next waypoint from the cached long-distance path.
    Generate path ONCE at the start, then follow it step by step.
    Uses path distance (not straight-line distance) to select waypoints.
    """
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index

    
    # Generate path ONCE if we don't have one for this destination
    if (_long_distance_path_cache is None or 
        _long_distance_destination != destination_key):
        
        waypoints = get_long_distance_waypoints(destination_key)
        if not waypoints:
            return None
        
        _long_distance_path_cache = waypoints
        _long_distance_destination = destination_key
        _long_distance_waypoint_index = 0
    
    # Check if we've reached the end of the path
    if _long_distance_waypoint_index >= len(_long_distance_path_cache):
        print(f"[LONG_DISTANCE] Reached end of path ({_long_distance_waypoint_index}/{len(_long_distance_path_cache)})")
        
        # Check if we're close to destination
        from ..helpers.navigation import get_nav_rect
        dest_rect = get_nav_rect(destination_key)
        if dest_rect and len(dest_rect) == 4:
            min_x, max_x, min_y, max_y = dest_rect
            dest_center_x = (min_x + max_x) // 2
            dest_center_y = (min_y + max_y) // 2
            
            player_x, player_y = player.get_x(), player.get_y()
            if isinstance(player_x, int) and isinstance(player_y, int):
                distance_to_dest = _calculate_distance(player_x, player_y, dest_center_x, dest_center_y)
                if distance_to_dest <= 15:  # Close to destination
                    print(f"[LONG_DISTANCE] Close to destination ({distance_to_dest:.1f} tiles), path complete!")
                    clear_long_distance_cache()
                else:
                    print(f"[LONG_DISTANCE] Still far from destination ({distance_to_dest:.1f} tiles), path may be incomplete")
        
        return None
    
    # Get player position
    player_x, player_y = player.get_x(), player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        # Fallback to next waypoint if we can't get player position
        current_waypoint = _long_distance_path_cache[_long_distance_waypoint_index]
        _long_distance_waypoint_index += 1
        return [current_waypoint]
    
    # Find the waypoint with the shortest path distance from player position
    # Only check waypoints from current index forward (cannot go backwards)
    start_index = _long_distance_waypoint_index  # Cannot go behind current index
    end_index = min(len(_long_distance_path_cache), _long_distance_waypoint_index + 25)  # Look ahead 25 waypoints
    
    shortest_path_distance = float('inf')
    best_index = _long_distance_waypoint_index  # Default to current index
    
    for i in range(start_index, end_index):
        path_distance = _calculate_path_distance_to_waypoint(i, player_x, player_y)
        if path_distance < shortest_path_distance:
            shortest_path_distance = path_distance
            best_index = i
    
    # Constraint: Don't jump more than 20 waypoints ahead of where we were
    max_allowed_index = _long_distance_waypoint_index + 20
    if best_index > max_allowed_index:
        best_index = max_allowed_index
    
    # Update our cached position in the path to the waypoint with shortest path distance
    _long_distance_waypoint_index = best_index
    
    # Calculate how many waypoints to look ahead for the target (max 20, or remaining waypoints if less)
    waypoints_ahead = min(20, len(_long_distance_path_cache) - best_index - 1)
    
    if waypoints_ahead > 0:
        target_index = best_index + waypoints_ahead
        target_waypoint = _long_distance_path_cache[target_index]
    else:
        # If we're near the end of the path, use the last waypoint
        target_waypoint = _long_distance_path_cache[-1]
        target_index = len(_long_distance_path_cache) - 1
    
    distance_to_target = _calculate_distance(player_x, player_y, target_waypoint[0], target_waypoint[1])
    
    # Simple progress indicator
    progress = f"{target_index + 1}/{len(_long_distance_path_cache)}"
    print(f"[TRAVEL] {progress} → {target_waypoint} ({distance_to_target:.0f}t)")
    
    return [target_waypoint]


def clear_long_distance_cache():
    """Clear the long-distance path cache."""
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index
    _long_distance_path_cache = None
    _long_distance_destination = None
    _long_distance_waypoint_index = 0
    print("[LONG_DISTANCE_CACHE] Cache cleared")

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
        path_waypoints: List of waypoints in the path
        
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
    
    # Find the path segment that goes through this door tile
    for i in range(len(path_waypoints) - 1):
        current_wp = path_waypoints[i]
        next_wp = path_waypoints[i + 1]
        
        # Check if this path segment goes through the door tile
        if (current_wp.get('x') == door_x and current_wp.get('y') == door_y) or \
           (next_wp.get('x') == door_x and next_wp.get('y') == door_y):
            
            # Calculate the direction of movement through the door
            dx = next_wp.get('x') - current_wp.get('x')
            dy = next_wp.get('y') - current_wp.get('y')
            
            # Determine which side of the door tile the path is crossing
            blocking_orientation = 0
            
            if dx > 0:  # Moving East (right)
                blocking_orientation |= 4  # East wall blocks eastward movement
            elif dx < 0:  # Moving West (left)
                blocking_orientation |= 1  # West wall blocks westward movement
            
            if dy < 0:  # Moving South (down)
                blocking_orientation |= 8  # South wall blocks southward movement
            elif dy > 0:  # Moving North (up)
                blocking_orientation |= 2  # North wall blocks northward movement
            
            # Check if the door has walls on the sides that would block this movement
            if (door_orientation_a & blocking_orientation) or (door_orientation_b & blocking_orientation):
                print(f"[DOOR_CHECK] Door at ({door_x}, {door_y}) is blocking path (orientation: {door_orientation_a}/{door_orientation_b}, blocking: {blocking_orientation})")
                return True
            else:
                print(f"[DOOR_CHECK] Door at ({door_x}, {door_y}) is NOT blocking path (orientation: {door_orientation_a}/{door_orientation_b}, blocking: {blocking_orientation})")
                return False
    
    # If door is not on the path, it's not blocking
    return False


def _handle_door_opening(door_plan: dict) -> bool:
    """
    Handle door opening logic with retry mechanism and recently traversed door tracking.
    
    Args:
        door_plan: Door plan containing door information
        
    Returns:
        True if door was opened successfully or skipped, False if door opening failed
    """
    global _most_recently_traversed_door
    
    if not door_plan:
        return True  # No door to handle
    
    # Check if this is the same door we just traversed
    if _is_same_door(door_plan):
        print(f"[DEBUG] Skipping recently traversed door at ({door_plan.get('x')}, {door_plan.get('y')})")
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
    # Get the current path to check if door is blocking
    player_x, player_y = player.get_x(), player.get_y()
    if isinstance(player_x, int) and isinstance(player_y, int):
        wps, dbg_path = ipc.path(goal=(door_x, door_y), visualize=False)
        if wps and not _is_door_blocking_path(door_plan, wps):
            _most_recently_traversed_door = door_plan
            print(f"[DOOR_CHECK] Door at ({door_x}, {door_y}) is not blocking path, skipping door opening")
            return True  # Door is not blocking, continue with normal movement
    
    # Door is closed, try to open it with retry logic
    door_opened = not door_plan["door"].get("closed")
    door_retry_count = 0
    max_door_retries = 3

    while not door_opened and door_retry_count < max_door_retries:
        # Get fresh door data for each retry attempt
        wps, dbg_path = ipc.path(goal=(door_x, door_y))
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
            print(f"[DOOR_OPENING] No valid world coordinates for door, skipping")
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
                    print(f"[DEBUG] Door opened successfully, marking as recently traversed")
                    door_opened = True
                    break
                time.sleep(0.1)

        if not door_opened:
            door_retry_count += 1
            if door_retry_count < max_door_retries:
                print(f"[DEBUG] Door opening attempt {door_retry_count} failed, retrying...")
                time.sleep(0.5)

    if not door_opened:
        # Door opening failed after retries
        print(f"[DEBUG] Failed to open door after {max_door_retries} attempts")
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
    
    # If wall_object is null, door doesn't exist (is open)
    # If wall_object exists, door is still there (is closed)
    og_door_id = door_plan['door'].get('id')
    new_door_id = resp['wall_object'].get('id')
    if not og_door_id == new_door_id:
        return True  # Door ID changed, it's open
    
    return False  # Door still exists with same ID, it's closed


def go_to(rect_or_key: str | tuple | list, center: bool = False) -> dict | None:
    """
    One movement click toward an area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        center: If True, go to center of region instead of stopping at boundary
    """
    import time
    start_time = time.time()
    print(f"[GO_TO_TIMING] Starting go_to for {rect_or_key}")
    
    # accept key or explicit (minX, maxX, minY, maxY)
    step_start = time.time()
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None
    print(f"[GO_TO_TIMING] Rect setup took {time.time() - step_start:.3f}s")
        
    # Get player position
    step_start = time.time()
    player_x, player_y = player.get_x(), player.get_y()
    print(f"[GO_TO_TIMING] Player position lookup took {time.time() - step_start:.3f}s")
    
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        print(f"[GO_TO] Could not get player position")
        return None
    
    # Check how many waypoints are left in the long-distance path
    step_start = time.time()
    waypoint_batch = _get_next_long_distance_waypoints(rect_key)
    waypoints_remaining = len(_long_distance_path_cache) - _long_distance_waypoint_index if _long_distance_path_cache else 0
    print(f"[GO_TO_TIMING] Long distance waypoint check took {time.time() - step_start:.3f}s")
    
    if waypoints_remaining > 0:
        print(f"[GO_TO_TIMING] Using LONG DISTANCE path (waypoints remaining: {waypoints_remaining})")
        # LONG DISTANCE: Get cached path and click ~25 tiles away
        if not waypoint_batch:
            # Fallback to short-distance pathfinding by treating as short distance
            distance = 25  # Force short-distance pathfinding
            print(f"[GO_TO_TIMING] No waypoint batch, falling back to short distance")
        else:
            # Use the single waypoint from the path
            target_waypoint = waypoint_batch[0]
            print(f"[GO_TO_TIMING] Using target waypoint: {target_waypoint}")
            
            # Project waypoint to screen coordinates
            step_start = time.time()
            waypoint_wps, _ = ipc.path(rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1))
            print(f"[GO_TO_TIMING] Waypoint path generation took {time.time() - step_start:.3f}s")
            
            if waypoint_wps:
                step_start = time.time()
                waypoint_proj, _ = ipc.project_many(waypoint_wps)
                waypoint_proj = _merge_door_into_projection(waypoint_wps, waypoint_proj)
                waypoint_usable = [p for p in waypoint_proj if isinstance(p, dict) and p.get("canvas")]
                print(f"[GO_TO_TIMING] Waypoint projection and filtering took {time.time() - step_start:.3f}s")
                
                if waypoint_usable:
                    chosen = waypoint_usable[-1]
                    world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
                    
                    # Check for doors on the long-distance path and handle them
                    gx = chosen["world"].get("x")
                    gy = chosen["world"].get("y")
                    
                    if isinstance(gx, int) and isinstance(gy, int):
                        step_start = time.time()
                        wps, dbg_path = ipc.path(goal=(gx, gy), visualize=False)
                        door_plan = _first_blocking_door_from_waypoints(wps)
                        print(f"[GO_TO_TIMING] Door check path generation took {time.time() - step_start:.3f}s")
                        
                        if door_plan:
                            step_start = time.time()
                            door_result = _handle_door_opening(door_plan)
                            print(f"[GO_TO_TIMING] Door opening took {time.time() - step_start:.3f}s")
                            if not door_result:
                                return None  # Door opening failed
                else:
                    world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
            else:
                world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
    
    else:
        print(f"[GO_TO_TIMING] Using SHORT DISTANCE path")
        # SHORT DISTANCE: Use standard local pathing
        step_start = time.time()
        wps, dbg_path = ipc.path(rect=tuple(rect))
        print(f"[GO_TO_TIMING] Short distance path generation took {time.time() - step_start:.3f}s")
        
        if not wps:
            return None
        
        step_start = time.time()
        proj, dbg_proj = ipc.project_many(wps)
        proj = _merge_door_into_projection(wps, proj)
        print(f"[GO_TO_TIMING] Short distance projection and door merge took {time.time() - step_start:.3f}s")
        
        usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
        if not usable:
            return None
        
        # Check for doors on the path and handle them
        if usable:
            last_waypoint = usable[-1]
            gx = last_waypoint["world"].get("x")
            gy = last_waypoint["world"].get("y")
            
            if isinstance(gx, int) and isinstance(gy, int):
                step_start = time.time()
                wps, dbg_path = ipc.path(goal=(gx, gy), visualize=False)
                door_plan = _first_blocking_door_from_waypoints(wps)
                print(f"[GO_TO_TIMING] Short distance door check path generation took {time.time() - step_start:.3f}s")
                
                if door_plan:
                    step_start = time.time()
                    door_result = _handle_door_opening(door_plan)
                    print(f"[GO_TO_TIMING] Short distance door opening took {time.time() - step_start:.3f}s")
                    if not door_result:
                        return None  # Door opening failed
        
        # Pick from the last 3 waypoints (or all if less than 3)
        if len(usable) <= 3:
            chosen = random.choice(usable)
        else:
            chosen = random.choice(usable[-3:])
        
        world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
    
    from ..services.click_with_camera import click_ground_with_camera
    
    step_start = time.time()
    result = click_ground_with_camera(
        world_coords=world_coords,
        description=f"Move toward {rect_key}",
        aim_ms=700,
        waypoint_path=wps
    )
    print(f"[GO_TO_TIMING] click_ground_with_camera took {time.time() - step_start:.3f}s")
    print(f"[GO_TO_TIMING] Total go_to execution took {time.time() - start_time:.3f}s")
    
    return result


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
        
        return door_plan
    
    return None

def go_to_closest_bank() -> dict | None:
    """Go to the closest bank."""
    player_data = ipc.get_player()
    bank_key = closest_bank_key()
    if bank_key:
        return go_to(bank_key)

def in_area(area_key: str) -> bool:
    """Check if player is in the specified area."""
    
    # Get player position
    player_x, player_y = player.get_x(), player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Get area rectangle
    rect = get_nav_rect(area_key)
    if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
        return False
    
    min_x, max_x, min_y, max_y = rect
    return min_x <= player_x <= max_x and min_y <= player_y <= max_y


def get_long_distance_waypoints(destination_key: str) -> list[tuple[int, int]] | None:
    """
    Generate waypoints for long-distance travel using the collision map pathfinder.
    This calls the pathfinder.py directly to get a complete path.
    """
    try:
        # Import pathfinder functions
        from ..pathfinder import (
            load_collision_data, 
            get_walkable_tiles, 
            astar_pathfinding
        )
        from ..helpers.navigation import get_nav_rect
        
        # Get current player position
        current_pos = get_player_position()
        if not current_pos:
            return None
        
        # Get destination coordinates
        dest_rect = get_nav_rect(destination_key)
        if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
            return None
        
        min_x, max_x, min_y, max_y = dest_rect
        dest_center_x = (min_x + max_x) // 2
        dest_center_y = (min_y + max_y) // 2
        destination = (dest_center_x, dest_center_y)
        
        # Load collision data
        collision_data = load_collision_data()
        if not collision_data:
            return None
        
        # Get walkable tiles
        walkable_tiles, blocked_tiles = get_walkable_tiles(collision_data)
        
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
        print(f"[LONG_DISTANCE] Error generating waypoints: {e}")
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
    print(f"[GO_TO_AND_FIND_NPC] Going to {rect_or_key} to find {npc_name}")

    from ..actions.npc import closest_npc_by_name
    if closest_npc_by_name(npc_name):
        return True
    
    # First, go to the destination area
    if not in_area(rect_or_key) and not closest_npc_by_name(npc_name):
        go_to_result = go_to(rect_or_key, center=center)
        if go_to_result is None:
            print(f"[GO_TO_AND_FIND_NPC] Failed to go to {rect_or_key}")
            return None
        return False

    if in_area(rect_or_key) and not closest_npc_by_name(npc_name):
        move_to_random_tile_in_area(rect_or_key)
        return False

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
