# ilbot/ui/simple_recorder/actions/travel.py
import random
import time

from .runtime import emit
# Camera import removed - using new camera system
from ..helpers.context import get_payload, get_ui
from ..helpers.navigation import get_nav_rect, closest_bank_key, bank_rect, player_in_rect, _merge_door_into_projection, _first_blocking_door_from_waypoints
from ..helpers.ipc import ipc_path, ipc_project_many
from ..services.camera_integration import dispatch_with_camera

# Global variable to track the most recently traversed door
_most_recently_traversed_door = None

# Global variables for long-distance path caching
_long_distance_path_cache = None
_long_distance_destination = None
_long_distance_waypoint_index = 0


def _find_closest_waypoint_index(player_x: int, player_y: int, waypoints: list[tuple[int, int]]) -> int:
    """
    Find the index of the waypoint closest to the player's current position.
    
    Args:
        player_x: Player's current X coordinate
        player_y: Player's current Y coordinate
        waypoints: List of (x, y) waypoint tuples
        
    Returns:
        Index of the closest waypoint
    """
    if not waypoints:
        return 0
    
    min_distance = float('inf')
    closest_index = 0
    
    for i, (wx, wy) in enumerate(waypoints):
        distance = _calculate_distance(player_x, player_y, wx, wy)
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    
    return closest_index


def _smooth_path(path: list[tuple[int, int]], min_distance: int = 5) -> list[tuple[int, int]]:
    """
    Smooth a path by removing unnecessary waypoints that are too close together.
    
    This helps reduce path oscillation by:
    1. Removing waypoints that are too close to each other
    2. Keeping waypoints that maintain path direction
    3. Ensuring minimum distance between waypoints
    
    Args:
        path: List of (x, y) waypoint tuples
        min_distance: Minimum distance between waypoints
        
    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]  # Always keep the start point
    
    i = 0
    while i < len(path) - 1:
        # Look ahead to find the next waypoint that's far enough away
        j = i + 1
        while j < len(path) - 1:
            distance = _calculate_distance(
                path[i][0], path[i][1],
                path[j][0], path[j][1]
            )
            if distance >= min_distance:
                break
            j += 1
        
        # Add the waypoint at position j
        smoothed.append(path[j])
        i = j
    
    # Always keep the end point
    if smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    
    return smoothed


# Removed complex path stability logic - we now just follow the path step by step


def _get_next_long_distance_waypoints(destination_key: str, payload: dict, batch_size: int = 35) -> list[tuple[int, int]] | None:
    """
    Get the next waypoint from the cached long-distance path.
    Generate path ONCE at the start, then follow it step by step.
    """
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index
    
    # Generate path ONCE if we don't have one for this destination
    if (_long_distance_path_cache is None or 
        _long_distance_destination != destination_key):
        
        print(f"[LONG_DISTANCE] Generating path ONCE for {destination_key}")
        waypoints = get_long_distance_waypoints(destination_key, payload)
        if not waypoints:
            print(f"[LONG_DISTANCE] Failed to generate path for {destination_key}")
            return None
        
        _long_distance_path_cache = waypoints
        _long_distance_destination = destination_key
        _long_distance_waypoint_index = 0
        print(f"[LONG_DISTANCE] Path generated with {len(waypoints)} waypoints - will follow this path")
    
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
            
            player_x, player_y = _get_player_position(payload)
            if isinstance(player_x, int) and isinstance(player_y, int):
                distance_to_dest = _calculate_distance(player_x, player_y, dest_center_x, dest_center_y)
                if distance_to_dest <= 15:  # Close to destination
                    print(f"[LONG_DISTANCE] Close to destination ({distance_to_dest:.1f} tiles), path complete!")
                    clear_long_distance_cache()
                else:
                    print(f"[LONG_DISTANCE] Still far from destination ({distance_to_dest:.1f} tiles), path may be incomplete")
        
        return None
    
    # Find a waypoint that's a good distance away (20-30 tiles)
    player_x, player_y = _get_player_position(payload)
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        # Fallback to next waypoint if we can't get player position
        current_waypoint = _long_distance_path_cache[_long_distance_waypoint_index]
        _long_distance_waypoint_index += 1
        return [current_waypoint]
    
    # Look ahead in the path to find a waypoint that's 20-30 tiles away
    target_waypoint = None
    target_index = _long_distance_waypoint_index
    
    for i in range(_long_distance_waypoint_index, len(_long_distance_path_cache)):
        waypoint = _long_distance_path_cache[i]
        distance = _calculate_distance(player_x, player_y, waypoint[0], waypoint[1])
        
        if distance >= 25:  # Found a waypoint that's 25+ tiles away
            target_waypoint = waypoint
            target_index = i
            break
        elif distance >= 20:  # Found a waypoint that's 20+ tiles away (acceptable)
            target_waypoint = waypoint
            target_index = i
    
    # If no waypoint is far enough, use the last waypoint in the path
    if target_waypoint is None:
        target_waypoint = _long_distance_path_cache[-1]
        target_index = len(_long_distance_path_cache) - 1
    
    # Update our position in the path to the target waypoint
    _long_distance_waypoint_index = target_index + 1
    
    distance_to_target = _calculate_distance(player_x, player_y, target_waypoint[0], target_waypoint[1])
    print(f"[LONG_DISTANCE] Moving to waypoint {target_index + 1}/{len(_long_distance_path_cache)}: {target_waypoint} (distance: {distance_to_target:.1f} tiles)")
    
    return [target_waypoint]


def clear_long_distance_cache():
    """Clear the long-distance path cache."""
    global _long_distance_path_cache, _long_distance_destination, _long_distance_waypoint_index
    _long_distance_path_cache = None
    _long_distance_destination = None
    _long_distance_waypoint_index = 0
    print("[LONG_DISTANCE_CACHE] Cache cleared")


def is_long_distance_path_complete():
    """Check if we've reached the end of the cached long-distance path."""
    global _long_distance_path_cache, _long_distance_waypoint_index
    if _long_distance_path_cache is None:
        return True
    return _long_distance_waypoint_index >= len(_long_distance_path_cache)


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


def _handle_door_opening(door_plan: dict, fresh_payload: dict, ui) -> bool:
    """
    Handle door opening logic with retry mechanism and recently traversed door tracking.
    
    Args:
        door_plan: Door plan containing door information
        fresh_payload: Current game state payload
        ui: UI instance for dispatching actions
        
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
    
    # Door is closed, try to open it with retry logic
    door_opened = not door_plan["door"].get("closed")
    door_retry_count = 0
    max_door_retries = 3

    while not door_opened and door_retry_count < max_door_retries:
        # Get fresh door data for each retry attempt
        fresh_payload = get_payload()
        wps, dbg_path = ipc_path(fresh_payload, goal=(door_x, door_y))
        fresh_door_plan = _first_blocking_door_from_waypoints(wps)
        
        if not fresh_door_plan:
            # No door found, it might be open already
            door_opened = True
            break
            
        # Use fresh door data for clicking
        d = (fresh_door_plan.get("door") or {})
        b = d.get("bounds")
        if isinstance(b, dict):
            step = emit({
                "action": "open-door",
                "click": {"type": "rect-center"},
                "target": {"domain": "object", "name": d.get("name") or "Door", "bounds": b},
                "postconditions": [f"doorOpen@{door_x},{door_y} == true"],
                "timeout_ms": 2000
            })
        else:
            # Fallback to canvas point if bounds not available
            c = d.get("canvas")
            if not (isinstance(c, dict) and isinstance(c.get("x"), (int, float)) and isinstance(c.get("y"), (int, float))):
                break  # Can't get coordinates, skip door opening
            step = emit({
                "action": "open-door",
                "click": {"type": "point", "x": int(c["x"]), "y": int(c["y"])},
                "target": {"domain": "object", "name": d.get("name") or "Door"},
                "postconditions": [f"doorOpen@{door_x},{door_y} == true"],
                "timeout_ms": 2000
            })

        door_result = dispatch_with_camera(step, ui=get_ui(), payload=get_payload(), aim_ms=420)
        if door_result:
            # Wait up to 5 seconds for door to open
            door_wait_start = time.time()
            while (time.time() - door_wait_start) * 1000 < 5000:
                fresh_payload = get_payload()
                if _door_is_open(fresh_door_plan, fresh_payload):
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


def _door_is_open(door_plan: dict, payload: dict) -> bool:
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
    from ..helpers.ipc import ipc_send
    resp = ipc_send({
        "cmd": "door_state",
        "door_x": door_x,
        "door_y": door_y,
        "door_p": door_p
    }, payload)
    
    if not resp or not resp.get("ok"):
        return True  # If IPC fails, assume door is open
    
    # If wall_object is null, door doesn't exist (is open)
    # If wall_object exists, door is still there (is closed)
    og_door_id = door_plan['door'].get('id')
    new_door_id = resp['wall_object'].get('id')
    if not og_door_id == new_door_id:
        return True  # Door ID changed, it's open
    
    return False  # Door still exists with same ID, it's closed


def go_to(rect_or_key: str | tuple | list, payload: dict | None = None, ui=None, center: bool = False) -> dict | None:
    """
    One movement click toward an area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        payload: Game state payload
        ui: UI instance for dispatching actions
        center: If True, go to center of region instead of stopping at boundary
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # accept key or explicit (minX, maxX, minY, maxY)
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None
        
    # Get fresh payload and player position
    fresh_payload = get_payload()
    player_x, player_y = _get_player_position(fresh_payload)
    
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        print(f"[GO_TO] Could not get player position")
        return None
    
    # Calculate distance to destination
    min_x, max_x, min_y, max_y = rect
    dest_center_x = (min_x + max_x) // 2
    dest_center_y = (min_y + max_y) // 2
    distance = _calculate_distance(player_x, player_y, dest_center_x, dest_center_y)
    
    print(f"[GO_TO] Distance to {rect_key}: {distance}")
    
    if distance > 50:
        # LONG DISTANCE: Get cached path and click ~25 tiles away
        print(f"[GO_TO] Using long-distance pathfinding")
        
        # Get next waypoint from path
        waypoint_batch = _get_next_long_distance_waypoints(rect_key, fresh_payload, 1)
        if not waypoint_batch:
            print(f"[GO_TO] Could not get long-distance waypoint, falling back to short-distance")
            # Fallback to short-distance pathfinding by treating as short distance
            distance = 25  # Force short-distance pathfinding
        else:
            # Use the single waypoint from the path
            target_waypoint = waypoint_batch[0]
            print(f"[GO_TO] Moving to waypoint: {target_waypoint}")
            
            # Project waypoint to screen coordinates
            waypoint_wps, _ = ipc_path(fresh_payload, rect=(target_waypoint[0]-1, target_waypoint[0]+1, target_waypoint[1]-1, target_waypoint[1]+1))
            if waypoint_wps:
                waypoint_proj, _ = ipc_project_many(fresh_payload, waypoint_wps)
                waypoint_proj = _merge_door_into_projection(waypoint_wps, waypoint_proj)
                waypoint_usable = [p for p in waypoint_proj if isinstance(p, dict) and p.get("canvas")]
                if waypoint_usable:
                    chosen = waypoint_usable[-1]
                    world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
                    
                    # Check for doors on the long-distance path and handle them
                    gx = chosen["world"].get("x")
                    gy = chosen["world"].get("y")
                    
                    if isinstance(gx, int) and isinstance(gy, int):
                        wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
                        door_plan = _first_blocking_door_from_waypoints(wps)
                        if door_plan:
                            if not _handle_door_opening(door_plan, fresh_payload, ui):
                                return None  # Door opening failed
                else:
                    world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
            else:
                world_coords = {"x": target_waypoint[0], "y": target_waypoint[1], "p": 0}
    
    else:
        # SHORT DISTANCE: Use standard local pathing
        print(f"[GO_TO] Using local pathfinding")
        
        wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))
        if not wps:
            print(f"[GO_TO] No waypoints found for {rect_key}")
            return None
        
        proj, dbg_proj = ipc_project_many(fresh_payload, wps)
        proj = _merge_door_into_projection(wps, proj)
        
        usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
        if not usable:
            print(f"[GO_TO] No usable waypoints after projection")
            return None
        
        # Check for doors on the path and handle them
        if usable:
            last_waypoint = usable[-1]
            gx = last_waypoint["world"].get("x")
            gy = last_waypoint["world"].get("y")
            
            if isinstance(gx, int) and isinstance(gy, int):
                wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
                door_plan = _first_blocking_door_from_waypoints(wps)
                if door_plan:
                    if not _handle_door_opening(door_plan, fresh_payload, ui):
                        return None  # Door opening failed
        
        # Pick from the last 3 waypoints (or all if less than 3)
        if len(usable) <= 3:
            chosen = random.choice(usable)
        else:
            chosen = random.choice(usable[-3:])
        
        world_coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
    
    from ..services.click_with_camera import click_ground_with_camera
    return click_ground_with_camera(
        world_coords=world_coords,
        description=f"Move toward {rect_key}",
        ui=ui,
        payload=fresh_payload,
        aim_ms=700
    )


def _get_player_position(payload: dict) -> tuple[int | None, int | None]:
    """Get current player position from payload."""
    player = payload.get("player", {})
    return player.get("worldX"), player.get("worldY")


def _calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


def _should_click_next_waypoint(current_waypoint: dict, player_x: int, player_y: int, 
                               proximity_threshold: int = 2) -> bool:
    """
    Determine if we should click the next waypoint based on player proximity.
    
    Args:
        current_waypoint: Current waypoint being approached
        player_x: Player's current X coordinate
        player_y: Player's current Y coordinate
        proximity_threshold: Distance threshold to trigger next waypoint click
        
    Returns:
        True if should click next waypoint, False otherwise
    """
    if not current_waypoint:
        return False
    
    waypoint_x = current_waypoint.get("world", {}).get("x", current_waypoint.get("x"))
    waypoint_y = current_waypoint.get("world", {}).get("y", current_waypoint.get("y"))
    
    if not isinstance(waypoint_x, int) or not isinstance(waypoint_y, int):
        return False
    
    distance = _calculate_distance(player_x, player_y, waypoint_x, waypoint_y)
    return distance <= proximity_threshold


def _is_blocking_door(door: dict) -> bool:
    if not isinstance(door, dict):
        return False
    if not door.get("present"):
        return False
    return door.get("closed", True) is True


def _pick_click_from_door(door: dict):
    if not isinstance(door, dict):
        return None, None
    c = door.get("canvas")
    if not isinstance(c, dict):
        return None, None
    tc = door.get("tileCanvas")
    if not isinstance(tc, dict):
        return None, None
    b = door.get("bounds")
    if not isinstance(b, dict):
        return None, None
    return {"type": "point", "x": int(c["x"]), "y": int(c["y"])}, {"bounds": b}


def _first_blocking_door(proj_rows: list[dict], up_to_index: int | None = None) -> dict | None:
    """
    Scan projected rows (which now include door metadata) and return a click plan
    for the earliest blocking door. Treat missing 'closed' as blocking.
    """
    limit = len(proj_rows) if up_to_index is None else max(0, min(up_to_index + 1, len(proj_rows)))

    for i in range(limit):
        row = proj_rows[i] or {}
        door = (row.get("door") or {})
        if not door.get("present"):
            continue

        closed = door.get("closed")
        if closed is False:
            continue  # already open

        # Prefer rect-center if bounds exist, else use canvas point
        if isinstance(door.get("bounds"), dict):
            click = {"type": "rect-center"}
            target_anchor = {"bounds": door["bounds"]}
        elif isinstance(door.get("canvas"), dict) and \
             isinstance(door["canvas"].get("x"), (int, float)) and isinstance(door["canvas"].get("y"), (int, float)):
            click = {"type": "point", "x": int(door["canvas"]["x"]), "y": int(door["canvas"]["y"])}
            target_anchor = {}
        else:
            # no geometry → skip; we'll just walk this tick
            continue

        wx = row.get("world", {}).get("x", row.get("x"))
        wy = row.get("world", {}).get("y", row.get("y"))
        wp = row.get("world", {}).get("p", row.get("p"))

        name = door.get("name") or "Door"
        ident = {"id": door.get("id"), "name": name, "world": {"x": wx, "y": wy, "p": wp}}

        return {
            "index": i,
            "click": click,
            "target": {"domain": "object", "name": name, **target_anchor},
            # Your executor already supports postconditions (used in open_bank)
            # Wire a simple predicate your state loop can satisfy when the door flips open.
            "postconditions": [f"doorOpen@{wx},{wy} == true"],
            "timeout_ms": 2000
        }

    return None


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


def go_to_with_proximity_awareness(rect_or_key: str | tuple | list, payload: dict | None = None, 
                                 ui=None, proximity_threshold: int = 2) -> dict | None:
    """
    Enhanced go_to with proximity awareness to prevent getting stuck.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        payload: Game state payload
        ui: UI instance for dispatching actions
        proximity_threshold: Distance threshold to trigger next waypoint click
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get current player position
    player_x, player_y = _get_player_position(payload)
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        print(f"[DEBUG] Could not get player position, using basic go_to")
        return go_to(rect_or_key, payload, ui)
    
    # Check if we're already close to the target
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        min_x, max_x, min_y, max_y = rect_or_key
        if min_x <= player_x <= max_x and min_y <= player_y <= max_y:
            print(f"[DEBUG] Player already in target area")
            return None
    else:
        rect = get_nav_rect(str(rect_or_key))
        if isinstance(rect, (tuple, list)) and len(rect) == 4:
            min_x, max_x, min_y, max_y = rect
            if min_x <= player_x <= max_x and min_y <= player_y <= max_y:
                print(f"[DEBUG] Player already in target area")
        return None
    
    # Use regular go_to
    return go_to(rect_or_key, payload, ui)


def go_to_closest_bank(payload: dict | None = None, ui=None) -> dict | None:
    """Go to the closest bank."""
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    bank_key = closest_bank_key(payload)
    if bank_key:
        return go_to(bank_key, payload, ui)


def go_to_ge(payload: dict | None = None, ui=None) -> dict | None:
    """Go to the Grand Exchange."""
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    return go_to("GE", payload, ui)


def in_area(area_key: str, payload: dict | None = None) -> bool:
    """Check if player is in the specified area."""
    if payload is None:
        payload = get_payload()
    
    # Get player position
    player_x, player_y = _get_player_position(payload)
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Get area rectangle
    rect = get_nav_rect(area_key)
    if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
        return False
    
    min_x, max_x, min_y, max_y = rect
    return min_x <= player_x <= max_x and min_y <= player_y <= max_y


def get_long_distance_waypoints(destination_key: str, payload: dict | None = None) -> list[tuple[int, int]] | None:
    """
    Generate waypoints for long-distance travel using the collision map pathfinder.
    This calls the pathfinder.py directly to get a complete path.
    """
    try:
        # Import pathfinder functions
        from ..pathfinder import (
            load_collision_data, 
            get_walkable_tiles, 
            astar_pathfinding, 
            get_current_player_position
        )
        from ..helpers.navigation import get_nav_rect
        
        print(f"[LONG_DISTANCE] Generating path to {destination_key}")
        
        # Get current player position
        current_pos = get_current_player_position()
        if not current_pos:
            print("[LONG_DISTANCE] Could not get current player position")
            return None
        
        # Get destination coordinates
        dest_rect = get_nav_rect(destination_key)
        if not (isinstance(dest_rect, (tuple, list)) and len(dest_rect) == 4):
            print(f"[LONG_DISTANCE] Invalid destination area: {destination_key}")
            return None
        
        min_x, max_x, min_y, max_y = dest_rect
        dest_center_x = (min_x + max_x) // 2
        dest_center_y = (min_y + max_y) // 2
        destination = (dest_center_x, dest_center_y)
        
        print(f"[LONG_DISTANCE] From {current_pos} to {destination}")
        
        # Load collision data
        collision_data = load_collision_data()
        if not collision_data:
            print("[LONG_DISTANCE] Could not load collision data")
            return None
        
        # Get walkable tiles
        walkable_tiles, blocked_tiles = get_walkable_tiles(collision_data)
        print(f"[LONG_DISTANCE] Walkable tiles: {len(walkable_tiles)}")
        
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
                if (x, y) in walkable_tiles:
                    dest_walkable.append((x, y))
        
        if not current_walkable:
            print("[LONG_DISTANCE] No walkable tiles found near current position")
            return None
        
        if not dest_walkable:
            print("[LONG_DISTANCE] No walkable tiles found near destination")
            return None
        
        # Find closest walkable tiles
        start = find_closest_walkable(current_pos, current_walkable)
        goal = find_closest_walkable(destination, dest_walkable)
        
        print(f"[LONG_DISTANCE] Start: {start}, Goal: {goal}")
        
        # Use A* pathfinding directly
        print("[LONG_DISTANCE] Using A* pathfinding...")
        path = astar_pathfinding(start, goal, walkable_tiles)
        
        if not path:
            print("[LONG_DISTANCE] Pathfinding failed")
            return None
        
        print(f"[LONG_DISTANCE] Path found with {len(path)} waypoints")
        
        # Smooth the path to reduce waypoint density
        smoothed_path = _smooth_path(path)
        print(f"[LONG_DISTANCE] Smoothed path has {len(smoothed_path)} waypoints")
        
        return smoothed_path
        
    except Exception as e:
        print(f"[LONG_DISTANCE] Error generating waypoints: {e}")
        return None


def go_to_and_find_npc(rect_or_key: str | tuple | list, npc_name: str, payload: dict | None = None, 
                      ui=None, center: bool = False, search_radius: int = 10, 
                      max_search_attempts: int = 20) -> dict | None:
    """
    Go to an area and then search for an NPC by traversing around the area.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        npc_name: Name of the NPC to find (partial match, case-insensitive)
        payload: Game state payload
        ui: UI instance for dispatching actions
        center: If True, go to center of region instead of stopping at boundary
        search_radius: Radius around the destination to search for NPC
        max_search_attempts: Maximum number of search attempts before giving up
        
    Returns:
        NPC data if found, None otherwise
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    print(f"[GO_TO_AND_FIND_NPC] Going to {rect_or_key} to find {npc_name}")
    
    # First, go to the destination area
    go_to_result = go_to(rect_or_key, payload, ui, center)
    if go_to_result is None:
        print(f"[GO_TO_AND_FIND_NPC] Failed to go to {rect_or_key}")
        return None
    
    # Wait a moment for movement to start
    time.sleep(1.0)
    
    # Import NPC functions
    from .npc import closest_npc_by_name
    from ..helpers.navigation import get_nav_rect
    
    # Get the destination area bounds
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
    else:
        rect = get_nav_rect(str(rect_or_key))
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            print(f"[GO_TO_AND_FIND_NPC] Invalid area: {rect_or_key}")
            return None
    
    min_x, max_x, min_y, max_y = rect
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    print(f"[GO_TO_AND_FIND_NPC] Searching for {npc_name} in area {rect}")
    
    # Search for the NPC by moving around the area
    for attempt in range(max_search_attempts):
        # Get fresh payload
        fresh_payload = get_payload()
        if fresh_payload is None:
            continue
            
        # Check if NPC is already visible
        npc = closest_npc_by_name(npc_name, fresh_payload)
        if npc is not None:
            print(f"[GO_TO_AND_FIND_NPC] Found {npc_name} on attempt {attempt + 1}")
            return npc
        
        # Get current player position
        player_x, player_y = _get_player_position(fresh_payload)
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            print(f"[GO_TO_AND_FIND_NPC] Could not get player position on attempt {attempt + 1}")
            time.sleep(0.5)
            continue
        
        # Check if we're still in the target area
        if not (min_x <= player_x <= max_x and min_y <= player_y <= max_y):
            print(f"[GO_TO_AND_FIND_NPC] Player moved outside target area, going back")
            go_to(rect_or_key, fresh_payload, ui, center)
            time.sleep(1.0)
            continue
        
        # Generate a random search point within the search radius
        import random
        search_x = center_x + random.randint(-search_radius, search_radius)
        search_y = center_y + random.randint(-search_radius, search_radius)
        
        # Clamp to area bounds
        search_x = max(min_x, min(max_x, search_x))
        search_y = max(min_y, min(max_y, search_y))
        
        # Calculate distance to search point
        distance_to_search = _calculate_distance(player_x, player_y, search_x, search_y)
        
        # Only move if we're not already close to the search point
        if distance_to_search > 3:
            print(f"[GO_TO_AND_FIND_NPC] Attempt {attempt + 1}: Moving to search point ({search_x}, {search_y})")
            
            # Create a small rect around the search point
            search_rect = (search_x - 1, search_x + 1, search_y - 1, search_y + 1)
            go_to(search_rect, fresh_payload, ui, center=True)
            
            # Wait for movement
            time.sleep(1.5)
        else:
            print(f"[GO_TO_AND_FIND_NPC] Attempt {attempt + 1}: Already close to search point, waiting")
            time.sleep(0.5)
    
    print(f"[GO_TO_AND_FIND_NPC] Could not find {npc_name} after {max_search_attempts} attempts")
    return None
