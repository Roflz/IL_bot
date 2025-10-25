from typing import Optional, List, Dict, Any

from ilbot.ui.simple_recorder.actions.travel import _first_blocking_door_from_waypoints
from ilbot.ui.simple_recorder.constants import BANK_REGIONS, REGIONS
from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy
from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc
from ilbot.ui.simple_recorder.helpers.utils import rect_beta_xy
from ilbot.ui.simple_recorder.services.click_with_camera import click_object_with_camera


def _get_path_distance_to_object(obj: Dict[str, Any]) -> int:
    """
    Get the path distance (number of waypoints) to an object.
    
    Args:
        obj: Object data containing world coordinates
        
    Returns:
        Number of waypoints in path to object, or 999 if pathfinding fails
    """
    world_coords = obj.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        return 999
    
    try:
        wps, _ = ipc.path(goal=(world_coords["x"], world_coords["y"]), visualize=False)
        return len(wps) if wps else 999
    except Exception:
        return 999


def _get_distance_to_object(obj: Dict[str, Any]) -> float:
    """
    Get the straight-line distance to an object.
    
    Args:
        obj: Object data containing world coordinates
        
    Returns:
        Straight-line distance to object, or 999.0 if coordinates are invalid
    """
    world_coords = obj.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        return 999.0
    
    # Get player position
    try:
        player_resp = ipc.get_player()
        if not player_resp or not player_resp.get("ok"):
            return 999.0
        
        player_data = player_resp.get("player", {})
        player_x = player_data.get("x")
        player_y = player_data.get("y")
        
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return 999.0
        
        # Calculate straight-line distance
        dx = world_coords["x"] - player_x
        dy = world_coords["y"] - player_y
        distance = (dx * dx + dy * dy) ** 0.5
        
        return distance
    except Exception:
        return 999.0


def _find_closest_object_by_distance(name: str, types: List[str] | None = None, max_distance: float = None, max_objects: int = None, exact_match: bool = False) -> Dict[str, Any] | None:
    """
    Find the closest object by straight-line distance.
    Uses the efficient find_object command which already calculates distance and returns the closest match.
    
    Args:
        name: Name of the object to search for (partial match)
        types: Object types to search for
        max_distance: If specified, return first object found under this distance
        max_objects: Maximum number of objects to check for distance (for performance)
        exact_match: If True, only matches objects with exact name match
        
    Returns:
        Closest object by straight-line distance, or None if not found
    """
    if not name or not str(name).strip():
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    obj_resp = ipc.find_object(name, types=types, exact_match=exact_match)
    
    if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
        return None
    
    obj = obj_resp.get("object")
    if not obj:
        return None
    
    return obj


def _find_closest_object_in_area(name: str, area: str | tuple, types: List[str] | None = None) -> Dict[str, Any] | None:
    """
    Find the closest object within a specific area.
    Uses the efficient find_object_in_area command which only scans the specified area.
    
    Args:
        name: Name of the object to search for (partial match)
        area: Area name from constants.py (e.g., "FALADOR_BANK") or tuple (min_x, max_x, min_y, max_y)
        types: Object types to search for
        
    Returns:
        Closest object within the area, or None if not found
    """
    if not name or not str(name).strip():
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    # Resolve area coordinates
    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            print(f"[ERROR] Unknown area: {area}. Available areas: {list(BANK_REGIONS.keys()) + list(REGIONS.keys())}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format. Use area name or tuple (min_x, max_x, min_y, max_y)")
        return None

    # Use find_object_in_area which only scans the specified area
    # This is much faster than get_objects + area filtering
    obj_resp = ipc.find_object_in_area(name, min_x, max_x, min_y, max_y, types=types)
    
    if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
        return None
    
    obj = obj_resp.get("object")
    if not obj:
        return None
    
    return obj


def _find_closest_object_by_path(name: str, types: List[str] | None = None, max_path_distance: int = None, max_objects: int = None) -> Dict[str, Any] | None:
    """
    Find the closest object by path distance (number of waypoints).
    Uses the efficient find_object command instead of getting all objects.
    
    Args:
        name: Name of the object to search for (partial match)
        types: Object types to search for
        max_path_distance: If specified, return first object found under this path distance
        max_objects: Maximum number of objects to check for path distance (for performance)
        
    Returns:
        Closest object by path distance, or None if not found
    """
    if not name or not str(name).strip():
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    # Get all matching objects within radius
    print(f"[FIND_OBJECT] Searching for '{name}' with types {types} and radius 14")
    obj_resp = ipc.get_objects(name, types=types, radius=26)  # Increased radius to find more objects
    
    if not obj_resp or not obj_resp.get("ok"):
        # Fallback to original find_object command
        print(f"[FIND_OBJECT] get_objects failed, falling back to find_object")
        return ipc.find_object(name, types)
    
    objects = obj_resp.get("objects", [])
    if not objects:
        # Fallback to original find_object command
        print(f"[FIND_OBJECT] No objects found, falling back to find_object")
        return ipc.find_object(name, types)
    
    # Deduplicate objects by world coordinates (fix for Java bug)
    seen_locations = set()
    unique_objects = []
    for obj in objects:
        world = obj.get('world', {})
        location_key = (world.get('x'), world.get('y'), world.get('p'))
        if location_key not in seen_locations:
            seen_locations.add(location_key)
            unique_objects.append(obj)
    
    objects = unique_objects
    print(f"[FIND_OBJECT] Deduplicated from {len(obj_resp.get('objects', []))} to {len(objects)} unique objects")
    
    # Debug: Print first few objects and their distances
    print(f"[FIND_OBJECT] Found {len(objects)} objects, first 5 distances:")
    for i, obj in enumerate(objects[:5]):
        world = obj.get('world', {})
        print(f"  {i}: {obj.get('name', 'Unknown')} - distance: {obj.get('distance', 'Unknown')} - world: {world}")
    
    # Limit the number of objects to check for performance
    if max_objects is not None and len(objects) > max_objects:
        objects = objects[:max_objects]
        print(f"[FIND_OBJECT] Limited to first {max_objects} objects for performance")
    
    # Find the object with the shortest path distance
    best_object = ipc.find_object(name, types).get("object")
    shortest_path_distance = _get_path_distance_to_object(best_object)
    
    for obj in objects:
        path_distance = _get_path_distance_to_object(obj)
        if path_distance is not None:
            # If max_path_distance is specified and this object is under it, return immediately
            if max_path_distance is not None and path_distance <= max_path_distance:
                print(f"[FIND_OBJECT] Found '{obj.get('name')}' at path distance {path_distance} waypoints (under threshold {max_path_distance})")
                return obj
            
            # Track the best object overall
            if path_distance < shortest_path_distance:
                shortest_path_distance = path_distance
                best_object = obj
    
    if best_object:
        print(f"[FIND_OBJECT] Found '{best_object.get('name')}' at path distance {shortest_path_distance} waypoints")
        return best_object
    
    # Fallback to original find_object command if no path-based object found
    print(f"[FIND_OBJECT] No path-based object found, falling back to find_object")
    return ipc.find_object(name, types)


def object_exists(
    name: str,
    radius: int = 26,
    types: List[str] | None = None
) -> bool:
    """
    Check if a game object exists by (partial) name.
    Uses the efficient find_object command.
    
    Args:
        name: Name of the object to search for (partial match)
        radius: Search radius for IPC fallback (default: 26)
        types: Object types to search for (default: ["GAME"])
        
    Returns:
        True if object exists, False otherwise
    """
    if not name or not str(name).strip():
        return False
    if types is None:
        types = ["GAME"]
    
    # Use the efficient find_object command
    obj_resp = ipc.find_object(name, types=types)
    if obj_resp and obj_resp.get("ok") and obj_resp.get("found", False):
        return obj_resp['object'].get('distance') < radius


def click_object_closest_by_path_distance(name: str, action: str, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> Optional[dict]:
    """
    Click a specific action on an object by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    Uses path-based distance calculation to select the closest object.
    If a CLOSED door lies on the path to the object, click the earliest blocking door first.
    
    Args:
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by path distance
        fresh_obj = _find_closest_object_by_path(name, types=["GAME"])
        
        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue  # Try next attempt
        
        # Calculate path distance for logging
        path_distance = _get_path_distance_to_object(fresh_obj)
        print(f"[OBJECT_ACTION] Found object: {fresh_obj.get('name')} at path distance {path_distance} waypoints")
        print(f"[OBJECT_ACTION] Object actions: {fresh_obj.get('actions')}")
        
        # Find action index
        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None
        
        idx = action_index(fresh_obj.get("actions"), action.lower())
        print(f"[OBJECT_ACTION] Action '{action}' found at index: {idx}")
        if idx is None:
            print(f"[OBJECT_ACTION] Action '{action}' not found in object actions")
            continue  # Try next attempt

        # 1) Check for doors on the path to the object
        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object action with pathing logic
        obj_name = fresh_obj.get("name") or name
        
        # Get world coordinates for the object
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            print(f"[OBJECT_ACTION] No valid world coordinates for object, trying next attempt")
            continue
        
        # Get click coordinates and rectangle from target object
        rect = unwrap_rect(fresh_obj.get("clickbox")) or unwrap_rect(fresh_obj.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(fresh_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_obj.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_obj["canvas"]["x"]), int(fresh_obj["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue  # Try next attempt
        
        if idx == 0:
            # Desired action is default → use click_object_with_camera with no action
            print(f"[OBJECT_ACTION] Using left-click for action at index 0")
            result = click_object_with_camera(
                object_name=obj_name,
                action=None,
                world_coords=world_coords,
                click_coords=click_coords,
                click_rect=click_rect,
                aim_ms=420,
                exact_match=exact_match_target_and_action
            )
        else:
            # Need context menu → use click_object_with_camera with action and index
            print(f"[OBJECT_ACTION] Using context menu for action at index {idx}")
            result = click_object_with_camera(
                object_name=obj_name,
                action=action,
                world_coords=world_coords,
                click_coords=click_coords,
                click_rect=click_rect,
                aim_ms=420,
                exact_match=exact_match_target_and_action
            )
        
        print(f"[OBJECT_ACTION] Click result: {result}")
        
        if result:
            return result

    return None


def click_object_closest_by_distance(name: str, action: str, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> Optional[dict]:
    """
    Click a specific action on an object by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    Uses straight-line distance calculation to select the closest object.
    If a CLOSED door lies on the path to the object, click the earliest blocking door first.
    
    Args:
        name: Name of the object to search for
        action: Action to perform on the object
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
    """
    max_retries = 3

    for attempt in range(max_retries):
        # Find closest object by distance
        fresh_obj = _find_closest_object_by_distance(name, types=["GAME"], exact_match=exact_match_object)

        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue  # Try next attempt

        # Find action index
        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        # 1) Check for doors on the path to the object
        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object action with pathing logic
        obj_name = fresh_obj.get("name") or name

        # Get world coordinates for the object
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            print(f"[OBJECT_ACTION] No valid world coordinates for object, trying next attempt")
            continue

        # Get click coordinates and rectangle from target object
        rect = unwrap_rect(fresh_obj.get("clickbox")) or unwrap_rect(fresh_obj.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(fresh_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(
                fresh_obj.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_obj["canvas"]["x"]), int(fresh_obj["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue  # Try next attempt

        result = click_object_with_camera(
            object_name=obj_name,
            action=action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action
        )

        print(f"[OBJECT_ACTION] Click result: {result}")

        if result:
            return result

    return None


def click_object_closest_by_path_simple(name: str, prefer_action: str | None = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
    """
    Click a game object by (partial) name using path-based distance calculation.
    Simplified version without door handling.
    Selects the closest object by path distance (number of waypoints) rather than absolute distance.
    
    Args:
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
    """
    if not name or not str(name).strip():
        return None

    want = str(name).strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by path distance
        target = _find_closest_object_by_path(want, types=["GAME"], max_path_distance=8)
        
        if not target:
            print(f"[DEBUG] Object '{want}' not found")
            continue  # Try next attempt
        
        # Calculate path distance for logging
        path_distance = _get_path_distance_to_object(target)
        print(f"[DEBUG] Found object: {target.get('name')} at path distance {path_distance} waypoints")

        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        # Click the object (no door handling)
        idx = action_index(target.get("actions"), want_action) if want_action else None
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        # Get click coordinates and rectangle from target object
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue  # Try next attempt

        # Use simplified click with camera function
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action
        )

    return None


def click_object_in_area_simple(name: str, area: str | tuple, action: str = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
    """
    Click an object within a specific area (simple version without door handling).
    Uses the efficient find_object_in_area command.
    
    Args:
        name: Name of the object to search for
        area: Area name from constants.py (e.g., "FALADOR_BANK") or tuple (min_x, max_x, min_y, max_y)
        action: Action to perform on the object (optional)
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
        
    Returns:
        Click result or None if not found
    """
    if not name or not str(name).strip():
        return None

    # Resolve area coordinates
    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            print(f"[ERROR] Unknown area: {area}. Available areas: {list(BANK_REGIONS.keys()) + list(REGIONS.keys())}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format. Use area name or tuple (min_x, max_x, min_y, max_y)")
        return None

    # Find the closest object in the specified area
    target = _find_closest_object_in_area(name, area, types=["GAME"])
    
    if not target:
        print(f"[DEBUG] Object '{name}' not found in area ({min_x},{min_y}) to ({max_x},{max_y})")
        return None
    
    print(f"[DEBUG] Found object: {target.get('name')} in area")

    # Get world coordinates for the object
    world_coords = target.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        print(f"[DEBUG] No valid world coordinates for object")
        return None
    
    # Get click coordinates and rectangle from target object
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    if rect:
        cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                               rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
        click_coords = {"x": cx, "y": cy}
        click_rect = rect
    elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
        cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
        click_coords = {"x": cx, "y": cy}
        click_rect = None
    else:
        print(f"[DEBUG] No valid click coordinates for object")
        return None

    # Use click_object_with_camera function (simple version without door handling)
    return click_object_with_camera(
        object_name=name,
        action=action,
        world_coords=world_coords,
        click_coords=click_coords,
        click_rect=click_rect,
        aim_ms=420,
        exact_match=exact_match_target_and_action
    )


def click_object_in_area(name: str, area: str | tuple, action: str = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
    """
    Click an object within a specific area with door detection logic.
    Uses the efficient find_object_in_area command.
    
    Args:
        name: Name of the object to search for
        area: Area name from constants.py (e.g., "FALADOR_BANK") or tuple (min_x, max_x, min_y, max_y)
        action: Action to perform on the object (optional)
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
        
    Returns:
        Click result or None if not found
    """
    if not name or not str(name).strip():
        return None

    # Resolve area coordinates
    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            print(f"[ERROR] Unknown area: {area}. Available areas: {list(BANK_REGIONS.keys()) + list(REGIONS.keys())}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format. Use area name or tuple (min_x, max_x, min_y, max_y)")
        return None

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find the closest object in the specified area
        target = _find_closest_object_in_area(name, area, types=["GAME"])
        
        if not target:
            print(f"[OBJECT_ACTION] Object '{name}' not found in area ({min_x},{min_y}) to ({max_x},{max_y})")
            continue  # Try next attempt
        
        print(f"[OBJECT_ACTION] Found object: {target.get('name')} in area")
        print(f"[OBJECT_ACTION] Object actions: {target.get('actions')}")
        
        # Find action index
        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None
        
        idx = action_index(target.get("actions"), action.lower()) if action else None
        print(f"[OBJECT_ACTION] Action '{action}' found at index: {idx}")
        if action and idx is None:
            print(f"[OBJECT_ACTION] Action '{action}' not found in object actions")
            continue  # Try next attempt

        # 1) Check for doors on the path to the object
        gx, gy = target.get("world", {}).get("x"), target.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object action with pathing logic
        obj_name = target.get("name") or name
        
        # Get world coordinates for the object
        world_coords = target.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            print(f"[OBJECT_ACTION] No valid world coordinates for object, trying next attempt")
            continue
        
        # Get click coordinates and rectangle from target object
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue  # Try next attempt

        result = click_object_with_camera(
            object_name=obj_name,
            action=action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action
        )
        
        print(f"[OBJECT_ACTION] Click result: {result}")
        
        if result:
            return result

    return None


def click_object_closest_by_distance_simple(name: str, prefer_action: str | None = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
    """
    Click a game object by (partial) name using straight-line distance calculation.
    Simplified version without door handling.
    Selects the closest object by straight-line distance rather than path distance.
    
    Args:
        exact_match_object: If True, only matches objects with exact name match
        exact_match_target_and_action: If True, uses exact matching for target and action verification
    """
    if not name or not str(name).strip():
        return None

    want = str(name).strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by distance
        target = _find_closest_object_by_distance(want, types=["GAME"], exact_match=exact_match_object)
        
        if not target:
            print(f"[DEBUG] Object '{want}' not found")
            continue  # Try next attempt

        # Click the object (no door handling)
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        # Get click coordinates and rectangle from target object
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue  # Try next attempt

        # Use simplified click with camera function
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action
        )

    return None
