from typing import Optional, List, Dict, Any
from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc
from ilbot.ui.simple_recorder.helpers.navigation import _first_blocking_door_from_waypoints
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
        wps, _ = ipc.path(goal=(world_coords["x"], world_coords["y"]))
        return len(wps) if wps else 999
    except Exception:
        return 999


def _find_closest_object_by_path(name: str, types: List[str] | None = None) -> Dict[str, Any] | None:
    """
    Find the closest object by path distance (number of waypoints).
    Uses the efficient find_object command instead of getting all objects.
    
    Args:
        name: Name of the object to search for (partial match)
        types: Object types to search for
        
    Returns:
        Closest object by path distance, or None if not found
    """
    if not name or not str(name).strip():
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    # Use the efficient find_object command instead of get_closest_objects
    obj_resp = ipc.find_object(name, types=types)
    
    if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
        return None
    
    # find_object already returns the best match by distance, but we can still
    # calculate path distance for logging and potential future optimizations
    target = obj_resp.get("object")
    if not target:
        return None
    
    # Calculate path distance for logging
    path_distance = _get_path_distance_to_object(target)
    print(f"[FIND_OBJECT] Found '{target.get('name')}' at path distance {path_distance} waypoints")
    
    return target


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
    return obj_resp and obj_resp.get("ok") and obj_resp.get("found", False)


def click(
    name: str,
    prefer_action: str | None = None,
    ui=None,
) -> dict | None:
    """
    Click a game object by (partial) name using path-based distance calculation.
    Selects the closest object by path distance (number of waypoints) rather than absolute distance.
    If a CLOSED door lies on the path to the object, click the earliest blocking door first.
    """
    if not name or not str(name).strip():
        return None

    want = str(name).strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by path distance
        target = _find_closest_object_by_path(want, types=["GAME"])
        
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

        # 1) Check for doors on the path to the object
        gx, gy = target["world"].get("x"), target["world"].get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            from ..helpers.navigation import _first_blocking_door_from_waypoints
            from .travel import _handle_door_opening
            
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object
        idx = action_index(target.get("actions"), want_action) if want_action else None
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        # Use centralized click with camera function
        from ..services.click_with_camera import click_object_with_camera
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            world_coords=world_coords,
            aim_ms=420
        )

    return None


def click_object_action(name: str, action: str) -> Optional[dict]:
    """
    Click a specific action on an object by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    Uses path-based distance calculation to select the closest object.
    If a CLOSED door lies on the path to the object, click the earliest blocking door first.
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
                if not _handle_door_opening(door_plan):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object action with pathing logic
        obj_name = fresh_obj.get("name") or name
        
        # Get world coordinates for the object
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            print(f"[OBJECT_ACTION] No valid world coordinates for object, trying next attempt")
            continue
        
        if idx == 0:
            # Desired action is default → use click_object_with_camera with no action
            print(f"[OBJECT_ACTION] Using left-click for action at index 0")
            result = click_object_with_camera(
                object_name=obj_name,
                action=None,
                world_coords=world_coords,
                aim_ms=420
            )
        else:
            # Need context menu → use click_object_with_camera with action and index
            print(f"[OBJECT_ACTION] Using context menu for action at index {idx}")
            result = click_object_with_camera(
                object_name=obj_name,
                action=action,
                world_coords=world_coords,
                aim_ms=420
            )
        
        print(f"[OBJECT_ACTION] Click result: {result}")
        
        if result:
            return result

    return None
