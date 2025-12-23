from typing import Optional, List, Dict, Any

from actions.travel import _first_blocking_door_from_waypoints
from constants import BANK_REGIONS, REGIONS
from helpers.rects import unwrap_rect
from helpers.runtime_utils import ipc, dispatch
from helpers.ipc import get_last_interaction
from helpers.utils import rect_beta_xy, clean_rs
from services.click_with_camera import click_object_with_camera


def _normalize_object_names(name_or_names: str | List[str]) -> List[str]:
    """
    Normalize an object name (or list of names) into a cleaned list of candidate names.
    Accepts:
      - "bank booth"
      - ["bank booth", "grand exchange booth"]
    """
    if isinstance(name_or_names, list):
        names = name_or_names
    else:
        names = [name_or_names]

    out: List[str] = []
    for n in names:
        if not n:
            continue
        s = str(n).strip()
        if s:
            out.append(s)
    return out


def _normalize_actions(action_or_actions: str | List[str] | None) -> List[str]:
    """
    Normalize action(s) into a list of non-empty strings.
    Accepts:
      - "Use"
      - ["Take", "Check"]
      - None
    """
    if action_or_actions is None:
        return []
    if isinstance(action_or_actions, list):
        raw = action_or_actions
    else:
        raw = [action_or_actions]
    out: List[str] = []
    for a in raw:
        if not a:
            continue
        s = str(a).strip()
        if s:
            out.append(s)
    return out


def _has_exact_action(obj: Dict[str, Any], action: str) -> bool:
    """
    True if obj.actions contains action (case-insensitive exact match).
    """
    want = (action or "").strip().lower()
    if not want:
        return True
    for a in (obj.get("actions") or []):
        if not a:
            continue
        if str(a).strip().lower() == want:
            return True
    return False


def _has_any_exact_action(obj: Dict[str, Any], actions: List[str]) -> bool:
    """
    True if obj.actions contains ANY of the provided actions (case-insensitive exact match).
    If actions is empty, returns True.
    """
    wants = [str(a).strip().lower() for a in (actions or []) if a]
    if not wants:
        return True
    obj_actions = [str(a).strip().lower() for a in (obj.get("actions") or []) if a]
    for w in wants:
        if w in obj_actions:
            return True
    return False


def _pick_first_available_action(obj: Dict[str, Any], prefer_actions: List[str]) -> str | None:
    """
    Pick the first action from prefer_actions that exists on obj.actions.
    If prefer_actions is empty, returns None.
    """
    for a in (prefer_actions or []):
        if _has_exact_action(obj, a):
            return a
    return None


def object_has_action(
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
) -> bool:
    """
    True if the closest matching object currently exposes the given action.
    Intended for lightweight checks inside loops (e.g. "does Bar dispenser have Take right now?").
    """
    try:
        obj_resp = ipc.find_object(name, types=(types or ["GAME"]), exact_match=exact_match_object) or {}
        if not (obj_resp.get("ok") and obj_resp.get("found")):
            return False
        obj = obj_resp.get("object") or {}
        return _has_exact_action(obj, action)
    except Exception:
        return False


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


def _find_closest_object_by_distance(
    name: str | List[str],
    types: List[str] | None = None,
    max_distance: float = None,
    max_objects: int = None,
    exact_match: bool = False,
    *,
    required_action: str | List[str] | None = None,
    radius: int = 26,
) -> Dict[str, Any] | None:
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
    names = _normalize_object_names(name)
    if not names:
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    # If an action is required, we must scan objects and filter by their available actions.
    # This avoids ambiguous cases like multiple objects with the same name but only some having the action.
    req_actions = _normalize_actions(required_action)
    if req_actions:
        best_obj = None
        best_dist = float("inf")
        for n in names:
            resp = ipc.get_objects(n, types=types, radius=int(radius)) or {}
            if not resp.get("ok"):
                continue
            objs = resp.get("objects") or []
            if max_objects is not None and len(objs) > int(max_objects):
                objs = objs[: int(max_objects)]
            for obj in objs:
                if not _has_any_exact_action(obj, req_actions):
                    continue
                d = obj.get("distance")
                try:
                    d = float(d)
                except Exception:
                    d = _get_distance_to_object(obj)
                if max_distance is not None and d <= float(max_distance):
                    return obj
                if d < best_dist:
                    best_dist = d
                    best_obj = obj
        return best_obj

    best_obj = None
    best_dist = float("inf")
    for n in names:
        obj_resp = ipc.find_object(n, types=types, exact_match=exact_match)
        if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
            continue
        obj = obj_resp.get("object")
        if not obj:
            continue
        d = obj.get("distance")
        try:
            d = float(d)
        except Exception:
            d = _get_distance_to_object(obj)

        if max_distance is not None and d <= float(max_distance):
            return obj

        if d < best_dist:
            best_dist = d
            best_obj = obj

    return best_obj


def _find_closest_object_in_area(
    name: str | List[str],
    area: str | tuple,
    types: List[str] | None = None,
    *,
    required_action: str | List[str] | None = None,
    radius: int = 26,
) -> Dict[str, Any] | None:
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
    names = _normalize_object_names(name)
    if not names:
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
    req_actions = _normalize_actions(required_action)
    if req_actions:
        best_obj = None
        best_dist = float("inf")
        for n in names:
            resp = ipc.get_objects(n, types=types, radius=int(radius)) or {}
            if not resp.get("ok"):
                continue
            for obj in (resp.get("objects") or []):
                w = obj.get("world") or {}
                ox, oy = w.get("x"), w.get("y")
                if not (isinstance(ox, int) and isinstance(oy, int)):
                    continue
                if not (int(min_x) <= int(ox) <= int(max_x) and int(min_y) <= int(oy) <= int(max_y)):
                    continue
                if not _has_any_exact_action(obj, req_actions):
                    continue
                d = obj.get("distance")
                try:
                    d = float(d)
                except Exception:
                    d = _get_distance_to_object(obj)
                if d < best_dist:
                    best_dist = d
                    best_obj = obj
        return best_obj

    best_obj = None
    best_dist = float("inf")
    for n in names:
        obj_resp = ipc.find_object_in_area(n, min_x, max_x, min_y, max_y, types=types)
        if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
            continue
        obj = obj_resp.get("object")
        if not obj:
            continue
        d = obj.get("distance")
        try:
            d = float(d)
        except Exception:
            d = _get_distance_to_object(obj)
        if d < best_dist:
            best_dist = d
            best_obj = obj
    return best_obj


def _find_closest_object_by_path(name: str | List[str], types: List[str] | None = None, max_path_distance: int = None, max_objects: int = None) -> Dict[str, Any] | None:
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
    names = _normalize_object_names(name)
    if not names:
        return None
    if types is None:
        types = ["GAME"]  # Default to GAME objects

    # When multiple names are provided, select the best target across all candidates.
    best_object: Dict[str, Any] | None = None
    best_path_distance = 999

    for n in names:
        # Get all matching objects within radius
        print(f"[FIND_OBJECT] Searching for '{n}' with types {types} and radius 26")
        obj_resp = ipc.get_objects(n, types=types, radius=26) or {}
        objects = obj_resp.get("objects", []) if obj_resp.get("ok") else []

        if not objects:
            # Fallback: direct find_object for this name
            fallback = (ipc.find_object(n, types=types) or {}).get("object")
            if fallback:
                pd = _get_path_distance_to_object(fallback)
                if max_path_distance is not None and pd <= int(max_path_distance):
                    return fallback
                if pd < best_path_distance:
                    best_object = fallback
                    best_path_distance = pd
            continue

        # Deduplicate objects by world coordinates (fix for Java bug)
        seen_locations = set()
        unique_objects = []
        for obj in objects:
            world = obj.get("world", {})
            location_key = (world.get("x"), world.get("y"), world.get("p"))
            if location_key not in seen_locations:
                seen_locations.add(location_key)
                unique_objects.append(obj)
        objects = unique_objects

        if max_objects is not None and len(objects) > int(max_objects):
            objects = objects[: int(max_objects)]

        # Score by path distance
        for obj in objects:
            pd = _get_path_distance_to_object(obj)
            if max_path_distance is not None and pd <= int(max_path_distance):
                return obj
            if pd < best_path_distance:
                best_object = obj
                best_path_distance = pd

    if best_object:
        print(f"[FIND_OBJECT] Best match '{best_object.get('name')}' at path distance {best_path_distance} waypoints")
        return best_object

    return None


def object_exists(
    name: str | List[str],
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
    names = _normalize_object_names(name)
    if not names:
        return False
    if types is None:
        types = ["GAME"]
    
    for n in names:
        obj_resp = ipc.find_object(n, types=types)
    if obj_resp and obj_resp.get("ok") and obj_resp.get("found", False):
            try:
                return float(obj_resp["object"].get("distance")) < float(radius)
            except Exception:
                return True
    return False


def click_object_closest_by_path_distance(
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
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
        fresh_obj = _find_closest_object_by_path(name, types=(types or ["GAME"]))
        
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


def click_object_closest_by_path_distance_prefer_no_camera(
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_closest_by_path_distance_no_camera(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_closest_by_path_distance(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_closest_by_path_distance_no_camera(
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """
    No-camera variant of `click_object_closest_by_path_distance`.
    Same logic, but uses `click_object_no_camera(...)` for the final click.
    """
    max_retries = 3

    for _attempt in range(max_retries):
        fresh_obj = _find_closest_object_by_path(name, types=(types or ["GAME"]))
        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue

        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle:
                return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        idx = action_index(fresh_obj.get("actions"), action.lower())
        if idx is None:
            print(f"[OBJECT_ACTION] Action '{action}' not found in object actions")
            continue

        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        rect = unwrap_rect(fresh_obj.get("clickbox")) or unwrap_rect(fresh_obj.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
                alpha=2.0,
                beta=2.0,
            )
        elif isinstance(fresh_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_obj.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_obj["canvas"]["x"]), int(fresh_obj["canvas"]["y"])
        else:
            continue

        # Same behavior as original: idx==0 -> left click, else context menu action.
        click_action = None if idx == 0 else action
        result = click_object_no_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )
        if result:
            return result

    return None


def click_object_closest_by_distance(
    name: str | List[str],
    action: str | List[str],
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    require_action_on_object: bool = False,
) -> Optional[dict]:
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
    prefer_actions = _normalize_actions(action)
    if not prefer_actions:
        print(f"[OBJECT_ACTION] No action provided for object '{name}'")
        return None

    for attempt in range(max_retries):
        # Find closest object by distance
        fresh_obj = _find_closest_object_by_distance(
            name,
            types=(types or ["GAME"]),
            exact_match=exact_match_object,
            required_action=(prefer_actions if require_action_on_object else None),
        )

        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue  # Try next attempt

        chosen_action = _pick_first_available_action(fresh_obj, prefer_actions)
        if not chosen_action:
            print(f"[OBJECT_ACTION] None of actions {prefer_actions} found on object '{fresh_obj.get('name') or name}'")
            continue

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
            action=chosen_action,
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


def click_object_closest_by_distance_prefer_no_camera(
    name: str | List[str],
    action: str | List[str],
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    require_action_on_object: bool = False,
) -> Optional[dict]:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_closest_by_distance_no_camera(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
        require_action_on_object=require_action_on_object,
    )
    if res:
        return res
    return click_object_closest_by_distance(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
        require_action_on_object=require_action_on_object,
    )


def click_object_closest_by_distance_no_camera(
    name: str | List[str],
    action: str | List[str],
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    require_action_on_object: bool = False,
) -> Optional[dict]:
    """
    No-camera variant of `click_object_closest_by_distance`.
    Same logic, but uses `click_object_no_camera(...)` for the final click.
    """
    max_retries = 3
    prefer_actions = _normalize_actions(action)
    if not prefer_actions:
        print(f"[OBJECT_ACTION] No action provided for object '{name}'")
        return None

    for _attempt in range(max_retries):
        fresh_obj = _find_closest_object_by_distance(
            name,
            types=(types or ["GAME"]),
            exact_match=exact_match_object,
            required_action=(prefer_actions if require_action_on_object else None),
        )
        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue

        chosen_action = _pick_first_available_action(fresh_obj, prefer_actions)
        if not chosen_action:
            print(f"[OBJECT_ACTION] None of actions {prefer_actions} found on object '{fresh_obj.get('name') or name}'")
            continue

        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        result = click_object_no_camera(
            object_name=str(obj_name),
            action=chosen_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )
        if result:
            return result

    return None


def click_object_closest_by_path_simple(name: str | List[str], prefer_action: str | None = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
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

    # Keep original name for click target, but allow multiple candidates for search.
    names = _normalize_object_names(name)
    if not names:
        return None
    want = names[0].strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by path distance
        target = _find_closest_object_by_path(names, types=["GAME"], max_path_distance=8)
        
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


def click_object_closest_by_path_simple_prefer_no_camera(
    name: str | List[str],
    prefer_action: str | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_closest_by_path_simple_no_camera(
        name=name,
        prefer_action=prefer_action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_closest_by_path_simple(
        name=name,
        prefer_action=prefer_action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_closest_by_path_simple_no_camera(
    name: str | List[str],
    prefer_action: str | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    No-camera variant of `click_object_closest_by_path_simple`.
    Same logic, but uses `click_object_no_camera(...)` for the click.
    """
    if not name or not str(name).strip():
        return None

    names = _normalize_object_names(name)
    if not names:
        return None
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    for _attempt in range(max_retries):
        target = _find_closest_object_by_path(names, types=["GAME"], max_path_distance=8)
        if not target:
            continue

        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle:
                return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        idx = action_index(target.get("actions"), want_action) if want_action else None
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        # Keep behavior: if no prefer_action provided, do a left click.
        click_action = None if not want_action else (None if idx == 0 else want_action)

        return click_object_no_camera(
            object_name=str(name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )

    return None


def click_object_in_area_simple(name: str | List[str], area: str | tuple, action: str = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
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
    names = _normalize_object_names(name)
    if not names:
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
    target = _find_closest_object_in_area(names, area, types=["GAME"])
    
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


def click_object_in_area_simple_action_auto(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    In-area variant with auto-left-click behavior:
    - If prefer_action is None: do a normal left click.
    - If prefer_action is provided and it's the default action (index 0): left click.
    - Otherwise: right click + context-select the prefer_action.

    This mirrors the behavior of click_object_closest_by_* helpers, but constrains the object to an area.
    (No door handling; use click_object_in_area_action_auto for door handling.)
    """
    if not name or not str(name).strip():
        return None

    names = _normalize_object_names(name)
    if not names:
        return None

    raw_action = (prefer_action or "").strip()
    want_action = raw_action.lower()
    max_retries = 3

    def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
        if not needle:
            return None
        try:
            acts = [a.lower() for a in (actions or []) if a]
            return acts.index(needle) if needle in acts else None
        except Exception:
            return None

    for _attempt in range(max_retries):
        target = _find_closest_object_in_area(
            names,
            area,
            types=(types or ["GAME"]),
            required_action=(raw_action if raw_action else None),
        )
        if not target:
            continue

        world_coords = target.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        idx = action_index(target.get("actions"), want_action) if want_action else None
        # IMPORTANT:
        # If a prefer_action is provided, always context-select it.
        # Left-click can be swapped by RuneLite (e.g. to "Examine"), which breaks "auto-left-click" assumptions.
        click_action = None if not raw_action else raw_action

        # If an action was requested but not present, retry.
        if want_action and idx is None:
            continue

        # Fresh click point from this object (same as click_object_in_area_simple)
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (
                    rect.get("x", 0),
                    rect.get("x", 0) + rect.get("width", 0),
                    rect.get("y", 0),
                    rect.get("y", 0) + rect.get("height", 0),
                ),
                alpha=2.0,
                beta=2.0,
            )
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue

        obj_name = target.get("name") or (names[0] if names else name)
        return click_object_with_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
        )

    return None


def click_object_in_area_action_auto(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Door-handling + in-area + auto-left-click behavior.

    This is the in-area counterpart to click_object_closest_by_path_distance / click_object_closest_by_distance
    style helpers: it will open an earliest blocking door (if any), then click the object.
    """
    if not name or not str(name).strip():
        return None

    names = _normalize_object_names(name)
    if not names:
        return None

    raw_action = (prefer_action or "").strip()
    want_action = raw_action.lower()
    max_retries = 3

    def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
        if not needle:
            return None
        try:
            acts = [a.lower() for a in (actions or []) if a]
            return acts.index(needle) if needle in acts else None
        except Exception:
            return None

    for _attempt in range(max_retries):
        target = _find_closest_object_in_area(
            names,
            area,
            types=(types or ["GAME"]),
            required_action=(raw_action if raw_action else None),
        )
        if not target:
            continue

        world_coords = target.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        idx = action_index(target.get("actions"), want_action) if want_action else None
        # IMPORTANT:
        # If a prefer_action is provided, always context-select it.
        # Left-click can be swapped by RuneLite (e.g. to "Examine"), which breaks "auto-left-click" assumptions.
        click_action = None if not raw_action else raw_action

        if want_action and idx is None:
            continue

        # Door handling on the path to the object
        gx, gy = world_coords.get("x"), world_coords.get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (
                    rect.get("x", 0),
                    rect.get("x", 0) + rect.get("width", 0),
                    rect.get("y", 0),
                    rect.get("y", 0) + rect.get("height", 0),
                ),
                alpha=2.0,
                beta=2.0,
            )
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(target.get("canvas", {}).get("x"), (int, float)) and isinstance(target.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue

        obj_name = target.get("name") or (names[0] if names else name)
        return click_object_with_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
        )

    return None


def click_object_in_area_action_auto_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Prefer-no-camera in-area variant:
    - First attempts to click without camera movement.
    - If it fails, falls back to the camera-based click.
    """
    res = click_object_in_area_action_auto_no_camera(
        name=name,
        area=area,
        prefer_action=prefer_action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_in_area_action_auto(
        name=name,
        area=area,
        prefer_action=prefer_action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_action_auto_no_camera(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    No-camera + in-area + auto-left-click.
    Uses click_object_no_camera for the final click (still does door handling).
    """
    if not name or not str(name).strip():
        return None

    names = _normalize_object_names(name)
    if not names:
        return None

    raw_action = (prefer_action or "").strip()
    want_action = raw_action.lower()

    def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
        if not needle:
            return None
        try:
            acts = [a.lower() for a in (actions or []) if a]
            return acts.index(needle) if needle in acts else None
        except Exception:
            return None

    max_retries = 3
    for _attempt in range(max_retries):
        target = _find_closest_object_in_area(
            names,
            area,
            types=(types or ["GAME"]),
            required_action=(raw_action if raw_action else None),
        )
        if not target:
            continue

        world_coords = target.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        idx = action_index(target.get("actions"), want_action) if want_action else None
        # IMPORTANT:
        # If a prefer_action is provided, always context-select it.
        # Left-click can be swapped by RuneLite (e.g. to "Examine"), which breaks "auto-left-click" assumptions.
        click_action = None if not raw_action else raw_action
        if want_action and idx is None:
            continue

        gx, gy = world_coords.get("x"), world_coords.get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = target.get("name") or (names[0] if names else name)
        return click_object_no_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )

    return None


def click_object_in_area_simple_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_in_area_simple_no_camera(
        name=name,
        area=area,
        action=action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_in_area_simple(
        name=name,
        area=area,
        action=action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_simple_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    No-camera variant of `click_object_in_area_simple`.
    Same logic, but uses `click_object_no_camera(...)` for the click.
    """
    names = _normalize_object_names(name)
    if not names:
        return None

    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        return None

    target = _find_closest_object_in_area(names, area, types=["GAME"])
    if not target:
        return None

    world_coords = target.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        return None

    obj_name = target.get("name") or (names[0] if names else name)
    return click_object_no_camera(
        object_name=str(obj_name),
        action=action,
        world_coords=world_coords,
        door_plan=None,
        exact_match=exact_match_target_and_action,
    )


def click_object_in_area(name: str | List[str], area: str | tuple, action: str = None, exact_match_object: bool = False, exact_match_target_and_action: bool = False) -> dict | None:
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


def click_object_in_area_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_in_area_no_camera(
        name=name,
        area=area,
        action=action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_in_area(
        name=name,
        area=area,
        action=action,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    No-camera variant of `click_object_in_area`.
    Same door handling logic, but uses `click_object_no_camera(...)` for the final click.
    """
    if not name or not str(name).strip():
        return None

    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        return None

    max_retries = 3
    for _attempt in range(max_retries):
        target = _find_closest_object_in_area(name, area, types=["GAME"])
        if not target:
            continue

        gx, gy = target.get("world", {}).get("x"), target.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        world_coords = target.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        obj_name = target.get("name") or name
        result = click_object_no_camera(
            object_name=str(obj_name),
            action=action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )
        if result:
            return result

    return None


def click_object_closest_by_distance_simple(
    name: str | List[str],
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
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

    names = _normalize_object_names(name)
    if not names:
        return None
    want = names[0].strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Find closest object by distance
        target = _find_closest_object_by_distance(names, types=(types or ["GAME"]), exact_match=exact_match_object)
        
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


def click_object_closest_by_distance_simple_prefer_no_camera(
    name: str | List[str],
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    Prefer-no-camera variant:
    - First attempts a click without any camera movement.
    - If unsuccessful, falls back to the camera-based click.
    """
    res = click_object_closest_by_distance_simple_no_camera(
        name=name,
        prefer_action=prefer_action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )
    if res:
        return res
    return click_object_closest_by_distance_simple(
        name=name,
        prefer_action=prefer_action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_closest_by_distance_simple_no_camera(
    name: str | List[str],
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """
    No-camera variant of `click_object_closest_by_distance_simple`.
    Same logic, but uses `click_object_no_camera(...)` for the click.
    """
    if not name or not str(name).strip():
        return None

    names = _normalize_object_names(name)
    if not names:
        return None
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    for _attempt in range(max_retries):
        target = _find_closest_object_by_distance(names, types=(types or ["GAME"]), exact_match=exact_match_object)
        if not target:
            continue

        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle:
                return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        idx = action_index(target.get("actions"), want_action) if want_action else None
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        click_action = None if not want_action else (None if idx == 0 else want_action)
        return click_object_no_camera(
            object_name=str(name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
        )

    return None


def click_object_no_camera(
        object_name: str,
        action: str = None,
        world_coords: dict = None,
        door_plan: dict = None,  # Door-specific plan with coordinates
        aim_ms: int = 420,
        exact_match: bool = False
) -> dict | None:
    """
    Click an object **without any camera movement**.

    This function assumes the camera is already in a usable position and only:
    - derives a click point (door_plan bounds/canvas OR fresh object bounds/canvas)
    - performs a context-menu click via `dispatch`
    - verifies the interaction via `get_last_interaction`
    """

    try:
        if not world_coords:
            return None

        # Try multiple attempts without moving the camera.
        max_attempts = 3
        for _attempt in range(max_attempts):
            point = None

            # Door-specific click point
            if door_plan and door_plan.get("door"):
                door_data = door_plan["door"] or {}

                bounds = door_data.get("bounds") or {}
                if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                    cx, cy = rect_beta_xy(
                        (
                            bounds.get("x", 0),
                            bounds.get("x", 0) + bounds.get("width", 0),
                            bounds.get("y", 0),
                            bounds.get("y", 0) + bounds.get("height", 0),
                        ),
                        alpha=2.0,
                        beta=2.0,
                    )
                    point = {"x": cx, "y": cy}
                elif isinstance((door_data.get("canvas") or {}).get("x"), (int, float)) and isinstance(
                    (door_data.get("canvas") or {}).get("y"), (int, float)
                ):
                    point = {"x": int(door_data["canvas"]["x"]), "y": int(door_data["canvas"]["y"])}

                # Determine action based on door state if not provided
                if action is None:
                    action = "Open" if door_data.get("closed", True) else "Close"

            # Standard object click point (refresh from IPC)
            if point is None:
                objects_resp = ipc.get_object_at_tile(
                    x=world_coords["x"],
                    y=world_coords["y"],
                    plane=world_coords.get("p", 0),
                    # Don't rely on server-side name filtering here.
                    # Some objects (e.g. MLM Sack) can have a base name of "null" and only
                    # resolve correctly via impostors; we do local matching below.
                    name=None,
                )

                if not objects_resp or not objects_resp.get("ok") or not objects_resp.get("objects"):
                    continue

                want = clean_rs(object_name).strip().lower()
                matching_object = None
                soft_match = None
                for obj in (objects_resp.get("objects") or []):
                    nm = clean_rs(obj.get("name")).strip().lower()
                    if not want:
                        matching_object = obj
                        break

                    # Prefer exact equality if we see it (even when exact_match=False).
                    if nm == want:
                        matching_object = obj
                        break

                    # Otherwise fall back to soft matching when exact_match is disabled.
                    if not exact_match:
                        if (want in nm) or (nm in want):
                            if soft_match is None:
                                soft_match = obj

                if matching_object is None and not exact_match:
                    matching_object = soft_match

                if not matching_object:
                    continue

                bounds = matching_object.get("bounds") or {}
                if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                    cx, cy = rect_beta_xy(
                        (
                            bounds.get("x", 0),
                            bounds.get("x", 0) + bounds.get("width", 0),
                            bounds.get("y", 0),
                            bounds.get("y", 0) + bounds.get("height", 0),
                        ),
                        alpha=2.0,
                        beta=2.0,
                    )
                    point = {"x": cx, "y": cy}
                elif isinstance((matching_object.get("canvas") or {}).get("x"), (int, float)) and isinstance(
                    (matching_object.get("canvas") or {}).get("y"), (int, float)
                ):
                    point = {"x": int(matching_object["canvas"]["x"]), "y": int(matching_object["canvas"]["y"])}
                else:
                    continue

            if point is None:
                continue

            # If action is None, do a straight left click.
            if action is None:
                step = {
                    "action": "click-object",
                    "click": {"type": "point", "x": int(point["x"]), "y": int(point["y"])},
                    "target": ({"domain": "object", "name": object_name, "world": world_coords}),
                    "anchor": point,
                }
            else:
                step = {
                    "action": "click-object-context",
                    "option": action,
                    "click": {
                        "type": "context-select",
                        "x": point["x"],
                        "y": point["y"],
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match,
                    },
                    "target": ({"domain": "object", "name": object_name, "world": world_coords}),
                    "anchor": point,
                }

            result = dispatch(step)
            if not result:
                continue

            last_interaction = get_last_interaction() or {}

            clean_action = clean_rs(last_interaction.get("action", ""))
            if action is None:
                action_match = True
            else:
                action_match = (clean_action.lower() == (action or "").lower()) if exact_match else (
                    (action or "").lower() in clean_action.lower()
                )

            # For doors we store `target_name`; for normal objects your code sometimes uses `target`.
            li_target_name = (last_interaction.get("target_name") or "")
            li_target = clean_rs(last_interaction.get("target", ""))
            if exact_match:
                target_match = li_target_name.lower() == (object_name or "").lower() or li_target.lower() == (object_name or "").lower()
            else:
                # Allow either-direction containment to handle cases like:
                #   object_name="Empty sack" but RuneLite reports target_name="Sack"
                want = clean_rs(object_name or "").lower()
                tn = clean_rs(li_target_name).lower()
                tt = clean_rs(li_target).lower()
                target_match = (want in tn) or (want in tt) or (tn and (tn in want)) or (tt and (tt in want))

            if last_interaction and action_match and target_match:
                print(f"[CLICK] {object_name} ({action}) - interaction verified (no camera)")
                return result

        return None

    except Exception:
        raise