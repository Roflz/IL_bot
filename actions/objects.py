from typing import Optional, List, Dict, Any
import random

from actions.travel import (
    _first_blocking_door_from_waypoints, 
    go_to_tile, 
    _get_player_movement_state, 
    _calculate_click_location,
    _is_moving_toward_path,
    _verify_click_path,
    _should_click_again,
    _calculate_distance,
    _is_tile_on_path
)
from constants import BANK_REGIONS, REGIONS
from helpers.rects import unwrap_rect
from helpers.runtime_utils import ipc, dispatch
from helpers.ipc import get_last_interaction
from helpers.utils import rect_beta_xy, clean_rs
from services.click_with_camera import click_object_with_camera
from actions import player


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


def object_at_tile_has_action(
    x: int,
    y: int,
    plane: int,
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
) -> bool:
    """
    Check if an object at a specific tile has the given action.
    
    Args:
        x: World X coordinate
        y: World Y coordinate
        plane: World plane
        name: Object name(s) to match
        action: Action to check for
        types: Object types to filter by (default: ["GAME"])
        exact_match_object: Whether to use exact name matching
    
    Returns:
        True if the object at the specified tile has the action, False otherwise
    """
    try:
        if isinstance(name, list):
            name_to_check = name[0] if name else None
        else:
            name_to_check = name
        
        resp = ipc.get_object_at_tile(x, y, plane, name=name_to_check, types=(types or ["GAME"]))
        if not resp.get("ok"):
            return False
        
        objects_list = resp.get("objects", []) or []
        for obj in objects_list:
            obj_name = obj.get("name", "")
            # Check name match
            if exact_match_object:
                if obj_name != name_to_check:
                    continue
            else:
                if isinstance(name, list):
                    if not any(n.lower() in obj_name.lower() for n in name):
                        continue
                else:
                    if name.lower() not in obj_name.lower():
                        continue
            
            # Check if it has the action
            if _has_exact_action(obj, action):
                return True
        
        return False
    except Exception:
        return False


# Movement state tracking for object interactions (separate from go_to movement state)
_object_movement_state = {
    "last_object_tile": None,  # {"x": int, "y": int} - Last object we were moving toward
    "last_clicked_tile": None,  # {"x": int, "y": int} - Last tile clicked when moving toward object
    "current_path": None,  # List of waypoints from last click to object
    "intended_path": None,  # List of waypoints from player to object
    "is_moving": False,  # Whether we're currently moving toward an object
    "last_movement_check_ts": None,  # Timestamp of last movement check
}

def _clear_object_movement_state():
    """Clear object movement state (call when switching to a different object)."""
    global _object_movement_state
    _object_movement_state = {
        "last_object_tile": None,
        "last_clicked_tile": None,
        "current_path": None,
        "intended_path": None,
        "is_moving": False,
        "last_movement_check_ts": None,
    }

def _move_toward_object_if_needed(
    obj_world_coords: dict,
    click_coords: dict,
    click_rect: dict = None,
    pre_calculated_path: list = None,
    path_radius: int = 5,
    path_center_bias: float = 0.7,
    # Running parameters
    running_min_path: int = 5,
    running_mid_path: int = 10,
    running_max_path: int = 20,
    running_min_click_prob: float = 0.1,
    running_mid_click_prob: float = 0.8,
    running_max_click_prob: float = 1.0,
    # Walking parameters
    walking_min_path: int = 3,
    walking_mid_path: int = 12,
    walking_max_path: int = 15,
    walking_min_click_prob: float = 0.2,
    walking_mid_click_prob: float = 0.2,
    walking_max_click_prob: float = 1.0,
    # Stationary parameters (same as running by default)
    stationary_min_path: int = 5,
    stationary_mid_path: int = 10,
    stationary_max_path: int = 20,
    stationary_min_click_prob: float = 0.1,
    stationary_mid_click_prob: float = 0.8,
    stationary_max_click_prob: float = 1.0,
    # Manhattan distance guard
    manhattan_cutoff_min: float = 20.0,
    manhattan_cutoff_max: float = 25.0,
    # Probability variance
    probability_variance: float = 0.05
) -> bool:
    """
    Move toward an object using sophisticated movement system with movement-state-aware probability.
    
    Uses path distance (waypoint count) with movement-state-aware probability to decide whether to:
    - Path toward the object (using normal distribution around target)
    - Click the object directly (with movement compensation)
    
    Further from object = more likely to path, closer = more likely to click directly.
    Running allows clicking directly from further away than walking.
    
    Args:
        obj_world_coords: World coordinates of the object {"x": int, "y": int, "p": int}
        click_coords: Canvas click coordinates (will be adjusted with movement compensation)
        click_rect: Optional rectangle bounds for the object
        path_radius: Radius around target when pathing (for normal distribution)
        path_center_bias: Center bias for normal distribution when pathing
        running_min_path: Below this path distance, always click directly (running)
        running_mid_path: Midpoint for probability curve (running)
        running_max_path: Above this, use min_click_prob (running)
        running_min_click_prob: Minimum click-directly probability at max_path (running)
        running_mid_click_prob: Click-directly probability at mid_path (running)
        running_max_click_prob: Maximum click-directly probability at min_path (running)
        walking_min_path: Below this path distance, always click directly (walking)
        walking_mid_path: Midpoint for probability curve (walking)
        walking_max_path: Above this, use min_click_prob (walking)
        walking_min_click_prob: Minimum click-directly probability at max_path (walking)
        walking_mid_click_prob: Click-directly probability at mid_path (walking)
        walking_max_click_prob: Maximum click-directly probability at min_path (walking)
        stationary_min_path: Below this path distance, always click directly (stationary)
        stationary_mid_path: Midpoint for probability curve (stationary)
        stationary_max_path: Above this, use min_click_prob (stationary)
        stationary_min_click_prob: Minimum click-directly probability at max_path (stationary)
        stationary_mid_click_prob: Click-directly probability at mid_path (stationary)
        stationary_max_click_prob: Maximum click-directly probability at min_path (stationary)
        manhattan_cutoff_min: Minimum Manhattan distance for always-path guard
        manhattan_cutoff_max: Maximum Manhattan distance for always-path guard
        probability_variance: Variance to add to probability values (e.g., 0.05 = ±5%)
    
    Returns:
        True if we moved/pathing (and should retry clicking), False if we're ready to click directly
    """
    from actions.travel import (
        _should_path_toward_target_with_movement_state,
        _calculate_path_click_location_for_target,
        _verify_click_path,
        _get_player_movement_state,
        go_to_tile
    )
    
    obj_x = obj_world_coords.get("x")
    obj_y = obj_world_coords.get("y")
    
    if not isinstance(obj_x, int) or not isinstance(obj_y, int):
        return False
    
    # Get player position
    player_x = player.get_x()
    player_y = player.get_y()
    
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Calculate Manhattan distance to object (for guard check)
    dx = obj_x - player_x
    dy = obj_y - player_y
    manhattan_distance = abs(dx) + abs(dy)
    
    # Get path distance (waypoint count) to object
    # Reuse pre-calculated path if provided (e.g., from door checking)
    if pre_calculated_path is not None:
        intended_path = pre_calculated_path
        path_distance = len(intended_path) if intended_path else 999
    else:
        rect = (obj_x - 1, obj_x + 1, obj_y - 1, obj_y + 1)
        intended_path, _ = ipc.path(rect=rect, visualize=False)
        path_distance = len(intended_path) if intended_path else 999  # Use 999 if no path
    
    # Get movement state
    movement_state = _get_player_movement_state()
    
    # STEP 1: Check if we're already moving toward the object (similar to go_to_tile logic)
    if movement_state and movement_state.get("is_moving"):
        from actions.travel import _is_moving_toward_path, _movement_state
        final_dest = {"x": obj_x, "y": obj_y}
        
        # Reuse intended_path as path_to_dest to avoid recalculation
        path_to_dest = intended_path if intended_path else None
        
        # Check if we're already moving toward this object (reuse intended_path)
        if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest, 
                                   path_to_clicked=None, path_to_dest=path_to_dest):
            # Already moving toward object - don't click again
            print(f"[OBJECT_MOVEMENT] Already moving toward object at ({obj_x}, {obj_y}), skipping click")
            return True  # Return True to signal we're moving (retry later)
        
        # Check if we're close to last clicked tile (path distance) - if so, might need to click again
        last_clicked = _movement_state.get("last_clicked_tile")
        path_to_clicked = None
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
                    
                    # Dynamic thresholds based on movement speed (same as go_to_tile)
                    if is_running:
                        threshold = random.randint(8, 10)
                    else:
                        threshold = random.randint(3, 4)
                    
                    if path_length > threshold:
                        # Not close to last clicked tile - check if we're moving toward object
                        # Reuse both paths to avoid recalculation
                        if _is_moving_toward_path(movement_state, _movement_state.get("current_path"), final_dest,
                                                   path_to_clicked=path_to_clicked, path_to_dest=path_to_dest):
                            # Paths converge - we're moving correctly, don't click
                            return True
    
    # STEP 2: Decide whether to path toward target or click directly using movement-state-aware probability
    should_path = _should_path_toward_target_with_movement_state(
        path_distance=path_distance,
        manhattan_distance=manhattan_distance,
        movement_state=movement_state,
        running_min_path=running_min_path,
        running_mid_path=running_mid_path,
        running_max_path=running_max_path,
        running_min_click_prob=running_min_click_prob,
        running_mid_click_prob=running_mid_click_prob,
        running_max_click_prob=running_max_click_prob,
        walking_min_path=walking_min_path,
        walking_mid_path=walking_mid_path,
        walking_max_path=walking_max_path,
        walking_min_click_prob=walking_min_click_prob,
        walking_mid_click_prob=walking_mid_click_prob,
        walking_max_click_prob=walking_max_click_prob,
        stationary_min_path=stationary_min_path,
        stationary_mid_path=stationary_mid_path,
        stationary_max_path=stationary_max_path,
        stationary_min_click_prob=stationary_min_click_prob,
        stationary_mid_click_prob=stationary_mid_click_prob,
        stationary_max_click_prob=stationary_max_click_prob,
        manhattan_cutoff_min=manhattan_cutoff_min,
        manhattan_cutoff_max=manhattan_cutoff_max,
        probability_variance=probability_variance
    )
    
    if should_path:
        # Path toward target using sophisticated movement
        # Calculate click location using normal distribution around target
        click_location = _calculate_path_click_location_for_target(
            player_x, player_y, obj_x, obj_y, path_radius, path_center_bias
        )
        
        # Use the intended_path we already calculated above
        if intended_path:
            # Verify the click location paths correctly
            # Note: We don't have clicked_path pre-calculated, so _verify_click_path will calculate it
            # But we're reusing intended_path which is good
            final_dest = {"x": obj_x, "y": obj_y}
            if _verify_click_path(click_location, final_dest, intended_path, clicked_path=None):
                # Click location is valid - use go_to_tile with sophisticated movement
                print(f"[OBJECT_MOVEMENT] Pathing toward object (path_distance: {path_distance}, manhattan: {manhattan_distance}), clicking at ({click_location.get('x')}, {click_location.get('y')})")
                result = go_to_tile(click_location["x"], click_location["y"], arrive_radius=0)
                if result:
                    return True  # Successfully moved - retry
                # Movement failed - fall through to direct click
            else:
                # Click location doesn't path correctly - fall back to direct click
                print(f"[OBJECT_MOVEMENT] Calculated click location doesn't path correctly, clicking object directly")
        else:
            # Can't get path - fall back to direct click
            should_path = False
    
    if not should_path:
        # Click target directly - apply movement compensation to click coordinates
        movement_state = _get_player_movement_state()
        
        if movement_state and movement_state.get("is_moving"):
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
                
                # Adjust click position opposite to movement direction
                try:
                    # Get object's world tile projection
                    proj = ipc.project_world_tile(obj_x, obj_y)
                    if proj and proj.get("ok") and proj.get("canvas"):
                        canvas_x = proj["canvas"].get("x")
                        canvas_y = proj["canvas"].get("y")
                        
                        if isinstance(canvas_x, (int, float)) and isinstance(canvas_y, (int, float)):
                            # Adjust canvas coordinates opposite to movement direction
                            if dx_dir != 0:
                                canvas_x = int(canvas_x - (dx_dir / abs(dx_dir)) * compensation * 50)
                            if dy_dir != 0:
                                canvas_y = int(canvas_y - (dy_dir / abs(dy_dir)) * compensation * 50)
                            
                            click_coords["x"] = canvas_x
                            click_coords["y"] = canvas_y
                            print(f"[OBJECT_MOVEMENT] Clicking object directly (path_distance: {path_distance}, manhattan: {manhattan_distance}), applied movement compensation: ({canvas_x}, {canvas_y})")
                except Exception as e:
                    print(f"[OBJECT_MOVEMENT] Failed to apply movement compensation: {e}")
        
        # Track the object click in movement state (so we can verify we're moving toward it)
        from actions.travel import _movement_state
        import time
        _movement_state["last_clicked_tile"] = {"x": obj_x, "y": obj_y}
        _movement_state["final_destination"] = {"x": obj_x, "y": obj_y}
        if intended_path:
            _movement_state["current_path"] = intended_path
            _movement_state["intended_path"] = intended_path
        _movement_state["is_moving"] = True
        _movement_state["last_movement_check_ts"] = time.time()
        
        # Ready to click directly
        return False
    
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
    path_waypoints: list = None,
    movement_state: dict = None,
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
        
        # Use path from door handling if available, otherwise use provided path_waypoints
        obj_path = path_waypoints if path_waypoints is not None else (wps if 'wps' in locals() else None)
        
        # Check if we need to move closer to the object before clicking
        # Reuse path from door checking if available
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect, pre_calculated_path=obj_path):
            # We moved - retry the click attempt
            continue
        
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
                exact_match=exact_match_target_and_action,
                path_waypoints=obj_path,
                movement_state=movement_state
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
                exact_match=exact_match_target_and_action,
                path_waypoints=obj_path,
                movement_state=movement_state
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
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(fresh_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_obj.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_obj["canvas"]["x"]), int(fresh_obj["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        # Same behavior as original: idx==0 -> left click, else context menu action.
        click_action = None if idx == 0 else action
        result = click_object_no_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
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
    path_waypoints: list = None,
    movement_state: dict = None,
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
        obj_path = path_waypoints
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            # Use path from door handling if path_waypoints not provided
            if obj_path is None:
                obj_path = wps
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

        # Check if we need to move closer to the object before clicking
        # Reuse path from door checking if available
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect, pre_calculated_path=obj_path):
            # We moved - retry the click attempt
            continue

        result = click_object_with_camera(
            object_name=obj_name,
            action=chosen_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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
        obj_path_for_movement = None
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            obj_path_for_movement = wps  # Reuse for movement check
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue

        # Get click coordinates for movement check
        rect = unwrap_rect(fresh_obj.get("clickbox")) or unwrap_rect(fresh_obj.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                 rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
                alpha=2.0,
                beta=2.0,
            )
            click_coords = {"x": cx, "y": cy}
            click_rect = rect
        elif isinstance(fresh_obj.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_obj.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_obj["canvas"]["x"]), int(fresh_obj["canvas"]["y"])
            click_coords = {"x": cx, "y": cy}
            click_rect = None
        else:
            continue

        # Check if we need to move closer to the object before clicking
        # Reuse path from door checking if available
        # if _move_toward_object_if_needed(world_coords, click_coords, click_rect, pre_calculated_path=obj_path_for_movement):
        #     # We moved - retry the click attempt
        #     continue

        result = click_object_no_camera(
            object_name=str(obj_name),
            action=chosen_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
        )
        if result:
            return result

    return None


def click_object_closest_by_path_simple(
    name: str | List[str], 
    prefer_action: str | None = None, 
    exact_match_object: bool = False, 
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> dict | None:
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

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        # Use simplified click with camera function
        # Get path to object if not provided
        obj_path = path_waypoints
        if obj_path is None:
            try:
                obj_x = world_coords.get("x")
                obj_y = world_coords.get("y")
                if isinstance(obj_x, int) and isinstance(obj_y, int):
                    wps, _ = ipc.path(goal=(obj_x, obj_y), visualize=False)
                    obj_path = wps
            except Exception:
                pass
        
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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

        # Get click coordinates for movement check
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                 rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
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

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        # Keep behavior: if no prefer_action provided, do a left click.
        click_action = None if not want_action else (None if idx == 0 else want_action)

        return click_object_no_camera(
            object_name=str(name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
        )

    return None


def click_object_in_area_simple(
    name: str | List[str], 
    area: str | tuple, 
    action: str = None, 
    exact_match_object: bool = False, 
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> dict | None:
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

    # Check if we need to move closer to the object before clicking
    if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
        # We moved - return None to indicate we should retry
        return None

    # Use click_object_with_camera function (simple version without door handling)
    # Get path to object if not provided
    obj_path = path_waypoints
    if obj_path is None:
        try:
            obj_x = world_coords.get("x")
            obj_y = world_coords.get("y")
            if isinstance(obj_x, int) and isinstance(obj_y, int):
                wps, _ = ipc.path(goal=(obj_x, obj_y), visualize=False)
                obj_path = wps
        except Exception:
            pass
    
    return click_object_with_camera(
        object_name=name,
        action=action,
        world_coords=world_coords,
        click_coords=click_coords,
        click_rect=click_rect,
        aim_ms=420,
        exact_match=exact_match_target_and_action,
        path_waypoints=obj_path,
        movement_state=movement_state
    )


def click_object_in_area_simple_action_auto(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
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
        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue
        
        # Get path to object if not provided
        obj_path = path_waypoints
        if obj_path is None:
            try:
                obj_x = world_coords.get("x")
                obj_y = world_coords.get("y")
                if isinstance(obj_x, int) and isinstance(obj_y, int):
                    wps, _ = ipc.path(goal=(obj_x, obj_y), visualize=False)
                    obj_path = wps
            except Exception:
                pass
        
        return click_object_with_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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
    path_waypoints: list = None,
    movement_state: dict = None,
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
        obj_path = path_waypoints
        gx, gy = world_coords.get("x"), world_coords.get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _dbg_path = ipc.path(goal=(gx, gy))
            # Use path from door handling if path_waypoints not provided
            if obj_path is None:
                obj_path = wps
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
        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue
        
        return click_object_with_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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

        # Get click coordinates for movement check
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

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        obj_name = target.get("name") or (names[0] if names else name)
        return click_object_no_camera(
            object_name=str(obj_name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
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

    # Get click coordinates for movement check
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    if rect:
        cx, cy = rect_beta_xy(
            (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
             rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
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
        return None

    # Check if we need to move closer to the object before clicking
    if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
        # We moved - return None to indicate we should retry
        return None

    obj_name = target.get("name") or (names[0] if names else name)
    return click_object_no_camera(
        object_name=str(obj_name),
        action=action,
        world_coords=world_coords,
        door_plan=None,
        exact_match=exact_match_target_and_action,
        click_coords=click_coords,
    )


def click_object_in_area(
    name: str | List[str], 
    area: str | tuple, 
    action: str = None, 
    exact_match_object: bool = False, 
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> dict | None:
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
        obj_path = path_waypoints
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            # Use path from door handling if path_waypoints not provided
            if obj_path is None:
                obj_path = wps
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
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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

        # Get click coordinates for movement check
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                 rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
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

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        obj_name = target.get("name") or name
        result = click_object_no_camera(
            object_name=str(obj_name),
            action=action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
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
    path_waypoints: list = None,
    movement_state: dict = None,
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

        # Check if we need to move closer to the object before clicking
        if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
            # We moved - retry the click attempt
            continue

        # Use simplified click with camera function
        # Get path to object if not provided
        obj_path = path_waypoints
        if obj_path is None:
            try:
                obj_x = world_coords.get("x")
                obj_y = world_coords.get("y")
                if isinstance(obj_x, int) and isinstance(obj_y, int):
                    wps, _ = ipc.path(goal=(obj_x, obj_y), visualize=False)
                    obj_path = wps
            except Exception:
                pass
        
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            world_coords=world_coords,
            click_coords=click_coords,
            click_rect=click_rect,
            aim_ms=420,
            exact_match=exact_match_target_and_action,
            path_waypoints=obj_path,
            movement_state=movement_state
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

        # Get click coordinates for movement check
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        if rect:
            cx, cy = rect_beta_xy(
                (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), 
                 rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
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

        # Check if we need to move closer to the object before clicking
        # if _move_toward_object_if_needed(world_coords, click_coords, click_rect):
        #     # We moved - retry the click attempt
        #     continue

        click_action = None if not want_action else want_action
        return click_object_no_camera(
            object_name=str(name),
            action=click_action,
            world_coords=world_coords,
            door_plan=None,
            exact_match=exact_match_target_and_action,
            click_coords=click_coords,
        )

    return None


def click_object_no_camera(
        object_name: str,
        action: str = None,
        world_coords: dict = None,
        door_plan: dict = None,  # Door-specific plan with coordinates
        aim_ms: int = 420,
        exact_match: bool = False,
        click_coords: dict = None  # Optional pre-calculated click coordinates (with movement compensation applied)
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

            # Use provided click_coords if available (already has movement compensation applied)
            if click_coords and isinstance(click_coords.get("x"), (int, float)) and isinstance(click_coords.get("y"), (int, float)):
                point = {"x": int(click_coords["x"]), "y": int(click_coords["y"])}

            # Door-specific click point (only if click_coords not provided)
            if point is None and door_plan and door_plan.get("door"):
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

            # Standard object click point (refresh from IPC) - only if click_coords not provided
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