"""
Legacy versions of object click methods.

These methods bypass the new camera/movement system:
- No movement compensation (_move_toward_object_if_needed)
- No camera movement (uses click_object_no_camera or direct dispatch)
- Door handling is preserved (still works)

All methods maintain the same function signatures as their counterparts in objects.py.
"""

from typing import Optional, List, Dict, Any
from actions.objects import (
    # Helper functions (reuse from original)
    _normalize_object_names,
    _normalize_actions,
    _has_exact_action,
    _has_any_exact_action,
    _pick_first_available_action,
    _find_closest_object_by_distance,
    _find_closest_object_in_area,
    _find_closest_object_by_path,
    _get_path_distance_to_object,
    # Door handling (keep this)
    _first_blocking_door_from_waypoints,
    # Click function (legacy version)
    click_object_no_camera,
)
from constants import BANK_REGIONS, REGIONS
from helpers.rects import unwrap_rect
from helpers.runtime_utils import ipc, dispatch
from helpers.utils import rect_beta_xy, clean_rs


# ============================================================================
# LEGACY VERSIONS: Path-based distance methods
# ============================================================================

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
    LEGACY VERSION: Click object by path distance.
    No movement compensation, no camera movement.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        fresh_obj = _find_closest_object_by_path(name, types=(types or ["GAME"]))
        
        if not fresh_obj:
            print(f"[OBJECT_ACTION] Object '{name}' not found")
            continue
        
        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None
        
        idx = action_index(fresh_obj.get("actions"), action.lower())
        if idx is None:
            print(f"[OBJECT_ACTION] Action '{action}' not found in object actions")
            continue

        # Door handling (keep this)
        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue
        
        # Get click coordinates
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
            continue
        
        # LEGACY: Skip movement compensation - just click directly
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


def click_object_closest_by_path_distance_prefer_no_camera(
    name: str | List[str],
    action: str,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: Same as no_camera (no preference needed in legacy)."""
    return click_object_closest_by_path_distance_no_camera(
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
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_closest_by_path_distance(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


# ============================================================================
# LEGACY VERSIONS: Straight-line distance methods
# ============================================================================

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
    LEGACY VERSION: Click object by straight-line distance.
    No movement compensation, no camera movement.
    """
    max_retries = 3
    prefer_actions = _normalize_actions(action)
    if not prefer_actions:
        print(f"[OBJECT_ACTION] No action provided for object '{name}'")
        return None

    for attempt in range(max_retries):
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

        # Door handling (keep this)
        gx, gy = fresh_obj.get("world", {}).get("x"), fresh_obj.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue
        
        # Get click coordinates
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
            continue
        
        # LEGACY: Skip movement compensation - just click directly
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


def click_object_closest_by_distance_prefer_no_camera(
    name: str | List[str],
    action: str | List[str],
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    require_action_on_object: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: Same as no_camera (no preference needed in legacy)."""
    return click_object_closest_by_distance_no_camera(
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
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_closest_by_distance(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
        require_action_on_object=require_action_on_object,
    )


# ============================================================================
# LEGACY VERSIONS: Simple path-based methods
# ============================================================================

def click_object_closest_by_path_simple(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> Optional[dict]:
    """
    LEGACY VERSION: Simple path-based click (no door handling).
    No movement compensation, no camera movement.
    """
    names = _normalize_object_names(name)
    if not names:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        fresh_obj = _find_closest_object_by_path(name, types=(types or ["GAME"]))
        if not fresh_obj:
            continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue
        
        # Get click coordinates
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
            continue
        
        # LEGACY: Skip movement compensation - just click directly
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


def click_object_closest_by_path_simple_prefer_no_camera(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: Same as no_camera."""
    return click_object_closest_by_path_simple_no_camera(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_closest_by_path_simple_no_camera(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_closest_by_path_simple(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


# ============================================================================
# LEGACY VERSIONS: Area-based methods
# ============================================================================

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
    LEGACY VERSION: Click object in area (simple, no door handling).
    No movement compensation, no camera movement.
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
            print(f"[ERROR] Unknown area: {area}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format")
        return None

    target = _find_closest_object_in_area(names, area, types=["GAME"])
    if not target:
        return None
    
    world_coords = target.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        return None
    
    # Get click coordinates
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
        return None

    # LEGACY: Skip movement compensation - just click directly
    return click_object_no_camera(
        object_name=name,
        action=action,
        world_coords=world_coords,
        door_plan=None,
        exact_match=exact_match_target_and_action,
        click_coords=click_coords,
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
    LEGACY VERSION: Area click with auto action selection.
    No movement compensation, no camera movement.
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
        click_action = None if not raw_action else raw_action

        if want_action and idx is None:
            continue

        # Get click coordinates
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
        
        # LEGACY: Skip movement compensation - just click directly
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
    LEGACY VERSION: Area click with auto action and door handling.
    No movement compensation, no camera movement.
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

        # Door handling (keep this)
        gx, gy = world_coords.get("x"), world_coords.get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, _ = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                from .travel import _handle_door_opening
                if not _handle_door_opening(door_plan, wps):
                    continue

        idx = action_index(target.get("actions"), want_action) if want_action else None
        click_action = None if not raw_action else raw_action

        if want_action and idx is None:
            continue

        # Get click coordinates
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
        
        # LEGACY: Skip movement compensation - just click directly
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


def click_object_in_area_action_auto_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    prefer_action: str | None = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """LEGACY VERSION: Same as no_camera."""
    return click_object_in_area_action_auto_no_camera(
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
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_in_area_action_auto(
        name=name,
        area=area,
        prefer_action=prefer_action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_simple_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """LEGACY VERSION: Same as no_camera."""
    return click_object_in_area_simple_no_camera(
        name=name,
        area=area,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_simple_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_in_area_simple(
        name=name,
        area=area,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> dict | None:
    """
    LEGACY VERSION: Area click with door handling.
    No movement compensation, no camera movement.
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
            print(f"[ERROR] Unknown area: {area}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format")
        return None

    target = _find_closest_object_in_area(names, area, types=["GAME"])
    if not target:
        return None
    
    world_coords = target.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        return None

    # Door handling (keep this)
    gx, gy = world_coords.get("x"), world_coords.get("y")
    if isinstance(gx, int) and isinstance(gy, int):
        wps, _ = ipc.path(goal=(gx, gy))
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            from .travel import _handle_door_opening
            if not _handle_door_opening(door_plan, wps):
                return None
    
    # Get click coordinates
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
        return None

    # LEGACY: Skip movement compensation - just click directly
    return click_object_no_camera(
        object_name=name,
        action=action,
        world_coords=world_coords,
        door_plan=None,
        exact_match=exact_match_target_and_action,
        click_coords=click_coords,
    )


def click_object_in_area_prefer_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """LEGACY VERSION: Same as no_camera."""
    return click_object_in_area_no_camera(
        name=name,
        area=area,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_in_area_no_camera(
    name: str | List[str],
    area: str | tuple,
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> dict | None:
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_in_area(
        name=name,
        area=area,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


# ============================================================================
# LEGACY VERSIONS: Simple distance-based methods
# ============================================================================

def click_object_closest_by_distance_simple(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
) -> Optional[dict]:
    """
    LEGACY VERSION: Simple distance-based click (no door handling).
    No movement compensation, no camera movement.
    """
    names = _normalize_object_names(name)
    if not names:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        fresh_obj = _find_closest_object_by_distance(
            name,
            types=(types or ["GAME"]),
            exact_match=exact_match_object,
        )
        if not fresh_obj:
            continue

        obj_name = fresh_obj.get("name") or name
        world_coords = fresh_obj.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            continue
        
        # Get click coordinates
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
            continue
        
        # LEGACY: Skip movement compensation - just click directly
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


def click_object_closest_by_distance_simple_prefer_no_camera(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: Same as no_camera."""
    return click_object_closest_by_distance_simple_no_camera(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )


def click_object_closest_by_distance_simple_no_camera(
    name: str | List[str],
    action: str = None,
    *,
    types: List[str] | None = None,
    exact_match_object: bool = False,
    exact_match_target_and_action: bool = False,
) -> Optional[dict]:
    """LEGACY VERSION: No camera, no movement compensation."""
    return click_object_closest_by_distance_simple(
        name=name,
        action=action,
        types=types,
        exact_match_object=exact_match_object,
        exact_match_target_and_action=exact_match_target_and_action,
    )



