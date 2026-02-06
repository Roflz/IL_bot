"""
Centralized click with camera functionality.
Handles the 3-step process: camera movement, fresh coordinates, click.
"""
import logging
import random

from helpers import unwrap_rect
from helpers.ipc import get_last_interaction
from helpers.utils import clean_rs, sleep_exponential, rect_beta_xy
from services.camera_integration import aim_midtop_at_world, aim_camera_along_path, aim_camera_at_target
from helpers.runtime_utils import ipc, dispatch
from helpers.navigation import _merge_door_into_projection
from actions.movement import get_movement_direction


def click_object_with_camera(
    object_name: str, 
    action: str = None,
    world_coords: dict = None,
    click_coords: dict = None,  # {"x": cx, "y": cy} for the click point
    click_rect: dict = None,    # Rectangle bounds for anchor
    door_plan: dict = None,     # Door-specific plan with coordinates
    aim_ms: int = 420,
    exact_match: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    Handles doors specially when door_plan is provided.
    
    Args:
        path_waypoints: Optional list of waypoint dicts from pathfinding for path-aware camera
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
    """
    
    try:
        if not world_coords:
            return None

        # Try camera retry directions: first try without moving camera, then LEFT, then RIGHT
        camera_retry_directions = [None, "LEFT", "RIGHT"]
        camera_retry_duration = 0.5  # Move camera for 500ms
        
        for camera_retry_idx, direction in enumerate(camera_retry_directions):
            # Move camera if this is not the first attempt
            if direction is not None:
                try:
                    ipc.key_press(direction)
                    sleep_exponential(camera_retry_duration * 0.8, camera_retry_duration * 1.2, 1.0)
                    ipc.key_release(direction)
                except Exception:
                    pass  # Don't fail if camera movement fails
            
            # Try multiple attempts with different coordinates
            max_attempts = 3
            for attempt in range(max_attempts):
                # Move camera to target using path-aware logic if available
                from actions import player
                player_x = player.get_x()
                player_y = player.get_y()
                
                # Use new camera system
                # Calculate distance for mode detection
                from actions.travel import _get_player_movement_state
                if isinstance(player_x, int) and isinstance(player_y, int):
                    dx = abs(world_coords['x'] - player_x)
                    dy = abs(world_coords['y'] - player_y)
                    distance = dx + dy  # Manhattan distance
                else:
                    distance = None
                
                # Use new camera system with auto-detection
                # Get area center from camera state if available
                from services.camera_integration import get_camera_state
                camera_state, state_config, area_center, _ = get_camera_state()
                
                aim_camera_at_target(
                    target_world_coords=world_coords,
                    mode=None,  # Auto-detect
                    action_type="click_object",
                    distance_to_target=distance,
                    path_waypoints=path_waypoints,
                    movement_state=movement_state,
                    max_ms=aim_ms,
                    object_world_coords=world_coords,  # Pass object coords for OBJECT_INTERACTION state
                    area_center=area_center
                )
                
                # Handle doors specially if door_plan is provided
                if door_plan and door_plan.get('door'):
                    door_data = door_plan['door']
                    
                    # Use door's specific coordinates - check bounds first, then canvas
                    bounds = door_data.get('bounds', {})
                    if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                        cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                               bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
                        point = {"x": cx, "y": cy}
                    elif door_data.get('canvas'):
                        cx = int(door_data["canvas"]["x"])
                        cy = int(door_data["canvas"]["y"])
                        point = {"x": cx, "y": cy}
                    
                    # Check if click point is within any blocking UI widget
                    from helpers.widgets import is_point_in_blocking_widget
                    if is_point_in_blocking_widget(cx, cy):
                        # Click point is blocked by UI - adjust to nearest point outside widget bounds
                        adjusted = False
                        canvas_width = 1920  # Default, adjust if needed
                        canvas_height = 1080  # Default, adjust if needed
                        margin = 10
                        
                        for offset_x, offset_y in [
                            (0, -50), (0, 50), (-50, 0), (50, 0),  # Up, down, left, right
                            (-30, -30), (30, -30), (-30, 30), (30, 30),  # Diagonals
                            (0, -100), (0, 100), (-100, 0), (100, 0),  # Further in cardinal directions
                        ]:
                            test_x = cx + offset_x
                            test_y = cy + offset_y
                            
                            # Ensure test point is still within canvas bounds
                            if (margin <= test_x <= canvas_width - margin and 
                                margin <= test_y <= canvas_height - margin and
                                not is_point_in_blocking_widget(test_x, test_y)):
                                cx = test_x
                                cy = test_y
                                point = {"x": cx, "y": cy}
                                adjusted = True
                                print(f"[CLICK] Door - click point blocked by UI widget, adjusted to ({cx}, {cy})")
                                break
                        
                        if not adjusted:
                            # Couldn't find a valid point - this is a problem, but we'll try anyway
                            print(f"[WARNING] Door - click point ({cx}, {cy}) is blocked by UI widget and couldn't be adjusted")
                        
                    # Determine action based on door state
                    if action is None:
                        action = "Open" if door_data.get('closed', True) else "Close"

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
                            "exact_match": exact_match
                        },
                        "target": ({"domain": "object", "name": object_name, "world": world_coords}),
                        "anchor": point,
                    }

                    result = dispatch(step)
                        
                    if result:
                        # Check if the correct interaction was performed
                        last_interaction = get_last_interaction()
                        
                        # Use exact match or contains based on exact_match parameter
                        clean_action = clean_rs(last_interaction.get("action", ""))
                        want_action = (action or "").lower()
                        action_match = (clean_action.lower() == want_action) if exact_match else (
                            (want_action in clean_action.lower()) or (clean_action.lower() in want_action)
                        )
                        want = clean_rs(object_name).lower()
                        tgt = clean_rs(last_interaction.get("target_name", "")).lower()
                        target_match = (tgt == want) if exact_match else ((want in tgt) or (tgt and (tgt in want)))
                        
                        if (last_interaction and action_match and target_match):
                            print(f"[CLICK] {object_name} ({action}) - interaction verified")
                            return result
                        else:
                            print(f"[CLICK] {object_name} ({action}) - incorrect interaction, retrying...")
                            continue
                
                # Standard object handling for non-doors
                # Re-acquire screen coordinates after camera movement
                # Get the actual object at the tile to get its fresh screen coordinates
                objects_resp = ipc.get_object_at_tile(
                    x=world_coords['x'], 
                    y=world_coords['y'], 
                    plane=world_coords.get('p', 0),
                    # Don't rely on server-side name filtering here; some objects resolve via impostors.
                    name=None
                )
                
                if not objects_resp.get("ok") or not objects_resp.get("objects"):
                    continue
                
                # Find the matching object by name (prefer exact equality, else soft match when exact_match=False)
                want_name = clean_rs(object_name).strip().lower()
                matching_object = None
                soft_match = None
                for obj in (objects_resp.get("objects") or []):
                    nm = clean_rs(obj.get("name", "")).strip().lower()
                    if not want_name:
                        matching_object = obj
                        break
                    if nm == want_name:
                        matching_object = obj
                        break
                    if not exact_match and ((want_name in nm) or (nm in want_name)):
                        if soft_match is None:
                            soft_match = obj
                if matching_object is None and not exact_match:
                    matching_object = soft_match
                
                if not matching_object or not matching_object.get("canvas"):
                    continue
                
                # Use fresh coordinates from the object - check bounds first, then canvas
                bounds = matching_object.get("bounds", {})
                if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                    cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                           bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
                    point = {"x": cx, "y": cy}
                else:
                    # Fallback to canvas coordinates
                    fresh_coords = matching_object["canvas"]
                    cx = int(fresh_coords["x"])
                    cy = int(fresh_coords["y"])
                    point = {"x": cx, "y": cy}
                
                # Check if click point is within any blocking UI widget
                from helpers.widgets import is_point_in_blocking_widget
                if is_point_in_blocking_widget(cx, cy):
                    # Click point is blocked by UI - adjust to nearest point outside widget bounds
                    # Try moving the click point in different directions to find a valid location
                    adjusted = False
                    canvas_width = 1920  # Default, adjust if needed
                    canvas_height = 1080  # Default, adjust if needed
                    margin = 10
                    
                    for offset_x, offset_y in [
                        (0, -50), (0, 50), (-50, 0), (50, 0),  # Up, down, left, right
                        (-30, -30), (30, -30), (-30, 30), (30, 30),  # Diagonals
                        (0, -100), (0, 100), (-100, 0), (100, 0),  # Further in cardinal directions
                    ]:
                        test_x = cx + offset_x
                        test_y = cy + offset_y
                        
                        # Ensure test point is still within canvas bounds
                        if (margin <= test_x <= canvas_width - margin and 
                            margin <= test_y <= canvas_height - margin and
                            not is_point_in_blocking_widget(test_x, test_y)):
                            cx = test_x
                            cy = test_y
                            point = {"x": cx, "y": cy}
                            adjusted = True
                            print(f"[CLICK] Object - click point blocked by UI widget, adjusted to ({cx}, {cy})")
                            break
                    
                    if not adjusted:
                        # Couldn't find a valid point - this is a problem, but we'll try anyway
                        print(f"[WARNING] Object - click point ({cx}, {cy}) is blocked by UI widget and couldn't be adjusted")

                step = {
                    "action": "click-object-context",
                    "option": action,
                    "click": {
                        "type": "context-select",
                        # "index": int(action_index),
                        "x": point["x"],
                        "y": point["y"],
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match
                    },
                    "target": ({"domain": "object", "name": object_name, "world": world_coords}),
                    "anchor": point,
                }

                result = dispatch(step)
                
                if result:
                    # Check if the correct interaction was performed
                    last_interaction = get_last_interaction()
                    
                    # Use exact match or contains based on exact_match parameter
                    clean_action = clean_rs(last_interaction.get("action", ""))
                    want_action = (action or "").lower()
                    action_match = (clean_action.lower() == want_action) if exact_match else (
                        (want_action in clean_action.lower()) or (clean_action.lower() in want_action)
                    )
                    want = clean_rs(object_name).lower()
                    tgt = clean_rs(last_interaction.get("target", "")).lower()
                    target_match = (tgt == want) if exact_match else ((want in tgt) or (tgt and (tgt in want)))
                    
                    if (last_interaction and action_match and target_match):
                        print(f"[CLICK] {object_name} - interaction verified")
                        return result
                    else:
                        print(f"[CLICK] {object_name} - incorrect interaction, retrying...")
                        continue
        
        return None
    
    except Exception as e:
        raise


def click_npc_with_camera(
    npc_name: str,
    action: str = None,
    world_coords: dict = None,
    aim_ms: int = 420,
    exact_match: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None,
    disable_pitch: bool = False,
    area: str | tuple = None
) -> dict | None:
    """
    Click an NPC with camera movement and fresh coordinate recalculation.
    
    Args:
        path_waypoints: Optional list of waypoint dicts from pathfinding for path-aware camera
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
        disable_pitch: If True, pitch will not be adjusted during camera movement (default: False)
        area: Optional area constraint - can be:
            - String: Area name from constants.py (e.g., "GWD_BANDOS")
            - Tuple: (min_x, max_x, min_y, max_y) for custom coordinates
            If provided, only clicks NPCs within this area
    """
    try:
        # if not world_coords:
        #     return None

        # Check if NPC is already visible and within clickable area before moving camera
        npc_resp_precheck = ipc.find_npc(npc_name)
        skip_camera_movement = False
        
        if npc_resp_precheck and npc_resp_precheck.get("ok") and npc_resp_precheck.get("found"):
            target_precheck = npc_resp_precheck.get("npc")
            if target_precheck:
                # Verify NPC is within specified area (if area constraint provided)
                if area is not None:
                    target_world_precheck = target_precheck.get("world", {})
                    target_x_precheck = target_world_precheck.get("x")
                    target_y_precheck = target_world_precheck.get("y")
                    
                    if isinstance(target_x_precheck, int) and isinstance(target_y_precheck, int):
                        # Resolve area coordinates
                        min_x, max_x, min_y, max_y = None, None, None, None
                        if isinstance(area, str):
                            try:
                                from constants import REGIONS, BANK_REGIONS
                                if area in REGIONS:
                                    min_x, max_x, min_y, max_y = REGIONS[area]
                                elif area in BANK_REGIONS:
                                    min_x, max_x, min_y, max_y = BANK_REGIONS[area]
                            except ImportError:
                                pass
                        elif isinstance(area, tuple) and len(area) == 4:
                            min_x, max_x, min_y, max_y = area
                        
                        # Check if NPC is within area bounds
                        if min_x is not None and not (min_x <= target_x_precheck <= max_x and min_y <= target_y_precheck <= max_y):
                            # NPC is outside area, don't skip camera movement
                            target_precheck = None
                
                if target_precheck:
                    # Get screen coordinates from NPC
                    rect_precheck = unwrap_rect(target_precheck.get("clickbox")) or unwrap_rect(target_precheck.get("bounds"))
                    canvas_precheck = target_precheck.get("canvas", {})
                    
                    # Check if NPC has valid screen coordinates
                    screen_x = None
                    screen_y = None
                    
                    if rect_precheck:
                        # Use center of clickbox/bounds
                        screen_x = rect_precheck.get("x", 0) + rect_precheck.get("width", 0) // 2
                        screen_y = rect_precheck.get("y", 0) + rect_precheck.get("height", 0) // 2
                    elif isinstance(canvas_precheck.get("x"), (int, float)) and isinstance(canvas_precheck.get("y"), (int, float)):
                        screen_x = int(canvas_precheck["x"])
                        screen_y = int(canvas_precheck["y"])
                    
                    # Check if NPC is within 800x600 clickable area centered on screen
                    if screen_x is not None and screen_y is not None:
                        where = ipc.where() or {}
                        screen_width = int(where.get("w", 0))
                        screen_height = int(where.get("h", 0))
                        
                        if screen_width > 0 and screen_height > 0:
                            # 800x600 box centered on screen
                            clickable_width = 800
                            clickable_height = 600
                            center_x = screen_width // 2
                            center_y = screen_height // 2
                            
                            min_x = center_x - clickable_width // 2
                            max_x = center_x + clickable_width // 2
                            min_y = center_y - clickable_height // 2
                            max_y = center_y + clickable_height // 2
                            
                            # Check if NPC is within clickable area
                            if min_x <= screen_x <= max_x and min_y <= screen_y <= max_y:
                                skip_camera_movement = True
                                logging.debug(f"[CLICK_NPC] NPC {npc_name} already visible at ({screen_x}, {screen_y}), skipping camera movement")

        # Try camera retry directions: first try without moving camera, then LEFT, then RIGHT
        camera_retry_directions = [None, "LEFT", "RIGHT"]
        camera_retry_duration = 0.5  # Move camera for 500ms
        
        for camera_retry_idx, direction in enumerate(camera_retry_directions):
            # Move camera if this is not the first attempt
            if direction is not None:
                try:
                    ipc.key_press(direction)
                    sleep_exponential(camera_retry_duration * 0.8, camera_retry_duration * 1.2, 1.0)
                    ipc.key_release(direction)
                except Exception:
                    pass  # Don't fail if camera movement fails
            
            # Try multiple attempts with different coordinates
            max_attempts = 3
            for attempt in range(max_attempts):
                # Move camera to target using path-aware logic if available (skip if already visible)
                if not skip_camera_movement:
                    from actions import player
                    player_x = player.get_x()
                    player_y = player.get_y()
                    
                    # Use new camera system
                    # Calculate distance for mode detection
                    if isinstance(player_x, int) and isinstance(player_y, int):
                        dx = abs(world_coords['x'] - player_x)
                        dy = abs(world_coords['y'] - player_y)
                        distance = dx + dy  # Manhattan distance
                    else:
                        distance = None
                    
                    # Use new camera system with auto-detection
                    from services.camera_integration import get_camera_state
                    camera_state, state_config, area_center, _ = get_camera_state()
                    
                    aim_camera_at_target(
                        target_world_coords=world_coords,
                        mode=None,  # Auto-detect
                        action_type="click_npc",
                        distance_to_target=distance,
                        path_waypoints=path_waypoints,
                        movement_state=movement_state,
                        max_ms=aim_ms,
                        object_world_coords=world_coords,  # Pass object coords for OBJECT_INTERACTION state
                        area_center=area_center,
                        disable_pitch=disable_pitch
                    )
                
                # STEP 3: NPC finding and rect resampling
                npc_resp = ipc.find_npc(npc_name)
                
                if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
                    continue
                
                target = npc_resp.get("npc")
                
                if not target:
                    continue
                
                # Verify NPC is within specified area (if area constraint provided)
                if area is not None:
                    target_world = target.get("world", {})
                    target_x = target_world.get("x")
                    target_y = target_world.get("y")
                    
                    if not isinstance(target_x, int) or not isinstance(target_y, int):
                        logging.debug(f"[CLICK_NPC] NPC {npc_name} has invalid world coordinates, skipping")
                        continue
                    
                    # Resolve area coordinates
                    min_x, max_x, min_y, max_y = None, None, None, None
                    if isinstance(area, str):
                        # Try to import constants
                        try:
                            from constants import REGIONS, BANK_REGIONS
                            if area in REGIONS:
                                min_x, max_x, min_y, max_y = REGIONS[area]
                            elif area in BANK_REGIONS:
                                min_x, max_x, min_y, max_y = BANK_REGIONS[area]
                            else:
                                logging.warning(f"[CLICK_NPC] Unknown area name: {area}")
                                continue
                        except ImportError:
                            logging.warning(f"[CLICK_NPC] Could not import constants to resolve area: {area}")
                            continue
                    elif isinstance(area, tuple) and len(area) == 4:
                        min_x, max_x, min_y, max_y = area
                    else:
                        logging.warning(f"[CLICK_NPC] Invalid area format: {area}")
                        continue
                    
                    # Check if NPC is within area bounds
                    if not (min_x <= target_x <= max_x and min_y <= target_y <= max_y):
                        logging.debug(f"[CLICK_NPC] NPC {npc_name} at ({target_x}, {target_y}) is outside area bounds, skipping")
                        continue
                
                # Get FRESH coordinates
                rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
                npc_name_fresh = target.get("name") or npc_name
                
                if rect:
                    cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0),
                                           rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
                    anchor = {"bounds": rect}
                    point = {"x": cx, "y": cy}
                elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
                    cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
                    anchor = {}
                    point = {"x": cx, "y": cy}
                else:
                    continue

                # Small delay to ensure hover is registered
                sleep_exponential(0.05, 0.15, 1.5)
            
                # STEP 6: Dispatch click
                step = {
                    "action": "click-npc-context",
                    "click": {
                        "type": "context-select",
                        "x": point["x"],
                        "y": point["y"],
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match
                    },
                    "option": action,
                    "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
                    "anchor": point,
                }
                
                result = dispatch(step)
                
                if result:
                    # Check if the correct interaction was performed
                    last_interaction = get_last_interaction()
                    
                    # Use exact match or contains based on exact_match parameter
                    clean_action = clean_rs(last_interaction.get("action", ""))
                    want_action = clean_rs(action).lower()
                    action_match = (clean_action.lower() == want_action) if exact_match else (
                        (want_action in clean_action.lower()) or (clean_action.lower() in want_action)
                    )
                    want = clean_rs(npc_name).lower()
                    tgt = clean_rs(last_interaction.get("target", "")).lower()
                    target_match = (tgt == want) if exact_match else ((want in tgt) or (tgt and (tgt in want)))
                    
                    if (last_interaction and action_match and target_match):
                        print(f"[CLICK] {npc_name} - interaction verified")
                        return result
                    else:
                        print(f"[CLICK] {npc_name} - incorrect interaction, retrying...")
                        continue
        
        return None
    
    except Exception as e:
        raise


def click_npc_with_camera_no_reacquire(
    target_npc: dict,
    action: str = None,
    world_coords: dict = None,
    aim_ms: int = 420,
    exact_match: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None
) -> dict | None:
    """
    Click an NPC with camera movement, WITHOUT re-acquiring a (potentially different) NPC from IPC.

    This is meant to be used when the caller has already selected/filtered a specific NPC (e.g. not
    in combat, not 0% hp) and wants to click *that* NPC, not whichever IPC reports as "closest"
    at click-time.
    
    Args:
        path_waypoints: Optional list of waypoint dicts from pathfinding for path-aware camera
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
    """
    try:
        if not isinstance(target_npc, dict):
            return None

        if not world_coords:
            # Prefer explicit world_coords; otherwise attempt to use the NPC's attached world coords
            world_coords = target_npc.get("world")
        if not isinstance(world_coords, dict):
            return None

        npc_name = (target_npc.get("name") or "").strip() or "Unknown"

        # Try camera retry directions: first try without moving camera, then LEFT, then RIGHT
        camera_retry_directions = [None, "LEFT", "RIGHT"]
        camera_retry_duration = 0.5  # Move camera for 500ms

        for _, direction in enumerate(camera_retry_directions):
            # Move camera if this is not the first attempt
            if direction is not None:
                try:
                    ipc.key_press(direction)
                    sleep_exponential(camera_retry_duration * 0.8, camera_retry_duration * 1.2, 1.0)
                    ipc.key_release(direction)
                except Exception:
                    pass  # Don't fail if camera movement fails

            # Try multiple attempts with different coordinates
            max_attempts = 3
            for _attempt in range(max_attempts):
                # Move camera to target using path-aware logic if available
                from actions import player
                player_x = player.get_x()
                player_y = player.get_y()
                
                # Use new camera system
                # Calculate distance for mode detection
                if isinstance(player_x, int) and isinstance(player_y, int):
                    dx = abs(world_coords["x"] - player_x)
                    dy = abs(world_coords["y"] - player_y)
                    distance = dx + dy  # Manhattan distance
                else:
                    distance = None
                
                # Use new camera system with auto-detection
                from services.camera_integration import get_camera_state
                camera_state, state_config, area_center, _ = get_camera_state()
                
                aim_camera_at_target(
                    target_world_coords=world_coords,
                    mode=None,  # Auto-detect
                    action_type="click_npc",
                    distance_to_target=distance,
                    path_waypoints=path_waypoints,
                    movement_state=movement_state,
                    max_ms=aim_ms,
                    object_world_coords=world_coords,  # Pass object coords for OBJECT_INTERACTION state
                    area_center=area_center
                )

                # Pick click point from the already-selected NPC (no IPC npc query here)
                rect = unwrap_rect(target_npc.get("clickbox")) or unwrap_rect(target_npc.get("bounds"))
                anchor = {}
                point = None

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
                    anchor = {"bounds": rect}
                    point = {"x": cx, "y": cy}
                else:
                    canvas = target_npc.get("canvas") if isinstance(target_npc.get("canvas"), dict) else None
                    if isinstance((canvas or {}).get("x"), (int, float)) and isinstance((canvas or {}).get("y"), (int, float)):
                        point = {"x": int(canvas["x"]), "y": int(canvas["y"])}
                    else:
                        # Last resort: tile center projection (still no NPC re-acquire)
                        proj = ipc.project_world_tile(int(world_coords["x"]), int(world_coords["y"])) or {}
                        if proj.get("ok") and proj.get("onscreen") and isinstance(proj.get("canvas"), dict):
                            point = {"x": int(proj["canvas"]["x"]), "y": int(proj["canvas"]["y"])}

                if not point:
                    continue

                # Small delay to ensure hover is registered
                sleep_exponential(0.05, 0.15, 1.5)

                step = {
                    "action": "click-npc-context",
                    "click": {
                        "type": "context-select",
                        "x": point["x"],
                        "y": point["y"],
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                        "exact_match": exact_match,
                    },
                    "option": action,
                    "target": {"domain": "npc", "name": npc_name, **anchor, "world": world_coords},
                    "anchor": point,
                }

                result = dispatch(step)
                if not result:
                    continue

                # Check if the correct interaction was performed
                last_interaction = get_last_interaction() or {}

                # Action verify: if action is None (left click), don't require action match
                clean_action = clean_rs(last_interaction.get("action", ""))
                if action is None:
                    action_match = True
                else:
                    action_match = (clean_action == action) if exact_match else (action in clean_action)

                # Target verify (prefer target_name which is already cleaned in the plugin)
                target_name = clean_rs(last_interaction.get("target_name") or last_interaction.get("target") or "")
                if exact_match:
                    target_match = target_name.lower() == npc_name.lower()
                else:
                    want = clean_rs(npc_name).lower()
                    tgt = target_name.lower()
                    target_match = (want in tgt) or (tgt and (tgt in want))

                if action_match and target_match:
                    print(f"[CLICK] {npc_name} - interaction verified (no reacquire)")
                    return result

                print(f"[CLICK] {npc_name} - incorrect interaction, retrying (no reacquire)...")

        return None

    except Exception:
        raise


def verify_walk_here_available():
    """Check if 'Walk here' is the top menu option"""
    menu_response = ipc.get_menu()
    if not menu_response.get("ok") or not menu_response.get("open"):
        return False
    
    entries = menu_response.get("entries", [])
    if not entries:
        return False
    
    # Find the entry with visualIndex: 0 (top option)
    top_option = None
    for entry in entries:
        if entry.get("visualIndex") == 0:
            top_option = entry
            break
    
    if not top_option:
        return False
    
    # Check if top option is "Walk here"
    return (top_option.get("option") == "Walk here" and 
            top_option.get("type") == "WALK")


def verify_action_available(action: str, target: str):
    """Check if a specific action and target combination is available in the menu"""
    menu_response = ipc.get_menu()
    if not menu_response.get("ok") or not menu_response.get("open"):
        return False
    
    entries = menu_response.get("entries", [])
    if not entries:
        return False
    
    # Look for entry where clean_rs(target) matches and option matches action
    for entry in entries:
        entry_target = entry.get("target", "")
        entry_option = entry.get("option", "")
        
        # Clean the target from the menu entry
        clean_target = clean_rs(entry_target)
        
        # Check if both action and target match
        if clean_target == target and entry_option == action:
            return entry.get("visualIndex")
    
    return False


def click_ground_with_camera(
    world_coords: dict,
    description: str = "Move",
    aim_ms: int = 700,
    waypoint_path: list = None,
    verify_tile: bool = False,
    expected_tile: dict = None,
    verify_path: bool = False,
    path_waypoints: list = None,
    movement_state: dict = None
) -> dict | None:
    """
    Click ground in the general direction of the destination, then verify the clicked tile is on the path.
    Uses shift+left-click for walking (RuneLite plugin makes "Walk here" top action when shift is held).
    
    This function:
    - Gets path for visualization and verification (visualize=True)
    - Calculates direction from player to destination
    - Clicks somewhere in that general direction (NOT on waypoints)
    - Verifies the clicked tile is on/near the intended path
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
        
        # Get movement state if not provided (for camera compensation)
        if movement_state is None:
            # Try to get movement state from IPC directly to avoid circular imports
            try:
                resp = ipc.get_player()
                if resp and resp.get("ok") and resp.get("player"):
                    player_data = resp.get("player", {})
                    current_x = player_data.get("worldX")
                    current_y = player_data.get("worldY")
                    pose_animation = player_data.get("poseAnimation", -1)
                    is_running = player_data.get("isRunning", False)
                    orientation = player_data.get("orientation", 0)
                    
                    # Simple movement state - just check if running
                    movement_state = {
                        "is_running": is_running,
                        "orientation": orientation
                    }
            except Exception:
                movement_state = None
        
        # Use new camera system for ground clicks
        # Calculate distance for mode detection
        if isinstance(player_x, int) and isinstance(player_y, int):
            dx = abs(click_x - player_x)
            dy = abs(click_y - player_y)
            distance = dx + dy  # Manhattan distance
        else:
            distance = None
        
        # Use new camera system with auto-detection
        from services.camera_integration import get_camera_state
        camera_state, state_config, area_center, _ = get_camera_state()
        
        aim_camera_at_target(
            target_world_coords={"x": click_x, "y": click_y},
            mode=None,  # Auto-detect
            action_type="click_ground",
            distance_to_target=distance,
            path_waypoints=path_waypoints,
            movement_state=movement_state,
            max_ms=aim_ms,
            object_world_coords=None,
            area_center=area_center
        )
        
        # Project the click tile to get screen coordinates
        proj, _ = ipc.project_many([{"x": click_x, "y": click_y}])
        
        if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
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
        
        # Canvas dimensions (default to common sizes, will adjust if needed)
        canvas_width = 1920
        canvas_height = 1080
        
        # Check if target is off-screen and adjust click to canvas edge in that direction
        if base_x < 0 or base_x >= canvas_width or base_y < 0 or base_y >= canvas_height:
            # Target is off-screen - click at canvas edge in the direction of the target
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # Calculate direction vector from canvas center to target
            dx = base_x - canvas_center_x
            dy = base_y - canvas_center_y
            
            # Calculate which edge we'll hit first
            # Find intersection with canvas edges
            edge_distance_from_center = random.uniform(30, 80)  # Human-like: 30-80px from edge
            
            if base_y < 0:
                # Target is above canvas - click at top edge
                cy = int(edge_distance_from_center)
                # Keep X direction but clamp to canvas, with slight randomization along edge
                if abs(dx) > 0:
                    # Project X to edge while maintaining direction
                    ratio = abs(dy) / abs(dx) if dx != 0 else 1
                    edge_x = canvas_center_x + int(dx / abs(dx) * min(abs(dx), canvas_width // 2 - 50))
                    cx = edge_x + random.randint(-30, 30)  # Randomize position along top edge
                else:
                    cx = canvas_center_x + random.randint(-50, 50)
                cx = max(50, min(canvas_width - 50, cx))  # Clamp to canvas with margin
            elif base_y >= canvas_height:
                # Target is below canvas - click at bottom edge
                cy = int(canvas_height - edge_distance_from_center)
                if abs(dx) > 0:
                    ratio = abs(dy) / abs(dx) if dx != 0 else 1
                    edge_x = canvas_center_x + int(dx / abs(dx) * min(abs(dx), canvas_width // 2 - 50))
                    cx = edge_x + random.randint(-30, 30)
                else:
                    cx = canvas_center_x + random.randint(-50, 50)
                cx = max(50, min(canvas_width - 50, cx))
            elif base_x < 0:
                # Target is left of canvas - click at left edge
                cx = int(edge_distance_from_center)
                if abs(dy) > 0:
                    ratio = abs(dx) / abs(dy) if dy != 0 else 1
                    edge_y = canvas_center_y + int(dy / abs(dy) * min(abs(dy), canvas_height // 2 - 50))
                    cy = edge_y + random.randint(-30, 30)
                else:
                    cy = canvas_center_y + random.randint(-50, 50)
                cy = max(50, min(canvas_height - 50, cy))
            elif base_x >= canvas_width:
                # Target is right of canvas - click at right edge
                cx = int(canvas_width - edge_distance_from_center)
                if abs(dy) > 0:
                    ratio = abs(dx) / abs(dy) if dy != 0 else 1
                    edge_y = canvas_center_y + int(dy / abs(dy) * min(abs(dy), canvas_height // 2 - 50))
                    cy = edge_y + random.randint(-30, 30)
                else:
                    cy = canvas_center_y + random.randint(-50, 50)
                cy = max(50, min(canvas_height - 50, cy))
            
            print(f"[CLICK] Ground - target off-screen (base: {base_x}, {base_y}), clicking at canvas edge ({cx}, {cy}) in target direction")
        else:
            # Target is on-screen - use coordinates with minor clamping for safety
            # Add small randomization to click position
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
            # Try moving the click point in different directions to find a valid location
            adjusted = False
            for offset_x, offset_y in [
                (0, -50), (0, 50), (-50, 0), (50, 0),  # Up, down, left, right
                (-30, -30), (30, -30), (-30, 30), (30, 30),  # Diagonals
                (0, -100), (0, 100), (-100, 0), (100, 0),  # Further in cardinal directions
            ]:
                test_x = cx + offset_x
                test_y = cy + offset_y
                
                # Ensure test point is still within canvas bounds
                if (margin <= test_x <= canvas_width - margin and 
                    margin <= test_y <= canvas_height - margin and
                    not is_point_in_blocking_widget(test_x, test_y)):
                    cx = test_x
                    cy = test_y
                    adjusted = True
                    print(f"[CLICK] Ground - click point blocked by UI widget, adjusted to ({cx}, {cy})")
                    break
            
            if not adjusted:
                # Couldn't find a valid point - this is a problem, but we'll try anyway
                print(f"[WARNING] Ground - click point ({cx}, {cy}) is blocked by UI widget and couldn't be adjusted")
        
        sleep_exponential(0.05, 0.15, 1.5)
                
        # Press shift before clicking (RuneLite plugin makes "Walk here" top action when shift is held)
        try:
            ipc.key_press("SHIFT")
        except Exception:
            pass  # Don't fail if key press fails
        
        # Perform a simple left click (shift is held, so "Walk here" will be the action)
        result = ipc.click(cx, cy, button=1)
        
        # Release shift after clicking
        try:
            ipc.key_release("SHIFT")
        except Exception:
            pass  # Don't fail if key release fails
        
        if not result:
            return None
        
        # Wait a short delay for the game to process the click and update selected tile
        import time
        time.sleep(0.05)  # 50ms delay
        
        # Get the ACTUAL selected tile from the game state (not from IPC response)
        selected_tile = ipc.get_selected_tile()
        
        # Check if the correct interaction was performed
        last_interaction = get_last_interaction()
        if not last_interaction or last_interaction.get("action") != "Walk here":
            print(f"[CLICK] Ground - incorrect interaction: {last_interaction}")
            return None
        
        # Verify the clicked tile matches the intended target
        if selected_tile:
            actual_x = selected_tile.get("x")
            actual_y = selected_tile.get("y")
            intended_x = click_x
            intended_y = click_y
            
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
                    print(f"[CLICK] Ground - tile mismatch: intended ({intended_x}, {intended_y}), actual ({actual_x}, {actual_y}), distance: {distance}, tolerance: {max_tolerance}")
                    return None
                # Tile matches within tolerance - success
        else:
            # No tile selected - might be acceptable for long-range clicks
            player_distance = abs(player_x - click_x) + abs(player_y - click_y)
            if player_distance <= 5:
                # Close range without tile data is suspicious
                print(f"[CLICK] Ground - no selected tile for close-range click")
                return None
            # Long range without tile data might be okay (tile might not be selectable)
        
        return result
    
    except Exception as e:
        raise
    