"""
Centralized click with camera functionality.
Handles the 3-step process: camera movement, fresh coordinates, click.
"""
import random

from helpers import unwrap_rect
from helpers.utils import clean_rs, sleep_exponential, rect_beta_xy
from ..services.camera_integration import aim_midtop_at_world
from helpers import ipc, dispatch
from helpers.navigation import _merge_door_into_projection


def click_object_with_camera(
    object_name: str, 
    action: str = None,
    world_coords: dict = None,
    click_coords: dict = None,  # {"x": cx, "y": cy} for the click point
    click_rect: dict = None,    # Rectangle bounds for anchor
    door_plan: dict = None,     # Door-specific plan with coordinates
    aim_ms: int = 420,
    exact_match: bool = False
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    Handles doors specially when door_plan is provided.
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
                # Move camera to target
                aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
                
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
                        from helpers import get_last_interaction
                        last_interaction = get_last_interaction()
                        
                        # Use exact match or contains based on exact_match parameter
                        clean_action = clean_rs(last_interaction.get("action", ""))
                        action_match = (clean_action.lower() == action.lower()) if exact_match else (action.lower() in clean_action.lower())
                        target_match = (last_interaction.get("target_name", "").lower() == object_name.lower()) if exact_match else (object_name.lower() in last_interaction.get("target_name", "").lower())
                        
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
                    name=object_name
                )
                
                if not objects_resp.get("ok") or not objects_resp.get("objects"):
                    continue
                
                # Find the matching object by name
                matching_object = None
                for obj in objects_resp["objects"]:
                    if obj.get("name", "").lower() == object_name.lower():
                        matching_object = obj
                        break
                
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
                    from helpers import get_last_interaction
                    last_interaction = get_last_interaction()
                    
                    # Use exact match or contains based on exact_match parameter
                    clean_action = clean_rs(last_interaction.get("action", ""))
                    action_match = (clean_action.lower() == action.lower()) if exact_match else (action.lower() in clean_action.lower())
                    target_match = (clean_rs(last_interaction.get("target", "")).lower() == object_name.lower()) if exact_match else (object_name.lower() in clean_rs(last_interaction.get("target", "")).lower())
                    
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
    exact_match: bool = False
) -> dict | None:
    """
    Click an NPC with camera movement and fresh coordinate recalculation.
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
                # Move camera to target
                aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
                
                # STEP 3: NPC finding and rect resampling
                npc_resp = ipc.find_npc(npc_name)
                
                if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
                    continue
                
                target = npc_resp.get("npc")
                
                if not target:
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
                    from helpers import get_last_interaction
                    last_interaction = get_last_interaction()
                    
                    # Use exact match or contains based on exact_match parameter
                    clean_action = clean_rs(last_interaction.get("action", ""))
                    action_match = (clean_action == action) if exact_match else (action in clean_action)
                    target_match = (clean_rs(last_interaction.get("target", "")).lower() == npc_name.lower()) if exact_match else (npc_name.lower() in clean_rs(last_interaction.get("target", "")).lower())
                    
                    if (last_interaction and action_match and target_match):
                        print(f"[CLICK] {npc_name} - interaction verified")
                        return result
                    else:
                        print(f"[CLICK] {npc_name} - incorrect interaction, retrying...")
                        continue
        
        return None
    
    except Exception as e:
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
    waypoint_path: list = None
) -> dict | None:
    """
    Click ground with camera movement and fresh coordinate recalculation.
    If verify_walk_here_available returns false, retry with different tiles from waypoint path.
    """
    try:
        if not world_coords:
            return None
        
        # Try pathfinding with decreasing max waypoints (20 down to 15)
        max_attempts = 3  # 20, 19, 18, 17, 16, 15
        initial_max_wps = 18
        
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
            
            for attempt in range(max_attempts):
                current_max_wps = initial_max_wps - attempt
                
                # Re-run pathfinding logic like in go_to to get fresh waypoint
                wps, dbg_path = ipc.path(rect=(world_coords['x']-1, world_coords['x']+1, world_coords['y']-1, world_coords['y']+1), visualize=False)
                
                if not wps:
                    continue
                
                proj, dbg_proj = ipc.project_many(wps)
                proj = _merge_door_into_projection(wps, proj)
                
                usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
                if not usable:
                    continue
                
                # Pick from the last 3 waypoints (or all if less than 3) - same logic as go_to
                if len(usable) <= 3:
                    chosen = random.choice(usable)
                else:
                    chosen = random.choice(usable[-3:])
                
                coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
                
                # STEP 2: Camera movement
                aim_midtop_at_world(coords['x'], coords['y'], max_ms=aim_ms)
                
                proj, _ = ipc.project_many([{"x": coords['x'], "y": coords['y'], "p": coords.get('p', 0)}])
                
                # Get fresh coordinates - check bounds first, then canvas
                proj_data = proj[0]
                bounds = proj_data.get("bounds", {})
                if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                    cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                           bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
                else:
                    # Fallback to canvas coordinates
                    fresh_coords = proj_data["canvas"]
                    cx = int(fresh_coords["x"])
                    cy = int(fresh_coords["y"])
                
                sleep_exponential(0.05, 0.15, 1.5)
                
                step = {
                    "action": "click-ground-context",
                    "option": "Walk here",
                    "click": {
                        "type": "context-select",
                        "index": 0,
                        "x": cx,
                        "y": cy,
                        "row_height": 16,
                        "start_dy": 10,
                        "open_delay_ms": 120,
                    },
                    "target": {"domain": "ground", "name": "", "world": coords},
                    "anchor": {"x": cx, "y": cy},
                }
                
                result = dispatch(step)
                
                if result:
                    # Check if the correct interaction was performed
                    from helpers import get_last_interaction
                    last_interaction = get_last_interaction()
                    
                    if last_interaction and last_interaction.get("action") == "Walk here":
                        return result
                    else:
                        print(f"[CLICK] Ground - incorrect interaction, retrying...")
                        # Continue to next attempt in the same camera direction
                        continue
        return None
    
    except Exception as e:
        raise
    