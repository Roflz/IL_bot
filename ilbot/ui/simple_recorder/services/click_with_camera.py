"""
Centralized click with camera functionality.
Handles the 3-step process: camera movement, fresh coordinates, click.
"""
import time
from ..helpers.context import get_payload, get_ui
from ..helpers.ipc import ipc_send
from ..helpers.rects import unwrap_rect, rect_center_xy
from .camera_integration import aim_midtop_at_world_iterative


def _wait_for_camera_movement_complete(payload: dict, max_wait_ms: int = 2000, check_interval_ms: int = 50) -> None:
    """
    Wait for camera movement to complete by checking if camera values are stable.
    
    Args:
        payload: Game payload
        max_wait_ms: Maximum time to wait for camera movement to complete
        check_interval_ms: Interval between camera value checks
    """
    print(f"[CLICK_WITH_CAMERA] Waiting for camera movement to complete (max {max_wait_ms}ms)...")
    
    start_time = time.time()
    previous_scale = None
    previous_pitch = None
    previous_yaw = None
    stable_count = 0
    required_stable_checks = 3  # Need 3 consecutive stable readings
    
    while (time.time() - start_time) * 1000 < max_wait_ms:
        # Get current camera values
        camera_resp = ipc_send({"cmd": "get_camera"}, payload)
        if camera_resp and camera_resp.get("ok"):
            camera_data = camera_resp.get("camera", {})
            current_scale = camera_data.get("scale")
            current_pitch = camera_data.get("pitch")
            current_yaw = camera_data.get("yaw")
            
            # Check if values are different from previous
            if (previous_scale is not None and 
                previous_pitch is not None and 
                previous_yaw is not None):
                
                scale_changed = abs(current_scale - previous_scale) > 0.001
                pitch_changed = abs(current_pitch - previous_pitch) > 0.001
                yaw_changed = abs(current_yaw - previous_yaw) > 0.001
                
                if scale_changed or pitch_changed or yaw_changed:
                    # Camera is still moving
                    stable_count = 0
                    print(f"[CLICK_WITH_CAMERA] Camera still moving: scale={current_scale:.3f}, pitch={current_pitch:.3f}, yaw={current_yaw:.3f}")
                else:
                    # Camera values are stable
                    stable_count += 1
                    print(f"[CLICK_WITH_CAMERA] Camera stable check {stable_count}/{required_stable_checks}")
                    
                    if stable_count >= required_stable_checks:
                        print(f"[CLICK_WITH_CAMERA] Camera movement completed after {int((time.time() - start_time) * 1000)}ms")
                        return
            
            # Update previous values
            previous_scale = current_scale
            previous_pitch = current_pitch
            previous_yaw = current_yaw
        
        time.sleep(check_interval_ms / 1000.0)
    
    print(f"[CLICK_WITH_CAMERA] Camera movement wait timed out after {max_wait_ms}ms")


def click_object_with_camera(
    object_name: str, 
    action: str = None, 
    action_index: int = None,
    world_coords: dict = None,
    ui=None,
    payload: dict = None,
    aim_ms: int = 420
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    
    Args:
        object_name: Name of the object to click
        action: Specific action to perform (optional)
        action_index: Index of the action in context menu (optional)
        world_coords: World coordinates of the object
        ui: UI instance
        payload: Game payload
        aim_ms: Camera movement time in ms
        
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for {object_name}")
        return None
    
    # STEP 1: Move camera to aim at object (NO CLICK YET)
    print(f"[CLICK_WITH_CAMERA] Moving camera to aim at {object_name} at world coords ({world_coords['x']}, {world_coords['y']})")
    aim_midtop_at_world_iterative(world_coords['x'], world_coords['y'], max_ms=aim_ms, payload=payload)
    
    # Wait for camera movement to complete by checking if camera values are stable
    print(f"[CLICK_WITH_CAMERA] Waiting for camera movement to complete...")
    _wait_for_camera_movement_complete(payload)
    
    # STEP 2: After camera movement, get FRESH coordinates using OPTIMIZED IPC command
    print(f"[CLICK_WITH_CAMERA] Camera movement completed, getting fresh object coordinates")
    fresh_payload = get_payload()
    obj_resp = ipc_send({"cmd": "find_object", "name": object_name, "types": ["GAME"]}, fresh_payload)
    
    if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
        print(f"[CLICK_WITH_CAMERA] Could not find {object_name} after camera movement")
        return None
    
    target = obj_resp.get("object")
    
    if not target:
        print(f"[CLICK_WITH_CAMERA] Could not find {object_name} after camera movement")
        return None
    
    # Get FRESH coordinates using SAME logic as before
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    obj_name = target.get("name") or object_name
    
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
        cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        print(f"[CLICK_WITH_CAMERA] Could not get fresh coordinates for {object_name}")
        return None
    
    print(f"[CLICK_WITH_CAMERA] Fresh coordinates after camera movement: ({cx}, {cy})")
    
    # STEP 3: Click with FRESH coordinates (no camera movement)
    from ..actions.runtime import emit
    
    if not action or action_index is None or action_index == 0:
        step = emit({
            "action": "click-object",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "object", "name": obj_name, **anchor, "world": world_coords},
        })
    else:
        step = emit({
            "action": "click-object-context",
            "click": {
                "type": "context-select",
                "index": int(action_index),
                "x": point["x"],
                "y": point["y"],
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "target": ({"domain": "object", "name": obj_name, **anchor, "world": world_coords}
                       if rect else {"domain": "object", "name": obj_name, "world": world_coords}),
            "anchor": point,
        })
    
    # Click with fresh coordinates
    result = ui.dispatch(step)
    if result:
        print(f"[CLICK_WITH_CAMERA] Successfully clicked {object_name}")
    return result


def click_npc_with_camera(
    npc_name: str,
    action: str = None,
    action_index: int = None,
    world_coords: dict = None,
    ui=None,
    payload: dict = None,
    aim_ms: int = 420
) -> dict | None:
    """
    Click an NPC with camera movement and fresh coordinate recalculation.
    
    Args:
        npc_name: Name of the NPC to click
        action: Specific action to perform (optional)
        action_index: Index of the action in context menu (optional)
        world_coords: World coordinates of the NPC
        ui: UI instance
        payload: Game payload
        aim_ms: Camera movement time in ms
        
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for {npc_name}")
        return None
    
    # STEP 1: Move camera to aim at NPC (NO CLICK YET)
    print(f"[CLICK_WITH_CAMERA] Moving camera to aim at {npc_name} at world coords ({world_coords['x']}, {world_coords['y']})")
    aim_midtop_at_world_iterative(world_coords['x'], world_coords['y'], max_ms=aim_ms, payload=payload)
    
    # Wait for camera movement to complete by checking if camera values are stable
    print(f"[CLICK_WITH_CAMERA] Waiting for camera movement to complete...")
    _wait_for_camera_movement_complete(payload)
    
    # STEP 2: After camera movement, get FRESH coordinates using OPTIMIZED IPC command
    print(f"[CLICK_WITH_CAMERA] Camera movement completed, getting fresh NPC coordinates")
    fresh_payload = get_payload()
    npc_resp = ipc_send({"cmd": "find_npc", "name": npc_name}, fresh_payload)
    
    if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
        print(f"[CLICK_WITH_CAMERA] Could not find {npc_name} after camera movement")
        return None
    
    target = npc_resp.get("npc")
    
    if not target:
        print(f"[CLICK_WITH_CAMERA] Could not find {npc_name} after camera movement")
        return None
    
    # Get FRESH coordinates using SAME logic as before
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    npc_name_fresh = target.get("name") or npc_name
    
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
        cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        print(f"[CLICK_WITH_CAMERA] Could not get fresh coordinates for {npc_name}")
        return None
    
    print(f"[CLICK_WITH_CAMERA] Fresh coordinates after camera movement: ({cx}, {cy})")
    
    # STEP 3: Click with FRESH coordinates (no camera movement)
    from ..actions.runtime import emit
    
    if not action or action_index is None or action_index == 0:
        step = emit({
            "action": "click-npc",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
        })
    else:
        step = emit({
            "action": "click-npc-context",
            "click": {
                "type": "context-select",
                "index": int(action_index),
                "x": point["x"],
                "y": point["y"],
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "option": action,
            "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
            "anchor": point,
        })
    
    # Click with fresh coordinates
    result = ui.dispatch(step)
    if result:
        print(f"[CLICK_WITH_CAMERA] Successfully clicked {npc_name}")
    return result


def click_ground_with_camera(
    world_coords: dict,
    description: str = "Move",
    ui=None,
    payload: dict = None,
    aim_ms: int = 700
) -> dict | None:
    """
    Click ground with camera movement and fresh coordinate recalculation.
    
    Args:
        world_coords: World coordinates of the ground click
        description: Description of the movement
        ui: UI instance
        payload: Game payload
        aim_ms: Camera movement time in ms
        
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for ground click")
        return None
    
    # STEP 1: Move camera to aim at ground point (NO CLICK YET)
    print(f"[CLICK_WITH_CAMERA] Moving camera to aim at ground at world coords ({world_coords['x']}, {world_coords['y']})")
    aim_midtop_at_world_iterative(world_coords['x'], world_coords['y'], max_ms=aim_ms, payload=payload)
    
    # Wait for camera movement to complete by checking if camera values are stable
    print(f"[CLICK_WITH_CAMERA] Waiting for camera movement to complete...")
    _wait_for_camera_movement_complete(payload)
    
    # STEP 2: After camera movement, get FRESH coordinates using SAME IPC command
    print(f"[CLICK_WITH_CAMERA] Camera movement completed, getting fresh ground coordinates")
    fresh_payload = get_payload()
    
    # For ground clicks, we need to project the world coordinates to screen coordinates
    from ..helpers.ipc import ipc_project_many
    proj, _ = ipc_project_many(fresh_payload, [{"x": world_coords['x'], "y": world_coords['y'], "p": world_coords.get('p', 0)}])
    
    if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
        print(f"[CLICK_WITH_CAMERA] Could not project ground coordinates after camera movement")
        return None
    
    # Get FRESH screen coordinates
    fresh_coords = proj[0]["canvas"]
    cx = int(fresh_coords["x"])
    cy = int(fresh_coords["y"])
    
    print(f"[CLICK_WITH_CAMERA] Fresh coordinates after camera movement: ({cx}, {cy})")
    
    # STEP 3: Click with FRESH coordinates (no camera movement)
    from ..actions.runtime import emit
    
    step = emit({
        "action": "click-ground",
        "description": description,
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "ground", "name": f"Ground→{description}", "world": world_coords},
    })
    
    # Click with fresh coordinates
    result = ui.dispatch(step)
    if result:
        print(f"[CLICK_WITH_CAMERA] Successfully clicked ground")
    return result
