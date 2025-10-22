"""
Centralized click with camera functionality.
Handles the 3-step process: camera movement, fresh coordinates, click.
"""
import time
import random
import json
from datetime import datetime

from ..helpers.rects import unwrap_rect, rect_center_xy
from ..helpers.utils import clean_rs
from ..services.camera_integration import aim_midtop_at_world
from ..helpers.runtime_utils import ui, ipc, dispatch
from ..helpers.navigation import _merge_door_into_projection

# Timing instrumentation
_TIMING_ENABLED = True
_TIMING_FILE = "click_with_camera.timing.jsonl"

def _mark_timing(label: str) -> int:
    """Mark a timing point and return nanoseconds timestamp."""
    return time.perf_counter_ns()

def _emit_timing(data: dict):
    """Emit timing data as JSONL."""
    if not _TIMING_ENABLED:
        return
    
    data["ts"] = datetime.now().isoformat()
    print(f"[CLICK_TIMING] {json.dumps(data)}")
    
    # Also write to file
    try:
        with open(_TIMING_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass  # Don't fail if we can't write to file


def click_object_with_camera(
    object_name: str, 
    action: str = None,
    world_coords: dict = None,
    click_coords: dict = None,  # {"x": cx, "y": cy} for the click point
    click_rect: dict = None,    # Rectangle bounds for anchor
    door_plan: dict = None,     # Door-specific plan with coordinates
    aim_ms: int = 420
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    Handles doors specially when door_plan is provided.
    """
    
    try:
        if not world_coords:
            return None

        aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
        
        # Handle doors specially if door_plan is provided
        if door_plan and door_plan.get('door'):
            door_data = door_plan['door']
            
            # Use door's specific coordinates
            if door_data.get('canvas'):
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
                    },
                    "target": ({"domain": "object", "name": object_name, "world": world_coords}),
                    "anchor": point,
                }

                result = dispatch(step)
                
                if result:
                    print(f"[CLICK] {object_name} ({action})")
                return result
        
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
            return None
        
        # Find the matching object by name
        matching_object = None
        for obj in objects_resp["objects"]:
            if obj.get("name", "").lower() == object_name.lower():
                matching_object = obj
                break
        
        if not matching_object or not matching_object.get("canvas"):
            return None
        
        # Use fresh coordinates from the object
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
            },
            "target": ({"domain": "object", "name": object_name, "world": world_coords}),
            "anchor": point,
        }

        result = dispatch(step)
        
        if result:
            print(f"[CLICK] {object_name}")
        return result
    
    except Exception as e:
        raise


def click_npc_with_camera(
    npc_name: str,
    action: str = None,
    world_coords: dict = None,
    aim_ms: int = 420
) -> dict | None:
    """
    Click an NPC with camera movement and fresh coordinate recalculation.
    """
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "click_timing",
        "who": "npc",
        "action": action or "Talk-to",
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"aim_retries": 0, "rect_resamples": 0, "menu_retries": 0},
        "camera": {"yaw": None, "pitch": None, "scale": None},
        "context": {"player_running": False, "entity_moving": True}  # NPCs are typically moving
    }
    
    try:
        if not world_coords:
            timing_data["ok"] = False
            timing_data["error"] = "No world coordinates provided"
            _emit_timing(timing_data)
            return None
    
        # STEP 1: Target acquisition
        t1_target_acquired = _mark_timing("target_acquired")
        timing_data["dur_ms"]["resolve"] = (t1_target_acquired - t0_start) / 1_000_000
        
        # STEP 2: Camera movement
        t2_cam_begin = _mark_timing("cam_begin")
        try:
            aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
        finally:
            t3_cam_end = _mark_timing("cam_end")
            timing_data["dur_ms"]["cam"] = (t3_cam_end - t2_cam_begin) / 1_000_000
        
        # STEP 3: NPC finding and rect resampling
        t4_rect_resample = _mark_timing("rect_resample")
        npc_resp = ipc.find_npc(npc_name)
        
        if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
            timing_data["ok"] = False
            timing_data["error"] = f"Could not find {npc_name} after camera movement"
            _emit_timing(timing_data)
            return None
        
        target = npc_resp.get("npc")
        
        if not target:
            timing_data["ok"] = False
            timing_data["error"] = f"Could not find {npc_name} after camera movement"
            _emit_timing(timing_data)
            return None
        
        # Get FRESH coordinates
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
            timing_data["ok"] = False
            timing_data["error"] = f"Could not get fresh coordinates for {npc_name}"
            _emit_timing(timing_data)
            return None
        
        timing_data["dur_ms"]["resample"] = (t4_rect_resample - t3_cam_end) / 1_000_000

        # STEP 4: Hover positioning
        t5_hover_ready = _mark_timing("hover_ready")
        hover_result = ipc.click(cx, cy, hover_only=True)
        timing_data["dur_ms"]["hover"] = (t5_hover_ready - t4_rect_resample) / 1_000_000
        
        if not hover_result.get("ok"):
            timing_data["ok"] = False
            timing_data["error"] = "Hover click failed"
            _emit_timing(timing_data)
            return None

        # Small delay to ensure hover is registered
        time.sleep(0.1)

        # STEP 5: Menu verification
        t6_menu_open = _mark_timing("menu_open")
        t7_menu_verified = _mark_timing("menu_verified")
        action_index = verify_action_available(action, npc_name)
        timing_data["dur_ms"]["menu_open"] = (t6_menu_open - t5_hover_ready) / 1_000_000
        timing_data["dur_ms"]["menu_verify"] = (t7_menu_verified - t6_menu_open) / 1_000_000
        
        if not action_index and not action_index == 0:
            timing_data["ok"] = False
            timing_data["error"] = "Action not available in menu"
            _emit_timing(timing_data)
            return None
    
        # STEP 6: Dispatch click
        t8_click_down = _mark_timing("click_down")
        if not action or action_index is None or action_index == 0:
            step = {
                "action": "click-npc",
                "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
                "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
            }
        else:
            step = {
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
            }
        
        result = dispatch(step)
        t9_click_up = _mark_timing("click_up")
        timing_data["dur_ms"]["dispatch"] = (t9_click_up - t8_click_down) / 1_000_000
        
        # STEP 7: Post-acknowledgment
        t10_post_ack = _mark_timing("post_ack")
        timing_data["dur_ms"]["post_ack"] = (t10_post_ack - t9_click_up) / 1_000_000
        timing_data["dur_ms"]["total"] = (t10_post_ack - t0_start) / 1_000_000
        
        # Get camera state for context
        try:
            camera_resp = ipc.get_camera()
            if camera_resp and camera_resp.get("ok"):
                timing_data["camera"]["yaw"] = camera_resp.get("yaw")
                timing_data["camera"]["pitch"] = camera_resp.get("pitch")
                timing_data["camera"]["scale"] = camera_resp.get("scale")
        except Exception:
            pass
        
        _emit_timing(timing_data)
        
        if result:
            print(f"[CLICK] {npc_name}")
        return result
    
    except Exception as e:
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        _emit_timing(timing_data)
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
                    time.sleep(camera_retry_duration)
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
                
                fresh_coords = proj[0]["canvas"]
                cx = int(fresh_coords["x"])
                cy = int(fresh_coords["y"])
                
                time.sleep(0.1)
                
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
                    if direction is not None:
                        print(f"[CLICK] Ground (after camera {direction})")
                    else:
                        print(f"[CLICK] Ground")
                    return result
        return None
    
    except Exception as e:
        raise
    