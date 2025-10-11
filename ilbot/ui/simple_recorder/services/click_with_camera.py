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
    door_plan: dict = None,  # NEW: For door-specific handling
    aim_ms: int = 420
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    """
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "click_timing",
        "who": "object",
        "action": action or "Interact",
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"aim_retries": 0, "rect_resamples": 0, "menu_retries": 0},
        "camera": {"yaw": None, "pitch": None, "scale": None},
        "context": {"player_running": False, "entity_moving": False}
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
        
        # STEP 3: Object finding and rect resampling
        t4_rect_resample = _mark_timing("rect_resample")
        if ("door" in object_name.lower() or "gate" in object_name.lower()) and door_plan:
            target = door_plan.get("door", {})
            if not target:
                timing_data["ok"] = False
                timing_data["error"] = "No door data in door plan"
                _emit_timing(timing_data)
                return None
        else:
            # Normal object finding for non-doors
            obj_resp = ipc.find_object(object_name, ["GAME", "WALL"])
            
            if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
                timing_data["ok"] = False
                timing_data["error"] = f"Could not find {object_name} after camera movement"
                _emit_timing(timing_data)
                return None
            
            target = obj_resp.get("object")
            
            if not target:
                timing_data["ok"] = False
                timing_data["error"] = f"Could not find {object_name} after camera movement"
                _emit_timing(timing_data)
                return None
        
        timing_data["dur_ms"]["resample"] = (t4_rect_resample - t3_cam_end) / 1_000_000
    
        # STEP 4: Coordinate processing
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
            timing_data["ok"] = False
            timing_data["error"] = f"Could not get fresh coordinates for {object_name}"
            _emit_timing(timing_data)
            return None
        
        # STEP 5: Hover positioning
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

        # STEP 6: Menu verification
        t6_menu_open = _mark_timing("menu_open")
        t7_menu_verified = _mark_timing("menu_verified")
        action_index = verify_action_available(action, object_name)
        timing_data["dur_ms"]["menu_open"] = (t6_menu_open - t5_hover_ready) / 1_000_000
        timing_data["dur_ms"]["menu_verify"] = (t7_menu_verified - t6_menu_open) / 1_000_000
        
        if not action_index and not action_index == 0:
            timing_data["ok"] = False
            timing_data["error"] = "Action not available in menu"
            _emit_timing(timing_data)
            return None
    
        # STEP 7: Dispatch click
        t8_click_down = _mark_timing("click_down")
        if not action or action_index is None or action_index == 0:
            step = {
                "action": "click-object",
                "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
                "target": {"domain": "object", "name": obj_name, **anchor, "world": world_coords},
            }
        else:
            step = {
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
            }

        result = dispatch(step)
        t9_click_up = _mark_timing("click_up")
        timing_data["dur_ms"]["dispatch"] = (t9_click_up - t8_click_down) / 1_000_000
        
        # STEP 8: Post-acknowledgment (simplified - just mark completion)
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
            print(f"[CLICK] {object_name}")
        return result
    
    except Exception as e:
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        _emit_timing(timing_data)
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
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "click_timing",
        "who": "ground",
        "action": "Walk-here",
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"aim_retries": 0, "rect_resamples": 0, "menu_retries": 0},
        "camera": {"yaw": None, "pitch": None, "scale": None},
        "context": {"player_running": True, "entity_moving": False}  # Ground clicks are for movement
    }
    
    try:
        if not world_coords:
            timing_data["ok"] = False
            timing_data["error"] = "No world coordinates provided"
            _emit_timing(timing_data)
            return None
        
        # Try pathfinding with decreasing max waypoints (20 down to 15)
        max_attempts = 3  # 20, 19, 18, 17, 16, 15
        initial_max_wps = 18
        
        for attempt in range(max_attempts):
            attempt_start = time.time()
            current_max_wps = initial_max_wps - attempt
            print(f"[CLICK_GROUND_TIMING] Attempt {attempt + 1}/{max_attempts}: max_wps={current_max_wps}")
            
            # Re-run pathfinding logic like in go_to to get fresh waypoint
            step_start = time.time()
            wps, dbg_path = ipc.path(rect=(world_coords['x']-1, world_coords['x']+1, world_coords['y']-1, world_coords['y']+1), max_wps=current_max_wps, visualize=False)
            print(f"[CLICK_GROUND_TIMING] Path generation took {time.time() - step_start:.3f}s")
            
            if not wps:
                print(f"[CLICK_GROUND_TIMING] No waypoints generated, trying next attempt")
                continue
            
            step_start = time.time()
            proj, dbg_proj = ipc.project_many(wps)
            proj = _merge_door_into_projection(wps, proj)
            print(f"[CLICK_GROUND_TIMING] Projection and door merge took {time.time() - step_start:.3f}s")
            
            usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
            if not usable:
                print(f"[CLICK_GROUND_TIMING] No usable waypoints after projection, trying next attempt")
                continue
            
            # Pick from the last 3 waypoints (or all if less than 3) - same logic as go_to
            if len(usable) <= 3:
                chosen = random.choice(usable)
            else:
                chosen = random.choice(usable[-3:])
            
            coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
            print(f"[CLICK_GROUND_TIMING] Selected coordinates: {coords}")
        
            # STEP 1: Target acquisition (pathfinding already done)
            t1_target_acquired = _mark_timing("target_acquired")
            timing_data["dur_ms"]["resolve"] = (t1_target_acquired - t0_start) / 1_000_000
            
            # STEP 2: Camera movement
            t2_cam_begin = _mark_timing("cam_begin")
            try:
                aim_midtop_at_world(coords['x'], coords['y'], max_ms=aim_ms)
            finally:
                t3_cam_end = _mark_timing("cam_end")
                timing_data["dur_ms"]["cam"] = (t3_cam_end - t2_cam_begin) / 1_000_000
            
            # STEP 3: Projection and coordinate resampling
            t4_rect_resample = _mark_timing("rect_resample")
            
            # Line 510: Second projection call (redundant!)
            t_proj2_start = _mark_timing("proj2_start")
            proj, _ = ipc.project_many([{"x": coords['x'], "y": coords['y'], "p": coords.get('p', 0)}])
            t_proj2_end = _mark_timing("proj2_end")
            timing_data["dur_ms"]["proj2_call"] = (t_proj2_end - t_proj2_start) / 1_000_000
            print(f"[LINE_TIMING] Line 510 (ipc.project_many): {(t_proj2_end - t_proj2_start) / 1_000_000:.3f}ms")
            
            # Line 512: Projection validation
            t_validate_start = _mark_timing("validate_start")
            if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
                timing_data["counts"]["rect_resamples"] += 1
                continue  # Try next attempt
            t_validate_end = _mark_timing("validate_end")
            timing_data["dur_ms"]["validate_proj"] = (t_validate_end - t_validate_start) / 1_000_000
            print(f"[LINE_TIMING] Line 512 (validate projection): {(t_validate_end - t_validate_start) / 1_000_000:.3f}ms")
            
            # Line 517: Canvas coordinate extraction
            t_canvas_start = _mark_timing("canvas_start")
            fresh_coords = proj[0]["canvas"]
            t_canvas_end = _mark_timing("canvas_end")
            timing_data["dur_ms"]["canvas_extract"] = (t_canvas_end - t_canvas_start) / 1_000_000
            print(f"[LINE_TIMING] Line 517 (canvas extraction): {(t_canvas_end - t_canvas_start) / 1_000_000:.3f}ms")
            
            # Line 518: X coordinate conversion
            t_x_start = _mark_timing("x_start")
            cx = int(fresh_coords["x"])
            t_x_end = _mark_timing("x_end")
            timing_data["dur_ms"]["x_convert"] = (t_x_end - t_x_start) / 1_000_000
            print(f"[LINE_TIMING] Line 518 (x coordinate): {(t_x_end - t_x_start) / 1_000_000:.3f}ms")
            
            # Line 519: Y coordinate conversion
            t_y_start = _mark_timing("y_start")
            cy = int(fresh_coords["y"])
            t_y_end = _mark_timing("y_end")
            timing_data["dur_ms"]["y_convert"] = (t_y_end - t_y_start) / 1_000_000
            print(f"[LINE_TIMING] Line 519 (y coordinate): {(t_y_end - t_y_start) / 1_000_000:.3f}ms")
            
            # Line 520: Timing calculation
            t_timing_calc_start = _mark_timing("timing_calc_start")
            timing_data["dur_ms"]["resample"] = (t4_rect_resample - t3_cam_end) / 1_000_000
            t_timing_calc_end = _mark_timing("timing_calc_end")
            timing_data["dur_ms"]["timing_calc"] = (t_timing_calc_end - t_timing_calc_start) / 1_000_000
            print(f"[LINE_TIMING] Line 520 (timing calculation): {(t_timing_calc_end - t_timing_calc_start) / 1_000_000:.3f}ms")

            # STEP 4: Hover positioning
            t5_hover_ready = _mark_timing("hover_ready")
            
            # Line 524: Hover click call
            t_hover_call_start = _mark_timing("hover_call_start")
            hover_result = ipc.click(cx, cy, hover_only=True)
            t_hover_call_end = _mark_timing("hover_call_end")
            timing_data["dur_ms"]["hover_call"] = (t_hover_call_end - t_hover_call_start) / 1_000_000
            print(f"[LINE_TIMING] Line 524 (ipc.click hover): {(t_hover_call_end - t_hover_call_start) / 1_000_000:.3f}ms")
            
            # Line 525: Hover timing calculation
            t_hover_calc_start = _mark_timing("hover_calc_start")
            timing_data["dur_ms"]["hover"] = (t5_hover_ready - t4_rect_resample) / 1_000_000
            t_hover_calc_end = _mark_timing("hover_calc_end")
            timing_data["dur_ms"]["hover_calc"] = (t_hover_calc_end - t_hover_calc_start) / 1_000_000
            print(f"[LINE_TIMING] Line 525 (hover timing calc): {(t_hover_calc_end - t_hover_calc_start) / 1_000_000:.3f}ms")
            
            # Line 527: Hover result check
            t_hover_check_start = _mark_timing("hover_check_start")
            if not hover_result.get("ok"):
                continue  # Try next attempt
            t_hover_check_end = _mark_timing("hover_check_end")
            timing_data["dur_ms"]["hover_check"] = (t_hover_check_end - t_hover_check_start) / 1_000_000
            print(f"[LINE_TIMING] Line 527 (hover result check): {(t_hover_check_end - t_hover_check_start) / 1_000_000:.3f}ms")
            
            # Line 531: Sleep call
            t_sleep_start = _mark_timing("sleep_start")
            time.sleep(0.1)
            t_sleep_end = _mark_timing("sleep_end")
            timing_data["dur_ms"]["sleep_100ms"] = (t_sleep_end - t_sleep_start) / 1_000_000
            print(f"[LINE_TIMING] Line 531 (time.sleep 0.1): {(t_sleep_end - t_sleep_start) / 1_000_000:.3f}ms")
            
            # STEP 5: Menu verification
            t6_menu_open = _mark_timing("menu_open")
            t7_menu_verified = _mark_timing("menu_verified")
            
            # Line 536: Menu verification call
            t_menu_verify_start = _mark_timing("menu_verify_start")
            walk_here_available = verify_walk_here_available()
            t_menu_verify_end = _mark_timing("menu_verify_end")
            timing_data["dur_ms"]["menu_verify_call"] = (t_menu_verify_end - t_menu_verify_start) / 1_000_000
            print(f"[LINE_TIMING] Line 536 (verify_walk_here_available): {(t_menu_verify_end - t_menu_verify_start) / 1_000_000:.3f}ms")
            
            # Line 537: Menu open timing calculation
            t_menu_open_calc_start = _mark_timing("menu_open_calc_start")
            timing_data["dur_ms"]["menu_open"] = (t6_menu_open - t5_hover_ready) / 1_000_000
            t_menu_open_calc_end = _mark_timing("menu_open_calc_end")
            timing_data["dur_ms"]["menu_open_calc"] = (t_menu_open_calc_end - t_menu_open_calc_start) / 1_000_000
            print(f"[LINE_TIMING] Line 537 (menu_open timing calc): {(t_menu_open_calc_end - t_menu_open_calc_start) / 1_000_000:.3f}ms")
            
            # Line 538: Menu verify timing calculation
            t_menu_verify_calc_start = _mark_timing("menu_verify_calc_start")
            timing_data["dur_ms"]["menu_verify"] = (t7_menu_verified - t6_menu_open) / 1_000_000
            t_menu_verify_calc_end = _mark_timing("menu_verify_calc_end")
            timing_data["dur_ms"]["menu_verify_calc"] = (t_menu_verify_calc_end - t_menu_verify_calc_start) / 1_000_000
            print(f"[LINE_TIMING] Line 538 (menu_verify timing calc): {(t_menu_verify_calc_end - t_menu_verify_calc_start) / 1_000_000:.3f}ms")
            
            # Line 540: Menu availability check
            t_menu_check_start = _mark_timing("menu_check_start")
            if not walk_here_available:
                timing_data["counts"]["menu_retries"] += 1
                continue  # Try next attempt
            t_menu_check_end = _mark_timing("menu_check_end")
            timing_data["dur_ms"]["menu_check"] = (t_menu_check_end - t_menu_check_start) / 1_000_000
            print(f"[LINE_TIMING] Line 540 (menu availability check): {(t_menu_check_end - t_menu_check_start) / 1_000_000:.3f}ms")
            
            # STEP 6: Dispatch click
            t8_click_down = _mark_timing("click_down")
            
            # Line 546: Step dictionary creation
            t_step_dict_start = _mark_timing("step_dict_start")
            step = {
                "action": "click-ground",
                "description": description,
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {"domain": "ground", "name": f"Ground→{description}", "world": coords},
            }
            t_step_dict_end = _mark_timing("step_dict_end")
            timing_data["dur_ms"]["step_dict_create"] = (t_step_dict_end - t_step_dict_start) / 1_000_000
            print(f"[LINE_TIMING] Line 546 (step dictionary creation): {(t_step_dict_end - t_step_dict_start) / 1_000_000:.3f}ms")
            
            # Line 553: Dispatch call
            t_dispatch_start = _mark_timing("dispatch_start")
            result = dispatch(step)
            t_dispatch_end = _mark_timing("dispatch_end")
            timing_data["dur_ms"]["dispatch_call"] = (t_dispatch_end - t_dispatch_start) / 1_000_000
            print(f"[LINE_TIMING] Line 553 (dispatch call): {(t_dispatch_end - t_dispatch_start) / 1_000_000:.3f}ms")
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
                print(f"[CLICK] Ground")
                return result
    
        # All attempts failed
        timing_data["ok"] = False
        timing_data["error"] = "All attempts failed"
        _emit_timing(timing_data)
        return None
    
    except Exception as e:
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        _emit_timing(timing_data)
        raise
    