# camera_integration.py
import logging
import time
import json
from datetime import datetime

from ilbot.ui.simple_recorder.actions import player
from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc, dispatch

# Timing instrumentation
_TIMING_ENABLED = True
_TIMING_FILE = "camera_integration.timing.jsonl"

def _mark_timing(label: str) -> int:
    """Mark a timing point and return nanoseconds timestamp."""
    return time.perf_counter_ns()

def _emit_timing(data: dict):
    """Emit timing data as JSONL."""
    if not _TIMING_ENABLED:
        return
    
    data["ts"] = datetime.now().isoformat()
    print(f"[CAMERA_TIMING] {json.dumps(data)}")
    
    # Also write to file
    try:
        with open(_TIMING_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass  # Don't fail if we can't write to file

def aim_midtop_at_world(wx: int, wy: int, *, max_ms: int = 600):
    """
    Aim camera at world coordinates using key press -> continuous check -> key release.
    """
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "camera_timing",
        "target_x": wx,
        "target_y": wy,
        "max_ms": max_ms,
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"aim_retries": 0, "key_presses": 0, "projections": 0},
        "camera": {"yaw": None, "pitch": None, "scale": None}
    }
    
    try:
        # Check if target is at the same tile as player's current position
        player_x, player_y = player.get_player_position()
        if player_x == wx and player_y == wy:
            # Target is at same tile as player, no need to move camera
            t1_same_tile = _mark_timing("same_tile")
            timing_data["dur_ms"]["aim_total"] = (t1_same_tile - t0_start) / 1_000_000
            timing_data["camera"]["yaw"] = None
            timing_data["camera"]["pitch"] = None
            timing_data["camera"]["scale"] = None
            _emit_timing(timing_data)
            return

        # Get window size
        where = ipc.where() or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))
        if W <= 0 or H <= 0:
            timing_data["ok"] = False
            timing_data["error"] = "Invalid window size"
            _emit_timing(timing_data)
            return

        # Target position (center horizontally, 30% down vertically)
        center_x = W // 2
        target_y = int(H * 0.30)
        
        # Dead zones with buffer room
        pitch_dead_zone = 30  # New buffer for pitch
        
        # Continuous checking with timeout
        max_aim_time = 5.0  # Maximum 5 seconds for aiming
        check_interval = 0.05  # Check every 50ms for more responsive control
        start_time = time.time()
        
        # Track which keys are currently pressed
        pressed_keys = set()
        
        try:
            t1_first_projection = _mark_timing("first_projection")
            timing_data["dur_ms"]["first_projection"] = (t1_first_projection - t0_start) / 1_000_000
            
            while time.time() - start_time < max_aim_time:
                # Project world coordinates to screen
                proj = ipc.project_world_tile(int(wx), int(wy)) or {}
                timing_data["counts"]["projections"] += 1
                
                if not proj.get("ok"):
                    time.sleep(check_interval)
                    continue

                # Get current camera state
                camera = ipc.get_camera() or {}
                current_pitch = int(camera.get("pitch", 0))
                current_scale = int(camera.get("scale", 551))

                # Calculate screen position
                screen_x = int(proj['canvas'].get("x", 0))
                screen_y = int(proj['canvas'].get("y", 0))
                
                # Calculate offsets
                dx = screen_x - center_x
                dy = screen_y - target_y
                
                # Check if target point is in proper screen position
                x_in_range = 500 <= screen_x <= 1100
                y_in_range = screen_y < 500
                
                if x_in_range and y_in_range:
                    # Target point is in proper position, we're done
                    t2_aim_complete = _mark_timing("aim_complete")
                    timing_data["dur_ms"]["aim_total"] = (t2_aim_complete - t0_start) / 1_000_000
                    timing_data["camera"]["yaw"] = camera.get("yaw")
                    timing_data["camera"]["pitch"] = current_pitch
                    timing_data["camera"]["scale"] = current_scale
                    _emit_timing(timing_data)
                    return
                
                # Determine what adjustments are needed
                needs_pitch_up = current_pitch < (300 - pitch_dead_zone)
                # needs_pitch = not y_close and not needs_pitch_up
                needs_yaw = not x_in_range or not y_in_range
                needs_scale = not (500 <= current_scale <= 600)  # Check if scale is in reasonable range
                
                # Handle pitch adjustments
                if needs_pitch_up:
                    if "UP" not in pressed_keys:
                        # Release any other pitch keys first
                        if "DOWN" in pressed_keys:
                            ipc.key_release("DOWN")
                            pressed_keys.discard("DOWN")
                        # Press UP
                        ipc.key_press("UP")
                        pressed_keys.add("UP")
                        timing_data["counts"]["key_presses"] += 1
                # elif needs_pitch:
                #     pitch_key = "DOWN" if dy > 0 else "UP"
                #     if pitch_key not in pressed_keys:
                #         # Release the other pitch key first
                #         other_pitch_key = "DOWN" if pitch_key == "UP" else "UP"
                #         if other_pitch_key in pressed_keys:
                #             ipc_send({"cmd": "key_release", "key": other_pitch_key})
                #             pressed_keys.discard(other_pitch_key)
                #         # Press the needed pitch key
                #         ipc_send({"cmd": "key_press", "key": pitch_key})
                #         pressed_keys.add(pitch_key)
                else:
                    # Release pitch keys if we don't need them
                    for pitch_key in ["UP", "DOWN"]:
                        if pitch_key in pressed_keys:
                            ipc.key_release(pitch_key)
                            pressed_keys.discard(pitch_key)
                
                # Handle yaw adjustments
                if needs_yaw:
                    yaw_key = "LEFT" if dx > 0 else "RIGHT"
                    if yaw_key not in pressed_keys:
                        # Release the other yaw key first
                        other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                        if other_yaw_key in pressed_keys:
                            ipc.key_release(other_yaw_key)
                            pressed_keys.discard(other_yaw_key)
                        # Press the needed yaw key
                        ipc.key_press(yaw_key)
                        pressed_keys.add(yaw_key)
                        timing_data["counts"]["key_presses"] += 1
                else:
                    # Release yaw keys if we don't need them
                    for yaw_key in ["LEFT", "RIGHT"]:
                        if yaw_key in pressed_keys:
                            ipc.key_release(yaw_key)
                            pressed_keys.discard(yaw_key)
                
                # Handle scale adjustments (scroll doesn't need press/release)
                if needs_scale:
                    if current_scale > 551:
                        ipc.scroll(-1)  # Zoom out
                    else:
                        ipc.scroll(1)   # Zoom in
                
                # Wait before next check
                time.sleep(check_interval)
        
        finally:
            # Always release all pressed keys when done
            for key in pressed_keys.copy():
                ipc.key_release(key)
                pressed_keys.discard(key)
            
            # Emit timing data if we haven't already
            if "aim_total" not in timing_data["dur_ms"]:
                t3_timeout = _mark_timing("timeout")
                timing_data["dur_ms"]["aim_total"] = (t3_timeout - t0_start) / 1_000_000
                timing_data["ok"] = False
                timing_data["error"] = "Timeout"
                _emit_timing(timing_data)

    except Exception as e:
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        _emit_timing(timing_data)
        logging.error(f"[AIM_CAMERA] Failed to aim camera at world coordinates ({wx}, {wy}): {e}")
        # Release any keys that might still be pressed
        try:
            for key in ["UP", "DOWN", "LEFT", "RIGHT"]:
                ipc.key_release(key)
        except Exception:
            pass

def dispatch_with_camera(step: dict, *, aim_ms: int = 600,
                        async_camera: bool = False, camera_delay_ms: int = 100):
    """
    Execute step with camera aiming.
    """
    # Extract target coordinates
    target = step.get("target", {})
    if not target:
        return dispatch(step)
    
    # Get world coordinates
    world_coords = target.get("world")
    if not world_coords:
        return dispatch(step)
    
    wx = world_coords.get("x")
    wy = world_coords.get("y")
    
    if wx is None or wy is None:
        return dispatch(step)
    
    # Aim camera at target
    aim_midtop_at_world(wx, wy, max_ms=aim_ms)
    
    # Execute the step
    return dispatch(step)