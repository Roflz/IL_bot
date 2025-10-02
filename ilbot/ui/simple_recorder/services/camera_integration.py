# camera_integration.py
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send
from ilbot.ui.simple_recorder.helpers.context import get_payload

INVERT_PITCH = False
PITCH_UP_KEY   = "UP"
PITCH_DOWN_KEY = "DOWN"

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def aim_midtop_at_world(
    wx: int,
    wy: int,
    *,
    max_ms: int = 450,
    x_dead_frac: float = 0.05,
    y_target_frac: float = 0.24,   # keep target high → less need to pitch down
    y_band_frac: float = 0.12,     # wider band → fewer corrections
    # asym hysteresis: be stricter before pitching DOWN
    y_band_down_mult: float = 2.2,
    y_band_up_mult: float = 1.0,
    # make down nudges gentler + hard-cap them
    pitch_down_bias: float = 0.45,
    pitch_down_cap_ms: int = 95,
    # zoom regulation (Scale only): lock to fully zoomed out
    target_scale: int = 551,
    scale_tol: int = 1,
    payload: dict | None = None,
):
    """
    One-tick nudge:
      - yaw if off-center in X
      - pitch toward mid-top band with asymmetric hysteresis & capped DOWN holds
      - regulate zoom using ONLY camera.Scale: keep near 551 (fully zoomed out)
    """
    print(f"[CAMERA DEBUG] aim_midtop_at_world called with wx={wx}, wy={wy}")
    try:
        where = ipc_send({"cmd": "where"}) or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))
        print(f"[CAMERA DEBUG] Window size: W={W}, H={H}")
        if W <= 0 or H <= 0:
            print(f"[CAMERA DEBUG] Invalid window size, returning")
            return

        proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
        print(f"[CAMERA DEBUG] Projection result: {proj}")
        if not proj.get("ok"):
            print(f"[CAMERA DEBUG] Projection failed, returning")
            return

        if not proj.get("onscreen"):
            print(f"[CAMERA DEBUG] Target offscreen, rotating camera right")
            ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.6)})
            return

        cx = int(proj["canvas"]["x"])
        cy = int(proj["canvas"]["y"])
        print(f"[CAMERA DEBUG] Canvas coords: cx={cx}, cy={cy}")

        # if the projected point sits outside the window, treat it as offscreen
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            print(f"[CAMERA DEBUG] Target outside window bounds, rotating camera right")
            ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.85)})
            return

        target_x = W // 2
        target_y = int(H * y_target_frac)
        print(f"[CAMERA DEBUG] Target position: target_x={target_x}, target_y={target_y}")

        dx = cx - target_x
        dy = cy - target_y
        print(f"[CAMERA DEBUG] Offsets: dx={dx}, dy={dy}")

        # --- yaw ---
        x_dead = W * x_dead_frac
        print(f"[CAMERA DEBUG] X dead zone: {x_dead}")
        if abs(dx) > x_dead:
            yaw_ms = _clamp(int((abs(dx) / (W * 0.6)) * max_ms), 80, max_ms)
            yaw_key = "LEFT" if dx > 0 else "RIGHT"
            print(f"[CAMERA DEBUG] Yawing {yaw_key} for {yaw_ms}ms")
            result = ipc_send({"cmd": "keyHold", "key": yaw_key, "ms": yaw_ms})
            print(f"[CAMERA DEBUG] Yaw IPC result: {result}")
        else:
            print(f"[CAMERA DEBUG] No yaw needed (within dead zone)")

        # --- pitch with asymmetric hysteresis ---
        base_band   = H * y_band_frac
        y_band_down = base_band * y_band_down_mult   # stricter to pitch DOWN
        y_band_up   = base_band * y_band_up_mult
        print(f"[CAMERA DEBUG] Pitch bands: down={y_band_down}, up={y_band_up}")

        need_down = dy > y_band_down   # tile below target band (positive dy = below)
        need_up   = dy < -y_band_up    # tile above target band (negative dy = above)
        print(f"[CAMERA DEBUG] Pitch needs: down={need_down}, up={need_up}")

        if need_down or need_up:
            up_key, down_key = (PITCH_DOWN_KEY, PITCH_UP_KEY) if INVERT_PITCH else (PITCH_UP_KEY, PITCH_DOWN_KEY)
            pitch_ms = _clamp(int((abs(dy) / (H * 0.5)) * max_ms), 60, max_ms)
            if need_down:
                pitch_ms = int(pitch_ms * pitch_down_bias)
                pitch_ms = min(pitch_ms, pitch_down_cap_ms)   # <- kills the "last 2 dips"
                print(f"[CAMERA DEBUG] Pitching {down_key} for {pitch_ms}ms")
                result = ipc_send({"cmd": "keyHold", "key": down_key, "ms": max(30, pitch_ms)})
                print(f"[CAMERA DEBUG] Pitch DOWN IPC result: {result}")
            else:
                print(f"[CAMERA DEBUG] Pitching {up_key} for {pitch_ms}ms")
                result = ipc_send({"cmd": "keyHold", "key": up_key, "ms": max(40, pitch_ms)})
                print(f"[CAMERA DEBUG] Pitch UP IPC result: {result}")
        else:
            print(f"[CAMERA DEBUG] No pitch needed (within target band)")

        try:
            cam = (payload or {}).get("camera") or {}
            scale = cam.get("Scale", None)  # no 'Zoom' anywhere
            if isinstance(scale, (int, float)):
                # Only ever scroll OUT if we're tighter than the target.
                if scale > (target_scale + scale_tol):
                    ipc_send({"cmd": "scroll", "amount": -1})
                # Never scroll IN; do nothing otherwise.
        except Exception:
            pass

    except Exception as e:
        print(f"[CAMERA DEBUG] Exception in aim_midtop_at_world: {e}")
        import traceback
        traceback.print_exc()


def aim_midtop_at_world_iterative(
    wx: int,
    wy: int,
    *,
    max_ms: int = 450,
    max_iterations: int = 10,
    x_dead_frac: float = 0.05,
    y_target_frac: float = 0.24,
    y_band_frac: float = 0.12,
    y_band_down_mult: float = 2.2,
    y_band_up_mult: float = 1.0,
    pitch_down_bias: float = 0.45,
    pitch_down_cap_ms: int = 95,
    target_scale: int = 551,
    scale_tol: int = 1,
    payload: dict | None = None,
):
    """
    Iteratively move camera until target is in the correct position.
    Keeps moving until target is within the target area or max iterations reached.
    """
    print(f"[CAMERA DEBUG] aim_midtop_at_world_iterative called with wx={wx}, wy={wy}")
    
    for iteration in range(max_iterations):
        print(f"[CAMERA DEBUG] Iteration {iteration + 1}/{max_iterations}")
        
        try:
            where = ipc_send({"cmd": "where"}) or {}
            W, H = int(where.get("w", 0)), int(where.get("h", 0))
            if W <= 0 or H <= 0:
                print(f"[CAMERA DEBUG] Invalid window size, breaking")
                break

            proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
            if not proj.get("ok"):
                print(f"[CAMERA DEBUG] Projection failed, breaking")
                break

            if not proj.get("onscreen"):
                print(f"[CAMERA DEBUG] Target offscreen, rotating camera right")
                ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.6)})
                time.sleep(0.3)  # Wait for rotation
                continue

            cx = int(proj["canvas"]["x"])
            cy = int(proj["canvas"]["y"])

            # if the projected point sits outside the window, treat it as offscreen
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                print(f"[CAMERA DEBUG] Target outside window bounds, rotating camera right")
                ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.85)})
                time.sleep(0.3)  # Wait for rotation
                continue

            target_x = W // 2
            target_y = int(H * y_target_frac)
            
            dx = cx - target_x
            dy = cy - target_y
            
            print(f"[CAMERA DEBUG] Current position: cx={cx}, cy={cy}, target_x={target_x}, target_y={target_y}")
            print(f"[CAMERA DEBUG] Offsets: dx={dx}, dy={dy}")

            # Check if we're close enough to the target position
            x_dead = W * x_dead_frac
            base_band = H * y_band_frac
            y_band_down = base_band * y_band_down_mult
            y_band_up = base_band * y_band_up_mult
            
            x_close = abs(dx) <= x_dead
            y_close = -y_band_up <= dy <= y_band_down
            
            print(f"[CAMERA DEBUG] Close enough? x_close={x_close} (dx={dx} <= {x_dead}), y_close={y_close} (dy={dy} in [-{y_band_up}, {y_band_down}])")
            
            if x_close and y_close:
                print(f"[CAMERA DEBUG] Target is in correct position! Stopping camera movement.")
                break

            # Move camera
            moved = False
            
            # Yaw movement
            if not x_close:
                yaw_ms = _clamp(int((abs(dx) / (W * 0.6)) * max_ms), 80, max_ms)
                yaw_key = "LEFT" if dx > 0 else "RIGHT"
                print(f"[CAMERA DEBUG] Yawing {yaw_key} for {yaw_ms}ms")
                result = ipc_send({"cmd": "keyHold", "key": yaw_key, "ms": yaw_ms})
                print(f"[CAMERA DEBUG] Yaw IPC result: {result}")
                moved = True
                time.sleep(0.2)  # Wait for yaw movement

            # Pitch movement
            if not y_close:
                need_down = dy > y_band_down
                need_up = dy < -y_band_up
                
                if need_down or need_up:
                    up_key, down_key = (PITCH_DOWN_KEY, PITCH_UP_KEY) if INVERT_PITCH else (PITCH_UP_KEY, PITCH_DOWN_KEY)
                    pitch_ms = _clamp(int((abs(dy) / (H * 0.5)) * max_ms), 60, max_ms)
                    
                    if need_down:
                        pitch_ms = int(pitch_ms * pitch_down_bias)
                        pitch_ms = min(pitch_ms, pitch_down_cap_ms)
                        print(f"[CAMERA DEBUG] Pitching {down_key} for {pitch_ms}ms")
                        result = ipc_send({"cmd": "keyHold", "key": down_key, "ms": max(30, pitch_ms)})
                        print(f"[CAMERA DEBUG] Pitch DOWN IPC result: {result}")
                    else:
                        print(f"[CAMERA DEBUG] Pitching {up_key} for {pitch_ms}ms")
                        result = ipc_send({"cmd": "keyHold", "key": up_key, "ms": max(40, pitch_ms)})
                        print(f"[CAMERA DEBUG] Pitch UP IPC result: {result}")
                    moved = True
                    time.sleep(0.2)  # Wait for pitch movement

            if not moved:
                print(f"[CAMERA DEBUG] No movement needed, breaking")
                break
                
            # Wait for camera movement to settle
            time.sleep(0.3)
            
        except Exception as e:
            print(f"[CAMERA DEBUG] Exception in iteration {iteration + 1}: {e}")
            break
    
    print(f"[CAMERA DEBUG] Camera movement completed after {iteration + 1} iterations")


# if your current dispatch_with_camera doesn't pass payload, use this version:
def dispatch_with_camera(step: dict, *, ui, payload: dict, aim_ms: int = 450, 
                        async_camera: bool = False, camera_delay_ms: int = 100):
    """
    Dispatch action with camera movement to aim at target.
    
    Args:
        step: Action step to execute
        ui: UI instance
        payload: Game state payload
        aim_ms: Maximum time for camera aiming
        async_camera: If True, camera movement happens asynchronously (not recommended)
        camera_delay_ms: Delay before starting camera movement
        
    Returns:
        Dispatch result
    """
    # Always do camera movement first (synchronously) to ensure proper aiming
    try:
        # Handle both dict and list formats
        if isinstance(step, list) and len(step) > 0:
            step_dict = step[0]  # Take first element if it's a list
            print(f"[CAMERA DEBUG] Step was list format, extracted dict: {step_dict.get('action', 'unknown')}")
        elif isinstance(step, dict):
            step_dict = step
            print(f"[CAMERA DEBUG] Step was dict format: {step_dict.get('action', 'unknown')}")
        else:
            print(f"[CAMERA DEBUG] Unexpected step format: {type(step)}, skipping camera movement")
            step_dict = None
        
        if step_dict:
            tgt = (step_dict.get("target") or {}).get("world") or {}
            wx, wy = tgt.get("x"), tgt.get("y")
            print(f"[CAMERA DEBUG] Target world coords: x={wx}, y={wy}, type_x={type(wx)}, type_y={type(wy)}")
            print(f"[CAMERA DEBUG] Full target: {tgt}")
            print(f"[CAMERA DEBUG] Full step_dict: {step_dict}")
            
            if isinstance(wx, int) and isinstance(wy, int):
                print(f"[CAMERA DEBUG] Starting camera movement to ({wx}, {wy})")
                # Small delay before camera movement
                import time
                time.sleep(camera_delay_ms / 1000.0)
                
                # Aim camera at target with iterative movement
                aim_midtop_at_world_iterative(wx, wy, max_ms=aim_ms, payload=payload)
                print(f"[CAMERA DEBUG] Camera movement completed")
                
                # Wait for camera movement to settle
                time.sleep(200 / 1000.0)  # 200ms delay after camera movement
            else:
                print(f"[CAMERA DEBUG] Invalid world coordinates: x={wx}, y={wy}")
        else:
            print(f"[CAMERA DEBUG] No valid step dict found, skipping camera movement")
    except Exception as e:
        print(f"[CAMERA DEBUG] Exception in camera movement: {e}")
        import traceback
        traceback.print_exc()
    
    # Execute the main action with updated coordinates
    out = ui.dispatch(step)
    
    return out
