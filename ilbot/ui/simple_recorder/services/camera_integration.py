# camera_integration.py
import time
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send
from ilbot.ui.simple_recorder.helpers.context import get_payload

def aim_midtop_at_world(wx: int, wy: int, *, max_ms: int = 600, payload: dict | None = None):
    """
    Aim camera at world coordinates using key press -> continuous check -> key release.
    """
    try:
        import time
        
        # Get window size
        where = ipc_send({"cmd": "where"}) or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))
        if W <= 0 or H <= 0:
            return

        # Target position (center horizontally, 30% down vertically)
        center_x = W // 2
        target_y = int(H * 0.30)
        
        # Dead zones with buffer room
        x_dead_zone = int(W * 0.12)  # Increased from 0.08 to 0.12
        y_dead_zone = int(H * 0.18)  # Increased from 0.12 to 0.18
        scale_dead_zone = 20  # Increased from 10 to 20
        pitch_dead_zone = 30  # New buffer for pitch
        
        # Continuous checking with timeout
        max_aim_time = 5.0  # Maximum 5 seconds for aiming
        check_interval = 0.05  # Check every 50ms for more responsive control
        start_time = time.time()
        
        # Track which keys are currently pressed
        pressed_keys = set()
        
        try:
            while time.time() - start_time < max_aim_time:
                # Project world coordinates to screen
                proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
                if not proj.get("ok"):
                    time.sleep(check_interval)
                    continue

                # Get current camera state
                camera = ipc_send({"cmd": "get_camera"}) or {}
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
                    return  # Target point is in proper position, we're done
                
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
                            ipc_send({"cmd": "keyRelease", "key": "DOWN"})
                            pressed_keys.discard("DOWN")
                        # Press UP
                        ipc_send({"cmd": "keyPress", "key": "UP"})
                        pressed_keys.add("UP")
                # elif needs_pitch:
                #     pitch_key = "DOWN" if dy > 0 else "UP"
                #     if pitch_key not in pressed_keys:
                #         # Release the other pitch key first
                #         other_pitch_key = "DOWN" if pitch_key == "UP" else "UP"
                #         if other_pitch_key in pressed_keys:
                #             ipc_send({"cmd": "keyRelease", "key": other_pitch_key})
                #             pressed_keys.discard(other_pitch_key)
                #         # Press the needed pitch key
                #         ipc_send({"cmd": "keyPress", "key": pitch_key})
                #         pressed_keys.add(pitch_key)
                else:
                    # Release pitch keys if we don't need them
                    for pitch_key in ["UP", "DOWN"]:
                        if pitch_key in pressed_keys:
                            ipc_send({"cmd": "keyRelease", "key": pitch_key})
                            pressed_keys.discard(pitch_key)
                
                # Handle yaw adjustments
                if needs_yaw:
                    yaw_key = "LEFT" if dx > 0 else "RIGHT"
                    if yaw_key not in pressed_keys:
                        # Release the other yaw key first
                        other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                        if other_yaw_key in pressed_keys:
                            ipc_send({"cmd": "keyRelease", "key": other_yaw_key})
                            pressed_keys.discard(other_yaw_key)
                        # Press the needed yaw key
                        ipc_send({"cmd": "keyPress", "key": yaw_key})
                        pressed_keys.add(yaw_key)
                else:
                    # Release yaw keys if we don't need them
                    for yaw_key in ["LEFT", "RIGHT"]:
                        if yaw_key in pressed_keys:
                            ipc_send({"cmd": "keyRelease", "key": yaw_key})
                            pressed_keys.discard(yaw_key)
                
                # Handle scale adjustments (scroll doesn't need press/release)
                if needs_scale:
                    if current_scale > 551:
                        ipc_send({"cmd": "scroll", "amount": -1})  # Zoom out
                    else:
                        ipc_send({"cmd": "scroll", "amount": 1})   # Zoom in
                
                # Wait before next check
                time.sleep(check_interval)
        
        finally:
            # Always release all pressed keys when done
            for key in pressed_keys.copy():
                ipc_send({"cmd": "keyRelease", "key": key})
                pressed_keys.discard(key)

    except Exception as e:
        pass  # Silent fail

def dispatch_with_camera(step: dict, *, ui, payload: dict, aim_ms: int = 600, 
                        async_camera: bool = False, camera_delay_ms: int = 100):
    """
    Execute step with camera aiming.
    """
    # Extract target coordinates
    target = step.get("target", {})
    if not target:
        return ui.dispatch(step)
    
    # Get world coordinates
    world_coords = target.get("world")
    if not world_coords:
        return ui.dispatch(step)
    
    wx = world_coords.get("x")
    wy = world_coords.get("y")
    
    if wx is None or wy is None:
        return ui.dispatch(step)
    
    # Aim camera at target
    aim_midtop_at_world(wx, wy, max_ms=aim_ms)
    
    # Execute the step
    return ui.dispatch(step)