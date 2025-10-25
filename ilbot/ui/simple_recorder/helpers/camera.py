import logging
from typing import Optional, Dict, Any
import time
import json
from datetime import datetime

from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc

# Timing instrumentation
_TIMING_ENABLED = True
_TIMING_FILE = "camera.timing.jsonl"

def _mark_timing(label: str) -> int:
    """Mark a timing point and return nanoseconds timestamp."""
    return time.perf_counter_ns()

def _emit_timing(data: dict):
    """Emit timing data as JSONL."""
    if not _TIMING_ENABLED:
        return
    
    data["ts"] = datetime.now().isoformat()
    
    # Also write to file
    try:
        with open(_TIMING_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass  # Don't fail if we can't write to file


def get_camera_stats() -> Optional[Dict[str, Any]]:
    """
    Get real-time camera statistics directly from the client.
    """
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "camera_stats_timing",
        "ok": True,
        "error": None,
        "dur_ms": {},
        "camera": {"yaw": None, "pitch": None, "scale": None}
    }
    
    try:
        resp = ipc.get_camera()
        t1_complete = _mark_timing("complete")
        timing_data["dur_ms"]["total"] = (t1_complete - t0_start) / 1_000_000
        
        if resp and resp.get("ok"):
            timing_data["camera"]["yaw"] = resp.get("yaw")
            timing_data["camera"]["pitch"] = resp.get("pitch")
            timing_data["camera"]["scale"] = resp.get("scale")
            _emit_timing(timing_data)
            return resp
        else:
            timing_data["ok"] = False
            timing_data["error"] = f"Failed to get camera stats: {resp}"
            _emit_timing(timing_data)
            print(f"[CAMERA] Failed to get camera stats: {resp}")
            return None
    except Exception as e:
        t2_error = _mark_timing("error")
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        timing_data["dur_ms"]["total"] = (t2_error - t0_start) / 1_000_000
        _emit_timing(timing_data)
        print(f"[CAMERA] Error getting camera stats: {e}")
        return None


def read_camera_scale() -> Optional[int]:
    """
    Get the current camera scale (zoom level) in real-time.
    """
    stats = get_camera_stats()
    if stats and stats.get("ok"):
        return stats.get("scale")
    return None


def setup_camera_optimal(target_scale: int = 551, target_pitch: int = 383) -> bool:
    """
    Set up camera for optimal bot operation using continuous key holds with real-time checking.
    
    Args:
        target_scale: Target camera scale (lower = more zoomed out, default 551)
        target_pitch: Target camera pitch (higher = more upward angle, default 383)
        
    Returns:
        True if setup completed successfully, False otherwise
    """
    
    import time
    
    try:
        # Get initial camera state
        camera_stats = get_camera_stats()
        if not camera_stats or not camera_stats.get("ok"):
            print("[CAMERA] Could not get initial camera stats")
            return False
        
        current_scale = camera_stats.get("scale", 512)
        current_pitch = camera_stats.get("pitch", 0)
        
        # First, handle zoom (scale) - scroll out until we're close enough
        if current_scale > target_scale + 30:
            
            def scale_good():
                stats = get_camera_stats()
                if not stats or not stats.get("ok"):
                    return False
                return abs(stats.get("scale", 512) - target_scale) <= 30
            
            if not scroll_until_condition(scale_good, max_duration=8.0):
                print("[CAMERA] Zoom adjustment timed out")
                return False
        
        # Then, handle pitch - hold UP until we're close enough
        if current_pitch < target_pitch - 30:
            
            def pitch_good():
                stats = get_camera_stats()
                if not stats or not stats.get("ok"):
                    return False
                return abs(stats.get("pitch", 0) - target_pitch) <= 30
            
            if not hold_until_condition(pitch_good, "UP", max_duration=8.0):
                print("[CAMERA] Pitch adjustment timed out")
                return False
        
        # Final check
        final_stats = get_camera_stats()
        if final_stats and final_stats.get("ok"):
            return True

        print("[CAMERA] Camera setup failed - could not get final stats")
        return False
    
    except Exception as e:
        print(f"[CAMERA] Camera setup failed with error: {e}")
        return False

def hold_until_condition(condition_func, key, max_duration=5.0):
    """Hold a key until condition is met or max duration reached"""
    start_time = time.time()
    check_interval = 0.05  # Check every 50ms

    # Press the key down
    ipc.key_press(key)

    try:
        # Monitor the condition while key is held down
        while time.time() - start_time < max_duration:
            if condition_func():
                return True
            from .utils import sleep_exponential
            sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
    finally:
        # Always release the key when done
        ipc.key_release(key)

    return False

def scroll_until_condition(condition_func, max_duration=5.0):
    """Scroll until condition is met or max duration reached"""
    start_time = time.time()
    check_interval = 0.1  # Check every 100ms

    while time.time() - start_time < max_duration:
        if condition_func():
            return True

        # Send scroll command
        ipc.scroll(-1)
        time.sleep(check_interval)

    return False

def move_camera_random() -> bool:
    """Move the camera randomly with yaw, pitch, and scroll movements."""
    try:
        import random
        
        # Get initial camera position
        initial_stats = get_camera_stats()
        if not initial_stats or not initial_stats.get("ok"):
            print(f"[CAMERA] Could not get initial camera stats")
            return False
        
        initial_yaw = initial_stats.get("yaw", 0)
        initial_pitch = initial_stats.get("pitch", 0)
        
        # Random movements with minimum thresholds
        yaw_amount = random.randint(200, 400) if random.choice([True, False]) else random.randint(-400, -200)
        pitch_amount = random.randint(50, 100) if random.choice([True, False]) else random.randint(-100, -50)
        scroll_amount = random.randint(5, 10) if random.choice([True, False]) else random.randint(-10, -5)
        
        print(f"[CAMERA] Random movements - Yaw: {yaw_amount}, Pitch: {pitch_amount}, Scroll: {scroll_amount}")
        
        # Use existing functions for each movement
        success = True
        
        # Handle yaw movement
        if yaw_amount != 0:
            def yaw_condition():
                current_stats = get_camera_stats()
                if not current_stats or not current_stats.get("ok"):
                    return False
                current_yaw = current_stats.get("yaw", 0)
                yaw_change = abs(current_yaw - initial_yaw)
                return yaw_change >= abs(yaw_amount) / 10
            
            key = "RIGHT" if yaw_amount > 0 else "LEFT"
            direction = "right" if yaw_amount > 0 else "left"
            print(f"[CAMERA] Moving yaw {direction} by {abs(yaw_amount)} pixels")
            if not hold_until_condition(yaw_condition, key, max_duration=2.0):
                print(f"[CAMERA] Yaw movement timed out")
                success = False
        
        # Handle pitch movement
        if pitch_amount != 0:
            def pitch_condition():
                current_stats = get_camera_stats()
                if not current_stats or not current_stats.get("ok"):
                    return False
                current_pitch = current_stats.get("pitch", 0)
                pitch_change = abs(current_pitch - initial_pitch)
                return pitch_change >= abs(pitch_amount) / 10
            
            key = "UP" if pitch_amount > 0 else "DOWN"
            direction = "up" if pitch_amount > 0 else "down"
            print(f"[CAMERA] Moving pitch {direction} by {abs(pitch_amount)} pixels")
            if not hold_until_condition(pitch_condition, key, max_duration=2.0):
                print(f"[CAMERA] Pitch movement timed out")
                success = False
        
        # Handle scroll movement
        if scroll_amount != 0:
            scroll_direction = "in" if scroll_amount > 0 else "out"
            print(f"[CAMERA] Scrolling zoom {scroll_direction} by {abs(scroll_amount)} steps")
            
            for _ in range(abs(scroll_amount)):
                if scroll_amount > 0:
                    ipc.scroll(1)  # Scroll in
                else:
                    ipc.scroll(-1)  # Scroll out
                from .utils import sleep_exponential
                sleep_exponential(0.05, 0.15, 1.5)
            
            print(f"[CAMERA] Completed {abs(scroll_amount)} scroll steps")
        
        return success
        
    except Exception as e:
        print(f"[CAMERA] Could not move camera: {e}")
        return False