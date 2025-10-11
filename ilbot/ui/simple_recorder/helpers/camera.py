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
    print(f"[CAMERA_TIMING] {json.dumps(data)}")
    
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
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "camera_setup_timing",
        "target_scale": target_scale,
        "target_pitch": target_pitch,
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"key_presses": 0, "scrolls": 0, "checks": 0},
        "camera": {"yaw": None, "pitch": None, "scale": None}
    }
    
    print(f"[CAMERA] Setting up camera for optimal view (Scale: {target_scale}, Pitch: {target_pitch})...")
    
    import time
    
    def hold_until_condition(condition_func, key, max_duration=5.0):
        """Hold a key until condition is met or max duration reached"""
        start_time = time.time()
        check_interval = 0.05  # Check every 50ms
        
        # Press the key down
        ipc.key_press(key)
        timing_data["counts"]["key_presses"] += 1
        
        try:
            # Monitor the condition while key is held down
            while time.time() - start_time < max_duration:
                timing_data["counts"]["checks"] += 1
                if condition_func():
                    return True
                time.sleep(check_interval)
        finally:
            # Always release the key when done
            ipc.key_release(key)
        
        return False
    
    def scroll_until_condition(condition_func, max_duration=5.0):
        """Scroll until condition is met or max duration reached"""
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while time.time() - start_time < max_duration:
            timing_data["counts"]["checks"] += 1
            if condition_func():
                return True
            
            # Send scroll command
            ipc.scroll(-1)
            timing_data["counts"]["scrolls"] += 1
            time.sleep(check_interval)
        
        return False
    
    try:
        # Get initial camera state
        camera_stats = get_camera_stats()
        if not camera_stats or not camera_stats.get("ok"):
            timing_data["ok"] = False
            timing_data["error"] = "Could not get initial camera stats"
            _emit_timing(timing_data)
            print("[CAMERA] Could not get initial camera stats")
            return False
        
        current_scale = camera_stats.get("scale", 512)
        current_pitch = camera_stats.get("pitch", 0)
        
        print(f"[CAMERA] Starting from Scale: {current_scale}, Pitch: {current_pitch}")
        
        # First, handle zoom (scale) - scroll out until we're close enough
        if current_scale > target_scale + 30:
            print(f"[CAMERA] Zooming out from {current_scale} to ~{target_scale}")
            
            def scale_good():
                stats = get_camera_stats()
                if not stats or not stats.get("ok"):
                    return False
                return abs(stats.get("scale", 512) - target_scale) <= 30
            
            if not scroll_until_condition(scale_good, max_duration=8.0):
                timing_data["ok"] = False
                timing_data["error"] = "Zoom adjustment timed out"
                _emit_timing(timing_data)
                print("[CAMERA] Zoom adjustment timed out")
                return False
        
        # Then, handle pitch - hold UP until we're close enough
        if current_pitch < target_pitch - 30:
            print(f"[CAMERA] Pitching up from {current_pitch} to ~{target_pitch}")
            
            def pitch_good():
                stats = get_camera_stats()
                if not stats or not stats.get("ok"):
                    return False
                return abs(stats.get("pitch", 0) - target_pitch) <= 30
            
            if not hold_until_condition(pitch_good, "UP", max_duration=8.0):
                timing_data["ok"] = False
                timing_data["error"] = "Pitch adjustment timed out"
                _emit_timing(timing_data)
                print("[CAMERA] Pitch adjustment timed out")
                return False
        
        # Final check
        final_stats = get_camera_stats()
        if final_stats and final_stats.get("ok"):
            final_scale = final_stats.get("scale", 512)
            final_pitch = final_stats.get("pitch", 0)
            timing_data["camera"]["yaw"] = final_stats.get("yaw")
            timing_data["camera"]["pitch"] = final_pitch
            timing_data["camera"]["scale"] = final_scale
            t1_complete = _mark_timing("complete")
            timing_data["dur_ms"]["total"] = (t1_complete - t0_start) / 1_000_000
            _emit_timing(timing_data)
            print(f"[CAMERA] Camera setup complete! Scale: {final_scale}, Pitch: {final_pitch}")
            return True
        
        timing_data["ok"] = False
        timing_data["error"] = "Could not get final stats"
        _emit_timing(timing_data)
        print("[CAMERA] Camera setup failed - could not get final stats")
        return False
    
    except Exception as e:
        t2_error = _mark_timing("error")
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        timing_data["dur_ms"]["total"] = (t2_error - t0_start) / 1_000_000
        _emit_timing(timing_data)
        print(f"[CAMERA] Camera setup failed with error: {e}")
        return False