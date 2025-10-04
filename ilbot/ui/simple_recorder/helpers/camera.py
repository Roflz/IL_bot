from ilbot.ui.simple_recorder.helpers.ipc import ipc_send
from ilbot.ui.simple_recorder.helpers.utils import now_ms
from typing import Optional, Dict, Any

def get_camera_stats(payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Get real-time camera statistics directly from the client.
    """
    if payload is None:
        from ilbot.ui.simple_recorder.helpers.context import get_payload
        payload = get_payload()
    
    if not payload:
        return None
    
    try:
        resp = ipc_send({"cmd": "get_camera"})
        if resp and resp.get("ok"):
            return resp
        else:
            print(f"[CAMERA] Failed to get camera stats: {resp}")
            return None
    except Exception as e:
        print(f"[CAMERA] Error getting camera stats: {e}")
        return None


def read_camera_scale(payload: Optional[Dict] = None) -> Optional[int]:
    """
    Get the current camera scale (zoom level) in real-time.
    """
    stats = get_camera_stats(payload)
    if stats and stats.get("ok"):
        return stats.get("scale")
    return None


def setup_camera_optimal(payload: Optional[Dict] = None, target_scale: int = 551, target_pitch: int = 383) -> bool:
    """
    Set up camera for optimal bot operation using continuous key holds with real-time checking.
    
    Args:
        payload: Optional payload, will get fresh if None
        target_scale: Target camera scale (lower = more zoomed out, default 551)
        target_pitch: Target camera pitch (higher = more upward angle, default 383)
        
    Returns:
        True if setup completed successfully, False otherwise
    """
    if payload is None:
        from ilbot.ui.simple_recorder.helpers.context import get_payload
        payload = get_payload()
    
    if not payload:
        print("[CAMERA] No payload available for camera setup")
        return False
    
    print(f"[CAMERA] Setting up camera for optimal view (Scale: {target_scale}, Pitch: {target_pitch})...")
    
    import time
    import threading
    
    def hold_until_condition(condition_func, key, max_duration=5.0):
        """Hold a key until condition is met or max duration reached"""
        start_time = time.time()
        check_interval = 0.05  # Check every 50ms
        
        # Press the key down
        ipc_send({"cmd": "keyPress", "key": key})
        
        try:
            # Monitor the condition while key is held down
            while time.time() - start_time < max_duration:
                if condition_func():
                    return True
                time.sleep(check_interval)
        finally:
            # Always release the key when done
            ipc_send({"cmd": "keyRelease", "key": key})
        
        return False
    
    def scroll_until_condition(condition_func, max_duration=5.0):
        """Scroll until condition is met or max duration reached"""
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while time.time() - start_time < max_duration:
            if condition_func():
                return True
            
            # Send scroll command
            ipc_send({"cmd": "scroll", "amount": -1})
            time.sleep(check_interval)
        
        return False
    
    # Get initial camera state
    camera_stats = get_camera_stats(payload)
    if not camera_stats or not camera_stats.get("ok"):
        print("[CAMERA] Could not get initial camera stats")
        return False
    
    current_scale = camera_stats.get("scale", 512)
    current_pitch = camera_stats.get("pitch", 0)
    
    print(f"[CAMERA] Starting from Scale: {current_scale}, Pitch: {current_pitch}")
    
    # First, handle zoom (scale) - scroll out until we're close enough
    if current_scale > target_scale + 30:
        print(f"[CAMERA] Zooming out from {current_scale} to ~{target_scale}")
        
        def scale_good():
            stats = get_camera_stats(payload)
            if not stats or not stats.get("ok"):
                return False
            return abs(stats.get("scale", 512) - target_scale) <= 30
        
        if not scroll_until_condition(scale_good, max_duration=8.0):
            print("[CAMERA] Zoom adjustment timed out")
            return False
    
    # Then, handle pitch - hold UP until we're close enough
    if current_pitch < target_pitch - 30:
        print(f"[CAMERA] Pitching up from {current_pitch} to ~{target_pitch}")
        
        def pitch_good():
            stats = get_camera_stats(payload)
            if not stats or not stats.get("ok"):
                return False
            return abs(stats.get("pitch", 0) - target_pitch) <= 30
        
        if not hold_until_condition(pitch_good, "UP", max_duration=8.0):
            print("[CAMERA] Pitch adjustment timed out")
            return False
    
    # Final check
    final_stats = get_camera_stats(payload)
    if final_stats and final_stats.get("ok"):
        final_scale = final_stats.get("scale", 512)
        final_pitch = final_stats.get("pitch", 0)
        print(f"[CAMERA] Camera setup complete! Scale: {final_scale}, Pitch: {final_pitch}")
        return True
    
    print("[CAMERA] Camera setup failed - could not get final stats")
    return False