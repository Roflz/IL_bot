# camera_integration.py
import logging
import time
import random
import math
import threading
import queue

from helpers.runtime_utils import ipc, dispatch
from helpers.utils import sleep_exponential, exponential_number, normal_number

# Timing instrumentation
_TIMING_ENABLED = True
_TIMING_FILE = "camera_integration.timing.jsonl"

# Camera data recording for calibration
_CAMERA_RECORDING_ENABLED = False
_CAMERA_RECORDING_FILE = "camera_calibration_data.jsonl"

# Camera States
CAMERA_STATE_IDLE_ACTIVITY = "IDLE_ACTIVITY"
CAMERA_STATE_OBJECT_INTERACTION = "OBJECT_INTERACTION"
CAMERA_STATE_LONG_TRAVEL = "LONG_TRAVEL"
CAMERA_STATE_AREA_ACTIVITY = "AREA_ACTIVITY"
CAMERA_STATE_PHASE_TRANSITION = "PHASE_TRANSITION"

# Global camera state (can be set by plans)
_current_camera_state = None
_current_camera_state_config = None
_current_area_center = None  # Area center for AREA_ACTIVITY state
_current_interaction_object = None  # Current object being interacted with {"x": int, "y": int, "plane": int}
_last_idle_camera_check_ts = 0  # Timestamp of last idle camera check
_last_idle_camera_position = None  # Last calculated optimal camera position
_last_idle_character_position = None  # Last character position when camera was adjusted
_last_idle_object_position = None  # Last object position when camera was adjusted

# Thread-based camera movement system
_camera_movement_queue = queue.Queue()  # Thread-safe queue for camera movements
_camera_thread = None  # Background thread for camera timing
_camera_thread_lock = threading.Lock()  # Lock for thread management
_camera_thread_running = False  # Flag to control thread lifecycle

# Camera movement constants
MIN_DISTANCE_TO_CONSIDER = 3  # tiles - objects closer than this are ignored

# Camera movement calibration constants (from calibration_results_20251227_161207.txt)
# YAW: 0.5768 units/ms overall average = 1.73 ms/yaw unit
YAW_MS_PER_UNIT = 1.73  # milliseconds per yaw unit
YAW_UNITS_PER_MS = 0.575  # yaw units per millisecond (for durations >= 200ms)

# PITCH: 0.2736 units/ms overall average = 3.65 ms/pitch unit
PITCH_MS_PER_UNIT = 3.65  # milliseconds per pitch unit
PITCH_UNITS_PER_MS = 0.275  # pitch units per millisecond (for durations >= 200ms)

# ZOOM: Variable based on current zoom level - use lookup table
# Zoom lookup table: (min_zoom, max_zoom) -> (units_per_scroll_in, units_per_scroll_out)
ZOOM_LOOKUP_TABLE = [
    # (min_zoom, max_zoom, units_per_scroll_in, units_per_scroll_out)
    (551, 1000, 50, 45),      # Low zoom: ~40-60 IN, ~43-46 OUT
    (1000, 2000, 95, 75),     # Mid-low zoom: ~55-131 IN, ~43-107 OUT
    (2000, 3000, 130, 105),   # Mid zoom: ~130-192 IN, ~99-153 OUT
    (3000, 4000, 195, 155),   # Mid-high zoom: ~190-268 IN, ~149-198 OUT
    (4000, 4409, 250, 200),   # High zoom: ~265-268 IN, ~198-302 OUT
]


def get_zoom_units_per_scroll(current_zoom, scroll_direction):
    """
    Get zoom units per scroll based on current zoom level.
    
    Args:
        current_zoom: Current zoom level (551-4409)
        scroll_direction: 1 for IN (zoom in), -1 for OUT (zoom out)
    
    Returns:
        float: Units per scroll for the given zoom level and direction
    """
    # Clamp zoom to valid range
    current_zoom = max(551, min(4409, current_zoom))
    
    # Find matching range in lookup table
    for min_zoom, max_zoom, units_in, units_out in ZOOM_LOOKUP_TABLE:
        if min_zoom <= current_zoom < max_zoom:
            return units_in if scroll_direction > 0 else units_out
    
    # Fallback: use last range (high zoom)
    return ZOOM_LOOKUP_TABLE[-1][2] if scroll_direction > 0 else ZOOM_LOOKUP_TABLE[-1][3]


def calculate_zoom_scroll_count(zoom_diff, current_zoom):
    """
    Calculate number of scrolls needed to achieve zoom_diff change.
    
    Args:
        zoom_diff: Desired zoom change (positive = zoom in, negative = zoom out)
        current_zoom: Current zoom level
    
    Returns:
        int: Number of scrolls needed (1-3 max)
    """
    if zoom_diff == 0:
        return 0
    
    scroll_direction = 1 if zoom_diff > 0 else -1
    units_per_scroll = get_zoom_units_per_scroll(current_zoom, scroll_direction)
    
    scroll_count = int(abs(zoom_diff) / units_per_scroll)
    return min(max(1, scroll_count), 3)  # Clamp between 1 and 3


# Test mode for camera movements
_camera_test_mode_enabled = False  # Set to True to enable test mode
_camera_test_trigger = False  # Set to True to trigger a test movement

# Camera configuration (configurable for testing)
CAMERA_CONFIG = {
    # Sweet spot position (2/3 up screen)
    "sweet_spot_y_ratio": 0.67,
    
    # Probability heat map parameters
    "heat_map_std_dev_ratio": 0.33,  # std_dev = screen_width * this ratio
    
    # Mode multipliers
    "intentional_mode_multiplier": (1.5, 2.0),  # Range for probability multiplier
    "idle_mode_multiplier": (0.3, 0.5),  # Range for probability multiplier
    
    # Distance thresholds for mode detection
    "intentional_distance_threshold": 3,  # tiles - closer than this = intentional
    "idle_distance_threshold": 10,  # tiles - farther than this = idle
    
    # Pitch ranges (configurable for testing)
    "pitch_close_min": 300,  # Close to target (< 5 tiles)
    "pitch_close_max": 500,
    "pitch_far_min": 100,  # Far from target (> 20 tiles)
    "pitch_far_max": 300,
    
    # Zoom ranges (configurable for testing)
    "zoom_close_min": 400,  # Close to target
    "zoom_close_max": 500,
    "zoom_far_min": 500,  # Far from target
    "zoom_far_max": 600,
    
    # Distance thresholds for pitch/zoom
    "close_distance_threshold": 5,  # tiles
    "far_distance_threshold": 20,  # tiles
    
    # Camera adjustment duration ranges (ms)
    "intentional_duration_min": 150,
    "intentional_duration_max": 400,
    "idle_duration_min": 50,
    "idle_duration_max": 200,
    
    # Movement following probability
    "movement_follow_probability": (0.10, 0.20),  # Range for following character direction
}

# Default camera state configurations
DEFAULT_CAMERA_STATE_CONFIGS = {
    CAMERA_STATE_IDLE_ACTIVITY: {
        "zoom_preference": "area_wide",  # "area_wide" | "close_up" | "medium" | "auto"
        "yaw_behavior": "follow_character",  # "follow_character" | "point_to_area_center" | "follow_path" | "point_to_object"
        "pitch_behavior": "auto",  # "overhead" | "angled" | "auto"
        "movement_frequency": "idle",  # "idle" | "moderate" | "active"
        "target_preference": "character",  # "character" | "area_center" | "object" | "path_ahead"
        "idle_probability": 0.7,  # 0.0-1.0 - probability of skipping camera adjustments
        "zoom_range": (500, 600),  # (min, max) zoom for area_wide
        "pitch_range": (200, 400),  # (min, max) pitch for idle activities
    },
    CAMERA_STATE_OBJECT_INTERACTION: {
        "zoom_preference": "medium",
        "yaw_behavior": "point_to_object",
        "pitch_behavior": "auto",
        "movement_frequency": "active",
        "target_preference": "object",
        "idle_probability": 0.0,  # Always adjust for object interactions
        "zoom_range": (450, 550),
        "pitch_range": (300, 500),
    },
    CAMERA_STATE_LONG_TRAVEL: {
        "zoom_preference": "area_wide",
        "yaw_behavior": "follow_path",
        "pitch_behavior": "auto",
        "movement_frequency": "moderate",
        "target_preference": "path_ahead",
        "idle_probability": 0.3,
        "zoom_range": (500, 600),
        "pitch_range": (150, 350),
    },
    CAMERA_STATE_AREA_ACTIVITY: {
        "zoom_preference": "area_wide",
        "yaw_behavior": "point_to_area_center",
        "pitch_behavior": "auto",
        "movement_frequency": "idle",
        "target_preference": "area_center",
        "idle_probability": 0.6,
        "zoom_range": (500, 600),
        "pitch_range": (200, 400),
    },
    CAMERA_STATE_PHASE_TRANSITION: {
        "zoom_preference": "auto",
        "yaw_behavior": "follow_character",
        "pitch_behavior": "auto",
        "movement_frequency": "active",
        "target_preference": "character",
        "idle_probability": 0.0,  # Always adjust during transitions
        "zoom_range": (450, 600),
        "pitch_range": (200, 500),
    },
}


def set_camera_state(state: str, config: dict = None, area_center: dict = None, interaction_object: dict = None):
    """
    Set the current camera state.
    
    Args:
        state: One of CAMERA_STATE_* constants
        config: Optional state-specific configuration dict (overrides defaults)
        area_center: Optional area center coordinates {"x": int, "y": int} for AREA_ACTIVITY state
        interaction_object: Optional object being interacted with {"x": int, "y": int, "plane": int}
    """
    global _current_camera_state, _current_camera_state_config, _current_area_center, _current_interaction_object
    
    _current_camera_state = state
    if config:
        # Merge with defaults
        default_config = DEFAULT_CAMERA_STATE_CONFIGS.get(state, {})
        _current_camera_state_config = {**default_config, **config}
    else:
        _current_camera_state_config = DEFAULT_CAMERA_STATE_CONFIGS.get(state, {}).copy()
    
    _current_area_center = area_center
    _current_interaction_object = interaction_object
    
    logging.debug(f"[CAMERA_STATE] Set to {state} with config: {_current_camera_state_config}, area_center: {area_center}, interaction_object: {interaction_object}")


def get_camera_state() -> tuple:
    """
    Get the current camera state and config.
    
    Returns:
        (state: str, config: dict, area_center: dict, interaction_object: dict) or (None, None, None, None) if not set
    """
    global _current_camera_state, _current_camera_state_config, _current_area_center, _current_interaction_object
    return _current_camera_state, _current_camera_state_config, _current_area_center, _current_interaction_object


def clear_camera_state():
    """Clear the current camera state (revert to default behavior)."""
    global _current_camera_state, _current_camera_state_config, _current_area_center, _current_interaction_object
    _current_camera_state = None
    _current_camera_state_config = None
    _current_area_center = None
    _current_interaction_object = None


def set_interaction_object(object_coords: dict):
    """
    Set the current object being interacted with.
    
    Args:
        object_coords: {"x": int, "y": int, "plane": int} or None to clear
    """
    global _current_interaction_object
    _current_interaction_object = object_coords


def set_camera_recording(enabled: bool):
    """
    Enable/disable camera data recording for calibration.
    
    Args:
        enabled: True to start recording, False to stop
    """
    global _CAMERA_RECORDING_ENABLED
    _CAMERA_RECORDING_ENABLED = enabled
    logging.info(f"[CAMERA_RECORDING] {'Enabled' if enabled else 'Disabled'}")


def _record_camera_data(camera_state: dict, objects_data: list):
    """
    Record camera state and object screen positions for calibration.
    
    Args:
        camera_state: {"yaw": int, "pitch": int, "zoom": int}
        objects_data: List of {"world": {"x": int, "y": int}, "screen": {"x": int, "y": int} or None, "type": str}
    """
    if not _CAMERA_RECORDING_ENABLED:
        return
    
    try:
        import json
        from actions import player
        
        player_x = player.get_x()
        player_y = player.get_y()
        
        record = {
            "timestamp": time.time(),
            "player": {"x": player_x, "y": player_y},
            "camera": camera_state,
            "objects": objects_data
        }
        
        with open(_CAMERA_RECORDING_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logging.warning(f"[CAMERA_RECORDING] Failed to record data: {e}")


def _camera_thread_worker():
    """
    Background thread that handles precise camera key timing.
    Processes movements from the queue and executes them with exact timing.
    """
    global _camera_thread_running
    
    _camera_thread_running = True
    active_keys = {}  # Track active key holds: {key: release_time}
    
    while _camera_thread_running:
        try:
            # Check for new movements (non-blocking with timeout)
            try:
                movement = _camera_movement_queue.get(timeout=0.1)
            except queue.Empty:
                movement = None
            
            # Process movement if available
            if movement:
                if movement.get("type") == "key_hold":
                    key = movement.get("key")
                    duration_ms = movement.get("duration_ms", 0)
                    cancel_opposite = movement.get("cancel_opposite")
                    
                    # Cancel opposite key if it's active
                    if cancel_opposite and cancel_opposite in active_keys:
                        ipc.key_release(cancel_opposite)
                        active_keys.pop(cancel_opposite, None)
                    
                    # If key is already active, cancel it first
                    if key in active_keys:
                        ipc.key_release(key)
                        active_keys.pop(key, None)
                    
                    # Press key and schedule release
                    ipc.key_press(key)
                    release_time = time.time() + (duration_ms / 1000.0)
                    active_keys[key] = release_time
                elif movement.get("type") == "scroll":
                    scroll_amount = movement.get("amount", 0)
                    scroll_count = movement.get("count", 1)
                    for _ in range(scroll_count):
                        ipc.scroll(scroll_amount)
                        time.sleep(0.05)  # Small delay between scrolls
            
            # Check for keys that need to be released
            current_time = time.time()
            keys_to_release = []
            for key, release_time in active_keys.items():
                if current_time >= release_time:
                    keys_to_release.append(key)
            
            # Release keys that have reached their duration
            for key in keys_to_release:
                ipc.key_release(key)
                active_keys.pop(key, None)
            
            # Small sleep to prevent busy-waiting
            time.sleep(0.01)  # 10ms check interval
            
        except Exception as e:
            logging.warning(f"[CAMERA_THREAD] Error in camera thread: {e}")
            time.sleep(0.1)
    
    # Cleanup: release any remaining keys
    for key in list(active_keys.keys()):
        ipc.key_release(key)
    
    logging.info("[CAMERA_THREAD] Camera thread stopped")


def _ensure_camera_thread():
    """
    Ensures the camera thread is running.
    Starts it if it's not already running.
    """
    global _camera_thread, _camera_thread_running
    
    with _camera_thread_lock:
        if _camera_thread is None or not _camera_thread.is_alive():
            _camera_thread_running = False  # Reset flag
            
            # Wait a moment for old thread to finish
            if _camera_thread is not None:
                _camera_thread.join(timeout=0.5)
            
            # Start new thread
            _camera_thread = threading.Thread(
                target=_camera_thread_worker,
                name="CameraMovementThread",
                daemon=True  # Dies when main thread dies
            )
            _camera_thread.start()
            logging.info("[CAMERA_THREAD] Camera thread started")


def stop_camera_thread():
    """
    Stops the camera thread gracefully.
    Should be called on script shutdown.
    """
    global _camera_thread_running, _camera_thread
    
    with _camera_thread_lock:
        _camera_thread_running = False
        if _camera_thread is not None:
            _camera_thread.join(timeout=1.0)
            _camera_thread = None
        logging.info("[CAMERA_THREAD] Camera thread stopped")


def adjust_camera_continuous() -> bool:
    """
    Continuous camera adjustment that runs every loop.
    Uses proper calculations based on camera ranges and key hold durations.
    
    Camera ranges:
    - Yaw: 0-2048 (circular, wraps around)
    - Pitch: 128 (lowest/looking down) to 383 (highest/looking up)
    - Zoom: 4409 (zoomed in) to 551 (zoomed out)
    
    Returns:
        True if camera was adjusted, False otherwise
    """
    import math
    
    camera_state, state_config, area_center, interaction_object = get_camera_state()
    
    if not camera_state or not state_config:
        return False
    
    from actions import player
    
    player_x = player.get_x()
    player_y = player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Get current camera state
    camera = ipc.get_camera() or {}
    current_yaw = camera.get("yaw", 0)
    current_pitch = camera.get("pitch", 0)
    current_zoom = camera.get("scale", 551)
    
    # Get screen dimensions
    where = ipc.where() or {}
    screen_width = int(where.get("w", 0))
    screen_height = int(where.get("h", 0))
    if screen_width == 0 or screen_height == 0:
        return False
    
    # Screen center (where character is)
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    
    # Get objects to frame
    # Filter out objects very close to player (they're already in view)
    objects_to_frame = []
    
    # 1. Current interaction object (highest priority)
    if interaction_object:
        obj_x = interaction_object.get("x")
        obj_y = interaction_object.get("y")
        if isinstance(obj_x, int) and isinstance(obj_y, int):
            proj = ipc.project_world_tile(obj_x, obj_y) or {}
            if proj.get("ok"):
                screen_x = int(proj.get("canvas", {}).get("x", 0))
                screen_y = int(proj.get("canvas", {}).get("y", 0))
                objects_to_frame.append({
                    "type": "current_interaction",
                    "world": {"x": obj_x, "y": obj_y},
                    "screen": {"x": screen_x, "y": screen_y},
                    "desired": {
                        "x": screen_center_x,
                        "y": int(screen_center_y * 0.7)  # Slightly above center
                    },
                    "priority": 1.0
                })
            else:
                # Off-screen - needs to be brought into view
                objects_to_frame.append({
                    "type": "current_interaction",
                    "world": {"x": obj_x, "y": obj_y},
                    "screen": None,  # Off-screen
                    "desired": {
                        "x": screen_center_x,
                        "y": int(screen_center_y * 0.7)
                    },
                    "priority": 1.0
                })
    
    # 2. Next-phase objects (predict what we'll interact with next)
    next_phase_objects = _get_next_phase_objects(camera_state, player_x, player_y)
    for obj in next_phase_objects:
        obj_x = obj.get("x")
        obj_y = obj.get("y")
        if isinstance(obj_x, int) and isinstance(obj_y, int):
            # Calculate distance to player
            distance = math.sqrt((obj_x - player_x)**2 + (obj_y - player_y)**2)
            
            # Only consider if not too close
            if distance >= MIN_DISTANCE_TO_CONSIDER:
                proj = ipc.project_world_tile(obj_x, obj_y) or {}
                if proj.get("ok"):
                    screen_x = int(proj.get("canvas", {}).get("x", 0))
                    screen_y = int(proj.get("canvas", {}).get("y", 0))
                    objects_to_frame.append({
                        "type": "next_phase",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": {"x": screen_x, "y": screen_y},
                        "desired": {
                            "x": screen_center_x,  # Fixed position, not random
                            "y": int(screen_center_y * 0.5)  # Fixed position, not random
                        },
                        "priority": 0.6
                    })
                else:
                    # Off-screen but should be visible
                    objects_to_frame.append({
                        "type": "next_phase",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": None,
                        "desired": {
                            "x": screen_center_x,
                            "y": int(screen_center_y * 0.6)
                        },
                        "priority": 0.6
                    })
    
    # 3. Area center (for context, lower priority)
    if area_center:
        area_x = area_center.get("x")
        area_y = area_center.get("y")
        if isinstance(area_x, int) and isinstance(area_y, int):
            # Calculate distance to player
            distance = math.sqrt((area_x - player_x)**2 + (area_y - player_y)**2)
            
            # Only consider if not too close
            if distance >= MIN_DISTANCE_TO_CONSIDER:
                proj = ipc.project_world_tile(area_x, area_y) or {}
                if proj.get("ok"):
                    screen_x = int(proj.get("canvas", {}).get("x", 0))
                    screen_y = int(proj.get("canvas", {}).get("y", 0))
                    objects_to_frame.append({
                        "type": "area_center",
                        "world": {"x": area_x, "y": area_y},
                        "screen": {"x": screen_x, "y": screen_y},
                        "desired": {
                            "x": int(screen_width * 0.75),  # Top-right area
                            "y": int(screen_height * 0.25)
                        },
                        "priority": 0.3
                    })
    
    if not objects_to_frame:
        return False
    
    # Record camera data for calibration if enabled
    if _CAMERA_RECORDING_ENABLED:
        objects_data = []
        for obj in objects_to_frame:
            obj_data = {
                "type": obj.get("type"),
                "world": obj.get("world"),
                "screen": obj.get("screen"),
                "desired": obj.get("desired"),
                "priority": obj.get("priority")
            }
            objects_data.append(obj_data)
        
        _record_camera_data(
            camera_state={"yaw": current_yaw, "pitch": current_pitch, "zoom": current_zoom},
            objects_data=objects_data
        )
    
    # Calculate desired target positions directly from object positions
    desired_target_yaw = current_yaw
    desired_target_pitch = current_pitch
    desired_target_zoom = current_zoom
    
    # For on-screen objects: calculate yaw/pitch changes directly from pixel offsets
    # For off-screen objects: calculate yaw to point at object
    
    total_yaw_change = 0.0
    total_pitch_change = 0.0
    total_zoom_change = 0
    
    total_priority = sum(obj.get("priority", 1.0) for obj in objects_to_frame)
    
    for obj in objects_to_frame:
        priority = obj.get("priority", 1.0) / total_priority if total_priority > 0 else 1.0
        
        if obj["screen"] is None:
            # Off-screen: calculate yaw to point at object
            world = obj["world"]
            angle_rad = math.atan2(world["y"] - player_y, world["x"] - player_x)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
            yaw_diff = desired_yaw_rl - current_yaw
            while yaw_diff > 1024:
                yaw_diff -= 2048
            while yaw_diff < -1024:
                yaw_diff += 2048
            total_yaw_change += yaw_diff * priority
            
            # Pitch toward medium
            pitch_diff = (255 - current_pitch) * priority
            total_pitch_change += pitch_diff
            
            # Zoom toward zoomed out
            zoom_diff = (551 - current_zoom) * priority
            total_zoom_change += int(zoom_diff / 150)  # Convert to scrolls
        else:
            # On-screen: calculate changes directly from pixel offsets
            screen = obj["screen"]
            desired = obj["desired"]
            dx = desired["x"] - screen["x"]
            dy = desired["y"] - screen["y"]
            
            # Direct conversion: 2 pixels = 1 yaw unit, 4 pixels = 1 pitch unit
            if abs(dx) > 10:  # Dead zone
                yaw_change = (dx / 2.0) * priority
                total_yaw_change += yaw_change
            
            if abs(dy) > 10:  # Dead zone
                pitch_change = (dy / 4.0) * priority
                total_pitch_change += pitch_change
    
    # Calculate final targets
    if abs(total_yaw_change) > 1:
        desired_target_yaw = int((current_yaw + total_yaw_change) % 2048)
    
    if abs(total_pitch_change) > 1:
        desired_target_pitch = int(max(128, min(383, current_pitch + total_pitch_change)))
    
    if abs(total_zoom_change) > 0:
        zoom_change = total_zoom_change * 150
        desired_target_zoom = int(max(551, min(4409, current_zoom + zoom_change)))
    
    # Dead zones (minimum movement to trigger)
    yaw_dead_zone = 20
    pitch_dead_zone = 10
    zoom_dead_zone = 30
    
    # Calculate differences to desired target
    yaw_diff = desired_target_yaw - current_yaw
    while yaw_diff > 1024:
        yaw_diff -= 2048
    while yaw_diff < -1024:
        yaw_diff += 2048
    
    pitch_diff = desired_target_pitch - current_pitch
    zoom_diff = desired_target_zoom - current_zoom
    
    # Ensure camera thread is running
    _ensure_camera_thread()
    
    # Queue movements for the thread to execute
    movements_queued = False
    
    if abs(yaw_diff) > yaw_dead_zone:
        # Calculate hold duration using calibrated conversion factor
        yaw_hold_ms = abs(yaw_diff) * YAW_MS_PER_UNIT
        yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
        
        # Queue movement for thread
        _camera_movement_queue.put({
            "type": "key_hold",
            "key": yaw_key,
            "duration_ms": yaw_hold_ms,
            "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
        })
        movements_queued = True
    
    if abs(pitch_diff) > pitch_dead_zone:
        # Calculate hold duration using calibrated conversion factor
        pitch_hold_ms = abs(pitch_diff) * PITCH_MS_PER_UNIT
        pitch_key = "UP" if pitch_diff > 0 else "DOWN"
        
        # Queue movement for thread
        _camera_movement_queue.put({
            "type": "key_hold",
            "key": pitch_key,
            "duration_ms": pitch_hold_ms,
            "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
        })
        movements_queued = True
    
    if abs(zoom_diff) > zoom_dead_zone:
        # Zoom uses scroll - calculate using zoom lookup table
        scroll_amount = 1 if zoom_diff > 0 else -1
        scroll_count = calculate_zoom_scroll_count(zoom_diff, current_zoom)
        if scroll_count > 0:
            _camera_movement_queue.put({
                "type": "scroll",
                "amount": scroll_amount,
                "count": scroll_count
            })
            movements_queued = True
    
    return movements_queued


def test_camera_movement(
    camera_state: str = None,
    state_config: dict = None,
    area_center: dict = None,
    interaction_object: dict = None
) -> dict:
    """
    Test function for camera movements.
    Calculates camera movements, outputs before/after coordinates of target objects,
    and executes the movement.
    
    Args:
        camera_state: Optional camera state (if None, uses get_camera_state())
        state_config: Optional state config (if None, uses get_camera_state())
        area_center: Optional area center (if None, uses get_camera_state())
        interaction_object: Optional interaction object (if None, uses get_camera_state())
    
    Returns:
        Dictionary with:
        - "before": List of object screen positions before movement
        - "after": List of object screen positions after movement
        - "movements": List of movements that were executed
        - "success": Whether the test completed successfully
    """
    import math
    
    # Use provided parameters or get from global state
    if camera_state is None:
        camera_state, state_config, area_center, interaction_object = get_camera_state()
    elif state_config is None:
        # If camera_state provided but not config, get defaults
        state_config = DEFAULT_CAMERA_STATE_CONFIGS.get(camera_state, {}).copy()
    
    if not camera_state or not state_config:
        logging.warning("[CAMERA_TEST] No camera state set")
        return {"success": False, "error": "No camera state set"}
    
    from actions import player
    
    player_x = player.get_x()
    player_y = player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return {"success": False, "error": "Invalid player position"}
    
    # Get current camera state
    camera = ipc.get_camera() or {}
    current_yaw = camera.get("yaw", 0)
    current_pitch = camera.get("pitch", 0)
    current_zoom = camera.get("scale", 551)
    
    # Get screen dimensions
    where = ipc.where() or {}
    screen_width = int(where.get("w", 0))
    screen_height = int(where.get("h", 0))
    if screen_width == 0 or screen_height == 0:
        return {"success": False, "error": "Invalid screen dimensions"}
    
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    
    # Get objects to frame (same logic as adjust_camera_continuous)
    # Filter out objects very close to player (they're already in view)
    objects_to_frame = []
    
    # 1. Current interaction object
    if interaction_object:
        obj_x = interaction_object.get("x")
        obj_y = interaction_object.get("y")
        if isinstance(obj_x, int) and isinstance(obj_y, int):
            # Calculate distance to player
            distance = math.sqrt((obj_x - player_x)**2 + (obj_y - player_y)**2)
            
            # Only consider if not too close
            if distance >= MIN_DISTANCE_TO_CONSIDER:
                proj = ipc.project_world_tile(obj_x, obj_y) or {}
                if proj.get("ok"):
                    screen_x = int(proj.get("canvas", {}).get("x", 0))
                    screen_y = int(proj.get("canvas", {}).get("y", 0))
                    objects_to_frame.append({
                        "type": "current_interaction",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": {"x": screen_x, "y": screen_y},
                        "desired": {
                            "x": screen_center_x,
                            "y": int(screen_center_y * 0.7)
                        },
                        "priority": 1.0
                    })
                else:
                    objects_to_frame.append({
                        "type": "current_interaction",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": None,
                        "desired": {
                            "x": screen_center_x,
                            "y": int(screen_center_y * 0.7)
                        },
                        "priority": 1.0
                    })
    
    # 2. Next-phase objects
    next_phase_objects = _get_next_phase_objects(camera_state, player_x, player_y)
    for obj in next_phase_objects:
        obj_x = obj.get("x")
        obj_y = obj.get("y")
        if isinstance(obj_x, int) and isinstance(obj_y, int):
            # Calculate distance to player
            distance = math.sqrt((obj_x - player_x)**2 + (obj_y - player_y)**2)
            
            # Only consider if not too close
            if distance >= MIN_DISTANCE_TO_CONSIDER:
                proj = ipc.project_world_tile(obj_x, obj_y) or {}
                if proj.get("ok"):
                    screen_x = int(proj.get("canvas", {}).get("x", 0))
                    screen_y = int(proj.get("canvas", {}).get("y", 0))
                    objects_to_frame.append({
                        "type": "next_phase",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": {"x": screen_x, "y": screen_y},
                        "desired": {
                            "x": screen_center_x,
                            "y": int(screen_center_y * 0.5)
                        },
                        "priority": 0.6
                    })
                else:
                    objects_to_frame.append({
                        "type": "next_phase",
                        "world": {"x": obj_x, "y": obj_y},
                        "screen": None,
                        "desired": {
                            "x": screen_center_x,
                            "y": int(screen_center_y * 0.6)
                        },
                        "priority": 0.6
                    })
    
    # 3. Area center (always include, but check distance)
    if area_center:
        area_x = area_center.get("x")
        area_y = area_center.get("y")
        if isinstance(area_x, int) and isinstance(area_y, int):
            # Calculate distance to player
            distance = math.sqrt((area_x - player_x)**2 + (area_y - player_y)**2)
            
            # Only consider if not too close
            if distance >= MIN_DISTANCE_TO_CONSIDER:
                proj = ipc.project_world_tile(area_x, area_y) or {}
                if proj.get("ok"):
                    screen_x = int(proj.get("canvas", {}).get("x", 0))
                    screen_y = int(proj.get("canvas", {}).get("y", 0))
                    objects_to_frame.append({
                        "type": "area_center",
                        "world": {"x": area_x, "y": area_y},
                        "screen": {"x": screen_x, "y": screen_y},
                        "desired": {
                            "x": int(screen_width * 0.75),
                            "y": int(screen_height * 0.25)
                        },
                        "priority": 0.3
                    })
    
    if not objects_to_frame:
        return {"success": False, "error": "No objects to frame"}
    
    # Record BEFORE positions
    before_positions = []
    for obj in objects_to_frame:
        if obj["screen"] is not None:
            before_positions.append({
                "type": obj["type"],
                "world": obj["world"],
                "screen": obj["screen"].copy(),
                "desired": obj["desired"].copy()
            })
        else:
            before_positions.append({
                "type": obj["type"],
                "world": obj["world"],
                "screen": None,
                "desired": obj["desired"].copy()
            })
    
    # Calculate desired camera movements (same logic as adjust_camera_continuous)
    total_yaw_change = 0.0
    total_pitch_change = 0.0
    total_zoom_change = 0
    
    total_priority = sum(obj.get("priority", 1.0) for obj in objects_to_frame)
    
    for obj in objects_to_frame:
        priority = obj.get("priority", 1.0) / total_priority if total_priority > 0 else 1.0
        
        if obj["screen"] is None:
            # Off-screen: calculate yaw to point at object
            world = obj["world"]
            angle_rad = math.atan2(world["y"] - player_y, world["x"] - player_x)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
            yaw_diff = desired_yaw_rl - current_yaw
            while yaw_diff > 1024:
                yaw_diff -= 2048
            while yaw_diff < -1024:
                yaw_diff += 2048
            total_yaw_change += yaw_diff * priority
            
            pitch_diff = (255 - current_pitch) * priority
            total_pitch_change += pitch_diff
            
            zoom_diff = (551 - current_zoom) * priority
            total_zoom_change += int(zoom_diff / 150)
        else:
            # On-screen: calculate changes from pixel offsets
            screen = obj["screen"]
            desired = obj["desired"]
            dx = desired["x"] - screen["x"]
            dy = desired["y"] - screen["y"]
            
            if abs(dx) > 10:
                yaw_change = (dx / 2.0) * priority
                total_yaw_change += yaw_change
            
            if abs(dy) > 10:
                pitch_change = (dy / 4.0) * priority
                total_pitch_change += pitch_change
    
    # Calculate final targets
    desired_target_yaw = current_yaw
    desired_target_pitch = current_pitch
    desired_target_zoom = current_zoom
    
    if abs(total_yaw_change) > 1:
        desired_target_yaw = int((current_yaw + total_yaw_change) % 2048)
    
    if abs(total_pitch_change) > 1:
        desired_target_pitch = int(max(128, min(383, current_pitch + total_pitch_change)))
    
    if abs(total_zoom_change) > 0:
        zoom_change = total_zoom_change * 150
        desired_target_zoom = int(max(551, min(4409, current_zoom + zoom_change)))
    
    # Calculate differences
    yaw_diff = desired_target_yaw - current_yaw
    while yaw_diff > 1024:
        yaw_diff -= 2048
    while yaw_diff < -1024:
        yaw_diff += 2048
    
    pitch_diff = desired_target_pitch - current_pitch
    zoom_diff = desired_target_zoom - current_zoom
    
    # Dead zones
    yaw_dead_zone = 20
    pitch_dead_zone = 10
    zoom_dead_zone = 30
    
    # Record movements that will be executed
    movements = []
    
    # Ensure camera thread is running
    _ensure_camera_thread()
    
    # Execute movements
    if abs(yaw_diff) > yaw_dead_zone:
        yaw_hold_ms = abs(yaw_diff) * YAW_MS_PER_UNIT
        yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
        movements.append({
            "type": "yaw",
            "key": yaw_key,
            "duration_ms": int(yaw_hold_ms),
            "change": yaw_diff
        })
        _camera_movement_queue.put({
            "type": "key_hold",
            "key": yaw_key,
            "duration_ms": int(yaw_hold_ms),
            "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
        })
    
    if abs(pitch_diff) > pitch_dead_zone:
        pitch_hold_ms = abs(pitch_diff) * PITCH_MS_PER_UNIT
        pitch_key = "UP" if pitch_diff > 0 else "DOWN"
        movements.append({
            "type": "pitch",
            "key": pitch_key,
            "duration_ms": int(pitch_hold_ms),
            "change": pitch_diff
        })
        _camera_movement_queue.put({
            "type": "key_hold",
            "key": pitch_key,
            "duration_ms": int(pitch_hold_ms),
            "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
        })
    
    if abs(zoom_diff) > zoom_dead_zone:
        scroll_amount = 1 if zoom_diff > 0 else -1
        scroll_count = calculate_zoom_scroll_count(zoom_diff, current_zoom)
        if scroll_count > 0:
            movements.append({
                "type": "zoom",
                "amount": scroll_amount,
                "count": scroll_count,
                "change": zoom_diff
            })
            _camera_movement_queue.put({
                "type": "scroll",
                "amount": scroll_amount,
                "count": scroll_count
            })
    
    # Wait for movements to complete (add a small buffer)
    max_wait_time = 0
    for m in movements:
        if m["type"] in ["yaw", "pitch"]:
            max_wait_time = max(max_wait_time, m["duration_ms"] / 1000.0)
        elif m["type"] == "zoom":
            max_wait_time = max(max_wait_time, scroll_count * 0.1)
    
    if max_wait_time > 0:
        time.sleep(max_wait_time + 0.1)  # Small buffer
    
    # Record AFTER positions
    after_positions = []
    for obj in objects_to_frame:
        obj_x = obj["world"]["x"]
        obj_y = obj["world"]["y"]
        proj = ipc.project_world_tile(obj_x, obj_y) or {}
        if proj.get("ok"):
            screen_x = int(proj.get("canvas", {}).get("x", 0))
            screen_y = int(proj.get("canvas", {}).get("y", 0))
            after_positions.append({
                "type": obj["type"],
                "world": obj["world"],
                "screen": {"x": screen_x, "y": screen_y},
                "desired": obj["desired"].copy()
            })
        else:
            after_positions.append({
                "type": obj["type"],
                "world": obj["world"],
                "screen": None,
                "desired": obj["desired"].copy()
            })
    
    # Output results
    logging.info("=" * 60)
    logging.info("[CAMERA_TEST] Camera Movement Test Results")
    logging.info("=" * 60)
    logging.info(f"Camera State: {camera_state}")
    logging.info(f"Movements Executed: {len(movements)}")
    for m in movements:
        logging.info(f"  - {m['type']}: {m.get('key', m.get('amount', 'N/A'))} for {m.get('duration_ms', m.get('count', 'N/A'))}ms (change: {m.get('change', 'N/A')})")
    logging.info("")
    logging.info("Object Positions:")
    for i, (before, after) in enumerate(zip(before_positions, after_positions)):
        logging.info(f"  Object {i+1} ({before['type']}):")
        logging.info(f"    World: ({before['world']['x']}, {before['world']['y']})")
        if before["screen"]:
            logging.info(f"    Before: Screen ({before['screen']['x']}, {before['screen']['y']})")
        else:
            logging.info(f"    Before: OFF-SCREEN")
        if after["screen"]:
            logging.info(f"    After:  Screen ({after['screen']['x']}, {after['screen']['y']})")
        else:
            logging.info(f"    After:  OFF-SCREEN")
        logging.info(f"    Desired: ({before['desired']['x']}, {before['desired']['y']})")
        if before["screen"] and after["screen"]:
            dx = after["screen"]["x"] - before["screen"]["x"]
            dy = after["screen"]["y"] - before["screen"]["y"]
            logging.info(f"    Movement: ({dx:+d}, {dy:+d}) pixels")
        logging.info("")
    logging.info("=" * 60)
    
    return {
        "success": True,
        "before": before_positions,
        "after": after_positions,
        "movements": movements,
        "camera_before": {"yaw": current_yaw, "pitch": current_pitch, "zoom": current_zoom},
        "camera_after": ipc.get_camera() or {}
    }


def _get_next_phase_objects(camera_state: str, player_x: int, player_y: int) -> list:
    """
    Predict what objects we'll interact with in the next phase.
    
    Returns:
        List of {"x": int, "y": int} dicts for next-phase objects
    """
    next_objects = []
    
    # This is MLM-specific for now, but could be generalized
    if camera_state == CAMERA_STATE_IDLE_ACTIVITY:
        # Mining phase - next will be deposit (hopper) or collect (sack)
        # Try to find hopper and sack nearby
        try:
            # Hopper is typically near the mining area
            hopper_resp = ipc.get_objects("Hopper", types=["GAME"], radius=15) or {}
            if hopper_resp.get("ok"):
                hoppers = hopper_resp.get("objects", []) or []
                for h in hoppers[:1]:  # Just get first one
                    w = h.get("world", {}) or {}
                    hx, hy = w.get("x"), w.get("y")
                    if isinstance(hx, int) and isinstance(hy, int):
                        next_objects.append({"x": hx, "y": hy, "name": "Hopper"})
            
            # Sack is also nearby
            sack_resp = ipc.get_objects("Sack", types=["GAME"], radius=15) or {}
            if sack_resp.get("ok"):
                sacks = sack_resp.get("objects", []) or []
                for s in sacks[:1]:
                    w = s.get("world", {}) or {}
                    sx, sy = w.get("x"), w.get("y")
                    if isinstance(sx, int) and isinstance(sy, int):
                        next_objects.append({"x": sx, "y": sy, "name": "Sack"})
        except Exception:
            pass
    
    elif camera_state == CAMERA_STATE_OBJECT_INTERACTION:
        # Could be depositing, collecting, or banking
        # Try to find relevant next objects
        try:
            # If near hopper, next might be sack
            # If near sack, next might be bank
            # If near bank, next might be ore vein
            # This is context-dependent, could be improved
            pass
        except Exception:
            pass
    
    return next_objects


def adjust_camera_for_current_state() -> bool:
    """
    Intelligently adjust camera for idle activities.
    Calculates optimal viewing position that can see character and interacting object,
    and only adjusts when needed (character/object moved significantly, or camera is in bad position).
    
    Returns:
        True if camera was adjusted, False otherwise
    """
    global _last_idle_camera_check_ts, _last_idle_camera_position, _last_idle_character_position, _last_idle_object_position
    
    camera_state, state_config, area_center, interaction_object = get_camera_state()
    
    if not camera_state or not state_config:
        return False
    
    # Only check every 3-8 seconds (less frequent for idle activities)
    current_time = time.time()
    if current_time - _last_idle_camera_check_ts < normal_number(3.0, 8.0, center_bias=0.7):
        return False
    _last_idle_camera_check_ts = current_time
    
    from actions import player
    
    player_x = player.get_x()
    player_y = player.get_y()
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return False
    
    # Get character orientation
    player_data = ipc.get_player() or {}
    character_orientation = player_data.get("orientation")
    
    # Use interaction object if set, otherwise try to find it
    object_coords = None
    object_orientation = None
    
    if interaction_object:
        object_coords = {"x": interaction_object.get("x"), "y": interaction_object.get("y")}
    elif camera_state == CAMERA_STATE_IDLE_ACTIVITY:
        # Try to find nearby objects the player might be interacting with
        # Check if player has an animation (indicates interaction)
        anim = player_data.get("animation")
        if anim and anim != 808:  # Not idle
            # Find object in direction player is facing
            object_coords = _find_object_in_direction(player_x, player_y, character_orientation, radius=2)
    
    # Check if we need to adjust:
    # 1. Character moved significantly (>3 tiles)
    # 2. Object changed
    # 3. Camera is in a bad position (check current camera vs optimal)
    needs_adjustment = False
    
    if _last_idle_character_position:
        dx = abs(player_x - _last_idle_character_position.get("x", 0))
        dy = abs(player_y - _last_idle_character_position.get("y", 0))
        if dx > 3 or dy > 3:
            needs_adjustment = True
    else:
        needs_adjustment = True  # First time
    
    if object_coords and _last_idle_object_position:
        obj_dx = abs(object_coords.get("x", 0) - _last_idle_object_position.get("x", 0))
        obj_dy = abs(object_coords.get("y", 0) - _last_idle_object_position.get("y", 0))
        if obj_dx > 1 or obj_dy > 1:
            needs_adjustment = True
    elif object_coords != _last_idle_object_position:
        needs_adjustment = True
    
    # Check if current camera position is good
    if not needs_adjustment and _last_idle_camera_position:
        camera = ipc.get_camera() or {}
        current_yaw = camera.get("yaw", 0)
        current_pitch = camera.get("pitch", 0)
        current_zoom = camera.get("scale", 551)
        
        optimal_yaw = _last_idle_camera_position.get("yaw")
        optimal_pitch = _last_idle_camera_position.get("pitch")
        optimal_zoom = _last_idle_camera_position.get("zoom")
        
        # Check if camera is close to optimal position
        yaw_diff = abs(current_yaw - optimal_yaw) if optimal_yaw is not None else 0
        while yaw_diff > 1024:
            yaw_diff -= 2048
        while yaw_diff < -1024:
            yaw_diff += 2048
        yaw_diff = abs(yaw_diff)
        
        pitch_diff = abs(current_pitch - optimal_pitch) if optimal_pitch is not None else 0
        zoom_diff = abs(current_zoom - optimal_zoom) if optimal_zoom is not None else 0
        
        # Only adjust if camera is significantly off (more than 200 yaw units, 50 pitch, 30 zoom)
        if yaw_diff < 200 and pitch_diff < 50 and zoom_diff < 30:
            return False  # Camera is in good position, no adjustment needed
    
    if not needs_adjustment:
        return False
    
    # Calculate optimal camera position
    optimal_position = _calculate_optimal_idle_camera_position(
        player_x, player_y, character_orientation,
        object_coords, object_orientation,
        area_center, state_config
    )
    
    if not optimal_position:
        return False
    
    # Store positions
    _last_idle_character_position = {"x": player_x, "y": player_y}
    _last_idle_object_position = object_coords
    _last_idle_camera_position = optimal_position
    
    # Apply small, intentional adjustment
    return _apply_idle_camera_adjustment(optimal_position, state_config)


def _find_object_in_direction(player_x: int, player_y: int, orientation: int, radius: int = 2) -> dict:
    """
    Find object in the direction the player is facing.
    
    Returns:
        {"x": int, "y": int} or None
    """
    try:
        # Calculate direction based on orientation
        # Orientation: 0=South, 512=West, 1024=North, 1536=East
        if orientation < 256 or orientation >= 1792:  # South
            dir_x, dir_y = 0, -1
        elif 256 <= orientation < 768:  # West
            dir_x, dir_y = -1, 0
        elif 768 <= orientation < 1280:  # North
            dir_x, dir_y = 0, 1
        else:  # East
            dir_x, dir_y = 1, 0
        
        # Check tiles in direction
        for dist in range(1, radius + 1):
            check_x = player_x + (dir_x * dist)
            check_y = player_y + (dir_y * dist)
            
            obj_resp = ipc.get_object_at_tile(check_x, check_y)
            if obj_resp and obj_resp.get("ok") and obj_resp.get("object"):
                return {"x": check_x, "y": check_y}
        
        return None
    except Exception:
        return None


def _calculate_optimal_idle_camera_position(
    player_x: int, player_y: int, character_orientation: int,
    object_coords: dict, object_orientation: int,
    area_center: dict, state_config: dict
) -> dict:
    """
    Calculate optimal camera position for idle activities.
    Tries to position camera to see both character and object, with good viewing angle.
    
    Returns:
        {"yaw": int, "pitch": int, "zoom": int} or None
    """
    try:
        # Get current camera
        camera = ipc.get_camera() or {}
        current_yaw = camera.get("yaw", 0)
        current_pitch = camera.get("pitch", 0)
        current_zoom = camera.get("scale", 551)
        
        # Calculate target position (weighted average of character, object, and area center)
        target_x = player_x
        target_y = player_y
        
        if object_coords:
            # Weight: 40% character, 40% object, 20% area center
            target_x = int(player_x * 0.4 + object_coords.get("x", player_x) * 0.4)
            if area_center:
                target_x = int(target_x * 0.8 + area_center.get("x", target_x) * 0.2)
            
            target_y = int(player_y * 0.4 + object_coords.get("y", player_y) * 0.4)
            if area_center:
                target_y = int(target_y * 0.8 + area_center.get("y", target_y) * 0.2)
        elif area_center:
            # Weight: 60% character, 40% area center
            target_x = int(player_x * 0.6 + area_center.get("x", player_x) * 0.4)
            target_y = int(player_y * 0.6 + area_center.get("y", player_y) * 0.4)
        
        # Calculate yaw to target
        import math
        angle_rad = math.atan2(target_y - player_y, target_x - player_x)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
        
        # Adjust yaw slightly based on character orientation (to see character better)
        if character_orientation is not None:
            # Convert character orientation to camera yaw
            char_yaw = int((character_orientation + 1024) % 2048)
            # Blend: 70% target direction, 30% character facing direction
            yaw_diff = char_yaw - desired_yaw_rl
            while yaw_diff > 1024:
                yaw_diff -= 2048
            while yaw_diff < -1024:
                yaw_diff += 2048
            desired_yaw_rl = int(desired_yaw_rl + yaw_diff * 0.3)
            desired_yaw_rl = desired_yaw_rl % 2048
        
        # Calculate optimal pitch (higher pitch = more overhead view)
        # For idle activities, use medium-high pitch to see both character and object
        pitch_range = state_config.get("pitch_range", (200, 400))
        desired_pitch = normal_number(pitch_range[0], pitch_range[1], center_bias=0.6, output_type="int")
        
        # Calculate optimal zoom (zoom out for area activities)
        zoom_range = state_config.get("zoom_range", (500, 600))
        desired_zoom = normal_number(zoom_range[0], zoom_range[1], center_bias=0.7, output_type="int")
        
        return {
            "yaw": desired_yaw_rl,
            "pitch": desired_pitch,
            "zoom": desired_zoom
        }
    except Exception as e:
        logging.warning(f"[CAMERA] Error calculating optimal idle position: {e}")
        return None


def _apply_idle_camera_adjustment(optimal_position: dict, state_config: dict) -> bool:
    """
    Apply small, intentional camera adjustment to reach optimal position.
    Uses smaller movements and only adjusts if significantly off.
    
    Returns:
        True if adjustment was made, False otherwise
    """
    try:
        camera = ipc.get_camera() or {}
        current_yaw = camera.get("yaw", 0)
        current_pitch = camera.get("pitch", 0)
        current_zoom = camera.get("scale", 551)
        
        optimal_yaw = optimal_position.get("yaw")
        optimal_pitch = optimal_position.get("pitch")
        optimal_zoom = optimal_position.get("zoom")
        
        if optimal_yaw is None or optimal_pitch is None or optimal_zoom is None:
            return False
        
        # Calculate differences
        yaw_diff = optimal_yaw - current_yaw
        while yaw_diff > 1024:
            yaw_diff -= 2048
        while yaw_diff < -1024:
            yaw_diff += 2048
        
        pitch_diff = optimal_pitch - current_pitch
        zoom_diff = optimal_zoom - current_zoom
        
        # Only adjust if difference is significant (smaller thresholds for idle)
        yaw_threshold = 150  # Only adjust if more than 150 units off
        pitch_threshold = 40
        zoom_threshold = 30
        
        if abs(yaw_diff) < yaw_threshold and abs(pitch_diff) < pitch_threshold and abs(zoom_diff) < zoom_threshold:
            return False  # Already close enough, no adjustment needed
        
        # Apply small adjustments with shorter duration
        max_adjust_time = 0.3  # 300ms max for idle adjustments
        check_interval = 0.05
        start_time = time.time()
        pressed_keys = set()
        
        try:
            while time.time() - start_time < max_adjust_time:
                camera = ipc.get_camera() or {}
                current_yaw = camera.get("yaw", 0)
                current_pitch = camera.get("pitch", 0)
                current_zoom = camera.get("scale", 551)
                
                # Recalculate differences
                yaw_diff = optimal_yaw - current_yaw
                while yaw_diff > 1024:
                    yaw_diff -= 2048
                while yaw_diff < -1024:
                    yaw_diff += 2048
                
                pitch_diff = optimal_pitch - current_pitch
                zoom_diff = optimal_zoom - current_zoom
                
                # Check if close enough
                if abs(yaw_diff) < 50 and abs(pitch_diff) < 15 and abs(zoom_diff) < 10:
                    break
                
                # Apply yaw adjustment (smaller movements)
                if abs(yaw_diff) > 50:
                    yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
                    if yaw_key not in pressed_keys:
                        other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                        if other_yaw_key in pressed_keys:
                            ipc.key_release(other_yaw_key)
                            pressed_keys.discard(other_yaw_key)
                        ipc.key_press(yaw_key)
                        pressed_keys.add(yaw_key)
                else:
                    for yaw_key in ["LEFT", "RIGHT"]:
                        if yaw_key in pressed_keys:
                            ipc.key_release(yaw_key)
                            pressed_keys.discard(yaw_key)
                
                # Apply pitch adjustment
                if abs(pitch_diff) > 15:
                    pitch_key = "UP" if pitch_diff > 0 else "DOWN"
                    if pitch_key not in pressed_keys:
                        other_pitch_key = "DOWN" if pitch_key == "UP" else "UP"
                        if other_pitch_key in pressed_keys:
                            ipc.key_release(other_pitch_key)
                            pressed_keys.discard(other_pitch_key)
                        ipc.key_press(pitch_key)
                        pressed_keys.add(pitch_key)
                else:
                    for pitch_key in ["UP", "DOWN"]:
                        if pitch_key in pressed_keys:
                            ipc.key_release(pitch_key)
                            pressed_keys.discard(pitch_key)
                
                # Apply zoom adjustment (smaller scrolls)
                if abs(zoom_diff) > 10:
                    scroll_amount = 1 if zoom_diff > 0 else -1
                    scroll_count = min(abs(zoom_diff) // 15, 2)  # Max 2 scrolls at a time
                    for _ in range(scroll_count):
                        ipc.scroll(scroll_amount)
                        sleep_exponential(0.05, 0.1, 1.0)
                
                sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
        
        finally:
            # Release all keys
            for key in pressed_keys.copy():
                ipc.key_release(key)
                pressed_keys.discard(key)
        
        return len(pressed_keys) > 0 or abs(zoom_diff) > 10
        
    except Exception as e:
        logging.warning(f"[CAMERA] Error applying idle adjustment: {e}")
        return False


def _calculate_path_direction(
    path_waypoints: list,
    player_x: int,
    player_y: int,
    num_waypoints: int = 5
) -> dict:
    """
    Calculate overall path direction from waypoints.
    
    Args:
        path_waypoints: List of waypoint dicts from pathfinding
        player_x, player_y: Current player position
        num_waypoints: Number of waypoints to analyze (default: 5)
        
    Returns:
        {"dx": int, "dy": int, "angle": float} - Direction vector and angle in degrees
    """
    if not path_waypoints or len(path_waypoints) == 0:
        return {"dx": 0, "dy": 0, "angle": 0.0}
    
    # Analyze first N waypoints to determine overall direction
    waypoints_to_analyze = min(num_waypoints, len(path_waypoints))
    
    # Calculate average direction from player through waypoints
    total_dx = 0
    total_dy = 0
    
    for i in range(waypoints_to_analyze):
        wp = path_waypoints[i]
        wp_x = wp.get("x") or wp.get("world", {}).get("x")
        wp_y = wp.get("y") or wp.get("world", {}).get("y")
        
        if wp_x is None or wp_y is None:
            continue
        
        # Calculate direction from player to this waypoint
        if i == 0:
            # First waypoint - direction from player
            dx = wp_x - player_x
            dy = wp_y - player_y
        else:
            # Direction from previous waypoint
            prev_wp = path_waypoints[i - 1]
            prev_x = prev_wp.get("x") or prev_wp.get("world", {}).get("x")
            prev_y = prev_wp.get("y") or prev_wp.get("world", {}).get("y")
            if prev_x is None or prev_y is None:
                continue
            dx = wp_x - prev_x
            dy = wp_y - prev_y
        
        total_dx += dx
        total_dy += dy
    
    # Average direction
    if waypoints_to_analyze > 0:
        avg_dx = total_dx / waypoints_to_analyze
        avg_dy = total_dy / waypoints_to_analyze
    else:
        avg_dx = 0
        avg_dy = 0
    
    # Calculate angle in degrees (0-360, where 0 is north, 90 is east)
    import math
    angle_rad = math.atan2(avg_dy, avg_dx)
    angle_deg = math.degrees(angle_rad)
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    
    return {
        "dx": int(round(avg_dx)),
        "dy": int(round(avg_dy)),
        "angle": angle_deg
    }


def _calculate_path_lookahead_point(
    path_waypoints: list,
    player_x: int,
    player_y: int,
    final_dest_x: int,
    final_dest_y: int,
    movement_state: dict = None
) -> dict:
    """
    Calculate optimal camera look-ahead point based on path analysis.
    
    Args:
        path_waypoints: List of waypoint dicts from pathfinding
        player_x, player_y: Current player position
        final_dest_x, final_dest_y: Final destination
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
        
    Returns:
        {"x": int, "y": int} - World coordinates to aim camera at
    """
    if not path_waypoints or len(path_waypoints) == 0:
        # No path - just aim at destination
        return {"x": final_dest_x, "y": final_dest_y}
    
    # Calculate total path distance (Manhattan)
    total_distance = abs(final_dest_x - player_x) + abs(final_dest_y - player_y)
    
    # Determine look-ahead distance based on total distance
    if total_distance > 20:
        # Long distance: look 5-10 tiles ahead (with variation)
        look_ahead_tiles = random.randint(5, 10)
    elif total_distance > 5:
        # Medium distance: look 2-5 tiles ahead (with variation)
        look_ahead_tiles = random.randint(2, 5)
    else:
        # Short distance: look at destination or 1-2 tiles ahead
        look_ahead_tiles = random.randint(0, 2)
    
    # Adjust for movement speed if available
    if movement_state:
        is_running = movement_state.get("is_running", False)
        if is_running:
            # Running: look further ahead
            look_ahead_tiles = int(look_ahead_tiles * 1.3)
    
    # Find waypoint that's approximately look_ahead_tiles along the path
    # We'll use path distance (sum of waypoint distances) rather than straight-line
    accumulated_distance = 0
    target_distance = look_ahead_tiles
    
    for i, wp in enumerate(path_waypoints):
        # Extract waypoint coordinates
        wp_x = wp.get("x") or wp.get("world", {}).get("x")
        wp_y = wp.get("y") or wp.get("world", {}).get("y")
        
        if wp_x is None or wp_y is None:
            continue
        
        # Calculate distance from player to this waypoint
        if i == 0:
            # First waypoint - distance from player
            dist = abs(wp_x - player_x) + abs(wp_y - player_y)
        else:
            # Distance from previous waypoint
            prev_wp = path_waypoints[i - 1]
            prev_x = prev_wp.get("x") or prev_wp.get("world", {}).get("x")
            prev_y = prev_wp.get("y") or prev_wp.get("world", {}).get("y")
            if prev_x is None or prev_y is None:
                continue
            dist = abs(wp_x - prev_x) + abs(wp_y - prev_y)
        
        accumulated_distance += dist
        
        # If we've reached our target distance, use this waypoint
        if accumulated_distance >= target_distance:
            # Apply movement direction compensation if available
            if movement_state and movement_state.get("movement_direction"):
                movement_dir = movement_state.get("movement_direction")
                dx = movement_dir.get("dx", 0)
                dy = movement_dir.get("dy", 0)
                
                # Look slightly ahead in movement direction (0.5-1 tile)
                if dx != 0 or dy != 0:
                    compensation = random.uniform(0.5, 1.0)
                    wp_x = int(wp_x + (dx / max(abs(dx), abs(dy))) * compensation)
                    wp_y = int(wp_y + (dy / max(abs(dx), abs(dy))) * compensation)
            
            return {"x": wp_x, "y": wp_y}
    
    # If we didn't find a waypoint within look-ahead distance, use final destination
    return {"x": final_dest_x, "y": final_dest_y}


# ============================================================================
# NEW CAMERA SYSTEM - Probability-based with context awareness
# ============================================================================

def calculate_camera_movement_probability(
    screen_x: int,
    screen_y: int,
    screen_width: int,
    screen_height: int,
    mode: str = "intentional"
) -> float:
    """
    Calculate probability of moving camera based on distance from sweet spot.
    Uses normal distribution to create a "heat map" effect.
    
    Args:
        screen_x, screen_y: Target's screen coordinates
        screen_width, screen_height: Screen dimensions
        mode: "intentional" or "idle" (affects probability multiplier)
    
    Returns:
        Probability (0.0-1.0) of moving camera
    """
    # Sweet spot: center horizontally, 2/3 up vertically
    center_x = screen_width // 2
    sweet_spot_y = int(screen_height * CAMERA_CONFIG["sweet_spot_y_ratio"])
    
    # Check if off-screen (with margin)
    margin = 50
    if screen_x < -margin or screen_x > screen_width + margin or \
       screen_y < -margin or screen_y > screen_height + margin:
        return 1.0  # Always move if off-screen
    
    # Calculate distance from sweet spot
    dx = screen_x - center_x
    dy = screen_y - sweet_spot_y
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Use normal distribution for probability
    # std_dev = screen_width * ratio (creates heat map effect)
    std_dev = screen_width * CAMERA_CONFIG["heat_map_std_dev_ratio"]
    
    # Normal distribution: probability decreases as distance increases
    # Using complementary CDF approximation
    z_score = distance / std_dev if std_dev > 0 else 0
    # Approximate normal CDF: P(X > z) ≈ 1 - (1 + erf(z/√2)) / 2
    # Simplified: use exponential decay for probability
    base_probability = math.exp(-0.5 * z_score * z_score)
    # Invert so closer = lower probability
    base_probability = 1.0 - base_probability
    
    # Apply mode multiplier
    if mode == "intentional":
        multiplier = random.uniform(*CAMERA_CONFIG["intentional_mode_multiplier"])
    else:  # idle
        multiplier = random.uniform(*CAMERA_CONFIG["idle_mode_multiplier"])
    
    probability = base_probability * multiplier
    return max(0.0, min(1.0, probability))  # Clamp to [0, 1]


def auto_detect_camera_mode(
    target_world_coords: dict = None,
    distance_to_target: float = None,
    action_type: str = None
) -> str:
    """
    Auto-detect camera mode (intentional vs idle) based on context.
    
    Args:
        target_world_coords: World coordinates of target (optional)
        distance_to_target: Distance to target in tiles (optional)
        action_type: Type of action being performed (optional)
    
    Returns:
        "intentional" or "idle"
    """
    # Calculate distance if not provided
    if distance_to_target is None and target_world_coords:
        from actions import player
        player_x = player.get_x()
        player_y = player.get_y()
        if isinstance(player_x, int) and isinstance(player_y, int):
            target_x = target_world_coords.get("x")
            target_y = target_world_coords.get("y")
            if isinstance(target_x, int) and isinstance(target_y, int):
                dx = abs(target_x - player_x)
                dy = abs(target_y - player_y)
                distance_to_target = dx + dy  # Manhattan distance
    
    # Mode detection logic
    if distance_to_target is not None:
        if distance_to_target < CAMERA_CONFIG["intentional_distance_threshold"]:
            return "intentional"  # Close to destination
        elif distance_to_target > CAMERA_CONFIG["idle_distance_threshold"]:
            return "idle"  # Long travel
    
    # Action-based detection
    if action_type:
        intentional_actions = ["click_object", "click_npc", "arrive_at_bank", "arrive_at_destination"]
        if action_type in intentional_actions:
            return "intentional"
        idle_actions = ["long_travel", "waiting", "idle_activity"]
        if action_type in idle_actions:
            return "idle"
    
    # Default to intentional (more conservative)
    return "intentional"


def calculate_camera_adjustments(
    target_world_coords: dict,
    character_state: dict = None,
    distance_to_target: float = None,
    mode: str = "intentional"
) -> dict:
    """
    Calculate camera adjustments (yaw, pitch, zoom) based on target and mode.
    
    Args:
        target_world_coords: World coordinates of target {"x": int, "y": int}
        character_state: Optional character state (movement direction, orientation, etc.)
        distance_to_target: Distance to target in tiles
        mode: "intentional" or "idle"
    
    Returns:
        Dict with adjustments: {yaw_angle: float, pitch_adjustment: int, zoom_adjustment: int, duration_ms: int}
    """
    from actions import player
    
    # Get current camera state
    camera = ipc.get_camera() or {}
    current_yaw_rl = camera.get("yaw", 0)
    current_pitch = camera.get("pitch", 0)
    current_zoom = camera.get("scale", 551)
    
    # Get player position
    player_x = player.get_x()
    player_y = player.get_y()
    target_x = target_world_coords.get("x")
    target_y = target_world_coords.get("y")
    
    if not all(isinstance(v, int) for v in [player_x, player_y, target_x, target_y]):
        return None
    
    # Calculate distance if not provided
    if distance_to_target is None:
        dx = abs(target_x - player_x)
        dy = abs(target_y - player_y)
        distance_to_target = dx + dy
    
    # Calculate yaw adjustment (angle to target)
    import math
    angle_rad = math.atan2(target_y - player_y, target_x - player_x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    # Convert to RuneLite yaw units (0-2047)
    desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
    angle_diff_rl = desired_yaw_rl - current_yaw_rl
    
    # Normalize to -1024 to 1024 range
    while angle_diff_rl > 1024:
        angle_diff_rl -= 2048
    while angle_diff_rl < -1024:
        angle_diff_rl += 2048
    
    angle_diff_deg = abs(angle_diff_rl) * (360 / 2047)
    
    # Calculate pitch adjustment based on distance
    if distance_to_target < CAMERA_CONFIG["close_distance_threshold"]:
        # Close: higher pitch (look down)
        desired_pitch = normal_number(
            CAMERA_CONFIG["pitch_close_min"],
            CAMERA_CONFIG["pitch_close_max"],
            center_bias=0.7,
            output_type="int"
        )
    elif distance_to_target > CAMERA_CONFIG["far_distance_threshold"]:
        # Far: lower pitch (look up)
        desired_pitch = normal_number(
            CAMERA_CONFIG["pitch_far_min"],
            CAMERA_CONFIG["pitch_far_max"],
            center_bias=0.7,
            output_type="int"
        )
    else:
        # Medium: interpolate
        ratio = (distance_to_target - CAMERA_CONFIG["close_distance_threshold"]) / \
                (CAMERA_CONFIG["far_distance_threshold"] - CAMERA_CONFIG["close_distance_threshold"])
        close_pitch = (CAMERA_CONFIG["pitch_close_min"] + CAMERA_CONFIG["pitch_close_max"]) / 2
        far_pitch = (CAMERA_CONFIG["pitch_far_min"] + CAMERA_CONFIG["pitch_far_max"]) / 2
        desired_pitch = int(close_pitch + (far_pitch - close_pitch) * ratio)
    
    pitch_adjustment = desired_pitch - current_pitch
    
    # Calculate zoom adjustment based on distance
    if distance_to_target < CAMERA_CONFIG["close_distance_threshold"]:
        # Close: zoom in
        desired_zoom = normal_number(
            CAMERA_CONFIG["zoom_close_min"],
            CAMERA_CONFIG["zoom_close_max"],
            center_bias=0.7,
            output_type="int"
        )
    elif distance_to_target > CAMERA_CONFIG["far_distance_threshold"]:
        # Far: zoom out
        desired_zoom = normal_number(
            CAMERA_CONFIG["zoom_far_min"],
            CAMERA_CONFIG["zoom_far_max"],
            center_bias=0.7,
            output_type="int"
        )
    else:
        # Medium: interpolate
        ratio = (distance_to_target - CAMERA_CONFIG["close_distance_threshold"]) / \
                (CAMERA_CONFIG["far_distance_threshold"] - CAMERA_CONFIG["close_distance_threshold"])
        close_zoom = (CAMERA_CONFIG["zoom_close_min"] + CAMERA_CONFIG["zoom_close_max"]) / 2
        far_zoom = (CAMERA_CONFIG["zoom_far_min"] + CAMERA_CONFIG["zoom_far_max"]) / 2
        desired_zoom = int(close_zoom + (far_zoom - close_zoom) * ratio)
    
    zoom_adjustment = desired_zoom - current_zoom
    
    # Calculate duration based on mode
    if mode == "intentional":
        duration_ms = normal_number(
            CAMERA_CONFIG["intentional_duration_min"],
            CAMERA_CONFIG["intentional_duration_max"],
            center_bias=0.6,
            output_type="int"
        )
    else:  # idle
        duration_ms = exponential_number(
            CAMERA_CONFIG["idle_duration_min"],
            CAMERA_CONFIG["idle_duration_max"],
            lambda_param=1.5,
            output_type="int"
        )
    
    return {
        "yaw_angle_diff": angle_diff_deg,
        "yaw_angle_diff_rl": angle_diff_rl,
        "pitch_adjustment": pitch_adjustment,
        "zoom_adjustment": zoom_adjustment,
        "duration_ms": duration_ms
    }


def apply_camera_adjustment(adjustments: dict, mode: str = "intentional") -> bool:
    """
    Apply discrete camera adjustments (yaw, pitch, zoom).
    
    Args:
        adjustments: Dict from calculate_camera_adjustments()
        mode: "intentional" or "idle"
    
    Returns:
        True if adjustments were made, False otherwise
    """
    if not adjustments:
        return False
    
    yaw_diff_rl = adjustments.get("yaw_angle_diff_rl", 0)
    pitch_adj = adjustments.get("pitch_adjustment", 0)
    zoom_adj = adjustments.get("zoom_adjustment", 0)
    duration_ms = adjustments.get("duration_ms", 200)
    
    # Dead zone for yaw (skip if too small)
    if abs(yaw_diff_rl) < 57:  # ~10 degrees
        yaw_diff_rl = 0
    
    # Apply yaw adjustment
    if yaw_diff_rl != 0:
        yaw_key = "RIGHT" if yaw_diff_rl > 0 else "LEFT"
        yaw_duration = duration_ms / 1000.0  # Convert to seconds
        
        # Scale duration based on angle (larger angle = longer duration)
        angle_ratio = min(1.0, abs(yaw_diff_rl) / 1024.0)  # Max rotation is 1024 units
        yaw_duration = yaw_duration * (0.5 + 0.5 * angle_ratio)
        
        ipc.key_press(yaw_key)
        sleep_exponential(yaw_duration * 0.8, yaw_duration * 1.2, 1.0)
        ipc.key_release(yaw_key)
    
    # Apply pitch adjustment (if significant)
    if abs(pitch_adj) > 20:  # Dead zone
        pitch_key = "UP" if pitch_adj > 0 else "DOWN"
        pitch_duration = (abs(pitch_adj) / 500.0) * (duration_ms / 1000.0)  # Scale by adjustment size
        pitch_duration = max(0.05, min(0.3, pitch_duration))  # Clamp
        
        ipc.key_press(pitch_key)
        sleep_exponential(pitch_duration * 0.8, pitch_duration * 1.2, 1.0)
        ipc.key_release(pitch_key)
    
    # Apply zoom adjustment (scroll doesn't need press/release)
    if abs(zoom_adj) > 10:  # Dead zone
        scroll_amount = 1 if zoom_adj > 0 else -1
        scroll_count = min(abs(zoom_adj) // 10, 3)  # Max 3 scrolls at once
        for _ in range(scroll_count):
            ipc.scroll(scroll_amount)
            sleep_exponential(0.05, 0.15, 1.0)
    
    return yaw_diff_rl != 0 or abs(pitch_adj) > 20 or abs(zoom_adj) > 10


def aim_camera_at_target(
    target_world_coords: dict,
    mode: str = None,
    action_type: str = None,
    distance_to_target: float = None,
    path_waypoints: list = None,
    movement_state: dict = None,
    max_ms: int = 600,
    object_world_coords: dict = None,
    area_center: dict = None,
    disable_pitch: bool = False
) -> bool:
    """
    Main entry point for new camera system with state-based behavior.
    Uses camera state to determine behavior, then uses continuous feedback loop.
    Incorporates character orientation and path to influence camera positioning.
    
    Args:
        target_world_coords: World coordinates {"x": int, "y": int}
        mode: "intentional" or "idle" (None = auto-detect)
        action_type: Type of action (for auto-detection)
        distance_to_target: Distance in tiles (for auto-detection)
        path_waypoints: Optional list of waypoint dicts from pathfinding (for path-aware camera)
        movement_state: Optional movement state dict with orientation, movement_direction, etc.
        max_ms: Maximum time for camera aiming (default: 600ms)
        object_world_coords: Optional object world coordinates (for OBJECT_INTERACTION state)
        area_center: Optional area center coordinates (for AREA_ACTIVITY state)
        disable_pitch: If True, pitch will not be adjusted (default: False)
    
    Returns:
        True if camera was adjusted, False otherwise
    """
    try:
        from actions import player
        
        # Check for camera state
        camera_state, state_config, stored_area_center, _ = get_camera_state()
        
        # Use stored area_center if not provided
        if area_center is None and stored_area_center:
            area_center = stored_area_center
        
        # Check idle probability if state is set
        if camera_state and state_config:
            idle_prob = state_config.get("idle_probability", 0.0)
            if random.random() < idle_prob:
                return False  # Skip camera movement based on state config
        
        # Auto-detect mode if not provided
        if mode is None:
            mode = auto_detect_camera_mode(
                target_world_coords=target_world_coords,
                distance_to_target=distance_to_target,
                action_type=action_type
            )
        
        # Determine target coordinates based on camera state
        target_x = target_world_coords.get("x")
        target_y = target_world_coords.get("y")
        if not isinstance(target_x, int) or not isinstance(target_y, int):
            return False
        
        # State-based target selection
        if camera_state and state_config:
            target_preference = state_config.get("target_preference", "character")
            yaw_behavior = state_config.get("yaw_behavior", "follow_character")
            
            player_x = player.get_x()
            player_y = player.get_y()
            
            if isinstance(player_x, int) and isinstance(player_y, int):
                # Determine target based on state
                if target_preference == "area_center" and area_center:
                    target_x = area_center.get("x", target_x)
                    target_y = area_center.get("y", target_y)
                elif target_preference == "object" and object_world_coords:
                    target_x = object_world_coords.get("x", target_x)
                    target_y = object_world_coords.get("y", target_y)
                elif target_preference == "path_ahead" and path_waypoints:
                    lookahead = _calculate_path_lookahead_point(
                        path_waypoints=path_waypoints,
                        player_x=player_x,
                        player_y=player_y,
                        final_dest_x=target_x,
                        final_dest_y=target_y,
                        movement_state=movement_state
                    )
                    target_x = lookahead["x"]
                    target_y = lookahead["y"]
                elif target_preference == "character":
                    # Use character position (will be handled in state-specific logic)
                    pass
        else:
            # Default behavior: use path look-ahead if available
            if path_waypoints:
                player_x = player.get_x()
                player_y = player.get_y()
                if isinstance(player_x, int) and isinstance(player_y, int):
                    lookahead = _calculate_path_lookahead_point(
                        path_waypoints=path_waypoints,
                        player_x=player_x,
                        player_y=player_y,
                        final_dest_x=target_x,
                        final_dest_y=target_y,
                        movement_state=movement_state
                    )
                    target_x = lookahead["x"]
                    target_y = lookahead["y"]
        
        # Use state-specific camera adjustment
        if camera_state and state_config:
            return _aim_camera_with_state(
                target_world_coords={"x": target_x, "y": target_y},
                camera_state=camera_state,
                state_config=state_config,
                movement_state=movement_state,
                object_world_coords=object_world_coords,
                area_center=area_center,
                path_waypoints=path_waypoints,
                max_ms=max_ms,
                disable_pitch=disable_pitch
            )
        else:
            # Fallback to original behavior
            return _aim_camera_continuous_feedback(
                target_world_coords={"x": target_x, "y": target_y},
                mode=mode,
                movement_state=movement_state,
                max_ms=max_ms,
                disable_pitch=disable_pitch
            )
        
    except Exception as e:
        logging.error(f"[AIM_CAMERA] Failed to aim camera at target: {e}")
        return False


def _aim_camera_with_state(
    target_world_coords: dict,
    camera_state: str,
    state_config: dict,
    movement_state: dict = None,
    object_world_coords: dict = None,
    area_center: dict = None,
    path_waypoints: list = None,
    max_ms: int = 600,
    disable_pitch: bool = False
) -> bool:
    """
    State-specific camera adjustment with continuous feedback.
    Implements different behaviors based on camera state configuration.
    
    Args:
        target_world_coords: Base target world coordinates {"x": int, "y": int}
        camera_state: Current camera state constant
        state_config: State configuration dict
        movement_state: Optional movement state dict
        object_world_coords: Optional object coordinates (for OBJECT_INTERACTION)
        area_center: Optional area center coordinates (for AREA_ACTIVITY)
        path_waypoints: Optional path waypoints (for LONG_TRAVEL)
        max_ms: Maximum time for camera aiming
    
    Returns:
        True if camera was adjusted, False otherwise
    """
    from actions import player
    
    try:
        player_x = player.get_x()
        player_y = player.get_y()
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return False
        
        # Get character orientation
        character_orientation = None
        if movement_state:
            character_orientation = movement_state.get("orientation")
        if character_orientation is None:
            player_data = ipc.get_player() or {}
            character_orientation = player_data.get("orientation")
        
        # Determine target based on yaw_behavior
        yaw_behavior = state_config.get("yaw_behavior", "follow_character")
        target_x = target_world_coords.get("x")
        target_y = target_world_coords.get("y")
        
        if yaw_behavior == "follow_character":
            # Camera follows character orientation - use character position as target
            # Yaw will be based on character orientation
            target_x = player_x
            target_y = player_y
        elif yaw_behavior == "point_to_area_center" and area_center:
            target_x = area_center.get("x", target_x)
            target_y = area_center.get("y", target_y)
        elif yaw_behavior == "point_to_object" and object_world_coords:
            target_x = object_world_coords.get("x", target_x)
            target_y = object_world_coords.get("y", target_y)
        elif yaw_behavior == "follow_path" and path_waypoints:
            lookahead = _calculate_path_lookahead_point(
                path_waypoints=path_waypoints,
                player_x=player_x,
                player_y=player_y,
                final_dest_x=target_x,
                final_dest_y=target_y,
                movement_state=movement_state
            )
            target_x = lookahead["x"]
            target_y = lookahead["y"]
        
        # Get current camera state
        camera = ipc.get_camera() or {}
        current_yaw_rl = camera.get("yaw", 0)
        current_pitch = camera.get("pitch", 0)
        current_zoom = camera.get("scale", 551)
        
        # Calculate desired yaw based on yaw_behavior
        desired_yaw_rl = None
        if yaw_behavior == "follow_character" and character_orientation is not None:
            # Convert character orientation to camera yaw
            # Character orientation: 0=South, 512=West, 1024=North, 1536=East
            # Camera yaw: 0=North, increases clockwise
            # Map character orientation to camera yaw
            char_orient = character_orientation
            if char_orient == 0:  # South
                desired_yaw_rl = 1024  # Camera looks south (1024 in RL units)
            elif char_orient == 512:  # West
                desired_yaw_rl = 1536  # Camera looks west
            elif char_orient == 1024:  # North
                desired_yaw_rl = 0  # Camera looks north
            elif char_orient == 1536:  # East
                desired_yaw_rl = 512  # Camera looks east
            else:
                # Interpolate for other orientations
                desired_yaw_rl = int((char_orient + 1024) % 2048)
        else:
            # Calculate yaw to target
            import math
            angle_rad = math.atan2(target_y - player_y, target_x - player_x)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
        
        # Get zoom and pitch preferences
        zoom_preference = state_config.get("zoom_preference", "auto")
        pitch_behavior = state_config.get("pitch_behavior", "auto")
        zoom_range = state_config.get("zoom_range", (450, 600))
        pitch_range = state_config.get("pitch_range", (200, 400))
        
        # Calculate desired zoom
        desired_zoom = current_zoom
        if zoom_preference == "area_wide":
            desired_zoom = normal_number(zoom_range[0], zoom_range[1], center_bias=0.7, output_type="int")
        elif zoom_preference == "close_up":
            desired_zoom = normal_number(400, 500, center_bias=0.7, output_type="int")
        elif zoom_preference == "medium":
            desired_zoom = normal_number(450, 550, center_bias=0.7, output_type="int")
        # "auto" keeps current zoom
        
        # Calculate desired pitch
        desired_pitch = current_pitch
        if pitch_behavior == "overhead":
            desired_pitch = normal_number(400, 500, center_bias=0.7, output_type="int")
        elif pitch_behavior == "angled":
            desired_pitch = normal_number(200, 350, center_bias=0.7, output_type="int")
        elif pitch_behavior == "auto":
            desired_pitch = normal_number(pitch_range[0], pitch_range[1], center_bias=0.7, output_type="int")
        
        # Enforce minimum pitch of 280
        desired_pitch = max(280, desired_pitch)
        
        # Continuous feedback loop with state-specific adjustments
        max_aim_time = max(0.05, min(5.0, float(max_ms) / 1000.0))
        check_interval = 0.05
        start_time = time.time()
        pressed_keys = set()
        
        try:
            while time.time() - start_time < max_aim_time:
                # Get current camera state
                camera = ipc.get_camera() or {}
                current_yaw_rl = camera.get("yaw", 0)
                current_pitch = camera.get("pitch", 0)
                current_zoom = camera.get("scale", 551)
                
                # Check yaw adjustment
                if desired_yaw_rl is not None:
                    angle_diff_rl = desired_yaw_rl - current_yaw_rl
                    while angle_diff_rl > 1024:
                        angle_diff_rl -= 2048
                    while angle_diff_rl < -1024:
                        angle_diff_rl += 2048
                    
                    if abs(angle_diff_rl) > 57:  # ~10 degrees dead zone
                        yaw_key = "RIGHT" if angle_diff_rl > 0 else "LEFT"
                        if yaw_key not in pressed_keys:
                            other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                            if other_yaw_key in pressed_keys:
                                ipc.key_release(other_yaw_key)
                                pressed_keys.discard(other_yaw_key)
                            ipc.key_press(yaw_key)
                            pressed_keys.add(yaw_key)
                    else:
                        # Release yaw keys if close enough
                        for yaw_key in ["LEFT", "RIGHT"]:
                            if yaw_key in pressed_keys:
                                ipc.key_release(yaw_key)
                                pressed_keys.discard(yaw_key)
                
                # Check pitch adjustment (with minimum pitch constraint of 280)
                if not disable_pitch:
                    pitch_diff = desired_pitch - current_pitch
                    if abs(pitch_diff) > 20:  # Dead zone
                        pitch_key = "UP" if pitch_diff > 0 else "DOWN"
                        # Prevent pitch from going below 280
                        if pitch_key == "DOWN" and current_pitch <= 280:
                            # Don't allow DOWN if pitch is already at or below minimum
                            pitch_key = None
                        if pitch_key and pitch_key not in pressed_keys:
                            other_pitch_key = "DOWN" if pitch_key == "UP" else "UP"
                            if other_pitch_key in pressed_keys:
                                ipc.key_release(other_pitch_key)
                                pressed_keys.discard(other_pitch_key)
                            ipc.key_press(pitch_key)
                            pressed_keys.add(pitch_key)
                    else:
                        # Release pitch keys if close enough
                        for pitch_key in ["UP", "DOWN"]:
                            if pitch_key in pressed_keys:
                                ipc.key_release(pitch_key)
                                pressed_keys.discard(pitch_key)
                else:
                    # Release any pitch keys if pitch is disabled
                    for pitch_key in ["UP", "DOWN"]:
                        if pitch_key in pressed_keys:
                            ipc.key_release(pitch_key)
                            pressed_keys.discard(pitch_key)
                    pitch_diff = 0  # Set to 0 so pitch_ok check passes
                
                # Check zoom adjustment
                zoom_diff = desired_zoom - current_zoom
                if abs(zoom_diff) > 10:  # Dead zone
                    scroll_amount = 1 if zoom_diff > 0 else -1
                    scroll_count = min(abs(zoom_diff) // 10, 3)
                    for _ in range(scroll_count):
                        ipc.scroll(scroll_amount)
                        sleep_exponential(0.05, 0.15, 1.0)
                
                # Check if we're close enough to all targets
                angle_diff_rl_check = 0
                if desired_yaw_rl is not None:
                    angle_diff_rl_check = desired_yaw_rl - current_yaw_rl
                    while angle_diff_rl_check > 1024:
                        angle_diff_rl_check -= 2048
                    while angle_diff_rl_check < -1024:
                        angle_diff_rl_check += 2048
                
                yaw_ok = desired_yaw_rl is None or abs(angle_diff_rl_check) <= 57
                pitch_ok = disable_pitch or abs(pitch_diff) <= 20
                zoom_ok = abs(zoom_diff) <= 10
                
                if yaw_ok and pitch_ok and zoom_ok:
                    break  # All adjustments complete
                
                sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
        
        finally:
            # Release all keys
            for key in pressed_keys.copy():
                ipc.key_release(key)
                pressed_keys.discard(key)
        
        return len(pressed_keys) > 0 or abs(zoom_diff) > 10
        
    except Exception as e:
        logging.error(f"[AIM_CAMERA_STATE] Failed: {e}")
        return False


def _aim_camera_continuous_feedback(
    target_world_coords: dict,
    mode: str = "intentional",
    movement_state: dict = None,
    max_ms: int = 600,
    disable_pitch: bool = False
) -> bool:
    """
    Continuous feedback loop for camera adjustment.
    Incorporates character orientation and movement direction.
    
    Args:
        target_world_coords: World coordinates {"x": int, "y": int}
        mode: "intentional" or "idle"
        movement_state: Optional movement state dict with orientation, movement_direction, etc.
        max_ms: Maximum time for camera aiming
        disable_pitch: If True, pitch will not be adjusted (default: False)
    
    Returns:
        True if camera was adjusted, False otherwise
    """
    from actions import player
    
    try:
        target_x = target_world_coords.get("x")
        target_y = target_world_coords.get("y")
        if not isinstance(target_x, int) or not isinstance(target_y, int):
            return False
        
        # Get player position and orientation
        player_x = player.get_x()
        player_y = player.get_y()
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return False
        
        # Get character orientation if available
        character_orientation = None
        if movement_state:
            character_orientation = movement_state.get("orientation")
        if character_orientation is None:
            # Try to get from IPC
            player_data = ipc.get_player() or {}
            character_orientation = player_data.get("orientation")
        
        # Get screen dimensions
        where = ipc.where() or {}
        screen_width = int(where.get("w", 0))
        screen_height = int(where.get("h", 0))
        if screen_width == 0 or screen_height == 0:
            return False
        
        # Target screen position (sweet spot: center horizontally, 2/3 up vertically)
        target_screen_x = screen_width // 2
        target_screen_y = int(screen_height * CAMERA_CONFIG["sweet_spot_y_ratio"])
        
        # Continuous feedback loop
        max_aim_time = max(0.05, min(5.0, float(max_ms) / 1000.0))
        check_interval = 0.05  # Check every 50ms
        start_time = time.time()
        pressed_keys = set()
        
        try:
            while time.time() - start_time < max_aim_time:
                # Project target to screen
                proj = ipc.project_world_tile(target_x, target_y) or {}
                if not proj.get("ok"):
                    # Off-screen - rotate camera toward target using orientation
                    if character_orientation is not None:
                        # Calculate desired yaw based on target direction
                        import math
                        angle_rad = math.atan2(target_y - player_y, target_x - player_x)
                        angle_deg = math.degrees(angle_rad)
                        if angle_deg < 0:
                            angle_deg += 360
                        
                        # Convert to RuneLite yaw units
                        desired_yaw_rl = int((360 - angle_deg) % 360 * (2047 / 360))
                        
                        # Get current camera yaw
                        camera = ipc.get_camera() or {}
                        current_yaw_rl = camera.get("yaw", 0)
                        
                        # Calculate angle difference
                        angle_diff_rl = desired_yaw_rl - current_yaw_rl
                        while angle_diff_rl > 1024:
                            angle_diff_rl -= 2048
                        while angle_diff_rl < -1024:
                            angle_diff_rl += 2048
                        
                        # Rotate if difference is significant
                        if abs(angle_diff_rl) > 57:  # ~10 degrees
                            yaw_key = "RIGHT" if angle_diff_rl > 0 else "LEFT"
                            if yaw_key not in pressed_keys:
                                other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                                if other_yaw_key in pressed_keys:
                                    ipc.key_release(other_yaw_key)
                                    pressed_keys.discard(other_yaw_key)
                                ipc.key_press(yaw_key)
                                pressed_keys.add(yaw_key)
                    
                    sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
                    continue
                
                # Get current screen position
                screen_x = int(proj.get("canvas", {}).get("x", 0))
                screen_y = int(proj.get("canvas", {}).get("y", 0))
                
                # Get current camera state for pitch check
                camera = ipc.get_camera() or {}
                current_pitch = int(camera.get("pitch", 0))
                
                # Calculate offsets from target position
                dx = screen_x - target_screen_x
                dy = screen_y - target_screen_y
                
                # Dead zones - increased for more lenient camera movement
                # NPCs within this zone are considered "close enough" to the target position
                x_dead_zone = 100
                y_dead_zone = 100
                
                # Check if we're close enough
                if abs(dx) <= x_dead_zone and abs(dy) <= y_dead_zone:
                    # Close enough - release all keys and exit
                    break
                
                # Determine needed adjustments
                needs_yaw = abs(dx) > x_dead_zone
                needs_pitch = abs(dy) > y_dead_zone
                
                # Apply yaw adjustment
                if needs_yaw:
                    yaw_key = "LEFT" if dx > 0 else "RIGHT"
                    if yaw_key not in pressed_keys:
                        other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                        if other_yaw_key in pressed_keys:
                            ipc.key_release(other_yaw_key)
                            pressed_keys.discard(other_yaw_key)
                        ipc.key_press(yaw_key)
                        pressed_keys.add(yaw_key)
                else:
                    # Release yaw keys if not needed
                    for yaw_key in ["LEFT", "RIGHT"]:
                        if yaw_key in pressed_keys:
                            ipc.key_release(yaw_key)
                            pressed_keys.discard(yaw_key)
                
                # Apply pitch adjustment (with minimum pitch constraint of 280)
                if not disable_pitch:
                    if needs_pitch:
                        pitch_key = "UP" if dy > 0 else "DOWN"
                        # Prevent pitch from going below 280
                        if pitch_key == "DOWN" and current_pitch <= 280:
                            # Don't allow DOWN if pitch is already at or below minimum
                            pitch_key = None
                        if pitch_key and pitch_key not in pressed_keys:
                            other_pitch_key = "DOWN" if pitch_key == "UP" else "UP"
                            if other_pitch_key in pressed_keys:
                                ipc.key_release(other_pitch_key)
                                pressed_keys.discard(other_pitch_key)
                            ipc.key_press(pitch_key)
                            pressed_keys.add(pitch_key)
                    else:
                        # Release pitch keys if not needed
                        for pitch_key in ["UP", "DOWN"]:
                            if pitch_key in pressed_keys:
                                ipc.key_release(pitch_key)
                                pressed_keys.discard(pitch_key)
                else:
                    # Release any pitch keys if pitch is disabled
                    for pitch_key in ["UP", "DOWN"]:
                        if pitch_key in pressed_keys:
                            ipc.key_release(pitch_key)
                            pressed_keys.discard(pitch_key)
                    # Set needs_pitch to False so the dead zone check passes
                    needs_pitch = False
                
                # Wait before next check
                sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
        
        finally:
            # Always release all pressed keys when done
            for key in pressed_keys.copy():
                ipc.key_release(key)
                pressed_keys.discard(key)
        
        return len(pressed_keys) > 0  # Return True if we made any adjustments
        
    except Exception as e:
        logging.error(f"[AIM_CAMERA_CONTINUOUS] Failed: {e}")
        return False


def aim_midtop_at_world(
    wx: int, 
    wy: int, 
    *, 
    max_ms: int = 600,
    path_waypoints: list = None,
    movement_state: dict = None,
    look_ahead_distance: int = None
):
    """
    Aim camera at world coordinates using key press -> continuous check -> key release.
    
    If path_waypoints are provided, calculates optimal look-ahead point along the path
    and aims at that instead of the raw coordinates. This provides smoother, more natural
    camera movement that follows the path direction.
    
    Args:
        wx, wy: World coordinates to aim at (or final destination if path_waypoints provided)
        max_ms: Maximum time for camera aiming
        path_waypoints: Optional list of waypoint dicts from pathfinding
        movement_state: Optional movement state dict with is_running, movement_direction, etc.
        look_ahead_distance: Optional override for look-ahead distance (in tiles)
    """
    
    try:
        # If path_waypoints are provided, calculate optimal look-ahead point
        target_wx, target_wy = wx, wy
        if path_waypoints:
            # Get player position for look-ahead calculation
            from actions import player
            player_x = player.get_x()
            player_y = player.get_y()
            
            if isinstance(player_x, int) and isinstance(player_y, int):
                # Calculate look-ahead point along path
                lookahead = _calculate_path_lookahead_point(
                    path_waypoints=path_waypoints,
                    player_x=player_x,
                    player_y=player_y,
                    final_dest_x=wx,
                    final_dest_y=wy,
                    movement_state=movement_state
                )
                target_wx = lookahead["x"]
                target_wy = lookahead["y"]
        
        # Get window size
        where = ipc.where() or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))

        # Target position (center horizontally, 30% down vertically)
        center_x = W // 2
        target_y = int(H * 0.30)
        
        # Dead zones with buffer room
        pitch_dead_zone = 30  # New buffer for pitch
        
        # Continuous checking with timeout
        # Respect caller-provided max_ms (default 600ms). Clamp to sane bounds.
        max_aim_time = max(0.05, min(5.0, float(max_ms) / 1000.0))
        check_interval = 0.05  # Check every 50ms for more responsive control
        start_time = time.time()
        
        # Track which keys are currently pressed
        pressed_keys = set()
        
        try:
            while time.time() - start_time < max_aim_time:
                # Project world coordinates to screen (use calculated look-ahead point if available)
                proj = ipc.project_world_tile(int(target_wx), int(target_wy)) or {}
                
                if not proj.get("ok"):
                    sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
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
                
                # Determine what adjustments are needed
                # NOTE: Pitch movement is DISABLED for non-jacobian travel methods
                # Only adjust yaw and scale, never pitch
                needs_yaw = not x_in_range or not y_in_range
                needs_scale = not (500 <= current_scale <= 600)  # Check if scale is in reasonable range

                # Exit early once we're "good enough"
                # Note: We don't check pitch here since we're not adjusting it
                if (x_in_range and y_in_range) and (not needs_yaw) and (not needs_scale):
                    break
                
                # Release any pitch keys that might be pressed (safety measure)
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
                
                # Wait before next check (use existing sleep util for variability)
                sleep_exponential(check_interval * 0.8, check_interval * 1.2, 1.0)
        
        finally:
            # Always release all pressed keys when done
            for key in pressed_keys.copy():
                ipc.key_release(key)
                pressed_keys.discard(key)

    except Exception as e:
        logging.error(f"[AIM_CAMERA] Failed to aim camera at world coordinates ({wx}, {wy}): {e}")
        # Release any keys that might still be pressed
        try:
            for key in ["UP", "DOWN", "LEFT", "RIGHT"]:
                ipc.key_release(key)
        except Exception:
            pass

def aim_camera_along_path(
    path_waypoints: list,
    player_x: int,
    player_y: int,
    movement_state: dict = None,
    max_rotation_ms: int = 300
):
    """
    Smoothly rotate camera to follow path direction.
    
    This function:
    - Calculates overall path direction from waypoints
    - Determines desired camera yaw angle
    - Rotates camera smoothly in that direction
    - Makes occasional, human-like adjustments
    
    Args:
        path_waypoints: List of waypoint dicts from pathfinding
        player_x, player_y: Current player position
        movement_state: Optional movement state dict
        max_rotation_ms: Maximum time for camera rotation (default: 300ms)
    """
    if not path_waypoints or len(path_waypoints) == 0:
        return
    
    try:
        # Calculate path direction
        path_dir = _calculate_path_direction(path_waypoints, player_x, player_y, num_waypoints=5)
        
        if path_dir["dx"] == 0 and path_dir["dy"] == 0:
            return  # No direction to follow
        
        # Get current camera state
        camera = ipc.get_camera() or {}
        current_yaw_rl = camera.get("yaw", 0)  # Camera yaw in RuneLite units (0-2047)
        
        # RuneLite camera yaw: 0-2047 range, where 0 = north, increases clockwise
        # Our path angle: 0-360 degrees, where 0 = north, increases counter-clockwise (standard math)
        # Convert path angle to RuneLite yaw units
        # RuneLite_yaw = (360 - math_angle) * (2047 / 360)  [convert to RuneLite units and reverse direction]
        path_angle_deg = path_dir["angle"]
        desired_yaw_rl = int((360 - path_angle_deg) % 360 * (2047 / 360))
        
        # Calculate angle difference (shortest rotation in RuneLite units)
        angle_diff_rl = desired_yaw_rl - current_yaw_rl
        
        # Normalize to -1024 to 1024 range (half rotation)
        while angle_diff_rl > 1024:
            angle_diff_rl -= 2048
        while angle_diff_rl < -1024:
            angle_diff_rl += 2048
        
        # Convert to degrees for easier calculation
        angle_diff_deg = abs(angle_diff_rl) * (360 / 2047)
        
        # If difference is small, don't rotate (dead zone)
        # 10 degrees in RuneLite units = ~57 units
        if angle_diff_deg < 10:
            return
        
        # Calculate rotation duration based on angle difference
        # Base duration proportional to angle (max_rotation_ms for 360 degrees)
        base_duration_seconds = (angle_diff_deg / 360.0) * (max_rotation_ms / 1000.0)
        
        # Use exponential timing for human-like variation
        duration_seconds = exponential_number(
            base_duration_seconds * 0.7,
            base_duration_seconds * 1.3,
            lambda_param=1.5,
            output_type="float"
        )
        
        # Clamp duration
        min_duration = 0.05  # 50ms minimum
        max_duration = max_rotation_ms / 1000.0  # Convert to seconds
        duration_seconds = max(min_duration, min(duration_seconds, max_duration))
        
        # Choose rotation direction
        # In RuneLite: LEFT decreases yaw (counter-clockwise), RIGHT increases yaw (clockwise)
        if angle_diff_rl > 0:
            # Need to rotate clockwise (increase yaw) - press RIGHT
            rotation_key = "RIGHT"
        else:
            # Need to rotate counter-clockwise (decrease yaw) - press LEFT
            rotation_key = "LEFT"
        
        # Perform smooth rotation
        ipc.key_press(rotation_key)
        sleep_exponential(duration_seconds * 0.8, duration_seconds * 1.2, 1.0)
        ipc.key_release(rotation_key)
        
    except Exception as e:
        logging.debug(f"[AIM_CAMERA_PATH] Failed to rotate camera along path: {e}")


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