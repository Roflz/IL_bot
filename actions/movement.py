"""
Movement direction tracking and precision walking utilities.
"""
import time
from typing import Optional, Tuple, Dict
from actions import player
from helpers.runtime_utils import ipc


# Global state for movement tracking
_movement_state = {
    "last_position": None,
    "last_position_ts": None,
    "movement_direction": None,  # (dx, dy) tuple
}


def get_movement_direction() -> Optional[Tuple[int, int]]:
    """
    Get the current movement direction based on recent position changes.
    
    Returns:
        Tuple (dx, dy) representing movement direction, or None if not moving
    """
    global _movement_state
    
    current_x = player.get_x()
    current_y = player.get_y()
    
    if current_x is None or current_y is None:
        return None
    
    now = time.time()
    last_pos = _movement_state.get("last_position")
    last_ts = _movement_state.get("last_position_ts")
    
    # Update position if enough time has passed (at least 100ms)
    if last_pos is None or last_ts is None or (now - last_ts) >= 0.1:
        if last_pos is not None:
            # Calculate direction
            dx = current_x - last_pos[0]
            dy = current_y - last_pos[1]
            
            # Only update if there's actual movement
            if dx != 0 or dy != 0:
                _movement_state["movement_direction"] = (dx, dy)
        
        _movement_state["last_position"] = (current_x, current_y)
        _movement_state["last_position_ts"] = now
    
    return _movement_state.get("movement_direction")


def clear_movement_state():
    """Clear movement tracking state."""
    global _movement_state
    _movement_state = {
        "last_position": None,
        "last_position_ts": None,
        "movement_direction": None,
    }

