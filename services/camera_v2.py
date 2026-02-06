"""
Camera System V2 - Autonomous background camera control.

This module provides an autonomous camera system that runs entirely in a background
thread, continuously monitoring interactions and keeping the camera positioned
optimally using the same movement rules as aim_midtop_at_world.
"""

import logging
import time
import threading
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from helpers.runtime_utils import ipc
from helpers.utils import sleep_exponential


# ============================================================================
# Constants and Configuration
# ============================================================================

@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    # Target screen position (same as aim_midtop_at_world)
    target_screen_x_ratio: float = 0.5  # Center horizontally
    target_screen_y_ratio: float = 0.30  # 30% down from top
    
    # Acceptable ranges (same as aim_midtop_at_world)
    x_range_min: int = 500
    x_range_max: int = 1100
    y_range_max: int = 500  # screen_y < 500
    scale_range_min: int = 500
    scale_range_max: int = 600
    
    # Movement rules
    pitch_disabled: bool = True  # Pitch movement is disabled
    check_interval: float = 0.05  # Check every 50ms
    max_adjustment_time: float = 5.0  # Max time to adjust camera
    
    # Zoom scroll settings
    zoom_scroll_delay: float = 0.05  # Delay between scrolls


# ============================================================================
# Interaction Tracking
# ============================================================================

@dataclass
class InteractionTarget:
    """Represents a current/next interaction target."""
    world_x: int
    world_y: int
    world_z: int = 0
    target_type: str = "unknown"  # "object", "npc", "ground", etc.
    timestamp: float = 0.0  # When this target was set


class InteractionTracker:
    """Tracks current, next, and last interactions."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.current_target: Optional[InteractionTarget] = None
        self.next_target: Optional[InteractionTarget] = None
        self.last_target: Optional[InteractionTarget] = None
    
    def set_current_target(self, world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
        """Set the current interaction target."""
        with self._lock:
            # Move current to last
            if self.current_target:
                self.last_target = self.current_target
            
            # Set new current
            self.current_target = InteractionTarget(
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                target_type=target_type,
                timestamp=time.time()
            )
    
    def set_next_target(self, world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
        """Set the next interaction target."""
        with self._lock:
            self.next_target = InteractionTarget(
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                target_type=target_type,
                timestamp=time.time()
            )
    
    def clear_current_target(self):
        """Clear the current target."""
        with self._lock:
            if self.current_target:
                self.last_target = self.current_target
            self.current_target = None
    
    def get_active_target(self) -> Optional[InteractionTarget]:
        """
        Get the active target to aim at.
        Priority: current_target > next_target > last_target
        """
        with self._lock:
            if self.current_target:
                return self.current_target
            elif self.next_target:
                return self.next_target
            elif self.last_target:
                # Only use last_target if it's recent (within 5 seconds)
                if time.time() - self.last_target.timestamp < 5.0:
                    return self.last_target
            return None


# ============================================================================
# Autonomous Camera Controller
# ============================================================================

class AutonomousCameraController:
    """
    Autonomous camera controller that runs entirely in a background thread.
    
    Continuously monitors interactions and adjusts camera to keep targets
    in the optimal screen position using the same rules as aim_midtop_at_world.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.interaction_tracker = InteractionTracker()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._pressed_keys = set()  # Track currently pressed keys
        
        # Integration with old system (for compatibility)
        self._old_interaction_object = None
    
    def start(self):
        """Start the autonomous camera thread."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._running = False  # Reset flag
                
                # Wait for old thread to finish if it exists
                if self._thread is not None:
                    self._thread.join(timeout=0.5)
                
                # Start new thread
                self._thread = threading.Thread(
                    target=self._autonomous_loop,
                    name="AutonomousCameraThread",
                    daemon=True  # Dies when main thread dies
                )
                self._thread.start()
                logging.info("[CAMERA_V2] Autonomous camera thread started")
    
    def stop(self):
        """Stop the autonomous camera thread gracefully."""
        with self._lock:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
                self._thread = None
            
            # Release all pressed keys
            self._release_all_keys()
            
            logging.info("[CAMERA_V2] Autonomous camera thread stopped")
    
    def _release_all_keys(self):
        """Release all currently pressed keys."""
        for key in list(self._pressed_keys):
            try:
                ipc.key_release(key)
            except Exception:
                pass
        self._pressed_keys.clear()
    
    def _autonomous_loop(self):
        """
        Main autonomous camera loop that runs continuously.
        Monitors interactions and adjusts camera to keep targets in optimal position.
        """
        self._running = True
        
        while self._running:
            try:
                # Check for interaction updates from old system (for compatibility)
                self._sync_with_old_system()
                
                # Get active target
                target = self.interaction_tracker.get_active_target()
                
                if target:
                    # Adjust camera to keep target in optimal position
                    self._adjust_camera_to_target(target)
                else:
                    # No target - release any pressed keys
                    self._release_all_keys()
                
                # Sleep before next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logging.warning(f"[CAMERA_V2] Error in autonomous loop: {e}")
                time.sleep(0.1)
        
        # Cleanup: release all keys
        self._release_all_keys()
    
    def _sync_with_old_system(self):
        """Sync with old camera_integration system for compatibility."""
        try:
            from services.camera_integration import get_camera_state
            _, _, _, interaction_object = get_camera_state()
            
            if interaction_object != self._old_interaction_object:
                self._old_interaction_object = interaction_object
                
                if interaction_object:
                    self.interaction_tracker.set_current_target(
                        world_x=interaction_object.get("x", 0),
                        world_y=interaction_object.get("y", 0),
                        world_z=interaction_object.get("plane", 0),
                        target_type="object"
                    )
                else:
                    self.interaction_tracker.clear_current_target()
        except Exception:
            pass  # Old system not available, continue
    
    def _adjust_camera_to_target(self, target: InteractionTarget):
        """
        Adjust camera to keep target in optimal screen position.
        Uses the same movement rules as aim_midtop_at_world.
        """
        try:
            # Get screen dimensions
            where = ipc.where() or {}
            W = int(where.get("w", 0))
            H = int(where.get("h", 0))
            if W == 0 or H == 0:
                return
            
            # Calculate target screen position (center X, 30% Y)
            center_x = W // 2
            target_y = int(H * self.config.target_screen_y_ratio)
            
            # Project world coordinates to screen
            proj = ipc.project_world_tile(target.world_x, target.world_y)
            if not proj or not proj.get("ok"):
                return
            
            # Get current screen position
            screen_x = int(proj.get('canvas', {}).get("x", 0))
            screen_y = int(proj.get('canvas', {}).get("y", 0))
            
            # Get current camera state
            camera = ipc.get_camera() or {}
            current_scale = int(camera.get("scale", 551))
            
            # Calculate offsets
            dx = screen_x - center_x
            dy = screen_y - target_y
            
            # Check if target is in proper screen position (same rules as aim_midtop_at_world)
            x_in_range = self.config.x_range_min <= screen_x <= self.config.x_range_max
            y_in_range = screen_y < self.config.y_range_max
            scale_in_range = self.config.scale_range_min <= current_scale <= self.config.scale_range_max
            
            # Exit early if already in good position
            if x_in_range and y_in_range and scale_in_range:
                self._release_all_keys()
                return
            
            # Determine what adjustments are needed
            needs_yaw = not x_in_range or not y_in_range
            needs_scale = not scale_in_range
            
            # Release pitch keys (pitch movement is disabled)
            for pitch_key in ["UP", "DOWN"]:
                if pitch_key in self._pressed_keys:
                    ipc.key_release(pitch_key)
                    self._pressed_keys.discard(pitch_key)
            
            # Handle yaw adjustments
            if needs_yaw:
                yaw_key = "LEFT" if dx > 0 else "RIGHT"
                if yaw_key not in self._pressed_keys:
                    # Release the other yaw key first
                    other_yaw_key = "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                    if other_yaw_key in self._pressed_keys:
                        ipc.key_release(other_yaw_key)
                        self._pressed_keys.discard(other_yaw_key)
                    # Press the needed yaw key
                    ipc.key_press(yaw_key)
                    self._pressed_keys.add(yaw_key)
            else:
                # Release yaw keys if we don't need them
                for yaw_key in ["LEFT", "RIGHT"]:
                    if yaw_key in self._pressed_keys:
                        ipc.key_release(yaw_key)
                        self._pressed_keys.discard(yaw_key)
            
            # Handle scale adjustments (scroll doesn't need press/release)
            if needs_scale:
                if current_scale > 551:
                    ipc.scroll(-1)  # Zoom out
                else:
                    ipc.scroll(1)   # Zoom in
                time.sleep(self.config.zoom_scroll_delay)
                
        except Exception as e:
            logging.warning(f"[CAMERA_V2] Error adjusting camera: {e}")
    
    def set_current_target(self, world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
        """Set the current interaction target (called by external code)."""
        self.interaction_tracker.set_current_target(world_x, world_y, world_z, target_type)
    
    def set_next_target(self, world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
        """Set the next interaction target (called by external code)."""
        self.interaction_tracker.set_next_target(world_x, world_y, world_z, target_type)
    
    def clear_current_target(self):
        """Clear the current target (called by external code)."""
        self.interaction_tracker.clear_current_target()


# ============================================================================
# Global Instance
# ============================================================================

_camera_controller: Optional[AutonomousCameraController] = None


def get_camera_controller() -> AutonomousCameraController:
    """Get or create the global camera controller instance."""
    global _camera_controller
    if _camera_controller is None:
        _camera_controller = AutonomousCameraController()
        _camera_controller.start()
    return _camera_controller


def set_current_interaction_target(world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
    """Convenience function to set current interaction target."""
    controller = get_camera_controller()
    controller.set_current_target(world_x, world_y, world_z, target_type)


def set_next_interaction_target(world_x: int, world_y: int, world_z: int = 0, target_type: str = "unknown"):
    """Convenience function to set next interaction target."""
    controller = get_camera_controller()
    controller.set_next_target(world_x, world_y, world_z, target_type)


def clear_interaction_target():
    """Convenience function to clear current interaction target."""
    controller = get_camera_controller()
    controller.clear_current_target()
