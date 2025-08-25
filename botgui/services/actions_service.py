"""
Actions Service for Bot GUI

This service handles real-time action recording and processing,
integrating with the existing bot controller and feature pipeline.
"""

import csv
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
from pynput import mouse, keyboard
import pygetwindow as gw
import logging

LOG = logging.getLogger(__name__)


class ActionsService:
    """
    Service for recording and processing user actions in real-time.
    
    Records mouse movements, clicks, key presses/releases, and scrolls
    when the bot is running, and provides processed action data
    in the same format as the training pipeline.
    """
    
    def __init__(self, controller):
        self.controller = controller
        self.is_recording = False
        self.recording_thread = None
        
        # Action data storage
        self.actions = []
        self.current_session_start = None
        
        # Action counters for the current session
        self.action_counts = {
            'total_actions': 0,
            'mouse_movements': 0,
            'clicks': 0,
            'key_presses': 0,
            'key_releases': 0,
            'scrolls': 0
        }
        
        # Input listeners
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Runelite window detection
        self.runelite_window = None
        self.last_move_time = 0
        self.move_threshold = 0.01  # 10ms throttle for mouse movements
        
        LOG.info("ActionsService initialized")
    
    def start_recording(self):
        """Start recording actions."""
        if self.is_recording:
            LOG.warning("Actions recording already started")
            return
        
        LOG.info("Starting actions recording...")
        
        # Clear previous session data
        self.actions.clear()
        self.action_counts = {k: 0 for k in self.action_counts.keys()}
        
        # Use a session-relative time origin for action events
        self.current_session_start = int(time.time() * 1000)
        
        # Find Runelite window
        if not self._find_runelite_window():
            LOG.warning("Runelite window not found, recording may not work properly")
        
        # Start input listeners
        self._start_listeners()
        
        self.is_recording = True
        LOG.info("Actions recording started")
    
    def stop_recording(self):
        """Stop recording actions."""
        if not self.is_recording:
            LOG.warning("Actions recording not started")
            return
        
        # Stop input listeners
        self._stop_listeners()
        
        self.is_recording = False
    
    def clear_data(self):
        """Clear all recorded action data."""
        self.actions.clear()
        self.action_counts = {k: 0 for k in self.action_counts.keys()}
        self.current_session_start = None
        LOG.info("Actions data cleared")
    
    def get_action_features(self) -> List[List[float]]:
        """
        Get action features as individual tensors for each timestep.
        
        Each timestep gets an action tensor representing actions in the 600ms window
        BEFORE that gamestate, processed exactly like shared_pipeline/actions.py.
        
        Returns:
            List of action tensors, one per timestep, in format:
            [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
        """
        # FIXED: Only check if actions exist, not if recording is active
        # We should be able to access previously recorded actions even when recording is stopped
        if not self.actions:
            return [[0.0]] * 10  # Return 10 empty tensors for T0-T9
        
        # FIXED: Use action timestamps instead of current gamestate timestamps
        # The actions were recorded at specific times, so we need to use those timestamps
        # not current gamestate timestamps which are from the current time
        
        # Get the timestamp range of recorded actions
        if not self.actions:
            return [[0.0]] * 10
        
        action_timestamps = [action.get('timestamp', 0) for action in self.actions]
        min_timestamp = min(action_timestamps)
        max_timestamp = max(action_timestamps)
        
        # Create 10 evenly spaced timesteps within the action time range
        # T0 = most recent actions, T9 = oldest actions
        timestep_duration = (max_timestamp - min_timestamp) // 10  # Duration per timestep
        
        # Create 10 timesteps (T0-T9) with 600ms windows
        action_tensors = []
        for i in range(10):
            # Calculate the center timestamp for this timestep
            # T0 = most recent (max_timestamp), T9 = oldest (min_timestamp)
            if i == 0:
                # T0: center around most recent actions
                center_timestamp = max_timestamp - (timestep_duration // 2)
            elif i == 9:
                # T9: center around oldest actions  
                center_timestamp = min_timestamp + (timestep_duration // 2)
            else:
                # T1-T8: evenly spaced between oldest and newest
                center_timestamp = max_timestamp - (i * timestep_duration) - (timestep_duration // 2)
            
            # Calculate the 600ms window around this center timestamp
            window_start = center_timestamp - 300  # 300ms before center
            window_end = center_timestamp + 300    # 300ms after center
            
            # Get actions in this window
            window_actions = []
            for action in self.actions:
                action_timestamp = action.get('timestamp', 0)
                if window_start <= action_timestamp <= window_end:
                    window_actions.append(action)
            
            # Sort actions by timestamp
            window_actions.sort(key=lambda a: a.get('timestamp', 0))
            
            # Convert to action tensor format: [count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
            action_tensor = [len(window_actions)]  # Start with action count
            
            for action in window_actions:
                # Timestamp (relative to window start)
                rel_timestamp = action.get('timestamp', 0) - window_start
                action_tensor.append(float(rel_timestamp))
                
                # Action type (encode as: 0=move, 1=click, 2=key_press, 3=key_release, 4=scroll)
                event_type = action.get('event_type', 'move')
                if event_type == 'move':
                    action_type = 0
                elif event_type == 'click':
                    action_type = 1
                elif event_type == 'key_press':
                    action_type = 2
                elif event_type == 'key_release':
                    action_type = 3
                elif event_type == 'scroll':
                    action_type = 4
                else:
                    action_type = 0
                action_tensor.append(float(action_type))
                
                # Coordinates
                action_tensor.append(float(action.get('x_in_window', 0)))
                action_tensor.append(float(action.get('y_in_window', 0)))
                
                # Button (encode as: 0=none, 1=left, 2=right, 3=middle)
                button = action.get('btn', '')
                if button == 'left':
                    button_code = 1
                elif button == 'right':
                    button_code = 2
                elif button == 'middle':
                    button_code = 3
                else:
                    button_code = 0
                action_tensor.append(float(button_code))
                
                # Key (simple hash for now)
                key = action.get('key', '')
                key_code = hash(key) % 10000 if key else 0
                action_tensor.append(float(key_code))
                
                # Scroll deltas
                action_tensor.append(float(action.get('scroll_dx', 0)))
                action_tensor.append(float(action.get('scroll_dy', 0)))
            
            action_tensors.append(action_tensor)
        
        # Count non-empty tensors
        non_empty_count = sum(1 for tensor in action_tensors if len(tensor) > 1)  # > 1 because [0] is empty
        
        # T0 is already most recent (index 0), T9 is oldest (index 9)
        # No need to reverse since we sorted gamestate_timestamps in reverse order
        
        return action_tensors

    def get_actions_in_window(self, window_start_ms: int, window_end_ms: int) -> List[Dict[str, Any]]:
        actions_in_window = []
        for action in self.actions:
            ts = action.get('timestamp', 0)
            if window_start_ms <= ts <= window_end_ms:
                actions_in_window.append(action)
        actions_in_window.sort(key=lambda a: a.get('timestamp', 0))
        return actions_in_window
    
    def get_action_summary(self) -> Dict[str, int]:
        """Get summary of recorded actions."""
        return self.action_counts.copy()
    
    def get_action_tensor_for_timestep(self, timestep: int, gamestate_timestamp: int) -> List[float]:
        """
        Get action tensor for a specific timestep.
        
        Args:
            timestep: Timestep index (0-9, where 0 is most recent)
            gamestate_timestamp: Timestamp of the gamestate for this timestep
            
        Returns:
            Action tensor for this timestep
        """
        if not self.is_recording or not self.actions:
            return [0.0]  # No actions
        
        # Calculate the target timestamp for this timestep
        # T0 is most recent, T9 is oldest
        target_timestamp = gamestate_timestamp - (timestep * 600)  # 600ms intervals
        
        # Find actions in the 600ms window BEFORE this timestamp
        window_start = target_timestamp - 600
        window_end = target_timestamp
        
        # Get actions in this window
        window_actions = []
        for action in self.actions:
            action_timestamp = action.get('timestamp', 0)
            if window_start <= action_timestamp <= window_end:
                window_actions.append(action)
        
        # Sort actions by timestamp
        window_actions.sort(key=lambda a: a.get('timestamp', 0))
        
        # Convert to action tensor format: [count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
        action_tensor = [len(window_actions)]  # Start with action count
        
        for action in window_actions:
            # Timestamp (relative to window start)
            rel_timestamp = action.get('timestamp', 0) - window_start
            action_tensor.append(float(rel_timestamp))
            
            # Action type (encode as: 0=move, 1=click, 2=key_press, 3=key_release, 4=scroll)
            event_type = action.get('event_type', 'move')
            if event_type == 'move':
                action_type = 0
            elif event_type == 'click':
                action_type = 1
            elif event_type == 'key_press':
                action_type = 2
            elif event_type == 'key_release':
                action_type = 3
            elif event_type == 'scroll':
                action_type = 4
            else:
                action_type = 0
            action_tensor.append(float(action_type))
            
            # Coordinates
            action_tensor.append(float(action.get('x_in_window', 0)))
            action_tensor.append(float(action.get('y_in_window', 0)))
            
            # Button (encode as: 0=none, 1=left, 2=right, 3=middle)
            button = action.get('btn', '')
            if button == 'left':
                button_code = 1
            elif button == 'right':
                button_code = 2
            elif button == 'middle':
                button_code = 3
            else:
                button_code = 0
            action_tensor.append(float(button_code))
            
            # Key (simple hash for now)
            key = action.get('key', '')
            key_code = hash(key) % 10000 if key else 0
            action_tensor.append(float(key_code))
            
            # Scroll deltas
            action_tensor.append(float(action.get('scroll_dx', 0)))
            action_tensor.append(float(action.get('scroll_dy', 0)))
        
        return action_tensor
    
    def _find_runelite_window(self) -> bool:
        """Find the Runelite window."""
        try:
            all_windows = gw.getAllWindows()
            runelite_windows = []
            
            for window in all_windows:
                title = window.title
                if (title.startswith('Runelite - ') or 
                    title.startswith('RuneLite - ') or
                    title == 'RuneLite' or
                    title == 'Runelite' or
                    'runelite' in title.lower() or
                    'runescape' in title.lower()):
                    runelite_windows.append(window)
            
            if runelite_windows:
                self.runelite_window = runelite_windows[0]
                print(f"🎯 RUNELITE WINDOW FOUND: '{self.runelite_window.title}' at ({self.runelite_window.left}, {self.runelite_window.top}) {self.runelite_window.width}x{self.runelite_window.height}")
                LOG.info(f"Found Runelite window: {self.runelite_window.title}")
                return True
            
            print("❌ RUNELITE WINDOW NOT FOUND: No Runelite windows detected")
            LOG.warning("No Runelite window found")
            return False
            
        except Exception as e:
            LOG.error(f"Error finding Runelite window: {e}")
            return False
    
    def _check_window_focus(self) -> bool:
        """Check if Runelite window is focused."""
        if not self.runelite_window:
            return False
        
        try:
            return self.runelite_window.isActive
        except Exception:
            return False
    
    def _get_relative_coordinates(self, x: int, y: int) -> tuple:
        """Convert screen coordinates to window-relative coordinates."""
        if not self.runelite_window:
            return x, y
        
        try:
            window_x = self.runelite_window.left
            window_y = self.runelite_window.top
            relative_x = x - window_x
            relative_y = y - window_y
            return relative_x, relative_y
        except Exception:
            return x, y
    
    def _start_listeners(self):
        """Start mouse and keyboard listeners with non-blocking mode."""
        try:
            # Use non-blocking listeners to prevent interference
            self.mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
                suppress=False  # Don't suppress events
            )
            self.mouse_listener.start()
            
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
                suppress=False  # Don't suppress events
            )
            self.keyboard_listener.start()
            
            LOG.info("Input listeners started (non-blocking mode)")
        except Exception as e:
            LOG.error(f"Failed to start input listeners: {e}")
    
    def _stop_listeners(self):
        """Stop mouse and keyboard listeners."""
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
                self.mouse_listener = None
            
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
        except Exception as e:
            LOG.error(f"Failed to stop input listeners: {e}")
    
    def _on_mouse_move(self, x, y):
        """Handle mouse movement events."""
        if not self.is_recording or not self._check_window_focus():
            return
        
        current_time = time.time()
        if current_time - self.last_move_time < self.move_threshold:
            return
        
        self.last_move_time = current_time
        
        rel_x, rel_y = self._get_relative_coordinates(x, y)
        ts_abs = int(current_time * 1000)
        t_sess = ts_abs - self.current_session_start
        
        action = {
            'timestamp': t_sess,  # legacy column (session-relative)
            'timestamp_abs_ms': ts_abs,
            't_session_ms': t_sess,
            'event_type': 'move',
            'x_in_window': rel_x,
            'y_in_window': rel_y,
            'btn': '',
            'key': '',
            'scroll_dx': 0,
            'scroll_dy': 0
        }
        
        # Use thread-safe append
        with threading.Lock():
            self.actions.append(action)
            self.action_counts['mouse_movements'] += 1
            self.action_counts['total_actions'] += 1
            

    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        if not self.is_recording or not self._check_window_focus():
            return
        
        # Only record press events (not releases)
        if not pressed:
            return
        
        rel_x, rel_y = self._get_relative_coordinates(x, y)
        ts_abs = int(time.time() * 1000)
        t_sess = ts_abs - self.current_session_start
        
        button_name = str(button).split('.')[-1] if button else ''
        
        action = {
            'timestamp': t_sess,
            'timestamp_abs_ms': ts_abs,
            't_session_ms': t_sess,
            'event_type': 'click',
            'x_in_window': rel_x,
            'y_in_window': rel_y,
            'btn': button_name,
            'key': '',
            'scroll_dx': 0,
            'scroll_dy': 0
        }
        
        # Use thread-safe append
        with threading.Lock():
            self.actions.append(action)
            self.action_counts['clicks'] += 1
            self.action_counts['total_actions'] += 1
            

    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events."""
        if not self.is_recording or not self._check_window_focus():
            return
        
        rel_x, rel_y = self._get_relative_coordinates(x, y)
        ts_abs = int(time.time() * 1000)
        t_sess = ts_abs - self.current_session_start
        
        action = {
            'timestamp': t_sess,
            'timestamp_abs_ms': ts_abs,
            't_session_ms': t_sess,
            'event_type': 'scroll',
            'x_in_window': rel_x,
            'y_in_window': rel_y,
            'btn': '',
            'key': '',
            'scroll_dx': dx,
            'scroll_dy': dy
        }
        
        # Use thread-safe append
        with threading.Lock():
            self.actions.append(action)
            self.action_counts['scrolls'] += 1
            self.action_counts['total_actions'] += 1
            

    
    def _on_key_press(self, key):
        """Handle key press events."""
        if not self.is_recording or not self._check_window_focus():
            return
        
        ts_abs = int(time.time() * 1000)
        t_sess = ts_abs - self.current_session_start
        key_name = str(key).replace("'", "") if hasattr(key, 'char') and key.char else str(key)
        
        action = {
            'timestamp': t_sess,
            'timestamp_abs_ms': ts_abs,
            't_session_ms': t_sess,
            'event_type': 'key_press',
            'x_in_window': 0,
            'y_in_window': 0,
            'btn': '',
            'key': key_name,
            'scroll_dx': 0,
            'scroll_dy': 0
        }
        
        # Use thread-safe append
        with threading.Lock():
            self.actions.append(action)
            self.action_counts['key_presses'] += 1
            self.action_counts['total_actions'] += 1
            

    
    def _on_key_release(self, key):
        """Handle key release events."""
        if not self.is_recording or not self._check_window_focus():
            return
        
        ts_abs = int(time.time() * 1000)
        t_sess = ts_abs - self.current_session_start
        key_name = str(key).replace("'", "") if hasattr(key, 'char') and key.char else str(key)
        
        action = {
            'timestamp': t_sess,
            'timestamp_abs_ms': ts_abs,
            't_session_ms': t_sess,
            'event_type': 'key_release',
            'x_in_window': 0,
            'y_in_window': 0,
            'btn': '',
            'key': key_name,
            'scroll_dx': 0,
            'scroll_dy': 0
        }
        
        # Use thread-safe append
        with threading.Lock():
            self.actions.append(action)
            self.action_counts['key_releases'] += 1
            self.action_counts['total_actions'] += 1
            

    
    def save_actions(self, filepath: str = "data/actions.csv"):
        """Save recorded actions to CSV file."""
        if not self.actions:
            LOG.info("No actions to save")
            return
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'timestamp_abs_ms', 't_session_ms', 'timestamp',
                        'event_type', 'x_in_window', 'y_in_window',
                        'btn', 'key', 'scroll_dx', 'scroll_dy'
                    ],
                )
                writer.writeheader()
                writer.writerows(self.actions)
            
            LOG.info(f"Saved {len(self.actions)} actions to {filepath}")
        except Exception as e:
            LOG.error(f"Failed to save actions: {e}")
    
    def get_processed_action_data(self) -> List[Dict]:
        """
        Get processed action data in the same format as the training pipeline.
        
        Returns:
            List of action data dictionaries for each gamestate
        """
        if not self.actions:
            return []
        
        # Group actions by gamestate timestamps (if available)
        # For now, return a single action summary
        return [{
            'action_count': self.action_counts['total_actions'],
            'mouse_movements': self.action_counts['mouse_movements'],
            'clicks': self.action_counts['clicks'],
            'key_presses': self.action_counts['key_presses'],
            'key_releases': self.action_counts['key_releases'],
            'scrolls': self.action_counts['scrolls']
        }]
