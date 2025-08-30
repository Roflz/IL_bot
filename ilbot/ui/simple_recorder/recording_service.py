"""Recording service for capturing user input and writing to CSV."""
import csv
import time
import threading
from datetime import datetime
from pynput import mouse, keyboard


class RecordingService:
    def __init__(self, window_info):
        self.window_info = window_info
        self.actions_file = None
        self.recording = False
        self.paused = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Mouse movement rate limiting (10ms minimum)
        self.last_mouse_move_time = 0
        self.mouse_move_interval = 0.01  # 10ms in seconds
        
        # Track last recorded position to avoid duplicates
        self.last_recorded_x = None
        self.last_recorded_y = None
        
        # Input listeners
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        
        # Thread safety
        self.lock = threading.Lock()
        
    def start_recording(self, actions_file):
        """Start recording input to the specified file."""
        self.actions_file = actions_file
        
        # Create CSV file and write header
        self.csv_file = open(actions_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
            'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
        ])
        
        # Start input listeners
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll
        )
        
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        self.recording = True
        self.paused = False
        
    def pause_recording(self):
        """Pause recording."""
        self.paused = True
        
    def resume_recording(self):
        """Resume recording."""
        self.paused = False
        
    def stop_recording(self):
        """Stop recording and cleanup."""
        self.recording = False
        
        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
            
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            
    def _write_action(self, event_type, x, y, btn="", key="", scroll_dx=0, scroll_dy=0, modifiers="", active_keys=""):
        """Write an action to the CSV file."""
        if not self.recording or self.paused or not self.csv_writer:
            return
            
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        
        with self.lock:
            self.csv_writer.writerow([
                timestamp, event_type, x, y, btn, key, scroll_dx, scroll_dy, modifiers, active_keys
            ])
            self.csv_file.flush()  # Ensure data is written immediately
            
    def _on_mouse_move(self, x, y):
        """Handle mouse movement events."""
        if not self.recording or self.paused:
            return
            
        # Convert to window-relative coordinates
        rel_x = x - self.window_info['x']
        rel_y = y - self.window_info['y']
        
        # Store last position for key events
        self.last_mouse_x = rel_x
        self.last_mouse_y = rel_y
        
        # Rate limiting for mouse movement (10ms minimum interval)
        current_time = time.time()
        if current_time - self.last_mouse_move_time < self.mouse_move_interval:
            return
        
        # Only record if within window bounds
        if (0 <= rel_x <= self.window_info['width'] and 
            0 <= rel_y <= self.window_info['height']):
            
            # Only record if position has actually changed
            if (self.last_recorded_x != rel_x or self.last_recorded_y != rel_y):
                self._write_action('move', rel_x, rel_y)
                self.last_mouse_move_time = current_time
                self.last_recorded_x = rel_x
                self.last_recorded_y = rel_y
            
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        if not self.recording or self.paused:
            return
            
        # Convert to window-relative coordinates
        rel_x = x - self.window_info['x']
        rel_y = y - self.window_info['x']
        
        # Store last position for key events
        self.last_mouse_x = rel_x
        self.last_mouse_y = rel_y
        
        # Only record if within window bounds
        if (0 <= rel_x <= self.window_info['width'] and 
            0 <= rel_y <= self.window_info['height']):
            
            # Convert button to string
            btn_str = str(button).replace('Button.', '')
            
            self._write_action('click', rel_x, rel_y, btn=btn_str)
            
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events."""
        if not self.recording or self.paused:
            return
            
        # Convert to window-relative coordinates
        rel_x = x - self.window_info['x']
        rel_y = y - self.window_info['y']
        
        # Store last position for key events
        self.last_mouse_x = rel_x
        self.last_mouse_y = rel_y
        
        # Only record if within window bounds
        if (0 <= rel_x <= self.window_info['width'] and 
            0 <= rel_y <= self.window_info['height']):
            
            self._write_action('scroll', rel_x, rel_y, scroll_dx=dx, scroll_dy=dy)
            
    def _on_key_press(self, key):
        """Handle key press events."""
        if not self.recording or self.paused:
            return
            
        # Convert key to string
        if hasattr(key, 'char'):
            key_str = key.char
        else:
            key_str = str(key).replace('Key.', '')
        print(key_str)
        # Use last known mouse position
        self._write_action('key_press', self.last_mouse_x, self.last_mouse_y, key=key_str)
        
    def _on_key_release(self, key):
        """Handle key release events."""
        if not self.recording or self.paused:
            return

        # Convert key to string
        if hasattr(key, 'char'):
            key_str = key.char
        else:
            key_str = str(key).replace('Key.', '')
        
        # Use last known mouse position
        self._write_action('key_release', self.last_mouse_x, self.last_mouse_y, key=key_str)
