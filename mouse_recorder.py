#!/usr/bin/env python3
"""
Mouse and Keyboard Recorder for OSRS Bot Training
Records mouse movements, clicks, scrolls, and keyboard events to actions.csv
Matches the exact behavior of the original gui_minimal.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import time
import threading
from pynput import mouse, keyboard
from pynput.keyboard import Key
import pygetwindow as gw
import os

class MouseRecorder:
    def __init__(self):
        self.recording = False
        self.actions = []
        self.click_queue = []  # Mouse events queue
        self.key_queue = []    # Keyboard events queue
        self.last_move_time = 0
        self.move_threshold = 0.01  # 10ms throttle for mouse moves
        self.runelite_window = None
        
        # Counters
        self.click_count = 0
        self.key_press_count = 0
        self.key_release_count = 0
        self.scroll_count = 0
        self.mouse_move_count = 0
        
        # Keyboard state tracking
        self.modifier_state = set()
        self.key_hold_times = {}
        
        # CSV file handling
        self.csvf = None
        self.csv_writer = None
        
        # Threading
        self.stop_event = threading.Event()
        self.recording_thread = None
        self.action_lock = threading.Lock()  # Thread safety for action recording

    def find_runelite_window(self):
        """Find the Runelite window"""
        try:
            all_windows = gw.getAllWindows()
            runelite_windows = []
            
            print(f"🔍 Scanning {len(all_windows)} windows for Runelite...")
            
            for window in all_windows:
                title = window.title
                try:
                    # Check different possible attribute names for visibility/active state
                    visible = getattr(window, 'visible', getattr(window, 'isVisible', 'unknown'))
                    active = getattr(window, 'isActive', getattr(window, 'active', 'unknown'))
                    print(f"  Window: '{title}' (visible: {visible}, active: {active})")
                except Exception as e:
                    print(f"  Window: '{title}' (properties unavailable: {e})")
                
                # Try multiple detection patterns
                if (title.startswith('Runelite - ') or 
                    title.startswith('RuneLite - ') or
                    title == 'RuneLite' or
                    title == 'Runelite' or
                    'runelite' in title.lower() or
                    'runescape' in title.lower()):
                    runelite_windows.append(window)
                    print(f"    ✓ MATCH: Found Runelite window!")
            
            print(f"🎯 Found {len(runelite_windows)} Runelite windows")
            
            if runelite_windows:
                # Use the first one found
                self.runelite_window = runelite_windows[0]
                print(f"🎮 Selected Runelite window: {self.runelite_window.title}")
                print(f"   Position: ({self.runelite_window.left}, {self.runelite_window.top})")
                print(f"   Size: {self.runelite_window.width} x {self.runelite_window.height}")
                return True
            
            print("⚠️ No Runelite window found. Make sure Runelite is running.")
            print("💡 Common Runelite window titles:")
            print("   - 'Runelite - username'")
            print("   - 'RuneLite - username'")
            print("   - 'RuneLite' (no username)")
            print("   - 'Runelite' (no username)")
            print("   - Any title containing 'runelite' or 'runescape'")
            return False
            
        except Exception as e:
            print(f"❌ Error finding Runelite window: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_window_focus(self):
        """Check if the detected Runelite window is focused"""
        if not self.runelite_window:
            return False
        try:
            # Check if the window is active
            active = getattr(self.runelite_window, 'isActive', getattr(self.runelite_window, 'active', False))
            return bool(active)
        except Exception:
            return False

    def get_relative_coordinates(self, x, y):
        """Convert screen coordinates to window-relative coordinates"""
        if not self.runelite_window:
            return x, y
        try:
            rel_x = x - self.runelite_window.left
            rel_y = y - self.runelite_window.top
            return rel_x, rel_y
        except Exception:
            return x, y

    def start_recording(self):
        """Start recording mouse and keyboard events"""
        if self.recording:
            return
        
        if not self.find_runelite_window():
            print("❌ Cannot start recording: Runelite window not found")
            return
        
        print("🎬 Starting recording...")
        self.recording = True
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        csv_path = "data/actions.csv"
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
                    'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
                ])
        
        # Open CSV file for writing (overwrite existing file)
        try:
            self.csvf = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csvf)
            # Write headers matching the original gui_minimal.py format
            self.csv_writer.writerow([
                'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
                'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
            ])
        except Exception as e:
            print(f"❌ Error opening CSV file: {e}")
            return
        
        # Reset counters
        self.click_count = 0
        self.key_press_count = 0
        self.key_release_count = 0
        self.scroll_count = 0
        self.mouse_move_count = 0
        
        # Start recording thread
        self.stop_event.clear()
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start listeners with non-blocking mode to prevent interference
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll,
            suppress=False  # Don't suppress events
        )
        self.mouse_listener.start()
        
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
            suppress=False  # Don't suppress events
        )
        self.keyboard_listener.start()
        
        print("✅ Recording started successfully")

    def stop_recording(self):
        """Stop recording and save actions"""
        if not self.recording:
            return
        
        print("⏹️ Stopping recording...")
        self.recording = False
        
        # Stop listeners
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        
        # Stop recording thread
        self.stop_event.set()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
        
        # Close CSV file
        if self.csvf:
            self.csvf.close()
            self.csvf = None
            self.csv_writer = None
        
        print(f"💾 Recording stopped. Saved actions to data/actions.csv")
        print(f"📊 Final counts - Clicks: {self.click_count}, Key presses: {self.key_press_count}, Key releases: {self.key_release_count}, Scrolls: {self.scroll_count}, Mouse moves: {self.mouse_move_count}")

    def on_mouse_move(self, x, y):
        """Handle mouse movement events - throttled to 10ms"""
        try:
            if not self.recording:
                return
            
            # Only record if Runelite window is focused
            if not self.check_window_focus():
                return
            
            current_time = time.time()
            # Throttle mouse move events to avoid flooding (10ms)
            if current_time - self.last_move_time > self.move_threshold:
                with self.action_lock:
                    self.mouse_move_count += 1
                    self.last_move_time = current_time
                
                # Get relative coordinates
                rel_x, rel_y = self.get_relative_coordinates(x, y)
                
                # Queue the event for processing
                event_data = {
                    'timestamp': int(time.time() * 1000),
                    'event_type': 'move',
                    'x_in_window': rel_x,
                    'y_in_window': rel_y,
                    'btn': '',
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0,
                    'modifiers': '',
                    'active_keys': ''
                }
                with self.action_lock:
                    self.click_queue.append(event_data)
                
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events - only record button presses, not releases"""
        try:
            if not self.recording:
                return
            
            # Only record if Runelite window is focused
            if not self.check_window_focus():
                return
            
            # Only record button presses, not releases
            if pressed:
                with self.action_lock:
                    self.click_count += 1
                
                btn_name = str(button).split('.')[-1]
                
                # Get relative coordinates
                rel_x, rel_y = self.get_relative_coordinates(x, y)
                
                # Safely get modifier and active keys strings
                try:
                    modifiers = self.get_modifier_string()
                except:
                    modifiers = ''
                
                try:
                    active_keys = self.get_active_keys_string()
                except:
                    active_keys = ''
                
                event_data = {
                    'timestamp': int(time.time() * 1000),
                    'event_type': 'click',
                    'x_in_window': rel_x,
                    'y_in_window': rel_y,
                    'btn': btn_name,
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0,
                    'modifiers': modifiers,
                    'active_keys': active_keys
                }
                with self.action_lock:
                    self.click_queue.append(event_data)
                
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        try:
            if not self.recording:
                return
            
            # Only record if Runelite window is focused
            if not self.check_window_focus():
                return
            
            with self.action_lock:
                self.scroll_count += 1
            
            # Get relative coordinates
            rel_x, rel_y = self.get_relative_coordinates(x, y)
            
            # Safely get modifier and active keys strings
            try:
                modifiers = self.get_modifier_string()
            except:
                modifiers = ''
            
            try:
                active_keys = self.get_active_keys_string()
            except:
                active_keys = ''
            
            event_data = {
                'timestamp': int(time.time() * 1000),
                'event_type': 'scroll',
                'x_in_window': rel_x,
                'y_in_window': rel_y,
                'btn': '',
                'key': '',
                'scroll_dx': dx,
                'scroll_dy': dy,
                'modifiers': modifiers,
                'active_keys': active_keys
            }
            with self.action_lock:
                self.click_queue.append(event_data)
            
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_key_press(self, key):
        """Handle key press events"""
        if not self.recording:
            return
        
        # Only record if Runelite window is focused
        if not self.check_window_focus():
            return
        
        try:
            # Improved key handling to capture ALL keys universally
            if hasattr(key, 'char') and key.char is not None:
                key_char = key.char
            elif hasattr(key, 'name') and key.name is not None:
                key_char = key.name
            else:
                key_char = str(key)
            
            # Ensure we always have a valid key identifier
            if not key_char or key_char == 'None':
                key_char = str(key)
            
            with self.action_lock:
                self.key_press_count += 1
            
            # Track modifier keys
            if key in [Key.shift, Key.ctrl, Key.alt, Key.cmd]:
                with self.action_lock:
                    self.modifier_state.add(str(key))
            
            # Track key hold times
            with self.action_lock:
                self.key_hold_times[key] = time.time()
            
            # Safely get modifier and active keys strings
            try:
                modifiers = self.get_modifier_string()
            except:
                modifiers = ''
            
            try:
                active_keys = self.get_active_keys_string()
            except:
                active_keys = ''
            
            event_data = {
                'timestamp': int(time.time() * 1000),
                'event_type': 'key_press',
                'x_in_window': 0,
                'y_in_window': 0,
                'btn': '',
                'key': key_char,
                'scroll_dx': 0,
                'scroll_dy': 0,
                'modifiers': modifiers,
                'active_keys': active_keys
            }
            with self.action_lock:
                self.key_queue.append(event_data)
            
        except Exception as e:
            # Log the error but continue recording
            print(f"Key press recording error: {e}, key: {key}")
            # Still record the key event with fallback
            fallback_key = str(key) if key else "unknown"
            event_data = {
                'timestamp': int(time.time() * 1000),
                'event_type': 'key_press',
                'x_in_window': 0,
                'y_in_window': 0,
                'btn': '',
                'key': fallback_key,
                'scroll_dx': 0,
                'scroll_dy': 0,
                'modifiers': '',
                'active_keys': ''
            }
            with self.action_lock:
                self.key_queue.append(event_data)

    def on_key_release(self, key):
        """Handle key release events"""
        if not self.recording:
            return
        
        # Only record if Runelite window is focused
        if not self.check_window_focus():
            return
        
        try:
            # Improved key handling to capture ALL keys universally
            if hasattr(key, 'char') and key.char is not None:
                key_char = key.char
            elif hasattr(key, 'name') and key.name is not None:
                key_char = key.name
            else:
                key_char = str(key)
            
            # Ensure we always have a valid key identifier
            if not key_char or key_char == 'None':
                key_char = str(key)
            
            with self.action_lock:
                self.key_release_count += 1
            
            # Remove modifier keys
            with self.action_lock:
                if str(key) in self.modifier_state:
                    self.modifier_state.discard(str(key))
            
            # Remove from hold times
            with self.action_lock:
                if key in self.key_hold_times:
                    del self.key_hold_times[key]
            
            # Safely get modifier and active keys strings
            try:
                modifiers = self.get_modifier_string()
            except:
                modifiers = ''
            
            try:
                active_keys = self.get_active_keys_string()
            except:
                active_keys = ''
            
            event_data = {
                'timestamp': int(time.time() * 1000),
                'event_type': 'key_release',
                'x_in_window': 0,
                'y_in_window': 0,
                'btn': '',
                'key': key_char,
                'scroll_dx': 0,
                'scroll_dy': 0,
                'modifiers': modifiers,
                'active_keys': active_keys
            }
            with self.action_lock:
                self.key_queue.append(event_data)
            
        except Exception as e:
            # Log the error but continue recording
            print(f"Key release recording error: {e}, key: {key}")
            # Still record the key event with fallback
            fallback_key = str(key) if key else "unknown"
            event_data = {
                'timestamp': int(time.time() * 1000),
                'event_type': 'key_release',
                'x_in_window': 0,
                'y_in_window': 0,
                'btn': '',
                'key': fallback_key,
                'scroll_dx': 0,
                'scroll_dy': 0,
                'modifiers': '',
                'active_keys': ''
            }
            with self.action_lock:
                self.key_queue.append(event_data)

    def get_modifier_string(self):
        """Get string representation of currently pressed modifier keys"""
        modifiers = []
        if Key.shift in self.modifier_state or 'shift' in self.modifier_state:
            modifiers.append('shift')
        if Key.ctrl in self.modifier_state or 'ctrl' in self.modifier_state:
            modifiers.append('ctrl')
        if Key.alt in self.modifier_state or 'alt' in self.modifier_state:
            modifiers.append('alt')
        if Key.cmd in self.modifier_state or 'cmd' in self.modifier_state:
            modifiers.append('cmd')
        return '+'.join(modifiers)

    def get_active_keys_string(self):
        """Get string representation of currently held keys"""
        if not self.key_hold_times:
            return ''
        
        # Get keys that have been held for more than 100ms
        current_time = time.time()
        active_keys = []
        for key, press_time in self.key_hold_times.items():
            if current_time - press_time > 0.1:
                if hasattr(key, 'char') and key.char is not None:
                    active_keys.append(key.char)
                elif hasattr(key, 'name') and key.name is not None:
                    active_keys.append(key.name)
                else:
                    active_keys.append(str(key))
        
        return '+'.join(active_keys)

    def recording_loop(self):
        """Main recording loop that processes queued events and writes to CSV"""
        while not self.stop_event.is_set():
            try:
                # Process mouse events
                with self.action_lock:
                    while self.click_queue and not self.stop_event.is_set():
                        event = self.click_queue.pop(0)
                        if self.csv_writer and self.csvf:
                            self.csv_writer.writerow([
                                event['timestamp'],
                                event['event_type'],
                                event['x_in_window'],
                                event['y_in_window'],
                                event['btn'],
                                event['key'],
                                event['scroll_dx'],
                                event['scroll_dy'],
                                event['modifiers'],
                                event['active_keys']
                            ])
                            self.csvf.flush()  # Ensure data is written immediately
                
                # Process keyboard events
                with self.action_lock:
                    while self.key_queue and not self.stop_event.is_set():
                        event = self.key_queue.pop(0)
                        if self.csv_writer and self.csvf:
                            self.csv_writer.writerow([
                                event['timestamp'],
                                event['event_type'],
                                event['x_in_window'],
                                event['y_in_window'],
                                event['btn'],
                                event['key'],
                                event['scroll_dx'],
                                event['scroll_dy'],
                                event['modifiers'],
                                event['active_keys']
                            ])
                            self.csvf.flush()  # Ensure data is written immediately
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                time.sleep(0.1)

    def get_action_type_counts(self):
        """Get current action type counts"""
        with self.action_lock:
            return {
                'move': self.mouse_move_count,
                'click': self.click_count,
                'scroll': self.scroll_count,
                'key_press': self.key_press_count,
                'key_release': self.key_release_count
            }

class RecorderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OSRS Mouse & Keyboard Recorder")
        self.root.geometry("600x600")
        self.root.resizable(True, True)
        
        self.recorder = MouseRecorder()
        self._setup_ui()
        
    def _setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="🎮 OSRS Action Recorder", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Text(self.root, height=6, width=70, wrap=tk.WORD)
        instructions.pack(pady=10, padx=10)
        instructions.insert(tk.END, """📋 Instructions:
1. Click "🔍 Detect Runelite Window" to find your Runelite client
2. Make sure Runelite is focused/active
3. Click "▶ Start Recording" to begin capturing actions
4. Perform actions in Runelite (move mouse, click, type, scroll)
5. Click "⏹ Stop Recording" when done
6. Actions are saved to data/actions.csv""")
        instructions.config(state=tk.DISABLED)
        
        # Runelite Window Detection
        window_frame = tk.LabelFrame(self.root, text="Runelite Window Detection")
        window_frame.pack(fill="x", padx=10, pady=5)
        
        self.window_status_label = tk.Label(window_frame, text="❓ Window not detected", fg="red")
        self.window_status_label.pack(pady=5)
        
        detect_button = tk.Button(window_frame, text="🔍 Detect Runelite Window", 
                                command=self.detect_window, bg="lightblue")
        detect_button.pack(pady=5)
        
        test_button = tk.Button(window_frame, text="🧪 Test Window Detection", 
                              command=self.test_detection, bg="lightyellow")
        test_button.pack(pady=5)
        
        # Recording Controls
        control_frame = tk.LabelFrame(self.root, text="Recording Controls")
        control_frame.pack(fill="x", padx=10, pady=5)
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="▶ Start Recording", 
                                    command=self.start_recording, bg="lightgreen", font=("Arial", 12))
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop Recording", 
                                   command=self.stop_recording, bg="lightcoral", font=("Arial", 12))
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Action Counts
        counts_frame = tk.LabelFrame(self.root, text="Action Counts (This Session)")
        counts_frame.pack(fill="x", padx=10, pady=5)
        
        self.move_count_label = tk.Label(counts_frame, text="Mouse Moves: 0")
        self.move_count_label.pack(anchor=tk.W)
        
        self.click_count_label = tk.Label(counts_frame, text="Clicks: 0")
        self.click_count_label.pack(anchor=tk.W)
        
        self.scroll_count_label = tk.Label(counts_frame, text="Scrolls: 0")
        self.scroll_count_label.pack(anchor=tk.W)
        
        self.key_press_label = tk.Label(counts_frame, text="Key Presses: 0")
        self.key_press_label.pack(anchor=tk.W)
        
        self.key_release_label = tk.Label(counts_frame, text="Key Releases: 0")
        self.key_release_label.pack(anchor=tk.W)
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready to detect window", fg="blue")
        self.status_label.pack(pady=10)
        
        # Update counts periodically
        self.root.after(100, self.update_status)
        
    def detect_window(self):
        """Detect Runelite window"""
        if self.recorder.find_runelite_window():
            self.window_status_label.config(text="✅ Runelite window detected", fg="green")
            self.status_label.config(text="Runelite window found! You can now start recording.", fg="green")
        else:
            self.window_status_label.config(text="❌ Runelite window not found", fg="red")
            self.status_label.config(text="Runelite window not found. Make sure it's running.", fg="red")
    
    def test_detection(self):
        """Test window detection with detailed output"""
        print("\n🧪 Testing window detection...")
        self.recorder.find_runelite_window()
    
    def start_recording(self):
        """Start recording"""
        if not self.recorder.runelite_window:
            messagebox.showerror("Error", "Please detect Runelite window first!")
            return
        
        self.recorder.start_recording()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Recording... Perform actions in Runelite", fg="red")
    
    def stop_recording(self):
        """Stop recording"""
        self.recorder.stop_recording()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Recording stopped. Actions saved to data/actions.csv", fg="blue")
    
    def update_status(self):
        """Update action count displays"""
        counts = self.recorder.get_action_type_counts()
        
        self.move_count_label.config(text=f"Mouse Moves: {counts['move']}")
        self.click_count_label.config(text=f"Clicks: {counts['click']}")
        self.scroll_count_label.config(text=f"Scrolls: {counts['scroll']}")
        self.key_press_label.config(text=f"Key Presses: {counts['key_press']}")
        self.key_release_label.config(text=f"Key Releases: {counts['key_release']}")
        
        # Schedule next update
        self.root.after(100, self.update_status)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = RecorderGUI()
    app.run()
