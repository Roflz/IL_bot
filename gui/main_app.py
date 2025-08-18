import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
import csv
import threading
import time
from datetime import datetime
from mss import mss, tools
from pynput import mouse, keyboard
from pynput.keyboard import Key
from PIL import Image
import numpy as np
import pyautogui
import tkinter.scrolledtext as scrolledtext
import torch
import traceback
try:
    # Try relative imports first (when running from bot_runelite_IL directory)
    from model.data_loader import ILSequenceDataset
    from model.model import ImitationHybridModel
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
except ImportError:
    # Fall back to absolute imports (when running from root directory)
    from bot_runelite_IL.model.data_loader import ILSequenceDataset
    from bot_runelite_IL.model.model import ImitationHybridModel
    from bot_runelite_IL.data_collection import DataCollector
    from bot_runelite_IL.preprocessing import DataPreprocessor

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import tkinter.filedialog as fd
import win32gui
import win32process
import psutil

# == CONFIG ==
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
SCREENSHOT_DIR = os.path.join(DATA_DIR, "runelite_screenshots")
ACTION_LOG = os.path.join(DATA_DIR, "actions.csv")
GAME_STATE_PATH = os.path.join(DATA_DIR, 'runelite_gamestate.json')
TREE_POS = (800, 400)
BANK_POS = (1000, 200)
INVENTORY_SIZE = 28

# --- Begin OsrsAutomationApp (formerly XPRecorder) ---

class OsrsAutomationApp:
    def __init__(self, root):
        self.root = root
        root.title("OSRS Automation Recorder")
        root.geometry("800x500")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Window focus configuration
        self.target_window_titles = ['RuneLite', 'Old School RuneScape', 'OSRS']
        self.selected_window_hwnd = None  # Store selected window handle
        self.available_windows = []  # List of available windows

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Main tab for input recording
        self.main_frame = tk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Input Recorder")

        # Interval slider
        slider_frame = tk.Frame(self.main_frame)
        slider_frame.pack(pady=5)
        tk.Label(slider_frame, text="Capture interval (s):").pack(side="left")
        self.interval_var = tk.DoubleVar(value=0.6)
        self.interval_slider = tk.Scale(
            slider_frame,
            variable=self.interval_var,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient="horizontal",
            length=200,
        )
        self.interval_slider.pack(side="left", padx=5)

        # Window focus configuration frame
        focus_frame = tk.LabelFrame(self.main_frame, text="Window Focus Configuration")
        focus_frame.pack(fill="x", padx=8, pady=4)
        
        # Window selection
        window_select_frame = tk.Frame(focus_frame)
        window_select_frame.pack(fill="x", padx=4, pady=2)
        tk.Label(window_select_frame, text="Target Window:").pack(side="left")
        self.window_var = tk.StringVar(value="Auto-detect OSRS")
        self.window_dropdown = ttk.Combobox(window_select_frame, textvariable=self.window_var, width=40, state="readonly")
        self.window_dropdown.pack(side="left", padx=4)
        self.window_dropdown.bind('<<ComboboxSelected>>', self.on_window_selection_change)
        self.refresh_windows_button = tk.Button(window_select_frame, text="Refresh", command=self.refresh_available_windows)
        self.refresh_windows_button.pack(side="left", padx=4)
        
        # Focus status details
        focus_details_frame = tk.Frame(focus_frame)
        focus_details_frame.pack(fill="x", padx=4, pady=2)
        self.focus_status_label = tk.Label(focus_details_frame, text="OSRS Focus: No", fg="red")
        self.focus_status_label.pack(side="left", padx=4)
        self.current_window_label = tk.Label(focus_details_frame, text="Current: None", fg="gray")
        self.current_window_label.pack(side="left", padx=4)
        self.target_window_label = tk.Label(focus_details_frame, text="Target: Auto-detect", fg="blue")
        self.target_window_label.pack(side="left", padx=4)
        self.refresh_focus_button = tk.Button(focus_details_frame, text="Refresh", command=self.refresh_focus_status)
        self.refresh_focus_button.pack(side="left", padx=4)
        
        # Debug button for window focus
        self.debug_focus_button = tk.Button(focus_frame, text="Debug Focus", command=self.debug_window_focus)
        self.debug_focus_button.pack(pady=2)
        
        # Find RuneLite windows button
        self.find_runelite_button = tk.Button(focus_frame, text="Find RuneLite Windows", command=self.find_and_show_runelite)
        self.find_runelite_button.pack(pady=2)
        
        # Test window detection button
        self.test_detection_button = tk.Button(focus_frame, text="Test Detection", command=self.test_window_detection)
        self.test_detection_button.pack(pady=2)
        
        # Toggle for focus checking
        self.focus_checking_enabled = tk.BooleanVar(value=True)
        self.focus_checking_checkbox = tk.Checkbutton(focus_frame, text="Enable Focus Checking", variable=self.focus_checking_enabled)
        self.focus_checking_checkbox.pack(pady=2)
        
        # Bind GUI interaction events to temporarily pause focus checking
        self.root.bind('<Button-1>', self.on_gui_interaction)
        self.root.bind('<Key>', self.on_gui_interaction)
        self.root.bind('<FocusIn>', self.on_gui_focus_in)
        self.root.bind('<FocusOut>', self.on_gui_focus_out)
        
        # Focus checking pause timer
        self.focus_checking_paused = False
        self.focus_pause_timer = None

        # Recording control
        self.recording = False
        self.record_button = tk.Button(self.main_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=5)

        # Initialize recording system variables
        self.click_queue = []
        self.key_queue = []
        self.stop_event = threading.Event()
        self.thread = None
        self.csvf = None
        self.csv_writer = None
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Mouse and keyboard tracking
        self.last_mouse_move_time_ns = 0
        self.modifier_state = set()
        self.key_hold_times = {}
        self.active_keys = set()
        
        # Constants for file paths
        self.DATA_DIR = os.path.abspath(os.path.join('bot_runelite_IL', 'data'))
        self.ACTION_LOG = os.path.abspath(os.path.join('bot_runelite_IL', 'data', 'actions.csv'))

        # Status counters frame
        status_frame = tk.Frame(self.main_frame)
        status_frame.pack(fill="x", pady=5)
        self.click_count = 0
        self.key_press_count = 0
        self.key_release_count = 0
        self.mouse_move_count = 0
        self.scroll_count = 0
        self.click_label = tk.Label(status_frame, text="Clicks: 0")
        self.key_press_label = tk.Label(status_frame, text="Key Presses: 0")
        self.key_release_label = tk.Label(status_frame, text="Key Releases: 0")
        self.mouse_move_label = tk.Label(status_frame, text="Mouse Moves: 0")
        self.scroll_label = tk.Label(status_frame, text="Scrolls: 0")
        for lbl in (self.click_label, self.key_press_label, self.key_release_label, self.mouse_move_label, self.scroll_label):
            lbl.pack(side="left", padx=5)

        # Add Bot Control tab
        self.bot_control_frame = tk.Frame(self.notebook)
        self.notebook.add(self.bot_control_frame, text="Bot Control")
        self.bot_status_label = tk.Label(self.bot_control_frame, text="Bot status: Idle")
        self.bot_status_label.pack(pady=5)
        self.start_bot_button = tk.Button(self.bot_control_frame, text="Start Bot", command=self.start_bot)
        self.start_bot_button.pack(side="left", padx=10, pady=5)
        self.stop_bot_button = tk.Button(self.bot_control_frame, text="Stop Bot", command=self.stop_bot, state="disabled")
        self.stop_bot_button.pack(side="left", padx=10, pady=5)

        # Add Debugging tab
        self.debug_frame = tk.Frame(self.notebook)
        self.notebook.add(self.debug_frame, text="Debugging")
        self.debug_text = scrolledtext.ScrolledText(self.debug_frame, height=10)
        self.debug_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.debug_button = tk.Button(self.debug_frame, text="Run Debug", command=self.run_debug)
        self.debug_button.pack(pady=5)

        # Add Model Training tab
        self.training_frame = tk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        self.train_status_label = tk.Label(self.training_frame, text="Training status: Idle")
        self.train_status_label.pack(pady=5)
        self.train_button = tk.Button(self.training_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side="left", padx=10, pady=5)
        self.stop_train_button = tk.Button(self.training_frame, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_train_button.pack(side="left", padx=10, pady=5)

        # Add Data Preprocessing tab
        self.preprocessing_frame = tk.Frame(self.notebook)
        self.notebook.add(self.preprocessing_frame, text="Data Preprocessing")
        input_frame = tk.LabelFrame(self.preprocessing_frame, text="Input Paths")
        input_frame.pack(fill="x", padx=8, pady=4)
        self.gamestate_dir_var = tk.StringVar(value=os.path.abspath(os.path.join('bot_runelite_IL', 'data', 'gamestates')))
        self.actions_csv_var = tk.StringVar(value=os.path.abspath(os.path.join('bot_runelite_IL', 'data', 'actions.csv')))
        tk.Label(input_frame, text="Gamestate Dir:").pack(side="left", padx=4)
        tk.Entry(input_frame, textvariable=self.gamestate_dir_var, width=40).pack(side="left", padx=4)
        tk.Label(input_frame, text="Actions CSV:").pack(side="left", padx=4)
        tk.Entry(input_frame, textvariable=self.actions_csv_var, width=40).pack(side="left", padx=4)
        self.preprocess_button = tk.Button(self.preprocessing_frame, text="Run Preprocessing", command=self.run_preprocessing)
        self.preprocess_button.pack(pady=8)
        self.preprocessing_progress = ttk.Progressbar(self.preprocessing_frame, orient="horizontal", length=400, mode="determinate")
        self.preprocessing_progress.pack(pady=4)

        # Add Data Inspector tab
        self.data_inspector_frame = tk.Frame(self.notebook)
        self.notebook.add(self.data_inspector_frame, text="Data Inspector")
        self.data_inspector_text = scrolledtext.ScrolledText(self.data_inspector_frame, height=20)
        self.data_inspector_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.load_data_button = tk.Button(self.data_inspector_frame, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=5)

        # Add Feature Inspector tab
        self.feature_inspector_frame = tk.Frame(self.notebook)
        self.notebook.add(self.feature_inspector_frame, text="Feature Inspector")
        self.feature_inspector_frame.rowconfigure(1, weight=1)
        self.feature_inspector_frame.columnconfigure(0, weight=1)

        # Top controls: directory selection and navigation
        feature_controls = tk.Frame(self.feature_inspector_frame)
        feature_controls.grid(row=0, column=0, sticky='ew', padx=8, pady=4)
        self.feature_data_dir_var = tk.StringVar(value=os.path.abspath(os.path.join('bot_runelite_IL', 'data', 'aligned')))
        tk.Label(feature_controls, text="Aligned Data Dir:").pack(side="left", padx=4)
        tk.Entry(feature_controls, textvariable=self.feature_data_dir_var, width=40).pack(side="left", padx=4)
        tk.Button(feature_controls, text="Browse", command=self.browse_feature_data_dir).pack(side="left", padx=4)
        self.load_features_button = tk.Button(feature_controls, text="Load Features", command=self.load_feature_data)
        self.load_features_button.pack(side="left", padx=4)

        # Example navigation
        nav_frame = tk.Frame(self.feature_inspector_frame)
        nav_frame.grid(row=1, column=0, sticky='ew', padx=8, pady=4)
        self.feature_example_idx_var = tk.IntVar(value=0)
        self.feature_example_label = tk.Label(nav_frame, text="Example: 0")
        self.feature_example_label.pack(side="left", padx=4)
        self.prev_feature_btn = tk.Button(nav_frame, text="Previous", command=lambda: self.show_feature_example(-1))
        self.prev_feature_btn.pack(side="left", padx=4)
        self.next_feature_btn = tk.Button(nav_frame, text="Next", command=lambda: self.show_feature_example(1))
        self.next_feature_btn.pack(side="left", padx=4)

        # Feature/state/action display
        display_frame = tk.Frame(self.feature_inspector_frame)
        display_frame.grid(row=2, column=0, sticky='nsew', padx=8, pady=4)
        self.feature_state_text = scrolledtext.ScrolledText(display_frame, height=8, width=40)
        self.feature_state_text.grid(row=0, column=0, padx=4, pady=4)
        self.feature_action_text = scrolledtext.ScrolledText(display_frame, height=8, width=40)
        self.feature_action_text.grid(row=0, column=1, padx=4, pady=4)
        self.state_features_text = scrolledtext.ScrolledText(display_frame, height=8, width=40)
        self.state_features_text.grid(row=1, column=0, padx=4, pady=4)
        self.action_features_text = scrolledtext.ScrolledText(display_frame, height=8, width=40)
        self.action_features_text.grid(row=1, column=1, padx=4, pady=4)

        # Screenshot display (optional, if used)
        self.feature_screenshot_label = tk.Label(display_frame)
        self.feature_screenshot_label.grid(row=2, column=0, columnspan=2, pady=4)

        # Placeholders for feature data
        self.feature_states = []
        self.feature_actions = []
        self.state_features = None
        self.action_features = None

        # Initialize window list
        self.refresh_available_windows()

    def refresh_available_windows(self):
        """Refresh the list of available windows in the dropdown"""
        try:
            # Use win32gui to enumerate all windows
            window_handles = []
            win32gui.EnumWindows(lambda hwnd, l: l.append(hwnd), window_handles)
            
            # Filter for visible windows with titles
            visible_windows = []
            for hwnd in window_handles:
                try:
                    # Check if window is visible
                    if not win32gui.IsWindowVisible(hwnd):
                        continue
                    
                    # Get window title
                    title = win32gui.GetWindowText(hwnd)
                    if not title or title.strip() == "":
                        continue
                    
                    # Skip system windows (very short titles or system classes)
                    if len(title) < 3:
                        continue
                        
                    # Get window class to filter out system windows
                    class_name = win32gui.GetClassName(hwnd)
                    if class_name in ['Shell_TrayWnd', 'Shell_SecondaryTrayWnd', 'DV2ControlHost']:
                        continue
                    
                    visible_windows.append((hwnd, title))
                except:
                    continue  # Skip windows that cause errors
            
            # Sort windows by title for better usability
            visible_windows.sort(key=lambda x: x[1].lower())
            
            # Store handles and titles separately
            self.available_windows = [hwnd for hwnd, title in visible_windows]
            window_titles = [title for hwnd, title in visible_windows]
            
            # Update dropdown
            self.window_dropdown['values'] = ['Auto-detect OSRS'] + window_titles
            self.window_dropdown.current(0)  # Select "Auto-detect OSRS" by default
            self.selected_window_hwnd = None  # Reset selected window handle
            self.update_focus_status()  # Update focus status based on new default
            
            print(f"Found {len(window_titles)} visible windows")
            
        except Exception as e:
            print(f"Error refreshing windows: {e}")
            # Fallback to empty list
            self.available_windows = []
            self.window_dropdown['values'] = ['Auto-detect OSRS']
            self.window_dropdown.current(0)

    def find_runelite_windows(self):
        """Find all RuneLite/OSRS windows specifically"""
        try:
            runelite_windows = []
            window_handles = []
            win32gui.EnumWindows(lambda hwnd, l: l.append(hwnd), window_handles)
            
            for hwnd in window_handles:
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        continue
                    
                    title = win32gui.GetWindowText(hwnd)
                    if not title:
                        continue
                    
                    # Check for RuneLite/OSRS windows
                    title_lower = title.lower()
                    if any(keyword in title_lower for keyword in ['runelite', 'old school runescape', 'osrs', 'jagex']):
                        runelite_windows.append((hwnd, title))
                        
                except:
                    continue
            
            return runelite_windows
            
        except Exception as e:
            print(f"Error finding RuneLite windows: {e}")
            return []

    def is_osrs_window_focused(self):
        """Check if the target window currently has focus"""
        try:
            # Check if focus checking is enabled
            if not self.focus_checking_enabled.get():
                return False
                
            # Check if focus checking is temporarily paused
            if self.focus_checking_paused:
                return False
                
            # Get the currently focused window
            focused_hwnd = win32gui.GetForegroundWindow()
            if focused_hwnd == 0:
                return False
                
            # Don't check focus if our own GUI window is focused
            if focused_hwnd == self.root.winfo_id():
                return False
                
            # Get window title
            window_title = win32gui.GetWindowText(focused_hwnd)
            
            # If a specific window is selected, check if that window has focus
            if self.selected_window_hwnd is not None:
                # Verify the selected window still exists
                try:
                    test_title = win32gui.GetWindowText(self.selected_window_hwnd)
                    if not test_title:  # Window no longer exists
                        self.selected_window_hwnd = None
                        self.window_var.set("Auto-detect OSRS")
                        return self.is_osrs_window_focused()  # Recursive call with auto-detect
                    return focused_hwnd == self.selected_window_hwnd
                except:
                    # Window handle is invalid, reset to auto-detect
                    self.selected_window_hwnd = None
                    self.window_var.set("Auto-detect OSRS")
                    return self.is_osrs_window_focused()  # Recursive call with auto-detect
            
            # Otherwise, use auto-detect with OSRS titles
            if not window_title:  # Empty title, skip
                return False
                
            # More comprehensive OSRS detection
            title_lower = window_title.lower()
            osrs_keywords = [
                'runelite', 'old school runescape', 'osrs', 'jagex',
                'runescape', 'rs3', 'rs2', '2007scape'
            ]
            return any(keyword in title_lower for keyword in osrs_keywords)
            
        except Exception as e:
            return False

    def refresh_focus_status(self):
        """Manually refresh the focus status display"""
        self.update_focus_status()
        self.root.update_idletasks()

    def get_current_window_info(self):
        """Get information about the currently focused window"""
        try:
            focused_hwnd = win32gui.GetForegroundWindow()
            if focused_hwnd == 0:
                return None, "No window focused"
                
            window_title = win32gui.GetWindowText(focused_hwnd)
            return focused_hwnd, window_title
        except Exception as e:
            return None, "Error"

    def get_window_details(self, hwnd):
        """Get detailed information about a specific window"""
        try:
            if hwnd == 0:
                return "Invalid window handle"
            
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            size = (rect[2] - rect[0], rect[3] - rect[1])
            
            return {
                'title': title,
                'class': class_name,
                'size': size,
                'rect': rect,
                'hwnd': hwnd
            }
        except Exception as e:
            return f"Error getting window details: {e}"

    def find_and_show_runelite(self):
        """Find and display all RuneLite/OSRS windows"""
        try:
            runelite_windows = self.find_runelite_windows()
            
            if not runelite_windows:
                messagebox.showinfo("RuneLite Windows", "No RuneLite/OSRS windows found.\n\nMake sure RuneLite is running and visible.")
                return
            
            # Create detailed info
            info = f"Found {len(runelite_windows)} RuneLite/OSRS window(s):\n\n"
            for i, (hwnd, title) in enumerate(runelite_windows, 1):
                try:
                    class_name = win32gui.GetClassName(hwnd)
                    rect = win32gui.GetWindowRect(hwnd)
                    size = (rect[2] - rect[0], rect[3] - rect[1])
                    info += f"{i}. Title: {title}\n"
                    info += f"   Class: {class_name}\n"
                    info += f"   Size: {size[0]}x{size[1]}\n"
                    info += f"   Handle: {hwnd}\n\n"
                except:
                    info += f"{i}. Title: {title}\n   (Error getting details)\n\n"
            
            # Add to debug tab if available
            if hasattr(self, 'debug_text'):
                self.debug_text.insert(tk.END, "=== RuneLite Windows Found ===\n" + info + "\n")
                self.debug_text.see(tk.END)
            
            print(info)
            messagebox.showinfo("RuneLite Windows Found", info)
            
        except Exception as e:
            print(f"Error finding RuneLite windows: {e}")
            messagebox.showerror("Error", f"Error finding RuneLite windows: {e}")

    def debug_window_focus(self):
        """Debug method to show detailed window focus information"""
        try:
            current_hwnd, current_title = self.get_current_window_info()
            target_hwnd = self.selected_window_hwnd
            
            debug_info = f"=== Window Focus Debug ===\n"
            debug_info += f"Current focused window:\n"
            if current_hwnd:
                current_details = self.get_window_details(current_hwnd)
                if isinstance(current_details, dict):
                    debug_info += f"  Title: {current_details['title']}\n"
                    debug_info += f"  Class: {current_details['class']}\n"
                    debug_info += f"  Size: {current_details['size']}\n"
                    debug_info += f"  Handle: {current_details['hwnd']}\n"
                else:
                    debug_info += f"  {current_details}\n"
            else:
                debug_info += f"  {current_title}\n"
            
            debug_info += f"\nTarget window:\n"
            if target_hwnd:
                target_details = self.get_window_details(target_hwnd)
                if isinstance(target_details, dict):
                    debug_info += f"  Title: {target_details['title']}\n"
                    debug_info += f"  Class: {target_details['class']}\n"
                    debug_info += f"  Size: {target_details['size']}\n"
                    debug_info += f"  Handle: {target_details['hwnd']}\n"
                else:
                    debug_info += f"  {target_details}\n"
            else:
                debug_info += f"  Auto-detect OSRS (keywords: {', '.join(['runelite', 'old school runescape', 'osrs', 'jagex', 'runescape', 'rs3', 'rs2', '2007scape'])})\n"
            
            # Check if current window matches OSRS keywords
            if current_title and current_title != "No window focused" and current_title != "Error":
                title_lower = current_title.lower()
                osrs_keywords = ['runelite', 'old school runescape', 'osrs', 'jagex', 'runescape', 'rs3', 'rs2', '2007scape']
                matches = [kw for kw in osrs_keywords if kw in title_lower]
                if matches:
                    debug_info += f"\nOSRS Detection: Current window matches keywords: {', '.join(matches)}\n"
                else:
                    debug_info += f"\nOSRS Detection: Current window does NOT match any OSRS keywords\n"
            
            debug_info += f"\nFocus match: {current_hwnd == target_hwnd if target_hwnd else 'Auto-detect'}\n"
            
            # Add to debug tab if available
            if hasattr(self, 'debug_text'):
                self.debug_text.insert(tk.END, debug_info + "\n")
                self.debug_text.see(tk.END)
            
            print(debug_info)
            
        except Exception as e:
            print(f"Error in debug_window_focus: {e}")

    def on_window_selection_change(self, event=None):
        """Handle window selection change in dropdown"""
        selected = self.window_var.get()
        
        if selected == "Auto-detect OSRS":
            self.selected_window_hwnd = None
            self.target_window_label.config(text="Target: Auto-detect", fg="blue")
        else:
            # Find the selected window and store its handle
            for hwnd in self.available_windows:
                if win32gui.GetWindowText(hwnd) == selected:
                    self.selected_window_hwnd = hwnd
                    self.target_window_label.config(text=f"Target: {selected[:30]}...", fg="green")
                    break
        
        self.update_focus_status()

    def update_focus_status(self):
        """Update the enhanced focus status indicators"""
        try:
            # Check if focus checking is disabled
            if not self.focus_checking_enabled.get():
                self.focus_status_label.config(text="Target Focus: Disabled", fg="gray")
                self.current_window_label.config(text="Current: Focus checking disabled", fg="gray")
                return
            
            # Check if focus checking is temporarily paused
            if self.focus_checking_paused:
                self.focus_status_label.config(text="Target Focus: Paused", fg="orange")
                self.current_window_label.config(text="Current: Focus checking paused", fg="orange")
                return
            
            # Get current window info
            current_hwnd, current_title = self.get_current_window_info()
            
            # Check if our own GUI is focused
            if current_hwnd == self.root.winfo_id():
                self.current_window_label.config(text="Current: OSRS Automation Recorder (GUI)", fg="blue")
                self.focus_status_label.config(text="Target Focus: GUI Focused", fg="orange")
                return
            
            # Update current window label
            if current_title and current_title not in ["No window focused", "Error"]:
                short_title = current_title[:30] + "..." if len(current_title) > 30 else current_title
                self.current_window_label.config(text=f"Current: {short_title}", fg="gray")
                
                # Check if current window matches OSRS keywords
                title_lower = current_title.lower()
                osrs_keywords = ['runelite', 'old school runescape', 'osrs', 'jagex', 'runescape', 'rs3', 'rs2', '2007scape']
                if any(keyword in title_lower for keyword in osrs_keywords):
                    self.current_window_label.config(fg="green")  # Highlight OSRS windows
                else:
                    self.current_window_label.config(fg="gray")
            else:
                self.current_window_label.config(text="Current: None", fg="gray")
            
            # Check if target window is focused
            is_focused = self.is_osrs_window_focused()
            
            # Update focus status
            if is_focused:
                self.focus_status_label.config(text="Target Focus: Yes", fg="green")
            else:
                self.focus_status_label.config(text="Target Focus: No", fg="red")
                
        except Exception as e:
            print(f"Error updating focus status: {e}")
            self.focus_status_label.config(text="Target Focus: Error", fg="red")

    def recording_loop(self):
        """Main recording loop that processes queued events and writes to CSV"""
        last_focus_update = 0
        while not self.stop_event.is_set():
            try:
                # Process mouse events
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
                
                # Update focus status every 100ms
                current_time = time.time()
                if current_time - last_focus_update > 0.1:
                    self.update_focus_status()
                    last_focus_update = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                time.sleep(0.1)

    def load_feature_data(self):
        import json
        import os
        import numpy as np
        data_dir = self.feature_data_dir_var.get()
        states_path = os.path.join(data_dir, 'states.json')
        actions_path = os.path.join(data_dir, 'actions.json')
        state_features_path = os.path.join(data_dir, 'state_features.npy')
        action_features_path = os.path.join(data_dir, 'action_features.npy')
        if not (os.path.exists(states_path) and os.path.exists(actions_path) and os.path.exists(state_features_path) and os.path.exists(action_features_path)):
            tk.messagebox.showerror("Missing Data", "Could not find all required files in the selected directory.")
            return
        with open(states_path, 'r') as f:
            self.feature_states = json.load(f)
        with open(actions_path, 'r') as f:
            self.feature_actions = json.load(f)
        self.state_features = np.load(state_features_path)
        self.action_features = np.load(action_features_path)
        self.feature_example_idx_var.set(0)
        self.show_feature_example(0)

    def show_feature_example(self, direction):
        idx = self.feature_example_idx_var.get()
        if direction != 0:
            idx += direction
        if idx < 0:
            idx = 0
        if idx >= len(self.feature_states):
            idx = len(self.feature_states) - 1
        self.feature_example_idx_var.set(idx)
        self.feature_example_label.config(text=f"Example: {idx}")
        self.clear_feature_display()
        # Display state and action JSON
        state = self.feature_states[idx]
        action = self.feature_actions[idx]
        self.feature_state_text.insert('1.0', json.dumps(state, indent=2))
        self.feature_action_text.insert('1.0', json.dumps(action, indent=2))
        # Display features
        if self.state_features is not None and idx < len(self.state_features):
            self.state_features_text.insert('1.0', np.array2string(self.state_features[idx], precision=3, separator=','))
        if self.action_features is not None and idx < len(self.action_features):
            self.action_features_text.insert('1.0', np.array2string(self.action_features[idx], precision=3, separator=','))
        # Optionally display screenshot if available
        data_dir = self.feature_data_dir_var.get()
        screenshot_path = os.path.join(data_dir, 'screenshots', f'{idx}.png')
        if os.path.exists(screenshot_path):
            from PIL import Image, ImageTk
            img = Image.open(screenshot_path)
            img = img.resize((200, 150))
            photo = ImageTk.PhotoImage(img)
            self.feature_screenshot_label.config(image=photo)
            self.feature_screenshot_label.image = photo
        else:
            self.feature_screenshot_label.config(image=None)
            self.feature_screenshot_label.image = None

    def clear_feature_display(self):
        self.feature_state_text.delete('1.0', tk.END)
        self.feature_action_text.delete('1.0', tk.END)
        self.state_features_text.delete('1.0', tk.END)
        self.action_features_text.delete('1.0', tk.END)
        self.feature_screenshot_label.config(image=None)
        self.feature_screenshot_label.image = None

    def on_closing(self):
        """Handle application closing"""
        if self.recording:
            self.stop_recording()
        self.root.destroy()

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording:
            self.record_button.config(text="Stop Recording")
            self.start_recording()
        else:
            self.record_button.config(text="Start Recording")
            self.stop_recording()

    def start_recording(self):
        """Start the recording process"""
        if self.recording:
            return
        
        # Ensure data directory exists
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.ACTION_LOG):
            with open(self.ACTION_LOG, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
                    'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
                ])
        
        # Open CSV file for writing
        try:
            self.csvf = open(self.ACTION_LOG, 'a', newline='')
            self.csv_writer = csv.writer(self.csvf)
        except Exception as e:
            return
        
        # Reset counters
        self.click_count = 0
        self.key_press_count = 0
        self.key_release_count = 0
        self.mouse_move_count = 0
        self.scroll_count = 0
        self.update_counters()
        
        # Start input listeners
        self.start_input_listeners()
        
        # Start recording thread
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.thread.start()
        
        self.recording = True
        self.record_button.config(text="Stop Recording")

    def stop_recording(self):
        """Stop the recording process"""
        if not self.recording:
            return
        
        self.recording = False
        self.stop_event.set()
        
        # Stop input listeners
        self.stop_input_listeners()
        
        # Close CSV file
        if self.csvf:
            self.csvf.close()
            self.csvf = None
        
        # Wait for recording thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.record_button.config(text="Start Recording")
        print("Recording stopped")

    def start_input_listeners(self):
        """Start mouse and keyboard listeners"""
        # Mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # Keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

    def stop_input_listeners(self):
        """Stop mouse and keyboard listeners"""
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

    def on_mouse_move(self, x, y):
        """Handle mouse movement events"""
        try:
            if not self.recording:
                return
            
            # Only record if OSRS window is focused
            if not self.is_osrs_window_focused():
                return
            
            current_time = time.time_ns()
            # Throttle mouse move events to avoid flooding
            if current_time - self.last_mouse_move_time_ns > 10000000:  # 10ms
                self.mouse_move_count += 1
                self.last_mouse_move_time_ns = current_time
                
                # Queue the event for processing
                event_data = {
                    'timestamp': int(time.time() * 1000),
                    'event_type': 'move',
                    'x_in_window': x,
                    'y_in_window': y,
                    'btn': '',
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0,
                    'modifiers': '',
                    'active_keys': ''
                }
                self.click_queue.append(event_data)
                
                self.update_counters()
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        try:
            if not self.recording:
                return
            
            # Only record if OSRS window is focused
            if not self.is_osrs_window_focused():
                return
            
            if pressed:
                self.click_count += 1
                btn_name = str(button).split('.')[-1]
                
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
                    'x_in_window': x,
                    'y_in_window': y,
                    'btn': btn_name,
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0,
                    'modifiers': modifiers,
                    'active_keys': active_keys
                }
                self.click_queue.append(event_data)
                
                self.update_counters()
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        try:
            if not self.recording:
                return
            
            # Only record if OSRS window is focused
            if not self.is_osrs_window_focused():
                return
            
            self.scroll_count += 1
            
            # Safely get modifier and active keys strings
            try:
                modifiers = self.get_modifier_string()
            except:
                modifiers = ''
            
            try:
                active_keys = self.get_active_keys_string()
            except:
                active_keys = ''
            
            self.click_queue.append({
                'timestamp': int(time.time() * 1000),
                'event_type': 'scroll',
                'x_in_window': x,
                'y_in_window': y,
                'btn': '',
                'key': '',
                'scroll_dx': dx,
                'scroll_dy': dy,
                'modifiers': modifiers,
                'active_keys': active_keys
            })
            
            self.update_counters()
        except Exception as e:
            # Continue recording even if this event fails
            pass

    def on_key_press(self, key):
        """Handle key press events"""
        if not self.recording:
            return
        
        # Only record if OSRS window is focused
        if not self.is_osrs_window_focused():
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
            
            self.key_press_count += 1
            
            # Track modifier keys
            if key in [Key.shift, Key.ctrl, Key.alt, Key.cmd]:
                self.modifier_state.add(str(key))
            
            # Track key hold times
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
            
            self.key_queue.append({
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
            })
            
            self.update_counters()
            
        except Exception as e:
            # Log the error but continue recording
            print(f"Key press recording error: {e}, key: {key}")
            # Still record the key event with fallback
            fallback_key = str(key) if key else "unknown"
            self.key_queue.append({
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
            })

    def on_key_release(self, key):
        """Handle key release events"""
        if not self.recording:
            return
        
        # Only record if OSRS window is focused
        if not self.is_osrs_window_focused():
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
            
            self.key_release_count += 1
            
            # Remove modifier keys
            if str(key) in self.modifier_state:
                self.modifier_state.discard(str(key))
            
            # Remove from hold times
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
            
            self.key_queue.append({
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
            })
            
            self.update_counters()
            
        except Exception as e:
            # Log the error but continue recording
            print(f"Key release recording error: {e}, key: {key}")
            # Still record the key event with fallback
            fallback_key = str(key) if key else "unknown"
            self.key_queue.append({
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
            })

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
            if current_time - press_time > 0.1:  # 100ms threshold
                try:
                    # Improved key handling to capture ALL keys universally
                    if hasattr(key, 'char') and key.char is not None:
                        key_char = key.char
                    elif hasattr(key, 'name') and key.name is not None:
                        key_char = key.name
                    else:
                        key_char = str(key)
                    
                    # Ensure we always have a valid key identifier
                    if key_char and key_char != 'None':
                        active_keys.append(str(key_char))
                except Exception:
                    # Fallback to string representation
                    try:
                        key_str = str(key)
                        if key_str and key_str != 'None':
                            active_keys.append(key_str)
                    except:
                        pass
        
        return '+'.join(active_keys)

    def update_counters(self):
        """Update the GUI counter labels"""
        try:
            self.click_label.config(text=f"Clicks: {self.click_count}")
            self.key_press_label.config(text=f"Key Presses: {self.key_press_count}")
            self.key_release_label.config(text=f"Key Releases: {self.key_release_count}")
            self.mouse_move_label.config(text=f"Mouse Moves: {self.mouse_move_count}")
            self.scroll_label.config(text=f"Scrolls: {self.scroll_count}")
            self.update_focus_status()
        except:
            pass  # GUI might not be ready yet

    def start_bot(self):
        """Start the bot"""
        self.bot_status_label.config(text="Bot status: Running")
        self.start_bot_button.config(state="disabled")
        self.stop_bot_button.config(state="normal")

    def stop_bot(self):
        """Stop the bot"""
        self.bot_status_label.config(text="Bot status: Idle")
        self.start_bot_button.config(state="normal")
        self.stop_bot_button.config(state="disabled")

    def run_debug(self):
        """Run debug functionality"""
        self.debug_text.insert(tk.END, "Debug started...\n")
        # Implementation to be added

    def start_training(self):
        """Start model training"""
        self.train_status_label.config(text="Training status: Running")
        self.train_button.config(state="disabled")
        self.stop_train_button.config(state="normal")

    def stop_training(self):
        """Stop model training"""
        self.train_status_label.config(text="Training status: Idle")
        self.train_button.config(state="normal")
        self.stop_train_button.config(state="disabled")

    def run_preprocessing(self):
        """Run data preprocessing"""
        self.preprocessing_progress['value'] = 0
        self.root.update_idletasks()

        collector = DataCollector(
            self.gamestate_dir_var.get(),
            self.actions_csv_var.get(),
            SCREENSHOT_DIR,
        )

        records, mismatches = collector.synchronize_records()
        self.preprocessing_progress['value'] = 100
        self.root.update_idletasks()

        msg = f"Aligned {len(records)} records."
        mismatch_lines = []
        for key, vals in mismatches.items():
            if vals:
                mismatch_lines.append(f"{key.replace('_', ' ').title()}: {len(vals)}")

        if mismatch_lines:
            msg += "\n" + "\n".join(mismatch_lines)
            messagebox.showwarning("Preprocessing", msg)
        else:
            messagebox.showinfo("Preprocessing", msg)

    def load_data(self):
        """Load data for inspection"""
        self.data_inspector_text.delete('1.0', tk.END)
        self.data_inspector_text.insert(tk.END, "Data loading...\n")
        # Implementation to be added

    def browse_feature_data_dir(self):
        """Browse for feature data directory"""
        from tkinter import filedialog
        directory = filedialog.askdirectory()
        if directory:
            self.feature_data_dir_var.set(directory)

    def test_window_detection(self):
        """Test window detection functionality"""
        self.debug_text.insert(tk.END, "Testing window detection...\n")
        self.debug_text.insert(tk.END, "Current focused window:\n")
        current_hwnd, current_title = self.get_current_window_info()
        if current_hwnd:
            current_details = self.get_window_details(current_hwnd)
            if isinstance(current_details, dict):
                self.debug_text.insert(tk.END, f"  Title: {current_details['title']}\n")
                self.debug_text.insert(tk.END, f"  Class: {current_details['class']}\n")
                self.debug_text.insert(tk.END, f"  Size: {current_details['size']}\n")
                self.debug_text.insert(tk.END, f"  Handle: {current_details['hwnd']}\n")
            else:
                self.debug_text.insert(tk.END, f"  {current_details}\n")
        else:
            self.debug_text.insert(tk.END, f"  {current_title}\n")
        self.debug_text.insert(tk.END, f"Target window: {self.window_var.get()}\n")
        self.debug_text.insert(tk.END, f"Is OSRS window focused: {self.is_osrs_window_focused()}\n")
        self.debug_text.insert(tk.END, "Test complete.\n")
        self.debug_text.see(tk.END)

    def on_gui_interaction(self, event):
        """Temporarily pause focus checking when GUI is interacted with"""
        if self.focus_checking_paused:
            return
        self.focus_checking_paused = True
        self.focus_pause_timer = threading.Timer(0.5, self.resume_focus_checking) # Resume after 0.5 seconds
        self.focus_pause_timer.start()

    def on_gui_focus_in(self, event):
        """Temporarily pause focus checking when GUI gains focus"""
        if self.focus_checking_paused:
            return
        self.focus_checking_paused = True
        self.focus_pause_timer = threading.Timer(0.5, self.resume_focus_checking) # Resume after 0.5 seconds
        self.focus_pause_timer.start()

    def on_gui_focus_out(self, event):
        """Temporarily pause focus checking when GUI loses focus"""
        if self.focus_checking_paused:
            return
        self.focus_checking_paused = True
        self.focus_pause_timer = threading.Timer(0.5, self.resume_focus_checking) # Resume after 0.5 seconds
        self.focus_pause_timer.start()

    def resume_focus_checking(self):
        """Resume focus checking after a GUI interaction"""
        self.focus_checking_paused = False
        self.focus_pause_timer = None
        self.update_focus_status() # Update status to reflect resumed focus

# --- End OsrsAutomationApp ---

# Main entry point
if __name__ == '__main__':
    root = tk.Tk()
    app = OsrsAutomationApp(root)
    root.mainloop() 
