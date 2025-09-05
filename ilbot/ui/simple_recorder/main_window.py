"""Main window for the simple recorder GUI."""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from datetime import datetime
try:
    from .recording_service import RecordingService
except ImportError:
    # Fallback for when running as script directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from ilbot.ui.simple_recorder.recording_service import RecordingService


class SimpleRecorderWindow:
    def __init__(self, root):
        self.root = root
        self.recording_service = None
        self.session_dir = None
        self.recording = False
        self.paused = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Left column (controls)
        main_frame.columnconfigure(1, weight=1)  # Right column (gamestate info)
        
        # Title
        title_label = ttk.Label(main_frame, text="Simple Bot Recorder", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create left frame for controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(1, weight=1)
        
        # Window detection section
        ttk.Label(left_frame, text="Window Detection:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.E, pady=(0, 10))
        
        self.detect_button = ttk.Button(left_frame, text="Detect Runelite Window", command=self.detect_window)
        self.detect_button.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.window_status = ttk.Label(left_frame, text="No window detected", foreground="red")
        self.window_status.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky=tk.E)
        
        # Session management section
        ttk.Label(left_frame, text="Session Management:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.E, pady=(0, 10))
        
        self.create_session_button = ttk.Button(left_frame, text="Create Session", command=self.create_session, state="disabled")
        self.create_session_button.grid(row=4, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.session_status = ttk.Label(left_frame, text="No session created", foreground="red")
        self.session_status.grid(row=5, column=0, columnspan=2, pady=(0, 10), sticky=tk.E)
        
        # Copy path button (initially disabled)
        self.copy_path_button = ttk.Button(left_frame, text="Copy Gamestates Path", command=self.copy_gamestates_path, state="disabled")
        self.copy_path_button.grid(row=6, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Recording controls section
        ttk.Label(left_frame, text="Recording Controls:", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.E, pady=(0, 10))
        
        # Recording buttons frame
        recording_frame = ttk.Frame(left_frame)
        recording_frame.grid(row=8, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        recording_frame.columnconfigure(0, weight=1)
        recording_frame.columnconfigure(1, weight=1)
        recording_frame.columnconfigure(2, weight=1)
        
        self.start_button = ttk.Button(recording_frame, text="Start Recording", command=self.start_recording, state="disabled")
        self.start_button.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        self.pause_button = ttk.Button(recording_frame, text="Pause Recording", command=self.pause_recording, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.end_session_button = ttk.Button(recording_frame, text="End Session", command=self.end_session, state="disabled")
        self.end_session_button.grid(row=0, column=2, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # Recording status
        self.recording_status = ttk.Label(left_frame, text="Not recording", foreground="red")
        self.recording_status.grid(row=9, column=0, columnspan=2, pady=(0, 20), sticky=tk.E)
        
        # Countdown label
        self.countdown_label = ttk.Label(left_frame, text="", font=("Arial", 14, "bold"))
        self.countdown_label.grid(row=10, column=0, columnspan=2, sticky=tk.E)
        
        # Live Gamestate Information section (right column)
        ttk.Label(main_frame, text="Live Gamestate Information:", font=("Arial", 12, "bold")).grid(row=1, column=1, sticky=tk.E, pady=(0, 10))
        
        # Create a frame for the gamestate info with a border
        gamestate_frame = ttk.LabelFrame(main_frame, text="Current Gamestate", padding="10")
        gamestate_frame.grid(row=2, column=1, pady=(0, 20), sticky=(tk.E, tk.N, tk.S))
        gamestate_frame.columnconfigure(1, weight=1)
        
        # Gamestate information labels (placeholder for now)
        self.gamestate_info_labels = {}
        
        # Create rows for different gamestate information
        info_rows = [
            ("Timestamp:", "timestamp"),
            ("Player Position:", "player_pos"),
            ("Camera Position:", "camera_pos"),
            ("Current Action:", "current_action"),
            ("Game State:", "game_state"),
            ("Inventory Items:", "inventory_count"),
            ("Combat Status:", "combat_status"),
            ("Last Event:", "last_event")
        ]
        
        for i, (label_text, key) in enumerate(info_rows):
            # Label
            ttk.Label(gamestate_frame, text=label_text, font=("Arial", 9, "bold")).grid(
                row=i, column=0, sticky=tk.E, pady=2, padx=(0, 10)
            )
            
            # Value label
            value_label = ttk.Label(gamestate_frame, text="N/A", foreground="gray", font=("Arial", 9))
            value_label.grid(row=i, column=1, sticky=tk.E, pady=2)
            self.gamestate_info_labels[key] = value_label
        
        # Update button for manual refresh
        self.refresh_gamestate_button = ttk.Button(
            gamestate_frame, 
            text="Refresh Gamestate Info", 
            command=self.refresh_gamestate_info,
            state="disabled"
        )
        self.refresh_gamestate_button.grid(row=len(info_rows), column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        
    def detect_window(self):
        """Detect the Runelite window."""
        try:
            self.detect_button.config(state="disabled", text="Detecting...")
            
            # Import here to avoid circular imports
            try:
                from .window_finder import WindowFinder
            except ImportError:
                # Fallback for when running as script directly
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
                from ilbot.ui.simple_recorder.window_finder import WindowFinder
            finder = WindowFinder()
            window_info = finder.find_runelite_window()
            
            if window_info:
                self.recording_service = RecordingService(window_info)
                self.window_status.config(text=f"Window detected: {window_info['title']}", foreground="green")
                self.create_session_button.config(state="normal")
            else:
                self.window_status.config(text="No Runelite window found", foreground="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect window: {e}")
        finally:
            self.detect_button.config(state="normal", text="Detect Runelite Window")
            
    def create_session(self):
        """Create a new recording session."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join("data", "recording_sessions", timestamp)
            gamestates_dir = os.path.join(self.session_dir, "gamestates")
            
            # Create directories
            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(gamestates_dir, exist_ok=True)
            
            self.session_status.config(text=f"Session created: {timestamp}", foreground="green")
            self.start_button.config(state="normal")
            self.end_session_button.config(state="normal")
            self.copy_path_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create session: {e}")
            
    def start_recording(self):
        """Start recording with countdown."""
        if not self.recording_service or not self.session_dir:
            messagebox.showerror("Error", "Please detect window and create session first")
            return
            
        self.recording = True
        self.paused = False
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.refresh_gamestate_button.config(state="disabled")  # Disable until recording actually starts
        self.recording_status.config(text="Starting recording...", foreground="orange")
        
        # Start countdown in separate thread
        countdown_thread = threading.Thread(target=self.countdown_and_start, daemon=True)
        countdown_thread.start()
        
    def countdown_and_start(self):
        """Countdown from 5 seconds and start recording."""
        for i in range(5, 0, -1):
            self.root.after(0, lambda count=i: self.countdown_label.config(text=str(count)))
            time.sleep(1)
            
        self.root.after(0, lambda: self.countdown_label.config(text=""))
        self.root.after(0, self.actually_start_recording)
        
    def actually_start_recording(self):
        """Actually start the recording after countdown."""
        try:
            actions_file = os.path.join(self.session_dir, "actions.csv")
            self.recording_service.start_recording(actions_file)
            
            self.recording_status.config(text="Recording...", foreground="green")
            self.refresh_gamestate_button.config(state="normal")  # Enable gamestate refresh when recording starts
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.recording_status.config(text="Recording failed", foreground="red")
            self.start_button.config(state="normal")
            self.pause_button.config(state="disabled")
            
    def pause_recording(self):
        """Pause or resume recording."""
        if not self.recording:
            return
            
        if self.paused:
            # Resume recording
            self.recording_service.resume_recording()
            self.pause_button.config(text="Pause Recording")
            self.recording_status.config(text="Recording...", foreground="green")
            self.paused = False
        else:
            # Pause recording
            self.recording_service.pause_recording()
            self.pause_button.config(text="Resume Recording")
            self.recording_status.config(text="Recording paused", foreground="orange")
            self.paused = True
            
    def end_session(self):
        """End the current recording session."""
        if self.recording:
            self.recording_service.stop_recording()
            self.recording = False
            self.paused = False
            
        self.recording_status.config(text="Session ended", foreground="red")
        self.start_button.config(state="disabled")
        self.pause_button.config(state="disabled")
        self.refresh_gamestate_button.config(state="disabled")  # Disable gamestate refresh when session ends
        self.countdown_label.config(text="")
        
        # Clear gamestate info when session ends
        for key, label in self.gamestate_info_labels.items():
            label.config(text="N/A", foreground="gray")
        
        # Reset session
        self.session_dir = None
        self.session_status.config(text="No session created", foreground="red")
        self.create_session_button.config(state="normal")
        self.copy_path_button.config(state="disabled")
        
    def copy_gamestates_path(self):
        """Copy the gamestates folder path to clipboard."""
        if not self.session_dir:
            return
            
        try:
            gamestates_path = os.path.join(self.session_dir, "gamestates")
            # Convert to absolute path
            abs_path = os.path.abspath(gamestates_path)
            
            # Copy to clipboard
            self.root.clipboard_clear()
            self.root.clipboard_append(abs_path + "\\")
            
            # Update status
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy path: {e}")
    
    def refresh_gamestate_info(self):
        """Refresh the live gamestate information display."""
        try:
            if not self.recording_service or not self.recording:
                # Clear all values when not recording
                for key, label in self.gamestate_info_labels.items():
                    label.config(text="N/A", foreground="gray")
                return
            
            # TODO: Implement actual gamestate data retrieval
            # For now, just show placeholder data
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Update labels with placeholder data
            self.gamestate_info_labels["timestamp"].config(text=current_time, foreground="black")
            self.gamestate_info_labels["player_pos"].config(text="(1234, 5678)", foreground="black")
            self.gamestate_info_labels["camera_pos"].config(text="(1200, 5600)", foreground="black")
            self.gamestate_info_labels["current_action"].config(text="Moving", foreground="blue")
            self.gamestate_info_labels["game_state"].config(text="In Game", foreground="green")
            self.gamestate_info_labels["inventory_count"].config(text="28/28", foreground="black")
            self.gamestate_info_labels["combat_status"].config(text="Not in combat", foreground="green")
            self.gamestate_info_labels["last_event"].config(text="Mouse move", foreground="black")
            
            
        except Exception as e:
            # Reset all labels to N/A on error
            for key, label in self.gamestate_info_labels.items():
                label.config(text="N/A", foreground="gray")
