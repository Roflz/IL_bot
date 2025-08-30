"""Main window for the simple recorder GUI."""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from datetime import datetime
from .recording_service import RecordingService


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
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Simple Bot Recorder", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Window detection section
        ttk.Label(main_frame, text="Window Detection:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        self.detect_button = ttk.Button(main_frame, text="Detect Runelite Window", command=self.detect_window)
        self.detect_button.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.window_status = ttk.Label(main_frame, text="No window detected", foreground="red")
        self.window_status.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Session management section
        ttk.Label(main_frame, text="Session Management:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(0, 10))
        
        self.create_session_button = ttk.Button(main_frame, text="Create Session", command=self.create_session, state="disabled")
        self.create_session_button.grid(row=5, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.session_status = ttk.Label(main_frame, text="No session created", foreground="red")
        self.session_status.grid(row=6, column=0, columnspan=2, pady=(0, 10))
        
        # Copy path button (initially disabled)
        self.copy_path_button = ttk.Button(main_frame, text="Copy Gamestates Path", command=self.copy_gamestates_path, state="disabled")
        self.copy_path_button.grid(row=7, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Recording controls section
        ttk.Label(main_frame, text="Recording Controls:", font=("Arial", 12, "bold")).grid(row=8, column=0, sticky=tk.W, pady=(0, 10))
        
        # Recording buttons frame
        recording_frame = ttk.Frame(main_frame)
        recording_frame.grid(row=9, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
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
        self.recording_status = ttk.Label(main_frame, text="Not recording", foreground="red")
        self.recording_status.grid(row=10, column=0, columnspan=2, pady=(0, 20))
        
        # Countdown label
        self.countdown_label = ttk.Label(main_frame, text="", font=("Arial", 14, "bold"))
        self.countdown_label.grid(row=11, column=0, columnspan=2)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
    def detect_window(self):
        """Detect the Runelite window."""
        try:
            self.detect_button.config(state="disabled", text="Detecting...")
            self.status_bar.config(text="Detecting Runelite window...")
            
            # Import here to avoid circular imports
            from .window_finder import WindowFinder
            finder = WindowFinder()
            window_info = finder.find_runelite_window()
            
            if window_info:
                self.recording_service = RecordingService(window_info)
                self.window_status.config(text=f"Window detected: {window_info['title']}", foreground="green")
                self.create_session_button.config(state="normal")
                self.status_bar.config(text=f"Window detected: {window_info['title']}")
            else:
                self.window_status.config(text="No Runelite window found", foreground="red")
                self.status_bar.config(text="No Runelite window found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect window: {e}")
            self.status_bar.config(text="Window detection failed")
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
            self.status_bar.config(text=f"Session created: {self.session_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create session: {e}")
            self.status_bar.config(text="Session creation failed")
            
    def start_recording(self):
        """Start recording with countdown."""
        if not self.recording_service or not self.session_dir:
            messagebox.showerror("Error", "Please detect window and create session first")
            return
            
        self.recording = True
        self.paused = False
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
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
            self.status_bar.config(text="Recording active")
            
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
            self.status_bar.config(text="Recording resumed")
            self.paused = False
        else:
            # Pause recording
            self.recording_service.pause_recording()
            self.pause_button.config(text="Resume Recording")
            self.recording_status.config(text="Recording paused", foreground="orange")
            self.status_bar.config(text="Recording paused")
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
        self.countdown_label.config(text="")
        self.status_bar.config(text="Session ended")
        
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
            self.status_bar.config(text=f"Copied to clipboard: {abs_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy path: {e}")
            self.status_bar.config(text="Failed to copy path")
