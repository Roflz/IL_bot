#!/usr/bin/env python3
"""
OSRS Bot Controller GUI
Safe interface for testing and controlling the imitation learning bot

TODO - MISSING PIECES FOR FULLY CORRECT PREDICTIONS:
1. Action enum mapping: Replace hardcoded type/button/key integers with exact training encodings
   - TYPE = {"click": 1, "scroll": 2, "key": 3}  # TODO: replace with your real encodings
   - BUTTON = {"left": 1, "right": 2, "middle": 3}  # TODO
   - KEYMAP = { ... }  # if you used a custom mapping in training

2. Plugin event field names: Confirm RuneLite JSON sends recent_actions/actions/last_actions with fields:
   timestamp (ms), type, x, y, button, key, scroll_dx, scroll_dy
   If names differ, adapt the packer to read the correct keys.

3. Coordinate space: Confirm training used raw live-view pixels (no normalization).
   If any scaling was used, apply the same transform before packing.

4. Timestamp units: We assume plugin/user timestamps are milliseconds.
   If seconds were used in training, adjust conversions accordingly.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font, filedialog
import threading
import time
import cv2
import numpy as np
import pyautogui
try:
    import pygetwindow as gw
    PYWINDOW_AVAILABLE = True
except ImportError:
    gw = None
    PYWINDOW_AVAILABLE = False
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import torch
from datetime import datetime
import csv

from live_feature_extractor import LiveFeatureExtractor
from collections import deque
from model_runner import ModelRunner

# Performance debugging
PERF_DEBUG = True  # set False to silence

def tnow():
    import time
    return time.perf_counter()

# Centralized path resolution
BASE_DIR = Path(__file__).resolve().parent

def rp(*parts: str) -> Path:
    """Repo-local absolute path builder."""
    return (BASE_DIR / Path(*parts)).resolve()

DATA_DIR = rp("data")
FEATURES_DIR = rp("data", "features")
MODEL_PATH = rp("training_results", "model_weights.pth")

# User input tracking (guarded import)
try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


class BotControllerGUI:
    """
    GUI controller for the OSRS imitation learning bot.
    """
    
    def __init__(self, root):
        print("[INIT] begin")
        self.root = root
        self.root.title("🤖 OSRS Bot Controller")
        self.root.geometry("1000x800")
        
        # Bot state
        self.bot = None
        self.is_running = False
        self.screenshot_thread = None
        self.bot_thread = None
        
        # Model
        self.model = None
        self.model_loaded = False
        
        # Screenshot region
        self.screenshot_region = None
        self.window_detected = False
        self.live_view_region = (0, 0, 800, 600)  # Default bounds for coordinate clamping
        
        # Live gamestate mode
        self.live_gamestate_mode = False
        self.live_feature_extractor = None
        self.gamestate_thread = None
        
        # Feature tracking state
        self.feature_tracking_active = False
        self.feature_tracking_thread = None
        
        # Live data state for canvas rendering
        self._latest_gamestate = None
        self._latest_features = None
        
        # Inference system
        self.gs_buf = deque(maxlen=10)      # ring buffer for gamestate features (128,)
        self.action_buf = deque(maxlen=10)  # ring buffer for action frames (101,8)
        self._pred_lock = threading.Lock()
        self._last_pred = None              # np.ndarray (101,8)
        self._infer_on = False              # inference toggle
        self._infer_thread = None           # inference thread
        self.predicted_actions = None       # latest predicted actions
        self._mappings_sorted = None        # cached feature mappings sorted by index
        self.pred_history = deque(maxlen=50)  # list of dicts {timestamp, actions: np.ndarray(N,8)}
        
        # Feature tracking resilience
        self._last_gamestate_ts = None
        self._stale_since = None  # time.time() when we first saw a stalled timestamp
        
        # Performance tracking
        self._last_perf_log = 0
        self._render_start_time = None
        
        # Render scheduling
        self._render_job = None
        self._render_pending = False
        
        # Queued logging
        self._log_queue = []
        self._log_flush_job = None
        
        # User input tracking
        self.user_events = deque(maxlen=200)  # each: dict with timestamp(ms), type, x, y, button, key, scroll_dx, scroll_dy
        self._mouse_listener = None
        self._key_listener = None
        
        # === Feature Dashboard Styles ===
        self.FEAT_STYLE = {
            "bg_panel": "#0D1117",   # canvas background
            "bg_group": "#111827",
            "border_group": "#2D3748",
            "fg_header": "#D1D9FF",
            "fg_label": "#AAB4C3",
            "fg_value": "#F1F5FB",
            "fg_muted": "#8A94A6",
            "ok": "#22C55E",   # slightly brighter green
            "warn": "#F59E0B",
            "err": "#EF4444",
        }

        self.FEAT_LAYOUT = {
            "padding": 16,      # outer padding
            "gap_group": 10,    # vertical gap between groups
            "gap_row": 4,       # vertical gap between rows within a group
            "gap_col": 12,      # horizontal gap between label/value columns
            "group_hdr_h": 22,  # header strip height (reduced)
            "corner": 8,        # rounded-ish corners (emulated)
            "min_col_width": 300,  # allows more columns for better balance
        }
        
        # === Cached Fonts ===
        self._font_header = font.Font(family="Segoe UI", size=9, weight="bold")
        self._font_label  = font.Font(family="Segoe UI", size=8, weight="bold")   # <- bold
        self._font_value  = font.Font(family="Segoe UI", size=8, weight="bold")   # <- bold
        self._font_small  = font.Font(family="Segoe UI", size=7, weight="normal")
        
        # === Text Mapping Deduplication ===
        self._warned_texts = set()  # Track which texts we've already warned about
        self._text_to_id = {}       # Cache text->id mappings for session
        # Seed common UI texts to avoid spam
        COMMON_UI_TEXTS = {"Close", "Cancel", "OK", "Apply", "Yes", "No"}
        for text in COMMON_UI_TEXTS:
            self._text_to_id[text] = hash(text) % 100000
        
        # Create GUI
        self.create_widgets()
        print("[INIT] widgets created")
        
        # Initialize
        self.check_model()
        self.load_feature_mappings()
        
        # Log startup paths
        self.log_startup_paths()
        
        # Log pygetwindow status
        if PYWINDOW_AVAILABLE:
            self.log("✅ pygetwindow library available")
        else:
            self.log("❌ pygetwindow library not available")
            self.log("   Automatic window detection disabled")
            self.log("   Use 'Set Manual Region' button to set screenshot area")
        
        # Log initial bot mode
        initial_bot = self.bot_mode_var.get()
        self.log(f"🤖 Initial bot mode: {initial_bot}")
        self.log(f"   📁 Will read from: data/{initial_bot}/runelite_gamestate.json")
        
        # Print initial paths and status
        self.print_paths_and_status()
        
        self.detect_game_window()
        
        # Create LiveFeatureExtractor
        self.live_feature_extractor = LiveFeatureExtractor(initial_bot)
        print("[INIT] extractor ready")
        
        print("[INIT] done")
    
    def log_startup_paths(self):
        """Log startup paths for diagnostics."""
        self.log("🔍 STARTUP PATHS:")
        self.log(f"   Model weights: {rp('training_results', 'model_weights.pth')}")
        self.log(f"   Feature mappings: {rp('data', 'features', 'feature_mappings.json')}")
        self.log(f"   ID mappings: {rp('data', 'features', 'id_mappings.json')}")
        
        # Get current bot mode paths
        current_bot = self.bot_mode_var.get()
        self.log(f"   Live JSON ({current_bot}): {rp('data', current_bot, 'runelite_gamestate.json')}")
        self.log(f"   Rolling folder ({current_bot}): {rp('data', current_bot, 'gamestates')}")
        
        # Also print to stdout for pre-freeze diagnostics
        print(f"[PATHS] Model weights: {rp('training_results', 'model_weights.pth')}")
        print(f"[PATHS] Feature mappings: {rp('data', 'features', 'feature_mappings.json')}")
        print(f"[PATHS] ID mappings: {rp('data', 'features', 'id_mappings.json')}")
        print(f"[PATHS] Live JSON ({current_bot}): {rp('data', current_bot, 'runelite_gamestate.json')}")
        print(f"[PATHS] Rolling folder ({current_bot}): {rp('data', current_bot, 'gamestates')}")
    
    def schedule_render(self, delay_ms=0):
        """Schedule a render with coalescing."""
        if self._render_job:
            self.root.after_cancel(self._render_job)
            self._render_job = None
        self._render_job = self.root.after(delay_ms, self._run_render)
    
    def _run_render(self):
        """Execute the scheduled render."""
        self._render_job = None
        
        t0 = time.perf_counter()
        
        try:
            # Print first render message
            if not hasattr(self, '_first_render_done'):
                print("[RENDER] first draw")
                self._first_render_done = True
            
            self._update_live_feature_table()
            
            # Print timing if render is slow
            dt = (time.perf_counter() - t0) * 1000
            if dt > 30:
                print(f"[RENDER] slow frame {dt:.1f}ms")
                    
        except Exception as e:
            self.log(f"❌ render error: {e}")
    
    def _flush_logs(self):
        """Flush queued log messages to the text widget."""
        self._log_flush_job = None
        if not self._log_queue:
            return
        
        # Batch insert to avoid per-line layout churn
        buf = "".join(self._log_queue)
        self._log_queue.clear()
        self.log_text.insert(tk.END, buf)
        self.log_text.see(tk.END)
    
    def _enqueue_log_text(self, s: str):
        """Enqueue text for batched logging."""
        self._log_queue.append(s)
        # Coalesce to ~60fps max; typical UI is fine with 50–150ms
        if self._log_flush_job is None:
            self._log_flush_job = self.root.after(80, self._flush_logs)
    
    def create_widgets(self):
        import tkinter as tk
        from tkinter import ttk, scrolledtext

        # ---- Root & main frame ----
        self.root.geometry("1280x800")  # optional sane default
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.rowconfigure(1, weight=1)      # body (paned windows)
        main.columnconfigure(0, weight=1)

        # ---- Title ----
        title = ttk.Label(main, text="🤖 OSRS Imitation Learning Bot Controller", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        # ---- Body: Horizontal split (left controls | right content) ----
        body_panes = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        body_panes.grid(row=1, column=0, sticky="nsew")

        # LEFT: Controls (fixed-ish width)
        left = ttk.Frame(body_panes)
        left.columnconfigure(0, weight=1)
        body_panes.add(left, weight=0)  # give almost all weight to the right pane

        # RIGHT: Vertical split (Live View | Logs)
        right_panes = ttk.Panedwindow(body_panes, orient=tk.VERTICAL)
        body_panes.add(right_panes, weight=1)

        # ---- LEFT: Controls content ----
        # Model status + load
        model_frame = ttk.LabelFrame(left, text="🧠 Model", padding=8)
        model_frame.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=0)

        self.model_status_label = ttk.Label(model_frame, text="❌ Model: Not Loaded", foreground="red")
        self.model_status_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.grid(row=1, column=0, sticky="ew")

        # Game window detection
        win_frame = ttk.LabelFrame(left, text="🖥️ Game Window", padding=8)
        win_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(1, weight=0)

        self.window_status_label = ttk.Label(win_frame, text="❌ No game window detected", foreground="red")
        self.window_status_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.detect_window_btn = ttk.Button(win_frame, text="Detect Window", command=self.detect_game_window)
        self.detect_window_btn.grid(row=1, column=0, sticky="ew")

        self.manual_region_btn = ttk.Button(win_frame, text="Set Manual Region", command=self.set_manual_region)
        self.manual_region_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        self.clear_region_btn = ttk.Button(win_frame, text="Clear Region", command=self.clear_window_region)
        self.clear_region_btn.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(win_frame, text="Use this if automatic detection fails", font=("Arial", 8), foreground="gray")\
            .grid(row=4, column=0, sticky="w", pady=(6, 0))

        # Bot actions
        actions = ttk.LabelFrame(left, text="🤖 Bot Actions", padding=8)
        actions.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(2, weight=0)

        self.test_prediction_btn = ttk.Button(actions, text="Test Prediction", command=self.test_prediction, state="disabled")
        self.test_prediction_btn.grid(row=0, column=0, sticky="ew")

        self.start_bot_btn = ttk.Button(actions, text="Start Bot", command=self.start_bot, state="disabled")
        self.start_bot_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.stop_bot_btn = ttk.Button(actions, text="Stop Bot", command=self.stop_bot, state="disabled")
        self.stop_bot_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # Settings
        settings = ttk.LabelFrame(left, text="⚙️ Settings", padding=8)
        settings.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(3, weight=1)  # allow this section to stretch to fill remaining left space if needed

        self.live_gamestate_var = tk.BooleanVar(value=False)
        self.live_gamestate_check = ttk.Checkbutton(
            settings, text="Live Gamestate Mode",
            variable=self.live_gamestate_var, command=self.toggle_live_gamestate_mode
        )
        self.live_gamestate_check.grid(row=0, column=0, columnspan=2, sticky="w")

        # Bot selection dropdown
        ttk.Label(settings, text="Bot Mode:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.bot_mode_var = tk.StringVar(value="bot1")
        self.bot_mode_combo = ttk.Combobox(
            settings, textvariable=self.bot_mode_var, 
            values=["bot1", "bot2", "bot3"], 
            state="readonly", width=8
        )
        self.bot_mode_combo.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self.bot_mode_combo.bind("<<ComboboxSelected>>", self.on_bot_mode_changed)
        
        # Refresh bot mode button
        self.refresh_bot_mode_btn = ttk.Button(
            settings, text="Refresh Bot Mode", 
            command=self.refresh_bot_mode, width=15
        )
        self.refresh_bot_mode_btn.grid(row=1, column=2, sticky="w", padx=(6, 0), pady=(6, 0))

        # Restart Live Tracking button
        self.restart_tracking_btn = ttk.Button(
            settings, text="Restart Live Tracking", 
            command=lambda: (self.stop_feature_tracking(), self.start_feature_tracking()), width=18
        )
        self.restart_tracking_btn.grid(row=2, column=2, sticky="w", padx=(6, 0), pady=(6, 0))

        # Diagnostics button
        self.diagnostics_btn = ttk.Button(
            settings, text="Diagnostics", 
            command=self.print_paths_and_status, width=15
        )
        self.diagnostics_btn.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

        # Raw features button
        self.raw_features_btn = ttk.Button(
            settings, text="Raw Features", 
            command=self.show_raw_features, width=15
        )
        self.raw_features_btn.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        
        # Debug live features button
        self.debug_live_btn = ttk.Button(
            settings, text="Debug Live", 
            command=self.debug_live_features, width=15
        )
        self.debug_live_btn.grid(row=2, column=4, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(settings, text="Action Interval (s):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.action_interval_var = tk.StringVar(value="3.0")
        self.action_interval_entry = ttk.Entry(settings, textvariable=self.action_interval_var, width=10)
        self.action_interval_entry.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(settings, text="Max Duration (s):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.max_duration_var = tk.StringVar(value="60")
        self.max_duration_entry = ttk.Entry(settings, textvariable=self.max_duration_var, width=10)
        self.max_duration_entry.grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        # ---- RIGHT TOP: Live View ----
        view = ttk.LabelFrame(right_panes, text="📸 Live View", padding=8)
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)  # image expands
        view.rowconfigure(1, weight=0)

        self.screenshot_label = ttk.Label(view, text="No screenshot available", background="black", foreground="white")
        self.screenshot_label.grid(row=0, column=0, sticky="nsew")

        self.screenshot_info_label = ttk.Label(view, text="", font=("Arial", 10))
        self.screenshot_info_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        right_panes.add(view, weight=3)  # give live view more weight than logs

        # ---- RIGHT BOTTOM: Tabbed Interface (Bot Logs | Live Feature Tracking) ----
        tabbed_frame = ttk.Frame(right_panes)
        right_panes.add(tabbed_frame, weight=1)

        # Create notebook for tabs
        self.logs_notebook = ttk.Notebook(tabbed_frame)
        self.logs_notebook.pack(fill=tk.BOTH, expand=True)

        # ---- TAB 1: Bot Logs ----
        log_frame = ttk.Frame(self.logs_notebook)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", borderwidth=0)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure text tags for formatting
        self.log_text.tag_configure("highlighted", 
                                   font=("Arial", 9, "bold"), 
                                   foreground="#0066cc")  # Blue color

        self.logs_notebook.add(log_frame, text="📝 Bot Logs")

        # ---- TAB 2: Live Feature Tracking ----
        feature_tracking_frame = ttk.Frame(self.logs_notebook)
        feature_tracking_frame.columnconfigure(0, weight=1)
        feature_tracking_frame.rowconfigure(1, weight=1)  # table grows

        # Top controls frame
        controls_frame = ttk.Frame(feature_tracking_frame)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        controls_frame.columnconfigure(8, weight=1)  # spacer

        # Control buttons
        ttk.Button(controls_frame, text="📋 Copy Table to Clipboard", 
                  command=self._copy_live_table_to_clipboard).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls_frame, text="💾 Export to CSV", 
                  command=self._export_live_table_to_csv).grid(row=0, column=1, padx=(0, 6))
        
        # Show Hash Translations toggle
        self.show_translations_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="🔍 Show Hash Translations", 
                       variable=self.show_translations_var).grid(row=0, column=2, padx=(0, 6))
        
        # Show Normalized Data toggle (no-op for now)
        self.show_normalized_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="📊 Show Normalized Data", 
                       variable=self.show_normalized_var, state="disabled").grid(row=0, column=3, padx=(0, 6))
        
        # Filter by Feature Group
        ttk.Label(controls_frame, text="Filter:").grid(row=0, column=4, padx=(6, 0))
        self.feature_group_filter = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(controls_frame, textvariable=self.feature_group_filter, 
                                   values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", 
                                          "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
                                   state="readonly", width=12)
        filter_combo.grid(row=0, column=5, padx=(0, 6))
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_live_table())
        
        # Search box
        ttk.Label(controls_frame, text="🔍 Search:").grid(row=0, column=6, padx=(6, 0))
        self.feature_search_var = tk.StringVar()
        self.feature_search_var.trace("w", lambda *args: self._filter_live_table())
        search_entry = ttk.Entry(controls_frame, textvariable=self.feature_search_var, width=15)
        search_entry.grid(row=0, column=7, padx=(0, 6))

        # Summary line above table
        self.live_feature_summary = ttk.Label(feature_tracking_frame, text="", font=("Arial", 9))
        self.live_feature_summary.grid(row=1, column=0, sticky="w", padx=8, pady=(4, 0))

        # Create Treeview for table display
        table_frame = ttk.Frame(feature_tracking_frame)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Create Treeview with scrollbars
        self.live_feature_tree = ttk.Treeview(table_frame, show="headings", height=20)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.live_feature_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.live_feature_tree.xview)
        self.live_feature_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.live_feature_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Configure columns
        self.live_feature_tree["columns"] = ["Feature", "Index", "Timestep 0", "Timestep 1", "Timestep 2", "Timestep 3", "Timestep 4", 
                                           "Timestep 5", "Timestep 6", "Timestep 7", "Timestep 8", "Timestep 9"]
        
        # Set column headings
        for col in self.live_feature_tree["columns"]:
            self.live_feature_tree.heading(col, text=col)
            if col == "Feature":
                self.live_feature_tree.column(col, width=200, minwidth=150)
            elif col == "Index":
                self.live_feature_tree.column(col, width=50, minwidth=50)
            else:
                self.live_feature_tree.column(col, width=100, minwidth=80)
        
        # Bind tooltip events
        self.live_feature_tree.bind('<Motion>', self.on_live_table_motion)
        self.live_feature_tree.bind('<Leave>', self.on_live_table_leave)
        
        # Tooltip variables
        self.live_tooltip = None
        self.live_tooltip_text = ""
        
        # Buffer status label
        self.buffer_status_label = ttk.Label(feature_tracking_frame, text="Live Features: Ready | Source: unknown | Last Update: —", font=("Arial", 9))
        self.buffer_status_label.grid(row=3, column=0, sticky="w", padx=8, pady=(4, 0))
        
        # Initialize rolling buffer for 10 timesteps
        self.live_feature_buffer = []
        self.live_feature_names = []
        self.live_feature_mappings = []
        self.live_id_mappings = {}
        
        # Load feature mappings and names
        self._load_live_feature_mappings()
        
        # Set up periodic refresh for real-time updates (every 500ms)
        self._setup_periodic_refresh()

        self.logs_notebook.add(feature_tracking_frame, text="📊 Live Feature Tracking")
        
        # ---- TAB 3: Predictions ----
        preds_frame = ttk.Frame(self.logs_notebook)
        preds_frame.columnconfigure(0, weight=1)
        preds_frame.rowconfigure(1, weight=1)  # table grows

        # Header row with controls
        header = ttk.Frame(preds_frame)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.columnconfigure(5, weight=1)  # spacer

        ttk.Label(header, text="Model Predictions", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.pred_status_label = ttk.Label(header, text="Waiting for buffer…")
        self.pred_status_label.grid(row=0, column=1, padx=(10,0))

        # Prediction and tracking toggles
        self.pred_enable_var = tk.BooleanVar(value=True)     # Run predictions
        self.track_input_var = tk.BooleanVar(value=False)    # Track my input for action frames

        ttk.Checkbutton(header, text="Run predictions", variable=self.pred_enable_var,
                        command=lambda: (self.start_inference() if self.pred_enable_var.get() else self._disable_predictions())).grid(row=0, column=2, padx=(12,0))

        ttk.Checkbutton(header, text="Track my input", variable=self.track_input_var,
                        command=lambda: (self.start_action_tracking() if self.track_input_var.get() else self.stop_action_tracking())).grid(row=0, column=3, padx=(12,0))

        # Control buttons
        self.auto_exec_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="Auto-execute", variable=self.auto_exec_var).grid(row=0, column=4, padx=(12,0))
        ttk.Button(header, text="Clear", command=self._clear_prediction_table).grid(row=0, column=5, padx=(12,0))
        ttk.Button(header, text="Export CSV", command=self._export_predictions_csv).grid(row=0, column=6, padx=(6,0))
        ttk.Button(header, text="Dump Frames", command=self._dump_last_10_frames).grid(row=0, column=7, padx=(6,0))

        # Table
        cols = ("#","timing","type","x","y","button","key","scroll_dx","scroll_dy")
        self.pred_table = ttk.Treeview(preds_frame, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (40,80,60,60,60,70,70,80,80)):
            self.pred_table.heading(c, text=c)
            self.pred_table.column(c, width=w, anchor="center")
        self.pred_table.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))

        # Footer (hint + totals)
        footer = ttk.Frame(preds_frame)
        footer.grid(row=2, column=0, sticky="ew", padx=8, pady=(0,8))
        self.pred_footer = ttk.Label(footer, text="0 actions")
        self.pred_footer.grid(row=0, column=0, sticky="w")
        ttk.Label(footer, text="Tip: double-click a row to preview on Live View").grid(row=0, column=1, sticky="e")

        self.logs_notebook.add(preds_frame, text="🤖 Predictions")
        
        # Bind double-click for action preview
        def _on_pred_row_dclick(event):
            sel = self.pred_table.selection()
            if not sel:
                return
            vals = self.pred_table.item(sel[0], "values")
            # vals: "#, timing, type, x, y, button, key, scroll_dx, scroll_dy"
            try:
                x, y = int(vals[3]), int(vals[4])
                # TODO: preview cursor marker in the Live View, if desired
                self.status_label.config(text=f"Preview action → type:{vals[2]} xy:({x},{y})")
            except Exception:
                pass

        self.pred_table.bind("<Double-1>", _on_pred_row_dclick)

        # ---- Status bar ----
        status = ttk.Frame(main, padding=(0, 6, 0, 0))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(status, text="Ready", font=("Arial", 12))
        self.status_label.grid(row=0, column=0, sticky="w")

        self.action_count_label = ttk.Label(status, text="Actions: 0")
        self.action_count_label.grid(row=0, column=1, padx=(20, 0))

        self.runtime_label = ttk.Label(status, text="Runtime: 0s")
        self.runtime_label.grid(row=0, column=2, padx=(20, 0))

        self.bot_mode_label = ttk.Label(status, text="Bot: bot1")
        self.bot_mode_label.grid(row=0, column=3, padx=(20, 0))

        # ---- Initial sash positions (after layout) ----
        self.root.update_idletasks()

        # Set left/right sash so left controls are visible but narrow
        try:
            body_panes.sashpos(0, 340)
        except Exception:
            pass

        # One-time placement for the vertical sash so logs are visible
        _placed = {"done": False}
        def _place_vertical_sash_once(_evt=None):
            if _placed["done"]:
                return
            right_panes.update_idletasks()
            h = right_panes.winfo_height()
            if h > 0:
                try:
                    # Put divider so Live View gets most height, but logs keep ~220px minimum
                    right_panes.sashpos(0, max(int(h * 0.65), h - 220))
                except Exception:
                    pass
                _placed["done"] = True

        right_panes.bind("<Configure>", _place_vertical_sash_once)


    # update_dashboard method removed - it relied on non-existent label widgets
    
    # update_dashboard_simple method removed - it relied on non-existent feature_display widget


    def feature_tracking_loop(self):
        """Main loop for feature tracking - runs every 600ms."""
        if not self.feature_tracking_active:
            return
        
        try:
            # Get latest gamestate
            if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
                gs = self.live_feature_extractor.get_latest_gamestate()
                
                if gs:
                    # Debug: log gamestate info
                    if not hasattr(self, '_gs_debug_logged'):
                        self.log(f"🔍 First gamestate received: {list(gs.keys())}")
                        self._gs_debug_logged = True
                    
                    # Extract features using the exact same logic as training
                    vec = self.live_feature_extractor.extract_live_features(gs)
                    
                    if vec is not None and len(vec) == 128:
                        # Debug: log feature extraction info
                        if not hasattr(self, '_vec_debug_logged'):
                            non_zero_count = int(np.count_nonzero(vec))
                            self.log(f"🔍 Features extracted: {non_zero_count}/128 non-zero values")
                            self.log(f"🔍 First few values: {vec[:10]}")
                            self._vec_debug_logged = True
                        
                        # Add to rolling buffer (maintain 10 timesteps)
                        # Ensure vec is a proper numpy array with correct shape
                        if isinstance(vec, np.ndarray):
                            vec_copy = vec.copy().astype(np.float32)
                        else:
                            vec_copy = np.array(vec, dtype=np.float32)
                        
                        # Validate the vector has correct length
                        if len(vec_copy) != 128:
                            self.log(f"⚠️ Invalid feature vector length: {len(vec_copy)}, expected 128")
                            return
                        
                        self.live_feature_buffer.append(vec_copy)
                        
                        # Keep only last 10 timesteps
                        if len(self.live_feature_buffer) > 10:
                            self.live_feature_buffer = self.live_feature_buffer[-10:]
                        
                        # Update buffer status
                        source_info = self.live_feature_extractor.get_data_source_info()
                        source_mode = source_info[0] if source_info else "unknown"
                        timestamp = gs.get('timestamp', 0)
                        if timestamp > 0:
                            time_str = datetime.fromtimestamp(timestamp / 1000).strftime('%H:%M:%S')
                        else:
                            time_str = '—'
                        
                        status_text = f"Live Features: Active | Source: {source_mode} | Last Update: {time_str} | Buffer: {len(self.live_feature_buffer)}/10"
                        self.buffer_status_label.config(text=status_text)
                        
                        # Schedule table update on main thread
                        print(f"[DEBUG] 🔄 Scheduling table update, buffer has {len(self.live_feature_buffer)} timesteps")
                        self.root.after(0, lambda: self._update_live_feature_table())
                        
                        # Also update action buffer for prediction
                        if len(self.action_buf) < 10:
                            self.action_buf.append(np.zeros((101, 8), dtype=np.float32))
                        
                        # Keep only last 10 action frames
                        if len(self.action_buf) > 10:
                            self.action_buf = self.action_buf[-10:]
                        
                        # Update action buffer status
                        if hasattr(self, 'pred_status_label'):
                            self.pred_status_label.config(text=f"Buffer: {len(self.action_buf)}/10 frames")
                        
                        # Log buffer status periodically
                        if len(self.live_feature_buffer) % 10 == 0:
                            self.log(f"📊 Feature buffer: {len(self.live_feature_buffer)}/10 timesteps")
                    
                    else:
                        # No gamestate available
                        print(f"[DEBUG] ❌ No valid feature vector extracted")
                        status_text = "Live Features: Waiting for data | Source: unknown | Last Update: — | Buffer: 0/10"
                        self.buffer_status_label.config(text=status_text)
                else:
                    # No feature extractor
                    print(f"[DEBUG] ❌ No gamestate received")
                    status_text = "Live Features: No extractor | Source: unknown | Last Update: — | Buffer: 0/10"
                    self.buffer_status_label.config(text=status_text)
                
        except Exception as e:
            print(f"[DEBUG] ❌ Exception in feature_tracking_loop: {e}")
            self.log(f"❌ Feature tracking error: {e}")
            status_text = f"Live Features: Error | Source: unknown | Last Update: — | Buffer: {len(self.live_feature_buffer)}/10"
            self.buffer_status_label.config(text=status_text)
        
        # Schedule next iteration
        if self.feature_tracking_active:
            self.root.after(600, self.feature_tracking_loop)
    
    def _update_buffer_status(self, feat_count: int, action_count: int, timestamp):
        """Update the buffer status label."""
        try:
            if hasattr(self, 'buffer_status_label'):
                # Get source info
                source = "unknown"
                if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
                    source = getattr(self.live_feature_extractor, '_source_mode', 'unknown')
                
                # Format timestamp
                if timestamp and timestamp != '-':
                    try:
                        ts_int = int(timestamp)
                        now_ms = int(time.time() * 1000)
                        age_ms = now_ms - ts_int
                        if age_ms < 1000:
                            age_str = f"{age_ms}ms ago"
                        elif age_ms < 60000:
                            age_str = f"{age_ms//1000}s ago"
                        else:
                            age_str = f"{age_ms//60000}m ago"
                    except:
                        age_str = "—"
                else:
                    age_str = "—"
                
                status_text = f"Live Features: Active | Source: {source} | Buffers: {feat_count}/10 features, {action_count}/10 actions | Last Update: {age_str}"
                self.buffer_status_label.config(text=status_text)
        except Exception as e:
            # Silently fail if label doesn't exist
            pass
    
    def feed_inference_buffers(self, gamestate: Dict, features: np.ndarray):
        """Feed the inference buffers with new gamestate and action data."""
        try:
            # Ensure features is a 128-D vector
            if isinstance(features, np.ndarray) and features.shape == (128,):
                self.gs_buf.append(features.astype(np.float32))
            else:
                # Fallback: create empty feature vector
                self.gs_buf.append(np.zeros(128, dtype=np.float32))

            try:
                # Use the new action helpers from LiveFeatureExtractor
                frame = self.live_feature_extractor.build_action_step(gamestate)
            except Exception as e:
                self.log(f"⚠️ Action frame build failed, using empty: {e}")
                frame = np.zeros((101, 8), dtype=np.float32)

            self.action_buf.append(frame.astype(np.float32))
            
            # Log buffer sizes after append
            self.log(f"Buffers: features {len(self.gs_buf)}/10, actions {len(self.action_buf)}/10")
            
            # Add stdout print to show buffer growth
            print(f"[BUF] features={len(self.gs_buf)} actions={len(self.action_buf)}")
            
        except Exception as e:
            self.log(f"❌ Error feeding inference buffers: {e}")
            # Add fallback data to keep buffers moving
            if len(self.gs_buf) < 10:
                self.gs_buf.append(np.zeros(128, dtype=np.float32))
            if len(self.action_buf) < 10:
                self.action_buf.append(np.zeros((101, 8), dtype=np.float32))


    def render_features_on_canvas(self, event=None):
        """Legacy method - now updates the live feature table instead."""
        # This method is kept for compatibility but now updates the table
        if (hasattr(self, 'live_feature_tree') and 
            self.live_feature_buffer and 
            len(self.live_feature_buffer) > 0 and
            all(isinstance(vec, (list, np.ndarray)) and len(vec) == 128 for vec in self.live_feature_buffer)):
            self._update_live_feature_table()

    def _draw_top_status_strip(self, c, w, gs):
        """Legacy method - no longer used with table-based system."""
        pass

    def _format_coords(self, x, y, plane=None):
        """Format coordinates safely."""
        try:
            if x is not None and y is not None:
                if plane is not None:
                    return f"{x:.0f}, {y:.0f}, {plane:.0f}"
                else:
                    return f"{x:.0f}, {y:.0f}"
            return "—"
        except:
            return "—"

    def _format_animation(self, anim):
        """Format animation safely."""
        if not anim:
            return "—"
        # Try to map animation ID to name if possible
        if isinstance(anim, (int, float)) and anim > 0:
            # Could add animation ID mapping here
            return f"ID:{anim}"
        return str(anim)

    def _format_hp(self, hp):
        """Format health points safely."""
        if hp is None or hp == -1:
            return "—"
        try:
            return f"{int(hp)}"
        except:
            return "—"

    def _format_energy(self, energy):
        """Format run energy safely."""
        if energy is None:
            return "—"
        try:
            return f"{int(energy)}%"
        except:
            return "—"

    def _format_movement(self, movement):
        """Format movement state safely."""
        if not movement:
            return "idle"
        movement_str = str(movement).lower()
        if "run" in movement_str:
            return "running"
        elif "walk" in movement_str:
            return "walking"
        return "idle"

    def _truncate_string(self, s, max_len):
        """Truncate string with ellipsis if too long."""
        if not s:
            return "—"
        s_str = str(s)
        if len(s_str) <= max_len:
            return s_str
        return s_str[:max_len-3] + "..."

    def _get_top_inventory_items(self, gs):
        """Get top 5 inventory items with counts."""
        try:
            inventory = gs.get('inventory', [])
            if not inventory:
                return "Empty"
            
            # Count items by ID
            item_counts = {}
            for item in inventory:
                if item and item.get('id', -1) != -1:
                    item_id = item['id']
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
            
            if not item_counts:
                return "Empty"
            
            # Get top 5 items
            top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Format as "name × count" or "ID × count"
            formatted = []
            for item_id, count in top_items:
                # Try to get name from inventory
                item_name = None
                for item in inventory:
                    if item and item.get('id') == item_id:
                        item_name = item.get('name')
                        break
                
                if item_name:
                    formatted.append(f"{item_name} × {count}")
                else:
                    formatted.append(f"ID:{item_id} × {count}")
            
            result = ", ".join(formatted)
            if len(item_counts) > 5:
                result += f" (+{len(item_counts)-5} more)"
            
            return result
            
        except Exception:
            return "—"

    def _format_nearby_objects(self, gs):
        """Format nearby objects (first 3 with distances)."""
        try:
            game_objects = gs.get('game_objects', [])
            objects = []
            
            # Sort by distance and take first 3
            sorted_objects = sorted(game_objects, key=lambda obj: obj.get('distance', float('inf')))
            
            for obj in sorted_objects[:3]:
                if isinstance(obj, dict):
                    name = obj.get('name', f"ID:{obj.get('id', '?')}")
                    distance = obj.get('distance')
                    if distance is not None:
                        objects.append(f"{name} ({distance:.1f})")
                    else:
                        objects.append(name)
            
            if not objects:
                return "—"
            
            # Return first 3 with +N more if needed
            result = ", ".join(objects[:3])
            if len(sorted_objects) > 3:
                result += f" (+{len(sorted_objects)-3} more)"
            
            return result
            
        except Exception:
            return "—"

    def _format_nearby_npcs(self, gs):
        """Format nearby NPCs (first 3 with distances)."""
        try:
            npcs = gs.get('npcs', [])
            
            if not npcs:
                return "—"
            
            # Sort by distance and take first 3
            sorted_npcs = sorted(npcs, key=lambda npc: npc.get('distance', float('inf')))
            
            # Format first 3 NPCs
            formatted = []
            for npc in sorted_npcs[:3]:
                if isinstance(npc, dict):
                    name = npc.get('name', f"ID:{npc.get('id', '?')}")
                    distance = npc.get('distance')
                    if distance is not None:
                        formatted.append(f"{name} ({distance:.1f})")
                    else:
                        formatted.append(name)
            
            result = ", ".join(formatted)
            if len(sorted_npcs) > 3:
                result += f" (+{len(sorted_npcs)-3} more)"
            
            return result
            
        except Exception:
            return "—"

    def _format_nearby_furnaces(self, gs):
        """Format nearby furnaces (first 3 with distances)."""
        try:
            furnaces = gs.get('furnaces', [])
            
            if not furnaces:
                return "—"
            
            # Format first 3 furnaces
            formatted = []
            for furnace in furnaces[:3]:
                if isinstance(furnace, dict):
                    name = furnace.get('name', f"ID:{furnace.get('id', '?')}")
                    distance = furnace.get('distance')
                    if distance is not None:
                        formatted.append(f"{name} ({distance:.1f})")
                    else:
                        formatted.append(name)
            
            result = ", ".join(formatted)
            if len(furnaces) > 3:
                result += f" (+{len(furnaces) - 3} more)"
            
            return result
            
        except Exception:
            return "—"

    def _format_timestamp(self, timestamp):
        """Format timestamp safely."""
        if not timestamp:
            return "—"
        try:
            # Convert to readable time
            dt = datetime.fromtimestamp(timestamp / 1000)  # Assume milliseconds
            return dt.strftime("%H:%M:%S")
        except:
            return str(timestamp)

    def _format_delta_ms(self, timestamp):
        """Format delta milliseconds since last tick."""
        if not timestamp:
            return "—"
        try:
            now_ms = int(time.time() * 1000)
            delta = now_ms - timestamp
            if delta < 1000:
                return f"{delta}ms"
            elif delta < 60000:
                return f"{delta//1000}s"
            else:
                return f"{delta//60000}m"
        except:
            return "—"

    def _get_live_json_basename(self):
        """Get basename of live JSON file."""
        try:
            if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
                path = self.live_feature_extractor.gamestate_file
                return path.name if path else "—"
            return "—"
        except:
            return "—"

    def _get_rolling_dir_basename(self):
        """Get basename of rolling directory."""
        try:
            if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
                path = self.live_feature_extractor.gamestates_dir
                return path.name if path else "—"
            return "—"
        except:
            return "—"
    
    def _get_inventory_summary(self, gs):
        """Get inventory summary from gamestate."""
        try:
            inventory = gs.get('inventory', [])
            if not inventory:
                return "Empty"
            
            # Count non-empty slots
            count = sum(1 for item in inventory if item.get('id', -1) != -1)
            if count == 0:
                return "Empty"
            
            # Show first non-empty item name
            for item in inventory:
                if item.get('id', -1) != -1:
                    name = item.get('name', 'Unknown')
                    return f"{count} items ({name})"
            
            return f"{count} items"
        except Exception:
            return "Empty"
    
    def _get_last_action(self, gs):
        """Get last action from gamestate."""
        try:
            # Try different possible paths for last action
            last_action = (gs.get('last_interaction', {}) or 
                          gs.get('last_action', {}) or 
                          gs.get('action', {}))
            
            if isinstance(last_action, dict):
                action_type = last_action.get('action') or last_action.get('type')
                item_name = last_action.get('item_name')
                
                if action_type and item_name:
                    return f"{action_type} ({item_name})"
                elif action_type:
                    return str(action_type)
            
            return "—"
        except Exception:
            return "—"
    
    def clear_dashboard(self):
        """Clear the live feature dashboard."""
        try:
            # Clear the feature buffer
            if hasattr(self, 'live_feature_buffer'):
                self.live_feature_buffer.clear()
            
            # Clear the action buffer
            if hasattr(self, 'action_buf'):
                self.action_buf.clear()
            
            # Update status
            if hasattr(self, 'buffer_status_label'):
                status_text = "Live Features: Cleared | Source: unknown | Last Update: — | Buffer: 0/10"
                self.buffer_status_label.config(text=status_text)
            
            self.log("🧹 Live feature dashboard cleared")
            
        except Exception as e:
            self.log(f"❌ Error clearing dashboard: {e}")

    def _rect(self, c, x1, y1, x2, y2, fill, outline=None, width=1):
        """Legacy method - no longer used with table-based system."""
        pass

    def _text(self, c, x, y, text, anchor="nw", fill=None, font=None):
        """Legacy method - no longer used with table-based system."""
        pass

    def _kv_row(self, c, x, y, label, value, lh=18, label_w=150, val_w=300, status=None):
        """Legacy method - no longer used with table-based system."""
        return y + lh

    def _group_panel(self, c, x, y, w, h, title):
        """
        Draw group background + header strip and return (content_x, content_y, content_w, header_bottom_y).
        """
        L = self.FEAT_LAYOUT; S = self.FEAT_STYLE
        # Legacy method - no longer used with table-based system
        return x + 12, y + 22, w - 24, y + 22

    def _bar(self, c, x, y, w, h, frac, bg="#1F2937", fg="#22C55E"):
        """Legacy method - no longer used with table-based system."""
        pass

    def _render_inventory_columns(self, c, cx, cy, cw, ch, top_items, all_counts):
        """Legacy method - no longer used with table-based system."""
        pass

    def _render_inventory_single_column(self, c, cx, cy, cw, ch, top_items, all_counts):
        """Legacy method - no longer used with table-based system."""
        pass

    def _ellipsis(self, s, font, max_px):
        """Truncate text with ellipsis to fit within max_px width"""
        if font.measure(s) <= max_px: 
            return s
        # binary chop to fit
        lo, hi = 0, len(s)
        best = s
        while lo < hi:
            mid = (lo + hi) // 2
            t = s[:mid] + "…"
            if font.measure(t) <= max_px:
                lo = mid + 1
                best = t
            else:
                hi = mid
        return best

    def _wrap_text_lines(self, text: str, font_obj, max_px: int) -> list[str]:
        """Greedy wrap text into lines that fit within max_px (no hyphenation)."""
        words = str(text).split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font_obj.measure(trial) <= max_px or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def _draw_wrapped(self, c, x, y, text, max_px, line_h, fill, font_obj):
        """Legacy method - no longer used with table-based system."""
        return y + line_h, 1

    def _compute_layout(self, groups, canvas_w, canvas_h, pad, gap_group, min_col_w):
        """Deprecated selector (kept for compatibility). We now pick columns after measuring."""
        n = len(groups)
        # Return a placeholder; real values will be computed later.
        cols = min(max(2, canvas_w // max(min_col_w, 280)), min(5, n)) or 2
        col_w = max(min_col_w, int((canvas_w - 2*pad - (cols-1)*gap_group) / max(1, cols)))
        rows = (n + cols - 1) // max(1, cols)
        return cols, col_w, rows

    def log(self, message: str):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.root.after(0, lambda: self._enqueue_log_text(log_message))
    
    def log_with_formatting(self, prefix: str, highlighted_text: str, suffix: str):
        """Add message to log with highlighted text in the middle."""
        timestamp = time.strftime("%H:%M:%S")
        
        def _do():
            # Insert the prefix
            self._enqueue_log_text(f"[{timestamp}] {prefix}")
            start_pos = self.log_text.index(tk.END + "-1c")
            
            # Insert the highlighted text with formatting
            self._enqueue_log_text(highlighted_text)
            end_pos = self.log_text.index(tk.END + "-1c")
            
            # Apply formatting to the highlighted text
            self.log_text.tag_add("highlighted", start_pos, end_pos)
            
            # Insert the suffix
            self._enqueue_log_text(suffix + "\n")
            self.log_text.see(tk.END)
        
        self.root.after(0, _do)
    
    def clear_log(self):
        """Clear the log."""
        self.log_text.delete(1.0, tk.END)
    
    def on_bot_mode_changed(self, event=None):
        """Handle bot mode selection change."""
        selected_bot = self.bot_mode_var.get()
        self.log(f"🤖 Bot mode changed to: {selected_bot}")
        
        # Update status bar
        self.bot_mode_label.config(text=f"Bot: {selected_bot}")
        
        # Update the live feature extractor if it exists
        if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
            self.live_feature_extractor = LiveFeatureExtractor(selected_bot)
            self.log(f"   📁 Updated feature extractor to use: {selected_bot}")
        
        # Stop feature tracking if active (since bot mode changed)
        if self.feature_tracking_active:
            self.stop_feature_tracking()
            self.log("   ⏹️  Feature tracking stopped due to bot mode change")
        
        # Log the expected folder path
        bot_folder = f"data/{selected_bot}"
        self.log(f"   📂 Will read gamestate from: {bot_folder}/runelite_gamestate.json")
    
    def refresh_bot_mode(self):
        """Manually refresh the bot mode and reinitialize feature extractor."""
        selected_bot = self.bot_mode_var.get()
        self.log(f"🔄 Refreshing bot mode: {selected_bot}")
        
        # Check if bot folder exists
        bot_folder = Path(f"data/{selected_bot}")
        gamestate_file = bot_folder / "runelite_gamestate.json"
        
        if bot_folder.exists():
            self.log(f"   ✅ Bot folder exists: {bot_folder}")
            if gamestate_file.exists():
                self.log(f"   ✅ Gamestate file found: {gamestate_file}")
                # Check file size and modification time
                file_size = gamestate_file.stat().st_size
                mod_time = time.ctime(gamestate_file.stat().st_mtime)
                self.log(f"   📊 File size: {file_size} bytes, Last modified: {mod_time}")
            else:
                self.log(f"   ⚠️  Gamestate file not found: {gamestate_file}")
                self.log("   💡 Make sure RuneLite plugin is set to the same bot mode")
        else:
            self.log(f"   ❌ Bot folder not found: {bot_folder}")
            self.log("   💡 Make sure RuneLite plugin is set to the same bot mode")
        
        # Reinitialize live feature extractor
        self.live_feature_extractor = LiveFeatureExtractor(selected_bot)
        self.log(f"   📁 Reinitialized feature extractor for: {selected_bot}")
        self.log(f"   📂 Reading from: data/{selected_bot}/runelite_gamestate.json")
    
    def show_raw_features(self):
        """Show raw feature values in a compact format."""
        try:
            # Get current bot mode and create feature extractor if needed
            selected_bot = self.bot_mode_var.get()
            if not hasattr(self, 'live_feature_extractor') or not self.live_feature_extractor:
                self.live_feature_extractor = LiveFeatureExtractor(selected_bot)
                self.log(f"📁 Created feature extractor for: {selected_bot}")
            
            # Get latest gamestate first
            gamestate = self.live_feature_extractor.get_latest_gamestate()
            if gamestate is None:
                self.log("❌ No gamestate data available")
                return
            
            features = self.live_feature_extractor.extract_live_features(gamestate)
            if features is None:
                self.log("❌ Failed to extract features")
                return
            
            feature_names = self.live_feature_extractor.get_feature_names()
            
            self.log("�� Raw Feature Values (with mappings):")
            self.log("   Format: [index] name: mapped_value (raw_value) or raw_value")
            
            # Show all features in a compact format with mapped values
            for i, (name, value) in enumerate(zip(feature_names, features)):
                if i % 10 == 0:  # Add spacing every 10 features
                    self.log("")
                
                # Get mapped value for better readability
                mapped_value = self.live_feature_extractor.interpret_feature(name, value)
                
                # Format the output based on whether we have a meaningful mapping
                if mapped_value != f"{value:.3f}" and mapped_value != f"{int(value)}" and mapped_value != f"{value:.0f}ms":
                    # We have a meaningful mapping - show as "mapped_name (raw_value)" with formatting
                    self.log_with_formatting(f"   [{i:3d}] {name}: ", mapped_value, f" ({value:.0f})")
                else:
                    # No meaningful mapping - just show the raw value
                    self.log(f"   [{i:3d}] {name}: {value:.3f}")
            
            self.log("")
            self.log(f"📊 Total: {len(features)} features")
            
            # Also show the actual gamestate structure for debugging
            self.log("\n🔍 Current Gamestate Structure (top-level keys):")
            if gamestate:
                for key in sorted(gamestate.keys()):
                    value = gamestate[key]
                    if isinstance(value, (list, dict)):
                        size = len(value) if hasattr(value, '__len__') else '?'
                        self.log(f"   {key}: {type(value).__name__} ({size} items)")
                    else:
                        self.log(f"   {key}: {type(value).__name__} = {value}")
            
        except Exception as e:
            self.log(f"❌ Raw features display failed: {e}")
    
    def debug_live_features(self):
        """Debug live feature updates and display issues."""
        try:
            self.log("🔍 Debugging Live Feature Updates...")
            
            # Check if we have a live feature extractor
            if not hasattr(self, 'live_feature_extractor') or not self.live_feature_extractor:
                self.log("❌ No live feature extractor available")
                return
            
            # Check current data source
            source_info = self.live_feature_extractor.get_data_source_info()
            self.log(f"📡 Data source: {source_info}")
            
            # Check if monitoring is active
            is_monitoring = self.live_feature_extractor.is_monitoring()
            self.log(f"📡 Monitoring active: {is_monitoring}")
            
            # Get latest gamestate
            gamestate = self.live_feature_extractor.get_latest_gamestate()
            if gamestate:
                self.log(f"📊 Latest gamestate timestamp: {gamestate.get('timestamp', 'N/A')}")
                self.log(f"📊 Gamestate keys: {list(gamestate.keys())}")
                
                # Check specific fields that should be updating
                camera_fields = ['camera_x', 'camera_y', 'camera_z', 'camera_pitch', 'camera_yaw']
                for field in camera_fields:
                    value = gamestate.get(field, 'NOT_FOUND')
                    self.log(f"   📷 {field}: {value}")
                
                # Check player fields
                player = gamestate.get('player', {})
                if player:
                    self.log(f"   👤 Player fields: {list(player.keys())}")
                    self.log(f"   👤 Player coords: ({player.get('world_x', 'N/A')}, {player.get('world_y', 'N/A')})")
                
                # Check phase context
                phase = gamestate.get('phase_context', {})
                if phase:
                    self.log(f"   🔄 Phase fields: {list(phase.keys())}")
                    self.log(f"   🔄 Current phase: {phase.get('cycle_phase', 'N/A')}")
                
            else:
                self.log("❌ No gamestate data available")
            
            # Check feature tracking status
            self.log(f"📊 Feature tracking active: {getattr(self, 'feature_tracking_active', False)}")
            self.log(f"📊 Live gamestate mode: {getattr(self, 'live_gamestate_mode', False)}")
            
            # Check if canvas exists and has latest data
            if hasattr(self, 'live_feature_tree'):
                self.log(f"📊 Live feature table exists: {self.live_feature_tree is not None}")
                if hasattr(self, '_latest_gamestate'):
                    self.log(f"📊 Latest gamestate cached: {self._latest_gamestate is not None}")
                if hasattr(self, '_latest_features'):
                    self.log(f"📊 Latest features cached: {self._latest_features is not None}")
            
            # Force a table update to see if it works
            self.log("🎨 Forcing feature table update...")
            self._update_live_feature_table()
            self.log("✅ Table update completed")
            
        except Exception as e:
            self.log(f"❌ Debug failed: {e}")
    
    def toggle_live_gamestate_mode(self):
        """Toggle between screenshot mode and live gamestate mode."""
        if self.live_gamestate_var.get():
            self.live_gamestate_mode = True
            selected_bot = self.bot_mode_var.get()
            self.log("🔄 Switching to Live Gamestate Mode")
            self.log("   This mode reads real OSRS data from RuneLite plugin")
            self.log(f"   Using bot mode: {selected_bot}")
            self.log(f"   📁 Reading from: data/{selected_bot}/runelite_gamestate.json")
            
            # Initialize live feature extractor with selected bot mode
            self.live_feature_extractor = LiveFeatureExtractor(selected_bot)
            print("[INIT] live extractor ready")
            
            # Log absolute paths for live JSON and rolling folder
            live_json_path = self.live_feature_extractor.gamestate_file
            rolling_folder_path = self.live_feature_extractor.gamestates_dir
            self.log(f"📁 Live JSON: {live_json_path}")
            self.log(f"📁 Rolling folder: {rolling_folder_path}")
            
            # Print paths and status for the new bot mode
            self.print_paths_and_status()
            
            # Start gamestate monitoring thread
            if hasattr(self.live_feature_extractor, 'start_monitor'):
                self.live_feature_extractor.start_monitor()
                print("[GS] start")
            else:
                self.log("❌ start_monitor method not found on extractor")
            
            # Start the GUI's own gamestate thread
            self.start_gamestate_thread()
            
            # Start feature tracking automatically
            self.start_feature_tracking()
            
            print("[INIT] threads scheduled")
            
            # Update GUI
            self.live_gamestate_check.config(text="Live Gamestate Mode ✅")
            
        else:
            self.live_gamestate_mode = False
            self.log("📸 Switching to Screenshot Mode")
            
            # Stop gamestate monitoring
            if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
                if hasattr(self.live_feature_extractor, 'stop_monitor'):
                    self.live_feature_extractor.stop_monitor()
                    print("[GS] stop")
                else:
                    self.log("❌ stop_monitor method not found on extractor")
            
            # Stop gamestate thread if active
            if self.gamestate_thread and self.gamestate_thread.is_alive():
                self.gamestate_thread.join(timeout=1.0)
                self.gamestate_thread = None
                self.log("⏹️  Gamestate thread stopped")
            
            # Stop feature tracking if active
            if self.feature_tracking_active:
                self.stop_feature_tracking()
            
            # Update GUI
            self.live_gamestate_check.config(text="Live Gamestate Mode")
    

    
    # toggle_feature_tracking method removed - references undefined track_features_var
    
    def start_gamestate_thread(self):
        """Start the GUI's own gamestate thread."""
        if not self.live_feature_extractor:
            self.log("❌ No live feature extractor available")
            self.log("   Please enable Live Gamestate Mode first")
            return
        
        # Guard threads
        if self.gamestate_thread and self.gamestate_thread.is_alive():
            return
        
        try:
            self.log("🚀 Starting gamestate thread...")
            self.log("   Auto-updating every 500ms")
            
            # Start gamestate thread
            self.gamestate_thread = threading.Thread(
                target=self.gamestate_monitor_loop,
                daemon=True
            )
            self.gamestate_thread.start()
            
            self.log("✅ Gamestate thread started!")
            
        except Exception as e:
            self.log(f"❌ Error starting gamestate thread: {e}")
            messagebox.showerror("Error", f"Failed to start gamestate thread: {e}")

    def gamestate_monitor_loop(self):
        """Main loop for monitoring gamestate and updating GUI."""
        print("[GST] loop start")
        self.log("   📊 Gamestate monitor loop started (500ms interval)")
        
        last_heartbeat = 0.0
        
        while self.live_gamestate_mode and self.gamestate_thread and self.gamestate_thread.is_alive():
            try:
                # Fetch gamestate via extractor
                gs = self.live_feature_extractor.get_latest_gamestate()
                if not gs:
                    time.sleep(0.2)
                    continue
                
                # Extract features (pure transformation, no disk I/O)
                features = self.live_feature_extractor.extract_live_features(gs)
                if features is None:
                    continue
                
                # Build action step (pure transformation, no disk I/O)
                action_step = self.live_feature_extractor.build_action_step(gs)
                
                # Update live data
                self._latest_gamestate = gs
                self._latest_features = features
                
                # Feed inference buffers
                self.feed_inference_buffers(gs, features)
                
                # Heartbeat every ~3s
                now = time.time()
                if now - last_heartbeat > 3.0:
                    ts = gs.get("timestamp", "-")
                    source = getattr(self.live_feature_extractor, '_source_mode', 'unknown')
                    print(f"[GST] tick ts={ts} source={source}", flush=True)
                    last_heartbeat = now
                
                # Update status label with buffer counts (via main thread)
                timestamp = gs.get('timestamp', '-')
                self.root.after(0, lambda: self._update_buffer_status(len(self.gs_buf), len(self.action_buf), timestamp))
                
                # Schedule render (via main thread)
                # Schedule table update for real-time updates
                self.root.after(0, lambda: self._update_live_feature_table())
                
                # Wait for next update (500ms)
                time.sleep(0.5)
                
            except Exception as e:
                self.log(f"❌ Gamestate monitor error: {e}")
                time.sleep(1.0)
        
        print("[GST] loop stop")
        self.log("   📊 Gamestate monitor loop ended")

    def start_feature_tracking(self):
        """Start live feature tracking."""
        if self.feature_tracking_active:
            self.log("📊 Feature tracking already active")
            return
        
        try:
            # Initialize live feature extractor if not already done
            if not hasattr(self, 'live_feature_extractor') or not self.live_feature_extractor:
                self.live_feature_extractor = LiveFeatureExtractor("bot1")
                self.live_feature_extractor.start_monitor()
                self.log("🔧 Live feature extractor initialized")
            
            # Clear existing buffers
            self.live_feature_buffer.clear()
            self.action_buf.clear()
            
            # Initialize status
            self.feature_tracking_active = True
            status_text = "Live Features: Starting | Source: unknown | Last Update: — | Buffer: 0/10"
            self.buffer_status_label.config(text=status_text)
            
            # Start the tracking loop
            self.feature_tracking_loop()
            
            self.log("🚀 Starting live feature tracking...")
            self.log("   📊 Feature tracking active (600ms interval)")
            self.log("   📊 Table will update as new data arrives")
            
        except Exception as e:
            self.log(f"❌ Failed to start feature tracking: {e}")
            self.feature_tracking_active = False
    
    def stop_feature_tracking(self):
        """Stop live feature tracking."""
        if not self.feature_tracking_active:
            self.log("📊 Feature tracking not active")
            return
        
        try:
            # Stop the tracking
            self.feature_tracking_active = False
            
            # Update status
            status_text = "Live Features: Stopped | Source: unknown | Last Update: — | Buffer: 0/10"
            self.buffer_status_label.config(text=status_text)
            
            self.log("🛑 Live feature tracking stopped")
            self.log("   📊 Feature tracking inactive")
            
        except Exception as e:
            self.log(f"❌ Error stopping feature tracking: {e}")

    def cleanup(self):
        """Clean up resources when GUI is closed."""
        try:
            # Stop all threads
            self.feature_tracking_active = False
            self._infer_on = False
            
            # Wait for threads to finish
            if hasattr(self, 'feature_tracking_thread') and self.feature_tracking_thread:
                self.feature_tracking_thread.join(timeout=1.0)
            if hasattr(self, '_infer_thread') and self._infer_thread:
                self._infer_thread.join(timeout=1.0)
            if hasattr(self, 'screenshot_thread') and self.screenshot_thread:
                self.screenshot_thread.join(timeout=1.0)
            
            # Stop action tracking listeners
            self.stop_action_tracking()
                
            self.log("🧹 Cleanup completed")
        except Exception as e:
            self.log(f"❌ Error during cleanup: {e}")
    
    def start_inference(self):
        """Start the inference loop."""
        if getattr(self, "_infer_on", False):
            return
        self._infer_on = True
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()
        print("[INF] start")
        self.log("🚀 Inference loop started")
    
    def stop_inference(self):
        """Stop the inference loop."""
        self._infer_on = False
        print("[INF] stop")
        self.log("⏹️  Inference loop stopped")
    
    def _inference_loop(self):
        """Inference loop running every 600ms."""
        runner = ModelRunner.instance()
        period = 0.6  # 600 ms
        while getattr(self, "_infer_on", False):
            t0 = time.time()
            
            # Always update buffer status in UI
            buf_gs, buf_act = len(self.gs_buf), len(self.action_buf)
            self.root.after(0, lambda: self.pred_status_label.config(
                text=f"Buffer: {buf_gs}/10 features, {buf_act}/10 actions"))
            
            # Check if predictions are enabled
            if not self.pred_enable_var.get():
                time.sleep(0.6)
                continue
            
            if len(self.gs_buf) == 10 and len(self.action_buf) == 10:
                # Log when buffers first become ready
                if not hasattr(self, '_buffers_ready_logged'):
                    self.log("Buffers ready; starting predictions...")
                    self._buffers_ready_logged = True
                
                try:
                    # Build tensors with correct shapes
                    features_tensor = np.stack(list(self.gs_buf), axis=0).reshape(1, -1, 128)      # (1,10,128)
                    action_tensor = np.stack(list(self.action_buf), axis=0).reshape(1, -1, 101, 8)  # (1,10,101,8)
                    
                    # Call ModelRunner
                    out = runner.predict(features_tensor, action_tensor)  # expect (1,101,8)
                    
                    # Ensure output is numpy array
                    if hasattr(out, 'detach'):
                        out_np = out.detach().to("cpu").numpy()
                    else:
                        out_np = np.asarray(out)
                    
                    # Verify shape
                    if out_np.shape != (1, 101, 8):
                        self.log(f"⚠️ Unexpected prediction shape: {out_np.shape}, expected (1,101,8)")
                        # Still try to use it by reshaping if possible
                        if out_np.size == 808:  # 101*8
                            out_np = out_np.reshape(1, 101, 8)
                        else:
                            # Create a fallback prediction
                            out_np = np.zeros((1, 101, 8), dtype=np.float32)
                            out_np[0, 0, 0] = 1  # 1 action
                    
                    # Log successful prediction
                    count = int(np.clip(out_np[0, 0, 0], 0, 100))
                    if count > 0:
                        first_action = out_np[0, 1]  # first action row
                        action_type = int(first_action[1])
                        x, y = int(first_action[2]), int(first_action[3])
                        self.log(f"Pred: {count} actions | first type={action_type} x={x} y={y}")
                    
                except Exception as e:
                    self.log(f"❌ Inference error: {e}")
                    # Create fallback prediction
                    out_np = np.zeros((1, 101, 8), dtype=np.float32)
                    out_np[0, 0, 0] = 1  # 1 action

                # store + notify UI
                with self._pred_lock:
                    self._last_pred = out_np[0]  # (101,8)
                self.root.after(0, self._on_new_prediction)

            # sleep remainder of 600 ms tick
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
    
    def _on_new_prediction(self):
        """Handle new prediction from inference loop."""
        with self._pred_lock:
            pred = None if self._last_pred is None else self._last_pred.copy()
        if pred is None:
            self.pred_status_label.config(text="Waiting for buffer…")
            return

        # Parse
        count = int(np.clip(round(pred[0,0]), 0, 100))
        actions = pred[1:1+count]  # (N,8)
        self.predicted_actions = actions

        # Update status widgets
        self.action_count_label.config(text=f"Pred: {count} actions")
        self.pred_status_label.config(text=f"{count} actions")
        if count:
            a0 = actions[0]
            self.status_label.config(text=f"Last pred → type:{int(round(a0[1]))} xy:({int(round(a0[2]))},{int(round(a0[3]))})")
        self.log(f"🎯 New prediction: {count} actions")

        # Fill the table (clear and reinsert)
        for iid in self.pred_table.get_children():
            self.pred_table.delete(iid)

        for i, row in enumerate(actions, start=1):
            t, typ, x, y, btn, key, sdx, sdy = row.tolist()
            # Clamp coordinates to Live View bounds
            clamped_x, clamped_y = self._clamp_xy(x, y)
            self.pred_table.insert("", "end", values=(
                i, f"{t:.3f}", int(round(typ)), clamped_x, clamped_y,
                int(round(btn)), int(round(key)), int(round(sdx)), int(round(sdy))
            ))

        self.pred_footer.config(text=f"{count} actions   |   updated {time.strftime('%H:%M:%S')}")

        # Save to history
        self.pred_history.append({"timestamp": time.time(), "actions": actions.copy()})

        # Optional auto-execute hook
        if self.auto_exec_var.get():
            self._enqueue_actions_for_execution(actions)

    def _clear_prediction_table(self):
        for iid in self.pred_table.get_children():
            self.pred_table.delete(iid)
        self.pred_footer.config(text="0 actions")
        self.pred_status_label.config(text="Cleared")

    def _export_predictions_csv(self):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(f"predictions_{ts}.csv")
            with path.open("w", encoding="utf-8") as f:
                f.write("idx,timing,type,x,y,button,key,scroll_dx,scroll_dy\n")
                for iid in self.pred_table.get_children():
                    vals = self.pred_table.item(iid, "values")
                    f.write(",".join(map(str, vals)) + "\n")
            self.log(f"📄 Exported predictions to {path.resolve()}")
            self.pred_status_label.config(text=f"Exported to {path.name}")
        except Exception as e:
            self.log(f"❌ Export failed: {e}")
            self.pred_status_label.config(text="Export failed")

    def _enqueue_actions_for_execution(self, actions: np.ndarray):
        # Placeholder: integrate with your bot executor if desired.
        # Expect actions shape (N,8) with fields:
        # [timing, type, x, y, button, key, scroll_dx, scroll_dy]
        pass

    def _disable_predictions(self):
        """Turn off predictions and clear the UI."""
        self.stop_inference()
        # Clear table/summary
        if hasattr(self, "pred_table"):
            for iid in self.pred_table.get_children():
                self.pred_table.delete(iid)
        if hasattr(self, "pred_footer"):
            self.pred_footer.config(text="0 actions")
        if hasattr(self, "pred_status_label"):
            self.pred_status_label.config(text="Predictions disabled")

    def _pack_actions_frame(self, events: list[dict], now_ms: int) -> np.ndarray:
        """
        Convert a list of raw action dicts (latest tick) into a (101,8) frame
        with row[0,0] = count and rows 1..count = [timing_s, type, x, y, button, key, scroll_dx, scroll_dy].
        Only include actions with 0..600ms age relative to now_ms.
        """
        frame = np.zeros((101, 8), dtype=np.float32)
        if not events:
            return frame

        # keep only events in the last 600ms
        window_ms = 600
        filtered = []
        for a in events:
            ts = int(a.get("timestamp", now_ms))
            if 0 <= (now_ms - ts) <= window_ms:
                filtered.append(a)
        if not filtered:
            return frame

        # cap at 100
        filtered = filtered[:100]
        frame[0, 0] = float(len(filtered))

        for i, a in enumerate(filtered, start=1):
            # seconds since event within the window (nonnegative)
            t = 0.0 if not now_ms else max(0.0, (now_ms - float(a.get("timestamp", now_ms))) / 1000.0)
            # NOTE: keep these enums identical to training encodings
            typ   = float(a.get("type", 0))
            x     = float(a.get("x", 0))
            y     = float(a.get("y", 0))
            btn   = float(a.get("button", 0))
            key   = float(a.get("key", 0))
            sdx   = float(a.get("scroll_dx", 0))
            sdy   = float(a.get("scroll_dy", 0))

            frame[i, 0] = t
            frame[i, 1] = typ
            frame[i, 2] = x
            frame[i, 3] = y
            frame[i, 4] = btn
            frame[i, 5] = key
            frame[i, 6] = sdx
            frame[i, 7] = sdy

        return frame

    def _validate_action_buffers(self) -> tuple[bool, str]:
        """Validate that action buffers are ready and properly formatted for inference."""
        if len(self.action_buf) < 10:
            return False, f"need 10 action frames, have {len(self.action_buf)}"

        for t, frame in enumerate(self.action_buf):
            if frame.shape != (101, 8):
                return False, f"frame {t} has wrong shape {frame.shape}"
            cnt = int(round(float(frame[0,0])))
            if not (0 <= cnt <= 100):
                return False, f"frame {t} invalid count {cnt}"
            # rows after count must be zero
            if cnt < 100:
                tail = frame[1+cnt:]
                if np.any(tail):
                    return False, f"frame {t} has nonzero padding after row {cnt}"
        return True, "ok"

    def _update_pred_buffer_status(self):
        """Update the prediction tab status with buffer progress and per-tick counts."""
        f, a = len(self.gs_buf), len(self.action_buf)
        counts = []
        for fr in list(self.action_buf)[-10:]:
            if fr.shape == (101,8):
                counts.append(str(int(round(fr[0,0]))))
            else:
                counts.append("?")
        
        status_text = f"Buffer: {f}/10 features, {a}/10 actions  |  per-tick counts: [{', '.join(counts)}]"
        self.pred_status_label.config(text=status_text)

    def _clamp_xy(self, x, y):
        """Clamp x,y coordinates to Live View bounds."""
        # Default bounds if not set
        x0, y0, w, h = getattr(self, 'live_view_region', (0, 0, 800, 600))
        return (min(max(int(x), 0), w-1), min(max(int(y), 0), h-1))

    def _dump_last_10_frames(self):
        """Dump the last 10 frames for debugging comparison with training data."""
        try:
            if len(self.gs_buf) < 10 or len(self.action_buf) < 10:
                self.log("❌ Need 10 frames in both buffers to dump")
                return

            dump_data = {
                "temporal_sequence": [list(frame) for frame in list(self.gs_buf)[-10:]],
                "action_input_sequence": [frame.tolist() for frame in list(self.action_buf)[-10:]],
                "now_ms": int(time.time() * 1000)
            }

            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(f"debug_frames_{ts}.json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(dump_data, f, indent=2)
            
            self.log(f"📄 Dumped last 10 frames to {path.resolve()}")
            self.pred_status_label.config(text=f"Dumped to {path.name}")
        except Exception as e:
            self.log(f"❌ Dump failed: {e}")
            self.pred_status_label.config(text="Dump failed")

    def start_action_tracking(self):
        """Start tracking user input for action frames."""
        if not PYNPUT_AVAILABLE:
            self.log("⚠️ pynput not available; cannot track user input.")
            self.track_input_var.set(False)
            return
        if self._mouse_listener or self._key_listener:
            return
        
        # compute live-view origin for coord mapping
        self._lv_ox = self.screenshot_label.winfo_rootx()
        self._lv_oy = self.screenshot_label.winfo_rooty()
        self._lv_w  = self.screenshot_label.winfo_width()
        self._lv_h  = self.screenshot_label.winfo_height()

        def to_lv_xy(screen_x, screen_y):
            x = int(screen_x - self._lv_ox)
            y = int(screen_y - self._lv_oy)
            x = max(0, min(self._lv_w - 1, x))
            y = max(0, min(self._lv_h - 1, y))
            return x, y

        def on_move(x, y):
            # we don't record passive move to keep volume low
            return

        def on_click(x, y, button, pressed):
            if not pressed:
                return
            lx, ly = to_lv_xy(x, y)
            self.user_events.append({
                "timestamp": int(time.time() * 1000),
                "type": 1,  # TODO: map to your "click" type enum
                "x": lx, "y": ly,
                "button": 1 if str(button).endswith(".left") else 2,
                "key": 0, "scroll_dx": 0, "scroll_dy": 0
            })

        def on_scroll(x, y, dx, dy):
            lx, ly = to_lv_xy(x, y)
            self.user_events.append({
                "timestamp": int(time.time() * 1000),
                "type": 2,  # TODO: map to your "scroll" type enum
                "x": lx, "y": ly,
                "button": 0, "key": 0, "scroll_dx": int(dx), "scroll_dy": int(dy)
            })

        def on_press(key):
            code = 0
            try:
                code = getattr(key, "vk", 0) or getattr(key, "value", 0)
            except Exception:
                pass
            self.user_events.append({
                "timestamp": int(time.time() * 1000),
                "type": 3,  # TODO: map to your "key" type enum
                "x": 0, "y": 0,
                "button": 0, "key": int(code), "scroll_dx": 0, "scroll_dy": 0
            })

        self._mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self._key_listener = keyboard.Listener(on_press=on_press)
        self._mouse_listener.start(); self._key_listener.start()
        self.log("🟢 Started tracking user input for action frames")

    def stop_action_tracking(self):
        """Stop tracking user input."""
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._key_listener:
            self._key_listener.stop()
            self._key_listener = None
        self.user_events.clear()
        self.log("⚪ Stopped tracking user input")

    
    def log_live_gamestate_info(self, gamestate: Dict, features: np.ndarray):
        """Log key information from live gamestate."""
        try:
            # Player info
            player = gamestate.get('player', {})
            if player:
                self.log(f"   👤 Player: ({player.get('world_x', 0)}, {player.get('world_y', 0)}) - {player.get('animation_name', 'Unknown')}")
            
            # Phase info
            phase_context = gamestate.get('phase_context', {})
            if phase_context:
                phase = phase_context.get('cycle_phase', 'unknown')
                self.log(f"   🔄 Phase: {phase}")
            
            # Bank status
            bank_open = gamestate.get('bank_open', False)
            self.log(f"   🏦 Bank: {'Open' if bank_open else 'Closed'}")
            
            # Inventory count
            inventory = gamestate.get('inventory', [])
            non_empty_slots = sum(1 for item in inventory if item and item.get('id', -1) != -1)
            self.log(f"   📦 Inventory: {non_empty_slots}/28 slots filled")
            
        except Exception as e:
            self.log(f"   ❌ Error logging gamestate info: {e}")
    
    def check_model(self):
        """Check if model file exists."""
        model_path = r"D:\cursor_projects\osrs_learner\bot_runelite_IL\training_results\model_weights.pth"
        if Path(model_path).exists():
            self.log(f"✅ Model file found: {model_path}")
            self.model_status_label.config(text=f"✅ Model: {model_path}", foreground="green")
        else:
            self.log(f"❌ Model file not found: {model_path}")
            self.model_status_label.config(text=f"❌ Model: Not Found", foreground="red")
    
    def load_feature_mappings(self):
        """Load and cache feature mappings sorted by feature_index."""
        try:
            mappings_path = "data/features/feature_mappings.json"
            if Path(mappings_path).exists():
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                # Sort by feature_index to ensure correct order
                self._mappings_sorted = sorted(mappings, key=lambda x: x["feature_index"])
                self.log(f"✅ Loaded {len(self._mappings_sorted)} feature mappings")
                
                # Early warning if mappings count doesn't match expected
                if len(self._mappings_sorted) != 128:
                    self.log(f"⚠️ Expected 128 mappings, found {len(self._mappings_sorted)}")
            else:
                self.log(f"❌ Feature mappings not found: {mappings_path}")
                self._mappings_sorted = []
        except Exception as e:
            self.log(f"❌ Error loading feature mappings: {e}")
            self._mappings_sorted = []
    
    def load_model(self):
        try:
            model_path = r"D:\cursor_projects\osrs_learner\bot_runelite_IL\training_results\model_weights.pth"
            if not Path(model_path).exists():
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
            self.log("🔄 Loading model via ModelRunner ...")
            # Force ModelRunner init (loads weights on CUDA if available)
            _ = ModelRunner.instance()
            self.model_loaded = True
            self.status_label.config(text="Model loaded")
            self.log("✅ Model ready for inference (ModelRunner)")
            self.test_prediction_btn.config(state="normal")
            self.start_bot_btn.config(state="normal")
        except Exception as e:
            self.log(f"❌ Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.status_label.config(text="Model load failed")
    
    def detect_game_window(self):
        """Detect OSRS game window."""
        if not PYWINDOW_AVAILABLE:
            self.log("❌ pygetwindow library not available")
            self.log("   Cannot detect game windows automatically")
            self.log("   Please use 'Set Manual Region' button instead")
            self.window_detected = False
            return
        
        try:
            self.log("🔍 Detecting game window...")
            self.log("   Searching for OSRS-related windows...")
            
            # Look for common OSRS window titles - be more specific to avoid editor windows
            osrs_titles = [
                "RuneLite", "Old School RuneScape", "OSRS", 
                "RuneScape", "Jagex Launcher"
            ]
            
            self.log(f"   Checking for: {', '.join(osrs_titles)}")
            
            # Get all windows first
            try:
                all_windows = gw.getAllWindows()
                self.log(f"   Found {len(all_windows)} total windows on system")
            except Exception as e:
                self.log(f"   ⚠️  Warning: Could not get all windows: {e}")
                self.log("   This might be due to pygetwindow version compatibility")
                all_windows = []
            
            game_window = None
            for title in osrs_titles:
                try:
                    self.log(f"   Searching for '{title}'...")
                    windows = gw.getWindowsWithTitle(title)
                    if windows:
                        # Skip editor windows - only get actual game windows
                        for window in windows:
                            window_title = getattr(window, 'title', '')
                            # Skip if it's clearly an editor window
                            if any(skip in window_title.lower() for skip in ['cursor', 'workspace', 'untitled', '.json']):
                                continue
                            game_window = window
                            self.log(f"   ✅ Found '{title}' window!")
                            break
                        if game_window:
                            break
                    else:
                        self.log(f"   ❌ No '{title}' window found")
                except Exception as search_error:
                    self.log(f"   ⚠️  Error searching for '{title}': {search_error}")
                    continue
            
            if game_window:
                try:
                    # Safely get window properties
                    window_title = getattr(game_window, 'title', 'Unknown Title')
                    window_left = getattr(game_window, 'left', 0)
                    window_top = getattr(game_window, 'top', 0)
                    window_width = getattr(game_window, 'width', 800)
                    window_height = getattr(game_window, 'height', 600)
                    
                    # Validate coordinates - ensure they're reasonable
                    if window_left < 0 or window_top < 0:
                        self.log(f"   ⚠️  Warning: Invalid coordinates ({window_left}, {window_top})")
                        self.log("   Using fallback coordinates")
                        window_left = max(0, window_left)
                        window_top = max(0, window_top)
                    
                    # Ensure minimum size
                    if window_width < 400 or window_height < 300:
                        self.log(f"   ⚠️  Warning: Window too small ({window_width}x{window_height})")
                        self.log("   Using fallback size")
                        window_width = max(800, window_width)
                        window_height = max(600, window_height)
                    
                    # Clamp coordinates to screen boundaries
                    screen_width, screen_height = pyautogui.size()
                    self.log(f"   Screen size: {screen_width}x{screen_height}")
                    
                    # Clamp coordinates to screen boundaries
                    window_left = max(0, min(window_left, screen_width - 100))
                    window_top = max(0, min(window_top, screen_height - 100))
                    window_width = min(window_width, screen_width - window_left)
                    window_height = min(window_height, screen_height - window_top)
                    
                    self.log(f"   Clamped region: ({window_left}, {window_top}) {window_width}x{window_height}")
                    
                    # Ensure minimum usable size after clamping
                    if window_width < 400 or window_height < 300:
                        self.log(f"   ⚠️  Window too small after clamping: {window_width}x{window_height}")
                        # Use a centered region that fits on screen
                        window_width = min(800, screen_width - 100)
                        window_height = min(600, screen_height - 100)
                        window_left = (screen_width - window_width) // 2
                        window_top = (screen_height - window_height) // 2
                        self.log(f"   Using centered fallback: ({window_left}, {window_top}) {window_width}x{window_height}")
                    
                    self.screenshot_region = (
                        window_left, window_top,
                        window_width, window_height
                    )
                    # Update live view region for coordinate clamping
                    self.live_view_region = (0, 0, window_width, window_height)
                    self.window_detected = True
                    self.window_status_label.config(
                        text=f"✅ {window_title}\n{window_width}x{window_height}",
                        foreground="green"
                    )
                    
                    self.log("✅ Game window detected successfully!")
                    self.log(f"   Window title: '{window_title}'")
                    self.log(f"   Window position: ({window_left}, {window_top})")
                    self.log(f"   Window size: {window_width}x{window_height}")
                    self.log(f"   Screenshot region set to: {self.screenshot_region}")
                    self.log(f"   Live view region set to: {self.live_view_region}")
                    
                except Exception as prop_error:
                    self.log(f"   ⚠️  Warning: Could not get all window properties: {prop_error}")
                    # Use fallback values
                    self.screenshot_region = (100, 100, 800, 600)
                    # Update live view region for coordinate clamping
                    self.live_view_region = (0, 0, 800, 600)
                    self.window_detected = True
                    self.window_status_label.config(
                        text=f"✅ {getattr(game_window, 'title', 'Unknown')}\n800x600 (fallback)",
                        foreground="green"
                    )
                    self.log("✅ Game window detected with fallback values!")
                    self.log(f"   Screenshot region set to fallback: (100, 100) 800x600")
                    self.log(f"   Live view region set to fallback: {self.live_view_region}")
                
                # Check window state (using available attributes)
                try:
                    # Check if window is visible (using available methods)
                    if hasattr(game_window, 'isVisible'):
                        if game_window.isVisible:
                            self.log("   Window is visible")
                        else:
                            self.log("   ⚠️  Window is not visible")
                    else:
                        self.log("   Window visibility: Unknown (attribute not available)")
                    
                    # Check if window is minimized
                    if hasattr(game_window, 'isMinimized'):
                        if game_window.isMinimized:
                            self.log("   ⚠️  Window is minimized")
                        else:
                            self.log("   Window is not minimized")
                    else:
                        self.log("   Window minimized: Unknown (attribute not available)")
                        
                except Exception as attr_error:
                    self.log(f"   ⚠️  Could not check window state: {attr_error}")
                    self.log("   Window state attributes not available in this pygetwindow version")
                
                # Start screenshot thread
                try:
                    self.start_screenshot_thread()
                except Exception as thread_error:
                    self.log(f"   ⚠️  Warning: Could not start screenshot thread: {thread_error}")
                    self.log("   Screenshots will not be available, but window detection succeeded")
            else:
                self.window_detected = False
                self.window_status_label.config(text="❌ No game window detected", foreground="red")
                self.log("❌ No game window detected")
                if all_windows:
                    self.log("   Available windows:")
                    try:
                        for i, window in enumerate(all_windows[:10]):  # Show first 10 windows
                            try:
                                window_title = getattr(window, 'title', 'Unknown')
                                window_width = getattr(window, 'width', 0)
                                window_height = getattr(window, 'height', 0)
                                if window_title and window_title.strip():
                                    self.log(f"     {i+1}. '{window_title}' ({window_width}x{window_height})")
                            except Exception as window_error:
                                self.log(f"     {i+1}. [Error reading window: {window_error}]")
                        if len(all_windows) > 10:
                            self.log(f"     ... and {len(all_windows) - 10} more windows")
                    except Exception as list_error:
                        self.log(f"   ⚠️  Error listing windows: {list_error}")
                else:
                    self.log("   No windows could be detected")
                
        except Exception as e:
            self.log(f"❌ Error detecting window: {e}")
            self.log("   This might be due to pygetwindow compatibility issues")
            self.log("   You can try using the 'Set Manual Region' button instead")
            self.window_detected = False
            
            # Try to provide helpful information
            try:
                self.log("   Attempting to get basic window information...")
                # Try a simpler approach
                test_windows = gw.getWindowsWithTitle("")
                if test_windows:
                    self.log(f"   Basic window detection works - found {len(test_windows)} windows")
                else:
                    self.log("   Basic window detection also failed")
            except Exception as fallback_error:
                self.log(f"   Fallback detection also failed: {fallback_error}")
                self.log("   Consider updating pygetwindow or using manual region setting")
    
    def set_manual_region(self):
        """Set screenshot region manually."""
        try:
            # Simple dialog for manual region
            dialog = tk.Toplevel(self.root)
            dialog.title("Set Manual Region")
            dialog.geometry("300x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Enter screenshot region:").pack(pady=10)
            
            frame = ttk.Frame(dialog)
            frame.pack(pady=10)
            
            ttk.Label(frame, text="X:").grid(row=0, column=0, padx=5)
            x_var = tk.StringVar(value="100")
            ttk.Entry(frame, textvariable=x_var, width=8).grid(row=0, column=1, padx=5)
            
            ttk.Label(frame, text="Y:").grid(row=0, column=2, padx=5)
            y_var = tk.StringVar(value="100")
            ttk.Entry(frame, textvariable=y_var, width=8).grid(row=0, column=2, padx=5)
            
            ttk.Label(frame, text="Width:").grid(row=1, column=0, padx=5, pady=5)
            width_var = tk.StringVar(value="800")
            ttk.Entry(frame, textvariable=width_var, width=8).grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(frame, text="Height:").grid(row=1, column=2, padx=5, pady=5)
            height_var = tk.StringVar(value="600")
            ttk.Entry(frame, textvariable=height_var, width=8).grid(row=1, column=2, padx=5, pady=5)
            
            def apply_region():
                try:
                    x = int(x_var.get())
                    y = int(y_var.get())
                    width = int(width_var.get())
                    height = int(height_var.get())
                    
                    self.screenshot_region = (x, y, width, height)
                    self.window_detected = True
                    self.window_status_label.config(
                        text=f"✅ Manual Region\n{width}x{height}",
                        foreground="green"
                    )
                    
                    self.log("✅ Manual region set successfully!")
                    self.log(f"   Screenshot region: ({x}, {y}) {width}x{height}")
                    
                    # Start screenshot thread
                    try:
                        self.start_screenshot_thread()
                    except Exception as thread_error:
                        self.log(f"   ⚠️  Warning: Could not start screenshot thread: {thread_error}")
                    
                    dialog.destroy()
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers")
            
            ttk.Button(dialog, text="Apply", command=apply_region).pack(pady=10)
            
        except Exception as e:
            self.log(f"❌ Error setting manual region: {e}")
    
    def clear_window_region(self):
        """Clear the currently detected window region."""
        self.screenshot_region = None
        self.window_detected = False
        self.window_status_label.config(text="❌ No game window detected", foreground="red")
        self.log("✅ Window region cleared.")
    

    def start_screenshot_thread(self):
        """Start thread for continuous screenshots."""
        try:
            if self.screenshot_thread and self.screenshot_thread.is_alive():
                self.log("📸 Screenshot thread already running")
                return
            
            if not self.screenshot_region:
                self.log("❌ No screenshot region set - cannot start screenshot thread")
                return
            
            self.screenshot_thread = threading.Thread(target=self.screenshot_loop, daemon=True)
            self.screenshot_thread.start()
            self.log("📸 Screenshot thread started")
            self.log(f"   Thread ID: {self.screenshot_thread.ident}")
            self.log(f"   Thread alive: {self.screenshot_thread.is_alive()}")
            self.log(f"   Screenshot region: {self.screenshot_region}")
            
        except Exception as e:
            self.log(f"❌ Error starting screenshot thread: {e}")
            raise
    
    def screenshot_loop(self):
        """Continuous screenshot loop."""
        screenshot_count = 0
        error_count = 0
        
        while self.window_detected and not self.is_running:
            try:
                if self.screenshot_region:
                    x, y, width, height = self.screenshot_region
                    
                    # Validate coordinates before taking screenshot
                    if x < 0 or y < 0 or width <= 0 or height <= 0:
                        self.log(f"   ❌ Invalid screenshot region: ({x}, {y}, {width}, {height})")
                        self.log("   Stopping screenshot thread due to invalid coordinates")
                        break
                    
                    # Check if coordinates are within reasonable bounds
                    screen_width, screen_height = pyautogui.size()
                    if x + width > screen_width or y + height > screen_height:
                        self.log(f"   ❌ Screenshot region extends beyond screen: ({x}, {y}, {width}, {height})")
                        self.log(f"   Screen size: {screen_width}x{screen_height}")
                        self.log("   Stopping screenshot thread due to out-of-bounds coordinates")
                        break
                    
                    screenshot = pyautogui.screenshot(region=(x, y, width, height))
                else:
                    screenshot = pyautogui.screenshot()
                
                screenshot_count += 1
                error_count = 0  # Reset error count on success
                
                # Convert to numpy and resize for display
                screenshot_np = np.array(screenshot)
                screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                
                # Resize for display (larger size when window detected)
                height, width = screenshot_np.shape[:2]
                if self.window_detected:
                    # When window detected, use much larger size to fill expanded view
                    max_size = 800
                else:
                    # Initial size
                    max_size = 300
                
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    screenshot_np = cv2.resize(screenshot_np, (new_width, new_height))
                
                # Convert to PIL for tkinter
                screenshot_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(screenshot_rgb)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # Update GUI (must be done in main thread)
                self.root.after(0, self.update_screenshot_display, tk_image, screenshot_np.shape)
                

                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                error_count += 1
                self.log(f"❌ Screenshot error #{error_count}: {e}")
                
                # If we get too many errors in a row, stop trying
                if error_count >= 5:
                    self.log("   ⚠️  Too many screenshot errors - stopping screenshot thread")
                    break
                
                time.sleep(2.0)
        
        self.log("   📸 Screenshot loop ended")
    
    def update_screenshot_display(self, image, shape):
        """Update screenshot display in main thread."""
        try:
            self.screenshot_label.config(image=image, text="")
            self.screenshot_label.image = image  # Keep reference
            
            height, width = shape[:2]
            self.screenshot_info_label.config(text=f"Size: {width}x{height}")
            
            # If window is detected, ensure the label fills the entire frame
            if self.window_detected:
                self.screenshot_label.grid_configure(sticky=(tk.W, tk.E, tk.N, tk.S))
            
        except Exception as e:
            self.log(f"❌ Error updating screenshot: {e}")
    
    def test_prediction(self):
        """Test a single prediction without executing actions."""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load model first")
            return
        
        try:
            self.log("🧪 Testing prediction...")
            self.status_label.config(text="Testing prediction...")
            
            if self.live_gamestate_mode and self.live_feature_extractor:
                # Use live gamestate data
                self.log("   📊 Using live gamestate data...")
                
                # Get latest gamestate
                gamestate = self.live_feature_extractor.get_latest_gamestate()
                if gamestate is None:
                    self.log("❌ No gamestate data available")
                    return
                
                # Extract features
                features = self.live_feature_extractor.extract_live_features(gamestate)
                if features is None:
                    self.log("❌ Failed to extract features from gamestate")
                    return
                
                # Make prediction
                prediction = self.predict_action_from_features(features)
                
                # Display results
                self.log("🎯 Live Gamestate Prediction Results:")
                self.log_live_gamestate_info(gamestate, features)
                
            else:
                # Use screenshot mode
                if not self.window_detected:
                    messagebox.showwarning("Warning", "Please detect game window first")
                    return
                
                self.log("   📸 Using screenshot data...")
                
                # Take screenshot
                try:
                    if self.screenshot_region:
                        x, y, width, height = self.screenshot_region
                        screenshot = pyautogui.screenshot(region=(x, y, width, height))
                    else:
                        screenshot = pyautogui.screenshot()
                    
                    screenshot_np = np.array(screenshot)
                    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                    
                except Exception as screenshot_error:
                    self.log(f"❌ Error taking screenshot: {screenshot_error}")
                    self.log("   Cannot test prediction without screenshot")
                    return
                
                # Extract features (simplified)
                features = self.extract_features(screenshot_np)
                
                # Make prediction
                prediction = self.predict_action(features)
                
                # Display results
                self.log("🎯 Screenshot Prediction Results:")
            
            # Display prediction results
            action_count = int(prediction[0, 0])
            self.log(f"  Predicted Actions: {action_count}")
            
            if action_count > 0:
                # Show first few actions
                for i in range(1, min(action_count + 1, 4)):
                    action = prediction[i]  # (8,) - [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
                    
                    action_type = int(action[1])
                    x, y = action[2], action[3]
                    button = int(action[4])
                    key = int(action[5])
                    scroll_dx, scroll_dy = action[6], action[7]
                    
                    self.log(f"  Action {i}: type={action_type}, pos=({x:.0f},{y:.0f})")
                    if button > 0:
                        self.log(f"    Button: {button}")
                    if key > 0:
                        self.log(f"    Key: {key}")
                    if abs(scroll_dx) > 0.1 or abs(scroll_dy) > 0.1:
                        self.log(f"    Scroll: ({scroll_dx:.1f}, {scroll_dy:.1f})")
                
                if action_count > 3:
                    self.log(f"  ... and {action_count - 3} more actions")
            
            self.status_label.config(text="Prediction test completed")
            
        except Exception as e:
            self.log(f"❌ Prediction test failed: {e}")
            self.status_label.config(text="Prediction test failed")
    
    def extract_features(self, screenshot: np.ndarray) -> np.ndarray:
        """Extract 128 features from screenshot."""
        # Convert to grayscale
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Extract basic features (same as basic_bot.py)
        features = []
        
        # Basic image statistics (16 features)
        features.extend([
            gray.mean(), gray.std(), gray.min(), gray.max(),
            np.percentile(gray, 25), np.percentile(gray, 50), np.percentile(gray, 75),
            gray.shape[0], gray.shape[1], gray.size,
            np.sum(gray > 128), np.sum(gray < 64), np.sum(gray > 192),
            np.var(gray), np.median(gray), np.mean(np.abs(np.diff(gray)))
        ])
        
        # Edge detection (16 features)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            edges.mean(), edges.std(), edges.sum(), np.sum(edges > 0),
            np.sum(edges > 50), np.sum(edges > 100), np.sum(edges > 150),
            np.sum(edges > 200), np.sum(edges > 250),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
            cv2.Sobel(gray, cv2.CV_64F, 1, 0).var(),
            cv2.Sobel(gray, cv2.CV_64F, 0, 1).var(),
            cv2.Sobel(gray, cv2.CV_64F, 2, 0).var(),
            cv2.Sobel(gray, cv2.CV_64F, 0, 2).var(),
            cv2.Sobel(gray, cv2.CV_64F, 1, 1).var(),
            cv2.Sobel(gray, cv2.CV_64F, 2, 2).var()
        ])
        
        # Color features (16 features)
        if len(screenshot.shape) == 3:
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            features.extend([
                hsv[:,:,0].mean(), hsv[:,:,0].std(), hsv[:,:,1].mean(), hsv[:,:,1].std(),
                hsv[:,:,2].mean(), hsv[:,:,2].std(),
                np.sum(hsv[:,:,0] > 90), np.sum(hsv[:,:,0] < 30),
                np.sum(hsv[:,:,1] > 128), np.sum(hsv[:,:,1] < 64),
                np.sum(hsv[:,:,2] > 192), np.sum(hsv[:,:,2] < 64),
                np.sum((hsv[:,:,0] > 100) & (hsv[:,:,1] > 100)),
                np.sum((hsv[:,:,0] > 0) & (hsv[:,:,0] < 20)),
                np.sum((hsv[:,:,0] > 110) & (hsv[:,:,0] < 130)),
                np.sum((hsv[:,:,0] > 20) & (hsv[:,:,0] < 40))
            ])
        else:
            features.extend([0] * 16)
        
        # Texture features (16 features)
        # Gabor filter responses for texture analysis
        angles = [0, 45, 90, 135]
        frequencies = [0.1, 0.2, 0.3, 0.4]
        
        for angle in angles:
            for freq in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 3, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                features.append(filtered.mean())
                features.append(filtered.std())
        
        # Grid-based features (64 features)
        grid_size = 8
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * (gray.shape[0] // grid_size)
                y_end = (i + 1) * (gray.shape[0] // grid_size)
                x_start = j * (gray.shape[1] // grid_size)
                x_end = (j + 1) * (gray.shape[1] // grid_size)
                
                grid_section = gray[y_start:y_end, x_start:x_end]
                features.append(grid_section.mean())
        
        # Ensure exactly 128 features
        features = features[:128]
        if len(features) < 128:
            features.extend([0] * (128 - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def predict_action(self, features: np.ndarray) -> np.ndarray:
        """Use model to predict action."""
        # Create temporal sequence (10 timesteps of gamestate features)
        temporal_sequence = np.tile(features, (10, 1))  # (10, 128)
        
        # Create action sequence (10 timesteps of dummy actions for now)
        action_sequence = np.zeros((10, 101, 8), dtype=np.float32)
        action_sequence[:, 0, 0] = 1.0  # Set action count to 1
        
        # Convert to tensors
        temporal_tensor = torch.FloatTensor(temporal_sequence).unsqueeze(0)  # (1, 10, 128)
        action_tensor = torch.FloatTensor(action_sequence).unsqueeze(0)      # (1, 10, 101, 8)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(temporal_tensor, action_tensor)
        
        # Convert to numpy - prediction is (1, 101, 8)
        return prediction.cpu().numpy()[0]  # (101, 8)
    
    def start_bot(self):
        """Start the bot."""
        if not self.model_loaded or not self.window_detected:
            messagebox.showwarning("Warning", "Please load model and detect game window first")
            return
        
        try:
            # Get settings
            action_interval = float(self.action_interval_var.get())
            max_duration = int(self.max_duration_var.get())
            
            self.log(f"🚀 Starting bot...")
            self.log(f"   Action interval: {action_interval}s")
            self.log(f"   Max duration: {max_duration}s")
            
            # Confirm with user
            response = messagebox.askyesno(
                "Confirm Bot Start", 
                f"⚠️  WARNING: This will move your mouse and click!\n\n"
                f"Settings:\n"
                f"• Action interval: {action_interval}s\n"
                f"• Max duration: {max_duration}s\n\n"
                f"Make sure OSRS is visible and you're ready!\n\n"
                f"Start the bot?"
            )
            
            if not response:
                return
            
            # Start bot thread
            self.is_running = True
            self.bot_thread = threading.Thread(
                target=self.bot_loop, 
                args=(action_interval, max_duration),
                daemon=True
            )
            self.bot_thread.start()
            
            # Update UI
            self.start_bot_btn.config(state="disabled")
            self.stop_bot_btn.config(state="normal")
            self.status_label.config(text="Bot running")
            
        except Exception as e:
            self.log(f"❌ Error starting bot: {e}")
            messagebox.showerror("Error", f"Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the bot."""
        self.is_running = False
        self.log("⏹️  Stopping bot...")
        
        # Update UI
        self.start_bot_btn.config(state="normal")
        self.stop_bot_btn.config(state="disabled")
        self.status_label.config(text="Bot stopped")
    
    def bot_loop(self, action_interval: float, max_duration: int):
        """Main bot loop."""
        start_time = time.time()
        action_count = 0
        
        try:
            while self.is_running and (time.time() - start_time) < max_duration:
                try:
                    # Take screenshot
                    if self.screenshot_region:
                        x, y, width, height = self.screenshot_region
                        screenshot = pyautogui.screenshot(region=(x, y, width, height))
                    else:
                        screenshot = pyautogui.screenshot()
                    
                    screenshot_np = np.array(screenshot)
                    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                    
                    # Extract features
                    features = self.extract_features(screenshot_np)
                    
                    # Predict action
                    prediction = self.predict_action(features)
                    
                    # Execute action (safe mode - just log for now)
                    # TODO: Add toggle for actual action execution when ready
                    self.log(f"🎯 Action {action_count + 1}: {self.format_action(prediction)}")
                    
                    action_count += 1
                    
                except Exception as action_error:
                    self.log(f"❌ Error in action {action_count + 1}: {action_error}")
                    # Continue to next action instead of crashing
                    action_count += 1
                
                # Update status in main thread
                elapsed = time.time() - start_time
                remaining = max_duration - elapsed
                self.root.after(0, self.update_bot_status, action_count, elapsed, remaining)
                
                # Wait for next action
                if remaining > action_interval:
                    time.sleep(action_interval)
                
        except Exception as e:
            self.log(f"❌ Bot error: {e}")
        finally:
            self.log(f"✅ Bot finished. Total actions: {action_count}")
            self.root.after(0, self.bot_finished)
    
    def format_action(self, action_tensor: np.ndarray) -> str:
        """Format action tensor for logging."""
        # action_tensor is (101, 8) where index 0 contains action count
        action_count = int(action_tensor[0, 0])
        
        if action_count == 0:
            return "No actions predicted"
        
        parts = [f"Actions: {action_count}"]
        
        # Format the first few actions (up to 3 for readability)
        for i in range(1, min(action_count + 1, 4)):
            action = action_tensor[i]  # (8,) - [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
            
            action_type = int(action[1])
            x, y = action[2], action[3]
            button = int(action[4])
            key = int(action[5])
            scroll_dx, scroll_dy = action[6], action[7]
            
            action_desc = f"Action{i}: type={action_type}, pos=({x:.0f},{y:.0f})"
            if button > 0:
                action_desc += f", button={button}"
            if key > 0:
                action_desc += f", key={key}"
            if abs(scroll_dx) > 0.1 or abs(scroll_dy) > 0.1:
                action_desc += f", scroll=({scroll_dx:.1f},{scroll_dy:.1f})"
            
            parts.append(action_desc)
        
        if action_count > 3:
            parts.append(f"... and {action_count - 3} more actions")
        
        return " | ".join(parts)
    
    def update_bot_status(self, action_count: int, elapsed: float, remaining: float):
        """Update bot status in main thread."""
        self.action_count_label.config(text=f"Actions: {action_count}")
        self.runtime_label.config(text=f"Runtime: {elapsed:.1f}s")
        
        if remaining > 0:
            self.status_label.config(text=f"Bot running - {remaining:.1f}s remaining")
    
    def bot_finished(self):
        """Called when bot finishes."""
        self.start_bot_btn.config(state="normal")
        self.stop_bot_btn.config(state="disabled")
        self.status_label.config(text="Bot finished")

    def _setup_periodic_refresh(self):
        """Set up periodic refresh for real-time updates."""
        def periodic_refresh():
            if self.feature_tracking_active or self.live_gamestate_mode:
                # Only refresh if we're actively tracking or in live mode
                if (hasattr(self, 'live_feature_tree') and 
                    self.live_feature_buffer and 
                    len(self.live_feature_buffer) > 0 and
                    all(isinstance(vec, (list, np.ndarray)) and len(vec) == 128 for vec in self.live_feature_buffer)):
                    # Only update if we have valid data
                    self._update_live_feature_table()
                # Canvas rendering has been replaced with table updates
                pass
            # Schedule next refresh in 500ms
            self.root.after(500, periodic_refresh)
        
        # Start the periodic refresh
        self.root.after(500, periodic_refresh)

    def print_paths_and_status(self):
        from datetime import datetime
        # Remove the import of bot_paths since it doesn't exist
        
        def _fmt(p: Path):
            try:
                if p.exists():
                    ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    size = p.stat().st_size
                    return f"✅ {p} (mtime: {ts}, size: {size})"
                return f"❌ {p} (missing)"
            except Exception as e:
                return f"⚠️  {p} (error: {e})"
        
        def _fmt_dir(p: Path):
            try:
                if p.exists():
                    files = list(p.glob("*.json"))
                    if files:
                        latest = max(files, key=lambda f: f.stat().st_mtime)
                        latest_time = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        return f"✅ {p} ({len(files)} files, latest: {latest_time})"
                    else:
                        return f"⚠️  {p} (exists, 0 files)"
                return f"❌ {p} (missing)"
            except Exception as e:
                return f"⚠️  {p} (error: {e})"
        
        # Get current bot mode
        current_bot = self.bot_mode_var.get() if hasattr(self, 'bot_mode_var') else "bot1"
        
        # Get ModelRunner status
        runner = ModelRunner.instance()
        device_str = getattr(runner, 'device_str', 'unknown')
        seq_len = getattr(runner, 'seq_len', 'unknown')
        num_features = getattr(runner, 'num_features', 'unknown')
        model_action_rows = getattr(runner, 'model_action_rows', 'unknown')
        action_vec = getattr(runner, 'action_vec', 'unknown')
        
        # Get buffer status
        gs_buf_len = len(self.gs_buf) if hasattr(self, 'gs_buf') else 0
        action_buf_len = len(self.action_buf) if hasattr(self, 'action_buf') else 0
        
        # Get data source info
        data_source = "unknown"
        if hasattr(self, 'live_feature_extractor') and self.live_feature_extractor:
            try:
                source_info = self.live_feature_extractor.get_data_source_info()
                if isinstance(source_info, tuple):
                    data_source = f"{source_info[0]} ({source_info[1]})"
                else:
                    data_source = str(source_info)
            except:
                data_source = "error getting source info"
        
        self.log(f"\n🔍 PATHS & STATUS DIAGNOSTICS:")
        self.log(f"Model weights: {_fmt(MODEL_PATH)}")
        self.log(f"Feature mappings: {_fmt(rp('data', 'features', 'feature_mappings.json'))}")
        self.log(f"ID mappings: {_fmt(rp('data', 'features', 'id_mappings.json'))}")
        # Use existing path resolution instead of bot_paths
        self.log(f"Live gamestate ({current_bot}): {_fmt(rp('data', current_bot, 'runelite_gamestate.json'))}")
        self.log(f"Rolling buffer ({current_bot}): {_fmt_dir(rp('data', current_bot, 'gamestates'))}")
        self.log(f"\n🧠 MODEL RUNNER STATUS:")
        self.log(f"Device: {device_str}")
        self.log(f"Seq length (T): {seq_len}")
        self.log(f"Features: {num_features}")
        self.log(f"Action rows: {model_action_rows}")
        self.log(f"Action vec: {action_vec}")
        self.log(f"\n📊 BUFFER STATUS:")
        self.log(f"Feature buffer: {gs_buf_len}/10")
        self.log(f"Action buffer: {action_buf_len}/10")
        self.log(f"Data source: {data_source}")

    def _load_live_feature_mappings(self):
        """Load feature mappings and names for the live feature table."""
        print(f"[DEBUG] _load_live_feature_mappings called")
        try:
            base_dir = Path(__file__).resolve().parent
            mappings_path = base_dir / "data" / "features" / "feature_mappings.json"
            id_mappings_path = base_dir / "data" / "features" / "id_mappings.json"
            print(f"[DEBUG] Base dir: {base_dir}")
            print(f"[DEBUG] Mappings path: {mappings_path}")
            print(f"[DEBUG] ID mappings path: {id_mappings_path}")

            # Load feature mappings
            self.live_feature_mappings = []
            self.live_feature_names = [""] * 128
            if mappings_path.exists():
                print(f"[DEBUG] ✅ Feature mappings file exists")
                with open(mappings_path, "r", encoding="utf-8") as f:
                    self.live_feature_mappings = json.load(f)
                print(f"[DEBUG] ✅ Loaded {len(self.live_feature_mappings)} feature mappings")
                
                for mapping in self.live_feature_mappings:
                    if isinstance(mapping, dict):
                        idx = mapping.get("feature_index")
                        name = mapping.get("feature_name")
                        if isinstance(idx, int) and 0 <= idx < 128 and isinstance(name, str):
                            self.live_feature_names[idx] = name
                
                # Fill any empty slots with default labels to guarantee length 128 indexing
                for i in range(128):
                    if not self.live_feature_names[i]:
                        self.live_feature_names[i] = f"feature_{i}"
                
                print(f"[DEBUG] ✅ Feature names array populated with {len([n for n in self.live_feature_names if n != ''])} non-empty names")
                self.log(f"🔍 Loaded {len(self.live_feature_mappings)} feature mappings from {mappings_path}")
            else:
                print(f"[DEBUG] ❌ Feature mappings file NOT found")
                self.log(f"⚠️ Feature mappings file not found: {mappings_path}")
                # Ensure names array is always usable
                self.live_feature_names = [f"feature_{i}" for i in range(128)]

            # Load ID mappings
            self.live_id_mappings = {}
            if id_mappings_path.exists():
                print(f"[DEBUG] ✅ ID mappings file exists")
                with open(id_mappings_path, "r", encoding="utf-8") as f:
                    self.live_id_mappings = json.load(f)
                print(f"[DEBUG] ✅ Loaded ID mappings with {len(self.live_id_mappings)} categories")
                print(f"[DEBUG] ID mapping categories: {list(self.live_id_mappings.keys())}")
                self.log(f"🔍 Loaded ID mappings: {len(self.live_id_mappings)} categories from {id_mappings_path}")
                
                # Create hash reverse lookup for fast translation - EXACTLY like browse_training_data.py
                self.hash_reverse_lookup = {}
                if hasattr(self, 'live_id_mappings') and self.live_id_mappings:
                    print(f"[DEBUG] ✅ Creating hash reverse lookup")
                    
                    # Global hash mappings
                    if 'Global' in self.live_id_mappings and 'hash_mappings' in self.live_id_mappings['Global']:
                        hash_mappings = self.live_id_mappings['Global']['hash_mappings']
                        print(f"[DEBUG] ✅ Found Global hash_mappings with {len(hash_mappings)} entries")
                        for hash_key, original_string in hash_mappings.items():
                            try:
                                hash_key_int = int(hash_key)
                                if 'hash_mappings' not in self.hash_reverse_lookup:
                                    self.hash_reverse_lookup['hash_mappings'] = {}
                                self.hash_reverse_lookup['hash_mappings'][hash_key_int] = original_string
                            except (ValueError, TypeError):
                                pass
                    else:
                        print(f"[DEBUG] ❌ Global hash_mappings not found")
                    
                    # Feature-group-specific mappings
                    for feature_group, group_mappings in self.live_id_mappings.items():
                        if feature_group == 'Global':
                            continue
                        
                        print(f"[DEBUG] 🔍 Processing feature group: {feature_group}")
                        for mapping_type, mappings in group_mappings.items():
                            if isinstance(mappings, dict):
                                print(f"[DEBUG]   - {mapping_type}: {len(mappings)} entries")
                                for id_key, original_string in mappings.items():
                                    try:
                                        id_key_int = int(id_key)
                                        if mapping_type not in self.hash_reverse_lookup:
                                            self.hash_reverse_lookup[mapping_type] = {}
                                        self.hash_reverse_lookup[mapping_type][id_key_int] = original_string
                                    except (ValueError, TypeError):
                                        pass
                
                print(f"[DEBUG] ✅ Created hash reverse lookup with {len(self.hash_reverse_lookup)} mapping types")
                print(f"[DEBUG] Hash reverse lookup keys: {list(self.hash_reverse_lookup.keys())}")
                self.log(f"🔍 Created hash reverse lookup with {len(self.hash_reverse_lookup)} mapping types")
            else:
                print(f"[DEBUG] ❌ ID mappings file NOT found")
                self.log(f"⚠️ ID mappings file not found: {id_mappings_path}")
                self.hash_reverse_lookup = {}
        except Exception as e:
            print(f"[DEBUG] ❌ Exception in _load_live_feature_mappings: {e}")
            self.log(f"Warning: Failed to load live feature mappings: {e}")
            # Guarantee sane defaults to prevent index errors
            self.live_feature_names = [f"feature_{i}" for i in range(128)]
            self.live_feature_mappings = []
            self.live_id_mappings = {}
        
        print(f"[DEBUG] ✅ _load_live_feature_mappings completed")
        print(f"[DEBUG] Final state:")
        print(f"[DEBUG]   - live_feature_mappings: {len(self.live_feature_mappings)} features")
        print(f"[DEBUG]   - live_feature_names: {len(self.live_feature_names)} names")
        print(f"[DEBUG]   - live_id_mappings: {len(self.live_id_mappings)} categories")
        print(f"[DEBUG]   - hash_reverse_lookup: {len(self.hash_reverse_lookup)} mapping types")

    def _get_live_feature_category(self, feature_name):
        """Get feature category for live feature table."""
        # Find the feature in our mappings
        for feature_data in self.live_feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_name') == feature_name:
                return feature_data.get('feature_group', 'other')
        
        # Fallback to hardcoded logic if not found in mappings
        feature_name_lower = feature_name.lower()
        
        # Player state features (0-4)
        if "world_x" in feature_name_lower or "world_y" in feature_name_lower:
            return "Player"
        elif "animation" in feature_name_lower:
            return "Player"
        elif "moving" in feature_name_lower or "movement" in feature_name_lower:
            return "Player"
        
        # Interaction context features (5-8)
        elif "action_type" in feature_name_lower or "item_name" in feature_name_lower or "target" in feature_name_lower:
            return "Interaction"
        elif "time_since_interaction" in feature_name_lower:
            return "Phase Context"
        
        # Camera features (9-13)
        elif "camera" in feature_name_lower or "pitch" in feature_name_lower or "yaw" in feature_name_lower:
            return "Camera"
        
        # Inventory features (14-41)
        elif "inventory" in feature_name_lower or "slot" in feature_name_lower:
            return "Inventory"
        
        # Bank features (42-62)
        elif "bank" in feature_name_lower:
            return "Bank"
        
        # Phase context features (63-66)
        elif "phase" in feature_name_lower:
            return "Phase Context"
        
        # Game objects features (67-108)
        elif "game_object" in feature_name_lower or "furnace" in feature_name_lower or "bank_booth" in feature_name_lower:
            return "Game Objects"
        
        # NPC features (109-123)
        elif "npc" in feature_name_lower:
            return "NPCs"
        
        # Tab features (124)
        elif "tab" in feature_name_lower:
            return "Tabs"
        
        # Skills features (125-126)
        elif "level" in feature_name_lower or "xp" in feature_name_lower:
            return "Skills"
        
        # Timestamp feature (127)
        elif "timestamp" in feature_name_lower:
            return "Timestamp"
        
        else:
            return "other"

    def _apply_live_row_coloring(self, item, category):
        """Apply color coding to live table rows based on feature category."""
        colors = {
            "Player": "#e6f7ff",           # Light cyan for player state
            "Interaction": "#e6ffe6",      # Light green for interaction context
            "Camera": "#f0f0f0",           # Light gray for camera
            "Inventory": "#e6f3ff",        # Light blue for inventory items
            "Bank": "#ffe6e6",             # Light red for bank
            "Phase Context": "#e6ffe6",    # Light green for phase context
            "Game Objects": "#f0e6ff",     # Light purple for game objects
            "NPCs": "#fff2e6",             # Light orange for NPCs
            "Tabs": "#ffe6e6",             # Light red for tabs
            "Skills": "#fff7e6",           # Light yellow for skills
            "Timestamp": "#f5f5f5",        # Very light gray for timestamp
            "other": "#ffffff"             # White for other features
        }
        
        bg_color = colors.get(category, "#ffffff")
        self.live_feature_tree.tag_configure(category, background=bg_color)
        self.live_feature_tree.item(item, tags=(category,))

    def _format_live_value_for_display(self, value, feature_name, feature_idx):
        """Format a value for display in the live feature table."""
        try:
            if self.show_translations_var.get():
                # Try to find translation using ID mappings
                translation_found = False
                
                # Check hash mappings first
                hash_mappings = self.live_id_mappings.get('Global', {}).get('hash_mappings', {})
                try:
                    hash_key = int(float(value))
                    if str(hash_key) in hash_mappings:
                        return hash_mappings[str(hash_key)]
                except (ValueError, TypeError):
                    pass
                
                # Check feature-specific mappings based on feature index
                if 0 <= feature_idx <= 4:  # Player features (0-4)
                    if 'Player' in self.live_id_mappings and 'player_animation_ids' in self.live_id_mappings['Player']:
                        try:
                            id_key = int(float(value))
                            if str(id_key) in self.live_id_mappings['Player']['player_animation_ids']:
                                return self.live_id_mappings['Player']['player_animation_ids'][str(id_key)]
                        except (ValueError, TypeError):
                            pass
                elif 14 <= feature_idx <= 41:  # Inventory features (14-41)
                    if 'Inventory' in self.live_id_mappings and 'item_ids' in self.live_id_mappings['Inventory']:
                        try:
                            id_key = int(float(value))
                            if str(id_key) in self.live_id_mappings['Inventory']['item_ids']:
                                return self.live_id_mappings['Inventory']['item_ids'][str(id_key)]
                        except (ValueError, TypeError):
                            pass
                
                # Check other mappings
                for category in ['Bank', 'Game Objects', 'NPCs', 'Tabs', 'Phase Context']:
                    if category in self.live_id_mappings:
                        for mapping_type, mappings in self.live_id_mappings[category].items():
                            if isinstance(mappings, dict):
                                try:
                                    id_key = int(float(value))
                                    if str(id_key) in mappings:
                                        return mappings[str(id_key)]
                                except (ValueError, TypeError):
                                    pass
            
            # Return formatted raw value
            if isinstance(value, (int, float)):
                if value == int(value):
                    return str(int(value))
                else:
                    return f"{value:.3f}"
            else:
                return str(value)
                
        except Exception:
            return str(value)

    def _update_live_feature_table(self):
        """Update the live feature table with the exact same logic as browse_training_data.py"""
        print(f"[DEBUG] _update_live_feature_table called")
        print(f"[DEBUG] live_feature_buffer exists: {hasattr(self, 'live_feature_buffer')}")
        print(f"[DEBUG] live_feature_buffer length: {len(self.live_feature_buffer) if hasattr(self, 'live_feature_buffer') else 'N/A'}")
        
        if not self.live_feature_buffer or len(self.live_feature_buffer) < 10:
            print(f"[DEBUG] ❌ Not enough data in buffer, returning early")
            return
        
        # Clear table
        for item in self.live_feature_tree.get_children():
            self.live_feature_tree.delete(item)
        
        # Get the latest 10 timesteps (newest is last)
        latest_features = list(self.live_feature_buffer)[-10:]
        print(f"[DEBUG] Got {len(latest_features)} timesteps from buffer")
        
        # Populate table with feature data - EXACTLY like browse_training_data.py
        for feature_idx in range(128):  # 128 features
            # Get feature name if available
            feature_name = self.live_feature_names[feature_idx] if feature_idx < len(self.live_feature_names) else f'feature_{feature_idx}'
            
            # Get values for all 10 timesteps
            values = []
            for timestep_features in latest_features:
                if isinstance(timestep_features, (list, np.ndarray)) and len(timestep_features) > feature_idx:
                    values.append(timestep_features[feature_idx])
                else:
                    values.append(0.0)
            
            # Format values nicely using the exact same method as browse_training_data.py
            formatted_values = []
            show_translations = self.show_translations_var.get()
            print(f"[DEBUG] show_translations_var.get() returned: {show_translations}")
            
            # Debug specific features that should have translations
            debug_features = [14, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
            
            for i, value in enumerate(values):
                if feature_idx in debug_features and i == 0:  # Debug first value of interesting features
                    print(f"[DEBUG] 🔍 Formatting value {value} for feature {feature_idx} ({feature_name})")
                formatted_value = self.format_value_for_display(value, feature_idx, show_translations)
                formatted_values.append(formatted_value)
                if feature_idx in debug_features and i == 0:  # Debug first value of interesting features
                    print(f"[DEBUG] ✅ Formatted value: '{formatted_value}'")
            
            # Insert into table - EXACTLY like browse_training_data.py
            item = self.live_feature_tree.insert("", "end", values=[
                feature_name,           # "player_world_x"
                feature_idx,            # 0
                formatted_values[0],    # "3096" (Timestep 0)
                formatted_values[1],    # "3096" (Timestep 1)
                formatted_values[2],    # "3096" (Timestep 2)
                formatted_values[3],    # "3096" (Timestep 3)
                formatted_values[4],    # "3096" (Timestep 4)
                formatted_values[5],    # "3096" (Timestep 5)
                formatted_values[6],    # "3096" (Timestep 6)
                formatted_values[7],    # "3096" (Timestep 7)
                formatted_values[8],    # "3096" (Timestep 8)
                formatted_values[9]     # "3096" (Timestep 9)
            ])
            
            # Apply color coding based on feature category - EXACTLY like browse_training_data.py
            category = self.get_feature_category(feature_name)
            self.apply_row_coloring(item, category)
        
        print(f"[DEBUG] ✅ Table update completed")

    def _filter_live_table(self):
        """Filter the live feature table based on current filters."""
        # Only update if we have valid data
        if (self.live_feature_buffer and 
            len(self.live_feature_buffer) > 0 and
            all(isinstance(vec, (list, np.ndarray)) and len(vec) == 128 for vec in self.live_feature_buffer)):
            self._update_live_feature_table()

    def _copy_live_table_to_clipboard(self):
        """Copy the live feature table to clipboard."""
        try:
            # Get all visible rows
            rows = []
            for item in self.live_feature_tree.get_children():
                values = self.live_feature_tree.item(item)['values']
                rows.append('\t'.join(str(v) for v in values))
            
            # Join rows with newlines
            table_text = '\n'.join(rows)
            
            # Copy to clipboard
            self.root.clipboard_clear()
            self.root.clipboard_append(table_text)
            
            self.log("✅ Live feature table copied to clipboard")
            
        except Exception as e:
            self.log(f"❌ Failed to copy table: {e}")

    def _export_live_table_to_csv(self):
        """Export the live feature table to CSV."""
        try:
            # Get file path
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Live Feature Table"
            )
            
            if not file_path:
                return
            
            # Get all visible rows
            rows = []
            for item in self.live_feature_tree.get_children():
                values = self.live_feature_tree.item(item)['values']
                rows.append([str(v) for v in values])
            
            # Write CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(self.live_feature_tree["columns"])
                # Write data
                writer.writerows(rows)
            
            self.log(f"✅ Live feature table exported to {file_path}")
            
        except Exception as e:
            self.log(f"❌ Failed to export table: {e}")

    def on_live_table_motion(self, event):
        """Handle mouse motion over live table for tooltips."""
        # Get the item and column under the cursor
        item = self.live_feature_tree.identify_row(event.y)
        column = self.live_feature_tree.identify_column(event.x)
        
        if item and column:
            # Get the value at this position
            values = self.live_feature_tree.item(item)['values']
            col_idx = int(column[1]) - 1  # Convert column identifier to index
            
            if 0 <= col_idx < len(values):
                value = values[col_idx]
                
                if col_idx == 0:  # Feature name column
                    feature_name = value
                    category = self._get_live_feature_category(feature_name)
                    tooltip_text = f"Feature: {feature_name}\nCategory: {category}"
                elif col_idx == 1:  # Index column
                    tooltip_text = f"Feature Index: {value}"
                else:  # Timestep column
                    feature_idx = int(values[1])  # Get feature index from row
                    timestep_idx = col_idx - 2  # Adjust for Feature and Index columns
                    
                    if (timestep_idx < len(self.live_feature_buffer) and
                        isinstance(self.live_feature_buffer[timestep_idx], (list, np.ndarray)) and
                        len(self.live_feature_buffer[timestep_idx]) > feature_idx):
                        raw_value = self.live_feature_buffer[timestep_idx][feature_idx]
                        feature_name = values[0]
                        category = self._get_live_feature_category(feature_name)
                        
                        # Enhanced tooltip with translation info
                        tooltip_text = f"Timestep {timestep_idx}\n"
                        tooltip_text += f"Feature: {feature_name}\n"
                        tooltip_text += f"Category: {category}\n"
                        
                        # Show translation if available and enabled, otherwise show raw value
                        if self.show_translations_var.get():
                            # Try to find translation
                            translation_found = False
                            
                            # Check hash mappings first
                            hash_mappings = self.live_id_mappings.get('Global', {}).get('hash_mappings', {})
                            try:
                                hash_key = int(float(raw_value))
                                if str(hash_key) in hash_mappings:
                                    original_value = hash_mappings[str(hash_key)]
                                    tooltip_text += f"Value: {original_value}"
                                    translation_found = True
                            except (ValueError, TypeError):
                                pass
                            
                            # Check feature-specific mappings
                            if not translation_found:
                                if 0 <= feature_idx <= 4:  # Player features
                                    if 'Player' in self.live_id_mappings and 'player_animation_ids' in self.live_id_mappings['Player']:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.live_id_mappings['Player']['player_animation_ids']:
                                                name = self.live_id_mappings['Player']['player_animation_ids'][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                        except (ValueError, TypeError):
                                            pass
                                elif 14 <= feature_idx <= 41:  # Inventory features
                                    if 'Inventory' in self.live_id_mappings and 'item_ids' in self.live_id_mappings['Inventory']:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.live_id_mappings['Inventory']['item_ids']:
                                                name = self.live_id_mappings['Inventory']['item_ids'][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                        except (ValueError, TypeError):
                                            pass
                            
                            # If no translation found, show raw value
                            if not translation_found:
                                tooltip_text += f"Raw Value: {raw_value}"
                        else:
                            tooltip_text += f"Raw Value: {raw_value}"
                        
                        # Add feature-specific information
                        if "time_since_interaction" in feature_name:
                            if raw_value > 0:
                                tooltip_text += f"\nTime: {raw_value:.0f}ms since last interaction"
                        elif "phase_start_time" in feature_name:
                            if raw_value > 0:
                                tooltip_text += f"\nTime: {raw_value:.0f}ms since phase start"
                        elif "phase_duration" in feature_name:
                            if raw_value > 0:
                                tooltip_text += f"\nDuration: {raw_value:.0f}ms"
                    else:
                        tooltip_text = f"Timestep {timestep_idx}\nNo data available"
                
                # Show tooltip if text changed or if tooltip doesn't exist
                if tooltip_text != self.live_tooltip_text or not self.live_tooltip:
                    self.live_tooltip_text = tooltip_text
                    self.show_live_tooltip(event.x_root, event.y_root, tooltip_text)

    def on_live_table_leave(self, event):
        """Handle mouse leave from live table."""
        self.hide_live_tooltip()

    def show_live_tooltip(self, x, y, text):
        """Show tooltip for live table at specified coordinates."""
        self.hide_live_tooltip()
        
        # Create tooltip window
        self.live_tooltip = tk.Toplevel(self.root)
        self.live_tooltip.wm_overrideredirect(True)
        self.live_tooltip.wm_geometry(f"+{x+10}+{y+10}")
        
        # Wrap text to prevent very long tooltips
        wrapped_text = self.wrap_text(text, max_width=60)
        
        # Create tooltip label with wrapped text
        label = tk.Label(self.live_tooltip, text=wrapped_text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Tahoma", "8", "normal"), wraplength=400)
        label.pack(padx=5, pady=3)

    def hide_live_tooltip(self):
        """Hide the live tooltip."""
        if self.live_tooltip:
            self.live_tooltip.destroy()
            self.live_tooltip = None

    def wrap_text(self, text, max_width=60):
        """Wrap text to prevent very long lines in tooltips."""
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= max_width:
                wrapped_lines.append(line)
            else:
                # Split long lines at word boundaries
                words = line.split()
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= max_width:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                            current_line = word
                        else:
                            # If a single word is too long, just add it
                            wrapped_lines.append(word)
                
                if current_line:
                    wrapped_lines.append(current_line)
        
        return '\n'.join(wrapped_lines)

    def format_value_for_display(self, value, feature_idx=None, show_translation=True):
        """Format a value for display in the table with better readability and optional hash translation - EXACTLY like browse_training_data.py"""
        # DEBUG: Log what we're being called with
        debug_features = [14, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        
        if feature_idx in debug_features:
            print(f"[DEBUG] 🔍 format_value_for_display called for interesting feature: value={value}, feature_idx={feature_idx}, show_translation={show_translation}")
        
        if isinstance(value, (int, float)):
            # Convert to float to handle numpy types
            value = float(value)
            
            # If showing translations, try to find a translation using the new ID mappings structure
            if show_translation and hasattr(self, 'live_id_mappings') and self.live_id_mappings:
                print(f"[DEBUG] ✅ live_id_mappings exists and has {len(self.live_id_mappings)} groups")
                
                # Get feature info to determine the correct mapping category
                feature_name = None
                feature_group = None
                data_type = None
                
                if feature_idx is not None:
                    print(f"[DEBUG] 🔍 Looking for feature index {feature_idx} in live_feature_mappings")
                    if hasattr(self, 'live_feature_mappings'):
                        print(f"[DEBUG] ✅ live_feature_mappings exists with {len(self.live_feature_mappings)} features")
                        for feature_data in self.live_feature_mappings:
                            if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                                feature_name = feature_data.get('feature_name')
                                feature_group = feature_data.get('feature_group')
                                data_type = feature_data.get('data_type')
                                print(f"[DEBUG] ✅ Found feature: name='{feature_name}', group='{feature_group}', type='{data_type}'")
                                break
                        else:
                            print(f"[DEBUG] ❌ Feature index {feature_idx} NOT found in live_feature_mappings")
                    else:
                        print(f"[DEBUG] ❌ live_feature_mappings does NOT exist")
                else:
                    print(f"[DEBUG] ❌ feature_idx is None - cannot look up feature info")
                
                # Handle boolean values automatically
                if data_type == 'boolean':
                    if value == 1.0:
                        return "true"
                    elif value == 0.0:
                        return "false"
                
                # Check feature-group-specific mappings
                if feature_group and feature_group in self.live_id_mappings:
                    group_mappings = self.live_id_mappings[feature_group]
                    print(f"[DEBUG] ✅ Found group '{feature_group}' with mappings: {list(group_mappings.keys())}")
                    
                    # Check hash mappings first (for hashed strings)
                    if 'hash_mappings' in group_mappings:
                        try:
                            hash_key = int(float(value))
                            print(f"[DEBUG] 🔍 Looking for hash_key {hash_key} in group_mappings['hash_mappings']")
                            if str(hash_key) in group_mappings['hash_mappings']:
                                original_value = group_mappings['hash_mappings'][str(hash_key)]
                                print(f"[DEBUG] ✅ Found in hash_mappings: {hash_key} -> '{original_value}'")
                                return str(original_value)
                            else:
                                print(f"[DEBUG] ❌ hash_key {hash_key} NOT found in group_mappings['hash_mappings']")
                        except (ValueError, TypeError) as e:
                            print(f"[DEBUG] ❌ Error converting value {value} to hash_key: {e}")
                    
                    # Check specific mapping types based on feature group
                    if feature_group == "Player":
                        if 'player_animation_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['player_animation_ids']:
                                    name = group_mappings['player_animation_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                        if 'player_movement_direction_hashes' in group_mappings:
                            try:
                                hash_key = int(float(value))
                                if str(hash_key) in group_mappings['player_movement_direction_hashes']:
                                    name = group_mappings['player_movement_direction_hashes'][str(hash_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Interaction":
                        for mapping_type in ['action_type_hashes', 'item_name_hashes', 'target_hashes']:
                            if mapping_type in group_mappings:
                                try:
                                    hash_key = int(float(value))
                                    if str(hash_key) in group_mappings[mapping_type]:
                                        name = group_mappings[mapping_type][str(hash_key)]
                                        return str(name)
                                except (ValueError, TypeError):
                                    pass
                    
                    elif feature_group == "Inventory":
                        if 'item_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                print(f"[DEBUG] 🔍 Looking for id_key {id_key} in item_ids")
                                if str(id_key) in group_mappings['item_ids']:
                                    name = group_mappings['item_ids'][str(id_key)]
                                    print(f"[DEBUG] ✅ Found in item_ids: {id_key} -> '{name}'")
                                    if feature_idx in debug_features:
                                        print(f"[DEBUG] 🎯 TRANSLATION SUCCESS for feature {feature_idx}: {value} -> '{name}'")
                                    return str(name)
                                else:
                                    print(f"[DEBUG] ❌ id_key {id_key} NOT found in item_ids")
                            except (ValueError, TypeError) as e:
                                print(f"[DEBUG] ❌ Error converting value {value} to id_key: {e}")
                        if 'empty_slot_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['empty_slot_ids']:
                                    name = group_mappings['empty_slot_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Bank":
                        if 'slot_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['slot_ids']:
                                    name = group_mappings['slot_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                        # Only apply boolean mapping to features that are actually boolean
                        if data_type == 'boolean' and 'boolean_states' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['boolean_states']:
                                    name = group_mappings['boolean_states'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Game Objects":
                        if 'object_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['object_ids']:
                                    name = group_mappings['object_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "NPCs":
                        if 'npc_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['npc_ids']:
                                    name = group_mappings['npc_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Tabs":
                        if 'tab_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['tab_ids']:
                                    name = group_mappings['tab_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Phase Context":
                        if 'phase_type_hashes' in group_mappings:
                            try:
                                hash_key = int(float(value))
                                print(f"[DEBUG] 🔍 Looking for hash_key {hash_key} in phase_type_hashes")
                                if str(hash_key) in group_mappings['phase_type_hashes']:
                                    name = group_mappings['phase_type_hashes'][str(hash_key)]
                                    print(f"[DEBUG] ✅ Found in phase_type_hashes: {hash_key} -> '{name}'")
                                    if feature_idx in debug_features:
                                        print(f"[DEBUG] 🎯 TRANSLATION SUCCESS for feature {feature_idx}: {value} -> '{name}'")
                                    return str(name)
                                else:
                                    print(f"[DEBUG] ❌ hash_key {hash_key} NOT found in phase_type_hashes")
                            except (ValueError, TypeError) as e:
                                print(f"[DEBUG] ❌ Error converting value {value} to hash_key: {e}")
                else:
                    print(f"[DEBUG] ❌ feature_group '{feature_group}' NOT found in live_id_mappings")
                
                # Check global hash mappings as fallback
                if 'Global' in self.live_id_mappings and 'hash_mappings' in self.live_id_mappings['Global']:
                    try:
                        hash_key = int(float(value))
                        print(f"[DEBUG] 🔍 Looking for hash_key {hash_key} in Global hash_mappings")
                        if str(hash_key) in self.live_id_mappings['Global']['hash_mappings']:
                            original_value = self.live_id_mappings['Global']['hash_mappings'][str(hash_key)]
                            print(f"[DEBUG] ✅ Found in Global hash_mappings: {hash_key} -> '{original_value}'")
                            return str(original_value)
                        else:
                            print(f"[DEBUG] ❌ hash_key {hash_key} NOT found in Global hash_mappings")
                    except (ValueError, TypeError) as e:
                        print(f"[DEBUG] ❌ Error converting value {value} to hash_key: {e}")
                
                # Fallback: try the old feature-based lookup
                if show_translation and feature_idx is not None and hasattr(self, 'hash_reverse_lookup') and feature_idx in self.hash_reverse_lookup:
                    print(f"[DEBUG] 🔍 Trying fallback lookup in hash_reverse_lookup[{feature_idx}]")
                    if value in self.hash_reverse_lookup[feature_idx]:
                        original_value = self.hash_reverse_lookup[feature_idx][value]
                        print(f"[DEBUG] ✅ Found in hash_reverse_lookup[{feature_idx}]: {value} -> '{original_value}'")
                        return str(original_value)
                    else:
                        print(f"[DEBUG] ❌ value {value} NOT found in hash_reverse_lookup[{feature_idx}]")
                else:
                    if not hasattr(self, 'hash_reverse_lookup'):
                        print(f"[DEBUG] ❌ hash_reverse_lookup does NOT exist")
                    elif feature_idx is None:
                        print(f"[DEBUG] ❌ feature_idx is None for fallback lookup")
                    elif feature_idx not in self.hash_reverse_lookup:
                        print(f"[DEBUG] ❌ feature_idx {feature_idx} NOT in hash_reverse_lookup")
                
                print(f"[DEBUG] ❌ No translation found for value {value}")
            else:
                if not show_translation:
                    print(f"[DEBUG] ❌ show_translation is False")
                elif not hasattr(self, 'live_id_mappings'):
                    print(f"[DEBUG] ❌ live_id_mappings does NOT exist")
                elif not self.live_id_mappings:
                    print(f"[DEBUG] ❌ live_id_mappings is empty")
            
            # Handle special cases for non-translated values (applies to all numeric values)
            if value == 0:
                result = "0"
            elif value == -1:
                result = "-1"
            else:
                # If it's a whole number, show as integer (no .0)
                if value == int(value):
                    result = f"{int(value)}"
                else:
                    result = f"{value:.3f}"
            
            # Debug: show final result for interesting features
            if feature_idx in debug_features:
                print(f"[DEBUG] 🎯 Final result for feature {feature_idx}: '{result}'")
            
            return result
        
        return str(value)
    
    def get_feature_category(self, feature_name):
        """Get feature category from feature mappings instead of hardcoded logic - EXACTLY like browse_training_data.py"""
        # Find the feature in our mappings
        for feature_data in self.live_feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_name') == feature_name:
                return feature_data.get('feature_group', 'other')
        
        # Fallback to hardcoded logic if not found in mappings
        feature_name_lower = feature_name.lower()
        
        # Player state features (0-4)
        if "world_x" in feature_name_lower or "world_y" in feature_name_lower:
            return "Player"
        elif "animation" in feature_name_lower:
            return "Player"
        elif "moving" in feature_name_lower or "movement" in feature_name_lower:
            return "Player"
        
        # Interaction context features (5-8)
        elif "action_type" in feature_name_lower or "item_name" in feature_name_lower or "target" in feature_name_lower:
            return "Interaction"
        elif "time_since_interaction" in feature_name_lower:
            return "Phase Context"
        
        # Camera features (9-13)
        elif "camera" in feature_name_lower or "pitch" in feature_name_lower or "yaw" in feature_name_lower:
            return "Camera"
        
        # Inventory features (14-41)
        elif "inventory" in feature_name_lower or "slot" in feature_name_lower:
            return "Inventory"
        
        # Bank features (42-62)
        elif "bank" in feature_name_lower:
            return "Bank"
        
        # Phase context features (63-66)
        elif "phase" in feature_name_lower:
            return "Phase Context"
        
        # Game objects features (67-122)
        elif "game_object" in feature_name_lower or "furnace" in feature_name_lower or "bank_booth" in feature_name_lower:
            return "Game Objects"
        
        # NPC features (123-142)
        elif "npc" in feature_name_lower:
            return "NPCs"
        
        # Tab features (143)
        elif "tab" in feature_name_lower:
            return "Tabs"
        
        # Skills features (144-145)
        elif "level" in feature_name_lower or "xp" in feature_name_lower:
            return "Skills"
        
        # Timestamp feature (146)
        elif "timestamp" in feature_name_lower:
            return "Timestamp"
        
        else:
            return "other"
    
    def apply_row_coloring(self, item, category):
        """Apply color coding to table rows based on feature category - EXACTLY like browse_training_data.py"""
        colors = {
            "Player": "#e6f7ff",           # Light cyan for player state
            "Interaction": "#e6ffe6",      # Light green for interaction context
            "Camera": "#f0f0f0",           # Light gray for camera
            "Inventory": "#e6f3ff",        # Light blue for inventory items
            "Bank": "#ffe6e6",             # Light red for bank
            "Phase Context": "#e6ffe6",    # Light green for phase context
            "Game Objects": "#f0e6ff",     # Light purple for game objects
            "NPCs": "#fff2e6",             # Light orange for NPCs
            "Tabs": "#ffe6e6",             # Light red for tabs
            "Skills": "#fff7e6",           # Light yellow for skills
            "Timestamp": "#f5f5f5",        # Very light gray for timestamp
            "other": "#ffffff"             # White for other features
        }
        
        bg_color = colors.get(category, "#ffffff")
        self.live_feature_tree.tag_configure(category, background=bg_color)
        self.live_feature_tree.item(item, tags=(category,))


def main():
    """Main function."""
    print("[BOOT] starting")
    root = tk.Tk()
    print("[BOOT] tk created")
    app = BotControllerGUI(root)
    print("[BOOT] gui constructed")
    
    # Handle window close
    def on_closing():
        if app.is_running:
            if messagebox.askokcancel("Quit", "Bot is running. Stop and quit?"):
                app.stop_bot()
                app.cleanup()
                root.destroy()
        else:
            app.cleanup()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start GUI
    print("[BOOT] entering mainloop")
    root.mainloop()


if __name__ == "__main__":
    main()
