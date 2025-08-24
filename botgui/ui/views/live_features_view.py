#!/usr/bin/env python3
"""Live Features View - displays rolling 10x128 feature window using tksheet"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from ...util.formatting import format_value_for_display
from ..styles import create_dark_stringvar, create_dark_booleanvar

LOG = logging.getLogger(__name__)

try:
    from tksheet import Sheet
    TKSHEET_AVAILABLE = True
except ImportError as e:
    TKSHEET_AVAILABLE = False
    LOG.error("tksheet import failed: %s", e)
    raise ImportError("tksheet is required but not available. Install with: pip install tksheet") from e


class LiveFeaturesView(ttk.Frame):
    """View for displaying live feature data in a rolling 10x128 window using tksheet"""
    
    # --- Threading guard ----------------------------------------------------
    def _assert_main_thread(self, where: str):
        cur = threading.current_thread()
        main = threading.main_thread()
        
        if cur is not main:
            error_msg = f"Tkinter call from non-main thread in {where}: {cur.name} (main: {main.name})"
            print(f"❌ CRITICAL ERROR [live_features_view.py:50] {error_msg}")
            print(f"🔍 DEBUG [live_features_view.py:55] Current thread: {cur.name}, Main thread: {main.name}")
            # CRASH IMMEDIATELY - threading violations are critical data processing errors
            raise RuntimeError(error_msg)
        else:
            print(f"🔍 DEBUG [live_features_view.py:60] Main thread assertion passed in {where}")
    
    def __init__(self, parent, controller):
        # Precondition checks
        if parent is None:
            raise ValueError("parent cannot be None")
        if controller is None:
            raise ValueError("controller cannot be None")
        if not hasattr(controller, 'ui_state'):
            raise RuntimeError("controller.ui_state not initialized")
        
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.feature_names = None
        self.feature_groups = None
        self._last_window = None  # (10,128)
        
        # Color bits for tracking cell colors (cyan ⇄ white)
        self._color_bits: Optional[np.ndarray] = None
        
        # Schema set flag
        self._schema_set = False
        
        # UI state - sync with controller
        self.show_translations = self.controller.ui_state.show_translations
        self.feature_group_filter = "All"
        self.search_text = ""
        
        # Collapsible groups and favorites
        self.expanded_groups = set()  # Set of expanded group names
        self.favorite_features = set()  # Set of expanded group names
        self.group_rows = {}  # Map group names to their row items in the sheet
        
        # Recording state
        self.recording = False
        self.recording_session_active = False
        self.runelite_window = None
        self.click_count = 0
        self.key_press_count = 0
        self.scroll_count = 0
        self.mouse_move_count = 0
        
        # Session data
        self.session_start_time = None
        self.session_data = []
        
        # Long session support
        self.auto_save_interval = 300  # Auto-save every 5 minutes
        self.last_auto_save = 0
        self.session_file_counter = 0
        
        # Massive dataset optimization
        self.csv_buffer_size = 1000  # Buffer actions before writing
        self.csv_buffer = []
        self.last_csv_flush = 0
        self.csv_flush_interval = 10  # Flush every 10 seconds
        
        # Feature recording (live 128D feature vectors)
        self.feature_recording = False
        self.features_csv_writer = None
        self.features_csvf = None
        self.feature_buffer = []
        self.feature_count = 0
        self.last_feature_flush = 0
        
        # Action recording (mouse/keyboard events)
        self.action_recording = False
        self.csv_writer = None
        self.csvf = None
        
        # Gamestate recording (raw JSON files)
        self.gamestate_recording = False
        self.gamestate_session_dir = None
        self.gamestate_count = 0
        self.last_gamestate_check = 0
        self.gamestate_check_interval = 5  # Check for new gamestates every 5 seconds
        
        # Gamestate cleanup system (removes auto-dumped files until recording starts)
        self.gamestate_cleanup_active = False
        self.last_gamestate_cleanup = 0
        self.gamestate_cleanup_interval = 10  # Clean up every 10 seconds
        self.gamestate_cleanup_threshold = 50  # Remove files if more than 50 exist
        
        # Mouse movement rate limiting (10ms minimum interval)
        self.mouse_move_throttle = 0.01  # 10ms = 0.01 seconds
        self.last_mouse_capture = 0
        
        # Countdown state
        self.countdown_active = False
        self.countdown_seconds = 5  # Default 5 second countdown
        
        self._setup_ui()
        self._bind_events()
        
        # Sync translations variable with controller state
        self.translations_var.set(self.show_translations)
        
        # Load favorites from file
        self._load_favorites()
        
        # Initialize groups as collapsed by default (empty set means all collapsed)
        self.expanded_groups = set()  # All feature groups collapsed initially
        
        LOG.info("LiveFeaturesView: initialized with tksheet")
    
    def _setup_ui(self):
        """Setup the user interface with tksheet and integrated recorder"""
        # Configure grid weights - add second column for recorder
        self.grid_columnconfigure(0, weight=3)  # Feature table gets more space
        self.grid_columnconfigure(1, weight=1)  # Recorder gets less space
        self.grid_rowconfigure(3, weight=1)  # Table gets most space
        
        # Header
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Live Feature Tracking", 
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        
        # Controls frame
        controls_frame = ttk.Frame(self, style="Toolbar.TFrame")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Left controls
        ttk.Button(controls_frame, text="▶ Start", 
                  command=self._start_live_mode).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls_frame, text="⏹ Stop", 
                  command=self._stop_live_mode).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls_frame, text="🗑️ Clear Buffers", 
                  command=self._clear_buffers).grid(row=0, column=2, padx=(0, 12))
        
        # Center controls
        ttk.Button(controls_frame, text="📋 Copy Table", 
                  command=self._copy_to_clipboard).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(controls_frame, text="💾 Export CSV", 
                  command=self._export_to_csv).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(controls_frame, text="📁 Expand All", 
                  command=self._expand_all_groups).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(controls_frame, text="📂 Collapse All", 
                  command=self._collapse_all_groups).grid(row=0, column=5, padx=(0, 12))
        
        # Filter controls
        ttk.Label(controls_frame, text="Group:").grid(row=0, column=6, padx=(0, 4))
        self.group_combo = ttk.Combobox(controls_frame, values=["All"], width=15, state="readonly")
        self.group_combo.grid(row=0, column=7, padx=(0, 12))
        self.group_combo.set("All")
        
        ttk.Label(controls_frame, text="Search:").grid(row=0, column=8, padx=(0, 4))
        self.search_var = create_dark_stringvar(self)
        self.search_entry = ttk.Entry(controls_frame, textvariable=self.search_var, width=20)
        self.search_entry.grid(row=0, column=9, padx=(0, 12))
        
        # Right controls
        self.translations_var = create_dark_booleanvar(self, value=True)
        ttk.Checkbutton(controls_frame, text="Show Translations", 
                       variable=self.translations_var).grid(row=0, column=10)
        
        # Summary line
        self.summary_label = ttk.Label(self, text="Features: 0/128 | Buffer: 0/10 | Status: Ready", 
                                     font=("Arial", 9))
        self.summary_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Recorder controls (right side)
        self._setup_recorder_controls()
        
        # Build the columns: ["feature", "group", "t0","t1",...,"t9"]
        self._setup_columns()
        
        LOG.info("tksheet initialized; headers set; read-only; dark theme")
    
    def _setup_columns(self):
        """Setup the tksheet with proper columns"""
        # Create tksheet with proper headers - T0 (current) on the left, T9 (oldest) on the right
        headers = ["Feature", "Index", "Group", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        
        # Create parent frame for the sheet
        sheet_frame = ttk.Frame(self)
        sheet_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        sheet_frame.grid_columnconfigure(0, weight=1)
        sheet_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize tksheet
        self.sheet = Sheet(sheet_frame, headers=headers)
        self.sheet.grid(row=0, column=0, sticky="nsew")
        
        # Configure sheet properties
        self.sheet.enable_bindings()
        self.sheet.readonly = True  # Make cells read-only
        
        # Apply dark theme styling
        self.sheet.change_theme("dark")
        
        # Configure column widths
        column_widths = [200, 60, 100, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
        for i, width in enumerate(column_widths):
            self.sheet.column_width(column=i, width=width)
        # Ensure we start from 0 rows; some tksheet builds default to N blank rows.
        try:
            self.sheet.set_sheet_data([])
        except Exception:
            pass
    
    def _setup_recorder_controls(self):
        """Setup the integrated recorder controls on the right side"""
        # Main recorder frame
        recorder_frame = ttk.LabelFrame(self, text="🎮 Training Data Recorder", padding=(8, 8))
        recorder_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(8, 8), pady=(8, 8))
        recorder_frame.grid_columnconfigure(0, weight=1)
        
        # Runelite Window Detection
        window_frame = ttk.LabelFrame(recorder_frame, text="Runelite Window Detection")
        window_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        window_frame.grid_columnconfigure(1, weight=1)
        
        # Window detection status
        self.window_status_label = ttk.Label(recorder_frame, text="❌ Window not detected", 
                                          foreground="lightblue", font=("Arial", 12, "bold"))
        self.window_status_label.grid(row=8, column=0, pady=(2, 0))
        
        detect_button = ttk.Button(window_frame, text="🔍 Detect", 
                                  command=self._detect_runelite_window)
        detect_button.grid(row=0, column=1, padx=8, pady=5)
        
        # Recording Controls
        control_frame = ttk.LabelFrame(recorder_frame, text="Recording Session Controls")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        control_frame.grid_columnconfigure(1, weight=1)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, padx=8, pady=10)
        
        # Main recording session button
        self.start_session_button = ttk.Button(button_frame, text="🎬 Start Recording Session", 
                                             command=self._start_recording_session, style="Accent.TButton")
        self.start_session_button.grid(row=0, column=0, padx=5, pady=(0, 5))
        
        # Countdown display
        self.countdown_label = ttk.Label(recorder_frame, text="", 
                                       foreground="magenta", font=("Arial", 16, "bold"))
        self.countdown_label.grid(row=9, column=0, pady=(2, 0))
        
        # Countdown configuration
        countdown_frame = ttk.Frame(button_frame)
        countdown_frame.grid(row=2, column=0, padx=5, pady=(0, 5))
        
        ttk.Label(countdown_frame, text="Countdown:").grid(row=0, column=0, padx=(0, 5))
        self.countdown_var = tk.StringVar(value="5")
        countdown_spinbox = ttk.Spinbox(countdown_frame, from_=3, to=30, width=5, 
                                       textvariable=self.countdown_var, command=self._update_countdown)
        countdown_spinbox.grid(row=0, column=1, padx=(0, 5))
        ttk.Label(countdown_frame, text="seconds").grid(row=0, column=2)
        
        # Auto-save configuration
        autosave_frame = ttk.Frame(button_frame)
        autosave_frame.grid(row=3, column=0, padx=5, pady=(0, 5))
        
        ttk.Label(autosave_frame, text="Auto-save:").grid(row=0, column=0, padx=(0, 5))
        self.autosave_var = tk.StringVar(value="5")
        autosave_spinbox = ttk.Spinbox(autosave_frame, from_=1, to=30, width=5, 
                                       textvariable=self.autosave_var, command=self._update_autosave)
        autosave_spinbox.grid(row=0, column=1, padx=(0, 5))
        ttk.Label(autosave_frame, text="minutes").grid(row=0, column=2)
        
        # Individual controls
        controls_subframe = ttk.Frame(button_frame)
        controls_subframe.grid(row=4, column=0, padx=0, pady=0)
        
        self.start_recording_button = ttk.Button(controls_subframe, text="▶ Start Recording", 
                                               command=self._start_recording, state="disabled")
        self.start_recording_button.grid(row=0, column=0, padx=5)
        
        self.stop_recording_button = ttk.Button(controls_subframe, text="⏹ Stop Recording", 
                                              command=self._stop_recording, state="disabled")
        self.stop_recording_button.grid(row=0, column=1, padx=5)
        
        # Session management
        session_frame = ttk.Frame(control_frame)
        session_frame.grid(row=1, column=0, padx=8, pady=(5, 0))
        
        self.clear_session_button = ttk.Button(session_frame, text="🗑️ Clear Session", 
                                             command=self._clear_recording_session)
        self.clear_session_button.grid(row=0, column=0, padx=5)
        
        self.export_session_button = ttk.Button(session_frame, text="💾 Export Session", 
                                              command=self._export_recording_session, state="disabled")
        self.export_session_button.grid(row=0, column=1, padx=5)
        
        # Action Counts
        counts_frame = ttk.LabelFrame(recorder_frame, text="Action Counts (This Session)")
        counts_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        counts_frame.grid_columnconfigure(1, weight=1)
        
        self.move_count_label = ttk.Label(counts_frame, text="Mouse Moves: 0")
        self.move_count_label.grid(row=0, column=0, padx=8, pady=2, sticky="w")
        
        self.click_count_label = ttk.Label(counts_frame, text="Clicks: 0")
        self.click_count_label.grid(row=0, column=1, padx=8, pady=2, sticky="w")
        
        self.scroll_count_label = ttk.Label(counts_frame, text="Scrolls: 0")
        self.scroll_count_label.grid(row=1, column=0, padx=8, pady=2, sticky="w")
        
        self.key_press_label = ttk.Label(counts_frame, text="Key Presses: 0")
        self.key_press_label.grid(row=1, column=1, padx=8, pady=2, sticky="w")
        
        # Process step indicator
        self.process_step_label = ttk.Label(recorder_frame, text="Step 1/5: Start recording session", 
                                         foreground="#E74C3C", font=("Arial", 14, "bold"))  # Medium red
        self.process_step_label.grid(row=4, column=0, pady=(8, 0))
        
        # File paths and technical info with copy button
        file_info_frame = ttk.Frame(recorder_frame)
        file_info_frame.grid(row=5, column=0, pady=(6, 0), sticky="ew")
        file_info_frame.grid_columnconfigure(0, weight=1)  # Label gets most space
        
        self.file_info_label = ttk.Label(file_info_frame, text="", 
                                       foreground="#FF6B6B", font=("Arial", 12, "bold"))  # Light coral red
        self.file_info_label.grid(row=0, column=0, sticky="w")
        
        # Copy button for gamestate directory path
        self.copy_path_button = ttk.Button(file_info_frame, text="📋 Copy Path", 
                                         command=self._copy_gamestate_path, 
                                         style="Accent.TButton", width=12)
        self.copy_path_button.grid(row=0, column=1, padx=(10, 0), sticky="e")
        self.copy_path_button.config(state="disabled")  # Initially disabled until path is available
        
        # Watchdog status (detailed)
        self.watchdog_status_label = ttk.Label(recorder_frame, text="🔍 Watchdog: Initializing...", 
                                             foreground="#E85A71", font=("Arial", 11))  # Light burgundy
        self.watchdog_status_label.grid(row=6, column=0, pady=(6, 0))
        
        # Live mode status
        self.live_mode_label = ttk.Label(recorder_frame, text="🔄 Live Mode: Inactive", 
                                       foreground="#D66853", font=("Arial", 12, "bold"))  # Light brick red
        self.live_mode_label.grid(row=7, column=0, pady=(6, 0))
        
        # Recording status
        self.recorder_status_label = ttk.Label(recorder_frame, text="Ready to start recording session", 
                                            foreground="#C44569", font=("Arial", 12, "bold"))  # Medium burgundy
        self.recorder_status_label.grid(row=8, column=0, pady=(6, 0))
        
        # Action counts display
        self.action_counts_label = ttk.Label(recorder_frame, text="📊 Actions: 0 clicks, 0 keys, 0 scrolls, 0 moves", 
                                          foreground="#A93226", font=("Arial", 12, "bold"))  # Darker burgundy
        self.action_counts_label.grid(row=9, column=0, pady=(6, 0))
        
        # Start periodic updates
        self.after(100, self._update_recorder_status)
        
        # Start auto-save timer for long sessions
        self.after(1000, self._check_auto_save)
        
        # Start gamestate cleanup system (removes auto-dumped files until recording starts)
        self.gamestate_cleanup_active = True
        self.after(1000, self._cleanup_gamestates_if_needed)
        print(f"🧹 Started gamestate cleanup system - will remove files every {self.gamestate_cleanup_interval}s until recording starts")

    def _bind_events(self):
        """Bind UI events"""
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_change)
        self.search_var.trace("w", self._on_search_change)
        self.translations_var.trace("w", self._on_translations_change)
        
        # NEW: make group headers clickable in the sheet
        try:
            # Works on recent tksheet
            self.sheet.extra_bindings([
                ("cell_select", self._on_sheet_cell_click),
                ("double_click_cell", self._on_sheet_cell_click),
            ])
            LOG.info("tksheet extra_bindings for clicks installed")
        except Exception as e:
            LOG.warning("tksheet extra_bindings not available: %s (group toggling by click disabled)", e)
        
        # Bind to controller for live feature updates
        if hasattr(self.controller, 'bind_live_features_view'):
            self.controller.bind_live_features_view(self)
        
        # The controller's UI pump drives updates; no local queue poller needed
        # self._start_feature_listener()
    
    def _on_group_change(self, event):
        """Handle group filter change"""
        self.feature_group_filter = self.group_combo.get()
        self._refresh_table()
    
    def _on_search_change(self, *args):
        """Handle search text change"""
        self.search_text = self.search_var.get().lower()
        self._refresh_table()
    
    def _on_translations_change(self, *args):
        """Handle translations toggle change"""
        self.show_translations = self.translations_var.get()
        # Update controller state
        self.controller.ui_state.show_translations = self.show_translations
        self._refresh_table()
    
    def _on_sheet_cell_click(self, event):
        """
        Toggle a group when its header row is clicked. Works across tksheet versions by
        robustly parsing the event argument (dict or tuple).
        """
        try:
            # tksheet >= 6.x typically passes a dict, older may pass a tuple/list
            row = col = None
            if isinstance(event, dict):
                # Try to get row/col from the selected field first
                if 'selected' in event and hasattr(event['selected'], 'row') and hasattr(event['selected'], 'column'):
                    row = event['selected'].row
                    col = event['selected'].column
                else:
                    # Fallback to direct keys
                    row = event.get("row")
                    col = event.get("column")
            elif isinstance(event, (tuple, list)) and len(event) >= 2:
                row, col = event[0], event[1]

            if row is None:
                return

            # Check if click is on a group header row
            group_name = None
            for gname, ginfo in self.group_rows.items():
                if ginfo['header_row'] == row:
                    group_name = gname
                    break
            
            if group_name and (col in (None, 0)):
                self.toggle_group(group_name)  # this calls _refresh_table and logs new state
                return "break"  # prevent tksheet changing selection focus further on this click
        except Exception:
            LOG.exception("_on_sheet_cell_click failed")
    
    def set_schema(self, feature_names: List[str], feature_groups: List[str]):
        """
        Builds the 128 rows; sets "feature", "index", "group", blanks time cells.
        
        Args:
            feature_names: List of 128 feature names
            feature_groups: List of 128 feature groups
            
        Raises:
            ValueError: If lists don't have length 128
        """
        try:
            print("🔍 DEBUG: set_schema called")
            print(f"🔍 DEBUG: feature_names length: {len(feature_names)}")
            print(f"🔍 DEBUG: feature_groups length: {len(feature_groups)}")
            
            self._assert_main_thread("set_schema")
            print("🔍 DEBUG: Main thread assertion passed")
            
            # Validate inputs
            if len(feature_names) != 128:
                raise ValueError(f"feature_names must have length 128, got {len(feature_names)}")
            if len(feature_groups) != 128:
                raise ValueError(f"feature_groups must have length 128, got {len(feature_groups)}")
            
            print("🔍 DEBUG: Input validation passed")
            
            self.feature_names = list(feature_names)
            self.feature_groups = list(feature_groups)
            print("🔍 DEBUG: Feature names and groups stored")
            
            # Initialize color bits
            self._color_bits = np.zeros((10, 128), dtype=bool)
            print("🔍 DEBUG: Color bits initialized")
            
            # Mark schema as set
            self._schema_set = True
            print("🔍 DEBUG: Schema marked as set")
            
            # Build the collapsible table
            print("🔍 DEBUG: About to refresh table...")
            self._refresh_table()
            print("🔍 DEBUG: Table refreshed successfully")
            
            LOG.info("Schema set: %d feature names, %d feature groups", len(feature_names), len(feature_groups))
            print("🔍 DEBUG: set_schema completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in set_schema: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _realize_all_rows(self):
        """Create rows for all 128 features"""
        # Hard reset existing rows to avoid blank tail rows
        self._reset_sheet_rows()
        
        # Create rows for all features
        for i in range(128):
            name = self.feature_names[i]
            group = self.feature_groups[i]
            
            # Create row data: Feature, Index, Group, blank T0..T9
            row_data = [name, str(i), group] + [""] * 10  # 10 time columns
            
            # Insert row
            self.sheet.insert_row(row_data, idx=i)
        
        # Force the sheet to redraw
        self.sheet.refresh()
        
        LOG.info("Realized all 128 feature rows")
    
    def _reset_sheet_rows(self):
        """Hard reset the sheet to 0 rows (clear() only wipes values)."""
        try:
            # Preferred API — replaces all data and resets row count
            self.sheet.set_sheet_data([])
            return
        except Exception:
            pass
        # Fallback: delete all existing rows
        total = 0
        try:
            # some tksheet builds expose get_total_rows()
            total = self.sheet.get_total_rows()
        except Exception:
            try:
                data = self.sheet.get_sheet_data()
                total = len(data) if data is not None else 0
            except Exception:
                total = 0
        for _ in range(int(total)):
            try:
                self.sheet.delete_row(0)
            except Exception:
                break
    
    def update_translations_state(self, show: bool):
        """Enable/disable translations and re-render the current window."""
        self._assert_main_thread("update_translations_state")
        self.show_translations = bool(show)
        try:
            # keep the local checkbox (if present) in sync
            self.translations_var.set(self.show_translations)
        except Exception:
            pass

        # Force a full repaint using the last window so labels update immediately
        if getattr(self, "_last_window", None) is not None:
            mask = np.ones_like(self._last_window, dtype=bool)
            self.update_from_window(self._last_window, changed_mask=mask)
        else:
            if hasattr(self, "sheet"):
                self.sheet.refresh()
    
    def update_from_window(self, window: np.ndarray, changed_mask=None):
        """
        Update table with window data using changed_mask for change detection.
        
        Args:
            window: np.ndarray shape (10,128), time rows t0..t9, features columns f0..f127
            changed_mask: Boolean mask indicating changed cells, or None to compute automatically
            
        Raises:
            RuntimeError: If schema not set or shapes don't match
        """
        try:
            # Check if this is a duplicate update with the same data
            if (self._last_window is not None and 
                self._last_window.shape == window.shape and 
                np.array_equal(self._last_window, window)):
                print("🔍 DEBUG: Skipping duplicate update - same data")
                return
            
            print("🔍 DEBUG: update_from_window called")
            print(f"🔍 DEBUG: window shape: {window.shape}")
            
            self._assert_main_thread("update_from_window")
            
            # Check schema is set
            if self.feature_names is None:
                print("❌ ERROR: Schema not set - feature_names is None")
                raise RuntimeError("update_from_window called before set_schema()")
            print("🔍 DEBUG: Schema check passed")

            if window.shape != (10, 128):
                print(f"❌ ERROR: Invalid window shape: {window.shape}")
                raise ValueError(f"window shape {window.shape} != (10,128)")

            # If no mask provided, compute vs last window
            if changed_mask is None:
                if self._last_window is None or self._last_window.shape != window.shape:
                    changed_mask = np.ones_like(window, dtype=bool)
                    print("🔍 DEBUG: Using full change mask (first update or shape change)")
                else:
                    changed_mask = (window != self._last_window)
                    print(f"🔍 DEBUG: Computed change mask - {np.sum(changed_mask)} cells changed")
            else:
                print(f"🔍 DEBUG: Using provided change mask - {np.sum(changed_mask)} cells changed")

            # Check if there are actually any changes
            if not np.any(changed_mask):
                print("🔍 DEBUG: No changes detected, skipping update")
                self._last_window = window.copy()
                return

            # Iterate features, update all 10 time columns for changed positions
            updated = 0

            for f_idx in range(128):
                # Find the actual row in the sheet for this feature
                sheet_row = self._get_feature_sheet_row(f_idx)
                if sheet_row is None:
                    continue  # Feature is hidden (collapsed group)
                
                for t_idx in range(10):  # t0..t9 LEFT→RIGHT
                    if not changed_mask[t_idx, f_idx]:
                        continue

                    value = window[t_idx, f_idx]
                    
                    # write value (row=sheet_row, col=3+t_idx)
                    # Columns: 0=Feature, 1=Index, 2=Group, time starts at col=3
                    col = 3 + t_idx

                    # Try to translate the raw value if translations are enabled
                    mapped = None
                    if getattr(self, "show_translations", False):
                        try:
                            # Get the feature's group as a hint from the table's "Group" column (col index 2).
                            group_hint = None
                            if hasattr(self, "sheet"):
                                try:
                                    group_hint = self.sheet.get_cell_data(sheet_row, 2)
                                except Exception:
                                    group_hint = None

                            mapped = self.controller.mapping_service.translate(f_idx, value, group_hint=group_hint)
                        except Exception:
                            mapped = None  # fail safe: fall back to raw value

                    # Prefer mapped label when available; otherwise show the raw number
                    text = mapped if (mapped is not None and mapped != "") else f"{value:.0f}"
                    self.sheet.set_cell_data(sheet_row, col, text)

                    # flip color on each change
                    self._color_bits[t_idx, f_idx] = ~self._color_bits[t_idx, f_idx]
                    new_color = "#00b3b3" if self._color_bits[t_idx, f_idx] else "#ffffff"
                    self.sheet.highlight_cells(row=sheet_row, column=col, fg=new_color, redraw=False)

                    updated += 1

            print(f"🔍 DEBUG: Updated {updated} cells")

            # Refresh the sheet after all updates
            self.sheet.refresh()
            
            # Store last window for next comparison
            self._last_window = window.copy()

            # Persist latest feature vector when recording is active
            try:
                if getattr(self, "feature_recording", False) and getattr(self, "features_csv_writer", None):
                    # Convention used elsewhere: row 0 is the most recent timestep (T0)
                    latest_vec = window[0, :].tolist()
                    ts_ms = int(time.time() * 1000)
                    self._save_feature_vector(latest_vec, ts_ms)
            except Exception as e:
                LOG.warning("Failed to save live feature vector: %s", e)
            
            print("🔍 DEBUG: update_from_window completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in update_from_window: {e}")
            import traceback
            traceback.print_exc()
            raise
        self._update_actions_group()
    
    def _update_actions_group(self):
        """Update the Actions group with current values"""
        if "Actions" not in self.group_rows:
            return
        
        try:
            # Get current action tensors from controller
            action_tensors = self.controller.get_action_features()
            if not action_tensors or len(action_tensors) < 10:
                LOG.debug("No action tensors available for update")
                return
            
            # Update each action feature row
            for row_idx in self.group_rows["Actions"]['feature_rows']:
                # Get the feature name to determine which action type to count
                feature_name = self.sheet.get_cell_data(row_idx, 0)
                
                # Update T0-T9 columns with aggregated values from each timestep's tensor
                for t in range(10):
                    col_idx = 3 + t  # T0 starts at column 3
                    if t < len(action_tensors):
                        timestep_tensor = action_tensors[t]
                        if len(timestep_tensor) > 0:
                            if feature_name == "Action Count":
                                # Action count is always first element
                                value = timestep_tensor[0] if len(timestep_tensor) > 0 else 0
                                self.sheet.set_cell_data(row_idx, col_idx, f"{int(value)}")
                            else:
                                # For other features, count occurrences in the tensor
                                count = self._count_action_type_in_tensor(timestep_tensor, feature_name.lower().replace(" ", "_"))
                                self.sheet.set_cell_data(row_idx, col_idx, f"{count}")
                        else:
                            self.sheet.set_cell_data(row_idx, col_idx, "0")
                    else:
                        self.sheet.set_cell_data(row_idx, col_idx, "0")
            
            
        except Exception as e:
            LOG.error(f"Error updating Actions group: {e}")
    
    def _get_feature_sheet_row(self, feature_idx: int) -> Optional[int]:
        """Return the visible sheet row index for a feature, or None if hidden/collapsed."""
        # Defensive guards
        if not getattr(self, "_schema_set", False):
            return None
        if feature_idx is None or feature_idx < 0 or feature_idx >= len(self.feature_names):
            return None

        # Resolve group and ensure it's currently expanded
        try:
            group_name = self.feature_groups[feature_idx]
        except Exception:
            return None
        if group_name not in getattr(self, "expanded_groups", set()):
            return None

        group = getattr(self, "group_rows", {}).get(group_name)
        if not group:
            return None

        # Scan the group's feature rows for this feature's name
        feature_name = self.feature_names[feature_idx]
        for row_idx in group.get("feature_rows", []):
            try:
                if self.sheet.get_cell_data(row_idx, 0) == feature_name:
                    return row_idx
            except Exception:
                continue
        return None
    
    def _refresh_table(self):
        """Refresh the feature table with collapsible groups and favorites"""
        try:
            print("🔍 DEBUG: _refresh_table called")
            self._assert_main_thread("_refresh_table")
            print("🔍 DEBUG: Main thread assertion passed")
            
            if not self._schema_set:
                print("❌ ERROR: Schema not set in _refresh_table")
                LOG.error("LiveFeaturesView: CRITICAL ERROR - schema not set in _refresh_table")
                return
            
            print("🔍 DEBUG: Schema check passed")
            
            # Update group combo with unique groups
            if self.feature_groups:
                unique_groups = sorted(list(set(self.feature_groups)))
                current_groups = ["All"] + unique_groups
                self.group_combo['values'] = current_groups
                if self.feature_group_filter not in current_groups:
                    self.feature_group_filter = "All"
                    self.group_combo.set("All")
                print(f"🔍 DEBUG: Updated group combo with {len(current_groups)} groups")
            
            # Clear the sheet and rebuild with collapsible groups
            if hasattr(self, 'sheet'):
                print("🔍 DEBUG: About to HARD-RESET sheet rows and rebuild table.")
                self._reset_sheet_rows()
                print("🔍 DEBUG: Sheet rows reset to 0")
                self._build_collapsible_table()
                print("🔍 DEBUG: Collapsible table built")
            else:
                print("❌ ERROR: No sheet available in _refresh_table")
            
            # If we have current data, update the table with it
            if hasattr(self, '_last_window') and self._last_window is not None:
                try:
                    # Force an update with the current window data
                    mask = np.ones_like(self._last_window, dtype=bool)
                    self.update_from_window(self._last_window, changed_mask=mask)
                except Exception as e:
                    print(f"❌ ERROR updating table with current data: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("🔍 DEBUG: No current window data to update with")
            
            print("🔍 DEBUG: _refresh_table completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in _refresh_table: {e}")
            import traceback
            traceback.print_exc()
            LOG.exception("LiveFeaturesView: _refresh_table failed with error")
            # Don't re-raise - just log and return gracefully
    
    def _build_collapsible_table(self):
        """Build the table with collapsible feature groups"""
        if not self.feature_names or not self.feature_groups:
            LOG.debug("_build_collapsible_table: no feature names or groups")
            return
        
        LOG.info(f"_build_collapsible_table: building table with {len(self.feature_names)} features in {len(set(self.feature_groups))} groups")
        
        # Group features by their feature group
        grouped_features = {}
        for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
            if group not in grouped_features:
                grouped_features[group] = []
            grouped_features[group].append((i, name, group))
        
        # Sort groups by their first feature index (Player first, then Interaction, Camera, etc.)
        def get_group_order(group_name):
            if group_name == "Player":
                return 0
            elif group_name == "Interaction":
                return 1
            elif group_name == "Camera":
                return 2
            elif group_name == "Inventory":
                return 3
            elif group_name == "Bank":
                return 4
            elif group_name == "Phase Context":
                return 5
            elif group_name == "Game Objects":
                return 6
            elif group_name == "NPCs":
                return 7
            elif group_name == "Tabs":
                return 8
            elif group_name == "Skills":
                return 9
            elif group_name == "Timestamp":
                return 10
            elif group_name == "Actions":
                return 11
            else:
                return 999  # Unknown groups go last
        
        # Ensure Actions group is always included
        all_groups = set(grouped_features.keys())
        all_groups.add("Actions")  # Actions group is always available
        
        sorted_groups = sorted(all_groups, key=get_group_order)
        
        row_idx = 0
        self.group_rows = {}  # Track which rows belong to which groups
        
        for group_name in sorted_groups:
            # Skip Actions group here - it will be handled separately
            if group_name == "Actions":
                continue
                
            features = grouped_features[group_name]
            
            # Check if group should be shown based on filter
            if self.feature_group_filter != "All" and group_name != self.feature_group_filter:
                continue
            
            # Check if group should be shown based on search
            if self.search_text:
                group_has_match = any(self.search_text in name.lower() for _, name, _ in features)
                if not group_has_match:
                    continue
            
            # Add group header row
            is_expanded = group_name in self.expanded_groups
            expand_icon = "▼" if is_expanded else "▶"
            group_header = [f"{expand_icon} {group_name}", "", "", "", "", "", "", "", "", "", "", "", ""]
            
            self.sheet.insert_row(group_header, idx=row_idx)
            
            # Style the group header row
            self.sheet.highlight_cells(row=row_idx, column=0, bg="#4a5568", fg="#ffffff")
            self.sheet.highlight_cells(row=row_idx, column=1, bg="#4a5568", fg="#ffffff")
            self.sheet.highlight_cells(row=row_idx, column=2, bg="#4a5568", fg="#ffffff")
            
            # Store group row info
            self.group_rows[group_name] = {
                'header_row': row_idx,
                'feature_rows': [],
                'expanded': is_expanded
            }
            
            row_idx += 1
            
            # Add feature rows if group is expanded
            if is_expanded:
                for feature_idx, name, group in features:
                    # Check if feature matches search
                    if self.search_text and self.search_text not in name.lower():
                        continue
                    
                    # Create feature row data with current values
                    feature_row = [name, str(feature_idx), group]
                    
                    # Fill in current feature values if available
                    if hasattr(self, '_last_window') and self._last_window is not None:
                        try:
                            # Get the current value for this feature from the last window
                            current_value = self._last_window[-1, feature_idx]  # Use the most recent timestep
                            # Try to translate the value if translations are enabled
                            mapped = None
                            if getattr(self, "show_translations", False):
                                try:
                                    mapped = self.controller.mapping_service.translate(feature_idx, current_value, group_hint=group_name)
                                except Exception:
                                    mapped = None
                            # Use translated value if available, otherwise raw value
                            display_value = mapped if (mapped is not None and mapped != "") else f"{current_value:.0f}"
                            feature_row.extend([display_value] * 10)  # T0-T9 columns all show current value
                        except Exception as e:
                            feature_row.extend([""] * 10)  # T0-T9 columns
                    else:
                        feature_row.extend([""] * 10)  # T0-T9 columns
                    
                    # Insert feature row
                    self.sheet.insert_row(feature_row, idx=row_idx)
                    
                    # Style favorite features
                    if feature_idx in self.favorite_features:
                        self.sheet.highlight_cells(row=row_idx, column=0, bg="#2c5282", fg="#ffffff")
                    
                    # Store feature row info
                    self.group_rows[group_name]['feature_rows'].append(row_idx)
                    
                    row_idx += 1
        
        # Add Actions group if it's in the sorted groups
        if "Actions" in sorted_groups:
            self._add_actions_group(row_idx)
    
    def _add_actions_group(self, start_row_idx: int):
        """Add the Actions group to the table"""
        
        row_idx = start_row_idx
        
        # Add Actions group header
        expand_icon = "▼" if "Actions" in self.expanded_groups else "▶"
        group_header = [f"{expand_icon} Actions", "", "", "", "", "", "", "", "", "", "", "", ""]
        
        self.sheet.insert_row(group_header, idx=row_idx)
        
        # Style the group header row
        self.sheet.highlight_cells(row=row_idx, column=0, bg="#4a5568", fg="#ffffff")
        self.sheet.highlight_cells(row=row_idx, column=1, bg="#4a5568", fg="#ffffff")
        self.sheet.highlight_cells(row=row_idx, column=2, bg="#4a5568", fg="#ffffff")
        
        # Store group row info
        self.group_rows["Actions"] = {
            'header_row': row_idx,
            'feature_rows': [],
            'expanded': "Actions" in self.expanded_groups
        }
        
        row_idx += 1
        
        # Add action feature rows only if group is expanded
        if "Actions" in self.expanded_groups:
            # Define meaningful action features based on aggregated counts
            action_features = [
                ("Action Count", "count"),           # Total actions in window
                ("Mouse Movements", "mouse_movements"), # Sum of mouse movements in window
                ("Clicks", "clicks"),               # Sum of clicks in window
                ("Key Presses", "key_presses"),     # Sum of key presses in window
                ("Key Releases", "key_releases"),   # Sum of key releases in window
                ("Scrolls", "scrolls")              # Sum of scrolls in window
            ]
            
            for feature_name, feature_key in action_features:
                # Create action feature row
                feature_row = [feature_name, f"action_{feature_key}", "Actions"]
                
                # Get current action tensors from controller
                try:
                    action_tensors = self.controller.get_action_features()
                    if len(action_tensors) >= 10:  # Should have 10 timesteps (T0-T9)
                        # Fill T0-T9 columns with aggregated values from each timestep's tensor
                        for t in range(10):
                            if t < len(action_tensors):
                                timestep_tensor = action_tensors[t]
                                if len(timestep_tensor) > 0:
                                    if feature_key == "count":
                                        # Action count is always first element
                                        value = timestep_tensor[0] if len(timestep_tensor) > 0 else 0
                                        feature_row.append(f"{int(value)}")
                                    else:
                                        # For other features, count occurrences in the tensor
                                        count = self._count_action_type_in_tensor(timestep_tensor, feature_key)
                                        feature_row.append(f"{count}")
                                else:
                                    feature_row.append("0")
                            else:
                                feature_row.append("0")
                    else:
                        feature_row.extend(["0"] * 10)
                except Exception as e:
                    feature_row.extend(["0"] * 10)
                
                # Insert action feature row
                self.sheet.insert_row(feature_row, idx=row_idx)
                
                # Store feature row info
                self.group_rows["Actions"]['feature_rows'].append(row_idx)
                
                row_idx += 1
    
    def _count_action_type_in_tensor(self, tensor: List[float], action_type: str) -> int:
        """Count occurrences of a specific action type in an action tensor"""
        if not tensor or len(tensor) < 1:
            return 0
        
        count = 0
        # Action tensor structure: [count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, timestamp2, type2, x2, y2, button2, key2, scroll_dx2, scroll_dy2, ...]
        # Action types: 0=move, 1=click, 2=key_press, 3=key_release, 4=scroll
        
        # Start from index 2 (first action type) and step by 8 (each action has 8 elements)
        for i in range(2, len(tensor), 8):
            if i < len(tensor):
                action_type_code = int(tensor[i])
                
                if action_type == "mouse_movements" and action_type_code == 0:
                    count += 1
                elif action_type == "clicks" and action_type_code == 1:
                    count += 1
                elif action_type == "key_presses" and action_type_code == 2:
                    count += 1
                elif action_type == "key_releases" and action_type_code == 3:
                    count += 1
                elif action_type == "scrolls" and action_type_code == 4:
                    count += 1
        
        return count
    
    def _insert_feature_row(self, feature_idx: int, name: str, group: str, translate_func, row_idx: int):
        """Insert a single feature row into the sheet"""
        # Create row data with proper column mapping
        values = [name, str(feature_idx), group]
        
        # Add blank values for T0..T9 (will be filled by update_table)
        values.extend([""] * 10)
        
        # Insert row
        self.sheet.insert_row(values, idx=row_idx)
        
        # Apply styling based on favorite status
        if feature_idx in self.favorite_features:
            # Highlight favorite rows
            self.sheet.highlight_cells(row=row_idx, column=0, bg="#2c5282", fg="#ffffff")
    
    def _load_favorites(self):
        """Load favorites from file"""
        try:
            import json
            from pathlib import Path
            
            favorites_file = Path("data/favorites.json")
            if favorites_file.exists():
                with open(favorites_file, 'r') as f:
                    self.favorite_features = set(json.load(f))
            else:
                self.favorite_features = set()
        except Exception as e:
            LOG.error(f"Failed to load favorites: {e}")
            self.favorite_features = set()
    
    def _save_favorites(self):
        """Save favorites to file"""
        try:
            import json
            from pathlib import Path
            
            favorites_file = Path("data/favorites.json")
            favorites_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(favorites_file, 'w') as f:
                json.dump(list(self.favorite_features), f)
        except Exception as e:
            LOG.error(f"Failed to save favorites: {e}")
    
    def toggle_favorite(self, feature_idx: int):
        """Toggle favorite status for a feature"""
        if feature_idx in self.favorite_features:
            self.favorite_features.remove(feature_idx)
        else:
            self.favorite_features.add(feature_idx)
        
        self._save_favorites()
        self._refresh_table()
        self._update_summary()
    
    def toggle_group(self, group_name: str):
        """Toggle expansion state for a group"""
        try:
            print(f"🔍 DEBUG: toggle_group called for group: {group_name}")
            
            # Toggle the expansion state
            if group_name in self.expanded_groups:
                print(f"🔍 DEBUG: Collapsing group: {group_name}")
                self.expanded_groups.remove(group_name)
            else:
                print(f"🔍 DEBUG: Expanding group: {group_name}")
                self.expanded_groups.add(group_name)

            # Simpler and safer: rebuild the table from current state to avoid index drift
            self._refresh_table()
            self._update_summary()
            print(f"🔍 DEBUG: Group {group_name} toggle completed")
            
        except Exception as e:
            print(f"❌ ERROR in toggle_group: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to full refresh if something goes wrong
            print("🔍 DEBUG: Falling back to full table refresh")
            self._refresh_table()
            self._update_summary()
    
    def _expand_group_rows(self, group_name: str):
        """Efficiently expand a group by adding its feature rows"""
        try:
            print(f"🔍 DEBUG: _expand_group_rows called for group: {group_name}")
            
            if not hasattr(self, 'group_rows') or group_name not in self.group_rows:
                print(f"⚠️ WARNING: Group {group_name} not found in group_rows")
                return
            
            group_info = self.group_rows[group_name]
            header_row = group_info['header_row']
            
            # Special handling for Actions group
            if group_name == "Actions":
                print(f"🔍 DEBUG: Expanding Actions group with special handling")
                self._add_actions_group_rows(header_row)
                # Ensure Actions group is positioned correctly after expanding
                self._ensure_actions_group_position()
                return
            
            # Find the features for this group
            group_features = []
            for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
                if group == group_name:
                    group_features.append((i, name, group))
            
            if not group_features:
                print(f"⚠️ WARNING: No features found for group {group_name}")
                return
            
            print(f"🔍 DEBUG: Adding {len(group_features)} feature rows for group {group_name}")
            
            # Insert feature rows after the header row
            current_row = header_row + 1
            for feature_idx, name, group in group_features:
                # Create feature row data
                feature_row = [name, str(feature_idx), group]
                
                # Fill in current feature values if available
                if hasattr(self, '_last_window') and self._last_window is not None:
                    try:
                        current_value = self._last_window[-1, feature_idx]
                        # Try to translate the value if translations are enabled
                        mapped = None
                        if getattr(self, "show_translations", False):
                            try:
                                mapped = self.controller.mapping_service.translate(feature_idx, current_value, group_hint=group_name)
                            except Exception:
                                mapped = None
                        # Use translated value if available, otherwise raw value
                        display_value = mapped if (mapped is not None and mapped != "") else f"{current_value:.0f}"
                        feature_row.extend([display_value] * 10)  # T0-T9 columns
                    except Exception as e:
                        print(f"⚠️ WARNING: Failed to get current value for feature {feature_idx}: {e}")
                        feature_row.extend([""] * 10)  # T0-T9 columns
                else:
                    feature_row.extend([""] * 10)  # T0-T9 columns
                
                # Insert the feature row
                self.sheet.insert_row(feature_row, idx=current_row)
                
                # Style favorite features
                if feature_idx in self.favorite_features:
                    self.sheet.highlight_cells(row=current_row, column=0, bg="#2c5282", fg="#ffffff")
                
                # Store the row info
                group_info['feature_rows'].append(current_row)
                current_row += 1
            
            # Update the expand icon to show expanded state
            expand_icon = "▼"
            self.sheet.set_cell_data(header_row, 0, f"{expand_icon} {group_name}")
            
            # Ensure Actions group is positioned correctly after expanding
            self._ensure_actions_group_position()
            
            print(f"🔍 DEBUG: Successfully expanded group {group_name}")
            
        except Exception as e:
            print(f"❌ ERROR in _expand_group_rows: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_actions_group_rows(self, header_row: int):
        """Add action feature rows for the Actions group"""
        try:
            print(f"🔍 DEBUG: _add_actions_group_rows called for header row {header_row}")
            
            # Define meaningful action features based on aggregated counts
            action_features = [
                ("Action Count", "count"),           # Total actions in window
                ("Mouse Movements", "mouse_movements"), # Sum of mouse movements in window
                ("Clicks", "clicks"),               # Sum of clicks in window
                ("Key Presses", "key_presses"),     # Sum of key presses in window
                ("Key Releases", "key_releases"),   # Sum of key releases in window
                ("Scrolls", "scrolls")              # Sum of scrolls in window
            ]
            
            current_row = header_row + 1
            
            for feature_name, feature_key in action_features:
                # Create action feature row
                feature_row = [feature_name, f"action_{feature_key}", "Actions"]
                
                # Get current action tensors from controller
                try:
                    action_tensors = self.controller.get_action_features()
                    if len(action_tensors) >= 10:  # Should have 10 timesteps (T0-T9)
                        # Fill T0-T9 columns with aggregated values from each timestep's tensor
                        for t in range(10):
                            if t < len(action_tensors):
                                timestep_tensor = action_tensors[t]
                                if len(timestep_tensor) > 0:
                                    if feature_key == "count":
                                        # Action count is always first element
                                        value = timestep_tensor[0] if len(timestep_tensor) > 0 else 0
                                        feature_row.append(f"{int(value)}")
                                    else:
                                        # For other features, count occurrences in the tensor
                                        count = self._count_action_type_in_tensor(timestep_tensor, feature_key)
                                        feature_row.append(f"{count}")
                                else:
                                    feature_row.append("0")
                            else:
                                feature_row.append("0")
                    else:
                        feature_row.extend(["0"] * 10)
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to get action features: {e}")
                    feature_row.extend(["0"] * 10)
                
                # Insert action feature row
                self.sheet.insert_row(feature_row, idx=current_row)
                
                # Store feature row info
                self.group_rows["Actions"]['feature_rows'].append(current_row)
                
                current_row += 1
            
            # Update the expand icon to show expanded state
            expand_icon = "▼"
            self.sheet.set_cell_data(header_row, 0, f"{expand_icon} Actions")
            
            # Ensure Actions group is positioned correctly after expanding
            self._ensure_actions_group_position()
            
            print(f"🔍 DEBUG: Successfully added {len(action_features)} action feature rows")
            
        except Exception as e:
            print(f"❌ ERROR in _add_actions_group_rows: {e}")
            import traceback
            traceback.print_exc()
    
    def _collapse_group_rows(self, group_name: str):
        """Efficiently collapse a group by removing its feature rows"""
        try:
            print(f"🔍 DEBUG: _collapse_group_rows called for group: {group_name}")
            
            if not hasattr(self, 'group_rows') or group_name not in self.group_rows:
                print(f"⚠️ WARNING: Group {group_name} not found in group_rows")
                return
            
            group_info = self.group_rows[group_name]
            feature_rows = group_info['feature_rows']
            
            if not feature_rows:
                print(f"🔍 DEBUG: Group {group_name} has no feature rows to collapse")
                return
            
            print(f"🔍 DEBUG: Removing {len(feature_rows)} feature rows for group {group_name}")
            
            # Remove feature rows in reverse order to avoid index shifting issues
            for row_idx in reversed(feature_rows):
                try:
                    self.sheet.delete_row(row_idx)
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to delete row {row_idx}: {e}")
            
            # Clear the feature rows list
            group_info['feature_rows'].clear()
            
            # Update the expand icon to show collapsed state
            expand_icon = "▶"
            self.sheet.set_cell_data(group_info['header_row'], 0, f"{expand_icon} {group_name}")
            
            # Ensure Actions group is positioned correctly after collapsing
            self._ensure_actions_group_position()
            
            print(f"🔍 DEBUG: Successfully collapsed group {group_name}")
            
        except Exception as e:
            print(f"❌ ERROR in _collapse_group_rows: {e}")
            import traceback
            traceback.print_exc()
    
    def _expand_all_groups(self):
        """Expand all feature groups"""
        if self.feature_groups:
            unique_groups = set(self.feature_groups)
            for group_name in unique_groups:
                self.expanded_groups.add(group_name)
            self._refresh_table()
            self._update_summary()
    
    def _collapse_all_groups(self):
        """Collapse all feature groups"""
        self.expanded_groups.clear()
        self._refresh_table()
        self._update_summary()
    
    def _update_summary(self):
        """Update the summary label"""
        if not self._schema_set:
            summary = "Features: 0/128 | Buffer: 0/10 | Status: Ready"
        else:
            features_count = 128
            buffer_count = 10
            status = "Active"
            favorites_count = len(self.favorite_features)
            expanded_groups_count = len(self.expanded_groups)
            summary = f"Features: {features_count}/128 | Buffer: {buffer_count}/10 | Status: {status} | Favorites: {favorites_count} | Groups: {expanded_groups_count} expanded"
        
        self.summary_label.config(text=summary)
    
    def _copy_to_clipboard(self):
        """Copy table data to clipboard"""
        if not self._schema_set:
            return
        
        try:
            # Get visible features
            visible_features = []
            for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
                if self.feature_group_filter != "All" and group != self.feature_group_filter:
                    continue
                if self.search_text and self.search_text not in name.lower():
                    continue
                visible_features.append((i, name, group))
            
            # Build CSV-like string
            lines = []
            header = ["Feature", "Index", "Group"] + [f"T{t}" for t in range(10)]  # T0 to T9
            lines.append("\t".join(header))
            
            for feature_idx, name, group in visible_features:
                row = [name, str(feature_idx), group]
                # Add blank values for time columns (data not available in this context)
                row.extend([""] * 10)
                lines.append("\t".join(row))
            
            clipboard_text = "\n".join(lines)
            self.clipboard_clear()
            self.clipboard_append(clipboard_text)
            
        except Exception as e:
            print(f"Failed to copy to clipboard: {e}")
    
    def _export_to_csv(self):
        """Export table data to CSV file"""
        if not self._schema_set:
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Get visible features
            visible_features = []
            for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
                if self.feature_group_filter != "All" and group != self.feature_group_filter:
                    continue
                if self.search_text and self.search_text not in name.lower():
                    continue
                visible_features.append((i, name, group))
            
            # Write CSV
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                header = ["Feature", "Index", "Group"] + [f"T{t}" for t in range(10)]  # T0 to T9
                writer.writerow(header)
                
                # Data rows
                for feature_idx, name, group in visible_features:
                    row = [name, feature_idx, group]
                    # Add blank values for time columns (data not available in this context)
                    row.extend([""] * 10)
                    writer.writerow(row)
            
        except Exception as e:
            print(f"Failed to export CSV: {e}")
    
    def clear(self):
        """Clear all data from the view"""
        self.feature_names = None
        self.feature_groups = None
        self._color_bits = None
        self._schema_set = False
        self.sheet.clear()
        self._update_summary()

    def _start_live_mode(self):
        """Start live mode"""
        try:
            print("🔍 DEBUG: _start_live_mode called")
            LOG.info("LiveFeaturesView: Starting live mode...")
            
            # Check controller availability
            if not hasattr(self, 'controller') or not self.controller:
                print("❌ ERROR: No controller available")
                raise RuntimeError("Controller not available")
            
            print(f"🔍 DEBUG: Controller type: {type(self.controller)}")
            print(f"🔍 DEBUG: Controller has start_live_mode_for_recorder: {hasattr(self.controller, 'start_live_mode_for_recorder')}")
            
            if hasattr(self.controller, 'start_live_mode_for_recorder'):
                print("🔍 DEBUG: Calling start_live_mode_for_recorder...")
                self.controller.start_live_mode_for_recorder()
                print("🔍 DEBUG: start_live_mode_for_recorder completed")
            else:
                print("🔍 DEBUG: Calling start_live_mode...")
                self.controller.start_live_mode()
                print("🔍 DEBUG: start_live_mode completed")
            
            LOG.info("LiveFeaturesView: Live mode started successfully")
            print("🔍 DEBUG: About to update summary label...")
            self.summary_label.config(text="Features: 0/128 | Buffer: 0/10 | Status: Live Mode Active")
            print("🔍 DEBUG: Summary label updated successfully")
            
        except Exception as e:
            print(f"❌ ERROR in _start_live_mode: {e}")
            import traceback
            traceback.print_exc()
            LOG.error(f"Failed to start live mode: {e}")
            try:
                self.summary_label.config(text="Features: 0/128 | Buffer: 0/10 | Status: Failed to Start")
            except Exception as ui_error:
                print(f"❌ Failed to update summary label after error: {ui_error}")
                traceback.print_exc()

    def _stop_live_mode(self):
        """Stop live mode"""
        try:
            if hasattr(self.controller, 'stop_live_mode_for_recorder'):
                self.controller.stop_live_mode_for_recorder()
            else:
                self.controller.stop_live_mode()
        except Exception as e:
            LOG.error(f"Failed to stop live mode: {e}")

    def _clear_buffers(self):
        """Clear all buffers"""
        try:
            if hasattr(self.controller, 'clear_buffers'):
                self.controller.clear_buffers()
            # Clear the view as well
            self.clear()
        except Exception as e:
            LOG.error(f"Failed to clear buffers: {e}")
    
    # === Recorder Methods ===
    
    def _update_countdown(self):
        """Update countdown value from spinbox"""
        try:
            self.countdown_seconds = int(self.countdown_var.get())
        except ValueError:
            self.countdown_seconds = 5
            self.countdown_var.set("5")
    
    def _update_autosave(self):
        """Update auto-save interval from spinbox"""
        try:
            minutes = int(self.autosave_var.get())
            self.auto_save_interval = minutes * 60  # Convert to seconds
        except ValueError:
            self.auto_save_interval = 300  # Default 5 minutes
            self.autosave_var.set("5")
    
    def _count_gamestate_files(self, directory):
        """Count gamestate files in the directory"""
        try:
            import os
            if not os.path.exists(directory):
                return 0
            files = [f for f in os.listdir(directory) if f.endswith('.json')]
            return len(files)
        except Exception:
            return 0
    
    def _start_recording_session(self):
        """Start a complete recording session with feature extraction"""
        if self.recording_session_active:
            return
        
        try:
            print("🎬 Starting recording session...")
            print("🔍 DEBUG [live_features_view.py:1325] About to call _start_live_mode...")
            
            # Update step indicator
            self.process_step_label.config(text="Step 2/5: Starting live feature extraction...", foreground="orange")
            self.recorder_status_label.config(text="🔄 Initializing recording session...", foreground="orange")
            
            self.recording_session_active = True
            self.session_start_time = time.time()
            
            # Create session-specific directory
            import os
            import datetime
            from pathlib import Path
            
            session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = str(Path("data") / "recording_sessions" / session_timestamp)
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"🔍 DEBUG [live_features_view.py:1340] GAMESTATE DIRECTORY: {self.session_dir}")
            
            # Update controller to look in this session's gamestates directory
            if hasattr(self, 'controller') and self.controller:
                try:
                    self.controller.update_gamestates_directory(session_timestamp)
                    print(f"✅ Updated controller to use gamestates from: {self.session_dir}/gamestates/")
                    print(f"🔍 DEBUG [live_features_view.py:1345] Session directory path: {self.session_dir}")
                    print(f"🔍 DEBUG [live_features_view.py:1350] Gamestates subdirectory path: {self.session_dir}/gamestates/")
                    print(f"🔍 DEBUG [live_features_view.py:1355] Absolute gamestates path: {os.path.abspath(os.path.join(self.session_dir, 'gamestates'))}")
                except Exception as e:
                    print(f"⚠️ Failed to update controller gamestates directory: {e}")
            
            # Update file paths info
            data_dir = os.path.abspath("data")
            session_dir_abs = os.path.abspath(self.session_dir)
            
            # Enable copy button now that we have a session directory
            if hasattr(self, 'copy_path_button'):
                self.copy_path_button.config(state="normal")
            
            # Start feature extraction (live mode)
            self._start_live_mode()
            print("🔍 DEBUG [live_features_view.py:1370] _start_live_mode completed")
            
            # Start gamestate recording
            self._start_gamestate_recording()
            
            # NOTE: Action recording will start when user clicks "Start Recording" button
            # and after countdown completes - NOT automatically with session
            # NOTE: Feature recording will start when user clicks "Start Recording" button
            # and after countdown completes - NOT automatically with session
            
            # Update live mode status with more detailed info
            gamestate_count = self._count_gamestate_files(self.session_dir)
            if gamestate_count > 0:
                self.live_mode_label.config(text=f"🔄 Live Mode: Active - {gamestate_count} gamestates found", foreground="green")
            else:
                self.live_mode_label.config(text="🔄 Live Mode: Active - Waiting for RuneLite plugin", foreground="orange")
            
            # Enable recording controls
            self.start_recording_button.config(state="normal")
            self.clear_session_button.config(state="normal")
            self.export_session_button.config(state="normal")  # Enable export when session is active
            
            # Update UI
            self.start_session_button.config(text="⏸️ Pause Session", command=self._pause_recording_session)
            self.process_step_label.config(text="Step 2/5: ✅ Session active! Next: Detect Runelite window", foreground="green")
            self.recorder_status_label.config(text="✅ Session active - feature extraction running", foreground="green")
            
            print("✅ Recording session started - feature extraction active")
            print(f"📁 Session directory: {session_dir_abs}")
            
        except Exception as e:
            print(f"❌ ERROR [live_features_view.py:1390] starting recording session: {e}")
            import traceback
            traceback.print_exc()
            self.process_step_label.config(text="Step 1/5: ❌ Error starting session", foreground="red")
            self.recorder_status_label.config(text=f"❌ Error starting session: {e}", foreground="red")
            self.recording_session_active = False
    
    def _pause_recording_session(self):
        """Pause the recording session (stop feature extraction)"""
        if not self.recording_session_active:
            return
        
        try:
            print("⏸️ Pausing recording session...")
            
            # Stop feature extraction (live mode)
            self._stop_live_mode()
            
            # Pause feature recording (keep files open)
            if hasattr(self, 'feature_recording') and self.feature_recording:
                self.feature_recording = False
                print("⏸️ Feature recording paused (files remain open)")
            
            # Pause gamestate recording (keep monitoring)
            if hasattr(self, 'gamestate_recording') and self.gamestate_recording:
                self.gamestate_recording = False
                print("⏸️ Gamestate recording paused (monitoring continues)")
            
            # Pause action recording (keep files open)
            if hasattr(self, 'action_recording') and self.action_recording:
                self.action_recording = False
                print("⏸️ Action recording paused (files remain open)")
            
            # Pause pynput listeners (stop capturing new events)
            if hasattr(self, 'recording') and self.recording:
                self.recording = False
                print("⏸️ Input listeners paused")
            
            # Update UI
            self.start_session_button.config(text="▶ Resume Session", command=self._resume_recording_session)
            self.recorder_status_label.config(text="Session paused - recording stopped, files remain open", foreground="orange")
            
            print("⏸️ Recording session paused - all files remain open for resume")
            
        except Exception as e:
            print(f"❌ Error pausing recording session: {e}")
            import traceback
            traceback.print_exc()
    
    def _resume_recording_session(self):
        """Resume the recording session (restart feature extraction)"""
        if not self.recording_session_active:
            return
        
        try:
            print("▶ Resuming recording session...")
            
            # Use after_idle to prevent blocking the main thread
            self.after_idle(self._resume_recording_async)
            
        except Exception as e:
            print(f"❌ Error resuming recording session: {e}")
    
    def _resume_recording_async(self):
        """Actually resume the recording session (non-blocking)"""
        try:
            print("▶ Actually resuming recording session...")
            
            # Restart feature extraction (live mode)
            self._start_live_mode()
            
            # Resume gamestate recording (reactivate monitoring)
            if hasattr(self, 'gamestate_session_dir') and self.gamestate_session_dir:
                self.gamestate_recording = True
                print("✅ Gamestate recording resumed (monitoring reactivated)")
            
            # Resume action recording (reactivate without recreating files)
            if hasattr(self, 'csv_writer') and self.csv_writer:
                self.action_recording = True
                print("✅ Action recording resumed (existing files continue)")
            else:
                print("⚠️ No action recording files found, cannot resume")
            
            # Resume feature recording (reactivate without recreating files)
            if hasattr(self, 'features_csv_writer') and self.features_csv_writer:
                self.feature_recording = True
                print("✅ Feature recording resumed (existing files continue)")
            else:
                print("⚠️ No feature recording files found, cannot resume")
            
            # Restart pynput listeners
            try:
                from pynput import mouse, keyboard
                
                # Start mouse listener
                self.mouse_listener = mouse.Listener(
                    on_move=self._on_mouse_move,
                    on_click=self._on_mouse_click,
                    on_scroll=self._on_mouse_scroll,
                    suppress=False
                )
                self.mouse_listener.start()
                
                # Start keyboard listener
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                    suppress=False
                )
                self.keyboard_listener.start()
                
                # Set recording state
                self.recording = True
                print("✅ Input listeners resumed")
                
            except Exception as listener_error:
                print(f"❌ ERROR: Failed to resume input listeners: {listener_error}")
                import traceback
                traceback.print_exc()
            
            # Update UI
            self.start_session_button.config(text="⏸️ Pause Session", command=self._pause_recording_session)
            self.recorder_status_label.config(text="Recording session resumed - all systems active", foreground="green")
            
            print("▶ Recording session resumed successfully - continuing from where left off")
            
        except Exception as e:
            print(f"❌ Error resuming recording session: {e}")
            import traceback
            traceback.print_exc()
            self.recorder_status_label.config(text=f"❌ Error resuming session: {e}", foreground="red")
    
    def _clear_recording_session(self):
        """Clear the current recording session"""
        try:
            print("🗑️ Clearing recording session...")
            
            # Stop recording if active
            if self.recording:
                self._stop_recording()
            
            # Stop feature extraction
            self._stop_live_mode()
            
            # Stop feature recording
            self._stop_feature_recording()
            
            # Stop gamestate recording
            self._stop_gamestate_recording()
            
            # Stop action recording
            self._stop_action_recording()
            
            # Stop gamestate cleanup system
            
            # Stop gamestate cleanup system
            self.gamestate_cleanup_active = False
            print(f"🛑 Stopped gamestate cleanup system - session cleared")
            
            # Reset session state
            self.recording_session_active = False
            self.session_start_time = None
            self.session_data = []
            
            # Reset counters
            self.click_count = 0
            self.key_press_count = 0
            self.scroll_count = 0
            self.mouse_move_count = 0
            
            # Reset UI
            self.start_session_button.config(text="🎬 Start Recording Session", command=self._start_recording_session)
            self.start_recording_button.config(state="disabled")
            self.stop_recording_button.config(state="disabled")
            self.clear_session_button.config(state="normal")
            self.export_session_button.config(state="disabled")
            
            # Disable copy button when session is cleared
            if hasattr(self, 'copy_path_button'):
                self.copy_path_button.config(state="disabled")
            # Reset all status indicators
            self.process_step_label.config(text="Step 1/5: Click '🎬 Start Recording Session'", foreground="orange")
            self.recorder_status_label.config(text="📋 Session cleared - ready to start new session", foreground="cyan")
            self.file_info_label.config(text="📁 Files: Not initialized", foreground="cyan")
            self.live_mode_label.config(text="🔄 Live Mode: Inactive", foreground="yellow")
            
            # Clear feature table
            self.clear()
            
            print("🗑️ Recording session cleared")
            
        except Exception as e:
            print(f"❌ Error clearing recording session: {e}")
    
    def _export_recording_session(self):
        """Export the current recording session data"""
        if not hasattr(self, 'session_dir') or not self.session_dir:
            from tkinter import messagebox
            messagebox.showinfo("Info", "No active recording session to export")
            return
        
        try:
            from tkinter import messagebox, filedialog
            import os
            import shutil
            
            # Show info about what will be exported
            session_name = os.path.basename(self.session_dir)
            messagebox.showinfo("Export Session", 
                f"Session '{session_name}' will be exported.\n\n"
                f"Files in session:\n"
                f"• actions.csv - Recorded mouse/keyboard actions\n"
                f"• features.csv - Live extracted feature vectors\n"
                f"• gamestates/ - Raw gamestate JSON files\n\n"
                f"Click OK to select export location.")
            
            # Get export directory
            export_dir = filedialog.askdirectory(
                parent=self,
                title="Select Export Directory for Session"
            )
            
            if not export_dir:
                return
            
            # Create export subdirectory
            export_path = os.path.join(export_dir, f"exported_session_{session_name}")
            os.makedirs(export_path, exist_ok=True)
            
            # Copy all session files
            if os.path.exists(self.session_dir):
                for item in os.listdir(self.session_dir):
                    src = os.path.join(self.session_dir, item)
                    dst = os.path.join(export_path, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)
            
            # Create session summary
            summary_data = {
                "session_info": {
                    "session_name": session_name,
                    "start_time": self.session_start_time,
                    "duration": time.time() - self.session_start_time if self.session_start_time else 0,
                    "total_actions": self.click_count + self.key_press_count + self.scroll_count,
                    "feature_vectors": self.feature_count,
                    "gamestates": self.gamestate_count,
                    "counts": {
                        "mouse_moves": self.mouse_move_count,
                        "clicks": self.click_count,
                        "scrolls": self.scroll_count,
                        "key_presses": self.key_press_count
                    }
                },
                "files": {
                    "session_directory": self.session_dir,
                    "export_directory": export_path
                }
            }
            
            # Save summary
            summary_file = os.path.join(export_path, "session_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", 
                f"Session '{session_name}' exported successfully!\n\n"
                f"Location: {export_path}\n"
                f"Files: {len(os.listdir(export_path))} items\n"
                f"Actions: {summary_data['session_info']['total_actions']}\n"
                f"Features: {summary_data['session_info']['feature_vectors']}\n"
                f"Gamestates: {summary_data['session_info']['gamestates']}")
            
            print(f"💾 Session exported to {export_path}")
            
        except Exception as e:
            print(f"❌ Error exporting session: {e}")
            from tkinter import messagebox
            messagebox.showerror("Export Error", f"Failed to export session: {e}")
    
    def _detect_runelite_window(self):
        """Detect Runelite window"""
        try:
            # Update status to show we're working
            self.window_status_label.config(text="🔍 Searching...", foreground="orange")
            self.recorder_status_label.config(text="Searching for Runelite windows...", foreground="orange")
            
            # Use after_idle to prevent blocking the main thread
            self.after_idle(self._detect_runelite_window_async)
            
        except Exception as e:
            import traceback
            print(f"❌ Error starting window detection: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
            try:
                self.window_status_label.config(text="❌ Error", foreground="red")
                self.recorder_status_label.config(text=f"Error: {e}", foreground="red")
            except Exception as ui_error:
                print(f"❌ Failed to update UI labels: {ui_error}")
                traceback.print_exc()
    
    def _detect_runelite_window_async(self):
        """Detect Runelite window asynchronously"""
        try:
            import pygetwindow as gw
            
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
                try:
                    self.window_status_label.config(text="✅ Window detected", foreground="green")
                    
                    # Update step progress
                    if self.recording_session_active:
                        self.process_step_label.config(text="Step 3/5: ✅ Window detected! Next: Start recording", foreground="green")
                        self.recorder_status_label.config(text="✅ Ready to record! Click '▶ Start Recording'", foreground="green")
                    else:
                        self.process_step_label.config(text="Step 2/5: ✅ Window found! Start session first", foreground="lime")
                        self.recorder_status_label.config(text="✅ Window found! Start recording session first", foreground="lime")
                    
                    # Update file info with window details
                    window_info = f"🎮 Window: {self.runelite_window.title} ({self.runelite_window.width}x{self.runelite_window.height})"
                    # current_file_text = self.file_info_label.cget("text")
                    # if "Data:" in current_file_text:
                    #     self.file_info_label.config(text=f"{current_file_text} | {window_info}", foreground="blue")
                    # else:
                    #     self.file_info_label.config(text=window_info, foreground="blue")
                    
                    print(f"🎮 Selected Runelite window: {self.runelite_window.title}")
                    print(f"📐 Window size: {self.runelite_window.width}x{self.runelite_window.height}")
                    print(f"📍 Window position: ({self.runelite_window.left}, {self.runelite_window.top})")
                except Exception as ui_error:
                    print(f"❌ Error updating UI after window detection: {ui_error}")
                    import traceback
                    traceback.print_exc()
            else:
                try:
                    self.window_status_label.config(text="❌ Window not found", foreground="red")
                    self.process_step_label.config(text="Step 2/5: ❌ No Runelite window found", foreground="red")
                    self.recorder_status_label.config(text="❌ No Runelite window found. Make sure it's running.", foreground="red")
                    print("⚠️ No Runelite window found. Make sure Runelite is running.")
                except Exception as ui_error:
                    print(f"❌ Error updating UI for no window found: {ui_error}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            import traceback
            print(f"❌ Error finding Runelite window: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
            try:
                self.window_status_label.config(text="❌ Error", foreground="red")
                self.recorder_status_label.config(text=f"Error: {e}", foreground="red")
            except Exception as ui_error:
                print(f"❌ Failed to update UI labels after error: {ui_error}")
                traceback.print_exc()
    
    def _start_recording(self):
        """Start recording mouse and keyboard events"""
        print("🔍 DEBUG: _start_recording called")
        if self.recording:
            return
        
        if not self.runelite_window:
            from tkinter import messagebox
            messagebox.showerror("Error", "Please detect Runelite window first!")
            return
        
        if not self.recording_session_active:
            from tkinter import messagebox
            messagebox.showerror("Error", "Please start a recording session first!")
            return
        
        try:
            print("🔍 DEBUG: About to start countdown...")
            # Start countdown before recording
            self._start_countdown()
            print("🔍 DEBUG: Countdown started")
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            self.recorder_status_label.config(text=f"Error starting recording: {e}", foreground="red")
    
    def _start_countdown(self):
        """Start countdown before recording begins"""
        if self.countdown_active:
            return
        
        try:
            self.countdown_active = True
            self.countdown_seconds = int(self.countdown_var.get())
            
            # Disable controls during countdown
            self.start_recording_button.config(state="disabled")
            self.start_session_button.config(state="disabled")
            
            # Update progress and show countdown message
            self.process_step_label.config(text="Step 4/5: ⏰ Countdown starting...", foreground="orange")
            self.recorder_status_label.config(text="🚀 Get ready! Countdown starting...", foreground="orange")
            
            # Update file info with recording details
            import os
            # Use session directory paths if available
            if hasattr(self, 'session_dir') and self.session_dir:
                actions_file = os.path.abspath(os.path.join(self.session_dir, "actions.csv"))
                features_file = os.path.abspath(os.path.join(self.session_dir, "features.csv"))
            else:
                actions_file = os.path.abspath("data/actions.csv")
                features_file = os.path.abspath("data/features.csv")
            
            current_text = self.file_info_label.cget("text")
            # if "actions.csv" not in current_text:
            #     self.file_info_label.config(text=f"{current_text} | 📝 Actions: {actions_file} | 🧠 Features: {features_file}", foreground="blue")
            
            # Start countdown using after_idle to prevent blocking
            self.after_idle(self._run_countdown)
            
        except Exception as e:
            print(f"❌ Error starting countdown: {e}")
            self.countdown_active = False
            self.start_recording_button.config(state="normal")
            self.start_session_button.config(state="normal")
    
    def _run_countdown(self):
        """Run the countdown timer"""
        if not self.countdown_active:
            return
        
        if self.countdown_seconds > 0:
            # Show countdown
            self.countdown_label.config(text=f"⏰ {self.countdown_seconds}", foreground="red")
            self.process_step_label.config(text=f"Step 4/5: ⏰ Recording starts in {self.countdown_seconds}...", foreground="orange")
            self.recorder_status_label.config(text=f"⏰ Recording starts in {self.countdown_seconds} seconds - Get ready!", foreground="orange")
            
            # Decrement and continue
            self.countdown_seconds -= 1
            self.after(1000, self._run_countdown)
        else:
            # Countdown finished - start recording
            self.countdown_label.config(text="🎬 GO!", foreground="green")
            self.process_step_label.config(text="Step 5/5: 🎬 RECORDING ACTIVE!", foreground="red")
            self.recorder_status_label.config(text="🎬 RECORDING! Perform actions in Runelite", foreground="red")
            
            # Clear countdown after 2 seconds
            self.after(2000, self._clear_countdown)
            
            # Actually start the recording
            self._start_recording_after_countdown()
    
    def _clear_countdown(self):
        """Clear the countdown display"""
        self.countdown_label.config(text="")
        self.countdown_active = False
        
        # Re-enable controls
        self.start_recording_button.config(state="normal")
        self.start_session_button.config(state="normal")
    
    def _start_recording_after_countdown(self):
        """Start the actual recording after countdown finishes"""
        try:
            # Use after_idle to prevent blocking the main thread
            self.after_idle(self._start_recording_async)
            
        except Exception as e:
            print(f"❌ Error starting recording after countdown: {e}")
            self.recorder_status_label.config(text=f"Error starting recording: {e}", foreground="red")
    
    def _start_recording_async(self):
        """Start the actual recording asynchronously"""
        try:
            # Import recording libraries
            from pynput import mouse, keyboard
            import csv
            import os
            import time
            import threading
            
            print(f"🔍 DEBUG: Starting recording async - session_dir: {getattr(self, 'session_dir', 'None')}")
            
            # Use session directory if available, otherwise fallback to data directory
            if hasattr(self, 'session_dir') and self.session_dir:
                csv_path = os.path.join(self.session_dir, "actions.csv")
                print(f"🔍 DEBUG: Using session directory for actions.csv: {csv_path}")
            else:
                print(f"⚠️ WARNING: No session_dir, falling back to data directory")
                os.makedirs("data", exist_ok=True)
                csv_path = "data/actions.csv"
            
            # Verify directory exists and is writable
            csv_dir = os.path.dirname(csv_path)
            if not os.path.exists(csv_dir):
                print(f"❌ ERROR: Directory does not exist: {csv_dir}")
                raise RuntimeError(f"Directory does not exist: {csv_dir}")
            
            if not os.access(csv_dir, os.W_OK):
                print(f"❌ ERROR: Directory not writable: {csv_dir}")
                raise RuntimeError(f"Directory not writable: {csv_dir}")
            
            print(f"🔍 DEBUG: Creating actions.csv at: {csv_path}")
            
            # Initialize CSV file
            try:
                self.csvf = open(csv_path, 'w', newline='')
                print(f"✅ CSV file opened successfully: {csv_path}")
            except Exception as csv_error:
                print(f"❌ ERROR: Failed to open CSV file: {csv_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to open CSV file {csv_path}: {csv_error}")
            
            try:
                self.csv_writer = csv.writer(self.csvf)
                self.csv_writer.writerow([
                    'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
                    'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
                ])
                print(f"✅ CSV header written successfully")
            except Exception as header_error:
                print(f"❌ ERROR: Failed to write CSV header: {header_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to write CSV header: {header_error}")
            
            # CRITICAL: Initialize CSV buffer variables (this was missing!)
            try:
                self.csv_buffer = []
                self.last_csv_flush = time.time()
                print(f"✅ CSV buffer initialized - size: {self.csv_buffer_size}, flush_interval: {self.csv_flush_interval}")
            except Exception as buffer_error:
                print(f"❌ ERROR: Failed to initialize CSV buffer: {buffer_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to initialize CSV buffer: {buffer_error}")
            
            # Reset counters
            self.click_count = 0
            self.key_press_count = 0
            self.scroll_count = 0
            self.mouse_move_count = 0
            print(f"✅ Action counters reset")
            
            # Start recording
            self.recording = True
            self.start_recording_button.config(state="disabled")
            self.stop_recording_button.config(state="normal")
            print(f"✅ Recording state set to: {self.recording}")
            
            # CRITICAL: Start action recording now that countdown is complete
            print("🎬 Starting action recording after countdown...")
            try:
                self._start_action_recording()
                print("✅ Action recording started successfully")
            except Exception as action_error:
                print(f"❌ ERROR: Failed to start action recording: {action_error}")
                import traceback
                traceback.print_exc()
                # Don't fail the entire recording if action recording fails
            
            # CRITICAL: Start feature recording now that countdown is complete
            print("🚀 Starting feature recording after countdown...")
            try:
                self._start_feature_recording()
                print("✅ Feature recording started successfully")
            except Exception as feature_error:
                print(f"❌ ERROR: Failed to start feature recording: {feature_error}")
                import traceback
                traceback.print_exc()
                # Don't fail the entire recording if feature recording fails
            
            # Verify runelite window is set
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"❌ ERROR: No runelite window detected!")
                raise RuntimeError("No runelite window detected - cannot record actions")
            
            print(f"🔍 DEBUG: Runelite window: {self.runelite_window.title} at ({self.runelite_window.left}, {self.runelite_window.top})")
            
            # Start listeners in separate threads
            try:
                self.mouse_listener = mouse.Listener(
                    on_move=self._on_mouse_move,
                    on_click=self._on_mouse_click,
                    on_scroll=self._on_mouse_scroll,
                    suppress=False
                )
                self.mouse_listener.start()
                print(f"✅ Mouse listener started")
            except Exception as mouse_error:
                print(f"❌ ERROR: Failed to start mouse listener: {mouse_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to start mouse listener: {mouse_error}")
            
            try:
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                    suppress=False
                )
                self.keyboard_listener.start()
                print(f"✅ Keyboard listener started")
            except Exception as keyboard_error:
                print(f"❌ ERROR: Failed to start keyboard listener: {keyboard_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to start keyboard listener: {keyboard_error}")
            
            # Verify listeners are running
            if not self.mouse_listener.is_alive():
                raise RuntimeError("Mouse listener failed to start")
            if not self.keyboard_listener.is_alive():
                raise RuntimeError("Keyboard listener failed to start")
            
            print("✅ Recording started successfully after countdown")
            print(f"🔍 DEBUG: All systems ready - recording: {self.recording}, csv_writer: {self.csv_writer}, csv_buffer: {len(self.csv_buffer)}")
            
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR starting recording after countdown: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
            
            # Try to clean up on failure
            try:
                if hasattr(self, 'csvf') and self.csvf:
                    self.csvf.close()
                    self.csvf = None
                if hasattr(self, 'csv_writer'):
                    self.csv_writer = None
                if hasattr(self, 'csv_buffer'):
                    self.csv_buffer = []
                self.recording = False
            except Exception as cleanup_error:
                print(f"❌ ERROR during cleanup: {cleanup_error}")
            
            # Update UI to show error
            try:
                self.recorder_status_label.config(text=f"❌ CRITICAL ERROR: {e}", foreground="red")
            except Exception as ui_error:
                print(f"❌ Failed to update UI after error: {ui_error}")
            
            # Re-raise to prevent silent failure
            raise
    
    def _stop_recording(self):
        """Stop recording and save actions"""
        if not self.recording:
            return
        
        try:
            self.recording = False
            self.start_recording_button.config(state="normal")
            self.stop_recording_button.config(state="disabled")
            
            # Update progress and status
            import os
            # Get the actual actions file path that was used
            actions_file = getattr(self, 'csvf', None)
            if actions_file and hasattr(actions_file, 'name'):
                actions_file = actions_file.name
            else:
                actions_file = "data/actions.csv"  # fallback
            
            self.process_step_label.config(text="Step 5/5: ✅ Recording completed!", foreground="green")
            self.recorder_status_label.config(text=f"✅ Recording stopped - {self.click_count + self.key_press_count + self.scroll_count} actions saved", foreground="springgreen")
            
            # Update file info with completion details
            file_size = "Unknown"
            try:
                if os.path.exists(actions_file):
                    size_bytes = os.path.getsize(actions_file)
                    file_size = f"{size_bytes} bytes"
            except:
                pass
            
            # Show session directory structure
            if hasattr(self, 'session_dir') and self.session_dir:
                session_info = f"📁 Session: {os.path.basename(self.session_dir)}"
                # self.file_info_label.config(text=f"{session_info} | 💾 Actions: {os.path.basename(actions_file)} ({file_size}) | 🧠 Features: {self.feature_count} vectors", foreground="green")
            else:
                current_text = self.file_info_label.cget("text")
                if "actions.csv" in current_text:
                    # Replace the actions file info with completion status
                    parts = current_text.split(" | ")
                    new_parts = []
                    for part in parts:
                        if "actions.csv" in part:
                            new_parts.append(f"💾 Saved: {actions_file} ({file_size})")
                        else:
                            new_parts.append(part)
                    # self.file_info_label.config(text=" | ".join(new_parts), foreground="green")
            
            # Stop listeners
            if hasattr(self, 'mouse_listener'):
                self.mouse_listener.stop()
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
            
            # Stop feature recording
            if hasattr(self, 'feature_recording') and self.feature_recording:
                self._stop_feature_recording()
                print("🛑 Feature recording stopped")
            
            # Finalize CSV files with buffered data
            self._finalize_csv()
            self._finalize_features_csv()
            
            print(f"💾 Recording stopped. Saved actions to {actions_file}")
            print(f"📊 Final counts - Clicks: {self.click_count}, Key presses: {self.key_press_count}, Scrolls: {self.scroll_count}, Mouse moves: {self.mouse_move_count}")
            print(f"🧠 Feature vectors saved: {self.feature_count}")
            
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
    
    def _on_mouse_move(self, x, y):
        """Handle mouse movement events"""
        try:
            # Check recording state
            if not hasattr(self, 'action_recording') or not self.action_recording:
                print(f"🔍 DEBUG: Mouse move ignored - action_recording: {getattr(self, 'action_recording', 'undefined')}")
                return
            
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"🔍 DEBUG: Mouse move ignored - no runelite window")
                return
            
            # Get relative coordinates
            try:
                rel_x = x - self.runelite_window.left
                rel_y = y - self.runelite_window.top
            except Exception as coord_error:
                print(f"❌ ERROR: Failed to calculate relative coordinates: {coord_error}")
                import traceback
                traceback.print_exc()
                return
            
            # Check if mouse is over Runelite window
            if (0 <= rel_x <= self.runelite_window.width and 
                0 <= rel_y <= self.runelite_window.height):
                
                # Rate limiting: Only capture every 10ms minimum
                current_time = time.time()
                if current_time - self.last_mouse_capture >= self.mouse_move_throttle:
                    self.mouse_move_count += 1
                    self.last_mouse_capture = current_time
                    
                    # Write to CSV with buffering
                    try:
                        if hasattr(self, 'csv_writer') and self.csv_writer:
                            action_data = [
                                int(time.time() * 1000), 'move', rel_x, rel_y, 
                                '', '', 0, 0, '', ''
                            ]
                            self._write_csv_buffered(action_data)
                        else:
                            print(f"❌ ERROR: No CSV writer available for mouse move")
                            import traceback
                            traceback.print_exc()
                    except Exception as csv_error:
                        print(f"❌ ERROR: Failed to write mouse move to CSV: {csv_error}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Debug: Show when movements are being throttled (but not too frequently)
                    if self.mouse_move_count % 100 == 0:  # Only log every 100th throttled event
                        print(f"🔍 Mouse movement throttled: {((current_time - self.last_mouse_capture) * 1000):.1f}ms since last capture")
                    
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in mouse move handler: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        try:
            # Check recording state
            if not hasattr(self, 'action_recording') or not self.action_recording:
                print(f"🔍 DEBUG: Mouse click ignored - action_recording: {getattr(self, 'action_recording', 'undefined')}")
                return
            
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"🔍 DEBUG: Mouse click ignored - no runelite window")
                return
            
            if not pressed:
                print(f"🔍 DEBUG: Mouse click ignored - not pressed")
                return
            
            # Get relative coordinates
            try:
                rel_x = x - self.runelite_window.left
                rel_y = y - self.runelite_window.top
            except Exception as coord_error:
                print(f"❌ ERROR: Failed to calculate relative coordinates: {coord_error}")
                import traceback
                traceback.print_exc()
                return
            
            if (0 <= rel_x <= self.runelite_window.width and 
                0 <= rel_y <= self.runelite_window.height):
                
                self.click_count += 1
                
                try:
                    btn_name = str(button).split('.')[-1]
                except Exception as button_error:
                    print(f"❌ ERROR: Failed to get button name: {button_error}")
                    btn_name = str(button)
                
                # Write to CSV with buffering
                try:
                    if hasattr(self, 'csv_writer') and self.csv_writer:
                        action_data = [
                            int(time.time() * 1000), 'click', rel_x, rel_y, 
                            btn_name, '', 0, 0, '', ''
                        ]
                        self._write_csv_buffered(action_data)
                        
                        # Store in session data for auto-save
                        try:
                            self._update_session_data({
                                'timestamp': action_data[0],
                                'event_type': 'click',
                                'x': rel_x,
                                'y': rel_y,
                                'button': btn_name
                            })
                        except Exception as session_error:
                            print(f"❌ ERROR: Failed to update session data: {session_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"❌ ERROR: No CSV writer available for mouse click")
                        import traceback
                        traceback.print_exc()
                except Exception as csv_error:
                    print(f"❌ ERROR: Failed to write mouse click to CSV: {csv_error}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in mouse click handler: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        try:
            # Check recording state
            if not hasattr(self, 'action_recording') or not self.action_recording:
                print(f"🔍 DEBUG: Mouse scroll ignored - action_recording: {getattr(self, 'action_recording', 'undefined')}")
                return
            
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"🔍 DEBUG: Mouse scroll ignored - no runelite window")
                return
            
            # Get relative coordinates
            try:
                rel_x = x - self.runelite_window.left
                rel_y = y - self.runelite_window.top
            except Exception as coord_error:
                print(f"❌ ERROR: Failed to calculate relative coordinates: {coord_error}")
                import traceback
                traceback.print_exc()
                return
            
            if (0 <= rel_x <= self.runelite_window.width and 
                0 <= rel_y <= self.runelite_window.height):
                
                self.scroll_count += 1
                
                # Write to CSV with buffering
                try:
                    if hasattr(self, 'csv_writer') and self.csv_writer:
                        action_data = [
                            int(time.time() * 1000), 'scroll', rel_x, rel_y, 
                            '', '', dx, dy, '', ''
                        ]
                        self._write_csv_buffered(action_data)
                    else:
                        print(f"❌ ERROR: No CSV writer available for mouse scroll")
                        import traceback
                        traceback.print_exc()
                except Exception as csv_error:
                    print(f"❌ ERROR: Failed to write mouse scroll to CSV: {csv_error}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in mouse scroll handler: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            # Check recording state
            if not hasattr(self, 'action_recording') or not self.action_recording:
                print(f"🔍 DEBUG: Key press ignored - action_recording: {getattr(self, 'action_recording', 'undefined')}")
                return
            
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"🔍 DEBUG: Key press ignored - no runelite window")
                return
            
            # Get key character
            try:
                if hasattr(key, 'char') and key.char is not None:
                    key_char = key.char
                elif hasattr(key, 'name') and key.name is not None:
                    key_char = key.name
                else:
                    key_char = str(key)
                
                if not key_char or key_char == 'None':
                    key_char = str(key)
            except Exception as key_error:
                print(f"❌ ERROR: Failed to get key character: {key_error}")
                key_char = str(key)
            
            self.key_press_count += 1
            
            # Write to CSV with buffering
            try:
                if hasattr(self, 'csv_writer') and self.csv_writer:
                    action_data = [
                        int(time.time() * 1000), 'key_press', 0, 0, 
                        '', key_char, 0, 0, '', ''
                    ]
                    self._write_csv_buffered(action_data)
                else:
                    print(f"❌ ERROR: No CSV writer available for key press")
                    import traceback
                    traceback.print_exc()
            except Exception as csv_error:
                print(f"❌ ERROR: Failed to write key press to CSV: {csv_error}")
                import traceback
                traceback.print_exc()
                    
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in key press handler: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _on_key_release(self, key):
        """Handle key release events"""
        try:
            # Check recording state
            if not hasattr(self, 'action_recording') or not self.action_recording:
                print(f"🔍 DEBUG: Key release ignored - action_recording: {getattr(self, 'action_recording', 'undefined')}")
                return
            
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"🔍 DEBUG: Key release ignored - no runelite window")
                return
            
            # Get key character
            try:
                if hasattr(key, 'char') and key.char is not None:
                    key_char = key.char
                elif hasattr(key, 'name') and key.name is not None:
                    key_char = key.name
                else:
                    key_char = str(key)
                
                if not key_char or key_char == 'None':
                    key_char = str(key)
            except Exception as key_error:
                print(f"❌ ERROR: Failed to get key character: {key_error}")
                key_char = str(key)
            
            # Write to CSV with buffering
            try:
                if hasattr(self, 'csv_writer') and self.csv_writer:
                    action_data = [
                        int(time.time() * 1000), 'key_release', 0, 0, 
                        '', key_char, 0, 0, '', ''
                    ]
                    self._write_csv_buffered(action_data)
                else:
                    print(f"❌ ERROR: No CSV writer available for key release")
                    import traceback
                    traceback.print_exc()
            except Exception as csv_error:
                print(f"❌ ERROR: Failed to write key release to CSV: {csv_error}")
                import traceback
                traceback.print_exc()
                    
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in key release handler: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _copy_gamestate_path(self):
        """Copy the absolute gamestate directory path to clipboard"""
        try:
            import pyperclip
            import os
            
            if hasattr(self, 'session_dir') and self.session_dir:
                gamestate_path = os.path.abspath(os.path.join(self.session_dir, "gamestates"))
                pyperclip.copy(gamestate_path)
                print(f"✅ Copied gamestate path to clipboard: {gamestate_path}")
                
                # Show temporary success message
                original_text = self.copy_path_button.cget("text")
                self.copy_path_button.config(text="✅ Copied!")
                self.after(2000, lambda: self.copy_path_button.config(text=original_text))
            else:
                print(f"❌ No session directory available for copying")
                
        except ImportError:
            print(f"❌ ERROR: pyperclip not installed. Install with: pip install pyperclip")
            # Fallback: show path in console
            if hasattr(self, 'session_dir') and self.session_dir:
                import os
                gamestate_path = os.path.abspath(os.path.join(self.session_dir, "gamestates"))
                print(f"📋 Gamestate path (copy manually): {gamestate_path}")
        except Exception as e:
            print(f"❌ ERROR copying path to clipboard: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_recorder_status(self):
        """Update recorder status displays"""
        try:
            # Update action count displays
            try:
                self.move_count_label.config(text=f"Mouse Moves: {self.mouse_move_count}")
                self.click_count_label.config(text=f"Clicks: {self.click_count}")
                self.scroll_count_label.config(text=f"Scrolls: {self.scroll_count}")
                self.key_press_label.config(text=f"Key Presses: {self.key_press_count}")
            except Exception as ui_error:
                print(f"❌ Error updating action count displays: {ui_error}")
                import traceback
                traceback.print_exc()
            
            # Update watchdog status and directory info
            if hasattr(self, 'controller') and self.controller:
                try:
                    watchdog_display = self.controller.get_current_directory_display()
                    if hasattr(self, 'file_info_label'):
                        self.file_info_label.config(text=watchdog_display, foreground="#FF6B6B")  # Light coral red
                    
                    # Update detailed watchdog status
                    watchdog_status = self.controller.get_watchdog_status()
                    if hasattr(self, 'watchdog_status_label'):
                        if watchdog_status.get('available', False):
                            status_text = f"🔍 Watchdog: {'🟢 Active' if watchdog_status.get('watching', False) else '🔴 Inactive'}"
                            if watchdog_status.get('watcher_thread_alive', False):
                                status_text += " | 🧵 Watcher: 🟢"
                            else:
                                status_text += " | 🧵 Watcher: 🔴"
                            if watchdog_status.get('feature_thread_alive', False):
                                status_text += " | 🧵 Feature: 🟢"
                            else:
                                status_text += " | 🧵 Feature: 🔴"
                            
                            self.watchdog_status_label.config(text=status_text, foreground="#E85A71")  # Light burgundy
                        else:
                            self.watchdog_status_label.config(text=f"🔍 Watchdog: 🔴 {watchdog_status.get('error', 'Not available')}", foreground="#C44569")  # Medium burgundy
                except Exception as e:
                    print(f"❌ Error getting watchdog status: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        if hasattr(self, 'file_info_label'):
                            self.file_info_label.config(text=f"🔴 Error getting watchdog status: {e}", foreground="#C44569")  # Medium burgundy
                        if hasattr(self, 'watchdog_status_label'):
                            self.watchdog_status_label.config(text=f"🔍 Watchdog: 🔴 Error: {e}", foreground="#C44569")  # Medium burgundy
                    except Exception as ui_error:
                        print(f"❌ Failed to update watchdog UI labels: {ui_error}")
                        traceback.print_exc()
            
            # Update session duration if recording
            if self.recording and self.session_start_time:
                try:
                    duration = int(time.time() - self.session_start_time)
                    minutes = duration // 60
                    seconds = duration % 60
                    total_actions = self.click_count + self.key_press_count + self.scroll_count + self.mouse_move_count
                    
                    # Update status label with duration and action count
                    if hasattr(self, 'recorder_status_label'):
                        self.recorder_status_label.config(text=f"🎬 Recording... {minutes:02d}:{seconds:02d} | {total_actions} total actions", foreground="#E74C3C")  # Medium red
                    
                    # Update action counts label
                    if hasattr(self, 'action_counts_label'):
                        self.action_counts_label.config(text=f"📊 Actions: {self.click_count} clicks, {self.key_press_count} keys, {self.scroll_count} scrolls, {self.mouse_move_count} moves")
                except Exception as duration_error:
                    print(f"❌ Error updating session duration: {duration_error}")
                    import traceback
                    traceback.print_exc()
                
                # Update CSV stats every 10 seconds
                # if hasattr(self, 'file_info_label') and time.time() % 10 < 0.1:
                #     csv_stats = self._get_csv_stats()
                #     features_stats = self._get_features_csv_stats()
                #     gamestate_stats = self._get_gamestate_stats()
                #     
                #     current_text = self.file_info_label.cget("text")
                #     new_parts = []
                #     
                #     # Split existing text and update relevant parts
                #     if " | " in current_text:
                #         parts = current_text.split(" | ")
                #         for part in parts:
                #             if "📊 CSV:" in part:
                #                 new_parts.append(csv_stats)
                #             elif "🧠 Features:" in part:
                #                 new_parts.append(features_stats)
                #             elif "🎮 Gamestates:" in part:
                #                 new_parts.append(gamestate_stats)
                #             else:
                #                 new_parts.append(part)
                #     else:
                #         new_parts = [current_text]
                #     
                #     # Add missing stats if not present
                #     if not any("🧠 Features:" in part for part in new_parts):
                #         new_parts.append(features_stats)
                #     if not any("🎮 Gamestates:" in part for part in new_parts):
                #         new_parts.append(gamestate_stats)
                #     
                #     self.file_info_label.config(text=" | ".join(new_parts))
            
            # Schedule next update
            self.after(100, self._update_recorder_status)
        except Exception as e:
            import traceback
            print(f"❌ Critical error in _update_recorder_status: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
            # Try to schedule next update even after error to prevent complete freezing
            try:
                self.after(1000, self._update_recorder_status)  # Longer delay after error
            except Exception as schedule_error:
                print(f"❌ Failed to schedule next update: {schedule_error}")
                traceback.print_exc()
    
    def _check_auto_save(self):
        """Check if it's time for auto-save during long sessions"""
        try:
            if (self.recording and self.recording_session_active and 
                time.time() - self.last_auto_save >= self.auto_save_interval):
                
                self._auto_save_session()
                self.last_auto_save = time.time()
            
            # Schedule next check
            self.after(1000, self._check_auto_save)
            
        except Exception as e:
            import traceback
            print(f"❌ Auto-save check failed: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _auto_save_session(self):
        """Auto-save session data to prevent data loss"""
        try:
            if not self.session_data:
                return
                
            # Create auto-save filename
            timestamp = int(time.time())
            filename = f"data/session_autosave_{timestamp}.json"
            
            # Prepare auto-save data
            auto_save_data = {
                "auto_save_info": {
                    "timestamp": timestamp,
                    "session_duration": time.time() - self.session_start_time if self.session_start_time else 0,
                    "total_actions": len(self.session_data),
                    "counts": {
                        "mouse_moves": self.mouse_move_count,
                        "clicks": self.click_count,
                        "scrolls": self.scroll_count,
                        "key_presses": self.key_press_count
                    }
                },
                "actions": self.session_data
            }
            
            # Save to file
            import json
            with open(filename, 'w') as f:
                json.dump(auto_save_data, f, indent=2)
            
            # Update status
            # self.file_info_label.config(text=f"💾 Auto-saved: {filename}", foreground="green")
            print(f"💾 Auto-saved session data to {filename}")
            
        except Exception as e:
            import traceback
            print(f"❌ Auto-save failed: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _update_session_data(self, action_data):
        """Add action data to session for auto-save"""
        try:
            if self.recording_session_active:
                self.session_data.append(action_data)
        except Exception as e:
            import traceback
            print(f"❌ Failed to update session data: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _write_csv_buffered(self, action_data):
        """Write action data to CSV with buffering for massive datasets"""
        try:
            # Add to buffer
            self.csv_buffer.append(action_data)
            
            # Check if we should flush the buffer
            current_time = time.time()
            should_flush = (
                len(self.csv_buffer) >= self.csv_buffer_size or  # Buffer full
                (current_time - self.last_csv_flush) >= self.csv_flush_interval  # Time to flush
            )
            
            if should_flush:
                self._flush_csv_buffer()
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to write CSV buffer: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _flush_csv_buffer(self):
        """Flush the CSV buffer to disk"""
        try:
            if not self.csv_buffer or not hasattr(self, 'csv_writer'):
                return
            
            # Write all buffered actions at once
            for action_data in self.csv_buffer:
                self.csv_writer.writerow(action_data)
            
            # Flush to disk
            if hasattr(self, 'csvf') and self.csvf:
                self.csvf.flush()
            
            # Update status
            self.last_csv_flush = time.time()
            buffer_size = len(self.csv_buffer)
            
            # Log buffer flush (but not too frequently)
            if buffer_size >= self.csv_buffer_size:
                print(f"💾 Flushed {buffer_size} actions to CSV (buffer full)")
            
            # Clear buffer
            self.csv_buffer.clear()
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to flush CSV buffer: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _finalize_csv(self):
        """Finalize CSV file and ensure all data is written"""
        try:
            # Flush any remaining buffer
            if self.csv_buffer:
                self._flush_csv_buffer()
            
            # Close CSV file properly
            if hasattr(self, 'csvf') and self.csvf:
                self.csvf.close()
                self.csvf = None
                self.csv_writer = None
            
            print("✅ CSV file finalized and closed")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to finalize CSV: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _get_csv_stats(self):
        """Get statistics about the CSV file"""
        try:
            if not hasattr(self, 'csvf') or not self.csvf:
                return "No CSV file open"
            
            # Get file size
            import os
            csv_path = "data/actions.csv"
            if os.path.exists(csv_path):
                size_bytes = os.path.getsize(csv_path)
                size_mb = size_bytes / (1024 * 1024)
                
                # Estimate line count (rough estimate)
                estimated_lines = int(size_bytes / 100)  # Rough estimate: 100 bytes per line
                
                return f"📊 CSV: {size_mb:.1f} MB, ~{estimated_lines:,} lines"
            else:
                return "CSV file not found"
                
        except Exception as e:
            return f"Error getting CSV stats: {e}"
    
    def _save_gamestate(self, gamestate_data):
        """Save a gamestate to the session directory"""
        try:
            if not self.gamestate_recording or not self.gamestate_session_dir:
                return
            
            import os
            import json
            
            # Create timestamped filename
            timestamp = int(time.time() * 1000)
            filename = f"gamestate_{timestamp}.json"
            filepath = os.path.join(self.gamestate_session_dir, filename)
            
            # Save gamestate data
            with open(filepath, 'w') as f:
                json.dump(gamestate_data, f, indent=2)
            
            self.gamestate_count += 1
            
            # Update status every 10 gamestates
            if self.gamestate_count % 10 == 0:
                print(f"💾 Saved {self.gamestate_count} gamestates to {self.gamestate_session_dir}")
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to save gamestate: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _start_gamestate_recording(self):
        """Start recording gamestates to session directory"""
        try:
            import os
            
            # Use session directory if available, otherwise create a timestamped one
            if hasattr(self, 'session_dir') and self.session_dir:
                self.gamestate_session_dir = os.path.join(self.session_dir, "gamestates")
            else:
                timestamp = int(time.time())
                session_name = f"recording_session_{timestamp}"
                self.gamestate_session_dir = os.path.join("data", "bot1", "gamestates", session_name)
            
            os.makedirs(self.gamestate_session_dir, exist_ok=True)
            
            self.gamestate_recording = True
            self.gamestate_count = 0
            
            print(f"🎬 Started gamestate recording to: {self.gamestate_session_dir}")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to start gamestate recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _start_action_recording(self):
        """Start action recording automatically with the session"""
        try:
            print(f"🔍 DEBUG: Starting action recording - session_dir: {getattr(self, 'session_dir', 'None')}")
            
            # Check if we have a session directory
            if not hasattr(self, 'session_dir') or not self.session_dir:
                raise RuntimeError("No session directory available for action recording")
            
            # Check if runelite window is detected
            if not hasattr(self, 'runelite_window') or not self.runelite_window:
                print(f"⚠️ WARNING: No runelite window detected, action recording will be limited")
                # Don't fail completely, but note the limitation
            
            # Initialize CSV file for actions
            import os
            import csv
            import time
            
            csv_path = os.path.join(self.session_dir, "actions.csv")
            print(f"🔍 DEBUG: Creating actions.csv at: {csv_path}")
            
            # Verify directory exists and is writable
            csv_dir = os.path.dirname(csv_path)
            if not os.path.exists(csv_dir):
                raise RuntimeError(f"Directory does not exist: {csv_dir}")
            
            if not os.access(csv_dir, os.W_OK):
                raise RuntimeError(f"Directory not writable: {csv_dir}")
            
            # Create CSV file
            try:
                self.csvf = open(csv_path, 'w', newline='')
                print(f"✅ Actions CSV file opened successfully: {csv_path}")
            except Exception as csv_error:
                raise RuntimeError(f"Failed to open actions CSV file {csv_path}: {csv_error}")
            
            # Create CSV writer and write header
            try:
                self.csv_writer = csv.writer(self.csvf)
                self.csv_writer.writerow([
                    'timestamp', 'event_type', 'x_in_window', 'y_in_window', 
                    'btn', 'key', 'scroll_dx', 'scroll_dy', 'modifiers', 'active_keys'
                ])
                print(f"✅ Actions CSV header written successfully")
            except Exception as header_error:
                raise RuntimeError(f"Failed to write actions CSV header: {header_error}")
            
            # CRITICAL: Initialize CSV buffer variables
            try:
                self.csv_buffer = []
                self.last_csv_flush = time.time()
                print(f"✅ Actions CSV buffer initialized - size: {self.csv_buffer_size}, flush_interval: {self.csv_flush_interval}")
            except Exception as buffer_error:
                raise RuntimeError(f"Failed to initialize actions CSV buffer: {buffer_error}")
            
            # Reset action counters
            self.click_count = 0
            self.key_press_count = 0
            self.scroll_count = 0
            self.mouse_move_count = 0
            print(f"✅ Action counters reset")
            
            # Set action recording state
            self.action_recording = True
            print(f"✅ Action recording state set to: {self.action_recording}")
            
            # Stop the gamestate cleanup system since we're now actively recording
            self.gamestate_cleanup_active = False
            print(f"🛑 Stopped gamestate cleanup system - now actively recording")
            
            print(f"🎬 Action recording started successfully - file: {csv_path}")
            
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR starting action recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
            
            # Clean up on failure
            try:
                if hasattr(self, 'csvf') and self.csvf:
                    self.csvf.close()
                    self.csvf = None
                if hasattr(self, 'csv_writer'):
                    self.csv_writer = None
                if hasattr(self, 'csv_buffer'):
                    self.csv_buffer = []
                self.action_recording = False
            except Exception as cleanup_error:
                print(f"❌ ERROR during cleanup: {cleanup_error}")
            
            # Re-raise to prevent silent failure
            raise
    
    def _stop_gamestate_recording(self):
        """Stop recording gamestates"""
        try:
            self.gamestate_recording = False
            
            if self.gamestate_session_dir:
                print(f"💾 Stopped gamestate recording. Saved {self.gamestate_count} gamestates to: {self.gamestate_session_dir}")
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to stop gamestate recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _stop_action_recording(self):
        """Stop action recording"""
        try:
            if hasattr(self, 'action_recording') and self.action_recording:
                self.action_recording = False
                print(f"💾 Stopped action recording")
                
                # Finalize CSV file
                try:
                    self._finalize_csv()
                    print(f"✅ Actions CSV file finalized")
                except Exception as finalize_error:
                    print(f"❌ ERROR: Failed to finalize actions CSV: {finalize_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"🔍 DEBUG: Action recording was not active")
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to stop action recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _get_gamestate_stats(self):
        """Get statistics about recorded gamestates"""
        try:
            if not self.gamestate_session_dir:
                return "No gamestate session active"
            
            import os
            
            if os.path.exists(self.gamestate_session_dir):
                # Count files and get directory size
                files = [f for f in os.listdir(self.gamestate_session_dir) if f.endswith('.json')]
                file_count = len(files)
                
                # Calculate directory size
                total_size = 0
                for filename in files:
                    filepath = os.path.join(self.gamestate_session_dir, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                
                size_mb = total_size / (1024 * 1024)
                
                return f"🎮 Gamestates: {file_count} files, {size_mb:.1f} MB"
            else:
                return "Gamestate directory not found"
                
        except Exception as e:
            return f"Error getting gamestate stats: {e}"
    
    def _start_feature_recording(self):
        """Start recording live feature vectors to CSV"""
        try:
            print("🚀 Starting feature recording...")
            import os
            import csv
            
            # Use session directory if available, otherwise fallback to data directory
            if hasattr(self, 'session_dir') and self.session_dir:
                features_csv_path = os.path.join(self.session_dir, "features.csv")
            else:
                os.makedirs("data", exist_ok=True)
                features_csv_path = "data/features.csv"
            
            # Initialize features CSV file
            self.features_csvf = open(features_csv_path, 'w', newline='')
            self.features_csv_writer = csv.writer(self.features_csvf)
            
            # Create header: timestamp + 128 feature columns
            header = ['timestamp'] + [f'feature_{i}' for i in range(128)]
            self.features_csv_writer.writerow(header)
            
            self.feature_recording = True
            self.feature_count = 0
            self.feature_buffer = []
            self.last_feature_flush = time.time()
            
            print(f"🎬 Started feature recording to: {features_csv_path}")
            print(f"✅ feature_recording set to: {self.feature_recording}")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to start feature recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _stop_feature_recording(self):
        """Stop recording feature vectors"""
        try:
            self.feature_recording = False
            
            # Finalize features CSV
            self._finalize_features_csv()
            
            print(f"💾 Stopped feature recording. Saved {self.feature_count} feature vectors")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to stop feature recording: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _save_feature_vector(self, feature_vector, timestamp=None):
        """Save a feature vector to the features CSV"""
        try:
            print(f"💾 _save_feature_vector called: feature_recording={self.feature_recording}, has_writer={hasattr(self, 'features_csv_writer')}")
            
            if not self.feature_recording or not self.features_csv_writer:
                print(f"❌ Early return: feature_recording={self.feature_recording}, has_writer={hasattr(self, 'features_csv_writer')}")
                return
            
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            
            # Prepare feature row: timestamp + 128 feature values
            if hasattr(feature_vector, 'flatten'):
                features = feature_vector.flatten()
            else:
                features = feature_vector
            
            # Ensure we have exactly 128 features
            if len(features) != 128:
                LOG.warning(f"Feature vector has {len(features)} features, expected 128")
                print(f"⚠️ Feature vector has {len(features)} features, expected 128")
                return
            
            # Create row data
            feature_row = [timestamp] + features.tolist()
            
            # Add to buffer for batch writing
            self._write_features_csv_buffered(feature_row)
            
            self.feature_count += 1
            print(f"✅ Feature vector saved! Count: {self.feature_count}")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to save feature vector: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _write_features_csv_buffered(self, feature_row):
        """Write feature data to CSV with buffering"""
        try:
            # Add to buffer
            self.feature_buffer.append(feature_row)
            
            # Check if we should flush the buffer
            current_time = time.time()
            should_flush = (
                len(self.feature_buffer) >= self.csv_buffer_size or  # Buffer full
                (current_time - self.last_feature_flush) >= self.csv_flush_interval  # Time to flush
            )
            
            if should_flush:
                self._flush_features_csv_buffer()
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to write features CSV buffer: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _flush_features_csv_buffer(self):
        """Flush the features CSV buffer to disk"""
        try:
            if not self.feature_buffer or not self.features_csv_writer:
                return
            
            # Write all buffered features at once
            for feature_row in self.feature_buffer:
                self.features_csv_writer.writerow(feature_row)
            
            # Flush to disk
            if self.features_csvf:
                self.features_csvf.flush()
            
            # Update status
            self.last_feature_flush = time.time()
            buffer_size = len(self.feature_buffer)
            
            # Log buffer flush (but not too frequently)
            if buffer_size >= self.csv_buffer_size:
                print(f"💾 Flushed {buffer_size} feature vectors to CSV (buffer full)")
            
            # Clear buffer
            self.feature_buffer.clear()
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to flush features CSV buffer: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _finalize_features_csv(self):
        """Finalize features CSV file and ensure all data is written"""
        try:
            # Flush any remaining buffer
            if self.feature_buffer:
                self._flush_features_csv_buffer()
            
            # Close CSV file properly
            if self.features_csvf:
                self.features_csvf.close()
                self.features_csvf = None
                self.features_csv_writer = None
            
            print("✅ Features CSV file finalized and closed")
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to finalize features CSV: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _get_features_csv_stats(self):
        """Get statistics about the features CSV file"""
        try:
            if not self.features_csvf:
                return "No features CSV file open"
            
            # Get file size
            import os
            features_csv_path = "data/features.csv"
            if os.path.exists(features_csv_path):
                size_bytes = os.path.getsize(features_csv_path)
                size_mb = size_bytes / (1024 * 1024)
                
                # Estimate line count (each feature vector is ~1KB)
                estimated_lines = int(size_bytes / 1000)
                
                return f"🧠 Features: {size_mb:.1f} MB, ~{estimated_lines:,} vectors"
            else:
                return "Features CSV file not found"
                
        except Exception as e:
            return f"Error getting features CSV stats: {e}"
    
    def _start_feature_listener(self):
        """Start listening for feature updates from the controller"""
        try:
            # Check if controller has a UI queue
            if hasattr(self.controller, 'ui_queue'):
                # Start periodic check for messages
                self._check_for_feature_updates()
            else:
                LOG.warning("Controller has no UI queue for feature updates")
        except Exception as e:
            import traceback
            print(f"❌ Failed to start feature listener: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _check_for_feature_updates(self):
        """Check for feature updates from the controller"""
        try:
            # Check for messages in the UI queue
            if hasattr(self.controller, 'ui_queue'):
                queue_size = self.controller.ui_queue.qsize()
                if queue_size > 0:
                    print(f"📨 Found {queue_size} messages in UI queue")
                
                while not self.controller.ui_queue.empty():
                    try:
                        msg_type, data = self.controller.ui_queue.get_nowait()
                        print(f"📨 Processing message: {msg_type}")
                        if msg_type == "table_update":
                            # Extract feature data
                            window, changed_mask, feature_names, feature_groups = data
                            print(f"📊 Table update: window={window.shape if window is not None else None}")
                            self._handle_feature_update(window, changed_mask, feature_names, feature_groups)
                    except Exception as e:
                        LOG.error(f"Error processing feature update: {e}")
                        print(f"❌ Error processing feature update: {e}")
            else:
                print("⚠️ Controller has no UI queue")
            
            # Schedule next check
            self.after(100, self._check_for_feature_updates)
            
        except Exception as e:
            import traceback
            print(f"❌ Error in feature update check: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _cleanup_gamestates_if_needed(self):
        """Clean up auto-dumped gamestates until recording actually starts"""
        try:
            current_time = time.time()
            
            # Only run cleanup if it's time, cleanup is active, and we're not actively recording
            if (current_time - self.last_gamestate_cleanup >= self.gamestate_cleanup_interval and 
                self.gamestate_cleanup_active and 
                not self.gamestate_recording):
                
                if hasattr(self, 'session_dir') and self.session_dir:
                    gamestate_dir = os.path.join(self.session_dir, "gamestates")
                    
                    if os.path.exists(gamestate_dir):
                        # Count existing gamestate files
                        gamestate_files = [f for f in os.listdir(gamestate_dir) if f.endswith('.json')]
                        file_count = len(gamestate_files)
                        
                        # If we have too many files, clean them up
                        if file_count > self.gamestate_cleanup_threshold:
                            print(f"🧹 Cleaning up {file_count} auto-dumped gamestates (recording not started yet)")
                            
                            # Remove oldest files first (keep newest 10)
                            gamestate_files.sort(key=lambda f: os.path.getmtime(os.path.join(gamestate_dir, f)))
                            files_to_remove = gamestate_files[:-10]  # Keep newest 10
                            
                            removed_count = 0
                            for filename in files_to_remove:
                                try:
                                    filepath = os.path.join(gamestate_dir, filename)
                                    os.remove(filepath)
                                    removed_count += 1
                                except Exception as remove_error:
                                    print(f"⚠️ Failed to remove {filename}: {remove_error}")
                            
                            print(f"✅ Cleaned up {removed_count} gamestate files (kept newest 10)")
                            
                            # Update the cleanup timestamp
                            self.last_gamestate_cleanup = current_time
                
        except Exception as e:
            print(f"❌ Error during gamestate cleanup: {e}")
            import traceback
            traceback.print_exc()
        
        # Schedule next cleanup check
        self.after(1000, self._cleanup_gamestates_if_needed)
    
    def _handle_feature_update(self, window, changed_mask, feature_names, feature_groups):
        """Handle incoming feature data update"""
        try:
            print(f"🔍 Feature update received: window={window.shape if window is not None else None}, feature_recording={self.feature_recording}")
            
            # Update schema if needed
            if not self._schema_set and feature_names and feature_groups:
                self.set_schema(feature_names, feature_groups)
            
            # Update the feature window
            if window is not None and self._schema_set:
                self._update_feature_window(window, changed_mask)
                
                # Update summary label
                if hasattr(self, 'summary_label'):
                    self.summary_label.config(text=f"Features: {len(feature_names)}/128 | Buffer: {window.shape[0]}/10 | Status: Live Mode Active")
                
                # Save live feature vector if recording is active
                if self.feature_recording and window.shape[0] > 0:
                    print(f"💾 Saving feature vector: shape={window.shape}, feature_recording={self.feature_recording}")
                    # T0 features must be saved with the T0 gamestate's absolute timestamp
                    current_features = window[0]  # T0
                    ts = None
                    try:
                        fp = getattr(self.controller, "feature_pipeline", None)
                        if fp and hasattr(fp, "get_last_gamestate_timestamp"):
                            ts = fp.get_last_gamestate_timestamp()
                    except Exception:
                        ts = None
                    # Fallback (shouldn't normally happen): use wall clock
                    if ts is None:
                        import time as _time
                        ts = int(_time.time() * 1000)
                    self._save_feature_vector(current_features, ts)
                else:
                    print(f"❌ Not saving: feature_recording={self.feature_recording}, window.shape[0]={window.shape[0] if window is not None else 'None'}")
                
                # Save gamestate if recording is active
                if self.gamestate_recording and hasattr(self.controller, 'get_current_gamestate'):
                    try:
                        gamestate_data = self.controller.get_current_gamestate()
                        if gamestate_data:
                            self._save_gamestate(gamestate_data)
                    except Exception as e:
                        LOG.debug(f"Could not get current gamestate: {e}")
                    
        except Exception as e:
            import traceback
            print(f"❌ Error handling feature update: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    def _update_feature_window(self, window, changed_mask):
        """Update the feature window with new data"""
        try:
            if not self._schema_set:
                return
                
            # Update the table directly with the new data
            self.update_from_window(window, changed_mask)
            
        except Exception as e:
            import traceback
            print(f"❌ Error updating feature window: {e}")
            print(f"❌ Stack trace:")
            traceback.print_exc()
    
    
    
    def _ensure_actions_group_position(self):
        """Ensure the Actions group is positioned correctly relative to other groups"""
        try:
            print("🔍 DEBUG: _ensure_actions_group_position called")
            
            if not hasattr(self, 'group_rows') or "Actions" not in self.group_rows:
                print("🔍 DEBUG: No Actions group to reposition")
                return
            
            # Calculate where the Actions group should be positioned
            target_position = 0
            for group_name, group_info in self.group_rows.items():
                if group_name == "Actions":
                    continue
                
                target_position += 1  # Count header row
                # rely solely on self.expanded_groups; group_info['expanded'] can get stale
                if group_name in self.expanded_groups:
                    target_position += len(group_info['feature_rows'])  # Count feature rows
            
            # Check if Actions group is at the correct position
            current_actions_info = self.group_rows["Actions"]
            if current_actions_info['header_row'] == target_position:
                print("🔍 DEBUG: Actions group already at correct position")
                return
            
            print(f"🔍 DEBUG: Actions group needs repositioning from {current_actions_info['header_row']} to {target_position}")
            
            # Remove old Actions group
            if current_actions_info['feature_rows']:
                for row_idx in reversed(current_actions_info['feature_rows']):
                    try:
                        self.sheet.delete_row(row_idx)
                    except Exception as e:
                        print(f"⚠️ WARNING: Failed to delete Actions feature row {row_idx}: {e}")
            
            try:
                self.sheet.delete_row(current_actions_info['header_row'])
            except Exception as e:
                print(f"⚠️ WARNING: Failed to delete Actions header: {e}")
            
            # Add Actions group at the correct position
            self._add_actions_group(target_position)
            
            print(f"🔍 DEBUG: Actions group repositioned to row {target_position}")
            
        except Exception as e:
            print(f"❌ ERROR in _ensure_actions_group_position: {e}")
            import traceback
            traceback.print_exc()
