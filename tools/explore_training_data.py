#!/usr/bin/env python3
"""
Simple Training Data Pipeline Explorer
"""

import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import os, sys, time, traceback, logging
from pathlib import Path
import pandas as pd
import numpy as np

class SimpleDataExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Data Explorer")
        self.root.geometry("1200x800")
        # --- Logging: console + file ---
        log_path = os.path.join(os.path.dirname(__file__), "explorer_debug.log")
        self.logger = logging.getLogger("explore_training_data")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            sh = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt); sh.setFormatter(fmt)
            self.logger.addHandler(fh); self.logger.addHandler(sh)
        self._log("=== Explorer started ===")

        # Make Tk and Python always show exceptions
        def _report_callback_exception(exc, val, tb):
            self._log("Tk callback exception", level="error")
            traceback.print_exception(exc, val, tb)
            self._safe_status(f"Error: {val}")
        self.root.report_callback_exception = _report_callback_exception
        def _exhook(exctype, value, tb):
            self._log("sys.excepthook", level="error")
            traceback.print_exception(exctype, value, tb)
        sys.excepthook = _exhook
        
        # Default data directory
        self.data_dir = "D:/repos/bot_runelite_IL/data"
        
        self.setup_ui()
        # Initialize 3D data tracking
        self.current_gamestate = 0
        self.total_gamestates = 0
        # Navigation for 4D arrays (timestep)
        self.current_action_slice = 0   # used as TIMESTEP index for 4D numpy arrays
        self.total_action_slices = 0
        self.action_data = None
        self.show_mapped_values = True  # Toggle for mapped vs raw display
        self.current_file_type = None  # Track current file type
        self.state_features_scroll_position = 0  # Save scroll position for state features
        self.load_files()

    # --- Treeview column helper: ALWAYS set columns + displaycolumns together ---
    def _set_tree_columns(self, tree, cols):
        self._log("_set_tree_columns", cols=list(cols))
        tree.configure(columns=cols, displaycolumns=cols)

    # ---------- logging helpers ----------
    def _log(self, msg, level="debug", **ctx):
        try:
            if ctx:
                msg = f"{msg} | {ctx}"
            getattr(self.logger, level)(msg)
        except Exception:
            print(msg, ctx, file=sys.stderr)

    def _safe_status(self, text):
        try:
            self.status_var.set(text)
            self.root.update_idletasks()
        except Exception:
            pass
    
    def setup_ui(self):
        # Simple controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Data Directory:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=self.data_dir)
        ttk.Entry(control_frame, textvariable=self.dir_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.load_files).pack(side=tk.LEFT, padx=5)
        
        # File selector
        ttk.Label(control_frame, text="File:").pack(side=tk.LEFT, padx=(20,5))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(control_frame, textvariable=self.file_var, width=40)
        self.file_combo.pack(side=tk.LEFT, padx=5)
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        # Log user typing/enter as well
        self.file_combo.bind('<Return>', self.on_file_selected)
        
        # Gamestate navigation (for 3D data)
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(nav_frame, text="Gamestate:").pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="←", command=self.prev_gamestate).pack(side=tk.LEFT, padx=2)
        self.gamestate_var = tk.StringVar(value="0")
        gamestate_entry = ttk.Entry(nav_frame, textvariable=self.gamestate_var, width=10)
        gamestate_entry.pack(side=tk.LEFT, padx=5)
        gamestate_entry.bind('<Return>', lambda e: self.on_gamestate_entry_change())
        ttk.Button(nav_frame, text="→", command=self.next_gamestate).pack(side=tk.LEFT, padx=2)
        self.total_gamestates_label = ttk.Label(nav_frame, text="of 0")
        self.total_gamestates_label.pack(side=tk.LEFT, padx=5)
        
        # Bind entry field to update display
        self.gamestate_var.trace('w', self.on_gamestate_entry_change)
        
        # Toggle for mapped vs raw display
        ttk.Label(nav_frame, text="Display:").pack(side=tk.LEFT, padx=(20,5))
        self.mapped_var = tk.BooleanVar(value=True)
        self.mapped_check = ttk.Checkbutton(nav_frame, text="Mapped Values", variable=self.mapped_var, 
                                           command=self.on_mapped_toggle)
        self.mapped_check.pack(side=tk.LEFT, padx=5)

        # --- Secondary navigator for 4D arrays (action slice) ---
        ttk.Label(nav_frame, text="  Timestep:").pack(side=tk.LEFT, padx=(16, 5))
        self.slice_prev_btn = ttk.Button(nav_frame, text="←", command=self.prev_action_slice)
        self.slice_prev_btn.pack(side=tk.LEFT, padx=2)
        self.action_slice_var = tk.StringVar(value="0")
        self.slice_entry = ttk.Entry(nav_frame, textvariable=self.action_slice_var, width=6)
        self.slice_entry.pack(side=tk.LEFT, padx=4)
        self.slice_entry.bind('<Return>', lambda e: self.on_action_slice_entry_change())
        self.slice_next_btn = ttk.Button(nav_frame, text="→", command=self.next_action_slice)
        self.slice_next_btn.pack(side=tk.LEFT, padx=2)
        self.total_action_slices_label = ttk.Label(nav_frame, text="of 0")
        self.total_action_slices_label.pack(side=tk.LEFT, padx=5)

        # Hide/disable slice controls by default (only shown for 4D)
        for w in (self.slice_prev_btn, self.slice_entry, self.slice_next_btn, self.total_action_slices_label):
            w.state(["disabled"])
        
        # Data display
        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tree = ttk.Treeview(tree_frame, show="tree headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, padx=10)
        
    def _autosize_columns(self, tree, padding=20, max_px=260):
        """Resize Treeview columns to fit content (headers + visible rows)."""
        self._log("_autosize_columns: start", columns=list(tree["columns"]))
        # Safe font lookup across Tk builds
        try:
            style = ttk.Style(tree)
            font_name = style.lookup("Treeview", "font") or "TkDefaultFont"
            f = tkfont.nametofont(font_name)
        except Exception:
            try:
                f = tkfont.nametofont("TkDefaultFont")
            except Exception:
                f = None  # fall back to fixed widths below

        if not tree["columns"]:
            self._log("_autosize_columns: no columns, return early")
            return  # nothing to size yet
        # include #0 separately
        for col in ("#0",) + tuple(tree["columns"]):
            header = tree.heading(col, "text") or ""
            width = f.measure(header) if f else 120
            # measure visible rows (don't iterate the whole dataset for speed)
            for iid in tree.get_children():
                cell = tree.set(iid, col) if col != "#0" else tree.item(iid, "text")
                width = max(width, f.measure(str(cell)) if f else width)
            tree.column(col, width=min(width + padding, max_px))
        self._log("_autosize_columns: done",
                  item_count=len(tree.get_children()),
                  columns=list(tree["columns"]))
    
    # --- BEGIN: exact 4D numpy viewer behavior (copied/adapted from print_numpy_array.py) ---
    def _ensure_4d_order_batch_timestep_action_feature(self, data4d: np.ndarray) -> np.ndarray:
        """
        Return a view reordered as (batch, timestep, action, feature).
        Your action_targets.npy is (B, A, T, F). The print viewer expects (B, T, A, F).
        We swap axes 1 and 2 when needed so the UI logic matches.
        """
        if data4d.ndim != 4:
            return data4d
        # If dim-1 < dim-2 (e.g., 10 < 100), it's probably (A, T) -> swap to (T, A)
        return data4d.swapaxes(1, 2) if data4d.shape[1] < data4d.shape[2] else data4d

    def _configure_numpy_tree_4d(self, tree):
        """Columns like the print viewer: 8 feature headers; #0 label 'Action'."""
        cols = ("count", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
        tree.configure(columns=cols, displaycolumns=cols)
        tree.column("#0", width=100, stretch=tk.NO, anchor="w")
        tree.heading("#0", text="Action")
        for c in cols:
            tree.column(c, anchor=tk.CENTER, width=140, minwidth=120, stretch=True)
            tree.heading(c, text=c)

    def _fmt_num(self, v):
        if isinstance(v, (np.integer, int)):
            return str(int(v))
        if isinstance(v, (np.floating, float)):
            return f"{float(v):.6f}"
        return str(v)

    def _update_numpy_4d_view(self, batch_idx: int, timestep_idx: int):
        """
        Show all 'actions' for (batch, timestep) exactly like print_numpy_array.py:
        rows = actions, columns = 8 features.
        """
        self._log("_update_numpy_4d_view: enter", batch=batch_idx, timestep=timestep_idx)
        data = getattr(self, "current_numpy_data", None)
        if data is None or data.ndim != 4:
            self._log("_update_numpy_4d_view: no 4D data")
            return
        view = self._ensure_4d_order_batch_timestep_action_feature(data)  # -> (B, T, A, F)
        if not (0 <= batch_idx < view.shape[0]):
            self._log("_update_numpy_4d_view: bad batch", batch=batch_idx, max=view.shape[0]-1, level="error")
            return
        if not (0 <= timestep_idx < view.shape[1]):
            self._log("_update_numpy_4d_view: bad timestep", timestep=timestep_idx, max=view.shape[1]-1, level="error")
            return

        tree = self.tree
        tree.delete(*tree.get_children())
        self._configure_numpy_tree_4d(tree)

        # Color map (same semantics the print viewer uses)
        action_colors = {0: "lightblue", 1: "lightgreen", 2: "lightyellow", 3: "lightcoral"}

        # Iterate actions at this timestep
        A = view.shape[2]
        self._log("_update_numpy_4d_view: rendering", actions=A, features=view.shape[3])
        for i in range(A):
            row = view[batch_idx, timestep_idx, i, :]  # (8,)
            values = [self._fmt_num(v) for v in row]
            item = tree.insert("", tk.END, text=f"Action {i}", values=values)
            # Optional row coloring: only when count > 0
            try:
                action_count = int(float(values[0]))
                action_type = int(float(values[1]))
                if action_count > 0 and action_type in action_colors:
                    tag = f"action_{action_type}"
                    tree.tag_configure(tag, background=action_colors[action_type])
                    tree.item(item, tags=(tag,))
            except Exception:
                pass

        # Nice widths, but never let autosize crash the UI
        try:
            self._autosize_columns(tree)
        except Exception:
            self._log("_update_numpy_4d_view: autosize error", level="error")
            traceback.print_exc()
            pass
    # --- END: exact 4D numpy viewer behavior ---

    def browse_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.data_dir)
        if new_dir:
            self.data_dir = new_dir
            self.dir_var.set(new_dir)
            self.load_files()
    
    def load_files(self):
        self.data_dir = self.dir_var.get()
        self._log("load_files: scanning", data_dir=self.data_dir)
        if not os.path.exists(self.data_dir):
            self.status_var.set(f"Directory not found: {self.data_dir}")
            return
        
        # Find all data files
        files = []
        for folder in ["01_raw_data", "02_trimmed_data", "03_normalized_data", "04_sequences", "05_mappings", "06_final_training_data"]:
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.json', '.npy', '.csv')):
                        files.append(f"{folder}/{file}")
        
        self._log("load_files: found files", count=len(files))
        self.file_combo['values'] = files
        if files:
            self.file_combo.set(files[0])
            self._log("load_files: auto-select first file", file=files[0])
            try:
                self.on_file_selected()
            except Exception:
                self._log("load_files: on_file_selected crashed", level="error")
                traceback.print_exc()
        
        self.status_var.set(f"Found {len(files)} files")
    
    def prev_gamestate(self):
        if not self.total_gamestates:
            return
        print(f"PREV: current={self.current_gamestate}, total={self.total_gamestates}")
        if self.current_gamestate > 0:
            self.current_gamestate -= 1
            self.gamestate_var.set(str(self.current_gamestate))
            print(f"PREV: now showing gamestate {self.current_gamestate}")
            self._navigate_gamestate()
        else:
            print("PREV: already at first gamestate")
    
    def next_gamestate(self):
        if not self.total_gamestates:
            return
        if self.current_gamestate < self.total_gamestates - 1:
            self.current_gamestate += 1
            self.gamestate_var.set(str(self.current_gamestate))
            self._navigate_gamestate()
    
    def on_gamestate_entry_change(self, *args):
        """Handle manual entry in gamestate field"""
        self._log("on_gamestate_entry_change", args=args, total=self.total_gamestates)
        if not self.total_gamestates:
            return
        try:
            new_gamestate = int(self.gamestate_var.get())
            if 0 <= new_gamestate < self.total_gamestates:
                self.current_gamestate = new_gamestate
                self._navigate_gamestate()
        except ValueError:
            pass  # Invalid input, ignore
    
    def _navigate_gamestate(self):
        """Navigate to current gamestate based on file type"""
        self._log("_navigate_gamestate", current_file_type=self.current_file_type,
                  index=self.current_gamestate, total=self.total_gamestates)
        # Save current scroll position if we're in state features
        if self.current_file_type == "state_features":
            self.state_features_scroll_position = self.tree.yview()[0]
        
        if self.current_file_type == "state_features":
            self.display_current_gamestate_features()
        elif self.current_file_type == "action_data":
            self.display_current_gamestate()
        elif self.current_file_type in ("numpy_array", "numpy_array_4d"):
            # For numpy arrays, update the slice display without changing columns
            if hasattr(self, 'current_numpy_data'):
                if self.current_file_type == "numpy_array_4d":
                    self._log("_navigate_gamestate: 4D refresh",
                              batch=self.current_gamestate, timestep=self.current_action_slice)
                    self._update_numpy_4d_view(self.current_gamestate, self.current_action_slice)
                else:
                    self._log("_navigate_gamestate: 3D refresh", batch=self.current_gamestate)
                    mock_tab = type('MockTab', (), {'_tree': self.tree, '_data': self.current_numpy_data, '_data_type': 'numpy_array'})()
                    self.update_3d_slice_for_tab(mock_tab, self.current_gamestate)
        # Update status to show current gamestate
        self.status_var.set(f"Showing gamestate {self.current_gamestate} of {self.total_gamestates}")
    
    def on_mapped_toggle(self):
        """Handle toggle between mapped and raw display"""
        self._log("on_mapped_toggle", mapped=self.mapped_var.get())
        self.show_mapped_values = self.mapped_var.get()
        if self.action_data:
            self.display_current_gamestate()
    
    def on_file_selected(self, event=None):
        sel = self.file_var.get()
        self._log("on_file_selected: event", event=str(event), selection=sel)
        if not sel:
            return
        file_path = os.path.join(self.data_dir, sel)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else -1
        self._log("on_file_selected: resolved",
                  file_path=file_path, exists=exists, size=size)
        try:
            self.load_file(file_path)
        except Exception:
            self._log("on_file_selected: load_file crashed", level="error", file=file_path)
            traceback.print_exc()
            self._safe_status(f"Error while loading {os.path.basename(file_path)}")
            try:
                messagebox.showerror("Explorer Error", f"Failed to load:\n{file_path}\n\n{traceback.format_exc()}")
            except Exception:
                pass
    
    def load_file(self, file_path):
        try:
            self._safe_status(f"Loading {os.path.basename(file_path)}...")
            self._log("load_file: start", path=file_path)
            
            # Load file
            if file_path.endswith('.npy'):
                try:
                    data = np.load(file_path, allow_pickle=False)
                except Exception:
                    self._log("np.load failed, retry with allow_pickle=True", level="warning")
                    data = np.load(file_path, allow_pickle=True)
                self._log("load_file: npy loaded",
                          ndim=int(getattr(data, 'ndim', -1)),
                          shape=tuple(getattr(data, 'shape', ())), dtype=str(getattr(data, 'dtype', 'na')))
                self.display_numpy(data)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self._log("load_file: json loaded", type=type(data).__name__)
                self.display_json(data)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                self._log("load_file: csv loaded", rows=len(data), cols=len(data.columns))
                self.display_csv(data)
            else:
                self._log("load_file: unsupported extension", level="warning")
                pass
            
            self._safe_status(f"Loaded {os.path.basename(file_path)}")
            
        except Exception as e:
            self._log("load_file: exception", level="error", error=str(e))
            traceback.print_exc()
            self._safe_status(f"Error: {str(e)}")
    
    def display_numpy(self, data):
        self._log("display_numpy: enter",
                  ndim=int(getattr(data, 'ndim', -1)),
                  shape=tuple(getattr(data, 'shape', ())))
        # Hard reset the tree to avoid duplicated/old headers
        self.tree.delete(*self.tree.get_children())
        # Avoid setting empty () columns which can cause Tk errors on some builds
        
        # Check if this is a features file (2D array with 128 features)
        if len(data.shape) == 2 and data.shape[1] == 128:
            self._log("display_numpy: 2D features view")
            # This is a features file - show as 2 columns with 128 rows
            self.tree.configure(columns=("feature_index", "feature_value"),
                                displaycolumns=("feature_index", "feature_value"))
            self.tree.column("#0", width=100, stretch=tk.NO)
            self.tree.column("feature_index", anchor=tk.CENTER, width=100)
            self.tree.column("feature_value", anchor=tk.CENTER, width=200)
            self.tree.heading("#0", text="Row")
            self.tree.heading("feature_index", text="Index")
            self.tree.heading("feature_value", text="Value")
            
            # Store the features data for gamestate navigation
            self.state_features_data = data
            self.current_file_type = "state_features"
            self.total_gamestates = data.shape[0]
            
            # Update gamestate counter
            self.gamestate_var.set(str(self.current_gamestate))
            self.total_gamestates_label.config(text=f"of {self.total_gamestates}")
            
            # Show features for current gamestate
            self.display_current_gamestate_features()
            return
        
        # Use the new numpy tree configuration for other numpy arrays
        self.configure_numpy_tree(self.tree, data)
        
        # Set file type for numpy arrays
        if len(data.shape) >= 3:
            self.current_file_type = "numpy_array"
            self.total_gamestates = data.shape[0]
            # Store the numpy data for navigation
            self.current_numpy_data = data
            # Update gamestate counter
            self.gamestate_var.set(str(self.current_gamestate))
            self.total_gamestates_label.config(text=f"of {self.total_gamestates}")

            # Enable/disable the secondary navigator
            if len(data.shape) == 4:
                # Match the print viewer: navigator controls TIMESTEP, not 'action set'
                self.current_file_type = "numpy_array_4d"
                view = self._ensure_4d_order_batch_timestep_action_feature(data)  # (B, T, A, F)
                self.total_action_slices = view.shape[1]  # number of timesteps
                self.current_action_slice = max(0, min(self.current_action_slice, self.total_action_slices - 1))
                self.action_slice_var.set(str(self.current_action_slice))
                self.total_action_slices_label.config(text=f"of {self.total_action_slices}")
                for w in (self.slice_prev_btn, self.slice_entry, self.slice_next_btn, self.total_action_slices_label):
                    w.state(["!disabled"])
                self._log("display_numpy: 4D mode enabled",
                          batches=data.shape[0], timesteps=view.shape[1], actions=view.shape[2], features=view.shape[3])
            else:
                self.total_action_slices = 0
                for w in (self.slice_prev_btn, self.slice_entry, self.slice_next_btn, self.total_action_slices_label):
                    w.state(["disabled"])
                self._log("display_numpy: 3D mode enabled",
                          batches=data.shape[0], timesteps=data.shape[1], features=data.shape[2])
        
        # For 3D+ arrays, we need to show initial data
        if len(data.shape) >= 3:
            # Show first slice for 3D arrays
            if len(data.shape) == 3:
                self._log("display_numpy: initial render 3D", batch=0)
                # Create a mock tab frame to use the update function
                mock_tab = type('MockTab', (), {
                    '_tree': self.tree,
                    '_data': data,
                    '_data_type': 'numpy_array'
                })()
                try:
                    self.update_3d_slice_for_tab(mock_tab, 0)
                except Exception:
                    self._log("display_numpy: update_3d_slice_for_tab crashed (3D)", level="error")
                    traceback.print_exc()
            # For 4D arrays, show first slice of first batch
            elif len(data.shape) == 4:
                self._log("display_numpy: initial render 4D", batch=0, timestep=0)
                try:
                    # Initial render for 4D arrays: batch=0, timestep=current_action_slice (0)
                    self._update_numpy_4d_view(0, 0)
                except Exception:
                    self._log("display_numpy: _update_numpy_4d_view crashed", level="error")
                    traceback.print_exc()
    
    def display_json(self, data):
        self.tree.delete(*self.tree.get_children())
        
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Check if this is action data format
                if "mouse_movements" in data[0] and "clicks" in data[0]:
                    # Store the action data for 3D navigation
                    self.action_data = data
                    self.total_gamestates = len(data)
                    self.current_file_type = "action_data"
                    
                    # Try to preserve the current gamestate if possible
                    if self.current_gamestate >= self.total_gamestates:
                        self.current_gamestate = 0
                    
                    self.gamestate_var.set(str(self.current_gamestate))
                    self.total_gamestates_label.config(text=f"of {self.total_gamestates}")
                    
                    # Display the current gamestate
                    self.display_current_gamestate()
                    return
                else:
                    # Regular list of dictionaries
                    keys = list(data[0].keys())
                    self.tree.configure(columns=keys, displaycolumns=keys)
                    self.tree.column("#0", width=100)
                    self.tree.heading("#0", text="Index")
                    
                    for key in keys:
                        self.tree.column(key, width=120)
                        self.tree.heading(key, text=key)
                    
                    for i, item in enumerate(data[:100]):  # Limit to first 100 items
                        values = [str(item.get(key, "")) for key in keys]
                        self.tree.insert("", tk.END, text=f"Row {i}", values=values)
            elif data and isinstance(data[0], list):
                # Check if this is action tensor format (list of lists)
                if len(data[0]) > 0 and isinstance(data[0][0], (int, float)):
                    # Store the action tensor data for 3D navigation
                    self.action_data = data
                    self.total_gamestates = len(data)
                    self.current_file_type = "action_data"
                    
                    # Try to preserve the current gamestate if possible
                    if self.current_gamestate >= self.total_gamestates:
                        self.current_gamestate = 0
                    
                    self.gamestate_var.set(str(self.current_gamestate))
                    self.total_gamestates_label.config(text=f"of {self.total_gamestates}")
                    
                    # Display the current gamestate
                    self.display_current_gamestate()
                    return
                else:
                    # Regular list of lists
                    self.tree.configure(columns=("index", "value"),
                                        displaycolumns=("index", "value"))
                    self.tree.column("#0", width=100)
                    self.tree.column("index", width=100)
                    self.tree.column("value", width=400)
                    self.tree.heading("#0", text="Row")
                    self.tree.heading("index", text="Index")
                    self.tree.heading("value", text="Value")
                    
                    for i, val in enumerate(data[:100]):
                        self.tree.insert("", tk.END, text=f"Row {i}", values=(i, str(val)))
            else:
                # Simple list
                self.tree.configure(columns=("index", "value"),
                                    displaycolumns=("index", "value"))
                self.tree.column("#0", width=0, stretch=tk.NO)
                self.tree.column("index", width=100)
                self.tree.column("value", width=400)
                self.tree.heading("index", text="Index")
                self.tree.heading("value", text="Value")
                
                for i, val in enumerate(data[:100]):
                    self.tree.insert("", tk.END, values=(i, str(val)))
        
        elif isinstance(data, dict):
            self.tree.configure(columns=("key", "value"),
                                displaycolumns=("key", "value"))
            self.tree.column("#0", width=0, stretch=tk.NO)
            self.tree.column("key", width=200)
            self.tree.column("value", width=400)
            self.tree.heading("key", text="Key")
            self.tree.heading("value", text="Value")
            
            for key, value in data.items():
                self.tree.insert("", tk.END, values=(key, str(value)))
    
    def display_current_gamestate(self):
        """Display the current gamestate's data"""
        if not self.action_data:
            return
            
        # Clear the table completely first
        self.tree.delete(*self.tree.get_children())
        
        gamestate = self.action_data[self.current_gamestate]
        
        # Check if this is action data format (with timestamps) or action tensor format
        if isinstance(gamestate, dict) and "mouse_movements" in gamestate:
            # Action data format - display timestamp-based view
            self._display_action_data_gamestate(gamestate)
        elif isinstance(gamestate, list):
            # Action tensor format - display tensor values
            self._display_action_tensor_gamestate(gamestate)
        
        # Force update the display
        self.root.update_idletasks()
    
    def _display_action_data_gamestate(self, gamestate):
        """Display action data gamestate with timestamps"""
        # Completely reset the tree structure
        self.tree.delete(*self.tree.get_children())
        # Configure columns directly (avoid empty-tuple reset which can error)
        cols = ("timestamp", "mouse_movements", "clicks", "key_presses", "key_releases", "scrolls")
        self.tree.configure(columns=cols, displaycolumns=cols)
        
        # Set up columns for timestamp view
        self.tree.column("#0", width=80)
        self.tree.heading("#0", text="Row")
        
        for col in ["timestamp", "mouse_movements", "clicks", "key_presses", "key_releases", "scrolls"]:
            self.tree.column(col, width=120)
            self.tree.heading(col, text=col)
        
        # Collect all actions with their timestamps
        all_actions = []
        for action_type in ["mouse_movements", "clicks", "key_presses", "key_releases", "scrolls"]:
            for action in gamestate.get(action_type, []):
                if "timestamp" in action:
                    all_actions.append((action["timestamp"], action_type, action))
        
        # Sort by timestamp
        all_actions.sort(key=lambda x: x[0])
        
        # Update status to show current gamestate
        self.status_var.set(f"Showing gamestate {self.current_gamestate} of {self.total_gamestates} - {len(all_actions)} actions")
        
        # Display each action on its own row
        for i, (timestamp, action_type, action) in enumerate(all_actions):
            # Initialize all columns to empty
            mouse_str = ""
            clicks_str = ""
            key_presses_str = ""
            key_releases_str = ""
            scrolls_str = ""
            
            # Fill in the appropriate column based on action type
            if action_type == "mouse_movements":
                mouse_str = f"({action.get('x', 0)}, {action.get('y', 0)})"
            elif action_type == "clicks":
                clicks_str = f"{action.get('button', '')}"
            elif action_type == "key_presses":
                key_presses_str = f"{action.get('key', '')}"
            elif action_type == "key_releases":
                key_releases_str = f"{action.get('key', '')}"
            elif action_type == "scrolls":
                scrolls_str = f"({action.get('dx', 0)}, {action.get('dy', 0)})"
            
            values = [
                str(timestamp),
                mouse_str,
                clicks_str,
                key_presses_str,
                key_releases_str,
                scrolls_str
            ]
            self.tree.insert("", tk.END, text=f"Row {i}", values=values)
    
    def _display_action_tensor_gamestate(self, gamestate):
        """Display action tensor gamestate with actions as rows and features as columns"""
        # Completely reset the tree structure
        self.tree.delete(*self.tree.get_children())
        
        # Get action count from index 0
        action_count = gamestate[0] if gamestate else 0
        
        # Set up columns: action number + 8 features with meaningful names
        columns = ["action", "timestamp", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy"]
        
        # Configure columns directly (avoid empty-tuple reset)
        self.tree.configure(columns=columns, displaycolumns=columns)
        
        self.tree.column("#0", width=80)
        self.tree.heading("#0", text="Row")
        
        for col in columns:
            self.tree.column(col, width=100)
            self.tree.heading(col, text=col)
        
        # Update status to show current gamestate
        self.status_var.set(f"Showing gamestate {self.current_gamestate} of {self.total_gamestates} - {action_count} actions")
        
        # Display action count row
        action_count_values = ["Action Count"] + [""] * 8
        self.tree.insert("", tk.END, text="Row 0", values=action_count_values)
        
        # Display each action as a row with its 8 features as columns
        for action_idx in range(action_count):
            start_idx = 1 + (action_idx * 8)
            end_idx = start_idx + 8
            
            # Get the 8 features for this action
            features = gamestate[start_idx:end_idx] if end_idx <= len(gamestate) else gamestate[start_idx:]
            
            # Pad with empty strings if we don't have all 8 features
            while len(features) < 8:
                features.append("")
            
            # Convert codes to human-readable values or keep raw values based on toggle
            if self.show_mapped_values:
                timestamp = features[0] if len(features) > 0 else ""
                action_type = self._convert_action_type_code(features[1]) if len(features) > 1 else ""
                x = features[2] if len(features) > 2 else ""
                y = features[3] if len(features) > 3 else ""
                button = self._convert_button_code(features[4]) if len(features) > 4 else ""
                key = self._convert_key_code(features[5]) if len(features) > 5 else ""
                scroll_dx = features[6] if len(features) > 6 else ""
                scroll_dy = features[7] if len(features) > 7 else ""
            else:
                # Show raw numeric values
                timestamp = features[0] if len(features) > 0 else ""
                action_type = features[1] if len(features) > 1 else ""
                x = features[2] if len(features) > 2 else ""
                y = features[3] if len(features) > 3 else ""
                button = features[4] if len(features) > 4 else ""
                key = features[5] if len(features) > 5 else ""
                scroll_dx = features[6] if len(features) > 6 else ""
                scroll_dy = features[7] if len(features) > 7 else ""
            
            # Create row values: action number + 8 features
            row_values = [f"Action {action_idx + 1}", timestamp, action_type, x, y, button, key, scroll_dx, scroll_dy]
            
            self.tree.insert("", tk.END, text=f"Row {action_idx + 1}", values=row_values)
    
    def _convert_action_type_code(self, code):
        """Convert action type code to string"""
        action_types = {
            0: 'move',
            1: 'click', 
            2: 'key_press',
            3: 'key_release',
            4: 'scroll'
        }
        return action_types.get(code, f'unknown({code})')
    
    def _convert_button_code(self, code):
        """Convert button code to string"""
        button_types = {
            0: '',
            1: 'left',
            2: 'right',
            3: 'middle'
        }
        return button_types.get(code, '')
    
    def _convert_key_code(self, code):
        """Convert key code to string using the same mapping as the pipeline"""
        if code == 0:
            return ''
        
        # Import the key mapper used by the pipeline
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.key_mapper import KeyboardKeyMapper
            
            # Create reverse mapping from the pipeline's key mapper
            reverse_mapping = {v: k for k, v in KeyboardKeyMapper.KEY_MAPPING.items()}
            
            # Convert float to int for lookup
            code_int = int(code) if isinstance(code, float) else code
            
            # Look up the key name
            if code_int in reverse_mapping:
                return reverse_mapping[code_int]
            else:
                return str(code)
        except ImportError:
            # Fallback if key_mapper not available
            return str(code)
    
    def display_csv(self, data):
        self.tree.delete(*self.tree.get_children())
        
        columns = data.columns.tolist()
        self.tree.configure(columns=columns, displaycolumns=columns)
        self.tree.column("#0", width=80)
        self.tree.heading("#0", text="Row")
        
        for col in columns:
            self.tree.column(col, width=120)
            self.tree.heading(col, text=col)
        
        for i in range(min(100, len(data))):  # Limit to first 100 rows
            row = data.iloc[i]
            values = [str(row[col]) for col in columns]
            self.tree.insert("", tk.END, text=f"Row {i}", values=values)
    
    def configure_numpy_tree(self, tree, data):
        """Configure tree for numpy array data"""
        self._log("configure_numpy_tree: enter",
                  ndim=int(getattr(data, 'ndim', -1)),
                  shape=tuple(getattr(data, 'shape', ())))
        # NOTE: avoid empty () reset; configure the target columns directly below
        
        if len(data.shape) == 1:
            tree.configure(columns=("index", "value"),
                           displaycolumns=("index", "value"))
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("index", anchor=tk.CENTER, width=100)
            tree.column("value", anchor=tk.CENTER, width=150)
            tree.heading("#0", text="")
            tree.heading("index", text="Index")
            tree.heading("value", text="Value")
            
            for i, val in enumerate(data):
                try:
                    formatted_val = f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val)
                except:
                    formatted_val = str(val)
                tree.insert("", tk.END, values=(i, formatted_val))
                
        elif len(data.shape) == 2:
            cols = tuple([f"col_{i}" for i in range(data.shape[1])])
            tree.configure(columns=cols, displaycolumns=cols)
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Row")
            
            for i, name in enumerate(cols):
                tree.column(name, anchor=tk.CENTER, width=120)  # Consistent width
                tree.heading(name, text=f"Col {i}")
            
            for i in range(data.shape[0]):
                try:
                    values = [f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val) for val in data[i]]
                except:
                    values = [str(val) for val in data[i]]
                tree.insert("", tk.END, text=f"Row {i}", values=values)
                
        elif len(data.shape) == 3:
            self._log("configure_numpy_tree: 3D")
            # 3D arrays: rows=timesteps, columns=features
            n_features = data.shape[2]
            # if 8 features, show nice names used in action arrays
            if n_features == 8:
                cols = ("count", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
            else:
                cols = tuple(f"F{i}" for i in range(n_features))
            self._set_tree_columns(tree, cols)
            tree.column("#0", width=100, stretch=tk.NO, anchor="w")
            tree.heading("#0", text="Timestep")
            for name in cols:
                tree.column(name, anchor=tk.CENTER, width=140, minwidth=120, stretch=True)
                tree.heading(name, text=name)
        else:
            self._log("configure_numpy_tree: 4D+")
            # For 4D+ arrays, show slice navigation for first two dimensions
            n_features = data.shape[-1]
            if n_features == 8:
                cols = ("count", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
            else:
                cols = tuple(f"F{i}" for i in range(n_features))
            self._set_tree_columns(tree, cols)
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Timestep")
            for name in cols:
                tree.column(name, anchor=tk.CENTER, width=140, minwidth=120, stretch=True)
                tree.heading(name, text=name)
    
    def update_3d_slice_for_tab(self, tab_frame, slice_idx):
        """Update a specific tab's 3D slice display"""
        self._log("update_3d_slice_for_tab: enter",
                  slice_idx=slice_idx)
        if not hasattr(tab_frame, '_tree') or not hasattr(tab_frame, '_data'):
            return
        
        tree = tab_frame._tree
        data = tab_frame._data
        data_type = tab_frame._data_type
        self._log("update_3d_slice_for_tab: data",
                  data_type=data_type,
                  ndim=int(getattr(data,'ndim',-1)),
                  shape=tuple(getattr(data,'shape',())))
        
        if data_type == "numpy_array":
            if not (0 <= slice_idx < data.shape[0]):
                self._log("update_3d_slice_for_tab: slice out of range")
                return
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            # Action-type colors used in both 3-D (targets-as-features) and 4-D views
            action_colors = {
                0: "lightblue",    # move
                1: "lightgreen",   # click
                2: "lightyellow",  # key_press/key_release
                3: "lightcoral"    # scroll
            }
            # Add ALL data for this slice
            if data.ndim == 3:
                self._log("update_3d_slice_for_tab: rendering 3D rows",
                          timesteps=data.shape[1], features=data.shape[2])
                # 3D array: data[slice_idx, i, j] - show ALL timesteps for this batch
                for i in range(data.shape[1]):  # timesteps
                    try:
                        # Format like print_numpy_array.py: ints as ints, floats to 6dp
                        def _fmt(v):
                            if isinstance(v, (np.integer, int)):
                                return str(int(v))
                            if isinstance(v, (np.floating, float)):
                                return f"{float(v):.6f}"
                            return str(v)
                        values = [_fmt(val) for val in data[slice_idx, i]]
                        item = tree.insert("", tk.END, text=f"Timestep {i}", values=values)
                        # Color if it looks like an action row (8 features)
                        if len(values) == 8:
                            try:
                                action_count = int(float(values[0]))
                                action_type = int(float(values[1]))
                                if action_count > 0 and action_type in action_colors:
                                    tree.tag_configure(f"action_{action_type}", background=action_colors[action_type])
                                    tree.item(item, tags=(f"action_{action_type}",))
                            except (ValueError, IndexError):
                                pass
                    except Exception:
                        self._log("update_3d_slice_for_tab: row render error (3D)", level="error", i=i)
                        traceback.print_exc()
                        values = [str(val) for val in data[slice_idx, i]]
                        tree.insert("", tk.END, text=f"Timestep {i}", values=values)
            elif data.ndim == 4:
                self._log("update_3d_slice_for_tab: rendering 4D rows",
                          A=data.shape[1], T=data.shape[2], F=data.shape[3])
                # 4D array: choose current action slice, then list all timesteps
                a = max(0, min(getattr(self, "current_action_slice", 0), data.shape[1] - 1))
                for i in range(data.shape[2]):  # timesteps
                    try:
                        def _fmt(v):
                            if isinstance(v, (np.integer, int)):
                                return str(int(v))
                            if isinstance(v, (np.floating, float)):
                                return f"{float(v):.6f}"
                            return str(v)
                        values = [_fmt(v) for v in data[slice_idx, a, i, :]]
                        item = tree.insert("", tk.END, text=f"Timestep {i}", values=values)
                        if len(values) == 8:
                            try:
                                action_count = int(float(values[0]))
                                action_type = int(float(values[1]))
                                if action_count > 0 and action_type in action_colors:
                                    tree.tag_configure(f"action_{action_type}", background=action_colors[action_type])
                                    tree.item(item, tags=(f"action_{action_type}",))
                            except (ValueError, IndexError):
                                pass
                    except Exception:
                        self._log("update_3d_slice_for_tab: row render error (4D)", level="error", i=i, a=a)
                        traceback.print_exc()
                        values = [str(v) for v in data[slice_idx, a, i, :]]
                        tree.insert("", tk.END, text=f"Timestep {i}", values=values)

        # auto-size after rows are inserted
        try:
            self._autosize_columns(tree)
        except Exception as e:
            # Don't let sizing kill the GUI; surface the problem in status bar
            self._log("update_3d_slice_for_tab: autosize error", level="error", error=str(e))
            self._safe_status(f"Autosize error: {e}")
    
    def display_current_gamestate_features(self):
        """Display the current gamestate's state features"""
        if not hasattr(self, 'state_features_data') or self.current_gamestate >= len(self.state_features_data):
            return
        
        # Get the features for the current gamestate
        features = self.state_features_data[self.current_gamestate]
        
        # Load feature mappings if not already loaded
        if not hasattr(self, 'feature_mappings'):
            try:
                feature_mappings_path = os.path.join(self.data_dir, 'features', 'feature_mappings.json')
                with open(feature_mappings_path, 'r') as f:
                    self.feature_mappings = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load feature mappings: {e}")
                self.feature_mappings = None
        
        # Clear the table and set up columns for state features
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(columns=("feature_index", "feature_value"),
                            displaycolumns=("feature_index", "feature_value"))
        self.tree.column("#0", width=100, stretch=tk.NO)
        self.tree.column("feature_index", anchor=tk.CENTER, width=100)
        self.tree.column("feature_value", anchor=tk.CENTER, width=200)
        self.tree.heading("#0", text="Row")
        self.tree.heading("feature_index", text="Index")
        self.tree.heading("feature_value", text="Value")
        
        # Display each feature as a row
        for i, feature_value in enumerate(features):
            try:
                formatted_value = f"{feature_value:.6f}" if isinstance(feature_value, (int, float, np.number)) else str(feature_value)
            except:
                formatted_value = str(feature_value)
            
            # Use actual feature name if available, otherwise fall back to "Feature {i}"
            if self.feature_mappings and i < len(self.feature_mappings):
                feature_name = self.feature_mappings[i].get('feature_name', f'Feature {i}')
            else:
                feature_name = f'Feature {i}'
            
            self.tree.insert("", tk.END, text=feature_name, values=(i, formatted_value))
        

        
        # Update status to show current gamestate
        self.status_var.set(f"Showing gamestate {self.current_gamestate} of {self.total_gamestates} - {len(features)} features")
        
        # Restore scroll position if we have a saved one
        if hasattr(self, 'state_features_scroll_position'):
            self.tree.yview_moveto(self.state_features_scroll_position)

    # ----- 4D slice navigation handlers -----
    def prev_action_slice(self):
        self._log("prev_action_slice", current=self.current_action_slice, total=self.total_action_slices)
        if self.total_action_slices <= 0:
            return
        if self.current_action_slice > 0:
            self.current_action_slice -= 1
            self.action_slice_var.set(str(self.current_action_slice))
            self._navigate_gamestate()

    def next_action_slice(self):
        self._log("next_action_slice", current=self.current_action_slice, total=self.total_action_slices)
        if self.total_action_slices <= 0:
            return
        if self.current_action_slice < self.total_action_slices - 1:
            self.current_action_slice += 1
            self.action_slice_var.set(str(self.current_action_slice))
            self._navigate_gamestate()

    def on_action_slice_entry_change(self, *args):
        self._log("on_action_slice_entry_change", value=self.action_slice_var.get(), total=self.total_action_slices)
        if self.total_action_slices <= 0:
            return
        try:
            new_idx = int(self.action_slice_var.get())
            if 0 <= new_idx < self.total_action_slices:
                self.current_action_slice = new_idx
                self._navigate_gamestate()
        except ValueError:
            pass

def main():
    try:
        root = tk.Tk()
        app = SimpleDataExplorer(root)
        root.mainloop()
    except Exception:
        print("FATAL in main():", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
