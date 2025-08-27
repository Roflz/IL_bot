import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import os, sys, time, traceback, logging
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple
# Optional plotting (for the Actions Graph tab)
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

class SimpleDataExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Data Explorer")
        self.root.geometry("1200x800")
        # Tabs (folder -> listbox) MUST be defined before setup_ui() uses them
        self.folders = [
            "01_raw_data",
            "02_trimmed_data",
            "03_normalized_data",
            "04_sequences",
            "05_mappings",
            "06_final_training_data",
        ]
        self.tab_lists = {}
        # summary + mapping state
        self.sequence_length = 10
        self.current_folder = None
        self.current_file_path = None
        self.current_numpy_data = None
        self.current_data = None     # current loaded data for summary updates
        self._full_metadata = None  # lazy-loaded features/gamestates_metadata.json
        self._slice_info = None     # {'start_idx_raw','end_idx_raw','count'}
        self._slice_cache = {}      # folder -> slice_info (memoize)
        self._counts_cache = {}     # folder -> (xs, ys) action counts memo
        self.summary_vars = {
            "dataset": tk.StringVar(value="-"),
            "shape": tk.StringVar(value="-"),
            "count": tk.StringVar(value="-"),      # underlying gamestates N
            "sequences": tk.StringVar(value="-"),  # B
            "range": tk.StringVar(value="-"),
            "current": tk.StringVar(value="-"),
            "timestamp": tk.StringVar(value="-"),
            "source_range": tk.StringVar(value="-"),
            "current_slice": tk.StringVar(value="-"),
            "sequence_range": tk.StringVar(value="-"),  # Sequence range and slice info
            "first_ts": tk.StringVar(value="-"),
            "last_ts": tk.StringVar(value="-")
        }
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
        
        # --- Session roots + default selection ---
        self.base_data_root = "D:/repos/bot_runelite_IL/data"
        self.sessions_root = Path(self.base_data_root) / "recording_sessions"
        # discover sessions (sorted ascending; newest is last)
        try:
            self.sessions = sorted([p.name for p in self.sessions_root.iterdir() if p.is_dir()])
        except Exception:
            self.sessions = []
        # default to latest session if present, else fall back to base root
        if self.sessions:
            self.session_name = self.sessions[-1]
            self.data_dir = str(self.sessions_root / self.session_name)
        else:
            self.session_name = ""
            self.data_dir = str(self.base_data_root)
        
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
        # Simple controls (top)
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Session chooser: combobox over all recording session folders
        ttk.Label(control_frame, text="Session:").pack(side=tk.LEFT)
        self.session_var = tk.StringVar(value=getattr(self, "session_name", ""))
        self.session_cb = ttk.Combobox(
            control_frame,
            textvariable=self.session_var,
            values=getattr(self, "sessions", []),
            state="readonly",
            width=24
        )
        self.session_cb.pack(side=tk.LEFT, padx=5)
        self.session_cb.bind("<<ComboboxSelected>>", self.on_session_change)

        # Read-only display of the resolved session path (for visibility)
        ttk.Label(control_frame, text="Folder:").pack(side=tk.LEFT, padx=(10, 2))
        self.dir_var = tk.StringVar(value=self.data_dir)
        dir_entry = ttk.Entry(control_frame, textvariable=self.dir_var, width=50, state="readonly")
        dir_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Refresh Sessions", command=self.refresh_sessions).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh Files", command=self.load_files).pack(side=tk.LEFT, padx=5)
        
        # ---------- TOP: file tabs (left) + summary pane (right) ----------
        top_paned = ttk.Panedwindow(self.root, orient="horizontal")
        top_paned.pack(fill=tk.X, padx=10, pady=(0,5))

        # Left: tabs + file list
        files_panel = ttk.Frame(top_paned)
        top_paned.add(files_panel, weight=3)
        self.notebook = ttk.Notebook(files_panel)
        self.notebook.pack(fill=tk.X, expand=True)
        for folder in self.folders:
            tab = ttk.Frame(self.notebook)
            label = folder.split('_', 1)[1] if '_' in folder else folder
            self.notebook.add(tab, text=label)
            lb = tk.Listbox(tab, height=8, exportselection=False)
            lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            sb = ttk.Scrollbar(tab, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb.bind('<<ListboxSelect>>', lambda e, f=folder: self.on_tab_file_select(f, e))
            self.tab_lists[folder] = lb

        # Right: info notebook with "Summary" + "Actions Graph"
        info_panel = ttk.Frame(top_paned)
        top_paned.add(info_panel, weight=2)
        self.info_nb = ttk.Notebook(info_panel)
        self.info_nb.pack(fill=tk.BOTH, expand=True)
        # Summary tab
        self.summary_tab = ttk.Frame(self.info_nb, padding=(10,10,10,10))
        self.info_nb.add(self.summary_tab, text="Summary")
        ttk.Label(self.summary_tab, text="Summary", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0,8))
        def row(label, var, r):
            ttk.Label(self.summary_tab, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Label(self.summary_tab, textvariable=var, foreground="#333", wraplength=360, justify="left").grid(row=r, column=1, sticky="w", pady=2)
        row("Dataset:", self.summary_vars["dataset"], 1)
        row("Shape:", self.summary_vars["shape"], 2)
        row("# Gamestates:", self.summary_vars["count"], 3)     # N
        row("Sequences:", self.summary_vars["sequences"], 4)    # B
        row("Source range (raw):", self.summary_vars["range"], 5)
        row("Source range:", self.summary_vars["source_range"], 6)
        ttk.Separator(self.summary_tab, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8,8))
        row("Current slice:", self.summary_vars["current"], 8)
        row("Current slice (source):", self.summary_vars["current_slice"], 9)
        row("Sequence range:", self.summary_vars["sequence_range"], 10)
        row("First timestamp:", self.summary_vars["first_ts"], 11)
        row("Last timestamp:", self.summary_vars["last_ts"], 12)
        row("Unix timestamp:", self.summary_vars["timestamp"], 13)
        for c in (0,1):
            self.summary_tab.grid_columnconfigure(c, weight=1)
        # Actions Graph tab
        self.graph_tab = ttk.Frame(self.info_nb)
        self.info_nb.add(self.graph_tab, text="Actions Graph")
        if _HAS_MPL:
            self._fig = Figure(figsize=(4.2, 2.6), dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._ax.set_xlabel("Gamestate")
            self._ax.set_ylabel("Action count")
            self._ax.grid(True, alpha=0.25)
            self._graph_cursor = None
            self._canvas = FigureCanvasTkAgg(self._fig, master=self.graph_tab)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.graph_tab, text="Matplotlib not available — install it to see the Actions Graph.").pack(padx=12, pady=12, anchor="w")
        
        # ---------- MIDDLE: navigation (3D/4D) ----------
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
        
        # ---------- BOTTOM: viewer (full width) ----------
        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # (summary is now at the top; viewer uses the whole bottom width)
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
        """Columns like the print viewer: 8 feature headers; #0 label 'Action'.
        First feature is now 'timestamp' (no action count)."""
        cols = ("timestamp", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
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
            # Optional row coloring keyed only by action_type now
            try:
                action_type = int(float(values[1]))
                if action_type in action_colors:
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
        
        # Populate tabs
        total = 0
        first_loaded = False
        for folder, lb in self.tab_lists.items():
            lb.delete(0, tk.END)
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith(('.json', '.npy', '.csv'))]
                files.sort()
                for f in files:
                    lb.insert(tk.END, f)
                total += len(files)
                # Auto-load the first file in the first non-empty tab
                if files and not first_loaded:
                    lb.selection_set(0)
                    self.on_tab_file_select(folder)
                    first_loaded = True
        self.status_var.set(f"Found {total} files across {len(self.tab_lists)} tabs")
    
    def on_tab_file_select(self, folder, event=None):
        """Load the selected file from a given folder tab"""
        lb = self.tab_lists.get(folder)
        if not lb:
            return
        sel = lb.curselection()
        if not sel:
            return
        filename = lb.get(sel[0])
        file_path = os.path.join(self.data_dir, folder, filename)
        self._log("on_tab_file_select", folder=folder, filename=filename, path=file_path)
        self.current_folder = folder
        self.current_file_path = file_path
        try:
            self.load_file(file_path)
            # update summary (dataset + shape + range)
            self._update_summary_on_file_load(file_path)
        except Exception:
            self._log("on_tab_file_select: load_file crashed", level="error", file=file_path)
            traceback.print_exc()
            self._safe_status(f"Error while loading {os.path.basename(file_path)}")
            try:
                messagebox.showerror("Explorer Error",
                                     f"Failed to load:\n{file_path}\n\n{traceback.format_exc()}")
            except Exception:
                pass
    
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
        elif self.current_file_type in ("action_data", "action_tensors"):
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
        # Update summary slice + timestamp
        try:
            self._update_summary_current_slice()
            # Always update the main summary to refresh sequence range info
            self._update_summary()
            self._update_actions_graph()  # move the cursor to new position
        except Exception:
            self._log("_navigate_gamestate: summary update failed", level="error")
    
    def on_mapped_toggle(self):
        """Handle toggle between mapped and raw display"""
        self._log("on_mapped_toggle", mapped=self.mapped_var.get())
        self.show_mapped_values = self.mapped_var.get()
        if self.action_data:
            self.display_current_gamestate()
    
    # (combobox-based on_file_selected removed in favor of tab file picker)
    
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
                # prevent stale shapes from persisting
                self.current_numpy_data = None
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                self._log("load_file: csv loaded", rows=len(data), cols=len(data.columns))
                self.display_csv(data)
                self.current_numpy_data = None
            else:
                self._log("load_file: unsupported extension", level="warning")
                pass
            
            self._safe_status(f"Loaded {os.path.basename(file_path)}")
            # Update summary main info after any successful load
            try:
                self._update_summary_on_file_load(file_path)
                # Also update the new summary format for JSON files
                if file_path.endswith('.json'):
                    self._update_summary()
                self._update_actions_graph()  # refresh chart for the selected dataset
            except Exception:
                self._log("load_file: summary update failed", level="error")
            
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
            # 2-D features -> do NOT retain current_numpy_data for summary
            self.current_numpy_data = None
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
        # Update summary main info for numpy
        try:
            self._update_summary_on_file_load(self.current_file_path or "")
        except Exception:
            self._log("display_numpy: summary update failed", level="error")
    
    def _display_json(self, filename: str, data_obj):
        """Route JSON displays with no fallbacks."""
        self.tree.delete(*self.tree.get_children())
        self._clear_columns()
        if self._is_action_tensor_file(filename):
            # Show the currently selected gamestate's flattened tensor as rows of 8
            idx = max(0, min(self.current_gamestate or 0, len(data_obj) - 1))
            self._display_action_tensors(data_obj[idx] if idx < len(data_obj) else [])
        elif filename.endswith("_action_data.json"):
            # Like before: render selected gamestate's actions as rows (derived from dicts)
            self._display_action_json_gamestate(data_obj)
        else:
            # Fallback only for *generic* JSON (not our action formats)
            self._display_generic_json(data_obj)

    def display_json(self, data):
        self.tree.delete(*self.tree.get_children())
        # clear stale ndarray reference so summary derives shape from clicked file
        self.current_numpy_data = None
        
        # Store the current data for summary updates
        self.current_data = data
        
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
            elif (isinstance(data[0], list) if data else True):
                # Treat as action tensors if (a) file is *_action_tensors.json OR
                # (b) first row is a numeric list (legacy heuristic).
                is_tensor_file = self._is_current_file_action_tensors()
                first_is_numeric_list = bool(data and len(data[0]) > 0 and isinstance(data[0][0], (int, float)))
                if is_tensor_file or first_is_numeric_list:
                    self.action_data = data
                    self.total_gamestates = len(data)
                    self.current_file_type = "action_tensors"
                    if self.current_gamestate >= self.total_gamestates:
                        self.current_gamestate = 0
                    self.gamestate_var.set(str(self.current_gamestate))
                    self.total_gamestates_label.config(text=f"of {self.total_gamestates}")
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
        if self.current_file_type == "action_tensors":
            self._display_action_tensor_gamestate(gamestate)
        elif isinstance(gamestate, dict) and "mouse_movements" in gamestate:
            # Action data format - display timestamp-based view
            self._display_action_data_gamestate(gamestate)
        elif isinstance(gamestate, list):
            # Action tensor format - display tensor values
            self._display_action_tensor_gamestate(gamestate)
        
        # Force update the display
        self.root.update_idletasks()
        
        # Update summary if we have current data
        if hasattr(self, 'current_data') and isinstance(self.current_data, list):
            try:
                self._update_summary()
            except Exception:
                self._log("display_current_gamestate: summary update failed", level="error")
    
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
        self.tree.delete(*self.tree.get_children())

        # Define columns and configure the tree properly
        cols = ['action', 'timestamp', 'type', 'x', 'y', 'button', 'key', 'scroll_dx', 'scroll_dy']
        self.tree.configure(columns=cols, displaycolumns=cols)
        
        # Set up column headers and widths
        self.tree.column("#0", width=80, stretch=tk.NO)
        self.tree.heading("#0", text="Action")
        
        for col in cols:
            self.tree.column(col, width=120, anchor=tk.CENTER)
            self.tree.heading(col, text=col)

        if not isinstance(gamestate, list) or len(gamestate) < 8:
            self._safe_status("No actions in this gamestate.")
            return

        # Debug logging
        self._log("_display_action_tensor_gamestate", 
                  gamestate_length=len(gamestate), 
                  current_gamestate=self.current_gamestate)

        start = 1 if len(gamestate) % 8 != 0 else 0
        for i in range(start, len(gamestate), 8):
            row = gamestate[i:i+8]
            action_idx = (i - start) // 8
            values = [str(action_idx)] + [str(x) for x in row]
            self.tree.insert("", "end", text=f"Action {action_idx}", values=values)

        count = (len(gamestate) - start) // 8
        self._safe_status(f"Gamestate {self.current_gamestate}: {count} actions")
    
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
                cols = ("timestamp", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
            else:
                cols = tuple(f"F{i}" for i in range(n_features))
            
            # Configure columns properly for large feature counts
            tree.configure(columns=cols, displaycolumns=cols)
            tree.column("#0", width=150, stretch=tk.NO, anchor="w")
            tree.heading("#0", text="Timestep")
            
            # Set reasonable column widths for large feature counts
            if n_features > 50:
                # For large feature counts, use smaller widths and enable horizontal scrolling
                col_width = 80
                for name in cols:
                    tree.column(name, anchor=tk.CENTER, width=col_width, minwidth=60, stretch=False)
                    tree.heading(name, text=name)
            else:
                # For smaller feature counts, use normal widths
                for name in cols:
                    tree.column(name, anchor=tk.CENTER, width=140, minwidth=120, stretch=True)
                    tree.heading(name, text=name)
        else:
            self._log("configure_numpy_tree: 4D+")
            # For 4D+ arrays, show slice navigation for first two dimensions
            n_features = data.shape[-1]
            if n_features == 8:
                cols = ("timestamp", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
            else:
                cols = tuple(f"F{i}" for i in range(n_features))
            
            # Configure columns properly for large feature counts
            tree.configure(columns=cols, displaycolumns=cols)
            tree.column("#0", width=150, stretch=tk.NO)
            tree.heading("#0", text="Timestep")
            
            # Set reasonable column widths for large feature counts
            if n_features > 50:
                # For large feature counts, use smaller widths and enable horizontal scrolling
                col_width = 80
                for name in cols:
                    tree.column(name, anchor=tk.CENTER, width=col_width, minwidth=60, stretch=False)
                    tree.heading(name, text=name)
            else:
                # For smaller feature counts, use normal widths
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
            
            # Add sequence context header
            n_timesteps = data.shape[1]
            n_features = data.shape[2]
            
            # Determine sequence type based on features
            if n_features == 8:
                sequence_type = "Action"
            elif n_features == 128:
                sequence_type = "Gamestate"
            else:
                sequence_type = "Feature"
            
            # Create context header
            context_text = f"{sequence_type} Sequence (0-{n_timesteps-1}) | Features: {n_features}"
            header_item = tree.insert("", "end", text=context_text, values=[""] * n_features)
            
            # Style the header
            tree.tag_configure("header", background="lightgray", font=("Arial", 9, "bold"))
            tree.item(header_item, tags=("header",))
            
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
                        
                        # Create descriptive timestep label
                        if i == 0:
                            timestep_label = f"Timestep {i} (start)"
                        elif i == data.shape[1] - 1:
                            timestep_label = f"Timestep {i} (current)"
                        else:
                            timestep_label = f"Timestep {i}"
                        
                        item = tree.insert("", tk.END, text=timestep_label, values=values)
                        # Color 8-feature rows by action_type (no count anymore)
                        if len(values) == 8:
                            try:
                                action_type = int(float(values[1]))
                                if action_type in action_colors:
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
                        
                        # Create descriptive timestep label with action slice info
                        if i == 0:
                            timestep_label = f"Timestep {i} (start) | Action {a}"
                        elif i == data.shape[2] - 1:
                            timestep_label = f"Timestep {i} (current) | Action {a}"
                        else:
                            timestep_label = f"Timestep {i} | Action {a}"
                        
                        item = tree.insert("", tk.END, text=timestep_label, values=values)
                        if len(values) == 8:
                            try:
                                action_type = int(float(values[1]))
                                if action_type in action_colors:
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

    # -------------------- Summary helpers --------------------
    def _load_full_metadata(self):
        """Load session-local metadata with absolute timestamps."""
        if self._full_metadata is not None:
            return self._full_metadata
        # Session-local only
        candidates = [
            os.path.join(self.data_dir, "05_mappings", "gamestates_metadata.json"),
            os.path.join(self.data_dir, "mappings", "gamestates_metadata.json"),  # legacy session name
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    self._full_metadata = json.load(open(p, "r"))
                    self._log("_load_full_metadata", count=len(self._full_metadata))
                    return self._full_metadata
                except Exception:
                    pass
        self._full_metadata = []
        return self._full_metadata

    def _load_slice_info(self, folder_path: Path) -> Optional[dict]:
        """Load slice info for exactly this folder. No fallbacks."""
        p = folder_path / "slice_info.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _infer_slice_info(self, folder: str) -> dict:
        """
        Best-effort inference of the source raw gamestate range for the selected folder.
        Returns dict with: {'start_idx_raw': int|None, 'end_idx_raw': int|None, 'count': int}
        """
        # Memoized?
        if folder in self._slice_cache:
            return self._slice_cache[folder]

        full_meta = self._load_full_metadata() or []
        full_ts   = [m.get("absolute_timestamp") for m in full_meta] if full_meta else []

        def _count_from_npy(path):
            try:
                return int(np.load(path, mmap_mode="r").shape[0])
            except Exception:
                return None

        def _range_from_json(json_path):
            """Match first/last gamestate timestamps against full metadata."""
            if not os.path.exists(json_path):
                return None
            try:
                arr = json.load(open(json_path, "r"))
                if not arr:
                    return None
                # handle either {"gamestate_timestamp": ...} or {"timestamp": ...}
                def _ts(rec):
                    return rec.get("gamestate_timestamp", rec.get("timestamp"))
                first_ts = _ts(arr[0]); last_ts = _ts(arr[-1])
                if first_ts in full_ts and last_ts in full_ts:
                    s = full_ts.index(first_ts); e = full_ts.index(last_ts)
                    return {"start_idx_raw": s, "end_idx_raw": e, "count": e - s + 1}
            except Exception as e:
                self._log("_range_from_json failed", level="error", path=json_path, error=str(e))
            return None

        def _match_rows_in_raw(sample_rows: np.ndarray, tol: float = 1e-6) -> int|None:
            """Find the starting index in raw state_features that matches the first K rows."""
            try:
                raw_path = os.path.join(self.data_dir, "01_raw_data", "state_features.npy")
                if not os.path.exists(raw_path):
                    return None
                raw = np.load(raw_path, mmap_mode="r")   # (N,128)
                if raw.ndim != 2 or raw.shape[1] != sample_rows.shape[1]:
                    return None
                K = min(3, sample_rows.shape[0])  # compare first 1..3 rows to be robust
                tgt = sample_rows[:K]
                N = raw.shape[0] - K + 1
                for s in range(max(N, 0)):
                    # quick reject using first row
                    if not np.allclose(raw[s], tgt[0], atol=tol, rtol=0):
                        continue
                    if K == 1 or np.allclose(raw[s:s+K], tgt, atol=tol, rtol=0):
                        return s
            except Exception as e:
                self._log("_match_rows_in_raw failed", level="error", error=str(e))
            return None

        def _denormalize_if_needed(norm_rows: np.ndarray) -> np.ndarray:
            """
            Undo the feature normalization we use: only time-like features were scaled /180.
            Uses data/features/feature_mappings.json to find those indices.
            """
            try:
                fmap = os.path.join(self.data_dir, "features", "feature_mappings.json")
                if not os.path.exists(fmap):
                    return norm_rows
                with open(fmap, "r") as f:
                    mappings = json.load(f)
                time_idx = []
                for m in mappings:
                    idx = m.get("feature_index")
                    name = m.get("feature_name", "")
                    dtype = m.get("data_type", "")
                    if idx is None: 
                        continue
                    if (dtype in ("time_ms","duration_ms") or
                        name in ("time_since_interaction","phase_start_time","phase_duration","timestamp")):
                        time_idx.append(idx)
                if not time_idx:
                    return norm_rows
                out = np.array(norm_rows, copy=True)
                out[:, time_idx] = out[:, time_idx] * 180.0
                return out
            except Exception as e:
                self._log("_denormalize_if_needed failed", level="error", error=str(e))
                return norm_rows

        if folder == "01_raw_data":
            n = _count_from_npy(os.path.join(self.data_dir, "01_raw_data", "state_features.npy"))
            if n is None:
                n = len(full_meta) or 0
            info = {"start_idx_raw": 0 if n else None, "end_idx_raw": (n - 1) if n else None, "count": n}
            self._slice_cache[folder] = info
            return info

    def _compute_action_shape(self, action_list: list) -> Tuple[int, int, int]:
        """Return (G, max_A, 8) for a list of per-gamestate action dicts."""
        G = len(action_list)
        def _count(gs):
            return (len(gs.get("mouse_movements", [])) +
                    len(gs.get("clicks", [])) +
                    len(gs.get("key_presses", [])) +
                    len(gs.get("key_releases", [])) +
                    len(gs.get("scrolls", [])))
        max_A = max((_count(gs) for gs in action_list), default=0)
        return (G, max_A, 8)

    def _is_action_tensor_file(self, filename: str) -> bool:
        return filename.endswith("_action_tensors.json")

    def _is_current_file_action_tensors(self) -> bool:
        """True if the selected file is *_action_tensors.json (e.g., raw/trimmed/normalized)."""
        try:
            name = os.path.basename(self.current_file_path or "")
            return name.endswith("_action_tensors.json") or name == "raw_action_tensors.json"
        except Exception:
            return False

    def _display_action_tensors(self, flat_list_for_gamestate: list):
        """Show one gamestate's flattened tensor as rows of 8 values."""
        cols = ("timestamp", "type", "x", "y", "button", "key", "scroll_dx", "scroll_dy")
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, stretch=True, anchor="center")
        self.tree.delete(*self.tree.get_children())
        # Handle older files that accidentally included a leading count
        start = 1 if (len(flat_list_for_gamestate) % 8 != 0) else 0
        row = []
        for i in range(start, len(flat_list_for_gamestate), 8):
            chunk = flat_list_for_gamestate[i:i+8]
            if len(chunk) < 8: break
            self.tree.insert("", "end", values=[f"{v}" for v in chunk])

    def _current_folder_path(self):
        """Get the current folder path."""
        if not self.current_folder:
            return Path(self.data_dir)
        return Path(self.data_dir) / self.current_folder

    def _summary_set(self, key: str, value: str):
        """Set a summary variable if it exists."""
        if key in self.summary_vars:
            self.summary_vars[key].set(value)

    def _clear_columns(self):
        """Clear all columns from the tree."""
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()

    def _display_action_json_gamestate(self, data_obj):
        """Display action JSON gamestate."""
        if not isinstance(data_obj, list) or not data_obj:
            return
        idx = max(0, min(self.current_gamestate or 0, len(data_obj) - 1))
        self._display_action_data_gamestate(data_obj[idx] if idx < len(data_obj) else {})

    def _display_generic_json(self, data_obj):
        """Display generic JSON data."""
        if isinstance(data_obj, list):
            if data_obj and isinstance(data_obj[0], dict):
                keys = list(data_obj[0].keys())
                self.tree.configure(columns=keys, displaycolumns=keys)
                self.tree.column("#0", width=100)
                self.tree.heading("#0", text="Index")
                
                for key in keys:
                    self.tree.column(key, width=120)
                    self.tree.heading(key, text=key)
                
                for i, item in enumerate(data_obj[:100]):  # Limit to first 100 items
                    values = [str(item.get(key, "")) for key in keys]
                    self.tree.insert("", tk.END, text=f"Row {i}", values=values)
            else:
                self.tree.configure(columns=("index", "value"), displaycolumns=("index", "value"))
                self.tree.column("#0", width=100)
                self.tree.column("index", width=100)
                self.tree.column("value", width=400)
                self.tree.heading("#0", text="Row")
                self.tree.heading("index", text="Index")
                self.tree.heading("value", text="Value")
                
                for i, val in enumerate(data_obj[:100]):
                    self.tree.insert("", tk.END, text=f"Row {i}", values=(i, str(val)))
        elif isinstance(data_obj, dict):
            self.tree.configure(columns=("key", "value"), displaycolumns=("key", "value"))
            self.tree.column("#0", width=0, stretch=tk.NO)
            self.tree.column("key", width=200)
            self.tree.column("value", width=400)
            self.tree.heading("key", text="Key")
            self.tree.heading("value", text="Value")
            
            for key, value in data_obj.items():
                self.tree.insert("", tk.END, values=(key, str(value)))

        if folder in ("02_trimmed_data", "03_normalized_data"):
            npy_name = "trimmed_features.npy" if folder == "02_trimmed_data" else "normalized_features.npy"
            n = _count_from_npy(os.path.join(self.data_dir, folder, npy_name)) or 0
            # best effort: align by timestamps if the action json exists
            json_name = "trimmed_raw_action_data.json" if folder == "02_trimmed_data" else "normalized_action_data.json"
            info = _range_from_json(os.path.join(self.data_dir, folder, json_name))
            if info:
                self._slice_cache[folder] = info
                return info
            # Fallback: match features back into raw
            try:
                fpath = os.path.join(self.data_dir, folder, npy_name)
                if os.path.exists(fpath):
                    arr = np.load(fpath, mmap_mode="r")
                    sample = arr[: min(arr.shape[0], 3)]
                    if folder == "03_normalized_data":
                        sample = _denormalize_if_needed(sample)
                    s = _match_rows_in_raw(sample)
                    if s is not None:
                        info = {"start_idx_raw": s, "end_idx_raw": s + n - 1, "count": n}
                        self._slice_cache[folder] = info
                        return info
            except Exception as e:
                self._log("feature match fallback failed", level="error", folder=folder, error=str(e))
            info = {"start_idx_raw": None, "end_idx_raw": None, "count": n}
            self._slice_cache[folder] = info
            return info

        if folder == "06_final_training_data":
            # Reconstruct underlying count from sequences: N = B + L
            seq = os.path.join(self.data_dir, folder, "gamestate_sequences.npy")
            n = None
            if os.path.exists(seq):
                try:
                    B = int(np.load(seq, mmap_mode="r").shape[0])
                    n = B + self.sequence_length
                except Exception:
                    n = None
            # anchor to trimmed range start if we can find it
            trimmed = self._infer_slice_info("02_trimmed_data")
            if trimmed and n:
                s = trimmed["start_idx_raw"]
                info = {"start_idx_raw": s, "end_idx_raw": s + n - 1, "count": n}
                self._slice_cache[folder] = info
                return info
            info = {"start_idx_raw": None, "end_idx_raw": None, "count": n or 0}
            self._slice_cache[folder] = info
            return info

        # Default/unknown
        info = {"start_idx_raw": None, "end_idx_raw": None, "count": 0}
        self._slice_cache[folder] = info
        return info

    def _update_summary(self):
        """Update the summary using only slice_info.json + the currently loaded data."""
        folder = self._current_folder_path()
        slice_info = self._load_slice_info(folder) or {}
        # Accept both old and new keys from slice_info.json
        start    = slice_info.get("start_idx_raw")
        end      = slice_info.get("end_idx_raw")
        total    = slice_info.get("count")
        first_ts = slice_info.get("first_ts", slice_info.get("first_timestamp"))
        last_ts  = slice_info.get("last_ts",  slice_info.get("last_timestamp"))

        # shape
        shape_text = "—"
        if isinstance(self.current_data, list):
            # action JSON
            G, A, D = self._compute_action_shape(self.current_data)
            shape_text = f"({G}, {A}, {D})"
        elif hasattr(self.current_data, "shape"):
            shape_text = str(self.current_data.shape)

        # current slice (source-indexed)
        cur = self.current_gamestate if isinstance(self.current_gamestate, int) else 0
        cur_src = (start + cur) if isinstance(start, int) else cur

        # Write directly into the visible summary fields (no hidden keys).
        self.summary_vars["shape"].set(shape_text)
        self.summary_vars["range"].set(
            f"gamestate {start} → {end}" if (start is not None and end is not None) else "—"
        )
        self.summary_vars["count"].set(str(total) if total is not None else "—")
        self.summary_vars["current"].set(str(cur_src))
        # Fill the "Source range:" with dataset-local indices (0..N-1)
        if isinstance(total, int) and total > 0:
            self.summary_vars["source_range"].set(f"0 → {total - 1}")
        else:
            self.summary_vars["source_range"].set("—")
        # Show first/last timestamps if present in slice_info.json
        self.summary_vars["first_ts"].set(str(first_ts) if first_ts is not None else "—")
        self.summary_vars["last_ts"].set(str(last_ts)   if last_ts  is not None else "—")
        # Update sequence range information
        sequence_range_text = "—"
        if hasattr(self, 'current_numpy_data') and self.current_numpy_data is not None:
            data = self.current_numpy_data
            if data.ndim == 3:
                # 3D: (batch, timesteps, features)
                n_timesteps = data.shape[1]
                n_features = data.shape[2]
                
                # Show the current batch (gamestate) being viewed
                current_batch = self.current_gamestate
                total_batches = data.shape[0]
                
                if n_features == 8:
                    sequence_range_text = f"Batch {current_batch}/{total_batches-1} | Action Sequence: 0-{n_timesteps-1}"
                elif n_features == 128:
                    sequence_range_text = f"Batch {current_batch}/{total_batches-1} | Gamestate Sequence: 0-{n_timesteps-1}"
                else:
                    sequence_range_text = f"Batch {current_batch}/{total_batches-1} | Feature Sequence: 0-{n_timesteps-1}"
                    
            elif data.ndim == 4:
                # 4D: (batch, timesteps, actions, features)
                n_timesteps = data.shape[1]
                n_actions = data.shape[2]
                n_features = data.shape[3]
                current_action_slice = getattr(self, "current_action_slice", 0)
                
                # Show the current batch (gamestate) being viewed
                current_batch = self.current_gamestate
                total_batches = data.shape[0]
                
                if n_features == 8:
                    sequence_range_text = f"Batch {current_batch}/{total_batches-1} | Action Sequence: 0-{n_timesteps-1} | Slice: {current_action_slice}/{n_actions-1}"
                else:
                    sequence_range_text = f"Batch {current_batch}/{total_batches-1} | Feature Sequence: 0-{n_timesteps-1} | Slice: {current_action_slice}/{n_actions-1}"
        
        self.summary_vars["sequence_range"].set(sequence_range_text)
        
        # Leave the per-slice unix timestamp to _update_summary_current_slice().
        # If slice_info has a slice-wide first_ts, show it as a fallback in "Unix timestamp".
        if first_ts is not None:
            self.summary_vars["timestamp"].set(str(first_ts))

    def _update_summary_on_file_load(self, file_path: str):
        """Update dataset/shape/range when a file is loaded."""
        if not file_path:
            return
        folder = self.current_folder or os.path.basename(os.path.dirname(file_path))
        self.summary_vars["dataset"].set(folder)
        # shape: always derive from the clicked file (prevents stale shapes)
        shp = "-"
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            try:
                shp = str(tuple(np.load(file_path, mmap_mode="r").shape))
            except Exception:
                shp = "-"
        elif ext == ".json":
            try:
                j = json.load(open(file_path, "r"))
                if isinstance(j, list):
                    shp = f"{len(j)} records"
                elif isinstance(j, dict):
                    shp = f"{len(j)} keys"
                else:
                    shp = type(j).__name__
            except Exception:
                shp = "-"
        elif ext == ".csv":
            try:
                # cheap count: first pass only
                import pandas as pd
                cols = len(pd.read_csv(file_path, nrows=1).columns)
                shp = f"{cols} columns"
            except Exception:
                shp = "-"
        self.summary_vars["shape"].set(shp)
        # sequences (B) when this file is a sequence tensor; also compute N = B + L
        B_val = None
        try:
            if ext == ".npy":
                arr = np.load(file_path, mmap_mode="r")
                if arr.ndim == 4 and arr.shape[1] == 10 and arr.shape[2] == 100 and arr.shape[3] == 8:
                    B_val = arr.shape[0]  # action_input_sequences
                elif arr.ndim == 3 and arr.shape[1] == 10 and arr.shape[2] == 128:
                    B_val = arr.shape[0]  # gamestate_sequences
                elif arr.ndim == 3 and arr.shape[1] == 100 and arr.shape[2] == 8:
                    B_val = arr.shape[0]  # action_targets
        except Exception:
            pass
        self.summary_vars["sequences"].set(str(B_val) if B_val is not None else "—")

        # range + count (N): infer slice; if this is a sequence tensor and in final set, prefer N=B+L
        self._slice_info = self._infer_slice_info(folder)
        si = self._slice_info or {}
        start = si.get("start_idx_raw")
        end   = si.get("end_idx_raw")
        N     = si.get("count", 0)
        # If we know B for a sequence tensor, recompute N = B + L (more reliable)
        if B_val is not None:
            N = (B_val + self.sequence_length)
            if start is not None:
                end = start + N - 1
        self.summary_vars["count"].set(str(N))
        if start is not None and end is not None:
            self.summary_vars["range"].set(f"gamestate {start} → {end}")
        else:
            self.summary_vars["range"].set("—")
        # current + ts
        self._update_summary_current_slice()
        # also reflect the current position on the graph
        try:
            self._update_actions_graph()
        except Exception:
            self._log("display_numpy: graph update failed", level="error")

    def _update_summary_current_slice(self):
        """Update the current slice and timestamp based on current indices + mapping."""
        si = self._slice_info or {}
        start_raw = si.get("start_idx_raw")
        if start_raw is None:
            self.summary_vars["current"].set("—")
            self.summary_vars["timestamp"].set("—")
            return

        b = int(self.current_gamestate or 0)
        s = int(self.current_action_slice or 0)
        L = int(self.sequence_length)

        # Default: 2D features / JSON -> single index
        label = None
        raw_idx_for_ts = start_raw + b

        if isinstance(self.current_numpy_data, np.ndarray):
            arr = self.current_numpy_data
            if arr.ndim == 4 and arr.shape[1] == L and arr.shape[2] == 100 and arr.shape[3] == 8:
                # action_input_sequences: window [t .. t+L-1], show range and in-window slice
                t0 = start_raw + b
                t1 = t0 + L - 1
                label = f"{t0}-{t1} ({s})"
                raw_idx_for_ts = t0 + s
            elif arr.ndim == 3 and arr.shape[1] == L and arr.shape[2] == 128:
                # gamestate_sequences: window [t .. t+L-1], show range
                t0 = start_raw + b
                t1 = t0 + L - 1
                label = f"{t0}-{t1}"
                raw_idx_for_ts = t0  # timestamp at window start
            elif arr.ndim == 3 and arr.shape[1] == 100 and arr.shape[2] == 8:
                # action_targets: single target time t = S + b + L
                t = start_raw + b + L
                label = str(t)
                raw_idx_for_ts = t

        if label is None:
            # 2D features / JSON fallback
            label = str(start_raw + b)
            raw_idx_for_ts = start_raw + b

        # Set "Current slice" (dataset-local label) and "Current slice (source)" (raw index)
        self.summary_vars["current"].set(label)
        self.summary_vars["current_slice"].set(str(raw_idx_for_ts))

        # Unix timestamp from metadata
        ts = "—"
        full = self._load_full_metadata() or []
        if 0 <= raw_idx_for_ts < len(full):
            ts = str(full[raw_idx_for_ts].get("absolute_timestamp", "—"))
        self.summary_vars["timestamp"].set(ts)

    # -------- Actions Graph helpers --------
    def _compute_counts_from_json(self, json_path: str):
        """Return counts per-gamestate from an action_data JSON (list of dicts)."""
        try:
            arr = json.load(open(json_path, "r"))
            ys = []
            for rec in arr:
                c = 0
                for key in ("mouse_movements","clicks","key_presses","key_releases","scrolls"):
                    v = rec.get(key, [])
                    if isinstance(v, list): c += len(v)
                ys.append(c)
            xs = list(range(len(ys)))
            return xs, ys
        except Exception as e:
            self._log("_compute_counts_from_json failed", level="error", path=json_path, error=str(e))
            return None, None

    def _compute_counts_from_tensors(self, json_path: str):
        """Return counts per-gamestate from an action_tensors JSON (list of flat lists)."""
        try:
            arr = json.load(open(json_path, "r"))
            ys = []
            for flat_list in arr:
                # Each gamestate is a flat list of 8-value chunks
                # Handle older files that accidentally included a leading count
                start = 1 if (len(flat_list) % 8 != 0) else 0
                action_count = (len(flat_list) - start) // 8
                ys.append(action_count)
            xs = list(range(len(ys)))
            return xs, ys
        except Exception as e:
            self._log("_compute_counts_from_tensors failed", level="error", path=json_path, error=str(e))
            return None, None

    def _compute_counts_from_sequences(self, folder: str):
        """
        Build counts per raw index for the final set using:
          - action_input_sequences.npy as primary (sum of 'count' over actions at each timestep)
          - action_targets.npy for the target step t+L (fills the tail)
        """
        try:
            seq_in = os.path.join(self.data_dir, folder, "action_input_sequences.npy")
            seq_tg = os.path.join(self.data_dir, folder, "action_targets.npy")
            if not os.path.exists(seq_in):
                return None, None
            L = self.sequence_length
            X = np.load(seq_in, mmap_mode="r")  # expect (B,10,100,8) or (B, A, T, F) then swapped by viewer
            if X.ndim != 4: return None, None
            # Make sure axis-1 is timestep as in the viewer (B, T, A, F)
            Xt = X if X.shape[1] == L else X.swapaxes(1,2)
            B = Xt.shape[0]; A = Xt.shape[2]
            N = B + L
            counts = np.full((N,), np.nan, dtype=float)
            # Fill from inputs: raw index r = b + s
            for b in range(B):
                # sum 'count' feature (col 0) over actions for each timestep
                # Xt[b, s, :, 0] -> counts at raw r=b+s
                step_counts = Xt[b, :, :, 0].sum(axis=1)  # shape (L,)
                for s in range(L):
                    r = b + s
                    if np.isnan(counts[r]):
                        counts[r] = float(step_counts[s])
            # Fill last tail (t = b + L) from targets if available
            if os.path.exists(seq_tg):
                T = np.load(seq_tg, mmap_mode="r")  # (B,100,8) or (B, A, F)
                if T.ndim == 3 and T.shape[-1] >= 1:
                    for b in range(min(B, T.shape[0])):
                        r = b + L
                        if np.isnan(counts[r]):
                            counts[r] = float(T[b, :, 0].sum())
            # Fallback: replace any remaining NaNs with 0
            counts = np.nan_to_num(counts, nan=0.0)
            xs = list(range(int(counts.shape[0])))
            return xs, counts.tolist()
        except Exception as e:
            self._log("_compute_counts_from_sequences failed", level="error", error=str(e))
            return None, None

    def _build_actions_graph_data(self):
        """Return (x_indices, per-gamestate counts) from the JSON that is currently loaded."""
        if not isinstance(self.current_data, list):
            return [], []
        counts = []
        first = self.current_data[0] if self.current_data else None
        if isinstance(first, dict):
            # action_data.json
            for gs in self.current_data:
                c = (len(gs.get("mouse_movements", [])) +
                     len(gs.get("clicks", [])) +
                     len(gs.get("key_presses", [])) +
                     len(gs.get("key_releases", [])) +
                     len(gs.get("scrolls", [])))
                counts.append(c)
        else:
            # *_action_tensors.json (each gs is a flat list of 8-value chunks)
            for flat_list in self.current_data:
                start = 1 if (len(flat_list) % 8 != 0) else 0  # tolerate old files with a leading count
                action_count = max(0, (len(flat_list) - start) // 8)
                counts.append(action_count)
        return list(range(len(counts))), counts

    def _get_action_counts_for_folder(self, folder: str):
        """Memoized counts for the active folder."""
        if folder in self._counts_cache:
            return self._counts_cache[folder]
        xs, ys = None, None
        if folder == "01_raw_data":
            # Try tensor file first, fall back to action data
            xs, ys = self._compute_counts_from_tensors(os.path.join(self.data_dir, folder, "raw_action_tensors.json"))
            if xs is None:
                xs, ys = self._compute_counts_from_json(os.path.join(self.data_dir, folder, "raw_action_data.json"))
        elif folder == "02_trimmed_data":
            # Try tensor file first, fall back to action data
            xs, ys = self._compute_counts_from_tensors(os.path.join(self.data_dir, folder, "trimmed_raw_action_tensors.json"))
            if xs is None:
                xs, ys = self._compute_counts_from_json(os.path.join(self.data_dir, folder, "trimmed_raw_action_data.json"))
        elif folder == "03_normalized_data":
            # Try tensor file first, fall back to action data
            xs, ys = self._compute_counts_from_tensors(os.path.join(self.data_dir, folder, "normalized_action_tensors.json"))
            if xs is None:
                xs, ys = self._compute_counts_from_json(os.path.join(self.data_dir, folder, "normalized_action_data.json"))
        elif folder == "06_final_training_data":
            xs, ys = self._compute_counts_from_sequences(folder)
        # Store even if None (avoid repeated work)
        self._counts_cache[folder] = (xs, ys)
        return xs, ys

    def _update_actions_graph(self):
        """Draw/refresh the actions graph for the current folder + position."""
        if not _HAS_MPL:
            return
        folder = self.current_folder or ""
        si = self._slice_info or self._infer_slice_info(folder) or {}
        start = si.get("start_idx_raw"); end = si.get("end_idx_raw"); N = si.get("count", 0)
        self._ax.clear()
        self._ax.set_xlabel("Gamestate"); self._ax.set_ylabel("Action count"); self._ax.grid(True, alpha=0.25)
        if not folder or start is None or end is None or N <= 0:
            self._ax.set_title("No range available")
            self._canvas.draw_idle()
            return
        xs, ys = self._build_actions_graph_data()
        if not xs or ys is None:
            self._ax.set_title("No action data for this folder")
            self._canvas.draw_idle()
            return
        # xs begin at 0 for that dataset; align to raw space
        raw_xs = [start + i for i in xs]
        self._ax.plot(raw_xs, ys, linewidth=1.25)
        # cursor at current raw index (based on the same logic as summary)
        cur_label = self.summary_vars["current"].get()
        cur_idx = None
        if "-" in cur_label:  # range like "5-14" or "5-14 (8)"
            base = cur_label.split(" ", 1)[0]
            a, b = base.split("-")
            # if "(s)" present, put cursor at the in-window position
            if "(" in cur_label and ")" in cur_label:
                try:
                    s = int(cur_label.split("(")[1].split(")")[0])
                    cur_idx = int(a) + s
                except Exception:
                    cur_idx = int(a)
            else:
                cur_idx = int(a)  # window start
        else:
            try:
                cur_idx = int(cur_label)
            except Exception:
                cur_idx = None
        if cur_idx is not None:
            self._ax.axvline(cur_idx, linestyle="--", linewidth=1.0)
        self._ax.set_xlim(start - 0.5, end + 0.5)
        
        # Create enhanced title with sequence context
        title = f"Action counts per gamestate [{start} … {end}]"
        
        # Add sequence range and slice information if available
        if hasattr(self, 'current_numpy_data') and self.current_numpy_data is not None:
            data = self.current_numpy_data
            if data.ndim == 3:
                # 3D: (batch, timesteps, features)
                n_timesteps = data.shape[1]
                n_features = data.shape[2]
                
                # Show the current batch being viewed
                current_batch = self.current_gamestate
                total_batches = data.shape[0]
                
                if n_features == 8:
                    title += f" | Batch {current_batch}/{total_batches-1} | Action Sequence: 0-{n_timesteps-1}"
                elif n_features == 128:
                    title += f" | Batch {current_batch}/{total_batches-1} | Gamestate Sequence: 0-{n_timesteps-1}"
                else:
                    title += f" | Batch {current_batch}/{total_batches-1} | Feature Sequence: 0-{n_timesteps-1}"
                    
            elif data.ndim == 4:
                # 4D: (batch, timesteps, actions, features)
                n_timesteps = data.shape[1]
                n_actions = data.shape[2]
                n_features = data.shape[3]
                current_action_slice = getattr(self, "current_action_slice", 0)
                
                # Show the current batch being viewed
                current_batch = self.current_gamestate
                total_batches = data.shape[0]
                
                if n_features == 8:
                    title += f" | Batch {current_batch}/{total_batches-1} | Action Sequence: 0-{n_timesteps-1} | Slice: {current_action_slice}/{n_actions-1}"
                else:
                    title += f" | Batch {current_batch}/{total_batches-1} | Feature Sequence: 0-{n_timesteps-1} | Slice: {current_action_slice}/{n_actions-1}"
        
        self._ax.set_title(title)
        self._canvas.draw_idle()

    def browse_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.data_dir)
        if new_dir:
            self.data_dir = new_dir
            self.dir_var.set(new_dir)
            self.load_files()

    def _scan_sessions(self):
        """Return (sessions_root_path, sorted_session_names)."""
        try:
            root = Path(self.base_data_root) / "recording_sessions"
            sessions = sorted([p.name for p in root.iterdir() if p.is_dir()])
            return root, sessions
        except Exception as e:
            self._log("_scan_sessions failed", level="error", error=str(e))
            return Path(self.base_data_root) / "recording_sessions", []

    def refresh_sessions(self):
        """Re-scan sessions and update the combobox; keep selection if possible."""
        self.sessions_root, self.sessions = self._scan_sessions()
        self.session_cb["values"] = self.sessions
        cur = self.session_var.get()
        if cur not in self.sessions:
            # select latest if current is missing
            if self.sessions:
                self.session_var.set(self.sessions[-1])
                self.on_session_change()
        else:
            # make sure dir reflects current choice
            self.data_dir = str(self.sessions_root / cur)
            self.dir_var.set(self.data_dir)
            self.load_files()

    def on_session_change(self, event=None):
        """When the user chooses a different session in the combobox."""
        sel = self.session_var.get().strip()
        if not sel:
            return
        self.session_name = sel
        self.data_dir = str(self.sessions_root / sel)
        self.dir_var.set(self.data_dir)
        self.load_files()

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
