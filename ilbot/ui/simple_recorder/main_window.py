# --- stdlib / third-party ---
import os
import re
import json
import time
import random
import ctypes
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

try:
    import pygetwindow as gw
except Exception:
    gw = None

import pyautogui
pyautogui.FAILSAFE = True  # moving mouse to a corner aborts

# --- project modules (split out; no behavior changes) ---
# constants
from .constants import (
    AUTO_REFRESH_MS,
    AUTO_RUN_TICK_MS,
    PRE_ACTION_DELAY_MS,
    RULE_WAIT_TIMEOUT_MS,
)

# small text/time helpers that table updates rely on
from .utils.common import (
    _clean_rs,
    _fmt_age_ms,
    _norm_name,
    _now_ms,
)

# rect/geometry helpers used across click resolution and table rendering
from .utils.rects import (
    _mk_rect,
    _rect_contains,
    _center_distance,
    _unwrap_rect,
    _rect_center_xy,
)

# Win32 POINT struct (used by cursor helpers)
from .utils.win32 import POINT

# IPC client (used by click/key execution)
from .services.ipc_client import RuneLiteIPC

# left-side status UI (window, gamestate dir, ipc port)
from .ui.status_panel import InstanceStatusPanel

# action plan selector/registry
from .action_plans import get_plan, _CRAFT_ANIMS
# Plans + helper primitives (re-exported here for UI/rules)
from .action_plans import (
    Plan,
    PLAN_REGISTRY,
    get_plan,
    _is_crafting_anim,
    _inv_has,
    _inv_count,
    _bank_slots,
    _norm_name,
)

from .session_ports import (
    _username_from_title,
    _session_dir_for_username,
    _autofill_port_for_username,
)



class SimpleRecorderWindow(ttk.Frame):
    def __init__(self, root, instance_index: int = 0):
        # Distinguish between a window host and an embedded container
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            host = root
            container = root
        else:
            try:
                host = root.winfo_toplevel()
            except Exception:
                host = root
            container = root

        # Build into the container, keep a handle to the real toplevel as self.root
        super().__init__(container, padding=12)
        self.root = host
        self.instance_index = int(instance_index)

        # Preserve your existing top-level setup
        try:
            self.root.title("Simple Bot Recorder")
        except Exception:
            pass
        try:
            self.root.minsize(1100, 600)
        except Exception:
            pass
        self._rl_window_rect = None  # {'x': int, 'y': int, 'width': int, 'height': int}
        self._rl_window_title = None

        # These are per-instance state holders:
        self.window_title_var = tk.StringVar(value="(none)")
        self.session_dir_var = tk.StringVar(value="")  # per-instance gamestate dir (editable)
        self.ipc_port_var = tk.StringVar(value="")  # free text, no scan required
        self.plan_var = tk.StringVar(value="SAPPHIRE_RINGS")  # per-instance plan

        # If you had a pre-existing self.session_dir, keep it in sync:
        self.session_dir = None  # or your existing default Path/str

        def _choose_session_dir():
            start_dir = self.session_dir_var.get() or os.getcwd()
            chosen = filedialog.askdirectory(initialdir=start_dir, title="Choose gamestate directory")
            if chosen:
                self.set_session_dir(chosen)

        # LEFT PANE: put the status panel at the top of your controls column (col=0)
        self.status_panel = InstanceStatusPanel(
            self,
            on_choose_dir=_choose_session_dir,
            window_title_var=self.window_title_var,
            session_dir_var=self.session_dir_var,
            ipc_port_var=self.ipc_port_var
        )
        self.status_panel.grid(row=0, column=0, sticky="nsew", padx=(8, 8), pady=(8, 8))

        # state
        self.session_dir: Path | None = None
        self.auto_refresh_on = False
        self._after_id = None
        self._last_crafting_ms = 0  # last time we observed crafting (anim or UI)
        self.input_enabled = False  # safety toggle
        self._canvas_offset = (0, 8)  # only use if your coords are CANVAS-relative
        self._rl_window_rect = None  # optional: set this in detect_window() if you find RuneLite's rect
        self.run_enabled = False
        self._run_var = tk.BooleanVar(value=self.run_enabled)
        self._last_action_time_ms = 0
        self._auto_run_after_id = None
        self._plan_ready_ms = 0  # don’t act until now (used for pre-action wait)
        self._step_state = None  # {'step': dict, 'stage': 'prewait'|'acting'|'postwait', 't0': int}

        # overall grid (two columns)
        self.grid(sticky="nsew")
        self.grid_columnconfigure(0, weight=0)            # controls (fixed)
        self.grid_columnconfigure(1, weight=1)            # info panel grows to the right
        self.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self._last_clicked_target = "—"
        self._last_click_epoch_ms = 0
        self._last_action_text = "—"
        self._auto_var = tk.BooleanVar(value=self.auto_refresh_on)
        self._input_var = tk.BooleanVar(value=self.input_enabled)

        self._build_controls_panel()
        self._build_info_panel()

        # start auto refresh
        self._schedule_auto_refresh()

    # ---------------- UI BUILDERS ----------------

    def _build_controls_panel(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # section: title
        title = ttk.Label(left, text="Simple Bot Recorder", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        # Window Detection
        win = ttk.LabelFrame(left, text="Window Detection")
        win.grid(row=1, column=0, sticky="ew", pady=6)
        win.grid_columnconfigure(0, weight=1)
        win.grid_columnconfigure(1, weight=0)

        # Combobox to pick 'RuneLite - <username>'
        ttk.Label(win, text="Target window:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.window_combo = ttk.Combobox(win, state="readonly", values=[])
        self.window_combo.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        self.window_combo.bind("<<ComboboxSelected>>", self._on_window_select)

        # Refresh/Detect button
        self.detect_button = ttk.Button(win, text="Refresh Windows", command=self.detect_window)
        self.detect_button.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 6))

        self.window_status = ttk.Label(win, text="No window selected", foreground="#2c7a7b")
        self.window_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        # IPC / Port Picker
        # IPC / Port Picker (free-text, no Scan)
        ipc = ttk.LabelFrame(left, text="IPC (RuneLite Plugin)")
        ipc.grid(row=2, column=0, sticky="ew", pady=6)
        ipc.grid_columnconfigure(0, weight=1)
        ipc.grid_columnconfigure(1, weight=0)

        ttk.Label(ipc, text="Port:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        # keep existing self.ipc_port_var (created in __init__)
        if not isinstance(getattr(self, "ipc_port_var", None), tk.StringVar):
            self.ipc_port_var = tk.StringVar(value="17000")

        self.ipc_port_entry = ttk.Entry(ipc, textvariable=self.ipc_port_var)
        self.ipc_port_entry.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        btns = ttk.Frame(ipc)
        btns.grid(row=1, column=1, sticky="e", padx=(0, 8), pady=(0, 6))
        ttk.Button(btns, text="Ping", command=self._ipc_ping).grid(row=0, column=0)
        self.ipc_status = ttk.Label(ipc, text="Not tested", foreground="#6b7280")
        self.ipc_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        # Input mode toggle stays the same
        self.input_mode_var = getattr(self, "input_mode_var", None) or tk.StringVar(value="ipc")
        ttk.Label(ipc, text="Input mode:").grid(row=3, column=0, sticky="w", padx=8, pady=(4, 0))
        mode_row = ttk.Frame(ipc)
        mode_row.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        ttk.Radiobutton(mode_row, text="IPC", variable=self.input_mode_var, value="ipc").grid(row=0, column=0,
                                                                                              padx=(0, 12))
        ttk.Radiobutton(mode_row, text="pyautogui", variable=self.input_mode_var, value="pyautogui").grid(row=0,
                                                                                                          column=1)
        # --- Action Plan picker ---
        planf = ttk.LabelFrame(left, text="Action Plan")
        planf.grid(row=3, column=0, sticky="ew", pady=6)
        planf.grid_columnconfigure(0, weight=1)

        ttk.Label(planf, text="Plan:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        from .action_plans import PLAN_REGISTRY  # top-level import is also fine
        plan_names = [("SAPPHIRE_RINGS", "Sapphire Rings"), ("GOLD_RINGS", "Gold Rings"), ("EMERALD_RINGS", "Emerald Rings"), ("GO_TO_GE", "Go to GE")]
        self.plan_combo = ttk.Combobox(
            planf,
            state="readonly",
            values=[label for _, label in plan_names]
        )
        # map label->id and id->label
        self._plan_label_to_id = {label: pid for pid, label in plan_names}
        self._plan_id_to_label = {pid: label for pid, label in plan_names}
        # default select
        self.plan_combo.set(self._plan_id_to_label[self.plan_var.get()])
        self.plan_combo.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        def _on_plan_change(*_):
            sel_label = self.plan_combo.get()
            self.plan_var.set(self._plan_label_to_id.get(sel_label, "SAPPHIRE_RINGS"))
            # optional: force a table refresh to reflect different head step summaries
            try:
                self.refresh_gamestate_info()
            except Exception:
                pass

        self.plan_combo.bind("<<ComboboxSelected>>", _on_plan_change)

        # --- Session Management ---
        sess = ttk.LabelFrame(left, text="Session")
        sess.grid(row=4, column=0, sticky="ew", pady=6)
        sess.grid_columnconfigure(0, weight=1)
        sess.grid_columnconfigure(1, weight=1)

        # read-only path display
        self.session_path_var = tk.StringVar(value="")
        ttk.Label(sess, text="Dir:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.session_path_entry = ttk.Entry(sess, textvariable=self.session_path_var, state="readonly")
        self.session_path_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6))

        self.copy_path_button = ttk.Button(sess, text="Copy Gamestates Path", command=self.copy_gamestates_path)
        self.copy_path_button.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        self.copy_path_button.state(["disabled"])

        self.session_status = ttk.Label(sess, text="No session", foreground="#2f855a")
        self.session_status.grid(row=2, column=1, sticky="w", padx=10, pady=6)

        # Recording Controls
        rec = ttk.LabelFrame(left, text="Recording Controls")
        rec.grid(row=5, column=0, sticky="ew", pady=6)
        for c in range(3):
            rec.grid_columnconfigure(c, weight=1)

        self.start_button = ttk.Button(rec, text="Start", command=self.start_recording, width=10)
        self.pause_button = ttk.Button(rec, text="Pause", command=self.pause_recording, width=10)
        self.end_button   = ttk.Button(rec, text="End",   command=self.end_session,     width=10)

        self.start_button.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.pause_button.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.end_button.grid(  row=0, column=2, padx=6, pady=6, sticky="ew")

        self.start_button.state(["disabled"])   # until a session exists
        self.pause_button.state(["disabled"])
        self.end_button.state(["disabled"])

        self.recording_status = ttk.Label(left, text="Recording paused", foreground="#d69e2e")
        self.recording_status.grid(row=6, column=0, sticky="w", pady=(4, 0))

        # ---- Debug Mouse Move ----
        dbg = ttk.LabelFrame(left, text="Debug Mouse Move")
        dbg.grid(row=7, column=0, sticky="ew", pady=6, padx=6)
        for c in range(3):
            dbg.grid_columnconfigure(c, weight=1)

        ttk.Label(dbg, text="X:").grid(row=0, column=0, sticky="e")
        self.debug_x = ttk.Entry(dbg, width=6)
        self.debug_x.grid(row=0, column=1, sticky="ew")

        ttk.Label(dbg, text="Y:").grid(row=1, column=0, sticky="e")
        self.debug_y = ttk.Entry(dbg, width=6)
        self.debug_y.grid(row=1, column=1, sticky="ew")

        self.debug_btn = ttk.Button(dbg, text="Move Mouse", command=self._debug_move_mouse)
        self.debug_btn.grid(row=0, column=2, rowspan=2, sticky="ew", padx=(6, 0))

    def _build_info_panel(self):
        # ---- styles (row colors) ----
        style = ttk.Style(self)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        # tags -> foreground colors
        self._row_tag_styles = {
            "bool":   {"foreground": "#1f7a1f"},   # green
            "status": {"foreground": "#d97706"},   # amber
            "inv":    {"foreground": "#2563eb"},   # blue
            "tile":   {"foreground": "#7c3aed"},   # purple
            "mouse":  {"foreground": "#059669"},   # teal/green
            "text":   {"foreground": "#374151"},   # slate
        }
        # Register tag styles
        for tag, opts in self._row_tag_styles.items():
            style.map(f"{tag}.TLabel", foreground=[("!disabled", opts["foreground"])])

        right = ttk.LabelFrame(self, text="Live Gamestate Information")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 0))  # <-- nsew + no right padding
        # allow children to expand
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        tv = ttk.Treeview(
            right,
            columns=("field", "value"),
            show="headings",
            height=10,
            selectmode="none",
        )
        tv.heading("field", text="Field")
        tv.heading("value", text="Value")
        tv.column("field", width=160, anchor="w", stretch=False)  # fixed label col
        tv.column("value", anchor="w", stretch=True)  # value col stretches
        tv.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # zebra striping
        tv.tag_configure("odd", background="#f7f7f7")
        tv.tag_configure("even", background="#ffffff")

        # color tags
        tv.tag_configure("bool",  foreground=self._row_tag_styles["bool"]["foreground"])
        tv.tag_configure("status",foreground=self._row_tag_styles["status"]["foreground"])
        tv.tag_configure("inv",   foreground=self._row_tag_styles["inv"]["foreground"])
        tv.tag_configure("tile",  foreground=self._row_tag_styles["tile"]["foreground"])
        tv.tag_configure("mouse", foreground=self._row_tag_styles["mouse"]["foreground"])
        tv.tag_configure("text",  foreground=self._row_tag_styles["text"]["foreground"])

        self.info_table = tv

        # Pretty labels (left column text) and row -> tag mapping
        self._pretty_labels = {
            "bank_open":         "Bank:",
            "crafting_open":     "Crafting UI:",
            "crafting_status":   "Crafting Status:",
            "inventory_summary": "Inventory:",
            "last_click": "Last Click:",
            "clicked_target": "Last Action:",
            "hovered_tile":      "Hovered Tile:",
            "mouse_pos":         "Mouse Cursor:",
            "last_interaction":  "Last Interaction:",
            "menu_entries":      "Menu Entries:",
            "phase": "Phase:",
            "next_action": "Action:",
            "sys_cursor_pos": "System Cursor:",
        }
        self._row_tags = {
            "bank_open": "bool",
            "crafting_open": "bool",
            "crafting_status": "status",
            "inventory_summary": "inv",
            "last_click": "mouse",
            "clicked_target": "text",
            "hovered_tile": "tile",
            "mouse_pos": "mouse",
            "last_interaction": "text",
            "menu_entries": "text",
            "phase": "status",
            "next_action": "text",
            "sys_cursor_pos": "mouse",
        }

        # Insert rows once and remember their iids for quick updates
        self.table_items = {}
        keys_in_order = [
            "bank_open",
            "crafting_open",
            "crafting_status",
            "inventory_summary",
            "last_click",
            "clicked_target",
            "hovered_tile",
            "mouse_pos",
            "sys_cursor_pos",
            "last_interaction",
            "menu_entries",
            "phase",
            "next_action",
        ]
        for i, key in enumerate(keys_in_order):
            zebra = "odd" if i % 2 else "even"
            color = self._row_tags[key]
            iid = tv.insert("", "end",
                            values=(self._pretty_labels[key], "—"),
                            tags=(zebra, color))
            self.table_items[key] = iid

        # buttons under table
        btns = ttk.Frame(right, padding=(10, 0, 10, 10))
        btns.grid(row=1, column=0, sticky="ew")
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.refresh_btn = ttk.Button(btns, text="Refresh Now", command=self.refresh_gamestate_info)
        self.refresh_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.auto_chk = ttk.Checkbutton(
            btns,
            text="Auto Update Table",
            variable=self._auto_var,
            command=self._on_auto_check
        )
        self.auto_chk.grid(row=0, column=1, sticky="w", padx=(6, 0))

        # NEW control row
        ctrl = ttk.Frame(right, padding=(10, 0, 10, 10))
        ctrl.grid(row=2, column=0, sticky="ew")
        ctrl.grid_columnconfigure(0, weight=1)
        ctrl.grid_columnconfigure(1, weight=1)

        self.exec_btn = ttk.Button(ctrl, text="Execute Action", command=self._execute_next_action)
        self.exec_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.input_chk = ttk.Checkbutton(
            ctrl,
            text="Enable Input",
            variable=self._input_var,
            command=self._on_input_check
        )
        self.input_chk.grid(row=0, column=0, sticky="w", padx=(0, 6))

        # self.input_toggle_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.exec_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Debug panel
        logf = ttk.LabelFrame(right, text="Action Debug")
        logf.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        right.grid_rowconfigure(3, weight=1)

        self.debug_text = ScrolledText(logf, height=6)
        self.debug_text.grid(row=0, column=0, sticky="nsew")

        # Auto-run row
        runrow = ttk.Frame(right, padding=(10, 0, 10, 10))
        runrow.grid(row=4, column=0, sticky="ew")
        runrow.grid_columnconfigure(0, weight=1)

        self.run_chk = ttk.Checkbutton(
            runrow,
            text="Auto-Run (loop actions)",
            variable=self._run_var,
            command=self._on_run_check
        )
        self.run_chk.grid(row=0, column=0, sticky="w")

    # ---------------- Actions / hooks ----------------

    def detect_window(self):
        """
        Refresh the list of RuneLite user windows ('RuneLite - <username>')
        and auto-select the current choice if present.
        """
        self._populate_window_combo()

        # If user had a selection, re-apply it after refresh
        sel = getattr(self, "_window_selected_title", None)
        if sel and sel in (self.window_combo["values"] or ()):
            try:
                idx = list(self.window_combo["values"]).index(sel)
                self.window_combo.current(idx)
                self._on_window_select()
            except Exception:
                pass

        # Debug dump of what we found
        titles = [t for t, _ in getattr(self, "_window_scan", [])]
        if titles:
            self._debug("Detected RuneLite windows:\n  - " + "\n  - ".join(titles))
        else:
            self._debug("Detected RuneLite windows: (none)")

    def create_session(self):
        """
        Create a new per-instance session dir under your standard base,
        and sync the left status with set_session_dir().
        """
        base = Path(r"D:\repos\bot_runelite_IL\data\recording_sessions")
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = base / stamp / "gamestates"
        session_dir.mkdir(parents=True, exist_ok=True)

        self.set_session_dir(str(session_dir))  # <-- sync left panel + internal state

        self.session_status.config(text=f"Session created: {stamp}")
        self.copy_path_button.state(["!disabled"])
        self.start_button.state(["!disabled"])
        self.end_button.state(["!disabled"])

    def copy_gamestates_path(self):
        if not self.session_dir:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.session_dir))
        self.root.update()

    def start_recording(self):
        self.recording_status.config(text="Recording…")
        self.pause_button.state(["!disabled"])

    def pause_recording(self):
        self.recording_status.config(text="Recording paused")

    def end_session(self):
        self.recording_status.config(text="Recording paused")
        self.start_button.state(["disabled"])
        self.pause_button.state(["disabled"])
        self.end_button.state(["disabled"])
        self.session_status.config(text="No session")
        self.session_dir = None

    # ---------------- Gamestate refresh ----------------

    def _toggle_auto_refresh(self):
        self.auto_refresh_on = not self.auto_refresh_on
        self.toggle_btn.config(text=f"Auto: {'On' if self.auto_refresh_on else 'Off'}")
        if self.auto_refresh_on:
            self._schedule_auto_refresh()
        elif self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_auto_refresh(self):
        if not self.auto_refresh_on:
            return
        try:
            self.refresh_gamestate_info()
        except Exception as e:
            # Never let a single bad refresh kill the loop
            self._debug(f"refresh_gamestate_info error: {e}")
        finally:
            self._after_id = self.root.after(AUTO_REFRESH_MS, self._schedule_auto_refresh)

    def _latest_gamestate_file(self) -> Path | None:
        """
        Return the newest *.json file **only** from THIS INSTANCE'S session_dir.
        No cross-instance/global fallback.
        """
        d = self.session_dir
        if not d:
            return None
        d = Path(d)
        if not d.exists() or not d.is_dir():
            return None

        newest: tuple[float, Path] | None = None
        for f in d.glob("*.json"):
            try:
                ts = f.stat().st_mtime
            except FileNotFoundError:
                continue
            if newest is None or ts > newest[0]:
                newest = (ts, f)
        return newest[1] if newest else None

    def refresh_gamestate_info(self):
        f = self._latest_gamestate_file()
        if not f:
            # clear rows gracefully (or just return)
            for iid in self.table_items.values():
                # leave left labels; set value to em dash
                current_tags = self.info_table.item(iid, "tags") or ()
                values = self.info_table.item(iid, "values")
                self.info_table.item(iid, values=(values[0], "—"), tags=current_tags)
            return

        try:
            text = f.read_text(encoding="utf-8")
        except FileNotFoundError:
            return  # it was pruned after selection; next tick will find another
        except Exception:
            return

        try:
            root = json.loads(text)
        except Exception:
            return

        payload = root.get("data", {})  # everything lives under "data"

        # Bank open/closed
        bank_open = bool(payload.get("bank", {}).get("bankOpen", False))

        # Crafting interface open/closed
        crafting_open = bool(payload.get("craftingInterfaceOpen", False))

        # Crafting status via animation (-1 idle, 899 crafting)
        anim_id = int(payload.get("player", {}).get("animation", -1))
        if anim_id == -1:
            crafting_status = "Idle (anim=-1)"
        elif anim_id == 899:
            crafting_status = "Crafting (anim=899)"
        else:
            crafting_status = f"Anim {anim_id}"

        # --- Crafting hysteresis (prevents flapping back to "Moving to furnace") ---
        now = _now_ms()
        if crafting_open or _is_crafting_anim(anim_id):
            self._last_crafting_ms = now

        CRAFT_GRACE_MS = 5000  # keep Crafting phase for up to 5s after last observed craft
        craft_recent = (now - self._last_crafting_ms) <= CRAFT_GRACE_MS

        # Inventory summary
        inv_slots = payload.get("inventory", {}).get("slots", [])
        name_counts: dict[str, int] = {}
        for s in inv_slots:
            n = s.get("itemName")
            q = int(s.get("quantity", 0) or 0)
            if n:
                name_counts[n] = name_counts.get(n, 0) + q
        inventory_summary = ", ".join(f"{k}: {v}" for k, v in sorted(name_counts.items())) or "0"

        # Hovered tile + first hovered game object name (if any)
        ht = payload.get("hoveredTile")
        if isinstance(ht, dict):
            hx, hy, hp = ht.get('worldX', '?'), ht.get('worldY', '?'), ht.get('plane', '?')
            obj_name = None
            gos = ht.get("gameObjects") or []
            for g in gos:
                name = _clean_rs((g or {}).get("name", "")) or ""
                if name and name.lower() != "null" and name.lower() != "unknown":
                    obj_name = name
                    break
            hovered_tile = f"({hx}, {hy}) p={hp}" + (f" • {obj_name}" if obj_name else "")
        else:
            hovered_tile = "—"

        # Mouse cursor position (canvas coords)
        mouse = payload.get("mouse")
        if isinstance(mouse, dict):
            mx, my = mouse.get("canvasX", None), mouse.get("canvasY", None)
            mouse_pos = f"({mx}, {my})" if mx is not None and my is not None else "—"
        else:
            mouse_pos = "—"

        # Last interaction (+ time since) with tags stripped
        li = payload.get("lastInteraction")
        if isinstance(li, dict):
            act = _clean_rs(li.get("action", ""))
            tgt = _clean_rs(li.get("target", ""))
            ts  = li.get("timestamp")  # expected ms since epoch (from your plugin)
            if isinstance(ts, (int, float)):
                age_ms = int(time.time() * 1000) - int(ts)
                age_txt = "  •  " + _fmt_age_ms(age_ms)
            else:
                age_txt = ""
            last_interaction = (f"{act} -> {tgt}".strip() or "—") + age_txt
        else:
            last_interaction = "—"

        # Menu entries (first few), with tags stripped
        entries = payload.get("menuEntries", [])
        if isinstance(entries, list) and entries:
            parts = []
            for e in entries[:4]:
                opt = _clean_rs((e or {}).get("option", ""))
                tgt = _clean_rs((e or {}).get("target", ""))
                txt = " ".join(p for p in (opt, tgt) if p).strip()
                if txt:
                    parts.append(txt)
            menu_entries = ", ".join(parts) if parts else "—"
        else:
            menu_entries = "—"

        # Last click (from plugin JSON: data["lastClick"])
        lc = payload.get("lastClick")
        if isinstance(lc, dict):
            cx, cy = lc.get("canvasX"), lc.get("canvasY")
            if cx is not None and cy is not None:
                since_ms = lc.get("sinceMs")
                if since_ms is None and lc.get("epochMs") is not None:
                    try:
                        now_ms = int(time.time() * 1000)
                        since_ms = now_ms - int(lc["epochMs"])
                    except Exception:
                        since_ms = None
                age_txt = f"  •  {_fmt_age_ms(int(since_ms))}" if isinstance(since_ms, (int, float)) else ""
                last_click = f"({cx}, {cy}){age_txt}"
            else:
                last_click = "—"
        else:
            last_click = "—"

        # Persisted comprehensive "Last Action"
        lc_obj = payload.get("lastClick")
        new_epoch = None
        if isinstance(lc_obj, dict):
            try:
                if lc_obj.get("epochMs") is not None:
                    new_epoch = int(lc_obj["epochMs"])
            except Exception:
                new_epoch = None

        if new_epoch is not None and new_epoch > self._last_click_epoch_ms:
            try:
                self._last_action_text = self._resolve_last_action(root) or "—"
            except Exception as e:
                self._last_action_text = "—"
                self._debug(f"_resolve_last_action error: {e}")
            self._last_click_epoch_ms = new_epoch

        # Use the persisted text (won’t disappear on later refreshes)
        last_action = self._last_action_text

        plan_impl = self._current_plan()
        phase = plan_impl.compute_phase(payload, craft_recent)
        plan = plan_impl.build_action_plan(payload, phase)

        # Compact summary for table: show first step (if any)
        if isinstance(plan, dict) and isinstance(plan.get("steps"), list) and plan["steps"]:
            first = plan["steps"][0]
            tgt = first.get("target", {}) or {}
            tname = tgt.get("name") or tgt.get("domain") or "target"
            click = first.get("click", {}) or {}
            if click.get("type") in ("rect-center", "rect-random"):
                rect = tgt.get("bounds") or tgt.get("clickbox")
                cx, cy = _rect_center_xy(_unwrap_rect(rect))
                summary = f"{first.get('action')} → {tname}" + (f" @ ({cx}, {cy})" if cx is not None else "")
            elif click.get("type") == "point":
                summary = f"{first.get('action')} → {tname} @ ({click.get('x')},{click.get('y')})"
            elif click.get("type") == "key":
                summary = f"{first.get('action')} → press {click.get('key')}"
            else:
                summary = f"{first.get('action')} → {tname}"
        else:
            summary = "—"

        # System cursor position (absolute Windows coords)
        sx, sy = self._get_system_cursor_pos()
        sys_cursor = f"({sx}, {sy})" if isinstance(sx, int) and isinstance(sy, int) else "—"

        # (Optional) Keep the full plan around for debugging / future UI
        self._last_action_plan = plan
        self._last_phase = phase

        # Track current plan head to detect changes between ticks
        try:
            steps = (plan.get("steps") or []) if isinstance(plan, dict) else []
            head_action = (steps[0].get("action") if steps else None)
        except Exception:
            head_action = None
        self._plan_head_action = head_action
        self._plan_phase = phase

        # Push to table rows
        self._table_set("bank_open",        "Open" if bank_open else "Closed")
        self._table_set("crafting_open",    "Open" if crafting_open else "Closed")
        self._table_set("crafting_status",  crafting_status)
        self._table_set("inventory_summary",inventory_summary)
        self._table_set("last_click", last_click)
        self._table_set("clicked_target", last_action)
        self._table_set("hovered_tile",     hovered_tile)
        self._table_set("mouse_pos",        mouse_pos)
        self._table_set("sys_cursor_pos", sys_cursor)
        self._table_set("last_interaction", last_interaction)
        self._table_set("menu_entries",     menu_entries)
        self._table_set("phase", phase)
        self._table_set("next_action", summary)


    def _table_set(self, key: str, value: str):
        """Update one row in the info table."""
        iid = self.table_items.get(key)
        if not iid:
            return
        # preserve existing zebra + color tags
        current_tags = self.info_table.item(iid, "tags") or ()
        self.info_table.item(iid, values=(self._pretty_labels[key], value), tags=current_tags)

    def _resolve_last_action(self, root_payload) -> str:
        """
        Returns a comprehensive last action string:
        'Click (cx, cy) -> <target> @ (tx, ty)'
        or 'Click (cx, cy) -> Ground • Tile (wx, wy) p=z'
        """
        data = root_payload.get("data", {}) if isinstance(root_payload, dict) else {}
        lc = data.get("lastClick")
        if not isinstance(lc, dict):
            return "—"

        cx, cy = lc.get("canvasX"), lc.get("canvasY")
        if not isinstance(cx, (int, float)) or not isinstance(cy, (int, float)):
            return "—"
        cx, cy = int(cx), int(cy)

        # Collect candidates identical to _resolve_last_click, but also compute target canvas center
        def rect_center(rect):
            x, y, w, h = rect
            return int(x + w / 2), int(y + h / 2)

        candidates = []

        # Inventory slots
        inv = data.get("inventory", {})
        for slot in (inv.get("slots", []) or []):
            b = slot.get("bounds") or {}
            rect = _mk_rect(b.get("bounds"))
            if rect and _rect_contains(rect, cx, cy):
                tx, ty = rect_center(rect)
                candidates.append({
                    "priority": 10,
                    "kind": "inventory_slot",
                    "name": slot.get("itemName"),
                    "slotId": slot.get("slotId"),
                    "rect": rect,
                    "tx": tx, "ty": ty,
                })

        # Bank slots
        bank = data.get("bank", {})
        for slot in (bank.get("slots", []) or []):
            b = slot.get("bounds") or {}
            rect = _mk_rect(b.get("bounds"))
            if rect and _rect_contains(rect, cx, cy):
                tx, ty = rect_center(rect)
                candidates.append({
                    "priority": 9,
                    "kind": "bank_slot",
                    "name": slot.get("itemName"),
                    "slotId": slot.get("slotId"),
                    "rect": rect,
                    "tx": tx, "ty": ty,
                })

        # Widgets (crafting etc.)
        cw = data.get("crafting_widgets", {})
        if isinstance(cw, dict):
            for widget_key, v in cw.items():
                rect = _mk_rect((v or {}).get("bounds") if isinstance(v, dict) else None)
                if rect and _rect_contains(rect, cx, cy):
                    tx, ty = rect_center(rect)
                    candidates.append({
                        "priority": 8,
                        "kind": "widget",
                        "name": widget_key,
                        "rect": rect,
                        "tx": tx, "ty": ty,
                    })

        # Hovered tile game objects
        ht = data.get("hoveredTile") or {}
        for obj in (ht.get("gameObjects", []) or []):
            rect = _mk_rect(obj.get("clickbox"))
            if rect and _rect_contains(rect, cx, cy):
                tx, ty = rect_center(rect)
                candidates.append({
                    "priority": 7,
                    "kind": "game_object",
                    "name": obj.get("name"),
                    "id": obj.get("id"),
                    "rect": rect,
                    "tx": tx, "ty": ty,
                })

        # If no strict hit, try nearest game object within small radius
        if not candidates:
            best = None
            best_d = 1e18
            for obj in (data.get("closestGameObjects") or []):
                rect = _mk_rect(obj.get("clickbox"))
                if rect:
                    d = _center_distance(rect, cx, cy)
                    tx, ty = rect_center(rect)
                else:
                    gx, gy = obj.get("canvasX"), obj.get("canvasY")
                    if not isinstance(gx, (int, float)) or not isinstance(gy, (int, float)):
                        continue
                    dx, dy = cx - gx, cy - gy
                    d = (dx * dx + dy * dy) ** 0.5
                    tx, ty = int(gx), int(gy)
                if d < best_d:
                    best_d, best = d, {"kind": "game_object_near", "name": obj.get("name"),
                                       "id": obj.get("id"), "tx": tx, "ty": ty}
            if best and best_d <= 40:
                best["priority"] = 6
                candidates.append(best)

        # If still nothing, ground tile (Walk here) using hoveredTile
        if not candidates:
            wx, wy, pl = ht.get("worldX"), ht.get("worldY"), ht.get("plane")
            if wx is not None and wy is not None:
                return f"Click ({cx}, {cy}) -> Ground • Tile ({wx}, {wy}) p={pl}"

        # Nothing matched and no ground tile available — bail safely
        if not candidates:
            return f"Click ({cx}, {cy}) -> Unknown"

        # Pick best candidate by priority then center proximity
        for c in candidates:
            rect = c.get("rect")
            c["_dist"] = _center_distance(rect, cx, cy) if rect else 0.0
        candidates.sort(key=lambda c: (-c["priority"], c["_dist"]))
        best = candidates[0]

        tx, ty = best.get("tx"), best.get("ty")
        kind = best.get("kind")
        name = best.get("name")
        slot_id = best.get("slotId")
        obj_id = best.get("id")

        if kind == "inventory_slot":
            nm = name if name else "empty"
            target = f"Inventory slot {slot_id}: {nm}"
        elif kind == "bank_slot":
            nm = name if name else "empty"
            target = f"Bank slot {slot_id}: {nm}"
        elif kind == "widget":
            target = f"Widget: {name}"
        elif kind in ("game_object", "game_object_near"):
            target = f"{'Nearest ' if kind.endswith('_near') else ''}Object: {name} (id {obj_id})" if name else f"Object id {obj_id}"
        else:
            target = name or kind or "—"

        if isinstance(tx, int) and isinstance(ty, int):
            return f"Click ({cx}, {cy}) -> {target} @ ({tx}, {ty})"
        else:
            return f"Click ({cx}, {cy}) -> {target}"

    def _toggle_input_enable(self):
        self.input_enabled = not self.input_enabled
        self.input_toggle_btn.config(text=f"Enable Input: {'ON' if self.input_enabled else 'OFF'}")

    def _screen_point_from_rect(self, rect: dict | None, jitter_px: int | None = None):
        """
        rect: {"x","y","width","height"} in SCREEN coords (assumed).
        jitter_px: if provided, add a random ± jitter within the rect edges.
        returns (x, y) int or (None, None) if invalid.
        """
        if not isinstance(rect, dict): return (None, None)
        try:
            x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["width"]), int(rect["height"])
            if w <= 0 or h <= 0: return (None, None)
            cx = x + w // 2
            cy = y + h // 2
            if jitter_px:
                # keep random point inside the rect; cap jitter to rect size
                jx = min(jitter_px, max(1, w // 3))
                jy = min(jitter_px, max(1, h // 3))
                rx = random.randint(x + 2, x + w - 3)
                ry = random.randint(y + 2, y + h - 3)
                return (rx, ry)
            return (cx, cy)
        except Exception:
            return (None, None)

    def _apply_canvas_offset(self, x: int | None, y: int | None):
        """
        If you discover your coords are CANVAS-relative, set self._canvas_offset=(ox,oy).
        Otherwise this is a no-op (adds (0,0)).
        """
        if x is None or y is None: return (None, None)
        ox, oy = self._canvas_offset
        return (int(x) + int(ox), int(y) + int(oy))

    def _ensure_ipc(self) -> bool:
        """
        Ensure self.ipc is configured using the **typed** port.
        Returns True if ready, False otherwise (and sets a red status).
        """
        port = self.get_ipc_port()
        if port is None or port <= 0 or port > 65535:
            try:
                self.ipc_status.config(text="Enter a valid IPC port (e.g., 17001)", foreground="#b91c1c")
            except Exception:
                pass
            return False

        if not hasattr(self, "ipc"):
            self.ipc = RuneLiteIPC(port=port, pre_action_ms=PRE_ACTION_DELAY_MS, timeout_s=2.0)
        else:
            self.ipc.port = port
            self.ipc.timeout_s = 2.0
        return True

    def _do_click_point(self, x: int, y: int, button: str = "left", move_ms: int = 80):
        mode = self.input_mode_var.get() if hasattr(self, "input_mode_var") else "ipc"

        # compute system coords once (used for logging, and for pyautogui mode)
        sys_x, sys_y = self._canvas_to_system(int(x), int(y))
        title = self._rl_window_title or "?"
        rl_rect = self._rl_window_rect

        if mode == "pyautogui":
            try:
                dur = max(0.0, float(move_ms) / 1000.0)
            except Exception:
                dur = 0.0
            try:
                pyautogui.moveTo(int(sys_x), int(sys_y), duration=dur)
                if button == "left":
                    pyautogui.click()
                elif button == "right":
                    pyautogui.click(button="right")
                else:
                    pyautogui.click()
            except Exception as e:
                self._debug(f"PYAUTOGUI click error: {type(e).__name__}: {e}")
            return

        # ----- IPC mode -----
        if not self._ensure_ipc():
            self._debug("IPC not ready: invalid or missing port")
            return
        # preflight ping so we fail fast with a clear message
        pong = self.ipc._send({"cmd": "ping"})
        if not (isinstance(pong, dict) and pong.get("ok")):
            try:
                self.ipc_status.config(text=f"Preflight fail @ {self.ipc.port}: {pong}", foreground="#b91c1c")
            except Exception:
                pass
            self._debug(f"IPC preflight failed on port {self.ipc.port}: {pong}")
            return

        btn = 1 if button == "left" else (3 if button == "right" else 1)
        self.ipc.focus()
        resp = self.ipc.click_canvas(int(x), int(y), button=btn)
        self._debug(f"IPC click → port={self.ipc.port} canvas=({int(x)},{int(y)}) resp={resp}")
        try:
            self.ipc_status.config(text=f"Click resp @ {self.ipc.port}: {resp}", foreground="#065f46" if resp.get("ok") else "#b91c1c")
        except Exception:
            pass


    def _do_press_key(self, key: str):
        mode = self.input_mode_var.get() if hasattr(self, "input_mode_var") else "ipc"

        if mode == "pyautogui":
            try:
                pyautogui.press(str(key))
            except Exception as e:
                self._debug(f"PYAUTOGUI key error: {type(e).__name__}: {e}")
            return

        # ----- IPC mode -----
        if not self._ensure_ipc():
            self._debug("IPC not ready: invalid or missing port")
            return
        pong = self.ipc._send({"cmd": "ping"})
        if not (isinstance(pong, dict) and pong.get("ok")):
            try:
                self.ipc_status.config(text=f"Preflight fail @ {self.ipc.port}: {pong}", foreground="#b91c1c")
            except Exception:
                pass
            self._debug(f"IPC preflight failed on port {self.ipc.port}: {pong}")
            return

        self.ipc.focus()
        title = self._rl_window_title or "?"
        rl_rect = self._rl_window_rect
        resp = self.ipc.key(key)
        try:
            self.ipc_status.config(text=f"Key resp @ {self.ipc.port}: {resp}", foreground="#065f46" if resp.get("ok") else "#b91c1c")
        except Exception:
            pass

    def _execute_next_action(self):
        if not self.input_enabled:
            self._table_set("next_action", "Input disabled (toggle it ON)")
            self._debug("Execute skipped: input disabled")
            return

        plan = getattr(self, "_last_action_plan", None)
        if not plan or not isinstance(plan, dict) or not plan.get("steps"):
            self._table_set("next_action", "No plan available")
            self._debug("Execute skipped: no plan available")
            return

        step = plan["steps"][0]
        click = step.get("click", {}) or {}
        target = step.get("target", {}) or {}
        ctype = (click.get("type") or "").lower()

        try:
            # pull current plan & first step
            step = (self._last_action_plan or {}).get("steps", [])[0]
            tgt = (step.get("target") or {})
            dbg = (tgt.get("debug") or {})

            ge = (dbg.get("ge_center") or {})
            me = (dbg.get("player") or {})
            vec = (dbg.get("goal_vec") or {})
            tw = (dbg.get("chosen_tile_world") or {})
            tc = (dbg.get("chosen_tile_canvas") or {})

            if ge and me and tw and tc and vec:
                self._debug(
                    "GE DEBUG → center=({gx},{gy})  player=({wx},{wy})  vec=({dx},{dy})  "
                    "pick_world=({tx},{ty},p={pl})  pick_canvas=({cx},{cy})".format(
                        gx=ge.get("x"), gy=ge.get("y"),
                        wx=me.get("x"), wy=me.get("y"),
                        dx=vec.get("dx"), dy=vec.get("dy"),
                        tx=tw.get("x"), ty=tw.get("y"), pl=tw.get("plane"),
                        cx=tc.get("x"), cy=tc.get("y"),
                    )
                )
        except Exception:
            # never let debug printing break action execution
            pass

        try:
            if ctype in ("rect-center", "rect-random"):
                rect = target.get("bounds") or target.get("clickbox")
                rect = rect if isinstance(rect, dict) else None
                jitter_px = int(click.get("jitter_px") or 0) if ctype == "rect-random" else None
                px, py = self._screen_point_from_rect(rect, jitter_px)
                px, py = self._apply_canvas_offset(px, py)
                if px is None:
                    self._table_set("next_action", "No rect available for click")
                    self._debug("Rect click skipped: no rect")
                    return
                self._debug(f"CLICK rect → px={px} py={py}")
                self._do_click_point(px, py)

            elif ctype in ("point", "canvas-point", "canvas_point"):
                px, py = click.get("x"), click.get("y")
                px, py = self._apply_canvas_offset(px, py)
                if not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
                    self._table_set("next_action", "Invalid point for click")
                    self._debug(f"Point click skipped: invalid coords {click}")
                    return

                # Optional: sanity log vs RL canvas bounds
                rl = self._rl_window_rect or {}
                self._debug(f"CLICK point → canvas=({click.get('x')},{click.get('y')}) "
                            f"final=({int(px)},{int(py)}) RLRect={rl} mode={getattr(self, 'input_mode_var', None) and self.input_mode_var.get()}")

                sys_before = self._get_system_cursor_pos()
                self._do_click_point(int(px), int(py))
                sys_after = self._get_system_cursor_pos()

            elif ctype == "key":
                key = (click.get("key") or "").lower()
                if not key:
                    self._table_set("next_action", "Invalid key")
                    self._debug("Keypress skipped: invalid key")
                    return
                self._debug(f"KEY → '{key}'")
                sys_before = self._get_system_cursor_pos()
                self._do_press_key(key)

            elif ctype == "world-tile":
                wx = int(click.get("worldX"))
                wy = int(click.get("worldY"))
                pl = int(click.get("plane", 0))
                # Ask IPC to project tile → canvas
                resp = self._ipc.project_world_tile(wx, wy, pl)
                self._log_debug(f"[GE] project {wx},{wy},{pl} -> {resp}")
                if resp.get("ok") and resp.get("onScreen"):
                    x, y = int(resp["x"]), int(resp["y"])
                    self._ipc.focus()  # optional
                    self._ipc.click_canvas(x, y, button=1)
                else:
                    self._log_debug(f"[GE] projection not clickable: {resp}")
                return  # handled

            else:
                self._table_set("next_action", f"Unsupported click.type: {ctype}")
                self._debug(f"Unsupported click type: {ctype} (step={step})")
                return

            # Refresh after action so the next step is proposed
            self.refresh_gamestate_info()

        except Exception as e:
            self._table_set("next_action", f"Action error: {e}")
            self._debug(f"Action error: {e}")

    def _on_auto_check(self):
        self.auto_refresh_on = bool(self._auto_var.get())
        if self.auto_refresh_on:
            self._schedule_auto_refresh()
        elif self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _on_input_check(self):
        self.input_enabled = bool(self._input_var.get())

    def _debug(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        # console
        print(line.strip())
        # GUI
        dt = getattr(self, "debug_text", None)
        if dt:
            dt.insert("end", line)
            dt.see("end")

    def _get_system_cursor_pos(self) -> tuple[int | None, int | None]:
        pt = POINT()
        ok = ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y)) if ok else (None, None)

    # --- Tk window rect (absolute screen coords) ---
    def _get_tk_window_rect(self) -> dict:
        # ensure layout is up-to-date
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        x = int(self.root.winfo_rootx())
        y = int(self.root.winfo_rooty())
        w = int(self.root.winfo_width())
        h = int(self.root.winfo_height())
        return {"x": x, "y": y, "width": w, "height": h}

    # optional setter you can call from detect_window() if you locate RuneLite
    def _set_runelite_window_rect(self, x: int, y: int, w: int, h: int):
        self._rl_window_rect = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}

    def _debug_move_mouse(self):
        try:
            x = int(self.debug_x.get())
            y = int(self.debug_y.get())
        except Exception:
            self._debug("Invalid X/Y input")
            return

        # absolute system coordinates (screen pixels)
        import pyautogui
        pyautogui.moveTo(x, y, duration=0.1)
        self._debug(f"Moved mouse to absolute ({x}, {y})")

    def _on_run_check(self):
        self.run_enabled = bool(self._run_var.get())
        # start or stop the loop
        if self.run_enabled:
            # kick the loop
            self._schedule_auto_run_tick()
            self._plan_ready_ms = _now_ms() + PRE_ACTION_DELAY_MS

        else:
            if self._auto_run_after_id:
                try:
                    self.root.after_cancel(self._auto_run_after_id)
                except Exception:
                    pass
                self._auto_run_after_id = None

    def _schedule_auto_run_tick(self):
        # Always reschedule; guard against re-entrancy
        if not self.run_enabled:
            return
        try:
            self._auto_run_tick()
        except Exception as e:
            self._debug(f"auto_run_tick error: {e}")
        finally:
            self._auto_run_after_id = self.root.after(AUTO_RUN_TICK_MS, self._schedule_auto_run_tick)

    def _auto_run_tick(self):
        # keep the UI fresh
        try:
            self.refresh_gamestate_info()
        except Exception as e:
            self._debug(f"refresh during auto-run error: {e}")

        data = self._latest_payload()
        bank_open = bool((data.get("bank") or {}).get("bankOpen", False))
        if bank_open:
            sapp = self._bank_count(data, "Sapphire")
            gold = self._bank_count(data, "Gold bar")
            # Stop as soon as either is out
            if sapp <= 0 or gold <= 0:
                missing = []
                if sapp <= 0: missing.append("Sapphires")
                if gold <= 0: missing.append("Gold bars")
                self._stop_auto_run(f"Out of {', '.join(missing)} in bank")
                return

        if not self.input_enabled or not self._action_ready():
            return

        plan = getattr(self, "_last_action_plan", None)
        if not plan or not isinstance(plan, dict):
            return
        steps = plan.get("steps") or []
        if not steps:
            return

        # If the plan/phase head changed since we started the current step, reset it
        if self._step_state:
            curr_action = self._step_state.get("step", {}).get("action")
            new_action = steps[0].get("action") if steps else None
            curr_phase = self._step_state.get("phase")
            new_phase = getattr(self, "_plan_phase", None)

            if (new_phase is not None and curr_phase != new_phase) or (new_action != curr_action):
                self._step_state = None
                # optionally seed pre-action delay so we don't hammer immediately
                self._mark_action_done()

        # Ensure we have a step in progress
        if not self._step_state and steps:
            self._begin_step(steps[0])

        # Ensure we have a step in progress
        if not self._step_state:
            self._begin_step(steps[0])

        # Advance the step state machine (prewait -> acting -> postwait)
        done = self._advance_step_state()
        # (When done, next tick will pick up the next freshly-built plan step)

    def _handle_wait_step(self, step: dict):
        """
        For 'wait-crafting-complete':
          Poll until inventory sapphires==0 or gold bars==0, or timeout.
        """
        name = (step.get("action") or "").lower()
        t0 = self._now_ms_local()

        def _poll():
            # stop conditions
            if not self.run_enabled:
                return
            if (self._now_ms_local() - t0) > RULE_WAIT_TIMEOUT_MS:
                self._debug("wait-step timeout reached")
                self._mark_action_done()
                return

            # Refresh state and re-evaluate condition
            try:
                self.refresh_gamestate_info()
            except Exception as e:
                self._debug(f"wait poll refresh error: {e}")

            # Rebuild a tiny view on inventory
            try:
                # last payload is inside the table refresh path; pull from newest json again:
                f = self._latest_gamestate_file()
                if not f:
                    self._auto_run_after_id = self.root.after(RULE_WAIT_TIMEOUT_MS, _poll)
                    return
                root = json.loads(f.read_text(encoding="utf-8"))
                data = root.get("data", {})
                sapp = _inv_count(data, "Sapphire")
                gold = _inv_count(data, "Gold bar")
                done = (sapp == 0) or (gold == 0)
            except Exception:
                done = False

            if done:
                self._debug("wait-step complete: materials consumed")
                self._mark_action_done()
                # fall through; main auto-run tick will kick again
            else:
                # keep polling
                self._auto_run_after_id = self.root.after(AUTO_RUN_TICK_MS, _poll)

        # Kick the poller
        _poll()

    def _now_ms_local(self) -> int:
        return int(time.time() * 1000)

    def _action_ready(self) -> bool:
        """Ready to launch next action after pre-action delay expires."""
        return self._now_ms_local() >= getattr(self, "_plan_ready_ms", 0)

    def _mark_action_done(self, delay_ms: int | None = None):
        """
        Schedule the next moment when an action is allowed.
        If delay_ms is None, use PRE_ACTION_DELAY_MS.
        """
        d = PRE_ACTION_DELAY_MS if delay_ms is None else int(delay_ms)
        self._plan_ready_ms = self._now_ms_local() + max(0, d)

    def _latest_payload(self) -> dict:
        """Read the newest JSON and return data{} (or {})."""
        f = self._latest_gamestate_file()
        if not f:
            return {}
        try:
            root = json.loads(f.read_text(encoding="utf-8"))
            return root.get("data", {}) or {}
        except Exception:
            return {}

    def _eval_rule(self, rule: str, data: dict) -> bool:
        """Evaluate a single rule string against the latest data dict."""
        rule = (rule or "").strip()

        # allow simple OR
        if " OR " in rule:
            return any(self._eval_rule(part, data) for part in rule.split(" OR "))

        # normalize helpers
        def _bool(path):
            if path == "bankOpen":
                return bool((data.get("bank") or {}).get("bankOpen", False))
            if path == "craftingInterfaceOpen":
                return bool(data.get("craftingInterfaceOpen", False))
            if path == "crafting in progress":
                anim = int((data.get("player") or {}).get("animation", -1))
                return anim in _CRAFT_ANIMS or bool(data.get("craftingInterfaceOpen", False))
            return False

        # simple booleans:  bankOpen == true/false
        m = re.fullmatch(r"(bankOpen|craftingInterfaceOpen)\s*==\s*(true|false)", rule, re.I)
        if m:
            key, val = m.group(1), m.group(2).lower() == "true"
            return _bool(key) == val

        # player.animation == 899
        m = re.fullmatch(r"player\.animation\s*==\s*(-?\d+)", rule, re.I)
        if m:
            want = int(m.group(1))
            anim = int((data.get("player") or {}).get("animation", -1))
            return anim == want

        # inventory contains 'Item'
        m = re.fullmatch(r"inventory\s+contains\s+'(.+)'", rule, re.I)
        if m:
            return _inv_has(data, m.group(1))

        # inventory does not contain 'Item'
        m = re.fullmatch(r"inventory\s+does\s+not\s+contain\s+'(.+)'", rule, re.I)
        if m:
            return not _inv_has(data, m.group(1))

        # inventory count('Item') <op> N
        m = re.fullmatch(r"inventory\s+count\('(.+)'\)\s*(==|>=|<=|>|<)\s*(\d+)", rule, re.I)
        if m:
            item, op, n = m.group(1), m.group(2), int(m.group(3))
            c = _inv_count(data, item)
            return {
                "==": c == n, ">=": c >= n, "<=": c <= n, ">": c > n, "<": c < n
            }[op]

        # crafting in progress (standalone)
        if rule.lower().strip() == "crafting in progress":
            return _bool("crafting in progress")

        # Unknown rule => be conservative (treat as False)
        self._debug(f"Unknown rule: {rule}")
        return False

    def _rules_ok(self, rules: list[str] | None, data: dict) -> bool:
        if not rules:
            return True
        return all(self._eval_rule(r, data) for r in rules if isinstance(r, str) and r.strip())

    def _begin_step(self, step: dict):
        self._step_state = {
            "step": step,
            "stage": "prewait",
            "t0": self._now_ms_local(),
            "phase": getattr(self, "_plan_phase", None),
        }

    def _advance_step_state(self):
        """
        Drives one step through:
          prewait (pre-delay only) -> acting -> postwait (rules or retry on timeout)
        Returns True when the step is fully finished, else False.
        """
        if not self._step_state:
            return True

        st   = self._step_state
        step = st["step"]
        stage= st["stage"]
        t0   = st["t0"]

        now  = self._now_ms_local()
        data = self._latest_payload()

        # --- local probe for readable rule status (mirrors _eval_rule patterns) ---
        def _probe_rule(rule: str) -> str:
            r = (rule or "").strip()
            # OR support: show each part
            if " OR " in r:
                parts = [p.strip() for p in r.split(" OR ") if p.strip()]
                statuses = [self._eval_rule(p, data) for p in parts]
                probes = [ _probe_rule(p) for p in parts ]
                return f"{' OR '.join(probes)}  => {any(statuses)}"

            # booleans
            m = re.fullmatch(r"(bankOpen|craftingInterfaceOpen)\s*==\s*(true|false)", r, re.I)
            if m:
                key, want = m.group(1), (m.group(2).lower()=="true")
                cur = bool((data.get("bank") or {}).get("bankOpen", False)) if key=="bankOpen" \
                      else bool(data.get("craftingInterfaceOpen", False))
                return f"{key}=={str(want).lower()}  (now {cur})  => {cur==want}"

            # crafting in progress / animation
            if r.lower().strip() == "crafting in progress":
                anim = int((data.get("player") or {}).get("animation", -1))
                cip  = bool(data.get("craftingInterfaceOpen", False))
                ok   = (anim in _CRAFT_ANIMS) or cip
                return f"crafting in progress  (anim={anim}, craftingUI={cip})  => {ok}"

            m = re.fullmatch(r"player\.animation\s*==\s*(-?\d+)", r, re.I)
            if m:
                want = int(m.group(1))
                anim = int((data.get("player") or {}).get("animation", -1))
                return f"player.animation=={want}  (now {anim})  => {anim==want}"

            # inventory contains / not contains
            m = re.fullmatch(r"inventory\s+contains\s+'(.+)'", r, re.I)
            if m:
                name = m.group(1)
                cnt = _inv_count(data, name)
                ok  = cnt > 0
                return f"inventory contains '{name}'  (count={cnt})  => {ok}"

            m = re.fullmatch(r"inventory\s+does\s+not\s+contain\s+'(.+)'", r, re.I)
            if m:
                name = m.group(1)
                cnt = _inv_count(data, name)
                ok  = cnt == 0
                return f"inventory does not contain '{name}'  (count={cnt})  => {ok}"

            # inventory count('Item') <op> N
            m = re.fullmatch(r"inventory\s+count\('(.+)'\)\s*(==|>=|<=|>|<)\s*(\d+)", r, re.I)
            if m:
                item, op, n = m.group(1), m.group(2), int(m.group(3))
                c = _inv_count(data, item)
                res = {"==": c == n, ">=": c >= n, "<=": c <= n, ">": c > n, "<": c < n}[op]
                return f"inventory count('{item}') {op} {n}  (now {c})  => {res}"

            # fallback: just show boolean result
            ok = self._eval_rule(r, data)
            return f"{r}  => {ok}"

        def _dump_rules(prefix: str, rules: list[str] | None):
            if not rules:
                self._debug(f"{prefix}: (none)")
                return
            lines = [ _probe_rule(r) for r in rules if isinstance(r, str) and r.strip() ]
            if lines:
                self._debug(f"{prefix}:\n  - " + "\n  - ".join(lines))
            else:
                self._debug(f"{prefix}: (none)")

        # 1) PREWAIT: ONLY honor pre-action delay, do NOT block on preconditions.
        if stage == "prewait":
            delay_ok = (now - t0) >= PRE_ACTION_DELAY_MS

            if delay_ok:
                st["stage"] = "acting"
            return False

        # 2) ACTING: perform once, then go to postwait
        if stage == "acting":
            self._execute_next_action()  # already refreshes
            st["stage"] = "postwait"
            st["t0"] = now
            return False

        # 3) POSTWAIT: wait for postconditions; on timeout, retry the action
        if stage == "postwait":
            post = step.get("postconditions")
            post_ok = self._rules_ok(post, data)

            if post_ok:
                self._step_state = None
                self._mark_action_done()  # uses PRE_ACTION_DELAY_MS
                return True

            if (now - t0) >= RULE_WAIT_TIMEOUT_MS:
                self._execute_next_action()
                st["t0"] = now
                return False

            # keep waiting
            return False

        # unknown stage: finish
        self._step_state = None
        return True

    def _set_runelite_window_rect(self, x: int, y: int, width: int, height: int):
        self._rl_window_rect = {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}

    def _canvas_to_system(self, cx: int | None, cy: int | None) -> tuple[int | None, int | None]:
        """
        Convert canvas/client-area coords to absolute screen coords using the
        detected RuneLite window and current canvas_offset.
        """
        if cx is None or cy is None:
            return (None, None)

        # 1) apply your known canvas_offset (titlebar/insets, etc.)
        ox, oy = self._canvas_offset if hasattr(self, "_canvas_offset") else (0, 0)
        rx, ry = int(cx) + int(ox), int(cy) + int(oy)

        # 2) add window origin if we have it
        rl = self._rl_window_rect if isinstance(self._rl_window_rect, dict) else None
        if not rl:
            return (rx, ry)
        return (int(rl["x"]) + rx, int(rl["y"]) + ry)

    def _scan_runelite_windows(self):
        """
        Return a list of (title, window_obj) for windows whose title matches
        'RuneLite - <username>' (case-insensitive, tolerant of spacing).
        We don't rely on .isVisible (pygetwindow can misreport); we only skip minimized or 0x0.
        """
        if gw is None:
            return []

        try:
            # pull everything once; titles may include IDEs etc.
            wins = gw.getAllWindows()
        except Exception as e:
            return []

        out = []
        pat = re.compile(r"^rune?lite\s*-\s*(.+)$", re.IGNORECASE)  # RuneLite / Runelite
        for w in wins:
            try:
                title = (w.title or "").strip()
                if not title:
                    continue
                m = pat.match(title)
                if not m:
                    continue
                # basic sanity: skip minimized/0-sized
                if getattr(w, "isMinimized", False):
                    continue
                if int(getattr(w, "width", 0)) <= 0 or int(getattr(w, "height", 0)) <= 0:
                    continue
                out.append((title, w))
            except Exception:
                continue

        # sort stable by title
        out.sort(key=lambda tw: tw[0].lower())
        return out

    def _populate_window_combo(self):
        items = self._scan_runelite_windows()
        titles = [t for t, _ in items]
        self._window_scan = items  # keep the (title, window) tuples

        # Update combo list
        try:
            self.window_combo["values"] = titles
        except Exception:
            pass

        if titles:
            # auto-select the first if nothing selected yet
            if not getattr(self, "_window_selected_title", None):
                self.window_combo.current(0)
                self._on_window_select()
            self.window_status.config(text=f"Found {len(titles)} RuneLite user window(s)")
        else:
            self.window_status.config(text="No RuneLite 'username' windows found")
            self._rl_window_rect = None
            self._rl_window_title = None

    def _on_window_select(self, *_):
        try:
            sel = self.window_combo.get().strip()
        except Exception:
            sel = ""

        self._window_selected_title = sel or None

        # locate selected window tuple
        target = None
        for title, w in getattr(self, "_window_scan", []):
            if title == sel:
                target = (title, w)
                break

        if not target:
            self.window_status.config(text="Pick a RuneLite window from the list")
            self._rl_window_rect = None
            self._rl_window_title = None
            return

        title, w = target
        try:
            self._set_runelite_window_rect(w.left, w.top, w.width, w.height)
            self._rl_window_title = title
            self.window_status.config(
                text=f"Using: {title} @ ({w.left},{w.top},{w.width}x{w.height})"
            )
            # ------- NEW: derive username → session dir -------
            username = _username_from_title(title)
            if username:
                p = _session_dir_for_username(username)
                self.session_dir = p
                self.session_dir_var.set(str(p))  # if you still bind in your status panel
                self.session_path_var.set(str(p))  # read-only field in Session section
                self.copy_path_button.state(["!disabled"])
                self.session_status.config(text=f"Session: {username}")

                # ------- NEW: auto-fill IPC port -------
                # 1) best: probe ports by username (requires 'info' support in IPC plugin)
                port = _autofill_port_for_username(username)
                # 2) fallback: instance-index based mapping
                if port is None:
                    idx = getattr(self, "instance_index", 0)  # ensure your InstancesManager sets this
                    port = 17000 + int(idx)

                # only set if empty or different (don't fight user edits)
                try:
                    cur = int(self.ipc_port_var.get().strip())
                except Exception:
                    cur = None
                if cur != port:
                    self.ipc_port_var.set(str(port))
                    # also keep self.ipc in sync if already constructed
                    if hasattr(self, "ipc"):
                        self.ipc.port = port

            self._debug(f"Selected RL window: '{title}' handle={getattr(w, 'hWnd', None)}")

        except Exception as e:
            self._debug(f"select window error: {e}")
            self.window_status.config(text="Failed to read window geometry")

    def _ipc_scan_ports(self, start=17000, end=17020, timeout_s=0.5):
        """
        Probe a small range of ports for a responding IPC plugin by sending {"cmd":"ping"}.
        Populates the combobox with responsive ports.
        """
        import socket, json
        found = []

        def try_ping(port):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as s:
                    s.sendall((json.dumps({"cmd": "ping"}) + "\n").encode("utf-8"))
                    s.shutdown(socket.SHUT_WR)
                    line = s.makefile("r", encoding="utf-8").readline().strip()
                    if line:
                        return json.loads(line)
            except Exception:
                return None
            return None

        for p in range(start, end + 1):
            resp = try_ping(p)
            if isinstance(resp, dict) and resp.get("ok"):
                found.append(str(p))

        if found:
            try:
                self.ipc_combo["values"] = found
                if self.ipc_port_var.get() not in found:
                    self.ipc_port_var.set(found[0])
            except Exception:
                pass
            self.ipc_status.config(text=f"Found {len(found)} IPC port(s): {', '.join(found)}", foreground="#065f46")
            self._debug("IPC scan found ports: " + ", ".join(found))
        else:
            self.ipc_status.config(text="No IPC listeners found in 17000-17020", foreground="#b91c1c")
            self._debug("IPC scan: none found (is the plugin enabled?)")

    def _ipc_ping(self):
        try:
            port = int(self.ipc_port_var.get().strip())
        except Exception:
            self.ipc_status.config(text="Invalid port", foreground="#b91c1c")
            return

        if not hasattr(self, "ipc"):
            self.ipc = RuneLiteIPC(port=port, pre_action_ms=PRE_ACTION_DELAY_MS, timeout_s=1.0)
        else:
            self.ipc.port = port
            self.ipc.timeout_s = 1.0

        pong = self.ipc._send({"cmd": "ping"})
        ok = isinstance(pong, dict) and pong.get("ok")
        self.ipc_status.config(text=f"Ping {port}: {pong}", foreground="#065f46" if ok else "#b91c1c")
        self._debug(f"IPC ping@{port} => {pong}")

    def _bank_count(self, payload: dict, name: str) -> int:
        n = _norm_name(name)
        return sum(
            int(s.get("quantity") or 0)
            for s in _bank_slots(payload)
            if _norm_name(s.get("itemName")) == n
        )

    def _stop_auto_run(self, reason: str):
        """Disable the Auto-Run checkbox and cancel the scheduler."""
        self._debug(f"AUTO-RUN STOPPED: {reason}")
        try:
            self._run_var.set(False)
        except Exception:
            pass
        # this will cancel self._auto_run_after_id if needed
        self._on_run_check()

    def set_session_dir(self, path_str: str):
        """Set the per-instance gamestate directory and sync UI."""
        self.session_dir = path_str
        self.session_dir_var.set(path_str)
        # If your table auto-refresh relies on a watcher/timer, trigger or mark dirty here:
        try:
            self._mark_gamestate_dir_changed()  # optional: if you have such a method
        except Exception:
            pass

    def set_window_title(self, title: str):
        """Reflect chosen target window."""
        self._rl_window_title = title
        self.window_title_var.set(title)

    def get_ipc_port(self) -> int | None:
        """Read the port as int from the free-text field; return None if invalid/empty."""
        s = (self.ipc_port_var.get() or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _current_plan(self):
        return get_plan(self.plan_var.get())
