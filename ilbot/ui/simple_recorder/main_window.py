import json
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

AUTO_REFRESH_MS = 1000  # 1s

class SimpleRecorderWindow(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=12)
        self.root = root
        self.root.title("Simple Bot Recorder")
        self.root.minsize(1100, 600)

        # state
        self.session_dir: Path | None = None
        self.auto_refresh_on = True
        self._after_id = None

        # overall grid (two columns)
        self.grid(sticky="nsew")
        self.grid_columnconfigure(0, weight=1)                   # controls
        self.grid_columnconfigure(1, weight=0, minsize=400)      # info panel width
        self.grid_rowconfigure(0, weight=1)

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

        self.detect_button = ttk.Button(win, text="Detect Runelite Window", command=self.detect_window)
        self.detect_button.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        self.window_status = ttk.Label(win, text="No window detected", foreground="#2c7a7b")
        self.window_status.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

        # Session Management
        sess = ttk.LabelFrame(left, text="Session Management")
        sess.grid(row=2, column=0, sticky="ew", pady=6)
        sess.grid_columnconfigure(0, weight=1)
        sess.grid_columnconfigure(1, weight=1)

        self.create_session_button = ttk.Button(sess, text="Create Session", command=self.create_session)
        self.create_session_button.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        self.copy_path_button = ttk.Button(sess, text="Copy Gamestates Path", command=self.copy_gamestates_path)
        self.copy_path_button.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.copy_path_button.state(["disabled"])

        self.session_status = ttk.Label(sess, text="No session", foreground="#2f855a")
        self.session_status.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        # Recording Controls
        rec = ttk.LabelFrame(left, text="Recording Controls")
        rec.grid(row=3, column=0, sticky="ew", pady=6)
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
        self.recording_status.grid(row=4, column=0, sticky="w", pady=(4, 0))

    def _build_info_panel(self):
        right = ttk.LabelFrame(self, text="Live Gamestate Information")
        right.grid(row=0, column=1, sticky="ns")

        info = ttk.Frame(right, padding=10)
        info.grid(sticky="n")
        info.grid_columnconfigure(0, weight=0)
        info.grid_columnconfigure(1, weight=1)

        rows = [
            ("Bank:",               "bank_open"),
            ("Crafting UI:",        "crafting_open"),
            ("Crafting Status:",    "crafting_status"),
            ("Inventory:",          "inventory_summary"),
            ("Hovered Tile:",       "hovered_tile"),
            ("Last Interaction:",   "last_interaction"),
            ("Menu Entries:",       "menu_entries"),
        ]

        self.gamestate_info_labels: dict[str, ttk.Label] = {}
        for r, (label_text, key) in enumerate(rows):
            ttk.Label(info, text=label_text, font=("Segoe UI", 10, "bold")).grid(
                row=r, column=0, sticky="w", padx=(0, 8), pady=2
            )
            lbl = ttk.Label(info, text="—", wraplength=280, justify="left")
            lbl.grid(row=r, column=1, sticky="w", pady=2)
            self.gamestate_info_labels[key] = lbl

        btns = ttk.Frame(right, padding=(10, 2, 10, 10))
        btns.grid(sticky="ew")
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.refresh_btn = ttk.Button(btns, text="Refresh Now", command=self.refresh_gamestate_info)
        self.toggle_btn  = ttk.Button(btns, text="Auto: On", command=self._toggle_auto_refresh)
        self.refresh_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.toggle_btn.grid( row=0, column=1, sticky="ew", padx=(6, 0))

    # ---------------- Actions / hooks ----------------

    def detect_window(self):
        # TODO: your existing detection
        self.window_status.config(text="Window detected")
        # (no change to info panel)

    def create_session(self):
        """
        Set the session dir (where gamestate .json files are written).
        Keep your existing logic; this version just creates a dated folder under ./data/recording_sessions/
        """
        from pathlib import Path

        base = Path(r"D:\\repos\\bot_runelite_IL\\data\\recording_sessions")
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = base / stamp / "gamestates"
        self.session_dir.mkdir(parents=True, exist_ok=True)
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
        self.refresh_gamestate_info()
        self._after_id = self.root.after(AUTO_REFRESH_MS, self._schedule_auto_refresh)

    def _latest_gamestate_file(self) -> Path | None:
        """
        Find the newest *.json file either in the current session_dir (if set),
        or in the newest session under data/recording_sessions/**/gamestates.
        """
        search_dirs: list[Path] = []
        if self.session_dir and self.session_dir.exists():
            search_dirs.append(self.session_dir)
        from pathlib import Path

        base = Path(r"D:\\repos\\bot_runelite_IL\\data\\recording_sessions")
        if base.exists():
            for run in sorted(base.glob("*/*/gamestates"), key=os.path.getmtime, reverse=True):
                search_dirs.append(run)
        if base.exists():
            for run in sorted(base.glob("*/*/gamestates"), key=os.path.getmtime, reverse=True):
                search_dirs.append(run)

        newest: tuple[float, Path] | None = None
        for d in search_dirs:
            for f in d.glob("*.json"):
                ts = f.stat().st_mtime
                if newest is None or ts > newest[0]:
                    newest = (ts, f)
        return newest[1] if newest else None

    def refresh_gamestate_info(self):
        f = self._latest_gamestate_file()
        if not f:
            # show blanks
            for k, lbl in self.gamestate_info_labels.items():
                lbl.config(text="—")
            return

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return

        # ---- map your exported JSON to UI rows (adjust keys to your schema) ----
        # Booleans
        bank_open      = bool(data.get("bank", {}).get("open"))
        crafting_open  = bool(data.get("crafting", {}).get("open"))

        # Crafting status via animation id (-1 idle, 899 crafting)
        anim_id = int(data.get("player", {}).get("animationId", -1))
        crafting_status = "Idle (anim=-1)" if anim_id == -1 else ("Crafting (anim=899)" if anim_id == 899 else f"Anim {anim_id}")

        # Inventory summary: "Ring mould: 1, Sapphire: 13, Gold bar: 13"
        inv_slots = data.get("inventory", {}).get("slots", [])
        name_counts = {}
        for s in inv_slots:
            n = s.get("itemName")
            q = int(s.get("quantity", 0) or 0)
            if n:
                name_counts[n] = name_counts.get(n, 0) + q
        inventory_summary = ", ".join(f"{k}: {v}" for k, v in sorted(name_counts.items())) or "0"

        # Hovered tile
        hovered_tile = tuple(data.get("hover", {}).get("tile", [])) or "—"

        # Last interaction + menu entries
        last_interaction = data.get("interaction", {}).get("last", "—")
        menu_entries = ", ".join(data.get("menu", {}).get("entries", [])[:4])  # first few

        # Update labels
        self.gamestate_info_labels["bank_open"].config(text="Open" if bank_open else "Closed")
        self.gamestate_info_labels["crafting_open"].config(text="Open" if crafting_open else "Closed")
        self.gamestate_info_labels["crafting_status"].config(text=crafting_status)
        self.gamestate_info_labels["inventory_summary"].config(text=inventory_summary)
        self.gamestate_info_labels["hovered_tile"].config(text=str(hovered_tile))
        self.gamestate_info_labels["last_interaction"].config(text=str(last_interaction))
        self.gamestate_info_labels["menu_entries"].config(text=menu_entries or "—")
