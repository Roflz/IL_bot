import json
import os
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

AUTO_REFRESH_MS = 100

# ---- helpers to clean RS color tags and format elapsed time ----
_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')

def _clean_rs(s: str | None) -> str:
    if not s:
        return ""
    return _RS_TAG_RE.sub('', s)

def _fmt_age_ms(ms_ago: int) -> str:
    # ms -> human short string
    if ms_ago < 1000:
        return f"{ms_ago} ms ago"
    s = ms_ago / 1000.0
    if s < 60:
        return f"{s:.1f}s ago"
    m = int(s // 60)
    s = int(s % 60)
    return f"{m}m {s}s ago"

def _mk_rect(d):
    """
    Accepts any dict like {"x":..., "y":..., "width":..., "height":...}
    Returns (x, y, w, h) or None if invalid.
    """
    if not isinstance(d, dict):
        return None
    try:
        x = int(d.get("x"))
        y = int(d.get("y"))
        w = int(d.get("width"))
        h = int(d.get("height"))
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except Exception:
        return None

def _rect_contains(rect, px, py):
    """rect=(x,y,w,h), point=(px,py)"""
    if rect is None:
        return False
    x, y, w, h = rect
    return (px >= x) and (py >= y) and (px < x + w) and (py < y + h)

def _center_distance(rect, px, py):
    """smaller is closer to the center of the rect"""
    if rect is None:
        return 1e18
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    dx, dy = (px - cx), (py - cy)
    return (dx * dx + dy * dy) ** 0.5


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
            "last_interaction",
            "menu_entries",
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
        or in the newest session under D:\repos\bot_runelite_IL\data\recording_sessions\*\gamestates
        """
        search_dirs: list[Path] = []

        # prefer the active session folder if present
        if self.session_dir and self.session_dir.exists():
            search_dirs.append(self.session_dir)

        base = Path(r"D:\repos\bot_runelite_IL\data\recording_sessions")
        if base.exists():
            # sessions look like <base>\<YYYYMMDD_HHMMSS>\gamestates
            for run in sorted(base.glob(r"*/gamestates"), key=os.path.getmtime, reverse=True):
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
            for lbl in self.gamestate_info_labels.values():
                lbl.config(text="—")
            return

        try:
            root = json.loads(f.read_text(encoding="utf-8"))
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
            self._last_action_text = self._resolve_last_action(root) or "—"
            self._last_click_epoch_ms = new_epoch

        # Use the persisted text (won’t disappear on later refreshes)
        last_action = self._last_action_text

        # Push to table rows
        self._table_set("bank_open",        "Open" if bank_open else "Closed")
        self._table_set("crafting_open",    "Open" if crafting_open else "Closed")
        self._table_set("crafting_status",  crafting_status)
        self._table_set("inventory_summary",inventory_summary)
        self._table_set("last_click", last_click)
        self._table_set("clicked_target", last_action)
        self._table_set("hovered_tile",     hovered_tile)
        self._table_set("mouse_pos",        mouse_pos)
        self._table_set("last_interaction", last_interaction)
        self._table_set("menu_entries",     menu_entries)


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
