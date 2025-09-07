import json
import os
import re
import time
import random
import pyautogui
import tkinter as tk
import ctypes
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from contextlib import suppress
try:
    import pygetwindow as gw
except Exception:
    gw = None



pyautogui.FAILSAFE = True  # moving mouse to a corner aborts
AUTO_REFRESH_MS = 100
# --- timings (simplified) ---
AUTO_RUN_TICK_MS     = 250    # scheduler tick for the UI/auto loop
PRE_ACTION_DELAY_MS  = 250    # delay before performing any action (click/key)
RULE_WAIT_TIMEOUT_MS = 10_000 # how long to wait for pre/post conditions before retry

# ---- helpers to clean RS color tags and format elapsed time ----
_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')

import socket
import json
import time

class RuneLiteIPC:
    def __init__(self, host="127.0.0.1", port=17000, pre_action_ms=250, timeout_s=2.0):
        self.host = host
        self.port = port
        self.pre_action_ms = pre_action_ms
        self.timeout_s = timeout_s

    def _send(self, obj: dict) -> dict:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
                s.sendall(data)
                s.shutdown(socket.SHUT_WR)
                resp = s.makefile("r", encoding="utf-8").readline().strip()
                return json.loads(resp) if resp else {"ok": False, "err": "empty-response"}
        except socket.timeout:
            return {"ok": False, "err": "timeout"}
        except ConnectionRefusedError:
            return {"ok": False, "err": "connection-refused"}
        except Exception as e:
            return {"ok": False, "err": f"{type(e).__name__}: {e}"}

    def ping(self) -> bool:
        try:
            r = self._send({"cmd":"ping"})
            return bool(r.get("ok"))
        except Exception:
            return False

    def focus(self):
        try:
            self._send({"cmd":"focus"})
        except Exception:
            pass

    def click_canvas(self, x:int, y:int, button:int=1, pre_ms: int | None = None):
        # pre-action delay here (you asked for this explicitly)
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            time.sleep(delay / 1000.0)
        return self._send({"cmd":"click", "x": int(x), "y": int(y), "button": int(button)})

    def key(self, k:str, pre_ms: int | None = None):
        delay = self.pre_action_ms if pre_ms is None else pre_ms
        if delay > 0:
            time.sleep(delay / 1000.0)
        return self._send({"cmd":"key", "k": str(k)})

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

def _compute_phase(payload: dict, craft_recent: bool) -> str:
    bank_open   = bool((payload.get("bank") or {}).get("bankOpen", False))
    craft_open  = bool(payload.get("craftingInterfaceOpen", False))

    has_mould   = _inv_has(payload, "Ring mould")
    has_gold    = _inv_count(payload, "Gold bar") > 0
    has_sapph   = _inv_count(payload, "Sapphire") > 0

    out_of_mats = (not has_gold) or (not has_sapph) or (not has_mould)

    # Your rules, with clear precedence:

    # Banking: highest precedence when open
    if bank_open:
        return "Banking"

    # Crafting: when UI is open OR recently crafting, until mats are gone
    if (craft_open or craft_recent):
        if not out_of_mats:
            return "Crafting"
        # no materials -> next trip is bank
        return "Moving to bank"

    # Moving to bank: missing mats (or no mould)
    if out_of_mats:
        return "Moving to bank"

    # Otherwise we're transitioning toward furnace
    return "Moving to furnace"

def _now_ms() -> int:
    return int(time.time() * 1000)

# Extendable list of known crafting animations
_CRAFT_ANIMS = {899}  # add more if you discover them

def _is_crafting_anim(anim_id: int) -> bool:
    return anim_id in _CRAFT_ANIMS

def _decide_action_from_phase(phase: str) -> str:
    if phase == "Moving to bank":
        return "→ Walk to nearest bank"
    elif phase == "Banking":
        return "→ Deposit outputs, withdraw materials"
    elif phase == "Moving to furnace":
        return "→ Walk to nearest furnace"
    elif phase == "Crafting":
        return "→ Click Make-All / wait for crafting"
    return "—"

def _norm_name(s: str | None) -> str:
    return _clean_rs(s or "").strip().lower()

def _inv_slots(payload: dict) -> list[dict]:
    return (payload.get("inventory", {}) or {}).get("slots", []) or []

def _inv_has(payload: dict, name: str) -> bool:
    n = _norm_name(name)
    return any(_norm_name(s.get("itemName")) == n for s in _inv_slots(payload))

def _inv_count(payload: dict, name: str) -> int:
    n = _norm_name(name)
    return sum(int(s.get("quantity") or 0) for s in _inv_slots(payload)
               if _norm_name(s.get("itemName")) == n)

def _first_inv_slot(payload: dict, name: str) -> dict | None:
    n = _norm_name(name)
    for s in _inv_slots(payload):
        if _norm_name(s.get("itemName")) == n:
            return s
    return None

def _bank_slots(payload: dict) -> list[dict]:
    return (payload.get("bank", {}) or {}).get("slots", []) or []

def _first_bank_slot(payload: dict, name: str) -> dict | None:
    n = _norm_name(name)
    best = None
    for s in _bank_slots(payload):
        if _norm_name(s.get("itemName")) == n:
            # Prefer highest quantity or lowest slotId for determinism
            if best is None:
                best = s
            else:
                q1 = int(s.get("quantity") or 0)
                q2 = int(best.get("quantity") or 0)
                if q1 > q2 or (q1 == q2 and int(s.get("slotId") or 9_999) < int(best.get("slotId") or 9_999)):
                    best = s
    return best

def _unwrap_rect(maybe_rect_dict: dict | None) -> dict | None:
    """
    Inventory/bank slots export as {"bounds": {...}} inside 'bounds'.
    Widgets export as {"bounds": {...}} directly.
    Objects export as 'clickbox' directly.
    This normalizes to a plain {"x","y","width","height"} or None.
    """
    if not isinstance(maybe_rect_dict, dict):
        return None
    # common cases
    if {"x","y","width","height"} <= set(maybe_rect_dict.keys()):
        return maybe_rect_dict
    inner = maybe_rect_dict.get("bounds")
    if isinstance(inner, dict) and {"x","y","width","height"} <= set(inner.keys()):
        return inner
    return None

def _rect_center_xy(rect: dict | None) -> tuple[int | None, int | None]:
    if not rect:
        return None, None
    try:
        return int(rect["x"] + rect["width"]/2), int(rect["y"] + rect["height"]/2)
    except Exception:
        return None, None

def _closest_object_by_names(payload: dict, names: list[str]) -> dict | None:
    wanted = [n.lower() for n in names]
    for obj in (payload.get("closestGameObjects") or []):
        nm = _norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj
    return None

def _craft_widget_rect(payload: dict, key: str) -> dict | None:
    w = (payload.get("crafting_widgets", {}) or {}).get(key)
    return _unwrap_rect((w or {}).get("bounds") if isinstance(w, dict) else None)

# --- Windows POINT struct (module-level) ---
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _build_action_plan(payload: dict, phase: str) -> dict:
    """
    Returns a structured plan:
    {
      "phase": "Crafting",
      "steps": [ {action-object}, ... ]
    }
    No side-effects. Purely derives targets from gamestate.
    """
    plan = {"phase": phase, "steps": []}

    # ---------- MOVING TO BANK ----------
    if phase == "Moving to bank":
        obj = _closest_object_by_names(payload, ["bank booth", "banker"])
        if obj:
            rect = _unwrap_rect(obj.get("clickbox"))
            cx, cy = _rect_center_xy(rect)
            step = {
                "action": "click-bank",
                "description": "Click nearest bank booth",
                "click": (
                    {"type": "rect-center"}
                    if rect else
                    {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}
                ),
                "target": {
                    "domain": "object",
                    "name": obj.get("name"),
                    "id": obj.get("id"),
                    "clickbox": rect,
                    "canvas": {"x": obj.get("canvasX"), "y": obj.get("canvasY")}
                },
                "preconditions": ["bankOpen == false"],
                "postconditions": ["bankOpen == true"],
                "confidence": 0.92 if rect else 0.6
            }
            plan["steps"].append(step)
        return plan

    # ---------- BANKING ----------
    if phase == "Banking":
        # Target counts (tune to taste)
        TARGET_SAPP = 13
        TARGET_GOLD = 13

        inv_sapp = _inv_count(payload, "Sapphire")
        inv_gold = _inv_count(payload, "Gold bar")
        has_mould = _inv_has(payload, "Ring mould")
        inv_ring = _first_inv_slot(payload, "Sapphire ring")

        # B1: Deposit outputs first
        if inv_ring:
            rect = _unwrap_rect(inv_ring.get("bounds"))
            plan["steps"].append({
                "action": "deposit-inventory-item",
                "description": "Deposit Sapphire ring from inventory",
                "click": {"type": "rect-center"} if rect else {"type": "none"},
                "target": {
                    "domain": "inventory",
                    "name": "Sapphire ring",
                    "slotId": inv_ring.get("slotId"),
                    "bounds": rect
                },
                "preconditions": ["bankOpen == true", "inventory contains 'Sapphire ring'"],
                "postconditions": ["inventory does not contain 'Sapphire ring'"],
                "confidence": 0.9 if rect else 0.4,
            })
            return plan  # single-step plan

        # B2: Top off Sapphires to TARGET_SAPP
        if inv_sapp < TARGET_SAPP:
            bank_sapp = _first_bank_slot(payload, "Sapphire")
            if bank_sapp:
                rect = _unwrap_rect(bank_sapp.get("bounds"))
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": f"Withdraw Sapphires (need {TARGET_SAPP - inv_sapp} more)",
                    "click": {"type": "rect-center"} if rect else {"type": "none"},
                    "target": {
                        "domain": "bank",
                        "name": "Sapphire",
                        "slotId": bank_sapp.get("slotId"),
                        "bounds": rect
                    },
                    "preconditions": ["bankOpen == true", f"inventory count('Sapphire') < {TARGET_SAPP}"],
                    "postconditions": [f"inventory count('Sapphire') >= {TARGET_SAPP}"],
                    "confidence": 0.9 if rect else 0.4,
                })
                return plan
            else:
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": "Could not find Sapphires in bank (scroll/search may be required)",
                    "click": {"type": "none"},
                    "target": {"domain": "bank", "name": "Sapphire"},
                    "preconditions": ["bankOpen == true"],
                    "postconditions": [],
                    "confidence": 0.0
                })
                return plan

        # B3: Top off Gold bars to TARGET_GOLD
        if inv_gold < TARGET_GOLD:
            bank_gold = _first_bank_slot(payload, "Gold bar")
            if bank_gold:
                rect = _unwrap_rect(bank_gold.get("bounds"))
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": f"Withdraw Gold bars (need {TARGET_GOLD - inv_gold} more)",
                    "click": {"type": "rect-center"} if rect else {"type": "none"},
                    "target": {
                        "domain": "bank",
                        "name": "Gold bar",
                        "slotId": bank_gold.get("slotId"),
                        "bounds": rect
                    },
                    "preconditions": ["bankOpen == true", f"inventory count('Gold bar') < {TARGET_GOLD}"],
                    "postconditions": [f"inventory count('Gold bar') >= {TARGET_GOLD}"],
                    "confidence": 0.9 if rect else 0.4,
                })
                return plan
            else:
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": "Could not find Gold bars in bank (scroll/search may be required)",
                    "click": {"type": "none"},
                    "target": {"domain": "bank", "name": "Gold bar"},
                    "preconditions": ["bankOpen == true"],
                    "postconditions": [],
                    "confidence": 0.0
                })
                return plan

        # B4: Ensure Ring mould present
        if not has_mould:
            bank_mould = _first_bank_slot(payload, "Ring mould")
            if bank_mould:
                rect = _unwrap_rect(bank_mould.get("bounds"))
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": "Withdraw Ring mould",
                    "click": {"type": "rect-center"} if rect else {"type": "none"},
                    "target": {
                        "domain": "bank",
                        "name": "Ring mould",
                        "slotId": bank_mould.get("slotId"),
                        "bounds": rect
                    },
                    "preconditions": ["bankOpen == true", "inventory does not contain 'Ring mould'"],
                    "postconditions": ["inventory contains 'Ring mould'"],
                    "confidence": 0.9 if rect else 0.4,
                })
                return plan
            else:
                plan["steps"].append({
                    "action": "withdraw-item",
                    "description": "Could not find Ring mould in bank (scroll/search may be required)",
                    "click": {"type": "none"},
                    "target": {"domain": "bank", "name": "Ring mould"},
                    "preconditions": ["bankOpen == true"],
                    "postconditions": [],
                    "confidence": 0.0
                })
                return plan

        # B5: Close bank when all conditions satisfied
        steps = {
            "action": "close-bank",
            "description": "Close bank with ESC",
            "click": {"type": "key", "key": "ESC"},
            "target": {"domain": "widget", "name": "bank_close"},
            "preconditions": [
                "bankOpen == true",
                "inventory contains 'Ring mould'",
                f"inventory count('Sapphire') >= {TARGET_SAPP}",
                f"inventory count('Gold bar') >= {TARGET_GOLD}",
                "inventory does not contain 'Sapphire ring'"
            ],
            "postconditions": ["bankOpen == false"],
            "confidence": 0.95
        }
        plan["steps"].append(steps)
        return plan

    # ---------- MOVING TO FURNACE ----------
    if phase == "Moving to furnace":
        obj = _closest_object_by_names(payload, ["furnace"])
        if obj:
            rect = _unwrap_rect(obj.get("clickbox"))
            cx, cy = _rect_center_xy(rect)
            step = {
                "action": "click-furnace",
                "description": "Click nearest furnace",
                "click": (
                    {"type": "rect-center"}
                    if rect else
                    {"type": "point", "x": int(obj.get("canvasX") or 0), "y": int(obj.get("canvasY") or 0)}
                ),
                "target": {
                    "domain": "object",
                    "name": obj.get("name"),
                    "id": obj.get("id"),
                    "clickbox": rect,
                    "canvas": {"x": obj.get("canvasX"), "y": obj.get("canvasY")}
                },
                "preconditions": [
                    "bankOpen == false",
                    "inventory contains 'Ring mould'",
                    "inventory count('Sapphire') > 0",
                    "inventory count('Gold bar') > 0"
                ],
                "postconditions": ["craftingInterfaceOpen == true"],
                "confidence": 0.92 if rect else 0.6
            }
            plan["steps"].append(step)
        return plan

    # ---------- CRAFTING ----------
    if phase == "Crafting":
        # C1: click make widget (when visible)
        make_rect = _craft_widget_rect(payload, "make_sapphire_rings")
        plan["steps"].append({
            "action": "click-make-widget",
            "description": "Click the 'Make sapphire rings' button",
            "click": {"type": "rect-center"} if make_rect else {"type": "none"},
            "target": {
                "domain": "widget",
                "name": "make_sapphire_rings",
                "bounds": make_rect
            },
            "preconditions": [
                "craftingInterfaceOpen == true",
                "inventory count('Sapphire') > 0",
                "inventory count('Gold bar') > 0"
            ],
            "postconditions": [
                "player.animation == 899 OR crafting in progress"
            ],
            "confidence": 0.95 if make_rect else 0.4
        })

        # C2: wait until materials gone
        plan["steps"].append({
            "action": "wait-crafting-complete",
            "description": "Wait until sapphires and gold bars are consumed",
            "click": {"type": "none"},
            "target": {"domain": "none", "name": "crafting_wait"},
            "preconditions": [
                "inventory count('Sapphire') > 0",
                "inventory count('Gold bar') > 0"
            ],
            "postconditions": [
                "inventory count('Sapphire') == 0 OR inventory count('Gold bar') == 0"
            ],
            "confidence": 1.0
        })
        return plan

    # Fallback (unknown phase)
    plan["steps"].append({
        "action": "idle",
        "description": "No actionable step for this phase",
        "click": {"type": "none"},
        "target": {"domain": "none", "name": "n/a"},
        "preconditions": [],
        "postconditions": [],
        "confidence": 0.0
    })
    return plan


class SimpleRecorderWindow(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=12)
        self.root = root
        self.root.title("Simple Bot Recorder")
        self.root.minsize(1100, 600)
        self._rl_window_rect = None  # {'x': int, 'y': int, 'width': int, 'height': int}
        self._rl_window_title = None

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
        ipc = ttk.LabelFrame(left, text="IPC (RuneLite Plugin)")
        ipc.grid(row=2, column=0, sticky="ew", pady=6)
        ipc.grid_columnconfigure(0, weight=1)
        ipc.grid_columnconfigure(1, weight=0)

        ttk.Label(ipc, text="Port:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.ipc_port_var = tk.StringVar(value="17000")
        self.ipc_combo = ttk.Combobox(ipc, state="readonly", textvariable=self.ipc_port_var, values=["17000"])
        self.ipc_combo.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        btns = ttk.Frame(ipc)
        btns.grid(row=1, column=1, sticky="e", padx=(0, 8), pady=(0, 6))
        ttk.Button(btns, text="Scan", command=self._ipc_scan_ports).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="Ping", command=self._ipc_ping).grid(row=0, column=1)
        self.ipc_status = ttk.Label(ipc, text="Not tested", foreground="#6b7280")
        self.ipc_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        # Input mode toggle
        self.input_mode_var = getattr(self, "input_mode_var", None) or tk.StringVar(value="ipc")
        ttk.Label(ipc, text="Input mode:").grid(row=3, column=0, sticky="w", padx=8, pady=(4, 0))
        mode_row = ttk.Frame(ipc)
        mode_row.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        ttk.Radiobutton(mode_row, text="IPC", variable=self.input_mode_var, value="ipc").grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(mode_row, text="pyautogui", variable=self.input_mode_var, value="pyautogui").grid(row=0, column=1)


        # Session Management
        sess = ttk.LabelFrame(left, text="Session Management")
        sess.grid(row=3, column=0, sticky="ew", pady=6)
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
        rec.grid(row=4, column=0, sticky="ew", pady=6)
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
        self.recording_status.grid(row=5, column=0, sticky="w", pady=(4, 0))

        # ---- Debug Mouse Move ----
        dbg = ttk.LabelFrame(left, text="Debug Mouse Move")
        dbg.grid(row=6, column=0, sticky="ew", pady=6, padx=6)
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
        try:
            self.refresh_gamestate_info()
        except Exception as e:
            # Never let a single bad refresh kill the loop
            self._debug(f"refresh_gamestate_info error: {e}")
        finally:
            self._after_id = self.root.after(AUTO_REFRESH_MS, self._schedule_auto_refresh)

    def _latest_gamestate_file(self) -> Path | None:
        search_dirs: list[Path] = []

        if self.session_dir and self.session_dir.exists():
            search_dirs.append(self.session_dir)

        base = Path(r"D:\repos\bot_runelite_IL\data\recording_sessions")
        if base.exists():
            for run in sorted(base.glob(r"*/gamestates"), key=os.path.getmtime, reverse=True):
                search_dirs.append(run)

        newest: tuple[float, Path] | None = None
        for d in search_dirs:
            for f in d.glob("*.json"):
                try:
                    ts = f.stat().st_mtime
                except FileNotFoundError:
                    continue  # file got pruned between glob and stat – skip it
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

        # Determine Phase
        phase = _compute_phase(payload, craft_recent)
        action = _decide_action_from_phase(phase)

        prev_phase = getattr(self, "_prev_phase_for_wait", None)
        if phase != prev_phase:
            self._plan_ready_ms = _now_ms() + PRE_ACTION_DELAY_MS
            self._prev_phase_for_wait = phase

        # Build standardized plan (no side-effects)
        plan = _build_action_plan(payload, phase)

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

    def _ensure_ipc(self):
        # helper: keep RuneLiteIPC in sync with the UI selection
        try:
            port = int(self.ipc_port_var.get().strip())
        except Exception:
            port = 17000
        if not hasattr(self, "ipc"):
            self.ipc = RuneLiteIPC(port=port, pre_action_ms=250, timeout_s=2.0)
        else:
            self.ipc.port = port

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
                self._debug(
                    f"PYAUTOGUI click -> canvas=({int(x)},{int(y)}) system=({sys_x},{sys_y}) "
                    f"window='{title}' RLRect={rl_rect}"
                )
            except Exception as e:
                self._debug(f"PYAUTOGUI click error: {type(e).__name__}: {e}")
            return

        # ----- IPC mode -----
        self._ensure_ipc()
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
        self._debug(
            f"IPC click (port={self.ipc.port}) -> canvas=({int(x)},{int(y)}) system=({sys_x},{sys_y}) "
            f"window='{title}' RLRect={rl_rect} resp={resp}"
        )
        try:
            self.ipc_status.config(text=f"Click resp @ {self.ipc.port}: {resp}", foreground="#065f46" if resp.get("ok") else "#b91c1c")
        except Exception:
            pass


    def _do_press_key(self, key: str):
        mode = self.input_mode_var.get() if hasattr(self, "input_mode_var") else "ipc"

        if mode == "pyautogui":
            try:
                pyautogui.press(str(key))
                self._debug(f"PYAUTOGUI key -> '{key}'")
            except Exception as e:
                self._debug(f"PYAUTOGUI key error: {type(e).__name__}: {e}")
            return

        # ----- IPC mode -----
        self._ensure_ipc()
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
        self._debug(f"IPC key (port={self.ipc.port}) -> '{key}' window='{title}' RLRect={rl_rect} resp={resp}")
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
        ctype = click.get("type")

        try:
            # log geometry each action
            self._log_window_geometry()

            if ctype in ("rect-center", "rect-random"):
                rect = target.get("bounds") or target.get("clickbox")
                rect = rect if isinstance(rect, dict) else None
                jitter_px = int(click.get("jitter_px") or 0) if ctype == "rect-random" else None
                px, py = self._screen_point_from_rect(rect, jitter_px)
                px, py = self._apply_canvas_offset(px, py)

                if px is None:
                    self._table_set("next_action", "No rect available for click")
                    self._log_click_debug(step, ctype, None, None, rect, self._canvas_offset,
                                          sys_before=self._get_system_cursor_pos(),
                                          sys_after=None,
                                          note="(no rect)")
                    return

                sys_before = self._get_system_cursor_pos()
                self._do_click_point(px, py)
                sys_after = self._get_system_cursor_pos()

                self._log_click_debug(step, ctype, px, py, rect, self._canvas_offset,
                                      sys_before=sys_before, sys_after=sys_after,
                                      note=f"(jitter={jitter_px})" if jitter_px else "")

            elif ctype == "point":
                px, py = click.get("x"), click.get("y")
                px, py = self._apply_canvas_offset(px, py)
                if not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
                    self._table_set("next_action", "Invalid point for click")
                    self._log_click_debug(step, ctype, None, None, None, self._canvas_offset,
                                          sys_before=self._get_system_cursor_pos(),
                                          sys_after=None,
                                          note="(invalid point)")
                    return

                sys_before = self._get_system_cursor_pos()
                self._do_click_point(int(px), int(py))
                sys_after = self._get_system_cursor_pos()

                self._log_click_debug(step, ctype, int(px), int(py), None, self._canvas_offset,
                                      sys_before=sys_before, sys_after=sys_after)

            elif ctype == "key":
                key = (click.get("key") or "").lower()
                if not key:
                    self._table_set("next_action", "Invalid key")
                    self._debug("Keypress skipped: invalid key")
                    return

                sys_before = self._get_system_cursor_pos()
                self._debug(
                    f"Phase='{getattr(self, '_last_phase', '?')}' action='{step.get('action')}' press key='{key}' "
                    f"sysBefore={sys_before}")
                self._do_press_key(key)
                sys_after = self._get_system_cursor_pos()
                # also log window rects for key actions
                self._log_window_geometry()

            else:
                self._table_set("next_action", f"Unsupported click.type: {ctype}")
                self._debug(f"Unsupported click.type: {ctype}")
                return

            # After action, refresh so next step is proposed
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

    def _log_click_debug(self, step, ctype, px=None, py=None, rect=None, used_offset=(0, 0),
                         sys_before=None, sys_after=None, note=""):
        tgt = (step.get("target") or {})
        tname = tgt.get("name") or tgt.get("domain") or "target"
        phase = getattr(self, "_last_phase", "?")
        rect_src = "bounds" if "bounds" in tgt else ("clickbox" if "clickbox" in tgt else "point")

        sys_planned = self._canvas_to_system(px, py)
        win_title = self._rl_window_title or "?"
        rl_rect = self._rl_window_rect

        self._debug(
            f"Phase='{phase}' action='{step.get('action')}' target='{tname}' "
            f"type={ctype} canvas=({px},{py}) system={sys_planned} rectSrc={rect_src} rect={rect} "
            f"offset={used_offset} window='{win_title}' RLRect={rl_rect} "
            f"sysBefore={sys_before} sysAfter={sys_after} {note}"
        )

    # --- Windows cursor position (absolute screen coords) ---
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

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

    # --- Debug dump of window geometry + offsets ---
    def _log_window_geometry(self):
        tk_rect = self._get_tk_window_rect()
        rl_rect = self._rl_window_rect if isinstance(self._rl_window_rect, dict) else None
        self._debug(
            "WindowRects  TK="
            f"{tk_rect}  RL={rl_rect}  canvas_offset={self._canvas_offset}"
        )

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
                self._debug(
                    f"plan/phase changed → reset step (from '{curr_action}' to '{new_action}', phase '{curr_phase}'→'{new_phase}')")
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
        self._debug(f"step begin → {step.get('action')} (phase={getattr(self, '_plan_phase', None)})")

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
            pre = step.get("preconditions")
            pre_defined = bool(pre)
            pre_ok = self._rules_ok(pre, data) if pre_defined else True

            self._debug(f"PREWAIT stage='{stage}' action='{step.get('action')}' "
                        f"delay_ok={delay_ok} elapsed={now - t0}ms")
            if pre_defined:
                _dump_rules("preconditions status", pre)

            if delay_ok:
                st["stage"] = "acting"
            return False

        # 2) ACTING: perform once, then go to postwait
        if stage == "acting":
            self._debug(f"ACTING action='{step.get('action')}'")
            self._execute_next_action()  # already refreshes
            st["stage"] = "postwait"
            st["t0"] = now
            return False

        # 3) POSTWAIT: wait for postconditions; on timeout, retry the action
        if stage == "postwait":
            post = step.get("postconditions")
            post_ok = self._rules_ok(post, data)
            remain = max(0, RULE_WAIT_TIMEOUT_MS - (now - t0))

            self._debug(f"POSTWAIT action='{step.get('action')}' post_ok={post_ok} "
                        f"elapsed={now - t0}ms remaining_to_retry={remain}ms")
            _dump_rules("postconditions status", post)

            if post_ok:
                self._debug("postconditions satisfied → advance")
                self._step_state = None
                self._mark_action_done()  # uses PRE_ACTION_DELAY_MS
                return True

            if (now - t0) >= RULE_WAIT_TIMEOUT_MS:
                self._debug("postwait timeout → retrying action now")
                self._execute_next_action()
                st["t0"] = now
                return False

            # keep waiting
            return False

        # unknown stage: finish
        self._debug(f"Unknown stage '{stage}' → finishing step")
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
            self._debug("pygetwindow not available; install: pip install pygetwindow")
            return []

        try:
            # pull everything once; titles may include IDEs etc.
            wins = gw.getAllWindows()
        except Exception as e:
            self._debug(f"scan windows error: {e}")
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
        """
        Called when user picks a 'RuneLite - <username>' entry in the combo.
        Caches rect/title for coord translation and debug.
        """
        try:
            sel = self.window_combo.get().strip()
        except Exception:
            sel = ""

        self._window_selected_title = sel or None

        # find the matching tuple
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
            # pygetwindow returns .left/.top/.width/.height
            self._set_runelite_window_rect(w.left, w.top, w.width, w.height)
            self._rl_window_title = title
            self.window_status.config(
                text=f"Using: {title} @ ({w.left},{w.top},{w.width}x{w.height})"
            )
            self._log_window_geometry()
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
