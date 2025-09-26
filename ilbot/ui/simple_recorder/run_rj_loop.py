# run_rj_loop.py
# Standalone runner for immediate-mode plans without main_window.py

import json
import threading
import time
import argparse
import sys
from pathlib import Path

from ilbot.ui.simple_recorder.helpers.context import set_payload, set_ui, get_payload
from ilbot.ui.simple_recorder.services.ipc_client import RuneLiteIPC
from ilbot.ui.simple_recorder.actions.timing import wait_until

# Import all available plans
from ilbot.ui.simple_recorder.plans.romeo_and_juliet import RomeoAndJulietPlan
from ilbot.ui.simple_recorder.plans.goblin_diplomacy import GoblinDiplomacyPlan
from ilbot.ui.simple_recorder.plans.tutorial_island import TutorialIslandPlan

# Plan registry - add new plans here
AVAILABLE_PLANS = {
    "romeo_and_juliet": RomeoAndJulietPlan,
    "goblin_diplomacy": GoblinDiplomacyPlan,
    "tutorial_island": TutorialIslandPlan,
    # Add more plans here as you create them
    # "cook_assistant": CookAssistantPlan,
    # "sheep_shearer": SheepShearerPlan,
}


def get_plan_class(plan_name: str):
    """Get plan class by name."""
    plan_class = AVAILABLE_PLANS.get(plan_name.lower())
    if plan_class is None:
        available = ", ".join(AVAILABLE_PLANS.keys())
        raise ValueError(f"Unknown plan '{plan_name}'. Available plans: {available}")
    return plan_class


def list_available_plans():
    """List all available plans with their descriptions."""
    print("Available plans:")
    for name, plan_class in AVAILABLE_PLANS.items():
        # Create a temporary instance to get the label
        try:
            temp_plan = plan_class()
            label = getattr(temp_plan, 'label', 'No description')
            print(f"  {name}: {label}")
        except Exception as e:
            print(f"  {name}: Error creating plan - {e}")


class UIShim:
    """
    Minimal UI adapter exposing:
      - debug(msg)
      - latest_payload()
      - dispatch(step_dict)   # executes immediately via IPC
    """

    def __init__(self, session_dir: str, port: int, canvas_offset=(0, 0)):
        self.session_dir = Path(session_dir)
        self.canvas_offset = tuple(canvas_offset or (0, 0))
        self.ipc = RuneLiteIPC(port=port, pre_action_ms=120, timeout_s=1.0)

        self._last_mtime = 0.0
        self._stop_refresher = False
        self._refresher = threading.Thread(target=self._payload_refresher, daemon=True)
        self._refresher.start()

    # ---------- utilities ----------
    def _payload_refresher(self):
        """Continuously update global payload when a newer gamestate JSON appears."""
        while not self._stop_refresher:
            try:
                d = self.session_dir
                if d.exists():
                    newest_f = None
                    newest_ts = self._last_mtime
                    for f in d.glob("*.json"):
                        try:
                            ts = f.stat().st_mtime
                        except FileNotFoundError:
                            continue
                        if ts > newest_ts:
                            newest_ts = ts
                            newest_f = f
                    if newest_f is not None:
                        # small debounce to let writer finish
                        time.sleep(0.01)
                        try:
                            root = json.loads(newest_f.read_text(encoding="utf-8"))
                            payload = (root.get("data") or {}) if isinstance(root, dict) else {}
                            # ensure __ipc is present, same as your normalize pattern
                            try:
                                payload["__ipc"] = self.ipc
                                payload["__ipc_port"] = getattr(self.ipc, "port", 0)
                            except Exception:
                                pass
                            set_payload(payload)
                            self._last_mtime = newest_ts
                        except Exception:
                            pass
            except Exception:
                pass
            # poll interval — tune as you like (20–50ms is typical)
            time.sleep(0.03)

    def normalize_payload(self, root: dict, ipc) -> dict:
        base = (root.get("data") if isinstance(root, dict) else None) or root or {}
        payload = dict(base)  # shallow copy
        payload["__ipc"] = ipc
        try:
            payload["__ipc_port"] = ipc.port if ipc else 0
        except Exception:
            pass
        return payload

    def debug(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    def latest_payload(self) -> dict:
        d = self.session_dir
        if not d.exists():
            return {}
        newest = None
        for f in d.glob("*.json"):
            try:
                ts = f.stat().st_mtime
            except FileNotFoundError:
                continue
            if newest is None or ts > newest[0]:
                newest = (ts, f)
        if not newest:
            return {}
        try:
            root = json.loads(newest[1].read_text(encoding="utf-8"))
            return self.normalize_payload(root, self.ipc)
        except Exception:
            raise(Exception)

    # ---------- step execution ----------
    def dispatch(self, step: dict):
        """
        Execute a single action step immediately.
        Supports: point, rect-center, rect-random, key, key-hold, type, scroll, wait.
        Returns lastInteraction data for click actions.
        """
        # NEW: allow None and sequences of steps
        if step is None:
            return
        if isinstance(step, (list, tuple)):
            for s in step:
                return self.dispatch(s)
            return

        if not isinstance(step, dict):
            # optional: self.debug("[STEP] non-dict step; skipping")
            return

        click = (step.get("click") or {})
        target = (step.get("target") or {})
        ctype = (click.get("type") or "").lower()
        

        # 'wait' doesn't require IPC readiness
        if ctype in ("point", "rect-center", "rect-random", "key", "key-hold", "keyhold", "type", "scroll"):
            if not self._ipc_ready():
                self.debug("[IPC] not ready; skipping step")
                return

        if ctype in ("rect-center", "rect-random"):
            rect = target.get("bounds") or target.get("clickbox")
            cx, cy = self._rect_center(rect)
            if cx is None:
                self.debug("[STEP] rect click: no bounds; skipping")
                return
            if ctype == "rect-random":
                try:
                    x = int(rect.get("x"));
                    y = int(rect.get("y"))
                    w = int(rect.get("width"));
                    h = int(rect.get("height"))
                    if w > 2 and h > 2:
                        import random
                        cx = x + random.randint(1, w - 2)
                        cy = y + random.randint(1, h - 2)
                except Exception:
                    pass
            return self._click_canvas(cx, cy)

        if ctype in ("point", "canvas-point", "canvas_point"):
            x, y = click.get("x"), click.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                x, y = int(x), int(y)
                x += int(self.canvas_offset[0]);
                y += int(self.canvas_offset[1])
                return self._click_canvas(x, y)
            else:
                self.debug("[STEP] point: invalid coords; skipping")
            return

        if ctype in ("key-hold", "keyhold"):
            key = (click.get("key") or "")
            dur = int(click.get("ms") or 180)
            if key:
                try:
                    self.ipc.focus()
                    self.ipc.key_hold(key, dur)
                except Exception as e:
                    self.debug(f"[IPC] key-hold error: {e}")
            return

        if ctype == "key":
            key = (click.get("key") or "")
            if key:
                try:
                    self.ipc.focus()
                    self.ipc.key(key if len(key) > 1 else key)
                except Exception as e:
                    self.debug(f"[IPC] key error: {e}")
            return

        if ctype == "type":
            text = (click.get("text") or "")
            per_ms = int(click.get("per_char_ms", 20))
            if text:
                try:
                    self.ipc.type(text, per_char_ms=per_ms)
                except Exception as e:
                    self.debug(f"[IPC] type error: {e}")
            return

        if ctype == "scroll":
            amt = int(click.get("amount", 0))
            try:
                self.ipc.focus()
                self.ipc.scroll(amt)
            except Exception as e:
                self.debug(f"[IPC] scroll error: {e}")
            return

        # --- right click at a point or rect center ---
        if ctype == "right-click":
            # anchor from rect bounds or explicit point
            rect = target.get("bounds") or target.get("clickbox")
            if rect:
                cx, cy = self._rect_center(rect)
                if cx is None:
                    self.debug("[STEP] right-click: no bounds; skipping")
                    return
                return self._click_canvas(cx, cy, button="right")

            x, y = click.get("x"), click.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                x, y = int(x), int(y)
                x += int(self.canvas_offset[0]); y += int(self.canvas_offset[1])
                return self._click_canvas(x, y, button="right")
            else:
                self.debug("[STEP] right-click: invalid coords; skipping")
            return

        # --- context menu select via live menu entries ---
        if ctype == "context-select":
            # 1) require explicit right-click anchor (canvas coords)
            x = click.get("x");
            y = click.get("y")
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return
            ax = int(x) + int(self.canvas_offset[0])
            ay = int(y) + int(self.canvas_offset[1])

            # 2) open the menu and wait
            self._click_canvas(ax, ay, button="right")
            time.sleep(int(click.get("open_delay_ms", 120)) / 1000.0)

            # 3) read live menu (entries[].rect has ABSOLUTE canvas coords from plugin)
            info = self.ipc._send({"cmd": "menu"}) or {}
            if not info.get("ok") or not info.get("open"):
                return

            entries = info.get("entries") or []
            want_opt = (click.get("option") or "").strip().lower()
            want_tgt = (click.get("target") or "").strip().lower()
            if not want_opt and not want_tgt:
                return

            def _match(e):
                eo = (e.get("option") or "").strip().lower()
                et = (e.get("target") or "").strip().lower()
                return ((not want_opt) or (eo == want_opt or want_opt in eo)) and \
                    ((not want_tgt) or (want_tgt in et))

            cand = [e for e in entries if _match(e)]
            if not cand:
                return

            r = cand[0].get("rect") or {}
            # rect.x / rect.y are ABSOLUTE canvas coords from the plugin logs:
            # e.g. rect=(751,409 110x15)
            cx = int(r["x"]) + int(r["w"]) // 2
            cy = int(r["y"]) + int(r["h"]) // 2

            # add canvas_offset once (same as you do for other clicks)
            cx += int(self.canvas_offset[0])
            cy += int(self.canvas_offset[1])

            return self._click_canvas(cx, cy, button="left")

        if ctype == "wait":
            ms = int(click.get("ms", 0))
            if ms > 0:
                time.sleep(ms / 1000.0)
            return

        # Unknown type: log and ignore
        self.debug(f"[STEP] unknown click type: {ctype}")

    # ---------- low-level helpers ----------
    def _ipc_ready(self) -> bool:
        try:
            pong = self.ipc._send({"cmd": "ping"})
            return isinstance(pong, dict) and pong.get("ok")
        except Exception:
            return False

    def _click_canvas(self, x: int, y: int, button: str = "left"):
        btn = 1 if button == "left" else 3
        
        try:
            self.ipc.click_canvas(int(x), int(y), button=btn)
            return True
        except Exception as e:
            self.debug(f"[IPC] click error: {e}")
            return None

    @staticmethod
    def _rect_center(bounds: dict | None):
        try:
            if not isinstance(bounds, dict):
                return (None, None)
            x = int(bounds.get("x")); y = int(bounds.get("y"))
            w = int(bounds.get("width")); h = int(bounds.get("height"))
            if w <= 0 or h <= 0:
                return (None, None)
            return (x + w // 2, y + h // 2)
        except Exception:
            return (None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Run RuneLite bot plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rj_loop.py romeo_and_juliet
  python run_rj_loop.py goblin_diplomacy --port 17001
  python run_rj_loop.py --list
  python run_rj_loop.py romeo_and_juliet --session-dir "D:\\data\\sessions\\player1\\gamestates\\"
        """
    )
    
    parser.add_argument(
        "plan", 
        nargs="?", 
        help="Plan name to run (use --list to see available plans)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available plans and exit"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=17000, 
        help="IPC port number (default: 17000)"
    )
    parser.add_argument(
        "--session-dir", 
        type=str, 
        default=r"D:\\repos\\bot_runelite_IL\\data\\recording_sessions\\gorillazzz33\\gamestates\\",
        help="Session directory path"
    )
    parser.add_argument(
        "--canvas-offset", 
        type=str, 
        default="0,0", 
        help="Canvas offset as x,y (default: 0,0)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=120, 
        help="Loop interval in milliseconds (default: 120)"
    )
    
    args = parser.parse_args()
    
    # Handle --list option
    if args.list:
        list_available_plans()
        return
    
    # Require plan name if not listing
    if not args.plan:
        parser.print_help()
        print("\nError: Plan name is required. Use --list to see available plans.")
        sys.exit(1)
    
    # Parse canvas offset
    try:
        offset_parts = args.canvas_offset.split(",")
        canvas_offset = (int(offset_parts[0]), int(offset_parts[1]))
    except (ValueError, IndexError):
        print(f"Error: Invalid canvas offset '{args.canvas_offset}'. Use format 'x,y'")
        sys.exit(1)
    
    # Get plan class
    try:
        plan_class = get_plan_class(args.plan)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create UI and plan instances
    ui = UIShim(
        session_dir=args.session_dir, 
        port=args.port, 
        canvas_offset=canvas_offset
    )
    set_ui(ui)
    plan = plan_class()
    
    ui.debug(f"Starting plan: {plan.label} ({plan.id})")
    ui.debug(f"Session dir: {args.session_dir}")
    ui.debug(f"IPC port: {args.port}")
    ui.debug(f"Canvas offset: {canvas_offset}")
    
    try:
        while True:
            # Always refresh payload for this tick
            payload = ui.latest_payload()
            set_payload(payload)
            
            # Let the plan decide the wait (ms)
            try:
                delay_ms = plan.loop(ui, payload)
            except Exception as e:
                ui.debug(f"[PLAN] error in loop: {e}")
                delay_ms = getattr(plan, "loop_interval_ms", args.interval)
            
            # Optional: log current phase
            try:
                ph = plan.compute_phase(payload, craft_recent=False)
                ui.debug(f"[phase] {ph}")
            except Exception:
                pass
            
            # Normalize delay
            try:
                delay_ms = int(delay_ms if delay_ms is not None else plan.loop_interval_ms)
            except Exception:
                delay_ms = getattr(plan, "loop_interval_ms", args.interval)
            delay_ms = max(10, delay_ms)
            
            time.sleep(delay_ms / 1000.0)
    except KeyboardInterrupt:
        ui.debug("Stopped by user.")
    except Exception as e:
        ui.debug(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
