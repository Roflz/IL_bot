# services/action_executor.py
# Complex action execution service for handling step dispatch

import time
import random
from typing import Optional, Dict, Any, Tuple


class ActionExecutor:
    """
    Handles complex action execution through IPC.
    Provides a dispatch method for executing various types of actions.
    """
    
    def __init__(self, ipc, canvas_offset=(0, 0)):
        self.ipc = ipc
        self.canvas_offset = tuple(canvas_offset or (0, 0))
    
    def debug(self, msg: str):
        """Debug logging with timestamp."""
        import logging
        logging.info(msg)
    
    def dispatch(self, step: dict):
        """
        Execute a single action step immediately.
        Supports: point, rect-center, rect-random, key, key-hold, type, scroll, wait.
        Returns lastInteraction data for click actions.
        """
        # Allow None and sequences of steps
        if step is None:
            return
        if isinstance(step, (list, tuple)):
            for s in step:
                return self.dispatch(s)
            return

        if not isinstance(step, dict):
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
                    x = int(rect.get("x"))
                    y = int(rect.get("y"))
                    w = int(rect.get("width"))
                    h = int(rect.get("height"))
                    if w > 2 and h > 2:
                        cx = x + random.randint(1, w - 2)
                        cy = y + random.randint(1, h - 2)
                except Exception:
                    pass
            return self._click_canvas(cx, cy)

        if ctype in ("point", "canvas-point", "canvas_point"):
            x, y = click.get("x"), click.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                x, y = int(x), int(y)
                x += int(self.canvas_offset[0])
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
                    # Use key press/release for better control
                    self.ipc.key_press(key)
                    time.sleep(dur / 1000.0)  # Convert ms to seconds
                    self.ipc.key_release(key)
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
                x += int(self.canvas_offset[0])
                y += int(self.canvas_offset[1])
                return self._click_canvas(x, y, button="right")
            else:
                self.debug("[STEP] right-click: invalid coords; skipping")
            return

        # --- context menu select via live menu entries ---
        if ctype == "context-select":
            # 1) require explicit right-click anchor (canvas coords)
            x = click.get("x")
            y = click.get("y")
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return
            ax = int(x) + int(self.canvas_offset[0])
            ay = int(y) + int(self.canvas_offset[1])

            # 2) Hover at the target coordinates first (like other methods)
            hover_result = self.ipc.click(ax, ay, hover_only=True)
            if not hover_result.get("ok"):
                return

            # Hover for 50 ms
            time.sleep(0.1)

            # 3) read live menu to see what options are available after hover
            info = self.ipc._send({"cmd": "menu"}) or {}
            entries = info.get("entries") or []
            from ilbot.ui.simple_recorder.helpers.utils import clean_rs
            want_opt = clean_rs((step.get("option") or "")).lower()
            want_tgt = clean_rs((target.get("name") or "")).lower()
            if not want_opt and not want_tgt:
                return

            def _match(e):
                eo = clean_rs((e.get("option") or "")).lower()
                et = clean_rs((e.get("target") or "")).lower()
                
                # Check if exact match is specified (defaults to current behavior if not specified)
                exact_match = click.get("exact_match", True)
                
                if exact_match:
                    # Exact match: option and target must match exactly
                    return ((not want_opt) or (eo == want_opt)) and \
                        ((not want_tgt) or (et == want_tgt))
                else:
                    # Partial match: option contains the wanted text (current behavior)
                    return ((not want_opt) or (want_opt in eo)) and \
                        ((not want_tgt) or (want_tgt in et))

            cand = [e for e in entries if _match(e)]
            if not cand:
                return

            # Check if the target option is at index 0 (first/default option)
            target_entry = cand[0]
            target_visual_index = target_entry.get('visualIndex')

            if target_visual_index == 0:
                # First option - use simple left-click instead of context menu
                click_result = self._click_canvas(ax, ay, button="left")
                
                # Get the last interaction data after the click
                interaction_data = self._get_last_interaction()
                
                return {
                    "click_result": click_result,
                    "interaction": interaction_data
                }
            else:
                # Menu not open after hover, need to right-click to open it
                self._click_canvas(ax, ay, button="right")
                time.sleep(int(click.get("open_delay_ms", 120)) / 1000.0)
                # Re-read menu after opening
                info = self.ipc._send({"cmd": "menu"}) or {}
                entries = info.get("entries") or []
                want_opt = clean_rs(step.get("option") or "").lower()
                want_tgt = clean_rs(target.get("name") or "").lower()
                if not want_opt and not want_tgt:
                    return
                cand = [e for e in entries if _match(e)]
                if not cand:
                    return

                # Check if the target option is at index 0 (first/default option)
                target_entry = cand[0]
                # # Not first option - use context menu selection
                r = target_entry.get('rect')or {}
                # rect.x / rect.y are ABSOLUTE canvas coords from the plugin logs:
                # e.g. rect=(751,409 110x15)
                cx = int(r["x"]) + int(r["w"]) // 2
                cy = int(r["y"]) + int(r["h"]) // 2

                # add canvas_offset once (same as you do for other clicks)
                cx += int(self.canvas_offset[0])
                cy += int(self.canvas_offset[1])

                # Perform the click and capture interaction data
                click_result = self._click_canvas(cx, cy, button="left")
                
                # Get the last interaction data after the click
                interaction_data = self._get_last_interaction()
                
                return {
                    "click_result": click_result,
                    "interaction": interaction_data
                }

        if ctype == "wait":
            ms = int(click.get("ms", 0))
            if ms > 0:
                time.sleep(ms / 1000.0)
            return

        # Unknown type: log and ignore
        self.debug(f"[STEP] unknown click type: {ctype}")

    # ---------- low-level helpers ----------
    def _ipc_ready(self) -> bool:
        """Check if IPC is ready."""
        try:
            pong = self.ipc._send({"cmd": "ping"})
            return isinstance(pong, dict) and pong.get("ok")
        except Exception:
            return False

    def _click_canvas(self, x: int, y: int, button: str = "left"):
        """Click at canvas coordinates."""
        btn = 1 if button == "left" else 3
        
        try:
            self.ipc.click_canvas(int(x), int(y), button=btn)
            return True
        except Exception as e:
            self.debug(f"[IPC] click error: {e}")
            return None

    @staticmethod
    def _rect_center(bounds: dict | None) -> Tuple[Optional[int], Optional[int]]:
        """Get center coordinates of a rectangle."""
        try:
            if not isinstance(bounds, dict):
                return (None, None)
            x = int(bounds.get("x"))
            y = int(bounds.get("y"))
            w = int(bounds.get("width"))
            h = int(bounds.get("height"))
            if w <= 0 or h <= 0:
                return (None, None)
            return (x + w // 2, y + h // 2)
        except Exception:
            return (None, None)

    def _get_last_interaction(self):
        """Get the last interaction data from the StateExporter2Plugin."""
        try:
            # Small delay to ensure the interaction is captured
            time.sleep(0.1)
            
            # Get the last interaction data via IPC
            result = self.ipc.get_last_interaction()
            if result and result.get("ok"):
                return result.get("interaction")
            return None
        except Exception as e:
            self.debug(f"[IPC] get_last_interaction error: {e}")
            return None
