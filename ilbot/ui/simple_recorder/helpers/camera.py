from .ipc import ipc_send
from .utils import now_ms

_CAM_CENTER_TOL_PX = 120
_CAM_NUDGE_COOLDOWN_MS = 350
_CAM_LAST_MS = 0
_canvas_wh_cache = {"w": 520, "h": 350, "ok": False}

def get_canvas_wh(payload: dict) -> tuple[int,int]:
    w = _canvas_wh_cache["w"]; h = _canvas_wh_cache["h"]
    try:
        if not _canvas_wh_cache["ok"]:
            ipc = payload.get("__ipc")
            if ipc:
                r = ipc._send({"cmd": "where"}) or {}
                if r.get("ok"):
                    w = int(r.get("w", w)); h = int(r.get("h", h))
                    _canvas_wh_cache.update({"w": w, "h": h, "ok": True})
    except Exception:
        pass
    return w, h

def player_canvas(payload: dict) -> tuple[int | None, int | None]:
    """
    Return (px, py) = player's canvas coordinates.
    Prefer tiles_15x15 (fast), fall back to IPC projection of player's world tile.
    """
    me = payload.get("player") or {}
    wx = int(me.get("worldX", 0)); wy = int(me.get("worldY", 0)); p = int(me.get("plane", 0))

    # Try tiles_15x15 first
    for t in (payload.get("tiles_15x15") or []):
        try:
            if int(t.get("worldX")) == wx and int(t.get("worldY")) == wy:
                cx, cy = t.get("canvasX"), t.get("canvasY")
                if isinstance(cx, int) and isinstance(cy, int) and cx >= 0 and cy >= 0:
                    return cx, cy
        except Exception:
            pass

    # Fallback IPC projection
    try:
        resp = ipc_send(payload, {"cmd": "tilexy_many", "tiles": [{"x": wx, "y": wy}]}) or {}
        r = (resp.get("results") or [None])[0]
        if r and r.get("ok") and r.get("onscreen"):
            c = r.get("canvas") or {}
            cx, cy = c.get("x"), c.get("y")
            if isinstance(cx, int) and isinstance(cy, int):
                return cx, cy
    except Exception:
        pass

    return None, None

def offscreen(cx: int|None, cy: int|None, w: int, h: int, margin: int = 4) -> bool:
    if not isinstance(cx, int) or not isinstance(cy, int):
        return True
    return (cx < -margin) or (cy < -margin) or (cx > w + margin) or (cy > h + margin)

def camera_recenter_steps(cx: int, w: int) -> list[dict]:
    center_x = w // 2
    if cx < center_x - _CAM_CENTER_TOL_PX:
        return [{"action":"camera-center","click":{"type":"key","key":"LEFT"}}]
    if cx > center_x + _CAM_CENTER_TOL_PX:
        return [{"action":"camera-center","click":{"type":"key","key":"RIGHT"}}]
    return []

def camera_rotate_to_view_steps(cx: int|None, cy: int|None, w: int, h: int) -> list[dict]:
    """
    If target is off-screen, rotate one tap toward the side it's off.
    Horizontal first; vertical pitch optional (kept simple for now).
    """
    if isinstance(cx, int):
        center_x = w // 2
        if cx < 0:  return [{"action":"camera-aim","click":{"type":"key","key":"LEFT"}}]
        if cx > w:  return [{"action":"camera-aim","click":{"type":"key","key":"RIGHT"}}]
        # if within [0,w], but vertically off: choose a tiny pan anyway to try revealing
        if cx < center_x: return [{"action":"camera-aim","click":{"type":"key","key":"LEFT"}}]
        else:             return [{"action":"camera-aim","click":{"type":"key","key":"RIGHT"}}]
    # unknown cx: gentle nudge alternating left/right (simple)
    import random
    return [{"action":"camera-aim","click":{"type":"key","key": random.choice(["LEFT","RIGHT"])}}]

def camera_nudge_budget() -> bool:
    global _CAM_LAST_MS
    now = now_ms()
    if now - _CAM_LAST_MS < _CAM_NUDGE_COOLDOWN_MS:
        return False
    _CAM_LAST_MS = now
    return True

def read_camera_scale(payload: dict) -> float|None:
    try:
        cam = payload.get("camera") or {}
        sc = cam.get("Scale")
        return float(sc) if sc is not None else None
    except Exception:
        return None

# --- Zoom state ---
_ZOOM_STATE = {
    "zooming": False,              # are we currently in a zoom adjustment loop?
    "last_scale": None,            # last observed camera scale
    "last_amt": 0,                 # last wheel amount we sent (+1 / -1)
    "increases_scale": None,       # True = +1 makes scale go up; False = +1 makes scale go down; None = unknown yet
    "last_sent_ms": 0,             # throttle
}

def zoom_steps_toward(
    payload: dict,
    target_scale: float,
    deadband_stop: float = 50.0,    # stop when inside this band
    deadband_resume: float = 70.0,  # start/continue only when outside this band
    max_steps: int = 2,
    min_interval_ms: int = 120
) -> list[dict]:
    """
    Higher scale = more zoomed IN.
      cur > target -> need to zoom OUT
      cur < target -> need to zoom IN
    Emits 0..max_steps wheel notches with hysteresis + auto direction learning.
    """
    global _ZOOM_STATE
    cur = read_camera_scale(payload)
    if cur is None:
        return []

    now = now_ms()

    # Learn how the wheel maps on this client (from the *previous* tick’s scroll)
    try:
        last_scale = _ZOOM_STATE["last_scale"]
        last_amt   = _ZOOM_STATE["last_amt"]
        if _ZOOM_STATE["increases_scale"] is None and last_scale is not None and last_amt != 0:
            delta = float(cur) - float(last_scale)
            # if +1 last time produced positive delta, then +1 increases scale
            if abs(delta) > 0.1:  # ignore tiny noise
                _ZOOM_STATE["increases_scale"] = (delta * last_amt) > 0
    except Exception:
        pass

    diff = float(cur) - float(target_scale)
    adiff = abs(diff)

    # Decide hysteresis gate
    if _ZOOM_STATE["zooming"]:
        # we keep zooming until we are within the tighter stop band
        if adiff <= deadband_stop:
            _ZOOM_STATE["zooming"] = False
            return []
    else:
        # we only (re)start zooming if we are outside the wider resume band
        if adiff <= deadband_resume:
            return []
        _ZOOM_STATE["zooming"] = True

    # Throttle
    if (now - _ZOOM_STATE["last_sent_ms"]) < min_interval_ms:
        return []

    # Desired logical direction
    want_in = diff < 0      # cur < target -> want zoom IN
    want_out = diff > 0     # cur > target -> want zoom OUT

    # Map to wheel notches using learned mapping; default assumption: +1 => IN (scale increases)
    inc = _ZOOM_STATE["increases_scale"]
    if inc is None:
        amt = +1 if want_in else -1
    else:
        if inc:
            amt = +1 if want_in else -1
        else:
            # inverted mapping on this machine
            amt = -1 if want_in else +1

    # Batch a little when far away
    steps_to_send = 2 if adiff > (4 * deadband_stop) else 1
    steps_to_send = min(steps_to_send, max_steps)

    steps = []
    for _ in range(steps_to_send):
        steps.append({
            "action": "camera-zoom",
            "click": {"type": "scroll", "amount": int(amt)},
            "debug": {
                "camera": "zoom-nav",
                "cur": float(cur),
                "target": float(target_scale),
                "diff": float(diff),
                "amt": int(amt),
                "adiff": float(adiff),
                "inc_map": inc,
                "hysteresis": {"stop": deadband_stop, "resume": deadband_resume},
            }
        })

    # update state for next tick’s learning/throttle
    _ZOOM_STATE["last_scale"] = float(cur)
    _ZOOM_STATE["last_amt"] = int(amt)
    _ZOOM_STATE["last_sent_ms"] = now

    return steps


# camera_keys.py (prefix-free)
import time
import random
from ilbot.ui.simple_recorder.helpers.context import get_payload
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send

# ---------------- basics ----------------

def _screen(payload=None):
    p = payload or get_payload()
    w = int(p.get("screenW") or p.get("canvasW") or 765)
    h = int(p.get("screenH") or p.get("canvasH") or 503)
    return w, h, w // 2, h // 2

def zoom(steps: int):
    """Mouse wheel zoom. Positive -> zoom in, Negative -> zoom out."""
    if steps:
        ipc_send({"cmd": "scroll", "amount": int(steps)})

def _tap(key: str, hold_ms: int = 60, reps: int = 1, pause: float = 0.02):
    for _ in range(max(1, reps)):
        ipc_send({"cmd": "keyHold", "key": key, "ms": int(hold_ms)})
        time.sleep(pause)

def yaw(delta_steps: int, step_ms: int = 60):
    """Rotate LEFT/RIGHT with tiny holds. Positive -> RIGHT, Negative -> LEFT."""
    key = "RIGHT" if delta_steps > 0 else "LEFT"
    _tap(key, hold_ms=step_ms, reps=abs(int(delta_steps)))

def pitch(delta_steps: int, step_ms: int = 80):
    """Tilt camera with UP/DOWN. Positive -> UP, Negative -> DOWN."""
    key = "UP" if delta_steps > 0 else "DOWN"
    _tap(key, hold_ms=step_ms, reps=abs(int(delta_steps)))

# ---------------- heuristics ----------------

def face_world_point(wx: int, wy: int, timeout_ms: int = 900, payload=None) -> bool:
    """
    Center target world tile horizontally using LEFT/RIGHT taps.
    Returns True if horizontally “good enough”, else False after timeout.
    """
    p = payload or get_payload()
    w, h, cx, _ = _screen(p)
    margin = max(48, w // 10)
    deadline = time.time() + timeout_ms / 1000.0

    while time.time() < deadline:
        proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
        if proj.get("ok") and proj.get("onscreen"):
            px = int(proj["canvas"]["x"])
            dx = px - cx
            if abs(dx) <= margin:
                return True
            taps = max(1, min(5, abs(dx) // 70))
            yaw(-taps if dx > 0 else +taps)  # if target is right of center, yaw LEFT
        else:
            yaw(random.choice((-2, 2)))  # exploratory
        time.sleep(0.05)
    return False

def ensure_clickable(point_or_rect: dict, payload=None, max_rot_ms: int = 800) -> bool:
    """
    Make a 2D point/rect center safer to click using ONLY arrow keys + scroll.
    Returns True if likely good to click now.
    Accepts {"x":..,"y":..} or {"bounds":{x,y,width,height}}; optional {"world":{"x":..,"y":..}}.
    """
    p = payload or get_payload()
    w, h, cx, _ = _screen(p)

    # Normalize to a point
    if "bounds" in point_or_rect:
        b = point_or_rect["bounds"]
        px = int(b["x"] + b["width"] / 2)
        py = int(b["y"] + b["height"] / 2)
    else:
        px, py = int(point_or_rect["x"]), int(point_or_rect["y"])

    # If already comfortably on screen, done
    if 24 <= px <= w - 24 and 24 <= py <= h - 24:
        return True

    # Nudge pitch/zoom if low or high
    if py > h - 40:
        pitch(+2)
        zoom(+1)
    elif py < 40:
        pitch(-1)

    # Prefer world-aware recenter
    world = point_or_rect.get("world")
    if isinstance(world, dict) and "x" in world and "y" in world:
        return face_world_point(int(world["x"]), int(world["y"]), timeout_ms=max_rot_ms, payload=p)

    # Fallback: heuristic horizontal recenter (no world coords)
    deadline = time.time() + max_rot_ms / 1000.0
    while time.time() < deadline:
        if 24 <= px <= w - 24 and 24 <= py <= h - 24:
            return True
        taps = max(1, min(4, abs(px - cx) // 80))
        yaw(+taps if px < cx else -taps)
        time.sleep(0.05)
        return True  # let caller re-fetch the point next tick
    return False

# ---------------- prep for walking ----------------

def prepare_for_walk(target_world_xy: tuple[int, int] | None, payload=None):
    """Before long walks: slight top-down bias + face rough goal."""
    pitch(+2)
    zoom(+1)
    if target_world_xy and len(target_world_xy) == 2:
        face_world_point(int(target_world_xy[0]), int(target_world_xy[1]), timeout_ms=600, payload=payload or get_payload())
