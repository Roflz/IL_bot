# camera_integration.py
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send

INVERT_PITCH = False
PITCH_UP_KEY   = "UP"
PITCH_DOWN_KEY = "DOWN"

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def aim_midtop_at_world(
    wx: int,
    wy: int,
    *,
    max_ms: int = 450,
    x_dead_frac: float = 0.10,
    y_target_frac: float = 0.24,   # keep target high → less need to pitch down
    y_band_frac: float = 0.12,     # wider band → fewer corrections
    # asym hysteresis: be stricter before pitching DOWN
    y_band_down_mult: float = 2.2,
    y_band_up_mult: float = 1.0,
    # make down nudges gentler + hard-cap them
    pitch_down_bias: float = 0.45,
    pitch_down_cap_ms: int = 95,
    # zoom regulation (Scale only): lock to fully zoomed out
    target_scale: int = 551,
    scale_tol: int = 1,
    payload: dict | None = None,
):
    """
    One-tick nudge:
      - yaw if off-center in X
      - pitch toward mid-top band with asymmetric hysteresis & capped DOWN holds
      - regulate zoom using ONLY camera.Scale: keep near 551 (fully zoomed out)
    """
    try:
        where = ipc_send({"cmd": "where"}) or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))
        if W <= 0 or H <= 0:
            return

        proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
        if not proj.get("ok"):
            return

        if not proj.get("onscreen"):
            ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.6)})
            return

        cx = int(proj["canvas"]["x"])
        cy = int(proj["canvas"]["y"])

        # if the projected point sits outside the window, treat it as offscreen
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.85)})
            return

        target_x = W // 2
        target_y = int(H * y_target_frac)

        dx = cx - target_x
        dy = cy - target_y

        # --- yaw ---
        x_dead = W * x_dead_frac
        if abs(dx) > x_dead:
            yaw_ms = _clamp(int((abs(dx) / (W * 0.6)) * max_ms), 80, max_ms)
            yaw_key = "LEFT" if dx > 0 else "RIGHT"
            ipc_send({"cmd": "keyHold", "key": yaw_key, "ms": yaw_ms})

        # --- pitch with asymmetric hysteresis ---
        base_band   = H * y_band_frac
        y_band_down = base_band * y_band_down_mult   # stricter to pitch DOWN
        y_band_up   = base_band * y_band_up_mult

        need_down = dy < -y_band_down   # tile above target band
        need_up   = dy >  y_band_up     # tile below target band

        if need_down or need_up:
            up_key, down_key = (PITCH_DOWN_KEY, PITCH_UP_KEY) if INVERT_PITCH else (PITCH_UP_KEY, PITCH_DOWN_KEY)
            pitch_ms = _clamp(int((abs(dy) / (H * 0.5)) * max_ms), 60, max_ms)
            if need_down:
                pitch_ms = int(pitch_ms * pitch_down_bias)
                pitch_ms = min(pitch_ms, pitch_down_cap_ms)   # <- kills the “last 2 dips”
                ipc_send({"cmd": "keyHold", "key": down_key, "ms": max(30, pitch_ms)})
            else:
                ipc_send({"cmd": "keyHold", "key": up_key,   "ms": max(40, pitch_ms)})

        try:
            cam = (payload or {}).get("camera") or {}
            scale = cam.get("Scale", None)  # no 'Zoom' anywhere
            if isinstance(scale, (int, float)):
                # Only ever scroll OUT if we're tighter than the target.
                if scale > (target_scale + scale_tol):
                    ipc_send({"cmd": "scroll", "amount": -1})
                # Never scroll IN; do nothing otherwise.
        except Exception:
            pass

    except Exception:
        pass


# if your current dispatch_with_camera doesn't pass payload, use this version:
def dispatch_with_camera(step: dict, *, ui, payload: dict, aim_ms: int = 450):
    out = ui.dispatch(step)
    try:
        tgt = (step[0].get("target") or {}).get("world") or {}
        wx, wy = tgt.get("x"), tgt.get("y")
        if isinstance(wx, int) and isinstance(wy, int):
            aim_midtop_at_world(wx, wy, max_ms=aim_ms, payload=payload)
    except Exception:
        pass
    return out
