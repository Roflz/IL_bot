# camera_midtop.py
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send

# flip this if your pitch feels inverted for UP/DOWN
INVERT_PITCH = False
PITCH_UP_KEY   = "UP"    # tilt so world goes up the screen
PITCH_DOWN_KEY = "DOWN"  # tilt so world goes down the screen

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def aim_midtop_at_world(wx: int, wy: int, *, max_ms: int = 450, x_dead_frac: float = 0.10, y_target_frac: float = 0.30):
    """
    Single-tick nudge: try to keep (wx, wy) around middle-top of the canvas.
    - yaw with LEFT/RIGHT if horizontally off-center
    - pitch with UP/DOWN to bring Y toward ~30% of the screen height
    - never blocks; just sends short keyHold(s)
    """
    try:
        # 1) Where & projection
        where = ipc_send({"cmd": "where"}) or {}
        W, H = int(where.get("w", 0)), int(where.get("h", 0))
        if W <= 0 or H <= 0:
            return

        proj = ipc_send({"cmd": "tilexy", "x": int(wx), "y": int(wy)}) or {}
        if not proj.get("ok"):
            return

        # If tile not on screen, swing yaw a bit and bail
        if not proj.get("onscreen"):
            # heuristic: bias yaw right (you can randomize or remember last direction)
            ipc_send({"cmd": "keyHold", "key": "RIGHT", "ms": int(max_ms * 0.6)})
            return

        cx = int(proj["canvas"]["x"])
        cy = int(proj["canvas"]["y"])

        # 2) Desired anchor is mid-top band
        target_x = W // 2
        target_y = int(H * y_target_frac)

        dx = cx - target_x
        dy = cy - target_y

        # --- yaw nudge (LEFT/RIGHT) ---
        # dead zone: no yaw if within x_dead_frac of width
        x_dead = W * x_dead_frac
        if abs(dx) > x_dead:
            # ms scales with normalized error; cap to max_ms
            yaw_ms = _clamp(int((abs(dx) / (W * 0.6)) * max_ms), 80, max_ms)
            yaw_key = "LEFT" if dx > 0 else "RIGHT"  # target is right of center -> turn LEFT
            ipc_send({"cmd": "keyHold", "key": yaw_key, "ms": yaw_ms})

        # --- pitch nudge (UP/DOWN) ---
        # y band around target_y (~middle-top). Wider band avoids fighting while walking.
        y_band = H * 0.08  # ≈8% of height
        if abs(dy) > y_band:
            pitch_ms = _clamp(int((abs(dy) / (H * 0.5)) * max_ms), 60, max_ms)
            # If the tile is LOWER than target band (dy > 0), we want to tilt UP (make world go up)
            up_key, down_key = (PITCH_DOWN_KEY, PITCH_UP_KEY) if INVERT_PITCH else (PITCH_UP_KEY, PITCH_DOWN_KEY)
            pitch_key = up_key if dy > 0 else down_key
            ipc_send({"cmd": "keyHold", "key": pitch_key, "ms": pitch_ms})

    except Exception:
        pass


def dispatch_with_camera(step: dict, *, ui, payload: dict, aim_ms: int = 450):
    """
    Fire the click immediately; then nudge camera toward the step's target world tile.
    Safe to use in go_to / click_npc / click-object flows.
    """
    # 1) Dispatch the click now (never block movement)
    out = ui.dispatch(step)

    # 2) If target has a world tile, aim once this tick
    try:
        tgt = (step[0].get("target") or {}).get("world") or {}
        wx, wy = tgt.get("x"), tgt.get("y")
        if isinstance(wx, int) and isinstance(wy, int):
            aim_midtop_at_world(wx, wy, max_ms=aim_ms)
    except Exception:
        pass

    return out
