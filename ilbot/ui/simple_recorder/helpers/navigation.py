from typing import Optional

from ilbot.ui.simple_recorder.actions import player
from ilbot.ui.simple_recorder.constants import *

def get_nav_rect(rect_or_key):
    """
    Resolve a navigation rectangle.

    Accepts:
      - tuple/list (minX, maxX, minY, maxY) -> returned as-is (normalized to tuple)
      - string key -> tries, in order:
          1) existing NAV_TARGETS (if present)
          2) BANK_REGIONS (if available/importable)
          3) REGIONS (if available/importable)
    Returns tuple(minX, maxX, minY, maxY) or None.
    """
    # 1) direct rect passed in
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        try:
            a, b, c, d = rect_or_key
            return (int(a), int(b), int(c), int(d))
        except Exception:
            return None

    # 2) key-based lookups
    if isinstance(rect_or_key, str):
        key = rect_or_key.strip()
        if not key:
            return None

        # b) BANK_REGIONS
        try:
            if isinstance(BANK_REGIONS, dict):
                if key in BANK_REGIONS:
                    r = BANK_REGIONS[key]
                    if isinstance(r, (tuple, list)) and len(r) == 4:
                        a, b, c, d = r
                        return (int(a), int(b), int(c), int(d))
                if key.upper() in BANK_REGIONS:
                    r = BANK_REGIONS[key.upper()]
                    if isinstance(r, (tuple, list)) and len(r) == 4:
                        a, b, c, d = r
                        return (int(a), int(b), int(c), int(d))
        except Exception:
            pass

        # c) REGIONS (generic pool of rects if you keep them there)
        try:
            if isinstance(REGIONS, dict):
                if key in REGIONS:
                    r = REGIONS[key]
                    if isinstance(r, (tuple, list)) and len(r) == 4:
                        a, b, c, d = r
                        return (int(a), int(b), int(c), int(d))
                if key.upper() in REGIONS:
                    r = REGIONS[key.upper()]
                    if isinstance(r, (tuple, list)) and len(r) == 4:
                        a, b, c, d = r
                        return (int(a), int(b), int(c), int(d))
        except Exception:
            pass

    return None

def player_xy() -> tuple[int | None, int | None]:
    from .ipc import ipc_send
    player_data = ipc_send({"cmd": "get_player"}) or {}
    p = player_data.get("player") or {}
    try:
        return int(p.get("worldX")), int(p.get("worldY"))
    except Exception:
        return (None, None)

def rect_center(rect: tuple[int,int,int,int]) -> tuple[int, int]:
    x0, x1, y0, y1 = rect
    return ( (x0 + x1)//2, (y0 + y1)//2 )

def player_in_rect(rect: tuple[int,int,int,int]) -> bool:
    x, y = player_xy()
    if x is None or y is None: return False
    x0, x1, y0, y1 = rect
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def closest_bank_key() -> str:
    x, y = player.get_player_position()
    # If we don't know player pos, just prefer the first entry.
    if x is None or y is None:
        return next(iter(BANK_REGIONS.keys()))
    best_key, best_d2 = None, 1e18
    for k, v in BANK_REGIONS.items():
        cx, cy = rect_center(v)
        dx, dy = cx - x, cy - y
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_key, best_d2 = k, d2
    return best_key

def bank_rect(key: str) -> tuple[int,int,int,int] | None:
    t = BANK_REGIONS.get((key or "").strip().upper())
    return t

def _merge_door_into_projection(wps: list[dict], proj: list[dict]) -> list[dict]:
    # Build quick lookup: (x,y,p) -> door
    door_by_w = {
        (w["x"], w["y"], w["p"]): w.get("door")
        for w in (wps or [])
        if isinstance(w, dict) and "x" in w and "y" in w and "p" in w and w.get("door")
    }

    out = []
    for row in (proj or []):
        # ensure world coords are on the row; if not, attach them from original path
        wx = row.get("world", {}).get("x", row.get("x"))
        wy = row.get("world", {}).get("y", row.get("y"))
        wp = row.get("world", {}).get("p", row.get("p"))
        if isinstance(wx, int) and isinstance(wy, int) and isinstance(wp, int):
            d = door_by_w.get((wx, wy, wp))
            if d:
                row = dict(row)  # copy so we don’t mutate shared structures
                row["door"] = d
        out.append(row)
    return out
