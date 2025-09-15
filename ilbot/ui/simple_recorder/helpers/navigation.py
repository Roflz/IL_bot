from ilbot.ui.simple_recorder.constants import *

def list_nav_targets_for_ui() -> list[tuple[str, str]]:
    """
    Build [(key, label), ...] for a UI combobox directly from constants.
    - Keys come from BANK_REGIONS first, then REGIONS (no NAV_TARGETS).
    - Labels are derived from key names, with a few friendly overrides.
    - Preferred keys (e.g., GE, EDGEVILLE_BANK) appear first if present.
    """
    keys: list[str] = []

    # Collect from constants (import paths may need tweaking for your tree)
    try:
        from .rects import BANK_REGIONS  # dict[str] -> (minX, maxX, minY, maxY)
        if isinstance(BANK_REGIONS, dict):
            keys.extend(list(BANK_REGIONS.keys()))
    except Exception:
        pass

    try:
        from .rects import REGIONS  # optional, broader pool of areas
        if isinstance(REGIONS, dict):
            for k in REGIONS.keys():
                if k not in keys:
                    keys.append(k)
    except Exception:
        pass

    # Nothing found? Return empty list rather than falling back to NAV_TARGETS.
    if not keys:
        return []

    # Preferred order at the top if present
    preferred = ["GE", "EDGEVILLE_BANK"]
    ordered = [k for k in preferred if k in keys]
    remaining = sorted(k for k in keys if k not in ordered)
    final_keys = ordered + remaining

    # Friendly label overrides; otherwise Title Case the key
    overrides = {
        "GE": "Grand Exchange (GE)",
        "EDGEVILLE_BANK": "Edgeville Bank",
    }
    def to_label(k: str) -> str:
        return overrides.get(k, k.replace("_", " ").title())

    return [(k, to_label(k)) for k in final_keys]

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

def player_xy(payload: dict) -> tuple[int | None, int | None]:
    p = payload.get("player") or {}
    try:
        return int(p.get("worldX")), int(p.get("worldY"))
    except Exception:
        return (None, None)

def rect_center(rect: tuple[int,int,int,int]) -> tuple[int, int]:
    x0, x1, y0, y1 = rect
    return ( (x0 + x1)//2, (y0 + y1)//2 )

def player_in_rect(payload: dict, rect: tuple[int,int,int,int]) -> bool:
    x, y = player_xy(payload)
    if x is None or y is None: return False
    x0, x1, y0, y1 = rect
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def closest_bank_key(payload: dict) -> str:
    x, y = player_xy(payload)
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

