# utils/rects.py
from typing import Tuple

def mk_rect(d):
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

def rect_contains(rect, px, py):
    """rect=(x,y,w,h), point=(px,py)"""
    if rect is None:
        return False
    x, y, w, h = rect
    return (px >= x) and (py >= y) and (px < x + w) and (py < y + h)

def center_distance(rect, px, py):
    """smaller is closer to the center of the rect"""
    if rect is None:
        return 1e18
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    dx, dy = (px - cx), (py - cy)
    return (dx * dx + dy * dy) ** 0.5

def unwrap_rect(maybe_rect_dict: dict | None) -> dict | None:
    """
    Inventory/bank slots export as {"bounds": {...}} inside 'bounds'.
    Widgets export as {"bounds": {...}} directly.
    Objects export as 'clickbox' directly.
    This normalizes to a plain {"x","y","width","height"} or None.
    """
    if not isinstance(maybe_rect_dict, dict):
        return None
    if {"x","y","width","height"} <= set(maybe_rect_dict.keys()):
        return maybe_rect_dict
    inner = maybe_rect_dict.get("bounds")
    if isinstance(inner, dict) and {"x","y","width","height"} <= set(inner.keys()):
        return inner
    return None

def rect_center_xy(rect: dict | None) -> Tuple[int | None, int | None]:
    if not rect:
        return (None, None)
    try:
        return int(rect["x"] + rect["width"]/2), int(rect["y"] + rect["height"]/2)
    except Exception:
        return (None, None)
