# nav_simple.py
from math import sqrt

def _dot(ax, ay, bx, by):
    return ax*bx + ay*by

def _norm(dx, dy):
    d = sqrt(dx*dx + dy*dy)
    if d <= 1e-6:
        return 0.0, 0.0
    return dx/d, dy/d

def choose_tile_toward(player_wx, player_wy, ge_wx, ge_wy, visible_tiles, min_dist=3, max_dist=18):
    """
    Pick the visible ground tile whose vector aligns best with GE direction.
    - Only consider tiles that are at least `min_dist` and at most `max_dist` Manhattan distance away (scene-ish).
    - visible_tiles: list of {"worldX","worldY","plane","canvas":{"x","y"}}
    Returns one item from visible_tiles or None.
    """
    dx = ge_wx - player_wx
    dy = ge_wy - player_wy
    vx, vy = _norm(dx, dy)  # desired direction

    best = None
    best_score = -1e9

    for t in visible_tiles or []:
        wx, wy = int(t.get("worldX", 0)), int(t.get("worldY", 0))
        c = t.get("canvas") or {}
        cx, cy = c.get("x"), c.get("y")
        if cx is None or cy is None:
            continue

        # ignore tiles too close/far to reduce misclicking right under the feet or too far
        manh = abs(wx - player_wx) + abs(wy - player_wy)
        if manh < min_dist or manh > max_dist:
            continue

        tx, ty = wx - player_wx, wy - player_wy
        ntx, nty = _norm(tx, ty)
        # score by alignment (dot product) and a small bias for distance (farther clicks = longer strides)
        align = _dot(vx, vy, ntx, nty)
        score = align * 1.0 + (manh * 0.05)
        if score > best_score:
            best_score = score
            best = t

    return best
