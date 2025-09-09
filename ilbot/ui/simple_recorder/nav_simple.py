# nav_simple.py
from math import sqrt

def _dot(ax, ay, bx, by):
    return ax*bx + ay*by

def _norm(dx, dy):
    d = sqrt(dx*dx + dy*dy)
    if d <= 1e-6:
        return 0.0, 0.0
    return dx/d, dy/d

def next_tile_toward(player_wx: int, player_wy: int,
                     target_wx: int, target_wy: int,
                     max_step: int = 15) -> tuple[int, int]:
    """
    Returns the world tile (wx, wy) at most `max_step` tiles away from player,
    along the straight-line direction toward target. If already very close,
    returns the target tile itself.
    """
    dx = target_wx - player_wx
    dy = target_wy - player_wy
    dist = (dx*dx + dy*dy) ** 0.5
    if dist <= 0.5:
        return target_wx, target_wy

    step = min(max_step, dist)
    # normalized direction, then step and round to nearest tile
    nx = dx / dist
    ny = dy / dist
    wx = int(round(player_wx + nx * step))
    wy = int(round(player_wy + ny * step))
    return wx, wy
