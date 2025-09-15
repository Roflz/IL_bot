# ilbot/ui/simple_recorder/actions/travel.py
from ..helpers.context import get_payload
from ..helpers.navigation import get_nav_rect, closest_bank_key, bank_rect, player_in_rect
from ..helpers.ipc import ipc_path, ipc_project_many


# actions/travel.py

from ..helpers.context import get_payload, get_ui
from ..helpers.navigation import closest_bank_key, bank_rect, player_in_rect

def go_to(rect_or_key, payload: dict | None = None, ui=None) -> dict | None:
    """
    Issue a single 'move toward area' click using IPC waypoints.
    - rect_or_key: either a key (str) OR a rect tuple (minX, maxX, minY, maxY)

    Returns ui.dispatch(step) on success, else None if no usable waypoint this tick.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    rect = get_nav_rect(rect_or_key)
    if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
        return None

    wps, dbg_path = ipc_path(payload, rect=tuple(rect))
    if not wps:
        return None

    proj, dbg_proj = ipc_project_many(payload, wps)
    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
    if not usable:
        return None

    # Farther is usually better for progress; pick among farthest few
    chosen = usable[-1] if len(usable) < 5 else usable[-3]
    cx, cy = int(chosen["canvas"]["x"]), int(chosen["canvas"]["y"])

    step = {
        "action": "click-ground",
        "description": f"Move toward {rect_or_key}",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "ground", "name": f"Waypoint→{rect_or_key}", "world": chosen["world"]},
        "debug": {"ipc_nav": {"dbg_path": dbg_path, "dbg_proj": dbg_proj}},
    }
    return ui.dispatch(step)


def go_to_closest_bank(payload: dict | None = None, ui=None) -> dict | None:
    """
    If not already in the nearest bank region, dispatch one move step toward it.
    Returns ui.dispatch(step) on success, else None if already inside or no waypoint this tick.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    key = closest_bank_key(payload)
    rect = bank_rect(key)
    if rect and player_in_rect(payload, rect):
        return None

    return go_to(key, payload, ui)

def go_to_ge(payload: dict | None = None, ui=None) -> dict | None:
    """
    If not already inside the Grand Exchange area, dispatch one move step toward it.
    Returns ui.dispatch(step) on success, else None if already inside or no GE rect is configured.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Try common keys you may have configured for the GE nav rectangle.
    rect_key = None
    rect = None
    for k in ("grand_exchange", "ge"):
        r = get_nav_rect(k)
        if isinstance(r, (tuple, list)) and len(r) == 4:
            rect_key, rect = k, r
            break

    if rect_key is None:
        return None  # no GE area configured

    if player_in_rect(payload, rect):
        return None  # already there

    return go_to(rect_key, payload, ui)

def in_area(area: tuple[int,int,int,int], payload: dict | None = None):
    if payload is None:
        payload = get_payload()
    return player_in_rect(payload, area)

