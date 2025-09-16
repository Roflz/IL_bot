# ilbot/ui/simple_recorder/actions/travel.py
from ..helpers.context import get_payload
from ..helpers.navigation import get_nav_rect, closest_bank_key, bank_rect, player_in_rect
from ..helpers.ipc import ipc_path, ipc_project_many


# actions/travel.py

from ..helpers.context import get_payload, get_ui
from ..helpers.navigation import closest_bank_key, bank_rect, player_in_rect

def _is_blocking_door(door: dict) -> bool:
    if not isinstance(door, dict):
        return False
    if not door.get("present"):
        return False
    # If 'closed' missing, be conservative and treat as blocking
    return door.get("closed", True) is True

def _pick_click_from_door(door: dict):
    """Prefer hull center, then tile center; return (x, y) or None."""
    if not isinstance(door, dict):
        return None
    c = door.get("canvas")
    if isinstance(c, dict) and "x" in c and "y" in c:
        return int(c["x"]), int(c["y"])
    tc = door.get("tileCanvas")
    if isinstance(tc, dict) and "x" in tc and "y" in tc:
        return int(tc["x"]), int(tc["y"])
    # As a last resort, if bounds are there, click rect center
    b = door.get("bounds")
    if isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
        return int(b["x"] + b["width"] // 2), int(b["y"] + b["height"] // 2)
    return None

def go_to(rect_or_key: str | tuple | list, payload: dict | None = None, ui=None) -> dict | None:
    """
    Issue one 'move toward area' click using IPC waypoints.
    If any waypoint up to the chosen one has a CLOSED door, click the door first.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # accept either nav key or raw rect (minX,maxX,minY,maxY)
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None

    # 1) pull path (preserving 'door' field in each waypoint)
    wps, dbg_path = ipc_path(payload, rect=tuple(rect))
    if not wps:
        return None

    # 2) project waypoints to canvas (merge door info!)
    proj, dbg_proj = ipc_project_many(payload, wps)
    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
    if not usable:
        return None

    # choose a waypoint a bit ahead to smooth movement
    chosen_idx_in_usable = -3 if len(usable) >= 5 else -1
    chosen = usable[chosen_idx_in_usable]
    # map back to absolute index in the proj list
    chosen_abs_idx = proj.index(chosen)

    # 3) scan from start to chosen_abs_idx for the earliest blocking door
    for i in range(0, chosen_abs_idx + 1):
        p = proj[i]
        door = p.get("door")
        if _is_blocking_door(door):
            click = _pick_click_from_door(door)
            if click:
                cx, cy = click
                step = {
                    "action": "open-door",
                    "description": f"Open door @ {p['world']['x']},{p['world']['y']}",
                    "click": {"type": "point", "x": cx, "y": cy},
                    "target": {
                        "domain": "object",
                        "name": str(door.get("name", "Door")),
                        "id": door.get("id"),
                        "orientation": door.get("orientation"),
                    },
                    "debug": {"door_idx": i, "world": p["world"], "ipc_nav": {"dbg_path": dbg_path, "dbg_proj": dbg_proj}},
                }
                return ui.dispatch(step)
            # if no click geometry, fall through to movement this tick

    # 4) no blocking door found ⇒ normal movement click
    cx, cy = int(chosen["canvas"]["x"]), int(chosen["canvas"]["y"])
    step = {
        "action": "click-ground",
        "description": f"Move toward {rect_key}",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "ground", "name": f"Waypoint→{rect_key}", "world": chosen.get("world")},
        "debug": {"ipc_nav": {"dbg_path": dbg_path, "dbg_proj": dbg_proj, "chosen_idx": chosen_abs_idx}},
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

