# ilbot/ui/simple_recorder/actions/travel.py
import random

from .runtime import emit
from ..helpers.camera import prepare_for_walk
from ..helpers.context import get_payload
from ..helpers.navigation import get_nav_rect, closest_bank_key, bank_rect, player_in_rect, _merge_door_into_projection
from ..helpers.ipc import ipc_path, ipc_project_many


# actions/travel.py

from ..helpers.context import get_payload, get_ui
from ..helpers.navigation import closest_bank_key, bank_rect, player_in_rect
from ..services.camera_integration import dispatch_with_camera


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

def _first_blocking_door(proj_rows: list[dict], up_to_index: int | None = None) -> dict | None:
    """
    Scan projected rows (which now include door metadata) and return a click plan
    for the earliest blocking door. Treat missing 'closed' as blocking.
    """
    limit = len(proj_rows) if up_to_index is None else max(0, min(up_to_index + 1, len(proj_rows)))

    for i in range(limit):
        row = proj_rows[i] or {}
        door = (row.get("door") or {})
        if not door.get("present"):
            continue

        closed = door.get("closed")
        if closed is False:
            continue  # already open

        # Prefer rect-center if bounds exist, else use canvas point
        if isinstance(door.get("bounds"), dict):
            click = {"type": "rect-center"}
            target_anchor = {"bounds": door["bounds"]}
        elif isinstance(door.get("canvas"), dict) and \
             isinstance(door["canvas"].get("x"), (int, float)) and isinstance(door["canvas"].get("y"), (int, float)):
            click = {"type": "point", "x": int(door["canvas"]["x"]), "y": int(door["canvas"]["y"])}
            target_anchor = {}
        else:
            # no geometry → skip; we’ll just walk this tick
            continue

        wx = row.get("world", {}).get("x", row.get("x"))
        wy = row.get("world", {}).get("y", row.get("y"))
        wp = row.get("world", {}).get("p", row.get("p"))

        name = door.get("name") or "Door"
        ident = {"id": door.get("id"), "name": name, "world": {"x": wx, "y": wy, "p": wp}}

        return {
            "index": i,
            "click": click,
            "target": {"domain": "object", "name": name, **target_anchor},
            # Your executor already supports postconditions (used in open_bank)
            # Wire a simple predicate your state loop can satisfy when the door flips open.
            "postconditions": [f"doorOpen@{wx},{wy} == true"],
            "timeout_ms": 2000
        }

    return None


def go_to(rect_or_key: str | tuple | list, payload: dict | None = None, ui=None) -> dict | None:
    """
    One movement click toward an area, door-aware.
    If a CLOSED door is on the returned segment before the chosen waypoint,
    click it first and wait (with timeout) for it to open.
    Also aims camera at the door/ground point before dispatch.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # accept key or explicit (minX, maxX, minY, maxY)
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None

    # Path + projection
    wps, dbg_path = ipc_path(payload, rect=tuple(rect))
    if not wps:
        return None

    proj, dbg_proj = ipc_project_many(payload, wps)
    proj = _merge_door_into_projection(wps, proj)

    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
    if not usable:
        return None

    # Choose randomly among the furthest 5 within the first <20 tiles (i.e., indices 15..19 when available)
    max_stride = min(19, len(usable) - 1)           # cap at index 19 (20 tiles from start)
    min_stride = max(0, max_stride - 4)             # last five within that window
    candidates = usable[min_stride:max_stride + 1]  # typical slice = [15..19]

    # Prefer tiles without a blocking door flag
    def _is_blocking(p):
        d = p.get("door") or {}
        if not d.get("present"):
            return False
        closed = d.get("closed")
        return (closed is True) or (closed is None)

    preferred = [p for p in candidates if not _is_blocking(p)]
    pool = preferred if preferred else candidates
    chosen = random.choice(pool)
    chosen_idx = proj.index(chosen)

    # Door before chosen waypoint
    door_plan = _first_blocking_door(usable, up_to_index=chosen_idx)
    if door_plan:
        t = (door_plan.get("target") or {})
        click = (door_plan.get("click") or {})
        b = t.get("bounds") or None
        c = t.get("canvas") or t.get("tileCanvas") or None

        # normalize timeout/postconditions
        timeout_ms = int(door_plan.get("timeout_ms") or 2000)
        postconds = door_plan.get("postconditions") or []

        # 1) Bounds-based click (preferred)
        if isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
            step = emit({
                "action": "open-door",
                "click": click if click else {"type": "rect-center"},
                "target": {
                    "domain": t.get("domain") or "object",
                    "name": t.get("name") or "Door",
                    "bounds": b,
                    "world": door_plan.get("world"),
                },
                "postconditions": postconds,
                "timeout_ms": timeout_ms,
            })
            return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)

        # 2) Canvas point click (fallback)
        if isinstance(c, dict) and "x" in c and "y" in c:
            step = emit({
                "action": "open-door",
                "click": {"type": "point", "x": int(c["x"]), "y": int(c["y"])},
                "target": {
                    "domain": t.get("domain") or "object",
                    "name": t.get("name") or "Door",
                    "world": door_plan.get("world"),
                },
                "postconditions": postconds,
                "timeout_ms": timeout_ms,
            })
            return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)

    # Normal ground move
    world_hint = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
    cx, cy = int(chosen["canvas"]["x"]), int(chosen["canvas"]["y"])
    step = emit({
        "action": "click-ground",
        "description": f"Move toward {rect_key}",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "ground", "name": f"Waypoint→{rect_key}",
                   "world": chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}},
    })

    return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=700)

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

