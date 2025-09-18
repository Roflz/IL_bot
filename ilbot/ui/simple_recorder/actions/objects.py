from typing import Optional, List, Dict, Any
from ilbot.ui.simple_recorder.actions.runtime import emit
from ilbot.ui.simple_recorder.helpers.context import get_ui, get_payload
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send, ipc_path
from ilbot.ui.simple_recorder.helpers.navigation import _first_blocking_door_from_waypoints
from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy
from ilbot.ui.simple_recorder.services.camera_integration import dispatch_with_camera


def click(
    name: str,
    prefer_action: str | None = None,
    payload: dict | None = None,
    ui=None,
) -> dict | None:
    """
    Click a game object by (partial) name (like open_bank).
    - If a CLOSED door lies on the path to the object, open it first (then return).
    - If prefer_action provided and not the first action, use context-select.
    - Otherwise left-click (rect-center if clickbox available, else canvas point).
    """
    if not name or not str(name).strip():
        return None
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    want = str(name).strip().lower()
    want_action = (prefer_action or "").strip().lower()

    # ---------- gather candidates ----------
    objs = (payload.get("closestGameObjects") or []) + (payload.get("gameObjects") or [])
    objs = [o for o in objs if (o.get("name") or "").strip()]

    # ---------- pick target by name (substring) locally ----------
    target = None
    for o in objs:
        nm = (o.get("name") or "").lower()
        if want in nm:
            target = o
            break

    # ---------- IPC fallback (broader scan), then choose closest ----------
    if target is None:
        req = {
            "cmd": "objects",
            "name": want,                 # substring match in plugin
            "radius": 26,                 # adjust as needed
            "types": ["WALL", "GAME", "DECOR", "GROUND"],
        }
        resp = ipc_send(req) or {}
        found = (resp.get("objects") or [])
        if not found:
            return None

        me = (payload.get("player") or {})
        me_x = me.get("worldX") if isinstance(me.get("worldX"), int) else payload.get("worldX")
        me_y = me.get("worldY") if isinstance(me.get("worldY"), int) else payload.get("worldY")
        me_p = me.get("plane")  if isinstance(me.get("plane"),  int) else payload.get("plane")

        def obj_wxy_p(o):
            w = o.get("world") or {}
            ox = w.get("x", o.get("worldX"))
            oy = w.get("y", o.get("worldY"))
            op = w.get("p", o.get("plane"))
            return ox, oy, op

        scored = []
        for o in found:
            ox, oy, op = obj_wxy_p(o)
            if isinstance(me_x, int) and isinstance(me_y, int) and isinstance(ox, int) and isinstance(oy, int):
                dist = abs(ox - me_x) + abs(oy - me_y)
            else:
                dist = 10**9
            same_plane = 0 if (isinstance(me_p, int) and op == me_p) else 1
            has_rect = 0 if (o.get("bounds") or o.get("clickbox")) else 1
            scored.append((same_plane, dist, has_rect, o))

        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        target = scored[0][3]

    # ---------- door check: path to object, open earliest blocking door ----------
    w = target.get("world") or {}
    gx = w.get("x", target.get("worldX"))
    gy = w.get("y", target.get("worldY"))
    if isinstance(gx, int) and isinstance(gy, int):
        wps, dbg_path = ipc_path(payload, goal=(gx, gy), max_wps=24)
        door_plan = _first_blocking_door_from_waypoints(wps)  # your existing helper
        if door_plan:
            d = (door_plan.get("door") or {})
            b = d.get("bounds")
            if isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
                step = emit({
                    "action": "open-door",
                    "click": {"type": "rect-center"},
                    "target": {"domain": "object", "name": d.get("name") or "Door", "bounds": b},
                    "postconditions": [],
                    "timeout_ms": 1200,
                })
            else:
                c = d.get("canvas") or d.get("tileCanvas")
                if not (isinstance(c, dict) and "x" in c and "y" in c):
                    return None
                step = emit({
                    "action": "open-door",
                    "click": {"type": "point", "x": int(c["x"]), "y": int(c["y"])},
                    "target": {"domain": "object", "name": d.get("name") or "Door"},
                    "postconditions": [],
                    "timeout_ms": 1200,
                })
            return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)

    # ---------- click geometry ----------
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target.get("canvasX"), (int, float)) and isinstance(target.get("canvasY"), (int, float)):
        cx, cy = int(target["canvasX"]), int(target["canvasY"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        return None

    # ---------- decide action / build step ----------
    def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
        if not needle: return None
        try:
            acts = [a.lower() for a in (actions or []) if a]
            return acts.index(needle) if needle in acts else None
        except Exception:
            return None

    acts = target.get("actions") or []
    idx = action_index(acts, want_action) if want_action else None
    obj_name = target.get("name") or name

    if not want_action or idx is None or idx == 0:
        # Simple left click
        step = emit({
            "action": "click-object",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "object", "name": obj_name, **anchor},
        })
    else:
        # Context menu select
        step = emit({
            "action": "click-object-context",
            "click": {
                "type": "context-select",
                "index": int(idx),
                "x": point["x"],
                "y": point["y"],
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "target": ({"domain": "object", "name": obj_name, **anchor}
                       if rect else {"domain": "object", "name": obj_name}),
            "anchor": point,
        })

    return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)
