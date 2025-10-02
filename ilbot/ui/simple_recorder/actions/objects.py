from typing import Optional, List, Dict, Any
from ilbot.ui.simple_recorder.actions.runtime import emit
from ilbot.ui.simple_recorder.helpers.context import get_ui, get_payload
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send, ipc_path
from ilbot.ui.simple_recorder.helpers.navigation import _first_blocking_door_from_waypoints
from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy
from ilbot.ui.simple_recorder.services.camera_integration import dispatch_with_camera


def object_exists(
    name: str,
    payload: dict | None = None,
    radius: int = 26,
    types: List[str] | None = None
) -> bool:
    """
    Check if a game object exists by (partial) name.
    
    Args:
        name: Name of the object to search for (partial match)
        payload: Optional payload, will get fresh if None
        radius: Search radius for IPC fallback (default: 26)
        types: Object types to search for (default: ["WALL", "GAME", "DECOR", "GROUND"])
        
    Returns:
        True if object exists, False otherwise
    """
    if not name or not str(name).strip():
        return False
    if payload is None:
        payload = get_payload()
    if types is None:
        types = ["WALL", "GAME", "DECOR", "GROUND"]

    want = str(name).strip().lower()

    # ---------- check payload first ----------
    objs = (payload.get("closestGameObjects") or []) + (payload.get("gameObjects") or [])
    objs = [o for o in objs if (o.get("name") or "").strip()]
    
    for o in objs:
        nm = (o.get("name") or "").lower()
        if want in nm:
            return True

    # ---------- IPC fallback for broader search ----------
    req = {
        "cmd": "objects",
        "name": want,
        "radius": radius,
        "types": types,
    }
    resp = ipc_send(req) or {}
    found = resp.get("objects") or []
    
    return len(found) > 0


def click(
    name: str,
    prefer_action: str | None = None,
    payload: dict | None = None,
    ui=None,
) -> dict | None:
    """
    Click a game object by (partial) name using direct IPC detection with pathing and door handling.
    If a CLOSED door lies on the path to the object, click the earliest blocking door first.
    """
    if not name or not str(name).strip():
        return None
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    want = str(name).strip().lower()
    want_action = (prefer_action or "").strip().lower()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Get fresh payload and object data on each retry
        fresh_payload = get_payload()
        
        # Use direct IPC to get game objects - OPTIMIZED VERSION
        from ..helpers.ipc import ipc_send
        
        # Get specific object using optimized find_object command
        obj_resp = ipc_send({"cmd": "find_object", "name": want, "types": ["GAME"]}, fresh_payload)
        
        if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
            print(f"[DEBUG] Object '{want}' not found")
            continue  # Try next attempt
        
        target = obj_resp.get("object")
        print(f"[DEBUG] Found object: {target.get('name')} at distance {target.get('distance')}")

        def action_index(actions: List[str] | None, needle: str) -> Optional[int]:
            if not needle: return None
            try:
                acts = [a.lower() for a in (actions or []) if a]
                return acts.index(needle) if needle in acts else None
            except Exception:
                return None

        # Target already found by optimized find_object command

        # 1) Check for doors on the path to the object
        gx, gy = target["world"].get("x"), target["world"].get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            from ..helpers.ipc import ipc_path
            from ..helpers.navigation import _first_blocking_door_from_waypoints
            from .travel import _handle_door_opening
            
            wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan, fresh_payload, ui):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the object
        idx = action_index(target.get("actions"), want_action) if want_action else None
        from ..helpers.rects import unwrap_rect, rect_center_xy
        rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
        obj_name = target.get("name") or name
        world_coords = {"x": target.get("world", {}).get("x"), "y": target.get("world", {}).get("y"), "p": target.get("world", {}).get("p", 0)}

        if rect:
            cx, cy = rect_center_xy(rect)
            anchor = {"bounds": rect}
            point = {"x": cx, "y": cy}
        elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            anchor = {}
            point = {"x": cx, "y": cy}
        else:
            continue  # Try next attempt

        from .runtime import emit
        from ..services.camera_integration import dispatch_with_camera
        if not want_action or idx is None or idx == 0:
            # Simple left click
            step = emit({
                "action": "click-object",
                "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
                "target": {"domain": "object", "name": obj_name, **anchor, "world": world_coords},
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
                "target": ({"domain": "object", "name": obj_name, **anchor, "world": world_coords}
                           if rect else {"domain": "object", "name": obj_name, "world": world_coords}),
                "anchor": point,
            })

        # Use centralized click with camera function
        from ..services.click_with_camera import click_object_with_camera
        return click_object_with_camera(
            object_name=name,
            action=want_action,
            action_index=idx,
            world_coords=world_coords,
            ui=ui,
            payload=fresh_payload,
            aim_ms=420
        )

    return None
