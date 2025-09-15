from .base import Plan
from ilbot.ui.simple_recorder.actions import *
from ..helpers.ipc import ipc_path, ipc_project_many


class GoToRectPlan(Plan):
    """
    Generic 'walk to rectangle' plan. Use it for GE and Edge Bank by passing a target rect in payload:
      payload["navTarget"] = {"name": "GE", "rect": (minX, maxX, minY, maxY)}
    If you prefer, you can keep tiny wrapper subclasses that set navTarget for you.
    """
    id = "GO_TO_RECT"
    label = "Go to Area"

    def _center(self, rect) -> tuple[int, int]:
        minx, maxx, miny, maxy = rect
        return (minx + maxx) // 2, (miny + maxy) // 2

    def compute_phase(self, payload: dict, craft_recent: bool) -> str:
        me = (payload.get("player") or {})
        tiles = payload.get("tiles_15x15") or []
        tgt = (payload.get("navTarget") or {})
        rect = tgt.get("rect")
        if not me or not tiles or not isinstance(rect, (tuple, list)) or len(rect) != 4:
            return "No Target"

        wx, wy = int(me.get("worldX", 0)), int(me.get("worldY", 0))
        minx, maxx, miny, maxy = rect
        if minx <= wx <= maxx and miny <= wy <= maxy:
            return "Arrived"
        return "Moving"

    def build_action_plan(self, payload: dict, phase: str) -> dict:
        plan = {"phase": phase, "steps": []}
        tgt = (payload.get("navTarget") or {})
        rect = tgt.get("rect")
        name = tgt.get("name") or "TARGET"

        if phase == "Arrived":
            plan["steps"].append({"action": "idle", "description": f"In {name}", "click": {"type":"none"},
                                  "target":{"domain":"none","name":"n/a"}, "preconditions":[], "postconditions":[], "confidence":1.0})
            return plan

        if phase != "Moving":
            plan["steps"].append({"action":"idle","description":"No actionable step","click":{"type":"none"},
                                  "target":{"domain":"none","name":"n/a"}, "preconditions":[], "postconditions":[], "confidence":0.0})
            return plan

        # ask IPC first
        wps, dbg_path = ipc_path(payload, rect=tuple(rect))
        if wps:
            proj, dbg_proj = ipc_project_many(payload, wps)
            usable = [p for p in proj if "canvas" in p]
            if usable:
                import random
                chosen = random.choice(usable[-5:] if len(usable) >= 5 else usable)
                cx, cy = int(chosen["canvas"]["x"]), int(chosen["canvas"]["y"])
                wx, wy, pl = int(chosen["world"]["x"]), int(chosen["world"]["y"]), int(chosen["world"]["p"])
                tx, ty = self._center(tuple(rect))
                plan["steps"].append({
                    "action": "click-ground",
                    "description": f"toward {name} {tx},{ty}",
                    "click": {"type": "point", "x": cx, "y": cy},
                    "target": {"domain":"ground","name":f"Tile→{name}({tx},{ty})",
                               "world":{"x":wx,"y":wy,"plane":pl}, "canvas":{"x":cx,"y":cy}},
                    "preconditions":[], "postconditions":[], "confidence":0.93
                })
                plan["debug"] = {"ipc_nav":{"dbg_path":dbg_path,"dbg_proj":dbg_proj}}
                return plan

        # fallback
        plan["steps"].append({"action":"idle","description":"No onscreen waypoint this tick","click":{"type":"none"},
                              "target":{"domain":"none","name":"n/a"}, "preconditions":[], "postconditions":[], "confidence":0.0})
        return plan
