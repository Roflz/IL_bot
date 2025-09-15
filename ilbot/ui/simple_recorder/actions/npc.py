from __future__ import annotations
from typing import Optional

from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.npc import closest_npc_by_name, npc_action_index
from ..helpers.rects import unwrap_rect, rect_center_xy

def click_npc(name: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Left-click an NPC by (partial) name. Uses rect-center if clickbox exists; else canvas point.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    npc = closest_npc_by_name(name, payload)
    if not npc:
        return None

    rect = unwrap_rect(npc.get("clickbox"))
    if rect:
        cx, cy = rect_center_xy(rect)
        step = emit({
            "action": "npc-click",
            "click": {"type": "rect-center"},
            "target": {"domain": "npc", "name": npc.get("name"), "bounds": rect},
        })
        return ui.dispatch(step)

    if isinstance(npc.get("canvasX"), (int, float)) and isinstance(npc.get("canvasY"), (int, float)):
        step = emit({
            "action": "npc-click",
            "click": {"type": "point", "x": int(npc["canvasX"]), "y": int(npc["canvasY"])},
            "target": {"domain": "npc", "name": npc.get("name")},
        })
        return ui.dispatch(step)

    return None

def click_npc_action(name: str, action: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    npc = closest_npc_by_name(name, payload)
    if not npc:
        return None

    idx = npc_action_index(npc, action)
    if idx is None:
        return None

    rect = unwrap_rect(npc.get("clickbox"))
    name_str = npc.get("name") or "NPC"

    # Determine anchor point
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(npc.get("canvasX"), (int, float)) and isinstance(npc.get("canvasY"), (int, float)):
        cx, cy = int(npc["canvasX"]), int(npc["canvasY"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        return None

    if idx == 0:
        # Desired action is default → a simple left click is enough
        step = emit({
            "action": "npc-action",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "npc", "name": name_str, **anchor},
        })
        return ui.dispatch(step)

    # Need context menu → right-click then select by index
    step = emit({
        "action": "npc-action-context",
        "click": {
            "type": "context-select",
            "index": int(idx),      # 0-based; your dispatcher calculates offset
            "x": cx,
            "y": cy,
            "row_height": 16,
            "start_dy": 18,
            "open_delay_ms": 120
        },
        "target": {"domain": "npc", "name": name_str, **anchor} if rect else {"domain": "npc", "name": name_str},
        "anchor": point  # keep explicit anchor as you do elsewhere
    })
    return ui.dispatch(step)
