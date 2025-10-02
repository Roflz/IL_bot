from __future__ import annotations
from typing import Optional, List, Union
import time

from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.door_traversal import check_door_traversal
from ..helpers.ipc import ipc_path, ipc_send
from ..helpers.navigation import _first_blocking_door_from_waypoints
from ..helpers.npc import closest_npc_by_name, npc_action_index
from ..helpers.rects import unwrap_rect, rect_center_xy
from .chat import dialogue_is_open, can_choose_option, can_continue, any_chat_active, option_exists, choose_option, continue_dialogue
from .travel import _handle_door_opening

from ..services.camera_integration import dispatch_with_camera


def _door_is_open(door_plan: dict, payload: dict) -> bool:
    """
    Check if a door is open by looking for it in the objects list.
    If the door is not found in objects, it's likely open.
    """
    if not door_plan or not door_plan.get("door"):
        return True
    
    door = door_plan.get("door", {})
    door_name = door.get("name", "")
    door_world = door.get("world", {})
    door_x = door_world.get("x")
    door_y = door_world.get("y")
    
    if not isinstance(door_x, int) or not isinstance(door_y, int):
        return True  # Can't verify, assume open
    
    # Check if door still exists in objects (if it does, it's closed)
    objects = payload.get("objects", [])
    for obj in objects:
        obj_world = obj.get("world", {})
        obj_x = obj_world.get("x")
        obj_y = obj_world.get("y")
        obj_name = obj.get("name", "")
        
        if (obj_x == door_x and obj_y == door_y and 
            door_name.lower() in obj_name.lower()):
            return False  # Door still exists, so it's closed
    
    return True  # Door not found, so it's open


def click_npc(name: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Left-click an NPC by (partial) name. If a CLOSED door lies on the path to the NPC,
    click the earliest blocking door first and wait briefly for it to open; otherwise,
    click the NPC.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    max_retries = 3
    expected_action = "Talk-to"
    
    for attempt in range(max_retries):
        # Get fresh payload and NPC data on each retry - OPTIMIZED VERSION
        fresh_payload = get_payload()
        
        # Use optimized find_npc command
        from ..helpers.ipc import ipc_send
        npc_resp = ipc_send({"cmd": "find_npc", "name": name}, fresh_payload)
        
        if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
            return None
        
        fresh_npc = npc_resp.get("npc")
        expected_target = fresh_npc.get("name", "")

        # 1) Check for doors on the path to the NPC
        gx, gy = fresh_npc.get("world", {}).get("x"), fresh_npc.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan, fresh_payload, ui):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the NPC with pathing logic
        rect = unwrap_rect(fresh_npc.get("clickbox"))
        world_coords = {"x": fresh_npc.get("world", {}).get("x"), "y": fresh_npc.get("world", {}).get("y"), "p": fresh_npc.get("world", {}).get("p", 0)}
        
        if rect:
            step = emit({
                "action": "npc-click",
                "click": {"type": "rect-center"},
                "target": {"domain": "npc", "name": fresh_npc.get("name"), "bounds": rect, "world": world_coords}
            })
            result = dispatch_with_camera(step, ui=ui, payload=fresh_payload, aim_ms=420)
        elif isinstance(fresh_npc.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_npc.get("canvas", {}).get("y"), (int, float)):
            step = emit({
                "action": "npc-click",
                "click": {"type": "point", "x": int(fresh_npc.get("canvas", {}).get("x")), "y": int(fresh_npc.get("canvas", {}).get("y"))},
                "target": {"domain": "npc", "name": fresh_npc.get("name"), "world": world_coords}
            })
            result = dispatch_with_camera(step, ui=ui, payload=fresh_payload, aim_ms=420)
        else:
            result = None
        
        if result:
            # Wait up to 600ms for lastInteraction to update and verify
            start_time = time.time()
            while (time.time() - start_time) * 1000 < 600:
                verify_payload = get_payload()
                last_interaction = verify_payload.get("lastInteraction")
                
                if last_interaction:
                    action = last_interaction.get("action", "")
                    target_name = last_interaction.get("target_name", "")
                    
                    # Check if the interaction matches what we expect
                    if expected_action in action and expected_target in target_name:
                        return result
                
                time.sleep(0.05)  # 50ms

    return None


def click_npc_simple(name: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Left-click an NPC by (partial) name. This version does NOT use pathing or door handling - direct click only.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    max_retries = 3
    expected_action = "Talk-to"
    
    for attempt in range(max_retries):
        # Get fresh payload and NPC data on each retry
        fresh_payload = get_payload()
        fresh_npc = closest_npc_by_name(name, fresh_payload)
        if not fresh_npc:
            return None

        expected_target = fresh_npc.get("name", "")

        # Click the NPC directly using centralized function
        world_coords = {"x": fresh_npc.get("worldX"), "y": fresh_npc.get("worldY"), "p": fresh_npc.get("worldP", 0)}
        from ..services.click_with_camera import click_npc_with_camera
        result = click_npc_with_camera(
            npc_name=fresh_npc.get("name"),
            world_coords=world_coords,
            ui=ui,
            payload=fresh_payload,
            aim_ms=420
        )
        
        if result:
            # Wait up to 600ms for lastInteraction to update and verify
            start_time = time.time()
            while (time.time() - start_time) * 1000 < 600:
                verify_payload = get_payload()
                last_interaction = verify_payload.get("lastInteraction")
                
                if last_interaction:
                    action = last_interaction.get("action", "")
                    target_name = last_interaction.get("target_name", "")
                    
                    # Check if the interaction matches what we expect
                    if expected_action in action and expected_target in target_name:
                        return result
                
                time.sleep(0.05)  # 50ms

    return None


def click_npc_action(name: str, action: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    If a CLOSED door lies on the path to the NPC, click the earliest blocking door first.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    max_retries = 3
    
    for attempt in range(max_retries):
        # Get fresh payload and NPC data on each retry - OPTIMIZED VERSION
        fresh_payload = get_payload()
        
        # Use optimized find_npc command
        from ..helpers.ipc import ipc_send
        npc_resp = ipc_send({"cmd": "find_npc", "name": name}, fresh_payload)
        
        if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
            return None
        
        fresh_npc = npc_resp.get("npc")
        print(f"[NPC_ACTION] Found NPC: {fresh_npc.get('name')} at distance {fresh_npc.get('distance')}")
        print(f"[NPC_ACTION] NPC actions: {fresh_npc.get('actions')}")
        
        idx = npc_action_index(fresh_npc, action)
        print(f"[NPC_ACTION] Action '{action}' found at index: {idx}")
        if idx is None:
            return None

        # 1) Check for doors on the path to the NPC
        gx, gy = fresh_npc.get("world", {}).get("x"), fresh_npc.get("world", {}).get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan, fresh_payload, ui):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the NPC action with pathing logic
        rect = unwrap_rect(fresh_npc.get("clickbox"))
        name_str = fresh_npc.get("name") or "NPC"

        # Determine anchor point
        print(f"[NPC_ACTION] Checking coordinates - rect: {rect}, canvas: {fresh_npc.get('canvas')}")
        
        if rect:
            cx, cy = rect_center_xy(rect)
            anchor = {"bounds": rect}
            point = {"x": cx, "y": cy}
            print(f"[NPC_ACTION] Using rect coordinates: ({cx}, {cy})")
        elif isinstance(fresh_npc.get("canvas", {}).get("x"), (int, float)) and isinstance(fresh_npc.get("canvas", {}).get("y"), (int, float)):
            cx, cy = int(fresh_npc.get("canvas", {}).get("x")), int(fresh_npc.get("canvas", {}).get("y"))
            anchor = {}
            point = {"x": cx, "y": cy}
            print(f"[NPC_ACTION] Using canvas coordinates: ({cx}, {cy})")
        else:
            print(f"[NPC_ACTION] No valid coordinates found, trying next attempt")
            continue  # Try next attempt

        if idx == 0:
            # Desired action is default → a simple left click is enough
            print(f"[NPC_ACTION] Using left-click for action at index 0")
            step = emit({
                "action": "npc-action",
                "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
                "target": {"domain": "npc", "name": name_str, **anchor},
            })
            result = dispatch_with_camera(step, ui=ui, payload=fresh_payload, aim_ms=420)
        else:
            # Need context menu → right-click then select by index
            print(f"[NPC_ACTION] Using context menu for action at index {idx}")
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
            result = dispatch_with_camera(step, ui=ui, payload=fresh_payload, aim_ms=420)
        
        print(f"[NPC_ACTION] Click result: {result}")
        
        if result:
            return result

    return None


def click_npc_action_simple(name: str, action: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    This version does NOT use pathing or door handling - direct click only.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Use optimized find_npc command (now returns closest NPC)
    from ..helpers.ipc import ipc_send
    npc_resp = ipc_send({"cmd": "find_npc", "name": name}, payload)
    
    if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
        return None
    
    npc = npc_resp.get("npc")
    print(f"[NPC_ACTION_SIMPLE] Found closest NPC: {npc.get('name')} at distance {npc.get('distance')}")
    
    idx = npc_action_index(npc, action)
    if idx is None:
        return None

    # Use centralized function for simple NPC action
    world_coords = {"x": npc.get("world", {}).get("x"), "y": npc.get("world", {}).get("y"), "p": npc.get("world", {}).get("p", 0)}
    from ..services.click_with_camera import click_npc_with_camera
    return click_npc_with_camera(
        npc_name=npc.get("name"),
        action=action,
        action_index=idx,
        world_coords=world_coords,
        ui=ui,
        payload=payload,
        aim_ms=420
    )

def chat_with_npc(npc_name: str, options: Optional[List[str]] = None, payload: Optional[dict] = None, ui=None, max_wait_ms: int = 4000) -> Optional[Union[int, dict]]:
    """
    Start a conversation with an NPC and handle the entire dialogue flow.
    
    Args:
        npc_name: Name of the NPC to talk to
        options: List of dialogue options to select if they appear (in order of preference)
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
        max_wait_ms: Maximum time to wait for dialogue to open
    
    Returns:
        - Return value from choose_option() or continue_dialogue() if successful
        - 1200 (delay) if an option was selected
        - None if dialogue failed to open or no chat is active
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    if options is None:
        options = []
    
    # Check if dialogue is already open or if we can choose options
    if not dialogue_is_open() and not can_choose_option() and not can_continue():
        # Click on the NPC to start conversation
        result = click_npc(npc_name)
        # result = check_door_traversal(3098, 3107, 0, max_time=3.0)
        if result is None:
            return None
        
        # Wait for dialogue to open
        start_time = time.time()
        while not dialogue_is_open() and not can_choose_option():
            if (time.time() - start_time) * 1000 > max_wait_ms:
                return None
            time.sleep(0.1)
        
        return result
    
    # Handle existing dialogue
    if any_chat_active():
        if can_choose_option():
            # Try to select one of the provided options in order of preference
            for option in options:
                if option_exists(option):
                    result = choose_option(option)
                    if result is not None:
                        return 1200  # Return delay value as in your example
                    break
            
            # If no preferred options found, return None
            return None
            
        elif can_continue():
            # Continue the dialogue
            result = continue_dialogue()
            return result
    
    return None
