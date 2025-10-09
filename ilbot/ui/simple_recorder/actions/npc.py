from __future__ import annotations
from typing import Optional, List, Union
import time

from ..helpers.runtime_utils import ipc
from ..helpers.navigation import _first_blocking_door_from_waypoints
from ..helpers.npc import closest_npc_by_name, npc_action_index
from ..helpers.rects import unwrap_rect, rect_center_xy
from .chat import dialogue_is_open, can_choose_option, can_continue, any_chat_active, option_exists, choose_option, continue_dialogue
from .travel import _handle_door_opening
from ..services.click_with_camera import click_npc_with_camera

from ..services.camera_integration import dispatch_with_camera


def click_npc_simple(name: str) -> Optional[dict]:
    """
    Left-click an NPC by (partial) name. This version does NOT use pathing or door handling - direct click only.
    """
    max_retries = 3
    expected_action = "Talk-to"
    
    for attempt in range(max_retries):
        fresh_npc = closest_npc_by_name(name)
        if not fresh_npc:
            return None

        expected_target = fresh_npc.get("name", "")

        # Click the NPC directly using centralized function
        world_coords = {"x": fresh_npc.get("worldX"), "y": fresh_npc.get("worldY"), "p": fresh_npc.get("worldP", 0)}
        from ..services.click_with_camera import click_npc_with_camera
        result = click_npc_with_camera(
            npc_name=fresh_npc.get("name"),
            world_coords=world_coords,
            aim_ms=420
        )
        
        if result:
            # Wait up to 600ms for lastInteraction to update and verify
            start_time = time.time()
            while (time.time() - start_time) * 1000 < 600:
                interaction_data = ipc.get_last_interaction()
                last_interaction = interaction_data.get("interaction")
                
                if last_interaction:
                    action = last_interaction.get("action", "")
                    target_name = last_interaction.get("target_name", "")
                    
                    # Check if the interaction matches what we expect
                    if expected_action in action and expected_target in target_name:
                        return result
                
                time.sleep(0.05)  # 50ms

    return None


def click_npc_action(name: str, action: str) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    If a CLOSED door lies on the path to the NPC, click the earliest blocking door first.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        # Use optimized find_npc command
        npc_resp = ipc.find_npc(name)
        
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
            wps, dbg_path = ipc.path(goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan):
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

        # Get world coordinates for the NPC
        world_coords = fresh_npc.get("world", {})
        if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
            print(f"[NPC_ACTION] No valid world coordinates for NPC, trying next attempt")
            continue
        
        if idx == 0:
            # Desired action is default → use click_npc_with_camera with no action
            print(f"[NPC_ACTION] Using left-click for action at index 0")
            result = click_npc_with_camera(
                npc_name=name_str,
                action=action,
                world_coords=world_coords,
                aim_ms=420
            )
        else:
            # Need context menu → use click_npc_with_camera with action and index
            print(f"[NPC_ACTION] Using context menu for action at index {idx}")
            result = click_npc_with_camera(
                npc_name=name_str,
                action=action,
                world_coords=world_coords,
                aim_ms=420
            )
        
        print(f"[NPC_ACTION] Click result: {result}")
        
        if result:
            return result

    return None


def click_npc_action_simple(name: str, action: str) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    This version does NOT use pathing or door handling - direct click only.
    """
    # Use optimized find_npc command (now returns closest NPC)
    npc_resp = ipc.find_npc(name)
    
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
        world_coords=world_coords,
        aim_ms=420
    )

def chat_with_npc(npc_name: str, options: Optional[List[str]] = None, max_wait_ms: int = 4000) -> Optional[Union[int, dict]]:
    """
    Start a conversation with an NPC and handle the entire dialogue flow.
    
    Args:
        npc_name: Name of the NPC to talk to
        options: List of dialogue options to select if they appear (in order of preference)
        max_wait_ms: Maximum time to wait for dialogue to open
    
    Returns:
        - Return value from choose_option() or continue_dialogue() if successful
        - 1200 (delay) if an option was selected
        - None if dialogue failed to open or no chat is active
    """
    if options is None:
        options = []
    
    # Check if dialogue is already open or if we can choose options
    if not dialogue_is_open() and not can_choose_option() and not can_continue():
        # Click on the NPC to start conversation
        result = click_npc_action(npc_name, "Talk-to")
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
