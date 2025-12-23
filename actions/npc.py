from __future__ import annotations
from typing import Optional, List, Union, Dict, Any
import time

from helpers.runtime_utils import ipc
from helpers.runtime_utils import dispatch
from helpers.npc import closest_npc_by_name, npc_action_index
from helpers.rects import unwrap_rect
from helpers.utils import sleep_exponential, rect_beta_xy, clean_rs
from helpers.ipc import get_last_interaction
from .chat import dialogue_is_open, can_choose_option, can_continue, any_chat_active, option_exists, choose_option, continue_dialogue
from .travel import _handle_door_opening, _first_blocking_door_from_waypoints
from services.click_with_camera import click_npc_with_camera
from constants import BANK_REGIONS, REGIONS


def _find_closest_npc_in_area(name: str, area: str | tuple) -> Dict[str, Any] | None:
    """
    Find the closest NPC within a specific area.
    
    Args:
        name: Name of the NPC to search for (partial match)
        area: Area name from constants.py (e.g., "FALADOR_BANK") or tuple (min_x, max_x, min_y, max_y)
        
    Returns:
        Closest NPC within the area, or None if not found
    """
    if not name or not str(name).strip():
        return None

    # Resolve area coordinates
    if isinstance(area, str):
        if area in BANK_REGIONS:
            min_x, max_x, min_y, max_y = BANK_REGIONS[area]
        elif area in REGIONS:
            min_x, max_x, min_y, max_y = REGIONS[area]
        else:
            print(f"[ERROR] Unknown area: {area}. Available areas: {list(BANK_REGIONS.keys()) + list(REGIONS.keys())}")
            return None
    elif isinstance(area, tuple) and len(area) == 4:
        min_x, max_x, min_y, max_y = area
    else:
        print(f"[ERROR] Invalid area format. Use area name or tuple (min_x, max_x, min_y, max_y)")
        return None

    # Get all NPCs and filter by area
    npcs_resp = ipc.get_npcs()
    if not npcs_resp or not npcs_resp.get("ok"):
        return None
    
    npcs = npcs_resp.get("npcs", [])
    if not npcs:
        return None
    
    # Filter NPCs by name and area
    matching_npcs = []
    for npc in npcs:
        npc_name = (npc.get("name") or "").lower()
        if name.lower() in npc_name:
            # Check if NPC is within the area bounds
            world_x = npc.get("worldX")
            world_y = npc.get("worldY")
            
            if (isinstance(world_x, int) and isinstance(world_y, int) and
                min_x <= world_x <= max_x and min_y <= world_y <= max_y):
                matching_npcs.append(npc)
    
    if not matching_npcs:
        return None
    
    # Return the closest NPC by distance
    closest_npc = min(matching_npcs, key=lambda npc: npc.get("distance", 999))
    return closest_npc


def click_npc_simple(name: str, action: str) -> Optional[dict]:
    """
    Left-click an NPC by (partial) name. This version does NOT use pathing or door handling - direct click only.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        fresh_npc = closest_npc_by_name(name)
        if not fresh_npc:
            return None

        expected_target = fresh_npc.get("name", "")

        # Click the NPC directly using centralized function
        world_coords = {"x": fresh_npc.get("worldX"), "y": fresh_npc.get("worldY"), "p": fresh_npc.get("worldP", 0)}
        from services.click_with_camera import click_npc_with_camera
        result = click_npc_with_camera(
            npc_name=fresh_npc.get("name"),
            world_coords=world_coords,
            aim_ms=420,
            action=action
        )

    return None


def click_npc_action(name: str, action: str) -> Optional[dict]:
    """
    Click a specific action on an NPC by auto-selecting:
      - Left-click if the desired action is the default (index 0).
      - Right-click + context-select if the desired action is at index > 0.
    If a CLOSED door lies on the path to the NPC, click the earliest blocking door first.
    """
    # Use optimized find_npc command
    npc_resp = ipc.find_npc(name)

    if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
        return None

    fresh_npc = npc_resp.get("npc")
    print(f"[NPC_ACTION] Found NPC: {fresh_npc.get('name')} at distance {fresh_npc.get('distance')}")
    print(f"[NPC_ACTION] NPC actions: {fresh_npc.get('actions')}")

    # 1) Check for doors on the path to the NPC
    gx, gy = fresh_npc.get("world", {}).get("x"), fresh_npc.get("world", {}).get("y")
    if isinstance(gx, int) and isinstance(gy, int):
        wps, dbg_path = ipc.path(goal=(gx, gy))
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            # Handle door opening with retry logic and recently traversed door tracking
            if not _handle_door_opening(door_plan, wps):
                # Door opening failed after retries, continue to next attempt
                return False
            else:
                return True

    # 2) Click the NPC action with pathing logic
    rect = unwrap_rect(fresh_npc.get("clickbox"))
    name_str = fresh_npc.get("name") or "NPC"

    # Determine anchor point
    print(f"[NPC_ACTION] Checking coordinates - rect: {rect}, canvas: {fresh_npc.get('canvas')}")

    # Get world coordinates for the NPC
    world_coords = fresh_npc.get("world", {})
    if not world_coords or not isinstance(world_coords.get("x"), int) or not isinstance(world_coords.get("y"), int):
        print(f"[NPC_ACTION] No valid world coordinates for NPC, trying next attempt")
        return None

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
    from services.click_with_camera import click_npc_with_camera
    return click_npc_with_camera(
        npc_name=npc.get("name"),
        action=action,
        world_coords=world_coords,
        aim_ms=420
    )


def click_npc_action_simple_prefer_no_camera(name: str, action: str, exact_match: bool = False) -> Optional[dict]:
    """
    Prefer-no-camera NPC click:
    - Attempts a context-select click using the NPC's current clickbox/canvas (no camera movement).
    - If verification fails, returns None (no camera fallback here).
    """
    npc_resp = ipc.find_npc(name)
    if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
        return None

    target = npc_resp.get("npc") or {}
    world = target.get("world") or {}
    if not isinstance(world.get("x"), int) or not isinstance(world.get("y"), int):
        return None

    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    point = None
    anchor = {}
    if rect:
        cx, cy = rect_beta_xy(
            (
                rect.get("x", 0),
                rect.get("x", 0) + rect.get("width", 0),
                rect.get("y", 0),
                rect.get("y", 0) + rect.get("height", 0),
            ),
            alpha=2.0,
            beta=2.0,
        )
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance((target.get("canvas") or {}).get("x"), (int, float)) and isinstance((target.get("canvas") or {}).get("y"), (int, float)):
        point = {"x": int(target["canvas"]["x"]), "y": int(target["canvas"]["y"])}
    else:
        return None

    sleep_exponential(0.05, 0.15, 1.5)
    step = {
        "action": "click-npc-context",
        "click": {
            "type": "context-select",
            "x": point["x"],
            "y": point["y"],
            "row_height": 16,
            "start_dy": 10,
            "open_delay_ms": 120,
            "exact_match": exact_match,
        },
        "option": action,
        "target": {"domain": "npc", "name": target.get("name") or name, **anchor, "world": {"x": world["x"], "y": world["y"], "p": int(world.get("p", 0))}},
        "anchor": point,
    }
    result = dispatch(step)
    if not result:
        return None

    last_interaction = get_last_interaction() or {}
    clean_action = clean_rs(last_interaction.get("action", ""))
    want_action = clean_rs(action).lower()
    action_match = (clean_action.lower() == want_action) if exact_match else (
        (want_action in clean_action.lower()) or (clean_action.lower() in want_action)
    )
    want = clean_rs(name).lower()
    tgt = clean_rs(last_interaction.get("target", "")).lower()
    target_match = (tgt == want) if exact_match else ((want in tgt) or (tgt and (tgt in want)))
    return result if (last_interaction and action_match and target_match) else None

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
        if result is None:
            return None
        
        # Wait for dialogue to open
        start_time = time.time()
        while not dialogue_is_open() and not can_choose_option():
            if (time.time() - start_time) * 1000 > max_wait_ms:
                return None
            sleep_exponential(0.05, 0.15, 1.5)
        
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
            
    if can_continue():
        # Continue the dialogue
        result = continue_dialogue()
        return result
    
    return None
