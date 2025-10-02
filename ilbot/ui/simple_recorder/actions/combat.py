from __future__ import annotations
from typing import Optional
import time

from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.ipc import ipc_send
from ..services.camera_integration import dispatch_with_camera


def attack_closest(npc_name: str | list, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Find the closest NPC with the given name(s) that is not in combat and attack it.
    
    Args:
        npc_name: Name(s) of the NPC(s) to attack (partial match allowed)
                 Can be a single string or a list of strings
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Convert single string to list for uniform handling
    if isinstance(npc_name, str):
        npc_names = [npc_name]
    else:
        npc_names = npc_name
    
    if not npc_names:
        print("[COMBAT] No NPC names provided")
        return None
    
    max_retries = 3
    
    for attempt in range(max_retries):
        # Get fresh payload and NPC data on each retry
        fresh_payload = get_payload()
        
        # Try each NPC name in the list
        for npc_name_to_try in npc_names:
            # Use optimized find_npc command to get closest NPC
            npc_resp = ipc_send({"cmd": "find_npc", "name": npc_name_to_try}, fresh_payload)
            
            if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
                print(f"[COMBAT] No NPC found with name containing '{npc_name_to_try}'")
                continue  # Try next name in the list
            
            npc = npc_resp.get("npc")
            print(f"[COMBAT] Found NPC: {npc.get('name')} at distance {npc.get('distance')}")
            
            # Check if NPC is already in combat
            if _is_npc_in_combat(npc, fresh_payload):
                print(f"[COMBAT] NPC {npc.get('name')} is already in combat, skipping")
                continue  # Try next name in the list
            
            # Check if NPC has 'Attack' action available
            actions = npc.get("actions", [])
            if "Attack" not in actions:
                print(f"[COMBAT] NPC {npc.get('name')} does not have 'Attack' action available")
                continue  # Try next name in the list
            
            # Attack the NPC
            result = _attack_npc(npc, fresh_payload, ui)
            if result:
                print(f"[COMBAT] Successfully attacked {npc.get('name')}")
                return result
            else:
                print(f"[COMBAT] Failed to attack {npc.get('name')}, trying next NPC...")
                continue  # Try next name in the list
        
        # If we get here, no NPCs were successfully attacked in this attempt
        print(f"[COMBAT] No attackable NPCs found in attempt {attempt + 1}, retrying...")
        time.sleep(0.5)  # Brief delay before retry
    
    print(f"[COMBAT] Failed to attack any NPC from list {npc_names} after {max_retries} attempts")
    return None


def _is_npc_in_combat(npc: dict, payload: dict) -> bool:
    """
    Check if an NPC is currently in combat using the IPC response data.
    
    Args:
        npc: NPC data dictionary from IPC response
        payload: Game state payload (unused, kept for compatibility)
        
    Returns:
        True if NPC is in combat, False otherwise
    """
    # Use the inCombat property from the IPC response
    in_combat = npc.get("inCombat", False)
    
    if in_combat:
        combat_target = npc.get("combatTarget", "Unknown")
        combat_target_type = npc.get("combatTargetType", "Unknown")
        print(f"[COMBAT] NPC {npc.get('name')} is in combat with {combat_target_type} '{combat_target}'")
    
    return in_combat


def _attack_npc(npc: dict, payload: dict, ui) -> Optional[dict]:
    """
    Perform the actual attack action on an NPC with pathing and door handling.
    
    Args:
        npc: NPC data dictionary
        payload: Game state payload
        ui: UI instance
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    from ..helpers.rects import unwrap_rect, rect_center_xy
    from ..helpers.ipc import ipc_path
    from ..helpers.navigation import _first_blocking_door_from_waypoints
    from .travel import _handle_door_opening
    
    # 1) Check for doors on the path to the NPC
    gx, gy = npc.get("world", {}).get("x"), npc.get("world", {}).get("y")
    if isinstance(gx, int) and isinstance(gy, int):
        wps, dbg_path = ipc_path(payload, goal=(gx, gy))
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            # Handle door opening with retry logic
            if not _handle_door_opening(door_plan, payload, ui):
                print(f"[COMBAT] Failed to open door on path to NPC")
                return None
    
    # Get NPC coordinates and bounds
    rect = unwrap_rect(npc.get("clickbox"))
    world_coords = {
        "x": npc.get("world", {}).get("x"), 
        "y": npc.get("world", {}).get("y"), 
        "p": npc.get("world", {}).get("p", 0)
    }
    
    # Determine click coordinates
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
        print(f"[COMBAT] Using rect coordinates: ({cx}, {cy})")
    elif isinstance(npc.get("canvas", {}).get("x"), (int, float)) and isinstance(npc.get("canvas", {}).get("y"), (int, float)):
        cx, cy = int(npc.get("canvas", {}).get("x")), int(npc.get("canvas", {}).get("y"))
        anchor = {}
        point = {"x": cx, "y": cy}
        print(f"[COMBAT] Using canvas coordinates: ({cx}, {cy})")
    else:
        print(f"[COMBAT] No valid coordinates found for NPC")
        return None
    
    # Find the index of the 'Attack' action
    actions = npc.get("actions", [])
    attack_index = None
    for i, action in enumerate(actions):
        if action and action.lower() == "attack":
            attack_index = i
            break
    
    if attack_index is None:
        print(f"[COMBAT] 'Attack' action not found in NPC actions: {actions}")
        return None
    
    # Create the attack step
    if attack_index == 0:
        # Attack is the default action - use left click
        print(f"[COMBAT] Using left-click for attack (index 0)")
        step = emit({
            "action": "npc-attack",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "npc", "name": npc.get("name"), **anchor, "world": world_coords},
        })
    else:
        # Attack is not default - use context menu
        print(f"[COMBAT] Using context menu for attack (index {attack_index})")
        step = emit({
            "action": "npc-attack-context",
            "click": {
                "type": "context-select",
                "option": "Attack",
                "x": cx,
                "y": cy,
                "row_height": 16,
                "start_dy": 18,
                "open_delay_ms": 120
            },
            "target": {"domain": "npc", "name": npc.get("name"), **anchor, "world": world_coords} if rect else {"domain": "npc", "name": npc.get("name"), "world": world_coords},
            "anchor": point
        })
    
    # Execute the attack with camera integration
    result = dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)
    
    if result:
        # Wait briefly to verify the attack was successful
        time.sleep(0.2)
        print(f"[COMBAT] Attack command executed successfully")
    
    return result
