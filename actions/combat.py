from __future__ import annotations
from typing import Optional

from .combat_interface import current_combat_style, select_combat_style
from .player import get_skill_level
from .travel import _first_blocking_door_from_waypoints
from helpers.runtime_utils import ipc
from helpers.utils import sleep_exponential


def attack_closest(npc_name: str | list) -> Optional[dict]:
    """
    Find the closest NPC with the given name(s) that is not in combat and attack it.
    
    Args:
        npc_name: Name(s) of the NPC(s) to attack (partial match allowed)
                 Can be a single string or a list of strings
        
    Returns:
        UI dispatch result if successful, None if failed
    """
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
        # Try each NPC name in the list
        for npc_name_to_try in npc_names:
            # Use optimized find_npc command to get closest NPC
            npc_resp = ipc.get_npcs(npc_name_to_try)
            
            if not npc_resp or not npc_resp.get("ok"):
                print(f"[COMBAT] No NPC found with name containing '{npc_name_to_try}'")
                continue  # Try next name in the list
            
            # Get the closest NPC from the npcs array
            npcs = npc_resp.get("npcs", [])
            if not npcs:
                print(f"[COMBAT] No NPCs in response for '{npc_name_to_try}'")
                continue  # Try next name in the list
            
            # Find the closest NPC (they should be sorted by distance)
            npc = npcs[0]
            print(f"[COMBAT] Found NPC: {npc.get('name')} at distance {npc.get('distance')}")
            
            # Check if NPC is already in combat
            if _is_npc_in_combat(npc):
                print(f"[COMBAT] NPC {npc.get('name')} is already in combat, skipping")
                continue  # Try next name in the list
            
            # Check if NPC has 'Attack' action available
            actions = npc.get("actions", [])
            if "Attack" not in actions:
                print(f"[COMBAT] NPC {npc.get('name')} does not have 'Attack' action available")
                continue  # Try next name in the list
            
            # Attack the NPC
            result = _attack_npc(npc)
            if result:
                print(f"[COMBAT] Successfully attacked {npc.get('name')}")
                return result
            else:
                print(f"[COMBAT] Failed to attack {npc.get('name')}, trying next NPC...")
                continue  # Try next name in the list
        
        # If we get here, no NPCs were successfully attacked in this attempt
        print(f"[COMBAT] No attackable NPCs found in attempt {attempt + 1}, retrying...")
        sleep_exponential(0.3, 0.8, 1.2)  # Brief delay before retry
    
    print(f"[COMBAT] Failed to attack any NPC from list {npc_names} after {max_retries} attempts")
    return None


def _is_npc_in_combat(npc: dict) -> bool:
    """
    Check if an NPC is currently in combat using the IPC response data.
    
    Args:
        npc: NPC data dictionary from IPC response
        
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


def _attack_npc(npc: dict) -> Optional[dict]:
    """
    Perform the actual attack action on an NPC with pathing and door handling.
    
    Args:
        npc: NPC data dictionary
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    # ipc_path is now available through the global ipc instance
    from .travel import _handle_door_opening
    from ..services.click_with_camera import click_npc_with_camera
    
    # 1) Check for doors on the path to the NPC
    gx, gy = npc.get("world", {}).get("x"), npc.get("world", {}).get("y")
    if isinstance(gx, int) and isinstance(gy, int):
        wps, dbg_path = ipc.path(goal=(gx, gy))
        door_plan = _first_blocking_door_from_waypoints(wps)
        if door_plan:
            # Handle door opening with retry logic
            if not _handle_door_opening(door_plan, wps):
                print(f"[COMBAT] Failed to open door on path to NPC")
                return None
    
    # Get NPC world coordinates
    world_coords = {
        "x": npc.get("world", {}).get("x"), 
        "y": npc.get("world", {}).get("y"), 
        "p": npc.get("world", {}).get("p", 0)
    }
    
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
    
    # Use click_npc_with_camera for the attack
    npc_name = npc.get("name", "Unknown")
    action = "Attack" if attack_index != 0 else None
    
    print(f"[COMBAT] Attacking {npc_name} using click_npc_with_camera")
    result = click_npc_with_camera(
        npc_name=npc_name,
        action=action,
        world_coords=world_coords,
        aim_ms=420
    )
    
    if result:
        # Wait briefly to verify the attack was successful
        sleep_exponential(0.1, 0.3, 1.5)
        print(f"[COMBAT] Attack command executed successfully")
    
    return result

def select_combat_style_for_training():
    """
    Uses:
        attack_level   = get_skill_level("attack")
        defence_level  = get_skill_level("defence")
        strength_level = get_skill_level("strength")
        current_style  = current_combat_style()
        select_combat_style(idx)

    Style mapping assumed from your notes:
        0 = Attack
        1 = Strength
        3 = Defence
    """
    # --- read current levels ---
    attack_level   = get_skill_level("attack")
    defence_level  = get_skill_level("defence")
    strength_level = get_skill_level("strength")

    # --- constants/rules ---
    # priority: strength > attack > defence
    CAPS = {"strength": None, "attack": 20, "defence": 5}
    STYLE = {"attack": 0, "strength": 1, "defence": 3}

    def ceil5(x: int) -> int:
        return ((x + 4) // 5) * 5

    def next_block_target(x: int) -> int:
        return x + 5 if x % 5 == 0 else ceil5(x)

    levels = {
        "attack": attack_level,
        "defence": defence_level,
        "strength": strength_level,
    }

    # 1) finish any mid-block (not multiple of 5) in priority order, within caps
    for skill in ("strength", "attack", "defence"):
        lvl = levels[skill]
        cap = CAPS[skill]
        if cap is not None and lvl >= cap:
            continue
        if lvl % 5 != 0:
            target = ceil5(lvl)
            if cap is not None:
                target = min(target, cap)
            if target > lvl:
                desired_style = STYLE[skill]
                if current_combat_style() != desired_style:
                    select_combat_style(desired_style)
                return skill, target

    # 2) start a new 5-level block in priority order, within caps
    for skill in ("strength", "attack", "defence"):
        lvl = levels[skill]
        cap = CAPS[skill]
        if cap is not None and lvl >= cap:
            continue
        target = next_block_target(lvl)
        if cap is not None:
            target = min(target, cap)
        if target > lvl:
            desired_style = STYLE[skill]
            if current_combat_style() != desired_style:
                select_combat_style(desired_style)
            return skill, target

    # 3) everything that can be capped is capped; keep Strength selected (no-op target)
    desired_style = STYLE["strength"]
    if current_combat_style() != desired_style:
        select_combat_style(desired_style)
    return "strength", strength_level
