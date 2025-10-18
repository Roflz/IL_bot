from __future__ import annotations
from typing import Optional
import time

from .travel import _first_blocking_door_from_waypoints
from ..helpers.runtime_utils import ipc

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
        time.sleep(0.5)  # Brief delay before retry
    
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
        time.sleep(0.2)
        print(f"[COMBAT] Attack command executed successfully")
    
    return result


def select_combat_style_for_training() -> None:
    """Select combat style based on current skill levels and training goals."""
    from .combat_interface import select_combat_style, current_combat_style
    from .player import get_skill_level
    
    # Get current skill levels
    attack_level = get_skill_level("attack")
    defence_level = get_skill_level("defence")
    strength_level = get_skill_level("strength")
    
    # Get current combat style
    current_style = current_combat_style()
    
    # Check if we should switch away from current style
    should_switch = False
    current_skill_level = 0
    
    if current_style == 0:  # Attack style
        current_skill_level = attack_level
        should_switch = (attack_level % 5 == 0 or attack_level >= 40)
    elif current_style == 1:  # Strength style
        current_skill_level = strength_level
        should_switch = (strength_level % 5 == 0)
    elif current_style == 3:  # Defence style
        current_skill_level = defence_level
        should_switch = (defence_level % 5 == 0 or defence_level >= 10)
    
    # Determine target style based on 5-level increments and max levels
    target_style = 1  # Default to Strength
    reason = f"Defaulting to Strength training (level {strength_level})"
    
    # Calculate which skills need training (not at 5-level increments)
    defence_needs_training = defence_level < 10 and (defence_level % 5 != 0)
    attack_needs_training = attack_level < 40 and (attack_level % 5 != 0)
    strength_needs_training = strength_level % 5 != 0
    
    # Priority 1: Train Defence if it needs training and is below max (10)
    if defence_needs_training and defence_level < 10:
        target_style = 3
        reason = f"Training Defence (level {defence_level}, needs 5-level increment)"
    # Priority 2: Train Attack if it needs training and is below max (40)
    elif attack_needs_training and attack_level < 40:
        target_style = 0
        reason = f"Training Attack (level {attack_level}, needs 5-level increment)"
    # Priority 3: Train Strength if it needs training
    elif strength_needs_training:
        target_style = 1
        reason = f"Training Strength (level {strength_level}, needs 5-level increment)"
    # Priority 4: If all skills are at 5-level increments, default to Strength
    else:
        target_style = 1
        reason = f"Training Strength (all skills at 5-level increments: Def {defence_level}, Att {attack_level}, Str {strength_level})"

    # Only switch if we need to change styles or if we should switch based on current skill level
    if should_switch and current_style != target_style:
        print(f"[COMBAT] {reason}")
        select_combat_style(target_style)
    else:
        print(f"[COMBAT] Keeping current style {current_style} (no switch needed)")
