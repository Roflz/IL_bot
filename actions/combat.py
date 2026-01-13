from __future__ import annotations
from typing import Optional

from .combat_interface import current_combat_style, select_combat_style
from .player import get_skill_level
from .travel import _first_blocking_door_from_waypoints
from helpers.runtime_utils import ipc, dispatch
from helpers.ipc import get_last_interaction
from helpers.utils import clean_rs, rect_beta_xy, sleep_exponential
from helpers import unwrap_rect
from services.camera_integration import aim_midtop_at_world, aim_camera_at_target


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
    aim_ms = 420
    camera_retry_directions = [None, "LEFT", "RIGHT"]
    camera_retry_duration = 0.5  # seconds
    
    for attempt in range(max_retries):
        # Try each NPC name in the list
        # Use optimized find_npc command to get closest NPC
        for npc_name_to_try in npc_names:
            for cam_dir in camera_retry_directions:
                # Optional small camera nudge before attempting aim+click
                if cam_dir is not None:
                    try:
                        ipc.key_press(cam_dir)
                        sleep_exponential(camera_retry_duration * 0.8, camera_retry_duration * 1.2, 1.0)
                        ipc.key_release(cam_dir)
                    except Exception:
                        pass

                # Multiple attempts within the same camera direction
                max_click_attempts = 3
                for _ in range(max_click_attempts):
                    npc_resp = ipc.get_npcs(npc_name_to_try)
                    if not npc_resp or not npc_resp.get("ok"):
                        print(f"[COMBAT] No NPC found with name containing '{npc_name_to_try}'")
                        break

                    npcs = npc_resp.get("npcs", []) or []
                    if not npcs:
                        print(f"[COMBAT] No NPCs in response for '{npc_name_to_try}'")
                        break

                    # Filter candidates first (these are already distance-sorted from IPC)
                    candidates = []
                    for n in npcs:
                        if _is_npc_in_combat(n):
                            continue
                        if _is_npc_known_dead(n):
                            continue
                        actions = n.get("actions", []) or []
                        if "Attack" not in actions:
                            continue
                        candidates.append(n)

                    if not candidates:
                        print(f"[COMBAT] No filtered candidates for '{npc_name_to_try}'")
                        break

                    # Pick the closest filtered candidate
                    chosen = candidates[0]
                    print(f"[COMBAT] Chosen NPC: {chosen.get('name')} at distance {chosen.get('distance')}")

                    # --- Door/path handling (same logic as old _attack_npc) ---
                    from .travel import _handle_door_opening

                    gx, gy = (chosen.get("world", {}) or {}).get("x"), (chosen.get("world", {}) or {}).get("y")
                    if isinstance(gx, int) and isinstance(gy, int):
                        wps, _dbg_path = ipc.path(goal=(gx, gy))
                        door_plan = _first_blocking_door_from_waypoints(wps)
                        if door_plan:
                            if not _handle_door_opening(door_plan, wps):
                                print(f"[COMBAT] Failed to open door on path to NPC")
                                continue

                    # --- Camera aim ---
                    world_coords = chosen.get("world") or {}
                    if not (isinstance(world_coords.get("x"), int) and isinstance(world_coords.get("y"), int)):
                        continue
                    
                    # Use new camera system
                    from actions import player
                    player_x = player.get_x()
                    player_y = player.get_y()
                    if isinstance(player_x, int) and isinstance(player_y, int):
                        dx = abs(world_coords["x"] - player_x)
                        dy = abs(world_coords["y"] - player_y)
                        distance = dx + dy  # Manhattan distance
                    else:
                        distance = None
                    
                    aim_camera_at_target(
                        target_world_coords=world_coords,
                        mode=None,  # Auto-detect
                        action_type="click_npc",
                        distance_to_target=distance
                    )

                    # --- Re-acquire after camera movement for FRESH screen bounds ---
                    npc_resp2 = ipc.get_npcs(npc_name_to_try)
                    if not npc_resp2 or not npc_resp2.get("ok"):
                        continue
                    npcs2 = npc_resp2.get("npcs", []) or []

                    # Re-apply the same filters (post-aim)
                    candidates2 = []
                    for n in npcs2:
                        if _is_npc_in_combat(n):
                            continue
                        if _is_npc_known_dead(n):
                            continue
                        actions = n.get("actions", []) or []
                        if "Attack" not in actions:
                            continue
                        candidates2.append(n)

                    if not candidates2:
                        continue

                    # Prefer the same tile as the pre-aim chosen NPC; otherwise nearest to that tile
                    tx, ty = world_coords["x"], world_coords["y"]
                    same_tile = None
                    for n in candidates2:
                        w = n.get("world") or {}
                        if w.get("x") == tx and w.get("y") == ty and w.get("p", 0) == world_coords.get("p", 0):
                            same_tile = n
                            break

                    if same_tile is not None:
                        target = same_tile
                    else:
                        def manhattan_to_target(n: dict) -> int:
                            w = n.get("world") or {}
                            nx, ny = w.get("x"), w.get("y")
                            if not (isinstance(nx, int) and isinstance(ny, int)):
                                return 999999
                            return abs(nx - tx) + abs(ny - ty)

                        target = min(candidates2, key=manhattan_to_target)

                    # Determine whether "Attack" is top option (left click) or needs context action
                    actions = target.get("actions", []) or []
                    attack_index = None
                    for i, a in enumerate(actions):
                        if a and a.lower() == "attack":
                            attack_index = i
                            break
                    if attack_index is None:
                        continue
                    action = "Attack" if attack_index != 0 else None

                    npc_name_fresh = target.get("name") or npc_name_to_try

                    # Pick a click point from fresh bounds/canvas
                    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
                    anchor = {}
                    point = None

                    if rect and rect.get("width", 0) > 0 and rect.get("height", 0) > 0:
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
                    else:
                        canvas = target.get("canvas") if isinstance(target.get("canvas"), dict) else None
                        tile_canvas = target.get("tileCanvas") if isinstance(target.get("tileCanvas"), dict) else None
                        if isinstance((canvas or {}).get("x"), (int, float)) and isinstance((canvas or {}).get("y"), (int, float)):
                            point = {"x": int(canvas["x"]), "y": int(canvas["y"])}
                        elif isinstance((tile_canvas or {}).get("x"), (int, float)) and isinstance((tile_canvas or {}).get("y"), (int, float)):
                            point = {"x": int(tile_canvas["x"]), "y": int(tile_canvas["y"])}
                        else:
                            proj = ipc.project_world_tile(int(world_coords["x"]), int(world_coords["y"])) or {}
                            if proj.get("ok") and proj.get("onscreen") and isinstance(proj.get("canvas"), dict):
                                point = {"x": int(proj["canvas"]["x"]), "y": int(proj["canvas"]["y"])}

                    if not point:
                        continue

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
                            "exact_match": False,
                        },
                        "option": action,
                        "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
                        "anchor": point,
                    }

                    result = dispatch(step)
                    if not result:
                        continue

                    last = get_last_interaction() or {}
                    clean_action = clean_rs(last.get("action", ""))
                    if action is None:
                        action_match = True
                    else:
                        action_match = action.lower() in clean_action.lower()

                    target_name = clean_rs(last.get("target_name") or last.get("target") or "")
                    target_match = npc_name_fresh.lower() in target_name.lower()

                    if action_match and target_match:
                        print(f"[COMBAT] Successfully attacked {npc_name_fresh}")
                        return result

                    print(f"[COMBAT] Incorrect interaction, retrying...")
        
        # If we get here, no NPCs were successfully attacked in this attempt
        print(f"[COMBAT] No attackable NPCs found in attempt {attempt + 1}, retrying...")
        sleep_exponential(0.3, 0.8, 1.2)  # Brief delay before retry
    
    print(f"[COMBAT] Failed to attack any NPC from list {npc_names} after {max_retries} attempts")
    return None


def _is_npc_known_dead(npc: dict) -> bool:
    """
    RuneLite only exposes NPC healthRatio/healthScale when a health bar is active.
    If healthScale > 0 then health is known; treat healthRatio <= 0 as dead/unattackable.
    For unknown health (-1/-1), do NOT filter the NPC out.
    """
    hr = npc.get("healthRatio", None)
    hs = npc.get("healthScale", None)

    if isinstance(hr, int) and isinstance(hs, int) and hs > 0:
        return hr <= 0

    return False


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
