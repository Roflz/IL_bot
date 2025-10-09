"""
Centralized click with camera functionality.
Handles the 3-step process: camera movement, fresh coordinates, click.
"""
import time
import random

from ..helpers.rects import unwrap_rect, rect_center_xy
from ..helpers.utils import clean_rs
from ..services.camera_integration import aim_midtop_at_world
from ..helpers.runtime_utils import ui, ipc, dispatch
from ..helpers.navigation import _merge_door_into_projection


def click_object_with_camera(
    object_name: str, 
    action: str = None,
    world_coords: dict = None,
    door_plan: dict = None,  # NEW: For door-specific handling
    aim_ms: int = 420
) -> dict | None:
    """
    Click an object with camera movement and fresh coordinate recalculation.
    """
    import time
    start_time = time.time()
    print(f"[CLICK_OBJECT_TIMING] Starting click_object_with_camera for {object_name}")
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for {object_name}")
        return None
    
    # STEP 1: Move camera to aim at object
    step_start = time.time()
    aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
    print(f"[CLICK_OBJECT_TIMING] Camera movement took {time.time() - step_start:.3f}s")
    
    # Special handling for doors - use door plan data directly
    step_start = time.time()
    if "door" in object_name.lower() and door_plan:
        print(f"[CLICK_WITH_CAMERA] Using door plan data for door interaction")
        target = door_plan.get("door", {})
        if not target:
            print(f"[CLICK_WITH_CAMERA] No door data in door plan")
            return None
    else:
        # Normal object finding for non-doors
        obj_resp = ipc.find_object(object_name, ["GAME", "WALL"])
        
        if not obj_resp or not obj_resp.get("ok") or not obj_resp.get("found"):
            print(f"[CLICK_WITH_CAMERA] Could not find {object_name} after camera movement")
            return None
        
        target = obj_resp.get("object")
        
        if not target:
            print(f"[CLICK_WITH_CAMERA] Could not find {object_name} after camera movement")
            return None
    print(f"[CLICK_OBJECT_TIMING] Object finding took {time.time() - step_start:.3f}s")
    
    # Get FRESH coordinates
    step_start = time.time()
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    obj_name = target.get("name") or object_name
    
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
        cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        print(f"[CLICK_WITH_CAMERA] Could not get fresh coordinates for {object_name}")
        return None
    print(f"[CLICK_OBJECT_TIMING] Coordinate processing took {time.time() - step_start:.3f}s")
    
    # Hover at the target coordinates
    step_start = time.time()
    hover_result = ipc.click(cx, cy, hover_only=True)
    print(f"[CLICK_OBJECT_TIMING] Hover click took {time.time() - step_start:.3f}s")
    
    if not hover_result.get("ok"):
        return None

    # Small delay to ensure hover is registered
    import time
    time.sleep(0.1)

    # Verify "Walk here" is available
    step_start = time.time()
    action_index = verify_action_available(action, object_name)
    print(f"[CLICK_OBJECT_TIMING] Action verification took {time.time() - step_start:.3f}s")
    
    if not action_index and not action_index == 0:
        return None
    
    if not action or action_index is None or action_index == 0:
        step = {
            "action": "click-object",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "object", "name": obj_name, **anchor, "world": world_coords},
        }
    else:
        step = {
            "action": "click-object-context",
            "click": {
                "type": "context-select",
                "index": int(action_index),
                "x": point["x"],
                "y": point["y"],
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "target": ({"domain": "object", "name": obj_name, **anchor, "world": world_coords}
                       if rect else {"domain": "object", "name": obj_name, "world": world_coords}),
            "anchor": point,
        }

    step_start = time.time()
    result = dispatch(step)
    print(f"[CLICK_OBJECT_TIMING] Dispatch took {time.time() - step_start:.3f}s")
    print(f"[CLICK_OBJECT_TIMING] Total click_object_with_camera took {time.time() - start_time:.3f}s")
    
    if result:
        print(f"[CLICK] {object_name}")
    return result


def click_npc_with_camera(
    npc_name: str,
    action: str = None,
    world_coords: dict = None,
    aim_ms: int = 420
) -> dict | None:
    """
    Click an NPC with camera movement and fresh coordinate recalculation.
    """
    import time
    start_time = time.time()
    print(f"[CLICK_NPC_TIMING] Starting click_npc_with_camera for {npc_name}")
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for {npc_name}")
        return None
    
    # STEP 1: Move camera to aim at NPC
    step_start = time.time()
    aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
    print(f"[CLICK_NPC_TIMING] Camera movement took {time.time() - step_start:.3f}s")
    
    # STEP 2: After camera movement, get FRESH coordinates
    # No longer using payload - get NPC data directly
    step_start = time.time()
    npc_resp = ipc.find_npc(npc_name)
    print(f"[CLICK_NPC_TIMING] NPC finding took {time.time() - step_start:.3f}s")
    
    if not npc_resp or not npc_resp.get("ok") or not npc_resp.get("found"):
        print(f"[CLICK_WITH_CAMERA] Could not find {npc_name} after camera movement")
        return None
    
    target = npc_resp.get("npc")
    
    if not target:
        print(f"[CLICK_WITH_CAMERA] Could not find {npc_name} after camera movement")
        return None
    
    # Get FRESH coordinates
    step_start = time.time()
    rect = unwrap_rect(target.get("clickbox")) or unwrap_rect(target.get("bounds"))
    npc_name_fresh = target.get("name") or npc_name
    
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
        cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        print(f"[CLICK_WITH_CAMERA] Could not get fresh coordinates for {npc_name}")
        return None
    print(f"[CLICK_NPC_TIMING] Coordinate processing took {time.time() - step_start:.3f}s")

    # Hover at the target coordinates
    step_start = time.time()
    hover_result = ipc.click(cx, cy, hover_only=True)
    print(f"[CLICK_NPC_TIMING] Hover click took {time.time() - step_start:.3f}s")
    
    if not hover_result.get("ok"):
        return None

    # Small delay to ensure hover is registered
    import time
    time.sleep(0.1)

    # Verify "Walk here" is available
    step_start = time.time()
    action_index = verify_action_available(action, npc_name)
    print(f"[CLICK_NPC_TIMING] Action verification took {time.time() - step_start:.3f}s")
    
    if not action_index and not action_index == 0:
        return None
    
    if not action or action_index is None or action_index == 0:
        step = {
            "action": "click-npc",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
        }
    else:
        step = {
            "action": "click-npc-context",
            "click": {
                "type": "context-select",
                "index": int(action_index),
                "x": point["x"],
                "y": point["y"],
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "option": action,
            "target": {"domain": "npc", "name": npc_name_fresh, **anchor, "world": world_coords},
            "anchor": point,
        }
    
    # Click with fresh coordinates
    step_start = time.time()
    result = dispatch(step)
    print(f"[CLICK_NPC_TIMING] Dispatch took {time.time() - step_start:.3f}s")
    print(f"[CLICK_NPC_TIMING] Total click_npc_with_camera took {time.time() - start_time:.3f}s")
    
    if result:
        print(f"[CLICK] {npc_name}")
    return result


def verify_walk_here_available():
    """Check if 'Walk here' is the top menu option"""
    menu_response = ipc.get_menu()
    if not menu_response.get("ok") or not menu_response.get("open"):
        return False
    
    entries = menu_response.get("entries", [])
    if not entries:
        return False
    
    # Find the entry with visualIndex: 0 (top option)
    top_option = None
    for entry in entries:
        if entry.get("visualIndex") == 0:
            top_option = entry
            break
    
    if not top_option:
        return False
    
    # Check if top option is "Walk here"
    return (top_option.get("option") == "Walk here" and 
            top_option.get("type") == "WALK")


def verify_action_available(action: str, target: str):
    """Check if a specific action and target combination is available in the menu"""
    menu_response = ipc.get_menu()
    if not menu_response.get("ok") or not menu_response.get("open"):
        return False
    
    entries = menu_response.get("entries", [])
    if not entries:
        return False
    
    # Look for entry where clean_rs(target) matches and option matches action
    for entry in entries:
        entry_target = entry.get("target", "")
        entry_option = entry.get("option", "")
        
        # Clean the target from the menu entry
        clean_target = clean_rs(entry_target)
        
        # Check if both action and target match
        if clean_target == target and entry_option == action:
            return entry.get("visualIndex")
    
    return False


def click_ground_with_camera(
    world_coords: dict,
    description: str = "Move",
    aim_ms: int = 700,
    waypoint_path: list = None
) -> dict | None:
    """
    Click ground with camera movement and fresh coordinate recalculation.
    If verify_walk_here_available returns false, retry with different tiles from waypoint path.
    """
    import time
    start_time = time.time()
    print(f"[CLICK_GROUND_TIMING] Starting click_ground_with_camera for {description}")
    
    if not world_coords:
        print(f"[CLICK_WITH_CAMERA] No world coordinates provided for ground click")
        return None
    
    # Try pathfinding with decreasing max waypoints (20 down to 15)
    max_attempts = 3  # 20, 19, 18, 17, 16, 15
    initial_max_wps = 18
    
    for attempt in range(max_attempts):
        attempt_start = time.time()
        current_max_wps = initial_max_wps - attempt
        print(f"[CLICK_GROUND_TIMING] Attempt {attempt + 1}/{max_attempts}: max_wps={current_max_wps}")
        
        # Re-run pathfinding logic like in go_to to get fresh waypoint
        step_start = time.time()
        wps, dbg_path = ipc.path(rect=(world_coords['x']-1, world_coords['x']+1, world_coords['y']-1, world_coords['y']+1), max_wps=current_max_wps, visualize=False)
        print(f"[CLICK_GROUND_TIMING] Path generation took {time.time() - step_start:.3f}s")
        
        if not wps:
            print(f"[CLICK_GROUND_TIMING] No waypoints generated, trying next attempt")
            continue
        
        step_start = time.time()
        proj, dbg_proj = ipc.project_many(wps)
        proj = _merge_door_into_projection(wps, proj)
        print(f"[CLICK_GROUND_TIMING] Projection and door merge took {time.time() - step_start:.3f}s")
        
        usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
        if not usable:
            print(f"[CLICK_GROUND_TIMING] No usable waypoints after projection, trying next attempt")
            continue
        
        # Pick from the last 3 waypoints (or all if less than 3) - same logic as go_to
        if len(usable) <= 3:
            chosen = random.choice(usable)
        else:
            chosen = random.choice(usable[-3:])
        
        coords = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
        print(f"[CLICK_GROUND_TIMING] Selected coordinates: {coords}")
        
        # STEP 1: Move camera to aim at ground point
        step_start = time.time()
        aim_midtop_at_world(coords['x'], coords['y'], max_ms=aim_ms)
        print(f"[CLICK_GROUND_TIMING] Camera movement took {time.time() - step_start:.3f}s")
        
        # For ground clicks, we need to project the world coordinates to screen coordinates
        step_start = time.time()
        proj, _ = ipc.project_many([{"x": coords['x'], "y": coords['y'], "p": coords.get('p', 0)}])
        print(f"[CLICK_GROUND_TIMING] Projection took {time.time() - step_start:.3f}s")
        
        if not proj or not isinstance(proj[0], dict) or not proj[0].get("canvas"):
            print(f"[CLICK_GROUND_TIMING] Projection failed, trying next attempt")
            continue  # Try next attempt
        
        # Get FRESH screen coordinates
        fresh_coords = proj[0]["canvas"]
        cx = int(fresh_coords["x"])
        cy = int(fresh_coords["y"])

        # Hover at the target coordinates
        step_start = time.time()
        hover_result = ipc.click(cx, cy, hover_only=True)
        print(f"[CLICK_GROUND_TIMING] Hover click took {time.time() - step_start:.3f}s")
        
        if not hover_result.get("ok"):
            print(f"[CLICK_GROUND_TIMING] Hover failed, trying next attempt")
            continue  # Try next attempt
        
        # Small delay to ensure hover is registered
        time.sleep(0.1)
        
        # Verify "Walk here" is available
        step_start = time.time()
        walk_here_available = verify_walk_here_available()
        print(f"[CLICK_GROUND_TIMING] Walk here verification took {time.time() - step_start:.3f}s")
        
        if not walk_here_available:
            print(f"[CLICK_GROUND_TIMING] Walk here not available, trying next attempt")
            continue  # Try next attempt
        
        step = {
            "action": "click-ground",
            "description": description,
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "ground", "name": f"Ground→{description}", "world": coords},
        }
        
        # Click with fresh coordinates
        step_start = time.time()
        result = dispatch(step)
        print(f"[CLICK_GROUND_TIMING] Dispatch took {time.time() - step_start:.3f}s")
        print(f"[CLICK_GROUND_TIMING] Attempt {attempt + 1} total took {time.time() - attempt_start:.3f}s")
        
        if result:
            print(f"[CLICK] Ground")
            print(f"[CLICK_GROUND_TIMING] Total click_ground_with_camera took {time.time() - start_time:.3f}s")
            return result
    
    # All attempts failed - try context-click "Walk here" on original tile as fallback
    print(f"[CLICK_WITH_CAMERA] All attempts failed, trying context-click 'Walk here' on original tile")
    fallback_start = time.time()
    
    # Move camera back to original coordinates
    step_start = time.time()
    aim_midtop_at_world(world_coords['x'], world_coords['y'], max_ms=aim_ms)
    print(f"[CLICK_GROUND_TIMING] Fallback camera movement took {time.time() - step_start:.3f}s")
    
    # Project original coordinates to screen
    step_start = time.time()
    proj, _ = ipc.project_many([{"x": world_coords['x'], "y": world_coords['y'], "p": world_coords.get('p', 0)}])
    print(f"[CLICK_GROUND_TIMING] Fallback projection took {time.time() - step_start:.3f}s")
    
    if proj and isinstance(proj[0], dict) and proj[0].get("canvas"):
        # Get screen coordinates for original tile
        fresh_coords = proj[0]["canvas"]
        cx = int(fresh_coords["x"])
        cy = int(fresh_coords["y"])
        
        # Hover at the target coordinates
        step_start = time.time()
        hover_result = ipc.click(cx, cy, hover_only=True)
        print(f"[CLICK_GROUND_TIMING] Fallback hover click took {time.time() - step_start:.3f}s")
        
        if hover_result.get("ok"):
            # Small delay to ensure hover is registered
            time.sleep(0.1)
            
            # Try context-click "Walk here"
            step = {
                "action": "click-ground-context",
                "description": f"{description} (context-click fallback)",
                "click": {
                    "type": "context-select",
                    "index": 0,  # "Walk here" is typically the first option
                    "x": cx,
                    "y": cy,
                    "row_height": 16,
                    "start_dy": 10,
                    "open_delay_ms": 120
                },
                "option": "Walk here",
                "target": {"domain": "ground", "name": f"Ground→{description}", "world": world_coords},
                "anchor": {"x": cx, "y": cy},
            }
            
            step_start = time.time()
            result = dispatch(step)
            print(f"[CLICK_GROUND_TIMING] Fallback dispatch took {time.time() - step_start:.3f}s")
            print(f"[CLICK_GROUND_TIMING] Fallback total took {time.time() - fallback_start:.3f}s")
            
            if result:
                print(f"[CLICK] Ground (context-click fallback)")
                print(f"[CLICK_GROUND_TIMING] Total click_ground_with_camera (with fallback) took {time.time() - start_time:.3f}s")
                return result
    
    print(f"[CLICK_WITH_CAMERA] Context-click fallback also failed")
    print(f"[CLICK_GROUND_TIMING] Total click_ground_with_camera (failed) took {time.time() - start_time:.3f}s")
    return None