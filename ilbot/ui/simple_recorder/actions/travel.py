# ilbot/ui/simple_recorder/actions/travel.py
import random
import time

from .runtime import emit
from ..helpers.camera import prepare_for_walk
from ..helpers.context import get_payload, get_ui
from ..helpers.navigation import get_nav_rect, closest_bank_key, bank_rect, player_in_rect, _merge_door_into_projection
from ..helpers.ipc import ipc_path, ipc_project_many
from ..services.camera_integration import dispatch_with_camera


def _get_hovered_tile_info(payload: dict) -> dict | None:
    """Get information about the currently hovered tile from payload."""
    return payload.get("hoveredTile")


def _validate_hovered_tile(expected_world_x: int, expected_world_y: int, payload: dict, tolerance: int = 3) -> bool:
    """
    Validate that the hovered tile matches the expected coordinates within tolerance.
    
    Args:
        expected_world_x: Expected world X coordinate
        expected_world_y: Expected world Y coordinate  
        payload: Game state payload
        tolerance: Maximum distance allowed between expected and hovered coordinates
        
    Returns:
        True if hovered tile is within tolerance of expected coordinates
    """
    hovered_tile = _get_hovered_tile_info(payload)
    if not hovered_tile:
        return False
        
    hovered_x = hovered_tile.get("worldX")
    hovered_y = hovered_tile.get("worldY")
    
    if not isinstance(hovered_x, int) or not isinstance(hovered_y, int):
        return False
    
    # Calculate distance between expected and hovered coordinates
    distance = _calculate_distance(expected_world_x, expected_world_y, hovered_x, hovered_y)
    return distance <= tolerance


def _get_expected_action_for_tile(world_x: int, world_y: int, payload: dict) -> str | None:
    """
    Determine the expected action for a tile based on its contents.
    
    Args:
        world_x: World X coordinate
        world_y: World Y coordinate
        payload: Game state payload
        
    Returns:
        Expected action string ("Walk here", "Open", etc.) or None
    """
    hovered_tile = _get_hovered_tile_info(payload)
    if not hovered_tile:
        return None
        
    # Check if this is the expected tile
    if not _validate_hovered_tile(world_x, world_y, payload):
        return None
    
    # Check for doors/gates on the tile
    game_objects = hovered_tile.get("gameObjects", [])
    for obj in game_objects:
        actions = obj.get("actions", [])
        if actions and "Open" in actions:
            return "Open"
    
    # Default to walk here for ground tiles
    return "Walk here"


def _verify_click_action(expected_action: str, payload: dict) -> bool:
    """
    Verify that the last interaction matches the expected action.
    
    Args:
        expected_action: Expected action string
        payload: Game state payload
        
    Returns:
        True if the last interaction matches expected action
    """
    last_interaction = payload.get("lastInteraction", {})
    if not last_interaction:
        return False
        
    action = last_interaction.get("action", "")
    return expected_action.lower() in action.lower()


def _hover_over_tile(step_dict: dict, ui) -> None:
    """
    Hover the mouse over the target tile coordinates.
    
    Args:
        step_dict: Step dictionary containing click coordinates
        ui: UI instance for dispatching hover actions
    """
    click_info = step_dict.get("click", {})
    if not click_info:
        return
    
    # Extract coordinates from the step
    x = click_info.get("x")
    y = click_info.get("y")
    
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        print(f"[DEBUG] No valid coordinates found for hover: {click_info}")
        return
    
    # Try different approaches to move the mouse to the target location
    try:
        # Method 1: Try using the IPC directly to move mouse
        from ..helpers.context import get_payload
        payload = get_payload()
        if payload and hasattr(ui, 'ipc') and hasattr(ui.ipc, 'move_mouse'):
            ui.ipc.move_mouse(int(x), int(y))
            print(f"[DEBUG] Moved mouse to coordinates ({x}, {y}) via IPC")
            return
        
        # Method 2: Try a right-click action (which moves mouse but doesn't click)
        hover_step = {
            "action": "right-click",
            "click": {"type": "point", "x": int(x), "y": int(y)},
            "target": step_dict.get("target", {})
        }
        ui.dispatch(hover_step)
        print(f"[DEBUG] Moved mouse via right-click action to ({x}, {y})")
        
    except Exception as e:
        print(f"[DEBUG] Mouse movement failed: {e}")
        # Method 3: Fallback - just wait and hope the coordinates are close enough
        print(f"[DEBUG] Using fallback - no mouse movement, relying on coordinate accuracy")


def _wait_for_camera_and_click(step: dict | list, ui, payload: dict | None = None, aim_ms: int = 450,
                              hover_delay_ms: int = 200, click_delay_ms: int = 100) -> dict | None:
    """
    Improved camera movement and clicking with better timing and validation.
    
    Args:
        step: Action step to execute (dict or list containing dict)
        ui: UI instance
        payload: Game state payload
        aim_ms: Maximum time for camera aiming
        hover_delay_ms: Delay after camera movement before clicking
        click_delay_ms: Additional delay before actual click
        
    Returns:
        Dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload() or {}
    
    # Handle both dict and list formats
    if isinstance(step, list) and len(step) > 0:
        step_dict = step[0]  # Take first element if it's a list
        print(f"[DEBUG] Step was list format, extracted dict: {step_dict.get('action', 'unknown')}")
    elif isinstance(step, dict):
        step_dict = step
        print(f"[DEBUG] Step was dict format: {step_dict.get('action', 'unknown')}")
    else:
        # Fallback to original behavior if format is unexpected
        print(f"[DEBUG] Unexpected step format: {type(step)}, using original dispatch_with_camera")
        return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=aim_ms)
    
    # Extract target information
    target = step_dict.get("target", {})
    world = target.get("world", {})
    world_x = world.get("x")
    world_y = world.get("y")
    
    if not isinstance(world_x, int) or not isinstance(world_y, int):
        # Fallback to original behavior if no world coordinates
        print(f"[DEBUG] No world coordinates found, using original dispatch_with_camera")
        return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=aim_ms)
    
    # Step 1: Move camera to target (asynchronous)
    try:
        from ..services.camera_integration import aim_midtop_at_world
        aim_midtop_at_world(world_x, world_y, max_ms=aim_ms, payload=payload)
    except Exception:
        pass  # Continue even if camera movement fails
    
    # Step 2: Wait for camera movement to settle
    time.sleep(hover_delay_ms / 1000.0)
    
    # Step 3: Skip hover validation for now - just proceed with click
    print(f"[DEBUG] Proceeding with click at target tile ({world_x}, {world_y})")
    fresh_payload = get_payload()
    
    # Step 4: Get expected action (optional validation)
    expected_action = _get_expected_action_for_tile(world_x, world_y, fresh_payload)
    if not expected_action:
        # Default to "Walk here" if we can't determine action
        expected_action = "Walk here"
        print(f"[DEBUG] Couldn't determine expected action, defaulting to 'Walk here'")
    
    # Step 5: Additional delay before clicking
    time.sleep(click_delay_ms / 1000.0)
    
    # Step 6: Get fresh coordinates right before clicking
    fresh_payload = get_payload()
    fresh_hovered_tile = _get_hovered_tile_info(fresh_payload)
    
    if fresh_hovered_tile:
        # Use the currently hovered tile coordinates for the click
        fresh_canvas_x = fresh_hovered_tile.get("canvasX")
        fresh_canvas_y = fresh_hovered_tile.get("canvasY")
        
        if isinstance(fresh_canvas_x, (int, float)) and isinstance(fresh_canvas_y, (int, float)):
            # Update the step with fresh coordinates
            step_dict["click"]["x"] = int(fresh_canvas_x)
            step_dict["click"]["y"] = int(fresh_canvas_y)
            print(f"[DEBUG] Updated click coordinates to fresh hovered tile: ({fresh_canvas_x}, {fresh_canvas_y})")
        else:
            # Try to get fresh coordinates via IPC
            try:
                from ..helpers.ipc import ipc_send
                ipc_result = ipc_send({
                    "cmd": "tilexy",
                    "x": world_x,
                    "y": world_y
                }, fresh_payload)
                
                if ipc_result and ipc_result.get("ok") and ipc_result.get("onscreen"):
                    fresh_canvas = ipc_result.get("canvas", {})
                    fresh_canvas_x = fresh_canvas.get("x")
                    fresh_canvas_y = fresh_canvas.get("y")
                    
                    if isinstance(fresh_canvas_x, (int, float)) and isinstance(fresh_canvas_y, (int, float)):
                        step_dict["click"]["x"] = int(fresh_canvas_x)
                        step_dict["click"]["y"] = int(fresh_canvas_y)
                        print(f"[DEBUG] Updated click coordinates via IPC: ({fresh_canvas_x}, {fresh_canvas_y})")
                    else:
                        print(f"[DEBUG] IPC returned invalid coordinates")
                else:
                    print(f"[DEBUG] IPC failed to get fresh coordinates")
            except Exception as e:
                print(f"[DEBUG] IPC fallback failed: {e}")
    else:
        # Try to get fresh coordinates via IPC as fallback
        try:
            from ..helpers.ipc import ipc_send
            ipc_result = ipc_send({
                "cmd": "tilexy",
                "x": world_x,
                "y": world_y
            }, fresh_payload)
            
            if ipc_result and ipc_result.get("ok") and ipc_result.get("onscreen"):
                fresh_canvas = ipc_result.get("canvas", {})
                fresh_canvas_x = fresh_canvas.get("x")
                fresh_canvas_y = fresh_canvas.get("y")
                
                if isinstance(fresh_canvas_x, (int, float)) and isinstance(fresh_canvas_y, (int, float)):
                    step_dict["click"]["x"] = int(fresh_canvas_x)
                    step_dict["click"]["y"] = int(fresh_canvas_y)
                    print(f"[DEBUG] Updated click coordinates via IPC fallback: ({fresh_canvas_x}, {fresh_canvas_y})")
                else:
                    print(f"[DEBUG] IPC fallback returned invalid coordinates")
            else:
                print(f"[DEBUG] IPC fallback failed, using original coordinates")
        except Exception as e:
            print(f"[DEBUG] IPC fallback failed: {e}")
            print(f"[DEBUG] Using original coordinates")
    
    # Step 7: Execute the click with fresh coordinates
    print(f"[DEBUG] Executing click at ({step_dict['click']['x']}, {step_dict['click']['y']}) with action: {step_dict.get('action', 'unknown')}")
    result = ui.dispatch(step_dict)
    print(f"[DEBUG] Click result: {result}")
    
    # Step 8: Verify the action was performed correctly
    if result:
        time.sleep(0.1)  # Brief wait for action to register
        verify_payload = get_payload()
        action_verified = _verify_click_action(expected_action, verify_payload)
        
        if not action_verified:
            # Debug: Log verification failure
            last_interaction = verify_payload.get("lastInteraction", {})
            actual_action = last_interaction.get("action", "None")
            print(f"[DEBUG] Action verification failed. Expected: '{expected_action}', Got: '{actual_action}'")
            print(f"[DEBUG] Click failed validation - returning None")
            return None
        else:
            print(f"[DEBUG] Action verification successful: '{expected_action}'")
    
    return result


def _is_blocking_door(door: dict) -> bool:
    if not isinstance(door, dict):
        return False
    if not door.get("present"):
        return False
    # If 'closed' missing, be conservative and treat as blocking
    return door.get("closed", True) is True

def _pick_click_from_door(door: dict):
    """Prefer hull center, then tile center; return (x, y) or None."""
    if not isinstance(door, dict):
        return None
    c = door.get("canvas")
    if isinstance(c, dict) and "x" in c and "y" in c:
        return int(c["x"]), int(c["y"])
    tc = door.get("tileCanvas")
    if isinstance(tc, dict) and "x" in tc and "y" in tc:
        return int(tc["x"]), int(tc["y"])
    # As a last resort, if bounds are there, click rect center
    b = door.get("bounds")
    if isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
        return int(b["x"] + b["width"] // 2), int(b["y"] + b["height"] // 2)
    return None

def _first_blocking_door(proj_rows: list[dict], up_to_index: int | None = None) -> dict | None:
    """
    Scan projected rows (which now include door metadata) and return a click plan
    for the earliest blocking door. Treat missing 'closed' as blocking.
    """
    limit = len(proj_rows) if up_to_index is None else max(0, min(up_to_index + 1, len(proj_rows)))

    for i in range(limit):
        row = proj_rows[i] or {}
        door = (row.get("door") or {})
        if not door.get("present"):
            continue

        closed = door.get("closed")
        if closed is False:
            continue  # already open

        # Prefer rect-center if bounds exist, else use canvas point
        if isinstance(door.get("bounds"), dict):
            click = {"type": "rect-center"}
            target_anchor = {"bounds": door["bounds"]}
        elif isinstance(door.get("canvas"), dict) and \
             isinstance(door["canvas"].get("x"), (int, float)) and isinstance(door["canvas"].get("y"), (int, float)):
            click = {"type": "point", "x": int(door["canvas"]["x"]), "y": int(door["canvas"]["y"])}
            target_anchor = {}
        else:
            # no geometry → skip; we’ll just walk this tick
            continue

        wx = row.get("world", {}).get("x", row.get("x"))
        wy = row.get("world", {}).get("y", row.get("y"))
        wp = row.get("world", {}).get("p", row.get("p"))

        name = door.get("name") or "Door"
        ident = {"id": door.get("id"), "name": name, "world": {"x": wx, "y": wy, "p": wp}}

        return {
            "index": i,
            "click": click,
            "target": {"domain": "object", "name": name, **target_anchor},
            # Your executor already supports postconditions (used in open_bank)
            # Wire a simple predicate your state loop can satisfy when the door flips open.
            "postconditions": [f"doorOpen@{wx},{wy} == true"],
            "timeout_ms": 2000
        }

    return None


def go_to(rect_or_key: str | tuple | list, payload: dict | None = None, ui=None, center: bool = False) -> dict | None:
    """
    One movement click toward an area, door-aware.
    If a CLOSED door is on the returned segment before the chosen waypoint,
    click it first and wait (with timeout) for it to open.
    Also aims camera at the door/ground point before dispatch.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        payload: Game state payload
        ui: UI instance for dispatching actions
        center: If True, go to center of region instead of stopping at boundary
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # accept key or explicit (minX, maxX, minY, maxY)
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None

    # Get fresh payload for path calculation
    fresh_payload = get_payload()
    
    # Path + projection with fresh data
    wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))
    if not wps:
        print(f"[DEBUG] No waypoints found for {rect_key}, refreshing path...")
        # Try once more with fresh payload
        time.sleep(0.1)
        fresh_payload = get_payload()
        wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))

    proj, dbg_proj = ipc_project_many(fresh_payload, wps)
    proj = _merge_door_into_projection(wps, proj)

    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
    if not usable:
        print(f"[DEBUG] No usable waypoints after projection, refreshing...")
        # Try once more with fresh payload
        time.sleep(0.1)
        fresh_payload = get_payload()
        wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))
        if wps:
            proj, dbg_proj = ipc_project_many(fresh_payload, wps)
            proj = _merge_door_into_projection(wps, proj)
            usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
        
        if not usable:
            print(f"[DEBUG] Still no usable waypoints after refresh, giving up")
            return None
    
    # If we have fewer than 10 waypoints, refresh the path
    if len(usable) < 10:
        print(f"[DEBUG] Only {len(usable)} waypoints remaining, refreshing path...")
        time.sleep(0.1)
        fresh_payload = get_payload()
        wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))
        if wps:
            proj, dbg_proj = ipc_project_many(fresh_payload, wps)
            proj = _merge_door_into_projection(wps, proj)
            usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
            print(f"[DEBUG] Refreshed path, now have {len(usable)} waypoints")
    
    # Check if player is at or near the last waypoint - if so, refresh the path
    player_x, player_y = _get_player_position(fresh_payload)
    if isinstance(player_x, int) and isinstance(player_y, int) and usable:
        last_waypoint = usable[-1]
        last_wp_x = last_waypoint.get("x")
        last_wp_y = last_waypoint.get("y")
        
        if isinstance(last_wp_x, int) and isinstance(last_wp_y, int):
            distance_to_last = _calculate_distance(player_x, player_y, last_wp_x, last_wp_y)
            print(f"[DEBUG] Distance to last waypoint: {distance_to_last}")
            
            # If player is close to the last waypoint, refresh the path
            if distance_to_last <= 3:  # Within 3 tiles of last waypoint
                print(f"[DEBUG] Player near last waypoint, refreshing path...")
                time.sleep(0.1)
                fresh_payload = get_payload()
                wps, dbg_path = ipc_path(fresh_payload, rect=tuple(rect))
                if wps:
                    proj, dbg_proj = ipc_project_many(fresh_payload, wps)
                    proj = _merge_door_into_projection(wps, proj)
                    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
                    print(f"[DEBUG] Refreshed path, now have {len(usable)} usable waypoints")
                else:
                    print(f"[DEBUG] No waypoints after refresh, checking if at destination...")
                    # Check if we're actually at the destination
                    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
                        rect = tuple(rect_or_key)
                        if player_in_rect(fresh_payload, rect):
                            print(f"[DEBUG] Player is at destination, no action needed")
                            return None
                    else:
                        rect_key = str(rect_or_key)
                        rect = get_nav_rect(rect_key)
                        if rect and player_in_rect(fresh_payload, rect):
                            print(f"[DEBUG] Player is at destination {rect_key}, no action needed")
        return None

    if center:
        # When going to center, choose from waypoints closer to the center of the region
        center_x = (rect[0] + rect[1]) / 2
        center_y = (rect[2] + rect[3]) / 2
        
        # Calculate distance from each waypoint to center and sort by distance
        def distance_to_center(wp):
            wx = wp.get("x", 0)
            wy = wp.get("y", 0)
            return ((wx - center_x) ** 2 + (wy - center_y) ** 2) ** 0.5
        
        # Sort by distance to center, take the closest 5
        sorted_by_distance = sorted(usable, key=distance_to_center)
        candidates = sorted_by_distance[-5:]
    else:
        # Choose randomly among the furthest 5 within the first <20 tiles (i.e., indices 15..19 when available)
        max_stride = min(19, len(usable) - 1)           # cap at index 19 (20 tiles from start)
        min_stride = max(0, max_stride - 4)             # last five within that window
        candidates = usable[min_stride:max_stride + 1]  # typical slice = [15..19]

    # Prefer tiles without a blocking door flag
    def _is_blocking(p):
        d = p.get("door") or {}
        if not d.get("present"):
            return False
        closed = d.get("closed")
        return (closed is True) or (closed is None)

    preferred = [p for p in candidates if not _is_blocking(p)]
    pool = preferred if preferred else candidates
    
    # Avoid clicking on the same waypoint repeatedly
    # Filter out waypoints that are too close to the player's current position
    if isinstance(player_x, int) and isinstance(player_y, int):
        filtered_pool = []
        for p in pool:
            wp_x = p.get("x")
            wp_y = p.get("y")
            if isinstance(wp_x, int) and isinstance(wp_y, int):
                distance = _calculate_distance(player_x, player_y, wp_x, wp_y)
                if distance > 1:  # Only consider waypoints more than 1 tile away
                    filtered_pool.append(p)
        
        if filtered_pool:
            pool = filtered_pool
            print(f"[DEBUG] Filtered waypoints to avoid clicking too close, {len(pool)} candidates remaining")
    
    chosen = random.choice(pool)
    chosen_idx = proj.index(chosen)

    # Door before chosen waypoint
    door_plan = _first_blocking_door(usable, up_to_index=chosen_idx)
    if door_plan:
        t = (door_plan.get("target") or {})
        click = (door_plan.get("click") or {})
        b = t.get("bounds") or None
        c = t.get("canvas") or t.get("tileCanvas") or None

        # normalize timeout/postconditions
        timeout_ms = int(door_plan.get("timeout_ms") or 2000)
        postconds = door_plan.get("postconditions") or []

        # 1) Bounds-based click (preferred)
        if isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
            step = emit({
                "action": "open-door",
                "click": click if click else {"type": "rect-center"},
                "target": {
                    "domain": t.get("domain") or "object",
                    "name": t.get("name") or "Door",
                    "bounds": b,
                    "world": door_plan.get("world"),
                },
                "postconditions": postconds,
                "timeout_ms": timeout_ms,
            })
            return _wait_for_camera_and_click(step, ui=ui, payload=fresh_payload, aim_ms=420, 
                                            hover_delay_ms=250, click_delay_ms=150)

        # 2) Canvas point click (fallback)
        if isinstance(c, dict) and "x" in c and "y" in c:
            step = emit({
                "action": "open-door",
                "click": {"type": "point", "x": int(c["x"]), "y": int(c["y"])},
                "target": {
                    "domain": t.get("domain") or "object",
                    "name": t.get("name") or "Door",
                    "world": door_plan.get("world"),
                },
                "postconditions": postconds,
                "timeout_ms": timeout_ms,
            })
            return _wait_for_camera_and_click(step, ui=ui, payload=fresh_payload, aim_ms=420,
                                            hover_delay_ms=250, click_delay_ms=150)

    # Normal ground move
    world_hint = chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}
    cx, cy = int(chosen["canvas"]["x"]), int(chosen["canvas"]["y"])
    step = emit({
        "action": "click-ground",
        "description": f"Move toward {rect_key}",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "ground", "name": f"Waypoint→{rect_key}",
                   "world": chosen.get("world") or {"x": chosen.get("x"), "y": chosen.get("y"), "p": chosen.get("p")}},
    })

    return _wait_for_camera_and_click(step, ui=ui, payload=fresh_payload, aim_ms=700,
                                    hover_delay_ms=300, click_delay_ms=200)


def _get_player_position(payload: dict) -> tuple[int | None, int | None]:
    """Get current player position from payload."""
    player = payload.get("player", {})
    return player.get("x"), player.get("y")


def _calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


def _should_click_next_waypoint(current_waypoint: dict, player_x: int, player_y: int, 
                               proximity_threshold: int = 2) -> bool:
    """
    Determine if we should click the next waypoint based on player proximity.
    
    Args:
        current_waypoint: Current waypoint being approached
        player_x: Player's current X coordinate
        player_y: Player's current Y coordinate
        proximity_threshold: Distance threshold to trigger next waypoint click
        
    Returns:
        True if player is close enough to current waypoint to click next one
    """
    waypoint_x = current_waypoint.get("x")
    waypoint_y = current_waypoint.get("y")
    
    if not isinstance(waypoint_x, int) or not isinstance(waypoint_y, int):
        return False
    
    distance = _calculate_distance(player_x, player_y, waypoint_x, waypoint_y)
    return distance <= proximity_threshold


def go_to_with_proximity_awareness(rect_or_key: str | tuple | list, payload: dict | None = None, 
                                 ui=None, center: bool = False, proximity_threshold: int = 2) -> dict | None:
    """
    Enhanced go_to function with proximity-aware waypoint clicking.
    
    This function intelligently clicks waypoints based on player proximity to the current target,
    preventing rapid clicking and improving movement efficiency.
    
    Args:
        rect_or_key: Region key or (minX, maxX, minY, maxY) tuple
        payload: Game state payload
        ui: UI instance for dispatching actions
        center: If True, go to center of region instead of stopping at boundary
        proximity_threshold: Distance threshold to trigger next waypoint click
        
    Returns:
        Dispatch result or None if no action needed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get current player position
    player_x, player_y = _get_player_position(payload)
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        # Fallback to original go_to if no player position
        return go_to(rect_or_key, payload, ui, center)
    
    # Check if we're already close to our destination
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        if player_in_rect(payload, rect):
            return None  # Already in target area
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if rect and player_in_rect(payload, rect):
            return None  # Already in target area
    
    # Get path and waypoints
    wps, dbg_path = ipc_path(payload, rect=tuple(rect))
    if not wps:
        return None
    
    proj, dbg_proj = ipc_project_many(payload, wps)
    proj = _merge_door_into_projection(wps, proj)
    
    usable = [p for p in proj if isinstance(p, dict) and p.get("canvas")]
    if not usable:
        return None
    
    # Find the waypoint the player is currently approaching
    current_target = None
    for i, waypoint in enumerate(usable):
        waypoint_x = waypoint.get("x")
        waypoint_y = waypoint.get("y")
        
        if isinstance(waypoint_x, int) and isinstance(waypoint_y, int):
            distance = _calculate_distance(player_x, player_y, waypoint_x, waypoint_y)
            if distance <= proximity_threshold * 2:  # Within reasonable range
                current_target = waypoint
                break
    
    # If no current target or player is close to current target, find next waypoint
    if current_target is None or _should_click_next_waypoint(current_target, player_x, player_y, proximity_threshold):
        # Use the original go_to logic to select next waypoint
        return go_to(rect_or_key, payload, ui, center)
    
    # Player is still approaching current target, no need to click yet
    return None


def go_to_closest_bank(payload: dict | None = None, ui=None) -> dict | None:
    """
    If not already in the nearest bank region, dispatch one move step toward it.
    Returns ui.dispatch(step) on success, else None if already inside or no waypoint this tick.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    key = closest_bank_key(payload)
    rect = bank_rect(key)
    if rect and player_in_rect(payload, rect):
        return None

    return go_to(key, payload, ui)

def go_to_ge(payload: dict | None = None, ui=None) -> dict | None:
    """
    If not already inside the Grand Exchange area, dispatch one move step toward it.
    Returns ui.dispatch(step) on success, else None if already inside or no GE rect is configured.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Try common keys you may have configured for the GE nav rectangle.
    rect_key = None
    rect = None
    for k in ("grand_exchange", "ge"):
        r = get_nav_rect(k)
        if isinstance(r, (tuple, list)) and len(r) == 4:
            rect_key, rect = k, r
            break

    if rect_key is None:
        return None  # no GE area configured

    if player_in_rect(payload, rect):
        return None  # already there

    return go_to(rect_key, payload, ui)

def in_area(rect_or_key: str | tuple | list, payload: dict | None = None, ui=None):
    if payload is None:
        payload = get_payload()

    # accept key or explicit (minX, maxX, minY, maxY)
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        rect = tuple(rect_or_key)
        rect_key = "custom"
    else:
        rect_key = str(rect_or_key)
        rect = get_nav_rect(rect_key)
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            return None
    return player_in_rect(payload, rect)

# --- Node-based travel (thin wrappers) ---
from ..helpers.nodes import node_rect, distance2_to  # new module

def go_to_node(name: str, payload: dict | None = None, ui=None, radius: int = 2):
    """
    Walk one step toward the named node (x,y,p) by wrapping the node tile as a small rect.
    Reuses existing go_to(...) which is door-aware, projection-based, and camera-integrated.
    """
    rect = node_rect(name, r=radius)
    if not rect:
        return None
    return go_to(rect, payload=payload, ui=ui)  # existing door-aware path→project→click flow

def follow_nodes(seq: list[str], payload: dict | None = None, ui=None,
                 arrive_d2: int = 9, radius: int = 2) -> dict | None:
    """
    Given a sequence of node names, attempt to reach the first not-yet-arrived node.
    - If within arrive_d2 (default=3 tiles -> 9), advance to the next.
    - Otherwise, issue one step toward that node (via go_to_node).
    Returns the dispatched step dict or None if already at final node.
    """
    if not isinstance(seq, (list, tuple)) or not seq:
        return None
    if payload is None:
        from ..helpers.context import get_payload
        payload = get_payload()

    # find first target not yet reached
    idx = 0
    while idx < len(seq):
        d2 = distance2_to(seq[idx], payload=payload)
        if d2 is None:
            break  # unknown player or node missing -> attempt anyway
        if d2 > arrive_d2:
            break
        idx += 1

    if idx >= len(seq):
        return None  # finished

    return go_to_node(seq[idx], payload=payload, ui=ui, radius=radius)

