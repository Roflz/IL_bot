# ilbot/ui/simple_recorder/actions/banking.py
import logging
from typing import List, Dict, Optional, Tuple

from . import ge, widgets, inventory, equipment
from .timing import wait_until
from .travel import in_area, _first_blocking_door_from_waypoints
from .widgets import click_widget
from helpers.bank import first_bank_slot, deposit_all_button_bounds, is_quantityx_selected, select_quantityx, \
    get_bank_quantity_mode, select_quantityx_custom, bank_note_selected, get_deposit_equipment_button, \
    get_bank_search_widgets, get_bank_tabs
from helpers.inventory import inv_has_any, norm_name, inv_count
from helpers.runtime_utils import ipc, dispatch
from helpers.rects import unwrap_rect
from helpers.utils import clean_rs, exponential_number, rect_beta_xy, sleep_exponential
from helpers.keyboard import press_enter, type_text
from helpers.widgets import widget_exists
from helpers.bank_inventory import find_bank_inventory_item
from helpers.inventory import inv_count


def is_open() -> bool:
    try:
        bank_data = ipc.get_bank()
        return bank_data.get('ok')
    except Exception as e:
        logging.error(f"[is_open] actions/bank.py: {e}")
        return False

def is_closed() -> bool:
    try:
        return not is_open()
    except Exception as e:
        logging.error(f"[is_closed] actions/bank.py: {e}")
        return True

def has_item(name: str) -> bool:
    try:
        return first_bank_slot(name)['quantity'] > 0
    except Exception as e:
        logging.error(f"[has_item] actions/bank.py: {e}")
        return False

def inv_has(name: str, min_qty: int = 1) -> bool:
    """
    Check if the bank has a specific item with optional minimum quantity.

    Args:
        name: Item name to check for
        min_qty: Minimum quantity required (default: 1)

    Returns:
        True if bank has the item in sufficient quantity, False otherwise
    """
    try:
        if not is_open():
            return False

        # Get bank inventory
        bank_inv = ipc.get_bank_inventory()
        if not bank_inv or not bank_inv.get("ok"):
            logging.error("[inv_has] actions/bank.py: Failed to get bank inventory from IPC")
            return False

        items = bank_inv.get("items", [])
        if not items:
            return False

        # Normalize the search name
        search_name = norm_name(name)

        # Count total quantity of the item
        total_qty = 0
        for item in items:
            item_name = item.get("name", "")
            if norm_name(item_name) == search_name:
                qty = int(item.get("quantity", 1))
                total_qty += qty

        return total_qty >= min_qty
    except Exception as e:
        logging.error(f"[inv_has] actions/bank.py: {e}")
        return False

def open_bank(prefer: str | None = None, randomize_closest: int | None = None, prefer_no_camera: bool = False) -> dict | None:
    """
    Open a bank using the new IPC system with proper response handling.
    
    Args:
        prefer: Preferred bank type ("bank booth", "banker", "bank chest", etc.)
        randomize_closest: If set to an integer X, randomly selects between the X closest booths/bankers/chests.
                          Uses weighted probability favoring the closest. Default None (no randomization).
        prefer_no_camera: If True, attempts to click without moving the camera. Default False (uses camera).
    """
    prefer = (prefer or "").strip().lower()
    want_booth = prefer in ("bank booth", "grand exchange booth")
    want_banker = prefer == "banker"
    want_chest = prefer in ("bank chest", "chest")

    # Auto-prefer banker in GE area
    try:
        if in_area("GE"):
            want_banker = True
            want_booth = False
    except Exception:
        pass

    if is_open():
        return

    # Get bank objects and NPCs using new IPC format
    booth_resp = ipc.get_objects("bank booth", ["GAME"])
    chest_resp = ipc.get_objects("bank chest", ["GAME"])
    banker_resp = ipc.get_npcs("banker")

    # Extract objects and NPCs from responses
    booths = booth_resp.get("objects", []) if booth_resp and booth_resp.get("ok") else []
    chests = chest_resp.get("objects", []) if chest_resp and chest_resp.get("ok") else []
    bankers = banker_resp.get("npcs", []) if banker_resp and banker_resp.get("ok") else []

    logging.info(f"[OPEN_BANK] Found {len(chests)} chests, {len(booths)} booths, {len(bankers)} bankers")

    # Find the best target
    target = None
    target_type = None

    # Helper function to find bank action index
    def find_bank_action(actions):
        if not actions:
            return None
        for i, action in enumerate(actions):
            if action and ("bank" in action.lower() or "use" in action.lower()):
                return i
        return None

    # Helper function to select from closest X items with randomization
    def select_from_closest(items, count: int | None = None):
        """Select from the closest items, optionally randomizing between the first X."""
        if not items:
            return None
        
        # Sort by distance
        sorted_items = sorted(items, key=lambda x: x.get("distance", float('inf')))
        
        if count is not None and len(sorted_items) >= count:
            # Randomize between first X items
            import random
            # Generate weights that favor closest but look random
            weights = []
            for i in range(count):
                # Weight decreases for later items
                base_weight = 1.0 / (1.0 + i * random.uniform(0.23, 0.41))
                weights.append(base_weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            selected = random.choices(sorted_items[:count], weights=weights, k=1)[0]
            return selected
        else:
            # Just return the closest
            return sorted_items[0]

    # Prefer bank chest when requested
    if want_chest and chests:
        valid_chests = []
        for chest in chests:
            actions = chest.get("actions", [])
            bank_idx = find_bank_action(actions)
            if bank_idx is not None:
                valid_chests.append(chest)
        
        if valid_chests:
            selected_chest = select_from_closest(valid_chests, randomize_closest)
            if selected_chest:
                target = selected_chest
                target_type = "object"

    # Try to find booth first (unless specifically wanting banker)
    if target is None and (not want_banker) and booths:
        # Collect all valid booths with bank actions
        valid_booths = []
        for booth in booths:
            name = booth.get("name", "").lower()
            actions = booth.get("actions", [])
            bank_idx = find_bank_action(actions)

            if bank_idx is not None and ("bank booth" in name or "grand exchange booth" in name):
                valid_booths.append(booth)
        
        if valid_booths:
            selected_booth = select_from_closest(valid_booths, randomize_closest)
            if selected_booth:
                logging.info(f"[OPEN_BANK] Selected booth: {selected_booth.get('name')} (distance: {selected_booth.get('distance', 'unknown')})")
                target = selected_booth
                target_type = "object"

    # Try any booth if no specific booth found
    if target is None and not want_banker and booths:
        # Collect all valid booths with bank actions
        valid_booths = []
        for booth in booths:
            actions = booth.get("actions", [])
            bank_idx = find_bank_action(actions)
            if bank_idx is not None:
                valid_booths.append(booth)
        
        if valid_booths:
            selected_booth = select_from_closest(valid_booths, randomize_closest)
            if selected_booth:
                logging.info(f"[OPEN_BANK] Selected any booth: {selected_booth.get('name')} (distance: {selected_booth.get('distance', 'unknown')})")
                target = selected_booth
                target_type = "object"

    # Try banker if no booth or specifically wanting banker
    if target is None and not want_booth and bankers:
        valid_bankers = []
        for banker in bankers:
            name = banker.get("name", "").lower()
            actions = banker.get("actions", [])
            bank_idx = find_bank_action(actions)

            if bank_idx is not None and ("banker" in name or want_banker):
                valid_bankers.append(banker)
        
        if valid_bankers:
            selected_banker = select_from_closest(valid_bankers, randomize_closest)
            if selected_banker:
                logging.info(f"[OPEN_BANK] Selected banker: {selected_banker.get('name')} (distance: {selected_banker.get('distance', 'unknown')})")
                target = selected_banker
                target_type = "npc"

    if target is None:
        logging.warning(f"[OPEN_BANK] No suitable bank target found")
        return


    # Handle door traversal if needed
    world_coords = target.get("world", {})
    gx, gy = world_coords.get("x"), world_coords.get("y")

    if isinstance(gx, int) and isinstance(gy, int):
        logging.info(f"[OPEN_BANK] Checking path to bank at ({gx}, {gy})")

        # Get path and check for doors
        wps, path_resp = ipc.path(goal=(gx, gy))
        if wps:
            from .travel import _handle_door_opening

            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                logging.info(f"[OPEN_BANK] Found blocking door, attempting to open")
                if not _handle_door_opening(door_plan, wps):
                    logging.warning(f"[OPEN_BANK] Failed to open door")
                    return

    # Click the bank target
    actions = target.get("actions", [])
    bank_idx = find_bank_action(actions)
    name = target.get("name", "Bank")

    # Determine action to use
    action = None
    if bank_idx is not None and bank_idx < len(actions):
        action = actions[bank_idx]

    # Use click methods based on prefer_no_camera setting
    world_coords_dict = {"x": gx, "y": gy, "p": 0}

    if target_type == "object":
        if prefer_no_camera:
            from actions.objects import click_object_no_camera
            result = click_object_no_camera(
                object_name=name,
                action=action,
                world_coords=world_coords_dict,
                aim_ms=420
            )
        else:
            from services.click_with_camera import click_object_with_camera
            result = click_object_with_camera(
                object_name=name,
                action=action,
                world_coords=world_coords_dict,
                aim_ms=420
            )
    else:  # NPC
        if prefer_no_camera:
            from actions.npc import click_npc_action_simple_prefer_no_camera
            result = click_npc_action_simple_prefer_no_camera(
                name=name,
                action=action,
                exact_match=False
            )
        else:
            from services.click_with_camera import click_npc_with_camera
            result = click_npc_with_camera(
                npc_name=name,
                action=action,
                world_coords=world_coords_dict,
                aim_ms=420
            )

    if result:
        logging.info(f"[OPEN_BANK] Click successful, waiting for bank interface")

        # Wait for bank interface to open with proper verification
        min_wait = int(exponential_number(150, 300, 1.0, "int"))
        max_wait = 10000  # Increased wait time for bank interface

        # Wait for bank interface to open
        bank_opened = wait_until(is_open, max_wait_ms=max_wait, min_wait_ms=min_wait)

        if bank_opened:
            logging.info(f"[OPEN_BANK] Bank interface opened successfully")

            # Handle any special widgets
            sleep_exponential(0.3, 0.8, 1.2)  # Let interface load

            from actions.widgets import click_widget

            if widget_exists(43515912):
                logging.info(f"[OPEN_BANK] Found special widget, clicking 43515933")
                click_result = click_widget(43515933)
                if click_result:
                    logging.info(f"[OPEN_BANK] Special widget clicked successfully")
                else:
                    logging.warning(f"[OPEN_BANK] Failed to click special widget")

            return result
        else:
            logging.warning(f"[OPEN_BANK] Bank interface did not open within {max_wait}ms")
            return None
    else:
        logging.error(f"[OPEN_BANK] Failed to open bank")
        return None




def deposit_inventory() -> dict | None:
    if not inv_has_any():
        return None
    rect = deposit_all_button_bounds()
    if not rect:
        return None
    cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0),
                           rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
    step = {
        "action": "bank-deposit-inv",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "bank-widget", "name": "Deposit Inventory", "bounds": rect},
        "postconditions": [],
    }
    return dispatch(step)


def withdraw_item(
    name: str | list,
    withdraw_x: int = 1,
    withdraw_all: bool = False,
) -> dict | None:
    # Handle list of items
    if isinstance(name, list):
        return withdraw_items(name, withdraw_x, withdraw_all)

    # Handle single item
    slot = first_bank_slot(name)
    if not slot:
        return None

    rect = unwrap_rect(slot.get("bounds"))
    if not rect:
        return None

    cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0),
                           rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
    item_name = str(name).strip()

    # --- Early exit: if bank_item_count <= withdraw_x and mode would withdraw more than available, just left-click ---
    if withdraw_x is not None and not withdraw_all:
        bank_item_count = get_item_count(item_name)
        if bank_item_count is not None and bank_item_count > 0:
            if bank_item_count <= withdraw_x:
                mode = get_bank_quantity_mode()
                mode_quantity = None
                
                if mode.get("mode") == 0:  # "1"
                    mode_quantity = 1
                elif mode.get("mode") == 1:  # "5"
                    mode_quantity = 5
                elif mode.get("mode") == 2:  # "10"
                    mode_quantity = 10
                elif mode.get("mode") == 3:  # "X" (custom)
                    mode_quantity = mode.get("x")
                elif mode.get("mode") == 4:  # "All"
                    # "All" would withdraw exactly bank_item_count, not more, so skip this optimization
                    mode_quantity = -1
                
                # If mode would withdraw more than available, just do simple left-click
                if mode_quantity == -1 or mode_quantity > bank_item_count:
                    step = {
                        "action": "withdraw-item",
                        "click": {"type": "point", "x": cx, "y": cy},
                        "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
                        "postconditions": [f"inventory contains '{item_name}'"],
                    }
                    return dispatch(step)

    # --- Withdraw-All (prefer Quantity: All + left-click) ---
    if withdraw_all:
        try:
            # Prefer selecting the bank Quantity: All button and then doing a simple left-click.
            ensure_quantity_button_set("All")
            step = {
                "action": "withdraw-item-all",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
                "postconditions": [f"inventory contains '{item_name}'"],
            }
            return dispatch(step)
        except Exception:
            # Fallback: Withdraw-All via live context menu
            step = {
                "action": "withdraw-item-all",
                "option": "withdraw-all",
                "click": {
                            "type": "context-select",        # uses live menu geometry via IPC "menu"
                    "target": item_name.lower(),     # substring match against menu target
                    "x": int(cx),                    # right-click anchor (canvas)
                    "y": int(cy),
                    "open_delay_ms": 120
                },
                "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
            }
            return dispatch(step)

    if withdraw_x == 5:
        step = {
            "action": "withdraw-item-5",
            "option": "withdraw-5",
            "click": {
                "type": "context-select",        # uses live menu geometry via IPC "menu"        # case-insensitive match
                "target": item_name.lower(),     # substring match against menu target
                "x": int(cx),                    # right-click anchor (canvas)
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        }
        return dispatch(step)

    if withdraw_x == 10:
        step = {
            "action": "withdraw-item-10",
            "option": "withdraw-10",
            "click": {
                "type": "context-select",        # uses live menu geometry via IPC "menu"        # case-insensitive match
                "target": item_name.lower(),     # substring match against menu target
                "x": int(cx),                    # right-click anchor (canvas)
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        }
        return dispatch(step)

    if withdraw_x == 1:
        step = {
            "action": "withdraw-item-1",
            "option": "withdraw-1",
            "click": {
                "type": "context-select",        # uses live menu geometry via IPC "menu"        # case-insensitive match
                "target": item_name.lower(),     # substring match against menu target
                "x": int(cx),                    # right-click anchor (canvas)
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        }
        return dispatch(step)

    # --- Withdraw-X via live context menu + type amount ---
    if withdraw_x is not None and withdraw_x > 5:

        if not get_bank_quantity_mode().get("x") == withdraw_x:
            select_quantityx_custom()
            if not wait_until(lambda: widget_exists(10616870), max_wait_ms=3000):
                return None
            sleep_exponential(0.3, 0.8, 1.2)
            type_text(str(withdraw_x))
            min_wait = int(exponential_number(400, 800, 1.0, "int"))
            if not wait_until(lambda: ge.buy_chatbox_text_input_contains(str(withdraw_x)), min_wait_ms=min_wait, max_wait_ms=3000):
                return None
            sleep_exponential(0.1, 0.3, 1.5)
            press_enter()
            min_wait = int(exponential_number(400, 800, 1.0, "int"))
            if not wait_until(lambda: is_quantityx_selected() and get_bank_quantity_mode().get("x") == withdraw_x, min_wait_ms=min_wait, max_wait_ms=3000):
                return None
            sleep_exponential(0.3, 0.8, 1.2)
        elif not is_quantityx_selected():
            select_quantityx()
            min_wait = int(exponential_number(400, 800, 1.0, "int"))
            if not wait_until(is_quantityx_selected, min_wait_ms=min_wait, max_wait_ms=3000):
                return None

        step = {
            "action": "withdraw-item",
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
            "postconditions": [f"inventory contains '{item_name}'"],
        }

        return dispatch(step)

    # --- Default: simple left-click withdraw (Withdraw-1 / custom shift-click config) ---
    mode = get_bank_quantity_mode()
    if mode.get("mode") == 3:
        quant = int(mode.get('x'))
    elif mode.get("mode") == 4:
        quant = 9999
    else:
        quant = int(mode.get('mode_name'))
    bank_quant = get_item_count(item_name)
    if bank_quant <= quant:
        step = {
            "action": "withdraw-item",
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
            "postconditions": [f"inventory contains '{item_name}'"],
        }
        sleep_exponential(0.1, 0.3, 1.2)
        result = dispatch(step)
    else:
        ensure_quantity_button_set("1")
        for click in range(withdraw_x):
            step = {
                "action": "withdraw-item",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
                "postconditions": [f"inventory contains '{item_name}'"],
            }
            sleep_exponential(0.1, 0.3, 1.2)
            result = dispatch(step)

    return result

import time

def withdraw_items(
    items,
    withdraw_all: bool = False,
    timeout: float = 15.0,
) -> dict | None:
    """
    Withdraw multiple items from the bank until the inventory contains the expected quantities
    or the timeout expires.

    Args:
        items: Either dict (item name -> quantity) or list of item names (withdraw all available).
        withdraw_all (bool): If True, withdraw all of each item regardless of quantity.
        timeout (float): Max seconds to wait before giving up (default 15s).

    Returns:
        Result of the last successful withdrawal, or None if all failed or timed out.
    """
    if not items:
        # Invalid items provided
        return None

    # Handle list of item names (withdraw all available)
    if isinstance(items, list):
        print(f"[BANK] Withdrawing all available quantities of {len(items)} items")

        last_result = None
        for item_name in items:
            if not isinstance(item_name, str):
                continue

            if has_item(item_name):
                bank_count = get_item_count(item_name)
                if bank_count == 1:
                    result = withdraw_item(item_name, withdraw_x=1)
                else:
                    result = withdraw_item(item_name, withdraw_all=True)

                if result:
                    last_result = result
                    print(f"[BANK] Withdrew {bank_count} {item_name}")
                else:
                    print(f"[BANK] Failed to withdraw {item_name}")

                sleep_exponential(0.2, 0.5, 1.2)

        return last_result

    # Handle dict format (original functionality)
    if not isinstance(items, dict):
        return None

    print(f"[BANK] Withdrawing {len(items)} items")

    last_result = None
    successful_withdrawals = 0
    start_time = time.time()

    while True:
        all_items_obtained = True

        for item, qty in items.items():
            if not isinstance(item, str) or not isinstance(qty, int) or qty <= 0:
                # Skipping invalid entry
                continue

            current_qty = inventory.count_unnoted_item(item)
            if current_qty < qty:
                all_items_obtained = False
                print(f"[BANK] Withdrawing {item}")

                result = withdraw_item(
                    name=item,
                    withdraw_x=qty if not withdraw_all else None,
                    withdraw_all=withdraw_all,
                )

                if result:
                    last_result = result
                    successful_withdrawals += 1
                    # Successfully withdrew item
                else:
                    print(f"[BANK] Failed to withdraw {item}")

                sleep_exponential(0.2, 0.5, 1.2)

        if all_items_obtained:
            print(f"[BANK] All requested items obtained")
            return last_result

        if time.time() - start_time > timeout:
            print("[BANK] Timed out waiting for items")
            return None

        time.sleep(0.5)


def _is_point_in_widget_bounds(x: int, y: int, bounds: dict) -> bool:
    """
    Check if a point (x, y) is within widget bounds.
    
    Args:
        x: X coordinate of the point
        y: Y coordinate of the point
        bounds: Widget bounds dict with x, y, width, height
    
    Returns:
        True if point is within bounds, False otherwise
    """
    if not bounds:
        return False
    
    bounds_x = bounds.get("x", 0)
    bounds_y = bounds.get("y", 0)
    bounds_width = bounds.get("width", 0)
    bounds_height = bounds.get("height", 0)
    
    return (bounds_x <= x <= bounds_x + bounds_width and 
            bounds_y <= y <= bounds_y + bounds_height)


def _find_path_click_outside_bank_interface(obj_world_coords: dict, bank_bounds: dict) -> dict | None:
    """
    Find a click point outside the bank interface that paths correctly toward the object.
    Uses sophisticated movement logic: clicks in an area that paths correctly, not on waypoints.
    This mimics human behavior of clicking around the interface to move toward an object.
    
    Args:
        obj_world_coords: World coordinates of the object {"x": int, "y": int, "p": int}
        bank_bounds: Bank interface widget bounds {"x": int, "y": int, "width": int, "height": int}
    
    Returns:
        World coordinates to click (outside bank interface) or None if not found
    """
    from actions import player
    from actions.travel import _calculate_click_location, _verify_click_path, _is_tile_on_path
    import random
    
    obj_x = obj_world_coords.get("x")
    obj_y = obj_world_coords.get("y")
    
    if not isinstance(obj_x, int) or not isinstance(obj_y, int):
        return None
    
    player_x = player.get_x()
    player_y = player.get_y()
    
    if not isinstance(player_x, int) or not isinstance(player_y, int):
        return None
    
    # Get intended path to object (for verification)
    rect = (obj_x - 1, obj_x + 1, obj_y - 1, obj_y + 1)
    intended_path, _ = ipc.path(rect=rect, visualize=False)
    
    if not intended_path or len(intended_path) == 0:
        return None
    
    final_dest = {"x": obj_x, "y": obj_y}
    
    # STEP 1: Calculate optimal click location using sophisticated movement logic
    # This clicks ahead of the destination, not on waypoints
    click_location = _calculate_click_location(player_x, player_y, obj_x, obj_y, is_precise=False)
    
    # Check if this click location is outside the bank interface
    proj = ipc.project_world_tile(click_location.get("x"), click_location.get("y"))
    if proj and proj.get("ok") and proj.get("canvas"):
        canvas_x = proj["canvas"].get("x")
        canvas_y = proj["canvas"].get("y")
        
        if isinstance(canvas_x, (int, float)) and isinstance(canvas_y, (int, float)):
            if not _is_point_in_widget_bounds(int(canvas_x), int(canvas_y), bank_bounds):
                # Click location is outside bank interface - verify it paths correctly
                if _verify_click_path(click_location, final_dest, intended_path):
                    return click_location
    
    # STEP 2: Click location is blocked - try alternative click locations
    # Try adjusting the click distance or direction to find a point outside the interface
    # that still paths correctly
    
    # Calculate direction to object
    dx = obj_x - player_x
    dy = obj_y - player_y
    distance = abs(dx) + abs(dy)
    
    if distance == 0:
        return None
    
    # Normalize direction
    if abs(dx) > abs(dy):
        unit_dx = 1 if dx > 0 else -1
        unit_dy = int(dy / abs(dx)) if dx != 0 else 0
    else:
        unit_dx = int(dx / abs(dy)) if dy != 0 else 0
        unit_dy = 1 if dy > 0 else -1
    
    # Try different click distances (closer and farther from destination)
    # to find one that's outside the bank interface
    click_distances = []
    
    # Generate candidate click distances based on total distance
    if distance > 20:
        # Long distance: try 3-12 tiles ahead
        click_distances = list(range(3, 13))
    elif distance > 5:
        # Medium distance: try 0-5 tiles ahead
        click_distances = list(range(0, 6))
    else:
        # Close distance: try 1-4 tiles ahead
        click_distances = list(range(1, 5))
    
    # Shuffle to add randomness
    random.shuffle(click_distances)
    
    # Try each click distance
    for click_dist in click_distances:
        # Calculate click location at this distance
        candidate_x = obj_x + (unit_dx * click_dist)
        candidate_y = obj_y + (unit_dy * click_dist)
        candidate_location = {"x": candidate_x, "y": candidate_y}
        
        # Check if this location is outside the bank interface
        proj = ipc.project_world_tile(candidate_x, candidate_y)
        if proj and proj.get("ok") and proj.get("canvas"):
            canvas_x = proj["canvas"].get("x")
            canvas_y = proj["canvas"].get("y")
            
            if isinstance(canvas_x, (int, float)) and isinstance(canvas_y, (int, float)):
                if not _is_point_in_widget_bounds(int(canvas_x), int(canvas_y), bank_bounds):
                    # Outside bank interface - verify it paths correctly
                    if _verify_click_path(candidate_location, final_dest, intended_path):
                        logging.info(f"[CLOSE_BANK] Found valid click location outside bank interface at ({candidate_x}, {candidate_y})")
                        return candidate_location
    
    # STEP 3: Try clicking in perpendicular directions (left/right of the path)
    # This mimics clicking around the interface
    perpendicular_directions = [
        (-unit_dy, unit_dx),   # 90 degrees clockwise
        (unit_dy, -unit_dx),   # 90 degrees counter-clockwise
    ]
    
    for perp_dx, perp_dy in perpendicular_directions:
        # Try clicking a few tiles to the side of the optimal path
        for side_offset in [2, 3, 4, 5]:  # Try 2-5 tiles to the side
            candidate_x = obj_x + (unit_dx * 3) + (perp_dx * side_offset)
            candidate_y = obj_y + (unit_dy * 3) + (perp_dy * side_offset)
            candidate_location = {"x": candidate_x, "y": candidate_y}
            
            # Check if this location is outside the bank interface
            proj = ipc.project_world_tile(candidate_x, candidate_y)
            if proj and proj.get("ok") and proj.get("canvas"):
                canvas_x = proj["canvas"].get("x")
                canvas_y = proj["canvas"].get("y")
                
                if isinstance(canvas_x, (int, float)) and isinstance(canvas_y, (int, float)):
                    if not _is_point_in_widget_bounds(int(canvas_x), int(canvas_y), bank_bounds):
                        # Outside bank interface - verify it paths correctly
                        if _verify_click_path(candidate_location, final_dest, intended_path):
                            logging.info(f"[CLOSE_BANK] Found valid click location (perpendicular) outside bank interface at ({candidate_x}, {candidate_y})")
                            return candidate_location
    
    # Couldn't find a valid click point outside the bank interface
    logging.info(f"[CLOSE_BANK] Could not find click location outside bank interface that paths correctly")
    return None


def close_bank(
    object_name: str | None = None,
    object_action: str | None = None,
    click_object_probability: float | tuple[float, float] | None = None,
    prefer_no_camera: bool = False
) -> dict | None:
    """
    Close the bank. Optionally can click an object instead (which auto-closes bank).
    
    If the bank interface is blocking the object, will either:
    1. Click a point outside the bank interface that's on a path to the object (human-like behavior)
    2. Fall back to closing the bank normally
    
    Args:
        object_name: Optional name of object to potentially click (e.g., "Furnace")
        object_action: Optional action to perform on object (e.g., "Smelt")
        click_object_probability: Probability to click object instead of closing bank.
                                 If tuple, uses random range (min, max). If float, uses that value.
                                 If None and object_name provided, uses default range (0.2347, 0.3891).
                                 If None and no object_name, always closes bank normally.
        prefer_no_camera: If True, uses prefer_no_camera variant for object click
    
    Returns:
        Dict with keys:
            - "action": "object_click" | "ground_click" | "normal_close" | "failed"
            - "result": Original result from the action (dict or None)
        Or None if bank is not open
    """
    try:
        # Check if bank is open first
        if not is_open():
            return None
        
        # If object_name and object_action provided, potentially click object instead
        if object_name and object_action:
            import random
            from constants import BANK_WIDGETS
            from actions.widgets import get_widget_bounds
            from actions import objects
            from services.click_with_camera import click_ground_with_camera
            
            # Determine probability
            if click_object_probability is None:
                # Default probability range
                click_probability = random.uniform(0.2347, 0.3891)
            elif isinstance(click_object_probability, tuple):
                # Range provided
                click_probability = random.uniform(click_object_probability[0], click_object_probability[1])
            else:
                # Single probability value
                click_probability = float(click_object_probability)
            
            # Randomly decide whether to click object directly
            click_object_directly = random.random() < click_probability
            
            if click_object_directly:
                # First, try to find the object and check if it's blocked by bank interface
                # Use IPC directly to find the object
                obj_resp = ipc.find_object(object_name, types=["GAME"], exact_match=False)
                obj = None
                if obj_resp and obj_resp.get("ok") and obj_resp.get("found"):
                    obj = obj_resp.get("object")
                    # Verify it has the required action
                    if obj and object_action:
                        obj_actions = [str(a).strip().lower() for a in (obj.get("actions") or []) if a]
                        if object_action.strip().lower() not in obj_actions:
                            obj = None  # Object doesn't have required action
                
                if obj:
                    obj_world_coords = obj.get("world", {})
                    obj_canvas = obj.get("canvas", {})
                    
                    # Get bank interface bounds
                    bank_bounds = get_widget_bounds(BANK_WIDGETS["UNIVERSE"])
                    
                    if bank_bounds and obj_canvas:
                        obj_canvas_x = obj_canvas.get("x")
                        obj_canvas_y = obj_canvas.get("y")
                        
                        # Check if object's click point is behind the bank interface
                        if isinstance(obj_canvas_x, (int, float)) and isinstance(obj_canvas_y, (int, float)):
                            if _is_point_in_widget_bounds(int(obj_canvas_x), int(obj_canvas_y), bank_bounds):
                                # Object is blocked by bank interface
                                logging.info(f"[CLOSE_BANK] Object '{object_name}' is blocked by bank interface, finding path click point outside...")
                                
                                # Try to find a click point outside the bank interface on path to object
                                path_click_coords = _find_path_click_outside_bank_interface(obj_world_coords, bank_bounds)
                                
                                if path_click_coords:
                                    # Click on ground outside bank interface (will move toward object)
                                    logging.info(f"[CLOSE_BANK] Clicking ground outside bank interface at ({path_click_coords.get('x')}, {path_click_coords.get('y')}) to move toward object")
                                    result = click_ground_with_camera(
                                        world_coords=path_click_coords,
                                        description=f"Move toward {object_name} (avoiding bank interface)",
                                        aim_ms=420
                                    )
                                    if result:
                                        return {"action": "ground_click", "result": result}
                                
                                # Couldn't find a good path click point - fall back to normal bank close
                                logging.info(f"[CLOSE_BANK] Could not find path click point, closing bank normally")
                                # Fall through to normal bank close
                            else:
                                # Object is not blocked - proceed with normal object click
                                if prefer_no_camera:
                                    result = objects.click_object_closest_by_distance_simple_no_camera(
                                        object_name, 
                                        prefer_action=object_action
                                    )
                                    if not result:
                                        # Fallback to prefer_no_camera variant if no_camera fails
                                        result = objects.click_object_closest_by_distance_simple_prefer_no_camera(
                                            object_name, 
                                            prefer_action=object_action
                                        )
                                else:
                                    result = objects.click_object_closest_by_distance_simple(
                                        object_name, 
                                        prefer_action=object_action
                                    )
                                return {"action": "object_click", "result": result}
                        else:
                            # Can't determine canvas coords - try normal click anyway
                            if prefer_no_camera:
                                result = objects.click_object_closest_by_distance_simple_no_camera(
                                    object_name, 
                                    prefer_action=object_action
                                )
                                if not result:
                                    # Fallback to prefer_no_camera variant if no_camera fails
                                    result = objects.click_object_closest_by_distance_simple_prefer_no_camera(
                                        object_name, 
                                        prefer_action=object_action
                                    )
                            else:
                                result = objects.click_object_closest_by_distance_simple(
                                    object_name, 
                                    prefer_action=object_action
                                )
                            return result
                    else:
                        # No bank bounds or object canvas - try normal click
                        if prefer_no_camera:
                            result = objects.click_object_closest_by_distance_simple_no_camera(
                                object_name, 
                                prefer_action=object_action
                            )
                            if not result:
                                # Fallback to prefer_no_camera variant if no_camera fails
                                result = objects.click_object_closest_by_distance_simple_prefer_no_camera(
                                    object_name, 
                                    prefer_action=object_action
                                )
                        else:
                            result = objects.click_object_closest_by_distance_simple(
                                object_name, 
                                prefer_action=object_action
                            )
                        return result
                else:
                    # Object not found - try normal click anyway (might work)
                    if prefer_no_camera:
                        result = objects.click_object_closest_by_distance_simple_no_camera(
                            object_name, 
                            prefer_action=object_action
                        )
                        if not result:
                            # Fallback to prefer_no_camera variant if no_camera fails
                            result = objects.click_object_closest_by_distance_simple_prefer_no_camera(
                                object_name, 
                                prefer_action=object_action
                            )
                    else:
                        result = objects.click_object_closest_by_distance_simple(
                            object_name, 
                            prefer_action=object_action
                        )
                    return result
        
        # Close bank normally
        step = {
            "action": "close-bank",
            "click": {"type": "key", "key": "esc"},
            "target": {"domain": "bank", "name": "Close"},
            "postconditions": ["bankOpen == false"],
        }
        result = dispatch(step)
        return {"action": "normal_close", "result": result}
        
    except Exception as e:
        logging.error(f"[close_bank] actions/bank.py: {e}")
        # Fallback to normal bank close
        step = {
            "action": "close-bank",
            "click": {"type": "key", "key": "esc"},
            "target": {"domain": "bank", "name": "Close"},
            "postconditions": ["bankOpen == false"],
        }
        result = dispatch(step)
        return {"action": "normal_close", "result": result}

def toggle_note_mode() -> dict | None:
    """Toggle bank note mode (withdraw as note/withdraw as item)"""
    from constants import BANK_WIDGETS

    # Get the bank note toggle widget directly
    widget_info = widgets.get_widget_info(BANK_WIDGETS["NOTE"])
    if not widget_info:
        return None

    widget_data = widget_info.get("data", {})
    bounds = widget_data.get("bounds")

    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        return None

    cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                           bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    step = {
        "action": "bank-note-toggle",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "bank-widget", "name": "Withdraw as Note", "bounds": bounds},
        "postconditions": [],
    }
    return dispatch(step)

def ensure_note_mode_disabled() -> dict | None:
    """Ensure bank note mode is disabled (withdraw as items, not notes)"""

    if bank_note_selected():
        return toggle_note_mode()
    return None


def ensure_note_mode_enabled() -> dict | None:
    """Ensure bank note mode is disabled (withdraw as items, not notes)"""

    while not bank_note_selected():
        toggle_note_mode()
        wait_until(bank_note_selected, max_wait_ms=exponential_number(0.3, 0.8, 1))

def deposit_equipment() -> dict | None:
    """Deposit all equipped items into the bank."""

    equipment_button = get_deposit_equipment_button()
    if not equipment_button:
        logging.warning("[DEPOSIT_EQUIPMENT] Equipment deposit button not found")
        return None

    bounds = equipment_button.get("bounds")
    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        logging.warning("[DEPOSIT_EQUIPMENT] Equipment deposit button has invalid bounds")
        return None

    cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                           bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    step = {
        "action": "bank-deposit-equipment",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "bank-widget", "name": "Deposit Equipment", "bounds": bounds},
        "postconditions": [],
    }
    return dispatch(step)

def interact(item_name: str | list, action: str | list) -> dict | None:
    """
    Interact with bank inventory items using context clicks.

    Args:
        item_name: Name of the item(s) to interact with (string or list)
        action: Action(s) to perform (string or list of strings)
                If list, will try each action until one is available

    Returns:
        Result of the interaction, or None if failed
    """
    # Handle list of items
    if isinstance(item_name, list):
        return _interact_multiple_items(item_name, action)

    # Handle single item
    return _interact_single_item(item_name, action)


def interact_inventory(item_name: str, action: str) -> dict | None:
    """
    Interact with player's inventory items when bank is open using context clicks.

    Args:
        item_name: Name of the item to interact with
        action: Action to perform (e.g., "Wield", "Wear", "Deposit-All", "Use")

    Returns:
        Result of the interaction, or None if failed
    """
    try:
        # Inner attempt loop with fresh coordinate recalculation
        max_attempts = 3
        for attempt in range(max_attempts):
            # Fresh coordinate recalculation
            widget_data = ipc.get_widget_children(983043)
            if not widget_data or not widget_data.get("ok"):
                print(f"[BANK] Failed to get bank inventory widget")
                continue

            children = widget_data.get("children", [])
            target_item = None

            # Find the target item in the inventory
            search_name = norm_name(item_name)
            for child in children:
                child_name = child.get("name", "").strip()
                if not child_name:
                    continue

                # Remove color codes from the name for comparison
                clean_name = child_name.replace("<col=ff9040>", "").replace("</col>", "").strip()
                if norm_name(clean_name) == search_name:
                    target_item = child
                    break

            if not target_item:
                print(f"[BANK] Item '{item_name}' not found in player inventory")
                continue

            # Get item bounds and canvas location
            bounds = target_item.get("bounds")
            canvas_location = target_item.get("canvasLocation")

            if not bounds or not canvas_location:
                print(f"[BANK] No valid bounds or canvas location for item '{item_name}'")
                continue

            # Use bounds for randomized coordinates, fallback to canvas
            if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
                cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                       bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
            else:
                cx, cy = int(canvas_location["x"]), int(canvas_location["y"])

            # Create the interaction step
            step = {
                "action": "bank-player-inventory-interact",
                "option": action,
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }

            result = dispatch(step)

            if result:
                # Check if the correct interaction was performed
                from helpers.ipc import get_last_interaction
                last_interaction = get_last_interaction()

                expected_action = action
                expected_target = item_name

                if (last_interaction and
                    last_interaction.get("action") == expected_action and
                    clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                    print(f"[CLICK] {expected_target} ({action}) - interaction verified")
                    return result
                else:
                    print(f"[CLICK] {expected_target} ({action}) - incorrect interaction, retrying...")
                    continue

        return None

    except Exception as e:
        logging.error(f"[interact_inventory] actions/bank.py: {e}")
        return None


def deposit_item_from_slot(
    item_name: str, 
    slot: dict,
    deposit_x: int = None,
    deposit_all: bool = False
) -> dict | None:
    """
    Deposit an item from a specific inventory slot.
    Replicates the withdrawal logic for intelligent deposit method selection.
    
    Args:
        item_name: Name of the item to deposit
        slot: Slot dict from bank widget children containing the item
        deposit_x: Specific quantity to deposit (1, 5, 10, or custom X)
        deposit_all: If True, deposit all of this item
        
    Returns:
        Result of the interaction, or None if failed
    """
    try:
        bounds = slot.get("bounds")
        canvas_location = slot.get("canvasLocation")
        
        if not bounds or not canvas_location:
            return None
        
        # Check quantity from inventory data
        inv_quantity = inv_count(item_name)
        
        if inv_quantity <= 0:
            return None
        
        cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                               bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        
        # --- Early exit: Check bank quantity mode vs inventory count ---
        mode = get_bank_quantity_mode()
        mode_quantity = None
        
        if mode.get("mode") == 0:  # "1"
            mode_quantity = 1
        elif mode.get("mode") == 1:  # "5"
            mode_quantity = 5
        elif mode.get("mode") == 2:  # "10"
            mode_quantity = 10
        elif mode.get("mode") == 3:  # "X" (custom)
            mode_quantity = mode.get("x")
        elif mode.get("mode") == 4:  # "All"
            mode_quantity = 9999  # Effectively unlimited
        
        # If bank quantity mode >= inventory count, just do simple left-click
        if mode_quantity is not None and mode_quantity >= inv_quantity:
            step = {
                "action": "bank-deposit-item-single",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            return dispatch(step)
        
        # --- Early exit: if inv_quantity <= deposit_x, just use Deposit-All ---
        if deposit_x is not None and not deposit_all:
            if inv_quantity <= deposit_x:
                # We want to deposit all (or more than we have), so use Deposit-All
                step = {
                    "action": "bank-deposit-item-all",
                    "option": "Deposit-All",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                return dispatch(step)
        
        # --- Deposit-All ---
        if deposit_all or (deposit_x is None and inv_quantity > 1):
            step = {
                "action": "bank-deposit-item-all",
                "option": "Deposit-All",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            return dispatch(step)
        
        # --- Deposit-5 ---
        if deposit_x == 5:
            step = {
                "action": "bank-deposit-item-5",
                "option": "Deposit-5",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            return dispatch(step)
        
        # --- Deposit-10 ---
        if deposit_x == 10:
            step = {
                "action": "bank-deposit-item-10",
                "option": "Deposit-10",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            return dispatch(step)
        
        # --- Deposit-1 ---
        if deposit_x == 1 or (deposit_x is None and inv_quantity == 1):
            step = {
                "action": "bank-deposit-item-single",
                "option": "Deposit-1",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            return dispatch(step)
        
        # --- Deposit-X (custom amount) ---
        if deposit_x is not None and deposit_x > 10:
            step = {
                "action": "bank-deposit-item-x",
                "option": "Deposit-X",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
            result = dispatch(step)
            if result:
                # Type the amount after Deposit-X is selected
                from helpers.keyboard import type_text, press_enter
                from helpers.utils import sleep_exponential, exponential_number
                from .timing import wait_until
                sleep_exponential(0.3, 0.8, 1.2)
                type_text(str(deposit_x))
                min_wait = int(exponential_number(400, 800, 1.0, "int"))
                # Wait for input to be typed (no specific check available for deposit, just wait)
                sleep_exponential(0.1, 0.3, 1.5)
                press_enter()
                sleep_exponential(0.3, 0.8, 1.2)
            return result
        
        # --- Default: use context-click with appropriate option ---
        if inv_quantity == 1:
            step = {
                "action": "bank-deposit-item-single",
                "option": "Deposit-1",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
        else:
            step = {
                "action": "bank-deposit-item-all",
                "option": "Deposit-All",
                "click": {
                    "type": "context-select",
                    "target": item_name.lower(),
                    "x": cx,
                    "y": cy,
                    "open_delay_ms": 120
                },
                "target": {
                    "domain": "bank-player-inventory",
                    "name": item_name,
                    "bounds": bounds
                },
                "postconditions": [],
            }
        
        return dispatch(step)
    except Exception as e:
        logging.error(f"[deposit_item_from_slot] actions/bank.py: {e}")
        return None


def _select_random_slot_for_item(item_name: str, bias_earlier: bool = True) -> dict | None:
    """
    Select a random slot for an item from the bank inventory widget.
    If bias_earlier is True, earlier slots have higher probability.
    
    Args:
        item_name: Name of the item to find
        bias_earlier: If True, weight selection toward earlier slots
        
    Returns:
        Slot dict if found, None otherwise
    """
    try:
        widget_data = ipc.get_widget_children(983043)
        if not widget_data or not widget_data.get("ok"):
            return None
        
        children = widget_data.get("children", [])
        search_name = norm_name(item_name)
        
        # Find all slots containing this item
        matching_slots = []
        for idx, child in enumerate(children):
            child_name = child.get("name", "").strip()
            if not child_name:
                continue
            
            clean_child_name = clean_rs(child_name)
            if norm_name(clean_child_name) == search_name:
                matching_slots.append((idx, child))
        
        if not matching_slots:
            return None
        
        if len(matching_slots) == 1:
            return matching_slots[0][1]
        
        # Randomly select with bias toward earlier slots
        if bias_earlier:
            import random
            weights = []
            for i in range(len(matching_slots)):
                # Weight decreases exponentially: first gets 1.0, second gets ~0.7, third gets ~0.5, etc.
                weight = 1.0 / (1.0 + i * 0.4)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            selected_slot = random.choices(matching_slots, weights=weights, k=1)[0][1]
        else:
            import random
            selected_slot = random.choice(matching_slots)[1]
        
        return selected_slot
    except Exception as e:
        logging.error(f"[_select_random_slot_for_item] actions/bank.py: {e}")
        return None


def deposit_unwanted_items(required_items: list, max_unique_for_bulk: int = 3) -> dict | None:
    """
    Smart deposit logic: only deposit items that are not in the required items list.
    If there are more than max_unique_for_bulk unique unwanted items, deposits all inventory.
    Otherwise, deposits each unwanted item individually with randomized slot selection.
    
    Args:
        required_items: List of item names that should be kept in inventory
        max_unique_for_bulk: If more than this many unique unwanted items, deposit all (default: 3)
        
    Returns:
        Result of the last deposit action, or None if no deposits were needed
    """
    try:
        from helpers.inventory import inv_slots
        from helpers.utils import norm_name
        
        # Check if we only have required items
        if inventory.has_only_items(required_items):
            return None
        
        # Get all items in inventory
        slots = inv_slots()
        required_normalized = {norm_name(name) for name in required_items}
        
        # Find unique unwanted items
        unwanted_items = set()
        for slot in slots:
            if slot.get("quantity", 0) > 0:
                item_name = slot.get("itemName", "")
                item_normalized = norm_name(item_name)
                if item_normalized not in required_normalized:
                    unwanted_items.add(item_name)  # Use original name for deposit
        
        if not unwanted_items:
            return None
        
        # If more than max_unique_for_bulk unique unwanted items, deposit all
        if len(unwanted_items) > max_unique_for_bulk:
            result = deposit_inventory()
            if result:
                from .timing import wait_until
                wait_until(inventory.is_empty, max_wait_ms=2000)
            return result
        
        # Deposit unwanted items individually with randomized slot selection
        last_result = None
        for item_name in unwanted_items:
            if not inventory.has_item(item_name):
                continue
            
            # Get inventory count for this item to pass to deposit methods
            inv_quantity = inv_count(item_name)
            if inv_quantity <= 0:
                continue
            
            # Try to get a random slot for this item
            selected_slot = _select_random_slot_for_item(item_name, bias_earlier=True)
            
            if selected_slot:
                # Deposit all of this unwanted item using intelligent deposit logic
                result = deposit_item_from_slot(item_name, selected_slot, deposit_all=True)
            else:
                # Fallback to standard deposit_item if slot selection fails
                result = deposit_item(item_name, deposit_all=True)
            
            if result:
                last_result = result
                # Wait for this item to be deposited
                from .timing import wait_until
                wait_until(lambda name=item_name: not inventory.has_item(name), max_wait_ms=2000)
        
        return last_result
    except Exception as e:
        logging.error(f"[deposit_unwanted_items] actions/bank.py: {e}")
        return None


def deposit_item(
    item_name: str,
    deposit_x: int = None,
    deposit_all: bool = False
) -> dict | None:
    """
    Deposit an item from player's inventory to the bank.
    Replicates the withdrawal logic for intelligent deposit method selection.

    Args:
        item_name: Name of the item to deposit
        deposit_x: Specific quantity to deposit (1, 5, 10, or custom X)
        deposit_all: If True, deposit all of this item

    Returns:
        Result of the interaction, or None if failed
    """
    try:
        # Inner attempt loop with fresh coordinate recalculation
        max_attempts = 3
        for attempt in range(max_attempts):
            # Fresh coordinate recalculation
            widget_data = ipc.get_widget_children(983043)
            if not widget_data or not widget_data.get("ok"):
                print(f"[BANK] Failed to get bank inventory widget")
                continue

            children = widget_data.get("children", [])
            target_item = None

            # Find the target item in the inventory
            search_name = norm_name(item_name)
            for child in children:
                child_name = child.get("name", "").strip()
                if not child_name:
                    continue

                # Remove color codes from the name for comparison
                clean_name = clean_rs(child_name)
                if norm_name(clean_name) == search_name:
                    target_item = child
                    break

            if not target_item:
                print(f"[BANK] Item '{item_name}' not found in player inventory")
                continue

            # Get item bounds and canvas location
            bounds = target_item.get("bounds")
            canvas_location = target_item.get("canvasLocation")

            if not bounds or not canvas_location:
                print(f"[BANK] No valid bounds or canvas location for item '{item_name}'")
                continue

            # Check quantity from inventory data
            inv_quantity = inv_count(item_name)

            if inv_quantity <= 0:
                print(f"[BANK] Item '{item_name}' has no quantity to deposit")
                continue

            cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                   bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)

            # --- Early exit: Check bank quantity mode vs inventory count ---
            mode = get_bank_quantity_mode()
            mode_quantity = None
            
            if mode.get("mode") == 0:  # "1"
                mode_quantity = 1
            elif mode.get("mode") == 1:  # "5"
                mode_quantity = 5
            elif mode.get("mode") == 2:  # "10"
                mode_quantity = 10
            elif mode.get("mode") == 3:  # "X" (custom)
                mode_quantity = mode.get("x")
            elif mode.get("mode") == 4:  # "All"
                mode_quantity = 9999  # Effectively unlimited
            
            # If bank quantity mode >= inventory count, just do simple left-click
            if mode_quantity is not None and mode_quantity >= inv_quantity:
                step = {
                    "action": "bank-deposit-item-single",
                    "click": {"type": "point", "x": cx, "y": cy},
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    return result
                continue

            # --- Early exit: if inv_quantity <= deposit_x, just use Deposit-All ---
            if deposit_x is not None and not deposit_all:
                if inv_quantity <= deposit_x:
                    # We want to deposit all (or more than we have), so use Deposit-All
                    step = {
                        "action": "bank-deposit-item-all",
                        "option": "Deposit-All",
                        "click": {
                            "type": "context-select",
                            "target": item_name.lower(),
                            "x": cx,
                            "y": cy,
                            "open_delay_ms": 120
                        },
                        "target": {
                            "domain": "bank-player-inventory",
                            "name": item_name,
                            "bounds": bounds
                        },
                        "postconditions": [],
                    }
                    result = dispatch(step)
                    if result:
                        return result
                    continue

            # --- Deposit-All ---
            if deposit_all or (deposit_x is None and inv_quantity > 1):
                step = {
                    "action": "bank-deposit-item-all",
                    "option": "Deposit-All",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    # Check if the correct interaction was performed
                    from helpers.ipc import get_last_interaction
                    last_interaction = get_last_interaction()
                    expected_action = "deposit-all"
                    expected_target = item_name
                    if (last_interaction and
                            expected_action in last_interaction.get("action").lower() and
                        clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                        return result
                    continue

            # --- Deposit-5 ---
            if deposit_x == 5:
                step = {
                    "action": "bank-deposit-item-5",
                    "option": "Deposit-5",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    return result
                continue

            # --- Deposit-10 ---
            if deposit_x == 10:
                step = {
                    "action": "bank-deposit-item-10",
                    "option": "Deposit-10",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    return result
                continue

            # --- Deposit-1 ---
            if deposit_x == 1 or (deposit_x is None and inv_quantity == 1):
                step = {
                    "action": "bank-deposit-item-single",
                    "option": "Deposit-1",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    return result
                continue

            # --- Deposit-X (custom amount) ---
            if deposit_x is not None and deposit_x > 10:
                step = {
                    "action": "bank-deposit-item-x",
                    "option": "Deposit-X",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
                result = dispatch(step)
                if result:
                    # Type the amount after Deposit-X is selected
                    from helpers.keyboard import type_text, press_enter
                    from helpers.utils import sleep_exponential, exponential_number
                    from .timing import wait_until
                    sleep_exponential(0.3, 0.8, 1.2)
                    type_text(str(deposit_x))
                    min_wait = int(exponential_number(400, 800, 1.0, "int"))
                    sleep_exponential(0.1, 0.3, 1.5)
                    press_enter()
                    sleep_exponential(0.3, 0.8, 1.2)
                    return result
                continue

            # --- Default: use context-click with appropriate option ---
            if inv_quantity == 1:
                step = {
                    "action": "bank-deposit-item-single",
                    "option": "Deposit-1",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
            else:
                step = {
                    "action": "bank-deposit-item-all",
                    "option": "Deposit-All",
                    "click": {
                        "type": "context-select",
                        "target": item_name.lower(),
                        "x": cx,
                        "y": cy,
                        "open_delay_ms": 120
                    },
                    "target": {
                        "domain": "bank-player-inventory",
                        "name": item_name,
                        "bounds": bounds
                    },
                    "postconditions": [],
                }
            
            result = dispatch(step)
            if result:
                return result

        return None

    except Exception as e:
        logging.error(f"[deposit_item] actions/bank.py: {e}")
        return None


def _interact_single_item(item_name: str, action: str | list) -> dict | None:
    """Interact with a single bank inventory item."""

    # Find the item in bank inventory
    item = find_bank_inventory_item(item_name)
    if not item:
        print(f"[BANK] Item '{item_name}' not found in bank inventory")
        return None

    # Get item bounds for clicking
    bounds = item.get("bounds")
    if not bounds:
        print(f"[BANK] No bounds found for item '{item_name}'")
        return None

    # Get canvas coordinates for context click
    canvas = item.get("canvas")
    if not canvas:
        print(f"[BANK] No canvas coordinates found for item '{item_name}'")
        return None

    # Handle single action
    if isinstance(action, str):
        return _try_action(item_name, action, bounds, canvas)

    # Handle list of actions - try each one until one works
    if isinstance(action, list):
        for action_option in action:
            if not isinstance(action_option, str):
                print(f"[BANK] Skipping invalid action: {action_option}")
                continue

            print(f"[BANK] Trying action '{action_option}' for item '{item_name}'")
            result = _try_action(item_name, action_option, bounds, canvas)
            if result:
                print(f"[BANK] Successfully used action '{action_option}' for item '{item_name}'")
                return result
            else:
                print(f"[BANK] Action '{action_option}' not available for item '{item_name}'")

        print(f"[BANK] No available actions found for item '{item_name}' from list: {action}")
        return None

    print(f"[BANK] Invalid action type: {type(action)}")
    return None


def _try_action(item_name: str, action: str, bounds: dict, canvas: dict) -> dict | None:
    """Try a specific action on an item."""
    print(f"[BANK] Attempting action '{action}' on item '{item_name}'")

    # Use bounds for randomized coordinates, fallback to canvas
    if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
        cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                               bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    else:
        cx, cy = int(canvas["x"]), int(canvas["y"])

    # Pre-check: if the desired menu entry isn't present, do NOT attempt the interaction.
    # This avoids opening the context menu / clicking when the bank menu option doesn't exist.
    try:
        # Hover at the target point to populate the live menu entries.
        hover_result = ipc.click(cx, cy, hover_only=True)
        if not (hover_result and hover_result.get("ok")):
            return None

        sleep_exponential(0.05, 0.15, 1.5)

        info = ipc._send({"cmd": "menu"}) or {}
        entries = info.get("entries") or []
        want_opt = clean_rs(action).lower()
        want_tgt = clean_rs(item_name).lower()

        def _match(e) -> bool:
            eo = clean_rs((e.get("option") or "")).lower()
            et = clean_rs((e.get("target") or "")).lower()
            # bank targets sometimes include extra text (quantity, tags); allow either-direction containment
            opt_ok = (eo == want_opt) or (want_opt in eo) or (eo in want_opt)
            tgt_ok = (not want_tgt) or (want_tgt in et) or (et in want_tgt)
            return opt_ok and tgt_ok

        if not any(_match(e) for e in entries):
            print(f"[BANK] Menu option '{action}' not present for '{item_name}', skipping")
            return None
    except Exception:
        # If menu probing fails, fall back to existing behavior (best effort).
        pass

    # Create context click step
    step = {
        "action": "bank-inventory-interact",
        "option": action,
        "click": {
            "type": "context-select",
            "target": item_name.lower(),
            "x": cx,
            "y": cy,
            "open_delay_ms": 120
        },
        "target": {"domain": "bank-inventory", "name": item_name, "bounds": bounds},
        "postconditions": [],
    }

    try:
        result = dispatch(step)
        return result
    except Exception as e:
        print(f"[BANK] Action '{action}' failed for item '{item_name}': {e}")
        return None


def _interact_multiple_items(item_names: list, action: str | list) -> dict | None:
    """
    Interact with multiple bank inventory items.

    Args:
        item_names: List of item names to interact with
        action: Action(s) to perform on each item (string or list)

    Returns:
        Result of the last successful interaction, or None if all failed
    """
    if not isinstance(item_names, list) or not item_names:
        print("[BANK] Invalid item list provided")
        return None

    print(f"[BANK] Interacting with {len(item_names)} items: {item_names}")
    print(f"[BANK] Using action(s): {action}")

    last_result = None
    successful_interactions = 0

    for item_name in item_names:
        if not isinstance(item_name, str):
            print(f"[BANK] Skipping invalid item: {item_name}")
            continue

        print(f"[BANK] Interacting with item: {item_name}")

        # Try to interact with this item
        result = _interact_single_item(item_name, action)

        if result:
            last_result = result
            successful_interactions += 1
            print(f"[BANK] Successfully interacted with: {item_name}")
        else:
            print(f"[BANK] Failed to interact with: {item_name}")

        # Small delay between interactions to avoid overwhelming the interface
        sleep_exponential(0.1, 0.3, 1.5)

    print(f"[BANK] Interaction complete: {successful_interactions}/{len(item_names)} items successful")
    return last_result


def get_bank_inventory() -> List[dict]:
    """
    Get all items in the bank (actual bank contents, not interface).

    Returns:
        List of bank items with name, quantity, and other details
    """

    resp = ipc.get_bank()
    if not resp or not resp.get("ok"):
        print(f"[BANK] Failed to get bank contents: {resp.get('err', 'Unknown error')}")
        return []

    bank_items = resp.get("items", [])
    print(f"[BANK] Retrieved {len(bank_items)} bank items")
    return bank_items


def get_item_count(item_name: str) -> int:
    """
    Get the count of a specific item in the bank.

    Args:
        item_name: Name of the item to count

    Returns:
        Number of the item in the bank, or 0 if not found
    """
    item = first_bank_slot(item_name)
    return item.get("quantity", 0)


def get_bank_contents() -> list:
    """
    Get all items in the bank.

    Returns:
        List of bank items, or empty list if failed
    """

    resp = ipc.get_bank()
    if not resp or not resp.get("ok"):
        print(f"[BANK] Failed to get bank contents: {resp.get('err', 'Unknown error')}")
        return []

    bank_items = resp.get("items", [])
    print(f"[BANK] Retrieved {len(bank_items)} bank items")
    return bank_items

def ensure_quantity_button_set(quantity: str):
    if quantity == "1":
        while not get_bank_quantity_mode().get("mode") == 0:
            set_quantity_button("1")
            wait_until(lambda: get_bank_quantity_mode().get("mode") == 0, max_wait_ms=exponential_number(0.4, 1, 1.5))
    elif quantity == "5":
        while not get_bank_quantity_mode().get("mode") == 1:
            set_quantity_button("5")
            wait_until(lambda: get_bank_quantity_mode().get("mode") == 1, max_wait_ms=exponential_number(0.4, 1, 1.5))
    elif quantity == "10":
        while not get_bank_quantity_mode().get("mode") == 2:
            set_quantity_button("10")
            wait_until(lambda: get_bank_quantity_mode().get("mode") == 2, max_wait_ms=exponential_number(0.4, 1, 1.5))
    elif quantity == "X":
        while not get_bank_quantity_mode().get("mode") == 3:
            set_quantity_button("X")
            wait_until(lambda: get_bank_quantity_mode().get("mode") == 3, max_wait_ms=exponential_number(0.4, 1, 1.5))
    elif quantity == "All":
        while not get_bank_quantity_mode().get("mode") == 4:
            set_quantity_button("All")
            wait_until(lambda: get_bank_quantity_mode().get("mode") == 4, max_wait_ms=exponential_number(0.4, 1, 1.5))


def set_quantity_button(quantity: str) -> dict | None:
    """
    Set the bank quantity button (1, 5, 10, X, All).

    Args:
        quantity: Quantity to set ("1", "5", "10", "X", "All")

    Returns:
        Result of the click action, or None if failed
    """
    from constants import BANK_WIDGETS

    # Map quantity strings to widget IDs
    quantity_map = {
        "1": "QUANTITY1",
        "5": "QUANTITY5",
        "10": "QUANTITY10",
        "X": "QUANTITYX",
        "All": "QUANTITYALL"
    }

    if quantity not in quantity_map:
        logging.warning(f"[SET_QUANTITY] Invalid quantity '{quantity}'. Must be one of: {list(quantity_map.keys())}")
        return None

    widget_key = quantity_map[quantity]
    widget_id = BANK_WIDGETS.get(widget_key)

    if not widget_id:
        logging.warning(f"[SET_QUANTITY] Widget ID not found for quantity '{quantity}'")
        return None

    # Get widget info
    click_widget(widget_id)

def search_bank(search_term: str) -> dict | None:
    """
    Search for items in the bank.

    Args:
        search_term: Term to search for

    Returns:
        Result of the search action, or None if failed
    """

    # Inner attempt loop with fresh coordinate recalculation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Fresh coordinate recalculation
        search_widgets = get_bank_search_widgets()
        search_box = None

        # Find the search box widget
        for widget in search_widgets:
            if widget.get("name") == "search_box":
                search_box = widget
                break

        if not search_box:
            logging.warning("[SEARCH_BANK] Bank search box not found")
            continue

        bounds = search_box.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.warning("[SEARCH_BANK] Bank search box has invalid bounds")
            continue

        # Click the search box first
        cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                               bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        step1 = {
            "action": "bank-search-click",
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "bank-widget", "name": "Search Box", "bounds": bounds},
            "postconditions": [],
        }

        result1 = dispatch(step1)

        if result1:
            # Check if the correct interaction was performed
            from helpers.ipc import get_last_interaction
            last_interaction = get_last_interaction()

            expected_action = "bank-search-click"
            expected_target = "Search Box"

            if (last_interaction and
                last_interaction.get("action") == expected_action and
                clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                print(f"[CLICK] {expected_target} - interaction verified")

                # Type the search term
                step2 = {
                    "action": "bank-search-type",
                    "click": {"type": "type", "text": search_term, "enter": False, "per_char_ms": 15, "focus": True}
                }
                return dispatch(step2)
            else:
                print(f"[CLICK] {expected_target} - incorrect interaction, retrying...")
                continue

    return None

def clear_bank_search() -> dict | None:
    """Clear the bank search."""
    return search_bank("")

def get_bank_tab_by_name(tab_name: str) -> dict | None:
    """
    Get a bank tab by name.

    Args:
        tab_name: Name of the tab to find

    Returns:
        Tab widget data, or None if not found
    """

    tabs = get_bank_tabs()
    search_name = tab_name.lower().strip()

    for tab in tabs:
        tab_text = tab.get("text", "").lower().strip()
        if tab_text == search_name:
            return tab
    return None

def click_bank_tab(tab_name: str) -> dict | None:
    """
    Click a bank tab by name.

    Args:
        tab_name: Name of the tab to click

    Returns:
        Result of the click action, or None if failed
    """
    # Inner attempt loop with fresh coordinate recalculation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Fresh coordinate recalculation
        tab = get_bank_tab_by_name(tab_name)
        if not tab:
            logging.warning(f"[CLICK_BANK_TAB] Bank tab '{tab_name}' not found")
            continue

        bounds = tab.get("bounds")
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.warning(f"[CLICK_BANK_TAB] Bank tab '{tab_name}' has invalid bounds")
            continue

        cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                               bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)

        step = {
            "action": "bank-click-tab",
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "bank-widget", "name": f"Tab {tab_name}", "bounds": bounds},
            "postconditions": [],
        }

        result = dispatch(step)

        if result:
            # Check if the correct interaction was performed
            from helpers.ipc import get_last_interaction
            last_interaction = get_last_interaction()

            expected_action = "bank-click-tab"
            expected_target = f"Tab {tab_name}"

            if (last_interaction and
                last_interaction.get("action") == expected_action and
                clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                print(f"[CLICK] {expected_target} - interaction verified")
                return result
            else:
                print(f"[CLICK] {expected_target} - incorrect interaction, retrying...")
                continue

    return None


# Equipment setup functions for complex plans
def setup_equipment_from_bank(equipment_config: Dict, plan_id: str = "UNKNOWN") -> Tuple[bool, List[str]]:
    """
    Set up equipment from bank based on skill levels and available items.

    Args:
        equipment_config: Equipment configuration dict
        plan_id: Plan ID for logging

    Returns:
        Tuple of (success, missing_items)
    """
    try:
        missing_items = []

        # Setup weapons
        if "weapon_tiers" in equipment_config and equipment_config["weapon_tiers"]:
            success, missing_weapons = _setup_weapons(equipment_config["weapon_tiers"], plan_id)
            if not success:
                missing_items.extend(missing_weapons)

        # Setup armor
        if "armor_tiers" in equipment_config and equipment_config["armor_tiers"]:
            success, missing_armor = _setup_armor(equipment_config["armor_tiers"], plan_id)
            if not success:
                missing_items.extend(missing_armor)

        # Setup jewelry
        if "jewelry_tiers" in equipment_config and equipment_config["jewelry_tiers"]:
            success, missing_jewelry = _setup_jewelry(equipment_config["jewelry_tiers"], plan_id)
            if not success:
                missing_items.extend(missing_jewelry)

        # Setup tools (with conditional equipping)
        if "tool_tiers" in equipment_config and equipment_config["tool_tiers"]:
            success, missing_tools = _setup_tools(equipment_config["tool_tiers"], plan_id, allow_fallback=True)
            if not success:
                missing_items.extend(missing_tools)

        return len(missing_items) == 0, missing_items

    except Exception as e:
        logging.error(f"[setup_equipment_from_bank] actions/bank.py: {e}")
        return False, [f"Equipment setup error: {e}"]


def _setup_weapons(weapon_tiers: List[Dict], plan_id: str) -> Tuple[bool, List[str]]:
    """Setup weapons based on attack level."""
    try:
        from .equipment import get_best_weapon_for_level

        target_weapon = get_best_weapon_for_level(weapon_tiers, plan_id)
        if not target_weapon:
            return True, []  # No weapon needed

        weapon_name = target_weapon["name"]

        # Check if we have the weapon
        if not (has_item(weapon_name) or inventory.has_item(weapon_name) or equipment.has_equipped(weapon_name)):
            return False, [weapon_name]

        # Equip the weapon if not already equipped
        if not equipment.has_equipped(weapon_name):
            if inventory.has_item(weapon_name):
                equipment.equip_item(weapon_name)
            elif has_item(weapon_name):
                withdraw_item(weapon_name, 1)
                equipment.equip_item(weapon_name)

        return True, []

    except Exception as e:
        logging.error(f"[_setup_weapons] actions/bank.py: {e}")
        return False, [f"Weapon setup error: {e}"]


def _setup_armor(armor_tiers: Dict, plan_id: str) -> Tuple[bool, List[str]]:
    """Setup armor based on defence level."""
    try:
        from .equipment import get_best_armor_for_level

        target_armor_dict = get_best_armor_for_level(armor_tiers, plan_id)
        if not target_armor_dict:
            return True, []  # No armor needed

        missing_items = []

        for armor_type, armor_item in target_armor_dict.items():
            armor_name = armor_item["name"]

            # Check if we have the armor piece
            if not (has_item(armor_name) or inventory.has_item(armor_name) or equipment.has_equipped(armor_name)):
                missing_items.append(armor_name)
                continue

            # Equip the armor if not already equipped
            if not equipment.has_equipped(armor_name):
                if inventory.has_item(armor_name):
                    equipment.equip_item(armor_name)
                elif has_item(armor_name):
                    withdraw_item(armor_name, 1)
                    equipment.equip_item(armor_name)

        return len(missing_items) == 0, missing_items

    except Exception as e:
        logging.error(f"[_setup_armor] actions/bank.py: {e}")
        return False, [f"Armor setup error: {e}"]


def _setup_jewelry(jewelry_tiers: Dict, plan_id: str) -> Tuple[bool, List[str]]:
    """Setup jewelry based on defence level."""
    try:
        from .equipment import get_best_armor_for_level

        target_jewelry_dict = get_best_armor_for_level(jewelry_tiers, plan_id)
        if not target_jewelry_dict:
            return True, []  # No jewelry needed

        missing_items = []

        for jewelry_type, jewelry_item in target_jewelry_dict.items():
            jewelry_name = jewelry_item["name"]

            # Check if we have the jewelry piece
            if not (has_item(jewelry_name) or inventory.has_item(jewelry_name) or equipment.has_equipped(jewelry_name)):
                missing_items.append(jewelry_name)
                continue

            # Equip the jewelry if not already equipped
            if not equipment.has_equipped(jewelry_name):
                if inventory.has_item(jewelry_name):
                    equipment.equip_item(jewelry_name)
                elif has_item(jewelry_name):
                    withdraw_item(jewelry_name, 1)
                    equipment.equip_item(jewelry_name)

        return len(missing_items) == 0, missing_items

    except Exception as e:
        logging.error(f"[_setup_jewelry] actions/bank.py: {e}")
        return False, [f"Jewelry setup error: {e}"]


def _setup_tools(tool_tiers: List[Tuple], plan_id: str, allow_fallback: bool = False) -> Tuple[bool, List[str]]:
    """Setup tools with optional fallback logic."""
    try:
        from .equipment import get_best_tool_for_level

        target_tool, _, _, _ = get_best_tool_for_level(tool_tiers, "woodcutting", plan_id)
        if not target_tool:
            return True, []  # No tool needed

        # Check if we have the best tool
        if has_item(target_tool) or inventory.has_item(target_tool) or equipment.has_equipped(target_tool):
            # Equip the tool if not already equipped
            if not equipment.has_equipped(target_tool):
                if inventory.has_item(target_tool):
                    equipment.equip_item(target_tool)
                elif has_item(target_tool):
                    withdraw_item(target_tool, 1)
                    equipment.equip_item(target_tool)
            return True, []

        # If we don't have the best tool and fallback is allowed, try to find a fallback
        if allow_fallback:
            fallback_tool = _find_fallback_tool(target_tool, tool_tiers)
            if fallback_tool:
                logging.info(f"[_setup_tools] Using fallback tool: {fallback_tool} instead of {target_tool}")
                if not equipment.has_equipped(fallback_tool):
                    if inventory.has_item(fallback_tool):
                        equipment.equip_item(fallback_tool)
                    elif has_item(fallback_tool):
                        withdraw_item(fallback_tool, 1)
                        equipment.equip_item(fallback_tool)
                return True, []  # Success with fallback

        return False, [target_tool]

    except Exception as e:
        logging.error(f"[_setup_tools] actions/bank.py: {e}")
        return False, [f"Tool setup error: {e}"]


def _find_fallback_tool(target_tool: str, tool_tiers: List[Tuple]) -> Optional[str]:
    """Find a fallback tool that we have available."""
    try:
        # Find the index of the target tool in the tiers
        target_index = -1
        for i, (tool_name, _, _, _) in enumerate(tool_tiers):
            if tool_name == target_tool:
                target_index = i
                break

        if target_index == -1:
            return None

        # Look for any tool we have that's lower tier (higher index)
        for i in range(target_index + 1, len(tool_tiers)):
            tool_name, _, _, _ = tool_tiers[i]
            if has_item(tool_name) or inventory.has_item(tool_name) or equipment.has_equipped(tool_name):
                return tool_name

        return None

    except Exception as e:
        logging.error(f"[_find_fallback_tool] actions/bank.py: {e}")
        return None


def check_sellable_items(sellable_items: Dict) -> Tuple[bool, Dict, Optional[str]]:
    """
    Check if we have items to sell and determine what equipment we could buy.

    Args:
        sellable_items: Dict of {item_name: min_quantity}

    Returns:
        Tuple of (has_sellable_items, items_to_sell, target_equipment)
    """
    try:
        items_to_sell = {}
        target_equipment = None

        for item_name, min_quantity in sellable_items.items():
            count = get_item_count(item_name)
            if count >= min_quantity:
                items_to_sell[item_name] = count
                logging.info(f"[check_sellable_items] Found {count} {item_name} to sell")

        if items_to_sell:
            # Determine what equipment we could buy with the proceeds
            # This logic would be plan-specific
            target_equipment = "Better equipment"  # Placeholder

        return len(items_to_sell) > 0, items_to_sell, target_equipment

    except Exception as e:
        logging.error(f"[check_sellable_items] actions/bank.py: {e}")
        return False, {}, None


def equip_item(item_name: str, slot: str) -> bool:
    """
    Equip an item to a specific slot using the appropriate action.

    Args:
        item_name: Name of the item to equip
        slot: Equipment slot (weapon, helmet, body, legs, shield, amulet, etc.)

    Returns:
        True if the equip action was successful, False otherwise
    """
    try:
        # Try both common equip actions; prefer the typical one for the slot.
        # Some items/contexts swap "Wear" vs "Wield", so we fall back to the other.
        actions = ["wield", "wear"] if slot == "weapon" else ["wear", "wield"]

        for action in actions:
            result = interact(item_name, action)
            if result is not None:
                return True

        return False

    except Exception as e:
        logging.error(f"[equip_item] actions/bank.py: Error equipping {item_name} to {slot}: {e}")
        return False


def has_materials_available(item_names: list[str]) -> dict[str, bool]:
    """
    Check if materials are available in either bank OR inventory.
    
    Useful for determining if a plan can continue before attempting to withdraw items.
    Checks both bank and inventory for each item.
    
    Args:
        item_names: List of item names to check for
        
    Returns:
        Dictionary mapping item names to availability (True if available in bank or inventory, False otherwise)
    """
    try:
        availability = {}
        
        for item_name in item_names:
            # Check bank
            bank_count = get_item_count(item_name)
            bank_has_item = bank_count is not None and int(bank_count) > 0
            
            # Check inventory
            inv_count_val = inv_count(item_name)
            inventory_has_item = inv_count_val is not None and int(inv_count_val) > 0
            
            # Available if in either bank or inventory
            availability[item_name] = bank_has_item or inventory_has_item
        
        return availability
        
    except Exception as e:
        logging.error(f"[has_materials_available] actions/bank.py: {e}")
        return {name: False for name in item_names}


def close_bank_no_camera(
    object_name: str | None = None,
    object_action: str | None = None,
    click_object_probability: float | tuple[float, float] | None = None,
) -> dict | None:
    """
    Close the bank without using camera movement. Guarantees no camera movement.
    
    If object_name and object_action are provided, may click the object directly (which auto-closes bank).
    If the object is blocked by the bank interface, closes bank normally instead of using camera.
    If the no-camera object click fails, closes bank normally instead of falling back to camera.
    
    Args:
        object_name: Optional name of object to potentially click (e.g., "Furnace")
        object_action: Optional action to perform on object (e.g., "Smelt")
        click_object_probability: Probability to click object instead of closing bank.
                                 If tuple, uses random range (min, max). If float, uses that value.
                                 If None and object_name provided, uses default range (0.2347, 0.3891).
                                 If None and no object_name, always closes bank normally.
    
    Returns:
        Dict with keys:
            - "action": "object_click" | "normal_close" | "failed"
            - "result": Original result from the action (dict or None)
        Or None if bank is not open
    """
    try:
        # Check if bank is open first
        if not is_open():
            return None
        
        # If object_name and object_action provided, potentially click object instead
        if object_name and object_action:
            import random
            from constants import BANK_WIDGETS
            from actions.widgets import get_widget_bounds
            from actions import objects
            
            # Determine probability
            if click_object_probability is None:
                # Default probability range
                click_probability = random.uniform(0.2347, 0.3891)
            elif isinstance(click_object_probability, tuple):
                # Range provided
                click_probability = random.uniform(click_object_probability[0], click_object_probability[1])
            else:
                # Single probability value
                click_probability = float(click_object_probability)
            
            # Randomly decide whether to click object directly
            click_object_directly = random.random() < click_probability
            
            if click_object_directly:
                # Get ALL objects matching the name with the required action
                from actions.objects import _normalize_actions, _has_any_exact_action
                
                req_actions = _normalize_actions(object_action) if object_action else []
                obj_resp = ipc.get_objects(object_name, types=["GAME"], radius=26) or {}
                matching_objects = []
                
                if obj_resp.get("ok"):
                    all_objs = obj_resp.get("objects") or []
                    for obj in all_objs:
                        # Filter by required action if specified
                        if req_actions:
                            if not _has_any_exact_action(obj, req_actions):
                                continue
                        matching_objects.append(obj)
                
                # Get bank interface bounds
                bank_bounds = get_widget_bounds(BANK_WIDGETS["UNIVERSE"])
                
                # Check all matching objects to find one that's unblocked
                unblocked_obj = None
                for obj in matching_objects:
                    obj_canvas = obj.get("canvas", {})
                    if not bank_bounds or not obj_canvas:
                        # Can't check blocking - use this object
                        unblocked_obj = obj
                        break
                    
                    obj_canvas_x = obj_canvas.get("x")
                    obj_canvas_y = obj_canvas.get("y")
                    
                    if isinstance(obj_canvas_x, (int, float)) and isinstance(obj_canvas_y, (int, float)):
                        if not _is_point_in_widget_bounds(int(obj_canvas_x), int(obj_canvas_y), bank_bounds):
                            # This object is not blocked - use it
                            unblocked_obj = obj
                            break
                    else:
                        # Can't determine if blocked - use this object
                        unblocked_obj = obj
                        break
                
                if unblocked_obj:
                    # Found an unblocked object - try no-camera click
                    result = objects.click_object_closest_by_distance_simple_no_camera(
                        object_name, 
                        prefer_action=object_action
                    )
                    if result:
                        return {"action": "object_click", "result": result}
                    # No-camera click failed - close bank normally instead of using camera
                    logging.info(f"[CLOSE_BANK_NO_CAMERA] No-camera object click failed, closing bank normally")
                    # Fall through to normal bank close
                elif matching_objects:
                    # All matching objects are blocked by bank interface - close bank normally
                    logging.info(f"[CLOSE_BANK_NO_CAMERA] All {len(matching_objects)} matching objects are blocked by bank interface, closing bank normally")
                    # Fall through to normal bank close
                else:
                    # No matching objects found - try no-camera click anyway (might work)
                    result = objects.click_object_closest_by_distance_simple_no_camera(
                        object_name, 
                        prefer_action=object_action
                    )
                    if result:
                        return {"action": "object_click", "result": result}
                    # No-camera click failed - close bank normally
                    logging.info(f"[CLOSE_BANK_NO_CAMERA] No matching objects found and no-camera click failed, closing bank normally")
                    # Fall through to normal bank close
        
        # Close bank normally (ESC key - never uses camera)
        step = {
            "action": "close-bank",
            "click": {"type": "key", "key": "esc"},
            "target": {"domain": "bank", "name": "Close"},
            "postconditions": ["bankOpen == false"],
        }
        result = dispatch(step)
        return {"action": "normal_close", "result": result}
        
    except Exception as e:
        logging.error(f"[close_bank_no_camera] actions/bank.py: {e}")
        # Fallback to normal bank close
        step = {
            "action": "close-bank",
            "click": {"type": "key", "key": "esc"},
            "target": {"domain": "bank", "name": "Close"},
            "postconditions": ["bankOpen == false"],
        }
        result = dispatch(step)
        return {"action": "normal_close", "result": result}


def close_bank_or_click_object(
    object_name: str,
    action: str,
    click_probability_range: tuple[float, float] = (0.2347, 0.3891),
    prefer_no_camera: bool = False
) -> dict | None:
    """
    Randomly choose between closing bank normally or clicking an object directly.
    
    Sometimes clicks the object directly (which auto-closes bank), other times closes bank first.
    Adds human-like variance to bank closing behavior.
    
    Args:
        object_name: Name of the object to potentially click
        action: Action to perform on the object (e.g., "Smelt", "Use")
        click_probability_range: Tuple of (min, max) probability to click object directly (default: ~23-39%)
        prefer_no_camera: If True, uses prefer_no_camera variant for object click
        
    Returns:
        Result of the action taken, or None if failed
    """
    try:
        import random
        from actions import objects
        
        # Randomly decide whether to click object directly
        click_probability = random.uniform(click_probability_range[0], click_probability_range[1])
        click_object_directly = random.random() < click_probability
        
        if click_object_directly:
            # Click object directly (will auto-close bank)
            if prefer_no_camera:
                result = objects.click_object_closest_by_distance_simple_prefer_no_camera(object_name, prefer_action=action)
            else:
                result = objects.click_object_closest_by_distance_simple(object_name, prefer_action=action)
            return result
        else:
            # Close bank normally
            return close_bank()
            
    except Exception as e:
        logging.error(f"[close_bank_or_click_object] actions/bank.py: {e}")
        # Fallback to normal bank close
        return close_bank()