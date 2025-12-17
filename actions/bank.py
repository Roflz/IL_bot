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

def open_bank(prefer: str | None = None) -> dict | None:
    """
    Open a bank using the new IPC system with proper response handling.
    """
    prefer = (prefer or "").strip().lower()
    want_booth = prefer in ("bank booth", "grand exchange booth")
    want_banker = prefer == "banker"

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
    banker_resp = ipc.get_npcs("banker")

    # Extract objects and NPCs from responses
    booths = booth_resp.get("objects", []) if booth_resp and booth_resp.get("ok") else []
    bankers = banker_resp.get("npcs", []) if banker_resp and banker_resp.get("ok") else []

    logging.info(f"[OPEN_BANK] Found {len(booths)} booths, {len(bankers)} bankers")

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

    # Try to find booth first (unless specifically wanting banker)
    if not want_banker and booths:
        for booth in booths:
            name = booth.get("name", "").lower()
            actions = booth.get("actions", [])
            bank_idx = find_bank_action(actions)

            if bank_idx is not None and ("bank booth" in name or "grand exchange booth" in name):
                logging.info(f"[OPEN_BANK] Selected booth: {booth.get('name')}")
                target = booth
                target_type = "object"
                break

    # Try any booth if no specific booth found
    if target is None and not want_banker and booths:
        for booth in booths:
            actions = booth.get("actions", [])
            bank_idx = find_bank_action(actions)
            if bank_idx is not None:
                logging.info(f"[OPEN_BANK] Selected any booth: {booth.get('name')}")
                target = booth
                target_type = "object"
                break

    # Try banker if no booth or specifically wanting banker
    if target is None and not want_booth and bankers:
        for banker in bankers:
            name = banker.get("name", "").lower()
            actions = banker.get("actions", [])
            bank_idx = find_bank_action(actions)

            if bank_idx is not None and ("banker" in name or want_banker):
                logging.info(f"[OPEN_BANK] Selected banker: {banker.get('name')}")
                target = banker
                target_type = "npc"
                break

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

    # Use click_with_camera methods for both objects and NPCs
    world_coords_dict = {"x": gx, "y": gy, "p": 0}

    if target_type == "object":
        from ..services.click_with_camera import click_object_with_camera
        result = click_object_with_camera(
            object_name=name,
            action=action,
            world_coords=world_coords_dict,
            aim_ms=420
        )
    else:  # NPC
        from ..services.click_with_camera import click_npc_with_camera
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
        max_wait = 3000  # Increased wait time for bank interface

        # Wait for bank interface to open
        bank_opened = wait_until(is_open, max_wait_ms=max_wait, min_wait_ms=min_wait)

        if bank_opened:
            logging.info(f"[OPEN_BANK] Bank interface opened successfully")

            # Handle any special widgets
            sleep_exponential(0.3, 0.8, 1.2)  # Let interface load

            from ..actions.widgets import click_widget

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

    # --- Withdraw-All via live context menu ---
    if withdraw_all:
        step = {
            "action": "withdraw-item-all",
            "option": "withdraw-all",
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


def close_bank() -> dict | None:
    step = {
        "action": "close-bank",
        "click": {"type": "key", "key": "esc"},
        "target": {"domain": "bank", "name": "Close"},
        "postconditions": ["bankOpen == false"],
    }
    return dispatch(step)

def toggle_note_mode() -> dict | None:
    """Toggle bank note mode (withdraw as note/withdraw as item)"""
    from ..constants import BANK_WIDGETS

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
                from ..helpers.ipc import get_last_interaction
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


def deposit_item(item_name: str) -> dict | None:
    """
    Deposit an item from player's inventory to the bank.
    Automatically chooses the appropriate deposit method based on quantity:
    - If quantity = 1: Uses left-click (Deposit-1)
    - If quantity > 1: Uses context-click (Deposit-All)

    Args:
        item_name: Name of the item to deposit

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
            quantity = inv_count(item_name)

            if quantity <= 0:
                print(f"[BANK] Item '{item_name}' has no quantity to deposit")
                continue

            cx, cy = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                   bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)

            # Choose deposit method based on quantity
            if quantity == 1:
                # Use simple left-click for single items
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
                print(f"[BANK] Depositing single item '{item_name}' (left-click)")
            else:
                # Use context-click for multiple items
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
                print(f"[BANK] Depositing all items '{item_name}' (quantity: {quantity})")

            result = dispatch(step)

            if result:
                # Check if the correct interaction was performed
                from ..helpers.ipc import get_last_interaction
                last_interaction = get_last_interaction()

                expected_action = "deposit-" if quantity == 1 else "deposit-all"
                expected_target = item_name

                if (last_interaction and
                    expected_action in last_interaction.get("action") and
                    clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                    print(f"[CLICK] {expected_target} - interaction verified")
                    return result
                else:
                    print(f"[CLICK] {expected_target} - incorrect interaction, retrying...")
                    continue

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
    from ..constants import BANK_WIDGETS

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
            from ..helpers.ipc import get_last_interaction
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
            from ..helpers.ipc import get_last_interaction
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
        # Determine the correct action based on slot type
        if slot == "weapon":
            action = "wield"
        else:
            action = "wear"

        # Use bank.interact to equip the item
        result = interact(item_name, action)
        return result is not None

    except Exception as e:
        logging.error(f"[equip_item] actions/bank.py: Error equipping {item_name} to {slot}: {e}")
        return False