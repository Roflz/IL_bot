# ilbot/ui/simple_recorder/actions/banking.py
import time
import logging
from typing import List

from . import ge, widgets
from .timing import wait_until
from .travel import in_area
from ..helpers.bank import first_bank_slot, deposit_all_button_bounds
from ..helpers.inventory import inv_has_any, norm_name
from ..helpers.runtime_utils import ipc, ui, dispatch
from ..helpers.rects import unwrap_rect, rect_center_xy
from ..services.camera_integration import dispatch_with_camera


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
        return first_bank_slot(name) is not None
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

    max_retries = 3
    
    for attempt in range(max_retries):
        logging.info(f"[OPEN_BANK] Attempt {attempt + 1}/{max_retries}")
        
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
                if action and "bank" in action.lower():
                    return i
            return None
        
        # Try to find booth first (unless specifically wanting banker)
        if not want_banker and booths:
            for booth in booths:
                name = booth.get("name", "").lower()
                actions = booth.get("actions", [])
                bank_idx = find_bank_action(actions)
                
                if bank_idx is not None and ("bank booth" in name or "grand exchange booth" in name):
                    target = booth
                    target_type = "object"
                    logging.info(f"[OPEN_BANK] Selected booth: {booth.get('name')}")
                    break
        
        # Try any booth if no specific booth found
        if target is None and not want_banker and booths:
            for booth in booths:
                actions = booth.get("actions", [])
                bank_idx = find_bank_action(actions)
                if bank_idx is not None:
                    target = booth
                    target_type = "object"
                    logging.info(f"[OPEN_BANK] Selected any booth: {booth.get('name')}")
                    break
        
        # Try banker if no booth or specifically wanting banker
        if target is None and not want_booth and bankers:
            for banker in bankers:
                name = banker.get("name", "").lower()
                actions = banker.get("actions", [])
                bank_idx = find_bank_action(actions)
                
                if bank_idx is not None and ("banker" in name or want_banker):
                    target = banker
                    target_type = "npc"
                    logging.info(f"[OPEN_BANK] Selected banker: {banker.get('name')}")
                    break
        
        if target is None:
            logging.warning(f"[OPEN_BANK] No suitable bank target found on attempt {attempt + 1}")
            continue
        
        # Handle door traversal if needed
        world_coords = target.get("world", {})
        gx, gy = world_coords.get("x"), world_coords.get("y")
        
        if isinstance(gx, int) and isinstance(gy, int):
            logging.info(f"[OPEN_BANK] Checking path to bank at ({gx}, {gy})")
            
            # Get path and check for doors
            wps, path_resp = ipc.path(goal=(gx, gy))
            if wps:
                from ..helpers.navigation import _first_blocking_door_from_waypoints
                from .travel import _handle_door_opening
                
                door_plan = _first_blocking_door_from_waypoints(wps)
                if door_plan:
                    logging.info(f"[OPEN_BANK] Found blocking door, attempting to open")
                    if not _handle_door_opening(door_plan):
                        logging.warning(f"[OPEN_BANK] Failed to open door, trying next attempt")
                        continue
        
        # Click the bank target
        actions = target.get("actions", [])
        bank_idx = find_bank_action(actions)
        name = target.get("name", "Bank")
        
        # Get click coordinates
        canvas = target.get("canvas", {})
        cx = canvas.get("x")
        cy = canvas.get("y")
        
        if not isinstance(cx, (int, float)) or not isinstance(cy, (int, float)):
            logging.warning(f"[OPEN_BANK] Invalid canvas coordinates: {canvas}")
            continue
        
        # Determine action to use
        action = None
        if bank_idx is not None and bank_idx < len(actions):
            action = actions[bank_idx]
        
        logging.info(f"[OPEN_BANK] Clicking {name} with action '{action}' at ({cx}, {cy})")
        
        # Use appropriate click function
        from ..services.click_with_camera import click_object_with_camera, click_npc_with_camera
        
        if target_type == "object":
            result = click_object_with_camera(
                object_name=name,
                action=action,
                world_coords={"x": gx, "y": gy, "p": 0},
                aim_ms=420
            )
        else:  # NPC
            result = click_npc_with_camera(
                npc_name=name,
                action=action,
                world_coords={"x": gx, "y": gy, "p": 0},
                aim_ms=420
            )
        
        if result:
            logging.info(f"[OPEN_BANK] Click successful, waiting for bank interface")
            
            # Wait for bank interface to open
            if wait_until(is_open, max_wait_ms=5000, min_wait_ms=200):
                logging.info(f"[OPEN_BANK] Bank interface opened successfully")
                
                # Handle any special widgets
                time.sleep(0.5)  # Let interface load
                
                from ..helpers.widgets import widget_exists
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
                logging.warning(f"[OPEN_BANK] Bank interface did not open, trying next attempt")
                continue
        else:
            logging.warning(f"[OPEN_BANK] Click failed, trying next attempt")
            continue
    
    logging.error(f"[OPEN_BANK] Failed to open bank after {max_retries} attempts")
    return None


def deposit_inventory() -> dict | None:
    if not inv_has_any():
        return None
    rect = deposit_all_button_bounds()
    if not rect:
        return None
    step = {
        "action": "bank-deposit-inv",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": "Deposit Inventory", "bounds": rect},
        "postconditions": [],
    }
    return dispatch(step)


def withdraw_item(
    name: str | list,
    withdraw_x: int | None = None,
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

    cx, cy = rect_center_xy(rect)
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

    # --- Withdraw-X via live context menu + type amount ---
    if withdraw_x is not None and withdraw_x > 1:
        # Step 1: Right-click and select "Withdraw-X"
        step1 = {
            "action": "withdraw-item-x-menu",
            "option": "withdraw-x",
            "click": {
                "type": "context-select",
                "target": item_name.lower(),
                "x": int(cx),
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        }
        dispatch(step1)
        if not wait_until(ge.chat_qty_prompt_active, min_wait_ms=300, max_wait_ms=3000):
            return None
        
        # Step 3: Type the quantity
        step3 = {
            "action": "type-withdraw-x",
            "click": {"type": "type", "text": str(int(withdraw_x)), "enter": False, "per_char_ms": 15, "focus": True}
        }
        dispatch(step3)
        if not wait_until(lambda: ge.buy_chatbox_text_input_contains(str(withdraw_x)), min_wait_ms=300, max_wait_ms=3000):
            return None

        
        # Step 4: Press Enter to confirm
        step4 = {"action": "confirm-withdraw-x", "click": {"type": "key", "key": "enter"}}
        dispatch(step4)
        if not wait_until(lambda: inv_has(item_name, min_qty=int(withdraw_x)), min_wait_ms=300, max_wait_ms=3000):
            return None
        return True

    # --- Default: simple left-click withdraw (Withdraw-1 / custom shift-click config) ---
    step = {
        "action": "withdraw-item",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        "postconditions": [f"inventory contains '{item_name}'"],
    }
    return dispatch_with_camera(step, aim_ms=420)


def withdraw_items(
    items: list,
    withdraw_x: int | None = None,
    withdraw_all: bool = False,
) -> dict | None:
    """
    Withdraw multiple items from the bank.
    
    Args:
        items: List of item names to withdraw
        withdraw_x: Amount to withdraw for each item (if None, uses default behavior)
        withdraw_all: If True, withdraw all of each item
        ui: UI instance
        
    Returns:
        Result of the last successful withdrawal, or None if all failed
    """
    if not isinstance(items, list) or not items:
        print("[BANK] withdraw_items: Invalid items list provided")
        return None
    
    print(f"[BANK] Withdrawing {len(items)} items: {items}")
    
    last_result = None
    successful_withdrawals = 0
    
    for item in items:
        if not isinstance(item, str):
            print(f"[BANK] Skipping invalid item: {item}")
            continue
            
        print(f"[BANK] Withdrawing item: {item}")
        
        # Try to withdraw this item
        result = withdraw_item(
            name=item,
            withdraw_x=withdraw_x,
            withdraw_all=withdraw_all,
        )
        
        if result:
            last_result = result
            successful_withdrawals += 1
            print(f"[BANK] Successfully withdrew: {item}")
        else:
            print(f"[BANK] Failed to withdraw: {item}")
        
        # Small delay between withdrawals to avoid overwhelming the interface
        import time
        time.sleep(0.3)
    
    print(f"[BANK] Withdrawal complete: {successful_withdrawals}/{len(items)} items successful")
    return last_result


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
    
    step = {
        "action": "bank-note-toggle",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": "Withdraw as Note", "bounds": bounds},
        "postconditions": [],
    }
    return dispatch(step)

def ensure_note_mode_disabled() -> dict | None:
    """Ensure bank note mode is disabled (withdraw as items, not notes)"""
    
    from ..helpers.bank import bank_note_selected
    if bank_note_selected():
        return toggle_note_mode()
    return None


def deposit_equipment() -> dict | None:
    """Deposit all equipped items into the bank."""
    from ..helpers.bank import get_deposit_equipment_button
    
    equipment_button = get_deposit_equipment_button()
    if not equipment_button:
        logging.warning("[DEPOSIT_EQUIPMENT] Equipment deposit button not found")
        return None
    
    bounds = equipment_button.get("bounds")
    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        logging.warning("[DEPOSIT_EQUIPMENT] Equipment deposit button has invalid bounds")
        return None
    
    step = {
        "action": "bank-deposit-equipment",
        "click": {"type": "rect-center"},
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
        # Get the bank inventory widget (ID 983043)
        widget_data = ipc.get_widget_children(983043)
        if not widget_data or not widget_data.get("ok"):
            print(f"[BANK] Failed to get bank inventory widget")
            return None
        
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
            return None
        
        # Get item bounds and canvas location
        bounds = target_item.get("bounds")
        canvas_location = target_item.get("canvasLocation")
        
        if not bounds or not canvas_location:
            print(f"[BANK] No valid bounds or canvas location for item '{item_name}'")
            return None
        
        # Create the interaction step
        step = {
            "action": "bank-player-inventory-interact",
            "option": action,
            "click": {
                "type": "context-select",
                "target": item_name.lower(),
                "x": int(canvas_location["x"]),
                "y": int(canvas_location["y"]),
                "open_delay_ms": 120
            },
            "target": {
                "domain": "bank-player-inventory", 
                "name": item_name, 
                "bounds": bounds
            },
            "postconditions": [],
        }
        
        print(f"[BANK] Interacting with inventory item '{item_name}' using action '{action}'")
        return dispatch(step)
        
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
        # Get the bank inventory widget (ID 983043)
        widget_data = ipc.get_widget_children(983043)
        if not widget_data or not widget_data.get("ok"):
            print(f"[BANK] Failed to get bank inventory widget")
            return None
        
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
            return None
        
        # Get item bounds and canvas location
        bounds = target_item.get("bounds")
        canvas_location = target_item.get("canvasLocation")
        
        if not bounds or not canvas_location:
            print(f"[BANK] No valid bounds or canvas location for item '{item_name}'")
            return None
        
        # Check quantity from inventory data
        from ..helpers.inventory import inv_count
        quantity = inv_count(item_name)
        
        if quantity <= 0:
            print(f"[BANK] Item '{item_name}' has no quantity to deposit")
            return None
        
        # Choose deposit method based on quantity
        if quantity == 1:
            # Use simple left-click for single items
            step = {
                "action": "bank-deposit-item-single",
                "click": {"type": "rect-center"},
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
                    "x": int(canvas_location["x"]),
                    "y": int(canvas_location["y"]),
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
        
        return dispatch(step)
        
    except Exception as e:
        logging.error(f"[deposit_item] actions/bank.py: {e}")
        return None


def _interact_single_item(item_name: str, action: str | list) -> dict | None:
    """Interact with a single bank inventory item."""
    from ..helpers.bank_inventory import find_bank_inventory_item

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
    
    # Create context click step
    step = {
        "action": "bank-inventory-interact",
        "option": action,
        "click": {
            "type": "context-select",
            "target": item_name.lower(),
            "x": int(canvas["x"]),
            "y": int(canvas["y"]),
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
        import time
        time.sleep(0.2)
    
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
    bank_items = get_bank_contents()
    if not bank_items:
        return 0
    
    total_count = 0
    for item in bank_items:
        if item.get("name") == item_name:
            total_count += item.get("quantity", 0)
    
    return total_count


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

def set_quantity_button(quantity: str) -> dict | None:
    """
    Set the bank quantity button (1, 5, 10, X, All).
    
    Args:
        quantity: Quantity to set ("1", "5", "10", "X", "All")
    
    Returns:
        Result of the click action, or None if failed
    """
    from ..helpers.bank import get_bank_quantity_buttons
    
    quantity_buttons = get_bank_quantity_buttons()
    target_button = None
    
    # Find the target quantity button
    for button_group in quantity_buttons:
        if button_group.get("name") == f"quantity{quantity}":
            buttons = button_group.get("buttons", [])
            for button in buttons:
                if button.get("visible", False):
                    target_button = button
                    break
            break
    
    if not target_button:
        logging.warning(f"[SET_QUANTITY] Quantity button '{quantity}' not found or not visible")
        return None
    
    bounds = target_button.get("bounds")
    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        logging.warning(f"[SET_QUANTITY] Quantity button '{quantity}' has invalid bounds")
        return None
    
    step = {
        "action": "bank-set-quantity",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": f"Quantity {quantity}", "bounds": bounds},
        "postconditions": [],
    }
    return dispatch(step)

def search_bank(search_term: str) -> dict | None:
    """
    Search for items in the bank.
    
    Args:
        search_term: Term to search for
    
    Returns:
        Result of the search action, or None if failed
    """
    from ..helpers.bank import get_bank_search_widgets
    
    search_widgets = get_bank_search_widgets()
    search_box = None
    
    # Find the search box widget
    for widget in search_widgets:
        if widget.get("name") == "search_box":
            search_box = widget
            break
    
    if not search_box:
        logging.warning("[SEARCH_BANK] Bank search box not found")
        return None
    
    bounds = search_box.get("bounds")
    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        logging.warning("[SEARCH_BANK] Bank search box has invalid bounds")
        return None
    
    # Click the search box first
    step1 = {
        "action": "bank-search-click",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": "Search Box", "bounds": bounds},
        "postconditions": [],
    }
    dispatch(step1)
    
    # Type the search term
    step2 = {
        "action": "bank-search-type",
        "click": {"type": "type", "text": search_term, "enter": False, "per_char_ms": 15, "focus": True}
    }
    return dispatch(step2)

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
    from ..helpers.bank import get_bank_tabs
    
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
    tab = get_bank_tab_by_name(tab_name)
    if not tab:
        logging.warning(f"[CLICK_BANK_TAB] Bank tab '{tab_name}' not found")
        return None
    
    bounds = tab.get("bounds")
    if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
        logging.warning(f"[CLICK_BANK_TAB] Bank tab '{tab_name}' has invalid bounds")
        return None
    
    step = {
        "action": "bank-click-tab",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": f"Tab {tab_name}", "bounds": bounds},
        "postconditions": [],
    }
    return dispatch(step)