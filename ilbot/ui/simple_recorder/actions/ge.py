# ilbot/ui/simple_recorder/actions/ge.py
# Tiny, reusable GE action builders (return single-click/keypress steps).
import logging
import time
from .widgets import get_widget_children_with_text, get_widget_children
from ..helpers.runtime_utils import ui, dispatch, ipc
from ..helpers.rects import rect_center_xy
from ..helpers.utils import closest_object_by_names
from ..helpers.widgets import rect_center_from_widget
from ..helpers.ge import (
    ge_open, ge_offer_open, ge_selected_item_is, ge_first_buy_slot_btn,
    widget_by_id_text_contains, ge_qty_button, ge_qty_matches, chat_qty_prompt_active, find_ge_plus5_bounds,
    widget_by_id_text, ge_buy_confirm_widget
)
from ..helpers.utils import press_enter
from ..actions import travel as trav, bank, inventory as inv
from ..actions.timing import wait_until
from ..constants import BANK_REGIONS

# ---------- simple reads ----------
def is_open() -> bool:
    # Check if GE main widget (465.1) exists and is visible
    ge_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    return bool(ge_data.get("ok", False) and ge_data.get("children"))

def is_closed() -> bool:
    return not is_open()

def get_ge_offer(slot: int) -> dict:
    """
    Get information about a specific GE offer slot.
    
    Args:
        slot: The offer slot number (0-7)
    
    Returns:
        Dict with 'open', 'item_name', and 'status' keys
    """
    # Get the offer slot widget ID (30474247 + slot)
    slot_widget_id = 30474247 + slot
    
    # Get children of the slot widget
    children_data = ipc.get_widget_children(slot_widget_id)
    if not children_data.get("ok"):
        return {"open": True, "item_name": None, "status": "OPEN"}
    
    children = children_data.get("children", [])
    
    # Default status is OPEN
    status = "OPEN"
    
    # Loop through all children widgets to check their text colors
    for child in children:
        text_color = child.get("textColor", "")
        
        # Check for status-indicating text colors
        if text_color == "5f00":
            status = "completed"
            break  # Found completed status, no need to check further
        elif text_color == "d88020":
            status = "in_progress"
            break
            # Don't break here, continue checking for completed status
        elif text_color == "8f0000":
            status = "aborted"
            break
            # Don't break here, continue checking for higher priority statuses
    
    # Try to get item name from the slot
    item_name = None
    for child in children:
        child_text = child.get("text", "").strip()
        
        # Look for item name - it's typically in a child with text and a specific color pattern
        # Based on the data, item names appear to have textColor 'ffb83f' or similar
        if (child_text and 
            child_text not in ["", "Empty", "Buy", "coins"] and
            len(child_text) > 2):  # Item names are typically longer than 2 characters
            item_name = child_text
            break
    
    return {
        "open": status == "OPEN",
        "item_name": item_name,
        "status": status
    }

def get_ge_offers() -> list[dict]:
    """
    Get information about all GE offer slots.
    
    Returns:
        List of dicts, one for each slot (0-7)
    """
    offers = []
    for slot in range(8):
        offers.append(get_ge_offer(slot))
    return offers

def offer_completed(slot: int = None) -> bool:
    """
    Check if a specific offer slot is completed, or if any offer is completed.
    
    Args:
        slot: Specific slot to check (0-7), or None to check all slots
    
    Returns:
        True if the specified slot is completed, or if any slot is completed
    """
    if slot is not None:
        offer_info = get_ge_offer(slot)
        return offer_info["status"] == "completed"
    else:
        offers = get_ge_offers()
        return any(offer["status"] == "completed" for offer in offers)

def offer_in_progress(slot: int = None) -> bool:
    """
    Check if a specific offer slot is in progress, or if any offer is in progress.
    
    Args:
        slot: Specific slot to check (0-7), or None to check all slots
    
    Returns:
        True if the specified slot is in progress, or if any slot is in progress
    """
    if slot is not None:
        offer_info = get_ge_offer(slot)
        return offer_info["status"] == "in_progress"
    else:
        offers = get_ge_offers()
        return any(offer["status"] == "in_progress" for offer in offers)

def offer_aborted(slot: int = None) -> bool:
    """
    Check if a specific offer slot is aborted, or if any offer is aborted.
    
    Args:
        slot: Specific slot to check (0-7), or None to check all slots
    
    Returns:
        True if the specified slot is aborted, or if any slot is aborted
    """
    if slot is not None:
        offer_info = get_ge_offer(slot)
        return offer_info["status"] == "aborted"
    else:
        offers = get_ge_offers()
        return any(offer["status"] == "aborted" for offer in offers)

def offer_open(slot: int = None) -> bool:
    """
    Check if a specific offer slot is open, or if any offer is open.
    
    Args:
        slot: Specific slot to check (0-7), or None to check all slots
    
    Returns:
        True if the specified slot is open, or if any slot is open
    """
    if slot is not None:
        offer_info = get_ge_offer(slot)
        return offer_info["open"]
    else:
        offers = get_ge_offers()
        return any(offer["open"] for offer in offers)

def selected_item_is(name: str) -> bool:
    return ge_selected_item_is(name)

def qty_is(qty: int) -> bool:
    return ge_qty_matches(qty)

def can_confirm() -> bool:
    return bool(widget_by_id_text_contains(30474266, "confirm"))

def has_collect() -> bool:
    return bool(widget_by_id_text_contains(30474246, "collect"))

def buy_chatbox_text_input_contains(substr: str) -> bool:
    """
    Check if the chatbox contains the given substring anywhere in its text.
    
    Args:
        substr: The substring to search for
    
    Returns:
        True if the substring is found in chatbox text, False otherwise
    """
    # Get the chatbox scroll contents widget (162.51)
    chatbox_widget_id = 10616870  # 162.51
    
    # Get child widgets of the chatbox
    children = get_widget_children(chatbox_widget_id).get('children')
    
    if not children:
        return False
    
    # Check all child widgets for the substring
    for child in children:
        child_text = child.get("text", "")
        child_name = child.get("name", "")
        
        # Check both text and name fields
        if child_text and substr.lower() in child_text.lower():
            return True
        if child_name and substr.lower() in child_name.lower():
            return True
    
    return False

def buy_chatbox_first_item_is(name: str) -> bool:
    """
    Check if the first item in the chatbox matches the given name.
    
    Args:
        name: The item name to check for
    
    Returns:
        True if the first item in chatbox matches the name, False otherwise
    """
    # Get the chatbox scroll contents widget (162.51)
    chatbox_widget_id = 10616883  # 162.51
    
    # Get child widgets of the chatbox
    from ..actions.widgets import get_widget_children
    children = get_widget_children(chatbox_widget_id).get('children')
    
    if not children or len(children) < 2:
        return False

    items = buy_chatbox_first_x_items(5)
    
    # Each item has 3 children, so check the second child (index 1) of the first item
    # The item name is in the text field of the second child
    first_item_text = children[0].get("text", "").strip()
    
    # Check if the first item matches the name (case-insensitive)
    return name.lower() == first_item_text.lower()

def buy_chatbox_first_x_items(count: int = 5) -> list:
    """
    Get the first X items from the chatbox.
    
    Args:
        count: Number of items to return (default: 5)
    
    Returns:
        List of item names from the chatbox, up to the specified count
    """
    # Get the chatbox scroll contents widget (162.51)
    chatbox_widget_id = 10616883  # 162.51
    
    # Get child widgets of the chatbox
    from ..actions.widgets import get_widget_children
    children = get_widget_children(chatbox_widget_id).get('children')
    
    if not children:
        return []
    
    items = []
    # Check every child and skip unwanted items
    for child in children:
        if len(items) >= count:
            break
            
        child_text = child.get("text", "").strip()
        if child_text:  # Only process non-empty text
            # Skip items with these patterns
            if (child_text == '' or 
                'previous search' in child_text.lower() or 
                'start typing' in child_text.lower()):
                continue
            
            items.append(child_text)
    
    return items

def buy_chatbox_item_in_first_x(item: str, x: int = 5) -> bool:
    """
    Check if a specific item appears in the first X items of the chatbox.
    
    Args:
        item: The item name to search for
        x: Number of items to check from the beginning (default: 5)
    
    Returns:
        True if the item is found in the first X items, False otherwise
    """
    # Get the first X items from the chatbox
    items = buy_chatbox_first_x_items(x)
    
    # Check if the target item is in the list (case-insensitive)
    item_lower = item.lower()
    for chatbox_item in items:
        if item_lower == chatbox_item.lower():
            return True
    
    return False

def click_chatbox_item_by_name(item: str, x: int = 5) -> dict | None:
    """
    Click on a specific item in the chatbox by searching for it in the first X items.
    
    Args:
        item: The item name to click on
        x: Number of items to search through (default: 5)
    
    Returns:
        Result of the click action, or None if item not found
    """
    # Get the chatbox scroll contents widget (162.51)
    chatbox_widget_id = 10616883  # 162.51
    
    # Get child widgets of the chatbox
    from ..actions.widgets import get_widget_children
    children = get_widget_children(chatbox_widget_id).get('children')
    
    if not children:
        return None
    
    # Search for the item by checking each child's text
    item_lower = item.lower()
    for i, child in enumerate(children):
        child_text = child.get("text", "").strip()
        if child_text and item_lower == child_text.lower():
            # Found the item, get its bounds and click
            bounds = child.get("bounds", {})
            if bounds:
                cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
                cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
                step = {
                    "id": "ge-click-chatbox-item",
                    "action": "click",
                    "description": f"Click item: {item}",
                    "click": {"type": "point", "x": cx, "y": cy},
                    "target": {"domain": "widget", "name": item, "bounds": bounds},
                    "preconditions": [], "postconditions": []
                }
                return dispatch(step)
    
    return None

# ---------- one-step actions ----------
def open_ge() -> dict | None:
    npc_data = ipc.get_npcs()

    # --- Prefer the Clerk; use live context menu to pick "Exchange" ---
    npc = None
    for cand in (npc_data.get("closestNPCs") or []) + (npc_data.get("npcs") or []):
        nm = (cand.get("name") or "").lower()
        nid = int(cand.get("id") or -1)
        if "grand exchange clerk" in nm or (2148 <= nid <= 2151):
            npc = cand
            break

    if npc and isinstance(npc['canvas'].get("x"), (int, float)) and isinstance(npc['canvas'].get("y"), (int, float)):
        cx = npc['canvas'].get("x")
        cy = npc['canvas'].get("y") - 8  # keep your slight lift
        step = {
            "id": "ge-open-clerk",
            "action": "open-ge-context",
            "description": "Open GE (clerk via context menu: Exchange)",
            "option": "exchange",  # case-insensitive
            "click": {
                "type": "context-select",   # uses IPC 'menu' to match rows by text + rect
                "target": "grand exchange clerk",  # substring match against target text
                "x": cx,                    # right-click anchor (canvas)
                "y": cy,
                "open_delay_ms": 120
            },
            "target": {"domain": "npc", "name": npc.get("name"), "id": npc.get("id")},
        }
        return dispatch(step)

    # --- Fallback: booth simple left-click (unchanged) ---
    booth = closest_object_by_names(["grand exchange booth"])
    if booth and isinstance(booth.get("canvasX"), (int, float)) and isinstance(booth.get("canvasY"), (int, float)):
        step = {
            "id": "ge-open-booth",
            "action": "click",
            "description": "Open GE (booth)",
            "click": {"type": "point", "x": int(booth["canvasX"]), "y": int(booth["canvasY"]) - 12},
            "target": {"domain": "object", "name": booth.get("name"), "id": booth.get("id")},
        }
        return dispatch(step)

    return None

def close_ge() -> dict | None:
    step = {
        "id": "ge-close",
        "action": "click",
        "description": "Close GE",
        "click": {"type": "key", "key": "escape"},
        "preconditions": [], "postconditions": []
    }
    return dispatch(step)

def get_offer_slots() -> dict:
    """
    Get GE offer slots organized by index (0-7).
    Returns a dictionary with keys 'INDEX_0' through 'INDEX_7' containing lists of widgets.
    """
    
    # Get all GE widgets
    offers_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not offers_data.get("ok"):
        return {}
    
    children = offers_data.get("children", [])
    
    # Initialize the result dictionary
    result = {}
    for i in range(8):
        result[f'INDEX_{i}'] = []
    
    # Group widgets by their parent index widget ID
    # Index widget IDs: 30474247 (index 0), 30474248 (index 1), etc.
    for child in children:
        widget_id = child.get("id", 0)
        
        # Check if this widget belongs to an index container
        # Index containers are 30474247-30474254 (INDEX_0 to INDEX_7)
        if 30474247 <= widget_id <= 30474254:
            index_num = widget_id - 30474247
            result[f'INDEX_{index_num}'].append(child)
    
    return result

def begin_buy_offer() -> dict | None:
    # Get offer slots organized by index
    offer_slots = get_offer_slots()
    
    # Find the first available offer slot (empty slot) across all indexes
    for index_name in ['INDEX_0', 'INDEX_1', 'INDEX_2', 'INDEX_3', 'INDEX_4', 'INDEX_5', 'INDEX_6', 'INDEX_7']:
        widgets = offer_slots.get(index_name, [])
        for widget in widgets:
            # Check if this is an empty slot (no item sprite or "Empty" text)
            sprite_id = widget.get("spriteId", -1)
            text = (widget.get("text") or "").strip()
            
            if sprite_id == 1108:  # Empty slot indicators
                bounds = widget.get("bounds", {})
                if bounds:
                    cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
                    cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
                    step = {
        "id": "ge-begin-buy",
        "action": "click",
        "description": "Open buy offer",
        "click": {"type": "point", "x": cx, "y": cy},
                        "target": {"domain": "widget", "name": "ge_offer_slot", "bounds": bounds},
        "preconditions": [], "postconditions": []
                    }
                    return dispatch(step)
    return None


def type_item_name(name: str) -> dict | None:
    if selected_item_is(name):
        return None

    step = {
        "id": "ge-type-item",
        "action": "type",
        "description": f"Type item: {name}",
        "click": {"type": "type", "text": name, "per_char_ms": 20},
    }
    return dispatch(step)


def set_quantity(qty: int = 0) -> dict | None:
    if qty_is(qty):
        return None

    if not chat_qty_prompt_active():
        dot = ge_qty_button()
        if dot:
            cx, cy = rect_center_from_widget(dot)
            step = {
                "id": "ge-qty-dot",
                "action": "click",
                "description": "Open quantity prompt",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {"domain": "widget", "name": "qty_button", "bounds": dot.get("bounds")},
                "preconditions": [], "postconditions": []
            }
            return dispatch(step)
        return None

    if not buy_chatbox_text_input_contains(str(qty)):
        step = {
            "id": "ge-qty-type",
            "action": "type",
            "description": f"Type buy quantity: {qty}",
            "click": {"type": "type", "text": str(qty), "per_char_ms": 20},
            "preconditions": [], "postconditions": []
        }
        return dispatch(step)

    else:
        return press_enter()


def click_plus5() -> dict | None:
    # Look for +5% button in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").strip().replace(" ", "")
        if text.lower() == "+5%":
            bounds = child.get("bounds", {})
            if bounds:
                cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
                cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
                step = {
        "id": "ge-plus5",
        "action": "click",
        "description": "+5% price",
        "click": {"type": "point", "x": cx, "y": cy},
                    "target": {"domain": "widget", "name": "+5%", "bounds": bounds},
        "preconditions": [], "postconditions": []
                }
                return dispatch(step)
    return None

def confirm_buy() -> dict | None:
    # Look for confirm button in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None

    children = setup_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").lower()
        if "confirm" in text:
            bounds = child.get("bounds", {})
            if bounds:
                cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
                cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
                step = {
        "id": "ge-confirm",
        "action": "click",
        "description": "Confirm buy offer",
        "click": {"type": "point", "x": cx, "y": cy},
                    "target": {"domain": "widget", "name": "confirm", "bounds": bounds},
        "preconditions": [], "postconditions": []
                }
                return dispatch(step)
    return None


def collect_to_inventory() -> dict | None:
    # Look for collect button in main GE container
    ge_data = ipc.get_widget_children(30474241)  # 465.1 GeOffers.CONTENTS
    if not ge_data.get("ok"):
        return None

    children = ge_data.get("children", [])
    for child in children:
        text = (child.get("text") or "").lower()
        if "collect" in text:
            bounds = child.get("bounds", {})
            if bounds:
                cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
                cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
                step = {
        "id": "ge-collect",
        "action": "click",
        "description": "Collect items to inventory",
        "click": {"type": "point", "x": cx, "y": cy},
                    "target": {"domain": "widget", "name": "collect", "bounds": bounds},
        "preconditions": [], "postconditions": []
                }
                return dispatch(step)
    return None

def find_ge_offer_by_item(item_name: str) -> dict | None:
    """
    Find the GE offer slot containing the specified item.
    
    Args:
        item_name: Name of the item to find
    
    Returns:
        Widget data for the offer slot, or None if not found
    """
    # Get all GE offer slots
    offer_slots = get_offer_slots()
    
    # Search through all offer slots
    for index_name in ['INDEX_0', 'INDEX_1', 'INDEX_2', 'INDEX_3', 'INDEX_4', 'INDEX_5', 'INDEX_6', 'INDEX_7']:
        widgets = offer_slots.get(index_name, [])
        for widget in widgets:
            # Check if this widget contains the item name
            widget_text = (widget.get("text") or "").strip()
            widget_name = (widget.get("name") or "").strip()
            
            if (item_name.lower() in widget_text.lower() or 
                item_name.lower() in widget_name.lower()):
                return widget
    
    return None

def abort_ge_offer(item_name: str) -> dict | None:
    """
    Abort a GE offer by right-clicking it and selecting 'Abort offer'.
    
    Args:
        item_name: Name of the item whose offer to abort
    
    Returns:
        Result of the abort action, or None if failed
    """
    # Find the offer slot containing the item
    offer_widget = find_ge_offer_by_item(item_name)
    if not offer_widget:
        return None
    
    bounds = offer_widget.get("bounds", {})
    if not bounds:
        return None
    
    cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
    cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    step = {
        "id": "ge-abort-offer",
        "action": "click-ground-context",
        "description": f"Abort offer for {item_name}",
        "click": {
            "type": "context-select",
            "index": 0,  # "Abort offer" is typically the first option
            "x": cx,
            "y": cy,
            "row_height": 16,
            "start_dy": 10,
            "open_delay_ms": 120
        },
        "option": "Abort offer",
        "target": {"domain": "widget", "name": "", "bounds": bounds},
        "anchor": {"x": cx, "y": cy},
    }
    return dispatch(step)

def modify_ge_offer(item_name: str) -> dict | None:
    """
    Modify a GE offer by right-clicking it and selecting 'Modify offer'.
    
    Args:
        item_name: Name of the item whose offer to modify
    
    Returns:
        Result of the modify action, or None if failed
    """
    # Find the offer slot containing the item
    offer_widget = find_ge_offer_by_item(item_name)
    if not offer_widget:
        return None
    
    bounds = offer_widget.get("bounds", {})
    if not bounds:
        return None
    
    cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
    cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    step = {
        "id": "ge-modify-offer",
        "action": "click-ground-context",
        "description": f"Modify offer for {item_name}",
        "click": {
            "type": "context-select",
            "index": 1,  # "Modify offer" is typically the second option
            "x": cx,
            "y": cy,
            "row_height": 16,
            "start_dy": 10,
            "open_delay_ms": 120
        },
        "option": "Modify offer",
        "target": {"domain": "widget", "name": "", "bounds": bounds},
        "anchor": {"x": cx, "y": cy},
    }
    return dispatch(step)

def set_custom_price(price: int) -> dict | None:
    """
    Set a custom price by clicking 'Enter price' button and typing the price.
    
    Args:
        price: The price to set
    
    Returns:
        Result of the price setting action, or None if failed
    """
    # Look for 'Enter price' button in setup widget children
    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
    if not setup_data.get("ok"):
        return None
    
    children = setup_data.get("children", [])
    
    # Find all widgets with "..." text and select the rightmost one (for price)
    price_widget = None
    rightmost_x = -1
    
    for child in children:
        text = (child.get("text") or "").strip().lower()
        if "..." in text:
            bounds = child.get("bounds", {})
            if bounds:
                x_pos = bounds.get("x", 0)
                # Keep track of the rightmost widget (highest x position)
                if x_pos > rightmost_x:
                    rightmost_x = x_pos
                    price_widget = child
    
    if price_widget:
        bounds = price_widget.get("bounds", {})
        cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
        cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
        step = {
            "id": "ge-enter-price",
            "action": "click",
            "description": "Click Enter price button",
            "click": {"type": "point", "x": cx, "y": cy},
            "target": {"domain": "widget", "name": "Enter price", "bounds": bounds},
            "preconditions": [], "postconditions": []
        }
        dispatch(step)
        if not wait_until(chat_qty_prompt_active):
            return None

        time.sleep(0.2)
        # Type the price
        price_step = {
            "id": "ge-type-price",
            "action": "type",
            "description": f"Type price: {price}",
            "click": {"type": "type", "text": str(price), "per_char_ms": 20},
        }
        dispatch(price_step)
        if not wait_until(lambda:buy_chatbox_text_input_contains(str(price))):
            return None
        press_enter()
        if not wait_until(lambda: not chat_qty_prompt_active()):
            return None
        return True
    
    return None


def buy_item_from_ge(item, retry_items: dict = None) -> bool | None:
    """
    Returns True when all items are in inventory.
    Otherwise performs exactly one small step toward buying them and returns None.
    Safe to call every tick.
    
    Args:
        item: Name of the item to buy (str) or dict of items to buy {item_name: (quantity, price_bumps, set_price)}
        retry_items: Dict of items that failed first attempt and need retry with set price
        ui: UI instance for dispatching actions
    """
    
    # Normalize input to dict
    if isinstance(item, str):
        items_to_buy = {item: (1, 5, 100)}  # (quantity, price_bumps, set_price)
    else:
        items_to_buy = item
    
    # Check if we already have all items
    if all(inv.has_item(item_name) for item_name in items_to_buy.keys()):
        close_ge()
        if is_closed():
            return True
        return None

    # Go to GE
    if not trav.in_area(BANK_REGIONS["GE"]):
        trav.go_to("GE")
        return None



    # Ensure coins: if we don't have any, grab from bank quickly (GE has a bank close by)
    if not inv.has_item("coins"):
        if bank.is_closed():
            bank.open_bank()
            wait_until(bank.is_open, max_wait_ms=6000)
            return None
        if bank.is_open():
            # make space then withdraw all coins
            if not inv.is_empty():
                bank.deposit_inventory()
                wait_until(inv.is_empty, max_wait_ms=4000, min_wait_ms=150)
                return None
            bank.withdraw_item("coins", withdraw_all=True)
            wait_until(lambda: inv.has_item("coins"), max_wait_ms=3000)
            return None
    elif bank.is_open():
        bank.close_bank()
        return None

    # Open GE
    if is_closed():
        open_ge()
        wait_until(is_open, max_wait_ms=5000)
        return None

    # Buy each item one by one
    items_list = list(items_to_buy.items())
    items_to_process = [(name, data) for name, data in items_list if not inv.has_item(name)]
    
    for i, (item_name, (item_quantity, item_price_bumps, item_set_price)) in enumerate(items_to_process):
        is_last_item = (i == len(items_to_process) - 1)
            
        # Open buy offer panel
        if not is_offer_screen_open():
            if has_collect():
                collect_to_inventory()
                return None
            begin_buy_offer()
            if not wait_until(offer_open, max_wait_ms=5000):
                return None
            return None

        # Type item name and select first result
        if not buy_chatbox_item_in_first_x(item_name, 5) and not selected_item_is(item_name):
            type_item_name(item_name)
            if not wait_until(lambda: buy_chatbox_first_item_is(item_name), min_wait_ms=600, max_wait_ms=3000):
                return None
            return None

        elif not buy_chatbox_item_in_first_x(item_name, 1) and not selected_item_is(item_name):
            # Search through the first 5 items to find and click the correct one
            result = click_chatbox_item_by_name(item_name, 5)
            if result:
                if not wait_until(lambda: selected_item_is(item_name)):
                    return None

        elif buy_chatbox_item_in_first_x(item_name, 1) and not selected_item_is(item_name):
            press_enter()
            if not wait_until(lambda: selected_item_is(item_name), min_wait_ms=600, max_wait_ms=3000):
                return None

        if selected_item_is(item_name):
            # Set quantity if needed
            if item_quantity > 1:
                if not qty_is(item_quantity):
                    set_quantity(item_quantity)
                    return None

            # Set price based on item_set_price from tuple
            if item_set_price > 0:
                # Use the set_price from the item tuple
                result = set_custom_price(item_set_price)
                if not result:
                    return None
                time.sleep(0.5)  # Wait for price to be set
            else:
                # Use +5% bumps when set_price is 0
                for _ in range(max(0, int(item_price_bumps))):
                    # Look for +5% button in setup widget children
                    setup_data = ipc.get_widget_children(30474266)  # 465.26 GeOffers.SETUP
                    if not setup_data.get("ok"):
                        break

                    children = setup_data.get("children", [])
                    plus = None
                    for child in children:
                        text = (child.get("text") or "").strip().replace(" ", "")
                        if text.lower() == "+5%":
                            plus = child

                        if not plus:
                            break
                        cx, cy = rect_center_from_widget(plus)
                        dispatch({
                            "id": "ge-plus5",
                            "action": "click",
                            "description": "+5% price",
                            "target": {"name": "+5%", "bounds": plus.get("bounds")},
                            "click": {"type": "point", "x": cx, "y": cy},
                        })
                        time.sleep(0.25)

            # Confirm buy
            confirm_buy()
            wait_until(lambda: ge_buy_confirm_widget() is None or popup_exists(), min_wait_ms=600, max_wait_ms=3000)

            if popup_exists():
                select_yes_popup()
            wait_until(lambda: ge_buy_confirm_widget() is None, min_wait_ms=600, max_wait_ms=3000)

            # Wait for collect button to appear (offer filled)
            if wait_until(has_collect, max_wait_ms=10000):
                collect_to_inventory()
                # Only return None if this is not the last item
                if not is_last_item:
                    return None

    # Close GE after all items are bought
    close_ge()
    return True


# ===== GE OFFER SCREEN ACTIONS =====

def create_buy_offer(item_name: str, quantity: int = 1, price_bumps: int = 0) -> dict | None:
    """
    Create a buy offer for an item.
    
    Args:
        item_name: Name of the item to buy
        quantity: Quantity to buy (default: 1)
        price_bumps: Number of +5% price bumps to apply (default: 0)
    
    Returns:
        Result of the offer creation, or None if failed
    """
    try:
        # Check if GE is open
        if not is_open():
            logging.error("[create_buy_offer] actions/ge.py: GE is not open")
            return None
        
        # Check if offer screen is open
        if not ge_offer_open():
            logging.error("[create_buy_offer] actions/ge.py: Offer screen is not open")
            return None
        
        # Click item search to select item
        result = click_item_search()
        if not result:
            logging.error("[create_buy_offer] actions/ge.py: Failed to click item search")
            return None
        
        # Wait for item selection
        if not wait_until(lambda: selected_item_is(item_name), max_wait_ms=5000):
            logging.error(f"[create_buy_offer] actions/ge.py: Item '{item_name}' not selected within timeout")
            return None
        
        # Set quantity if needed
        if quantity > 1:
            result = set_quantity(quantity)
            if not result:
                logging.error(f"[create_buy_offer] actions/ge.py: Failed to set quantity to {quantity}")
                return None
        
        # Apply price bumps
        for _ in range(price_bumps):
            result = click_plus5()
            if not result:
                logging.error("[create_buy_offer] actions/ge.py: Failed to apply price bump")
                return None
            time.sleep(0.25)  # Small delay between price bumps
        
        # Confirm the offer
        result = confirm_offer()
        if not result:
            logging.error("[create_buy_offer] actions/ge.py: Failed to confirm offer")
            return None
        
        logging.info(f"[create_buy_offer] actions/ge.py: Successfully created buy offer for {quantity}x {item_name}")
        return result
        
    except Exception as e:
        logging.error(f"[create_buy_offer] actions/ge.py: {e}")
        return None


def create_sell_offer(item_name: str, quantity: int = 1, price_bumps: int = 0) -> dict | None:
    """
    Create a sell offer for an item.
    
    Args:
        item_name: Name of the item to sell
        quantity: Quantity to sell (default: 1)
        price_bumps: Number of +5% price bumps to apply (default: 0)
    
    Returns:
        Result of the offer creation, or None if failed
    """
    try:
        # Check if GE is open
        if not is_open():
            logging.error("[create_sell_offer] actions/ge.py: GE is not open")
            return None
        
        # Check if offer screen is open
        if not ge_offer_open():
            logging.error("[create_sell_offer] actions/ge.py: Offer screen is not open")
            return None
        
        # Click item search to select item
        result = click_item_search()
        if not result:
            logging.error("[create_sell_offer] actions/ge.py: Failed to click item search")
            return None
        
        # Wait for item selection
        if not wait_until(lambda: selected_item_is(item_name), max_wait_ms=5000):
            logging.error(f"[create_sell_offer] actions/ge.py: Item '{item_name}' not selected within timeout")
            return None
        
        # Set quantity if needed
        if quantity > 1:
            result = set_quantity(quantity)
            if not result:
                logging.error(f"[create_sell_offer] actions/ge.py: Failed to set quantity to {quantity}")
                return None
        
        # Apply price bumps
        for _ in range(price_bumps):
            result = click_plus5()
            if not result:
                logging.error("[create_sell_offer] actions/ge.py: Failed to apply price bump")
                return None
            time.sleep(0.25)  # Small delay between price bumps
        
        # Confirm the offer
        result = confirm_offer()
        if not result:
            logging.error("[create_sell_offer] actions/ge.py: Failed to confirm offer")
            return None
        
        logging.info(f"[create_sell_offer] actions/ge.py: Successfully created sell offer for {quantity}x {item_name}")
        return result
        
    except Exception as e:
        logging.error(f"[create_sell_offer] actions/ge.py: {e}")
        return None


def get_offer_screen_status() -> dict:
    """
    Get the current status of the offer screen.
    
    Returns:
        Dictionary with status information
    """
    try:
        status = {
            "ge_open": is_open(),
            "offer_screen_open": ge_offer_open(),
            "selected_item": None,
            "current_quantity": None,
            "current_price": None,
            "can_confirm": can_confirm()
        }
        
        # Get selected item if offer screen is open
        if status["offer_screen_open"]:
            # This would need to be implemented to read the actual selected item
            # For now, just set to None
            status["selected_item"] = None
        
        return status
        
    except Exception as e:
        logging.error(f"[get_offer_screen_status] actions/ge.py: {e}")
        return {
            "ge_open": False,
            "offer_screen_open": False,
            "selected_item": None,
            "current_quantity": None,
            "current_price": None,
            "can_confirm": False
        }


def popup_exists() -> bool:
    """
    Check if the GE popup overlay exists.
    
    Returns:
        True if the popup overlay widget exists and is visible, False otherwise
    """
    from .widgets import widget_exists
    return widget_exists(18939907)  # Popupoverlay.WINDOW

def select_yes_popup() -> dict | None:
    """
    Click the 'Yes' button on the GE popup overlay and wait for it to disappear.
    
    Returns:
        Result of the click action, or None if failed
    """
    from .widgets import get_widget_info
    from .timing import wait_until
    
    # Get the Yes button widget info
    button_info = get_widget_info(18939912)  # Popupoverlay.BUTTON_1
    
    button_data = button_info.get("data", {})
    bounds = button_data.get("bounds", {})
    if not bounds:
        return None
    
    # Click the Yes button
    cx = bounds.get("x", 0) + bounds.get("width", 0) // 2
    cy = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    step = {
        "id": "ge-select-yes-popup",
        "action": "click",
        "description": "Click Yes on GE popup",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "widget", "name": "Yes button", "bounds": bounds},
        "preconditions": [], "postconditions": []
    }
    
    result = dispatch(step)
    if result:
        # Wait until the popup no longer exists
        if not wait_until(lambda: not popup_exists(), max_wait_ms=5000):
            return None
        return result
    
    return None

def click_item_search() -> dict | None:
    """Click the item search icon to open item selection."""
    from ..helpers.ge import click_item_search as helper_click_item_search
    return helper_click_item_search()


def adjust_quantity(direction: str) -> dict | None:
    """Adjust quantity using arrow buttons."""
    from ..helpers.ge import adjust_quantity as helper_adjust_quantity
    return helper_adjust_quantity(direction)


def set_price_percentage(percentage: int) -> dict | None:
    """Set price using percentage buttons."""
    from ..helpers.ge import set_price_percentage as helper_set_price_percentage
    return helper_set_price_percentage(percentage)


def adjust_price(direction: str) -> dict | None:
    """Adjust price using arrow buttons."""
    from ..helpers.ge import adjust_price as helper_adjust_price
    return helper_adjust_price(direction)


def set_market_price() -> dict | None:
    """Set price to current market price."""
    from ..helpers.ge import set_market_price as helper_set_market_price
    return helper_set_market_price()


def confirm_offer() -> dict | None:
    """Confirm and place the offer."""
    from ..helpers.ge import confirm_offer as helper_confirm_offer
    return helper_confirm_offer()


def close_offer_screen() -> dict | None:
    """Close the GE offer setup screen."""
    from ..helpers.ge import close_offer_screen as helper_close_offer_screen
    return helper_close_offer_screen()


def open_history() -> dict | None:
    """Open the GE history screen."""
    from ..helpers.ge import open_history as helper_open_history
    return helper_open_history()


def is_offer_screen_open() -> bool:
    """Check if the GE offer setup screen is open."""
    from ..helpers.ge import is_offer_screen_open as helper_is_offer_screen_open
    return helper_is_offer_screen_open()


def get_offer_screen_data() -> dict | None:
    """Get all offer screen widget data for analysis."""
    from ..helpers.ge import get_offer_screen_data as helper_get_offer_screen_data
    return helper_get_offer_screen_data()


def is_offer_ready() -> bool:
    """Check if the offer is ready to be confirmed (has item, quantity, and price)."""
    from ..helpers.ge import is_offer_ready as helper_is_offer_ready
    return helper_is_offer_ready()


def get_current_quantity() -> str | None:
    """Get the current quantity value from the offer screen."""
    from ..helpers.ge import get_current_quantity as helper_get_current_quantity
    return helper_get_current_quantity()


def get_current_price() -> str | None:
    """Get the current price value from the offer screen."""
    from ..helpers.ge import get_current_price as helper_get_current_price
    return helper_get_current_price()


def check_and_buy_required_items(required_items: list, item_requirements: dict, plan_id: str = "GE") -> dict | None:
    """
    Check if we have all required items and buy missing ones from GE.
    
    Args:
        required_items: List of item names to check for
        item_requirements: Dict mapping item names to (quantity, five_percent_pushes, set_price) tuples
                          Special keys: "unnoted_items" - list of items that must be unnoted
        plan_id: Plan identifier for logging
    
    Returns:
        dict with status information:
        - "status": "complete", "buying", "error", or "no_items_needed"
        - "items_to_buy": dict of items still needed (if status is "buying")
        - "error": error message (if status is "error")
    """
    try:
        # Check if we have all required items with correct quantities and types
        has_all_items = True
        unnoted_items = item_requirements.get("unnoted_items", [])
        
        for item in required_items:
            if item in unnoted_items:
                # Need unnoted version
                if not inv.has_unnoted_item(item):
                    has_all_items = False
                    break
            else:
                # Check quantity requirement
                required_qty = item_requirements[item][0] # Get quantity from tuple
                if not inv.has_item(item, min_qty=required_qty):
                    has_all_items = False
                    break
        
        if has_all_items:
            logging.info(f"[{plan_id}] Already have all required items")
            return {"status": "complete"}
        
        # Open bank and deposit all inventory first
        if bank.is_closed() and not inv.inv_has("coins"):
            bank.open_bank()
            return {"status": "buying", "items_to_buy": {}}  # Still working on banking
        
        if inv.inv_has("coins") and bank.is_open():
            bank.close_bank()
            return {"status": "buying", "items_to_buy": {}}  # Still working on banking
        
        # Deposit all inventory items to get accurate counts
        if bank.is_open():
            bank.deposit_inventory()
            time.sleep(0.5)  # Brief wait for deposit to complete
        
        # Calculate what we need to buy
        items_to_buy = {}
        
        for item in required_items:
            if item in unnoted_items:
                # Check for unnoted version
                item_count = bank.get_item_count(item)
                if item_count == 0:
                    required_qty, five_percent_pushes, set_price = item_requirements.get(item, (1, 5, 100))
                    items_to_buy[item] = (required_qty, five_percent_pushes, set_price)
            else:
                # Check quantity requirement
                required_qty, five_percent_pushes, set_price = item_requirements.get(item, (1, 5, 100))
                current_count = bank.get_item_count(item)
                needed_count = max(0, required_qty - current_count)
                if needed_count > 0:
                    items_to_buy[item] = (needed_count, five_percent_pushes, set_price)
        
        # If we don't need to buy anything, we're done
        if not items_to_buy:
            logging.info(f"[{plan_id}] Already have all required items")
            return {"status": "complete"}
        
        logging.info(f"[{plan_id}] Calculated items to buy: {items_to_buy}")
        
        # Use the centralized GE buying method
        logging.info(f"[{plan_id}] Buying required items from GE...")
        result = buy_item_from_ge(items_to_buy)
        
        if result is None:
            # Still working on buying items
            return {"status": "buying", "items_to_buy": items_to_buy}
        elif result is True:
            logging.info(f"[{plan_id}] Successfully bought all required items")
            return {"status": "complete"}
        else:
            logging.info(f"[{plan_id}] Failed to buy items, retrying...")
            return {"status": "buying", "items_to_buy": items_to_buy}
            
    except Exception as e:
        logging.error(f"[{plan_id}] Error in check_and_buy_required_items: {e}")
        return {"status": "error", "error": str(e)}