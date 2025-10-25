import time

from . import tab
from .tab import open_tab
from .timing import wait_until
from ..helpers.inventory import inv_has, inv_has_any, get_item_coordinates, inv_count, first_inv_slot
from ..helpers.runtime_utils import ipc, ui, dispatch
from ..helpers.tab import is_inventory_tab_open
from ..helpers.utils import sleep_exponential, rect_beta_xy, clean_rs
from ..helpers.rects import unwrap_rect, rect_center_xy

from typing import Optional, Callable, List

def has_item(name: str, min_qty: int = 1) -> bool:
    """
    True if inventory contains item `name` with quantity >= min_qty.
    """
    return inv_has(name, min_qty)

def has_items(items: dict, noted: bool = False) -> bool:
    """
    Check whether the inventory contains all specified items in the required quantities.

    Args:
        items (dict or set or list or tuple): Mapping of item name -> required quantity,
                                              or iterable of item names for ANY quantity.
        noted (bool): If True, check for noted items; otherwise check for unnoted items.

    Returns:
        bool: True if all items are present in the required quantities, False otherwise.
    """
    if not items:
        print("[INV] inventory_has_items: Invalid items provided")
        return False

    # Allow set/list/tuple as "require ANY amount of each item"
    if isinstance(items, (set, list, tuple)):
        items = {name: "any" for name in items}

    if not isinstance(items, dict):
        print("[INV] inventory_has_items: Invalid items type")
        return False

    for item, qty in items.items():
        # allow "any" or None as "at least 1 present"
        if isinstance(qty, str) and qty.lower() == "any":
            needed_any = True
        elif qty is None:
            needed_any = True
        else:
            needed_any = False

        if not isinstance(item, str):
            print(f"[INV] Skipping invalid entry: {item} -> {qty}")
            continue

        if noted:
            count = inv_count(item)
        else:
            count = count_unnoted_item(item)

        if needed_any:
            if count <= 0:
                print(f"[INV] Missing {item}: need ANY, have 0")
                return False
        else:
            if not isinstance(qty, int) or qty <= 0:
                print(f"[INV] Skipping invalid qty entry: {item} -> {qty}")
                continue
            if count < qty:
                print(f"[INV] Missing {item}: need {qty}, have {count}")
                return False

    return True

def has_any_items(item_names: List[str], min_qty: int = 1) -> bool:
    """
    Check if inventory contains ANY item from the list with quantity >= min_qty.
    
    Args:
        item_names: List of item names to check for
        min_qty: Minimum quantity required for the item (default: 1)
        
    Returns:
        True if ANY item is found with sufficient quantity, False otherwise
    """
    if not item_names:
        return False  # Empty list means no items to find
    
    # Check each item individually - return True as soon as we find one
    for item_name in item_names:
        if has_item(item_name, min_qty):
            return True
    
    return False


def has_noted_item(name: str) -> bool:
    """
    True if inventory contains noted version of item `name`.
    """
    resp = ipc.get_inventory()
    
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    target_name = (name or "").strip().lower()
    
    for slot in slots:
        item_name = (slot.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Check for noted items
        if target_name in item_name and slot.get("noted"):
            return True
    
    return False

def count_unnoted_item(name: str) -> int:
    """
    Count the total quantity of unnoted version of item `name`.
    This counts the normal item, not the noted version.
    """
    resp = ipc.get_inventory()
    
    if not resp or not resp.get("ok"):
        return 0
    
    slots = resp.get("slots", [])
    target_name = (name or "").strip().lower()
    total_count = 0
    
    for slot in slots:
        item_name = (slot.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Exact match and not noted
        if item_name == target_name and not slot.get("noted"):
            quantity = int(slot.get("quantity") or 0)
            total_count += quantity
    
    return total_count


def has_unnoted_item(name: str, min_qty: int = 1) -> bool:
    """
    True if inventory contains at least min_qty unnoted version of item `name`.
    This is the normal item, not the noted version.
    """
    return count_unnoted_item(name) >= min_qty



def is_empty(excepted_items: list[str] = None) -> bool:
    """
    True if the player's inventory is empty.
    """
    return not inv_has_any(excepted_items=excepted_items)


def get_empty_slots_count() -> int:
    """
    Returns the number of empty inventory slots.
    
    Returns:
        Number of empty slots (0-28, where 28 is completely empty)
    """
    try:
        # Use IPC command to get inventory data
        resp = ipc.get_inventory()
        
        if not resp or not resp.get("ok"):
            print(f"[INVENTORY] Failed to get inventory data: {resp.get('err', 'Unknown error')}")
            return 28  # Assume all slots empty if can't get data
        
        slots = resp.get("slots", [])
        
        # Count empty slots
        empty_count = 0
        for slot in slots:
            # Check if slot is empty
            item_name = slot.get("itemName", "").strip()
            quantity = int(slot.get("quantity", 0))
            
            # Slot is empty if no item name or quantity is 0
            if not item_name or quantity <= 0:
                empty_count += 1
        
        return empty_count
        
    except Exception as e:
        print(f"[INVENTORY] Error counting empty slots: {e}")
        # Fallback: assume all slots are empty if we can't read the data
        return 28

def use_item_on_item(item1_name: str, item2_name: str, max_retries: int = 3) -> Optional[dict]:
    """
    Use an item in the inventory on another item in the inventory.
    
    Args:
        item1_name: Name of the item to use (the "using" item)
        item2_name: Name of the item to use on (the "target" item)
        max_retries: Maximum number of retry attempts if interaction fails
    
    Returns:
        UI dispatch result or None if failed
    """
    # Check if both items exist in inventory
    if not has_item(item1_name):
        return None
    if not has_item(item2_name):
        return None

    result1 = interact(item1_name, "Use")

    if not result1:
        return None

    sleep_exponential(0.3, 0.8)

    from ilbot.ui.simple_recorder.actions import objects
    result2 = interact(item2_name, "Use", exact_match=False)

    return result2

def use_item_on_object(item_name: str, object_name: str, max_retries: int = 3) -> Optional[dict]:
    """
    Use an item in the inventory on a game object.
    
    Args:
        item_name: Name of the item to use from inventory
        object_name: Name of the game object to use the item on
        max_retries: Maximum number of retry attempts if interaction fails
    
    Returns:
        UI dispatch result or None if failed
    """
    # Check if item exists in inventory
    if not has_item(item_name):
        return None

    result1 = interact(item_name, "Use")

    if not result1:
        return None

    sleep_exponential(0.3, 0.8)

    from ilbot.ui.simple_recorder.actions import objects
    result2 = objects.click_object_closest_by_distance_simple(object_name, f"Use")
    
    return result2


def inventory_has_amount(item_name: str, expected_amount: int) -> bool:
    """
    Helper function for use with wait_until to check if inventory contains 
    the expected amount of an item.
    
    Args:
        item_name: Name of the item to check
        expected_amount: Expected quantity in inventory
    
    Returns:
        True if inventory contains at least the expected amount
    """
    current_count = inv_count(item_name)
    return current_count >= expected_amount


def interact(item_name: str, menu_option: str, exact_match: bool = False) -> Optional[dict]:
    """
    Context-click an inventory item and select a specific menu option.
    
    Args:
        item_name: Name of the item to interact with
        menu_option: Menu option to select (e.g., "Use", "Drop", "Examine")
        exact_match: If True, only matches exact target and action names; if False, uses substring matching
    
    Returns:
        UI dispatch result or None if failed
    """
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Opening inventory tab before interacting with {item_name}")
        open_tab("INVENTORY")
        # Wait a moment for the tab to open
        sleep_exponential(0.1, 0.3, 1.5)
    
    # Ensure we're still on inventory tab before proceeding
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Failed to open inventory tab, aborting interaction")
        return None
    
    # Inner attempt loop with fresh coordinate recalculation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Fresh coordinate recalculation
        item = first_inv_slot(item_name)
        if not item:
            continue
        
        bounds = item.get("bounds")
        if not bounds:
            continue
        
        # Calculate center coordinates
        x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                             bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        
        # Context-click the item
        step = {
            "action": "inventory-interact",
            "click": {"type": "point", "x": int(x), "y": int(y)},
            "target": {"domain": "inventory", "name": item_name, "menu_option": menu_option},
        }
        
        result = dispatch(step)
        
        if result:
            # Check if the correct interaction was performed
            from ..helpers.ipc import get_last_interaction
            last_interaction = get_last_interaction()
            
            expected_target = item_name
            expected_action = menu_option
            
            # Use exact match or contains based on exact_match parameter
            target_match = (clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()) if exact_match else (expected_target.lower() in clean_rs(last_interaction.get("target", "")).lower())
            action_match = (clean_rs(last_interaction.get("action", "")).lower() == expected_action.lower()) if exact_match else (expected_action.lower() in clean_rs(last_interaction.get("action", "")).lower())
            
            if last_interaction and target_match and action_match:
                print(f"[CLICK] {expected_target} ({menu_option}) - interaction verified")
                return result
            else:
                print(f"[CLICK] {expected_target} ({menu_option}) - incorrect interaction, retrying...")
                continue
    
    return None


def is_full() -> bool:
    """
    Check if inventory is full (no empty slots).
    
    Returns:
        True if inventory is full (0 empty slots), False otherwise
    """
    empty_slots = get_empty_slots_count()
    return empty_slots == 0
