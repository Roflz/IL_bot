import time
from ..helpers.inventory import inv_has, inv_has_any, get_item_coordinates, inv_count
from ..helpers.context import get_payload, get_ui  # for optional payload
from ..helpers.widgets import rect_center_from_widget
from .runtime import emit
from .tab import ensure_inventory_tab_open
from typing import Optional, Callable

def has_item(name: str, min_qty: int = 1, payload: dict | None = None) -> bool:
    """
    True if inventory contains item `name` with quantity >= min_qty.
    Falls back to simple presence when min_qty <= 1.
    Tries inv_count(...) if available; otherwise scans payload["inventory"]["items"].
    """
    if payload is None:
        payload = get_payload()
    if min_qty <= 1:
        return inv_has(payload, name)

    # Prefer an existing counter helper if your codebase provides it
    try:
        inv_count_fn = globals().get("inv_count") or globals().get("inventory_count")
        if callable(inv_count_fn):
            return (int(inv_count_fn(payload, name)) >= int(min_qty))
    except Exception:
        pass

    # Fallback: manual scan of the payload
    try:
        items = (((payload or {}).get("inventory") or {}).get("slots") or [])
        want = (name or "").strip().lower()
        total = 0
        for it in items:
            nm = (it.get("itemName") or "").strip().lower()
            if nm == want:
                total += int(it.get("quantity") or 1)
        return total >= min_qty
    except Exception:
        return False

def has_noted_item(name: str, payload: dict | None = None) -> bool:
    """
    True if inventory contains noted version of item `name`.
    Checks for items with "(noted)" suffix or similar patterns.
    """
    if payload is None:
        payload = get_payload()
    
    items = ((payload or {}).get("inventory") or {}).get("slots") or []
    target_name = (name or "").strip().lower()
    
    for item in items:
        item_name = (item.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Check for other noted patterns (some items might have different formats)
        if target_name in item_name and item.get("isNoted"):
            return True
    
    return False

def has_unnoted_item(name: str, payload: dict | None = None) -> bool:
    """
    True if inventory contains unnoted version of item `name`.
    This is the normal item, not the noted version.
    """
    if payload is None:
        payload = get_payload()
    
    items = ((payload or {}).get("inventory") or {}).get("slots") or []
    target_name = (name or "").strip().lower()
    
    for item in items:
        item_name = (item.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Exact match and not noted
        if item_name == target_name and not item.get("isNoted"):
            return True
    
    return False



def is_empty(payload: dict | None = None) -> bool:
    """
    True if the player's inventory is empty.
    Accepts an optional payload; if omitted, uses the current global payload.
    """
    if payload is None:
        payload = get_payload()
    try:
        # preferred helper
        return not inv_has_any(payload)
    except Exception:
        # conservative fallback if helper/payload structure changes
        inv = (payload or {}).get("inventory") or []
        return len(inv) == 0 or all(
            (not item) or int(item.get("quantity", 0)) <= 0
            for item in inv
        )

def use_item_on_item(item1_name: str, item2_name: str, payload: Optional[dict] = None, ui=None, max_retries: int = 3) -> Optional[dict]:
    """
    Use an item in the inventory on another item in the inventory.
    
    Args:
        item1_name: Name of the item to use (the "using" item)
        item2_name: Name of the item to use on (the "target" item)
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
        max_retries: Maximum number of retry attempts if interaction fails
    
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Ensure inventory tab is open before using items
    if not ensure_inventory_tab_open(payload, ui):
        return None
    
    # Check if both items exist in inventory
    if not has_item(item1_name, payload=payload):
        return None
    if not has_item(item2_name, payload=payload):
        return None
    
    # Find the items in inventory
    items = ((payload or {}).get("inventory") or {}).get("slots") or []
    item1 = None
    item2 = None
    
    for item in items:
        item_name = (item.get("itemName") or "").strip()
        if not item_name:  # Skip empty slots
            continue
            
        # Exact match for item1 (the "using" item)
        if item1 is None and item_name.lower() == item1_name.lower():
            item1 = item
        # Exact match for item2 (the "target" item)  
        elif item2 is None and item_name.lower() == item2_name.lower():
            item2 = item
            
        if item1 and item2:
            break
    
    if not item1 or not item2:
        return None
    
    # Get click coordinates for both items
    item1_coords = get_item_coordinates(item1)
    item2_coords = get_item_coordinates(item2)
    
    if not item1_coords or not item2_coords:
        return None
    
    # Click first item with retry logic
    for attempt in range(max_retries):
        # Get fresh payload and item coordinates on each retry
        fresh_payload = get_payload()
        fresh_items = ((fresh_payload or {}).get("inventory") or {}).get("slots") or []
        fresh_item1 = None
        
        for item in fresh_items:
            item_name = (item.get("itemName") or "").strip()
            if item_name and item_name.lower() == item1_name.lower():
                fresh_item1 = item
                break
        
        if not fresh_item1:
            return None
            
        fresh_item1_coords = get_item_coordinates(fresh_item1)
        if not fresh_item1_coords:
            return None
        
        step1 = emit({
            "action": "inventory-use-item",
            "click": {"type": "point", "x": fresh_item1_coords[0], "y": fresh_item1_coords[1]},
            "target": {"domain": "inventory", "name": item1_name},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": item1_name
        })
        result1 = ui.dispatch(step1)
        
        if result1:
            break
            
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    if not result1:
        return None
    
    # Click second item with retry logic
    for attempt in range(max_retries):
        # Get fresh payload and item coordinates on each retry
        fresh_payload = get_payload()
        fresh_items = ((fresh_payload or {}).get("inventory") or {}).get("slots") or []
        fresh_item2 = None
        
        for item in fresh_items:
            item_name = (item.get("itemName") or "").strip()
            if item_name and item_name.lower() == item2_name.lower():
                fresh_item2 = item
                break
        
        if not fresh_item2:
            return None
            
        fresh_item2_coords = get_item_coordinates(fresh_item2)
        if not fresh_item2_coords:
            return None
        
        step2 = emit({
            "action": "inventory-use-on-item",
            "click": {"type": "point", "x": fresh_item2_coords[0], "y": fresh_item2_coords[1]},
            "target": {"domain": "inventory", "name": item2_name},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": item2_name
        })
        result2 = ui.dispatch(step2)
        
        if result2:
            return result2
            
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    return None


def inventory_has_amount(item_name: str, expected_amount: int, payload: dict | None = None) -> bool:
    """
    Helper function for use with wait_until to check if inventory contains 
    the expected amount of an item.
    
    Args:
        item_name: Name of the item to check
        expected_amount: Expected quantity in inventory
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if inventory contains at least the expected amount
    """
    if payload is None:
        payload = get_payload()
    
    current_count = inv_count(payload, item_name)
    return current_count >= expected_amount
