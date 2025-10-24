from typing import Optional

from .runtime_utils import ipc
from .utils import norm_name

from .rects import unwrap_rect
from .widgets import rect_center_from_widget


def inv_slots() -> list[dict]:
    from .runtime_utils import ipc
    inventory_data = ipc.get_inventory()
    return inventory_data.get("slots", []) or []

def inv_has(name: str, min_qty: int = 1) -> bool:
    n = norm_name(name)
    if min_qty <= 1:
        return any(norm_name(s.get("itemName")) == n for s in inv_slots())
    else:
        return inv_count(name) >= min_qty

def inv_count(name: str) -> int:
    n = norm_name(name)
    return sum(int(s.get("quantity") or 0) for s in inv_slots()
               if norm_name(s.get("itemName")) == n)

def first_inv_slot(name: str) -> dict | None:
    n = norm_name(name)
    for s in inv_slots():
        if norm_name(s.get("itemName")) == n:
            return s
    return None

def inventory_has_foreign_items() -> bool:
    allowed = {"Ring mould", "Gold bar", "Sapphire", "Emerald"}
    for slot in inv_slots():
        if int(slot.get("quantity") or 0) > 0 and (slot.get("itemName") or "") not in allowed:
            return True
    return False

def inventory_ring_slots() -> list[dict]:
    out = []
    for s in inv_slots():
        nm = norm_name(s.get("itemName"))
        if "ring" in nm and "mould" not in nm and int(s.get("quantity") or 0) > 0:
            out.append(s)
    return out

def inv_slot_bounds(slot_id: int) -> dict | None:
    from .runtime_utils import ipc
    inventory_widgets_data = ipc_send({"cmd": "get_inventory_widgets"}) or {}
    iw = inventory_widgets_data.get(str(slot_id)) or {}
    return unwrap_rect(iw.get("bounds") if isinstance(iw, dict) else None)

def coins() -> int:
    return inv_count("Coins")

def inv_has_any(excepted_items: list[str] = None) -> bool:
    from .runtime_utils import ipc
    inventory_data = ipc.get_inventory()
    
    if not inventory_data or not inventory_data.get("ok"):
        return False
    
    # Normalize excepted items if provided
    excepted_normalized = set()
    if excepted_items:
        excepted_normalized = {norm_name(item) for item in excepted_items if item}
    
    # Check if there are any actual items (not empty slots)
    slots = inventory_data.get("slots", [])
    for slot in slots:
        # Empty slots have id: -1 and quantity: 0
        if slot.get("id", -1) != -1 and slot.get("quantity", 0) > 0:
            # If we have excepted items, check if this item is in the excepted list
            if excepted_items:
                item_name = slot.get("itemName", "")
                item_normalized = norm_name(item_name)
                if item_normalized not in excepted_normalized:
                    return True
            else:
                # No excepted items, return True for any item
                return True
    
    return False

def get_item_coordinates(item: dict) -> Optional[tuple[int, int]]:
    """Extract click coordinates from an inventory item using existing helper functions."""
    # Use the existing rect_center_from_widget function which handles nested bounds
    coords = rect_center_from_widget(item)
    if coords[0] is not None and coords[1] is not None:
        return (coords[0], coords[1])
    
    # Fallback to direct x, y fields if available
    x = item.get("x")
    y = item.get("y")
    if x is not None and y is not None:
        return (int(x), int(y))
    
    return None


def has_only_items(allowed_items: list[str]) -> bool:
    """
    Check if the inventory contains only the specified items (and empty slots).
    
    Args:
        allowed_items: List of item names that are allowed in the inventory
        
    Returns:
        True if inventory contains only the allowed items (and empty slots), False otherwise
    """
    if not isinstance(allowed_items, list):
        return False
    
    # Normalize the allowed items list
    allowed_normalized = {norm_name(item) for item in allowed_items if item}
    
    # Get all inventory slots
    slots = inv_slots()
    
    # Check each slot
    for slot in slots:
        quantity = int(slot.get("quantity", 0))
        
        # Skip empty slots
        if quantity <= 0:
            continue
        
        # Get the item name and normalize it
        item_name = slot.get("itemName", "")
        item_normalized = norm_name(item_name)
        
        # If the item is not in the allowed list, return False
        if item_normalized not in allowed_normalized:
            return False
    
    # If we get here, all non-empty slots contain only allowed items
    return True