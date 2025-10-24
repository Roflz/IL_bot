import time

from . import tab
from .tab import open_tab
from .timing import wait_until
from ..helpers.inventory import inv_has, inv_has_any, get_item_coordinates, inv_count, first_inv_slot, inv_slot_bounds
from ..helpers.runtime_utils import ipc, ui, dispatch
from ..helpers.tab import is_inventory_tab_open

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
    # Ensure inventory tab is open before using items
    if not tab.is_tab_open("INVENTORY"):
        if not tab.ensure_tab_open("INVENTORY"):
            return None
    
    # Check if both items exist in inventory
    if not has_item(item1_name):
        return None
    if not has_item(item2_name):
        return None
    
    # Find the items in inventory
    inventory_data = ipc.get_inventory() or {}
    items = inventory_data.get("slots", [])
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
        # Get fresh inventory data and item coordinates on each retry
        fresh_inventory_data = ipc.get_inventory() or {}
        fresh_items = fresh_inventory_data.get("slots") or []
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
        
        step1 = {
            "action": "inventory-use-item",
            "click": {"type": "point", "x": fresh_item1_coords[0], "y": fresh_item1_coords[1]},
            "target": {"domain": "inventory", "name": item1_name},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": item1_name
        }
        result1 = dispatch(step1)
        
        if result1:
            time.sleep(0.2)
            break
            
        if attempt < max_retries - 1:
            time.sleep(0.5)
    
    if not result1:
        return None
    
    # Click second item with retry logic
    for attempt in range(max_retries):
        # Get fresh inventory data and item coordinates on each retry
        fresh_inventory_data = ipc.get_inventory() or {}
        fresh_items = fresh_inventory_data.get("slots") or []
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
        
        step2 = {
            "action": "inventory-use-on-item",
            "click": {"type": "point", "x": fresh_item2_coords[0], "y": fresh_item2_coords[1]},
            "target": {"domain": "inventory", "name": item2_name},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": item2_name
        }
        result2 = dispatch(step2)
        
        if result2:
            return result2
            
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    return None


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
    # Ensure inventory tab is open before using items
    if not tab.is_tab_open("INVENTORY"):
        tab.open_inventory_tab()
        return None
    
    # Check if item exists in inventory
    if not has_item(item_name):
        return None
    
    # Find the item in inventory
    inventory_data = ipc.get_inventory() or {}
    items = inventory_data.get("slots", [])
    item = None
    
    for inv_item in items:
        inv_item_name = (inv_item.get("itemName") or "").strip()
        if not inv_item_name:  # Skip empty slots
            continue
            
        # Exact match for the item
        if inv_item_name.lower() == item_name.lower():
            item = inv_item
            break
    
    if not item:
        return None
    
    # Get click coordinates for the item
    item_coords = get_item_coordinates(item)
    if not item_coords:
        return None
    
    # Click the item with retry logic
    for attempt in range(max_retries):
        # Get fresh inventory data and item coordinates on each retry
        fresh_inventory_data = ipc.get_inventory() or {}
        fresh_items = fresh_inventory_data.get("slots") or []
        fresh_item = None
        
        for inv_item in fresh_items:
            inv_item_name = (inv_item.get("itemName") or "").strip()
            if inv_item_name and inv_item_name.lower() == item_name.lower():
                fresh_item = inv_item
                break
        
        if not fresh_item:
            return None
            
        fresh_item_coords = get_item_coordinates(fresh_item)
        if not fresh_item_coords:
            return None
        
        step1 = {
            "action": "inventory-use-item",
            "click": {"type": "point", "x": fresh_item_coords[0], "y": fresh_item_coords[1]},
            "target": {"domain": "inventory", "name": item_name},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": item_name
        }
        result1 = dispatch(step1)
        
        if result1:
            break
            
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    if not result1:
        return None
    
    # Now click on the game object with retry logic
    for attempt in range(max_retries):
        # Get fresh object data for object search
        objects_data = ipc.get_objects() or {}
        
        # Search for the game object
        objs = (objects_data.get("closestGameObjects") or []) + (objects_data.get("gameObjects") or [])
        target_obj = None
        
        for obj in objs:
            obj_name = (obj.get("name") or "").strip()
            if not obj_name:
                continue
                
            # Partial match for object name
            if object_name.lower() in obj_name.lower():
                target_obj = obj
                break
        
        # IPC fallback if not found in objects data
        if target_obj is None:
            resp = ipc.get_objects(object_name, ["WALL", "GAME", "DECOR", "GROUND"]) or {}
            found = resp.get("objects") or []
            if found:
                # Choose closest object
                player_data = ipc.get_player() or {}
                me = player_data.get("player") or {}
                me_x = me.get("worldX") if isinstance(me.get("worldX"), int) else player_data.get("worldX")
                me_y = me.get("worldY") if isinstance(me.get("worldY"), int) else player_data.get("worldY")
                
                def obj_wxy_p(o):
                    w = o.get("world") or {}
                    ox = w.get("x", o.get("worldX"))
                    oy = w.get("y", o.get("worldY"))
                    return ox, oy
                
                scored = []
                for o in found:
                    ox, oy = obj_wxy_p(o)
                    if isinstance(me_x, int) and isinstance(me_y, int) and isinstance(ox, int) and isinstance(oy, int):
                        dist = abs(ox - me_x) + abs(oy - me_y)
                    else:
                        dist = 10**9
                    scored.append((dist, o))
                
                scored.sort(key=lambda t: t[0])
                target_obj = scored[0][1]
        
        if not target_obj:
            return None
        
        # Get object click coordinates
        from ..helpers.rects import unwrap_rect, rect_center_xy
        rect = unwrap_rect(target_obj.get("clickbox")) or unwrap_rect(target_obj.get("bounds"))
        if rect:
            cx, cy = rect_center_xy(rect)
            point = {"x": cx, "y": cy}
            anchor = {"bounds": rect}
        elif isinstance(target_obj.get("canvasX"), (int, float)) and isinstance(target_obj.get("canvasY"), (int, float)):
            cx, cy = int(target_obj["canvasX"]), int(target_obj["canvasY"])
            point = {"x": cx, "y": cy}
            anchor = {}
        else:
            return None
        
        step2 = {
            "action": "inventory-use-on-object",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": "object", "name": object_name, **anchor},
            "max_retries": 1,
            "expected_action": "Use",
            "expected_target": object_name
        }
        result2 = dispatch(step2)
        
        if result2:
            return result2
            
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    return None


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


def interact(item_name: str, menu_option: str) -> Optional[dict]:
    """
    Context-click an inventory item and select a specific menu option.
    
    Args:
        item_name: Name of the item to interact with
        menu_option: Menu option to select (e.g., "Use", "Drop", "Examine")
    
    Returns:
        UI dispatch result or None if failed
    """
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Opening inventory tab before interacting with {item_name}")
        open_tab("INVENTORY")
        # Wait a moment for the tab to open
        time.sleep(0.2)
    
    # Ensure we're still on inventory tab before proceeding
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Failed to open inventory tab, aborting interaction")
        return None
    
    # Find the item in inventory
    item = first_inv_slot(item_name)
    if not item:
        return None
    
    # Get item bounds
    bounds = item.get("bounds")
    if not bounds:
        return None
    
    # Calculate center coordinates
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    # Context-click the item
    step = {
        "action": "inventory-interact",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "inventory", "name": item_name, "menu_option": menu_option},
    }
    return dispatch(step)


def is_full() -> bool:
    """
    Check if inventory is full (no empty slots).
    
    Returns:
        True if inventory is full (0 empty slots), False otherwise
    """
    empty_slots = get_empty_slots_count()
    return empty_slots == 0
