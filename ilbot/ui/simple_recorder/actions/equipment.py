# equipment.py (actions)

from __future__ import annotations
from typing import Optional, Union
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.utils import clean_rs
from ..helpers.widgets import widget_exists, get_widget_info
from ..helpers.ipc import ipc_send


def interact(item_name: str, menu_option: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Context-click an equipment inventory item and select a specific menu option.
    This interacts with unequipped items in the equipment interface, not equipped items.
    
    Args:
        item_name: Name of the equipment inventory item to interact with
        menu_option: Menu option to select (e.g., "Wear", "Examine", "Drop")
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Check if equipment interface is open
    if not equipment_interface_open(payload):
        return None
    
    # Find the item in equipment inventory
    item = find_equipment_inventory_item(item_name)
    if not item:
        return None
    
    # Get item bounds
    bounds = item.get('bounds')
    if not bounds:
        return None
    
    # Calculate center coordinates
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    # Context-click the item
    step = emit({
        "action": "equipment-interact",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "equipment", "name": item_name, "menu_option": menu_option},
    })
    return ui.dispatch(step)


def equipment_interface_open(payload: Optional[dict] = None) -> bool:
    """
    Check if the equipment interface is open and visible.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if equipment interface is open and visible, False otherwise
    """
    if payload is None:
        payload = get_payload()
    
    # Check if the equipment interface widget exists and is visible
    return widget_exists(5570560, payload)


def find_equipment_item(item_name: str, payload: Optional[dict] = None):
    """
    Find an equipped item by name using direct IPC detection.
    This looks for currently equipped items, not equipment inventory items.
    
    Args:
        item_name: Name of the equipped item to find
        payload: Optional payload, will get fresh if None
    
    Returns:
        Equipped item data dict if found, None otherwise
    """
    if payload is None:
        payload = get_payload()
    
    # Get equipment data using the get_equipment command (equipped items)
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    equipment = resp.get("equipment", {})
    slots = resp.get("slots", [])
    
    # Search through equipment slots
    for slot_data in slots:
        slot_name = clean_rs(slot_data.get("name", ""))
        if slot_name and slot_name.lower() == item_name.lower():
            return slot_data
    
    return None


def find_equipment_inventory_item(item_name: str, payload: Optional[dict] = None):
    """
    Find an equipment inventory item by name using direct IPC detection.
    This looks for unequipped items in the equipment interface.
    
    Args:
        item_name: Name of the equipment inventory item to find
        payload: Optional payload, will get fresh if None
    
    Returns:
        Equipment inventory item data dict if found, None otherwise
    """
    if payload is None:
        payload = get_payload()
    
    # Get equipment inventory data using the new get_equipment_inventory command
    resp = ipc_send({"cmd": "get_equipment_inventory"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    items = resp.get("items", [])
    
    # Search through equipment inventory items
    for item_data in items:
        item_name_clean = clean_rs(item_data.get("name", ""))
        if item_name_clean and item_name_clean.lower() == item_name.lower():
            return item_data
    
    return None


def get_equipment_item_bounds(slot_index: int, payload: Optional[dict] = None) -> Optional[dict]:
    """
    Get the bounds of an equipment item by slot index using direct IPC detection.
    
    Args:
        slot_index: Equipment slot index (0-27)
        payload: Optional payload, will get fresh if None
    
    Returns:
        Dictionary with bounds (x, y, width, height) or None if not found
    """
    if payload is None:
        payload = get_payload()
    
    # Get equipment widget children using IPC
    resp = ipc_send({"cmd": "get_widget_children", "widget_id": 5570560}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    children = resp.get("children", [])
    
    if slot_index >= len(children):
        return None
    
    child = children[slot_index]
    child_id = child.get("id")
    if not child_id:
        return None
    
    # Get detailed info for this child widget
    child_resp = ipc_send({"cmd": "get_widget_info", "widget_id": child_id}, payload)
    if not child_resp or not child_resp.get("ok"):
        return None
    
    child_data = child_resp.get("data", {})
    bounds = child_data.get("bounds")
    
    if not bounds:
        return None
    
    return {
        "x": bounds.get("x", 0),
        "y": bounds.get("y", 0),
        "width": bounds.get("width", 0),
        "height": bounds.get("height", 0)
    }


def has_equipment_item(item_name: str, payload: Optional[dict] = None) -> bool:
    """
    Check if a specific equipment item is equipped using direct IPC detection.
    
    Args:
        item_name: Name of the equipment item to check
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if item is equipped, False otherwise
    """
    return find_equipment_item(item_name, payload) is not None


def get_equipment_data(payload: Optional[dict] = None) -> Optional[dict]:
    """
    Get all equipment data using direct IPC detection.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        Equipment data dict with all slots, or None if failed
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    return resp.get("equipment", {})


def get_equipment_slot(slot_name: str, payload: Optional[dict] = None) -> Optional[dict]:
    """
    Get equipment data for a specific slot (e.g., "HEAD", "WEAPON", "BODY").
    
    Args:
        slot_name: Name of the equipment slot (HEAD, WEAPON, BODY, etc.)
        payload: Optional payload, will get fresh if None
    
    Returns:
        Equipment slot data dict, or None if not found
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return None
    
    equipment = resp.get("equipment", {})
    return equipment.get(slot_name.lower())


def list_equipped_items(payload: Optional[dict] = None) -> list[dict]:
    """
    Get a list of all currently equipped items.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        List of equipped item data dicts
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return []
    
    slots = resp.get("slots", [])
    equipped = []
    
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            equipped.append(slot_data)
    
    return equipped


def has_equipped(item_names: Union[str, list[str]], payload: Optional[dict] = None) -> bool:
    """
    Check if you have all the specified items equipped.
    
    Args:
        item_names: Single item name or list of item names to check for 
                   (e.g., "Bronze sword" or ["Bronze sword", "Wooden shield"])
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if all items are equipped, False otherwise
    """
    # Convert single string to list for consistent handling
    if isinstance(item_names, str):
        item_names = [item_names]
    
    if not item_names:
        return True
    
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    # Get all equipped item names
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name.lower())
    
    # Check if all requested items are equipped
    for item_name in item_names:
        if clean_rs(item_name).lower() not in equipped_names:
            return False
    
    return True


def has_any_equipped(item_names: list[str], payload: Optional[dict] = None) -> bool:
    """
    Check if you have any of the specified items equipped.
    
    Args:
        item_names: List of item names to check for (e.g., ["Bronze sword", "Iron sword"])
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if any item is equipped, False otherwise
    """
    if not item_names:
        return False
    
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    # Get all equipped item names
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name.lower())
    
    # Check if any requested item is equipped
    for item_name in item_names:
        if clean_rs(item_name).lower() in equipped_names:
            return True
    
    return False


def get_equipped_item_names(payload: Optional[dict] = None) -> list[str]:
    """
    Get a list of all currently equipped item names.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        List of equipped item names
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_equipment"}, payload)
    if not resp or not resp.get("ok"):
        return []
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name)
    
    return equipped_names
