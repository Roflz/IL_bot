"""
Bank Inventory Helper Functions
==============================

Helper functions for interacting with the bank inventory interface.
Similar to equipment inventory but for bank items.
"""

from ..helpers.ipc import ipc_send
from ..helpers.context import get_payload


def get_bank_inventory_items(payload: dict | None = None) -> list[dict]:
    """
    Get all items from the bank inventory interface.
    
    Args:
        payload: Game state payload (optional)
        
    Returns:
        List of bank inventory items with their details
    """
    if payload is None:
        payload = get_payload()
    
    # Send IPC command to get bank inventory items
    response = ipc_send({"cmd": "get_bank_inventory"}, payload)
    
    if not response or not response.get("ok"):
        print(f"[BANK_INVENTORY] Failed to get bank inventory: {response.get('err', 'unknown error')}")
        return []
    
    items = response.get("items", [])
    print(f"[BANK_INVENTORY] Found {len(items)} bank inventory items")
    
    return items


def find_bank_inventory_item(item_name: str, payload: dict | None = None) -> dict | None:
    """
    Find a specific item in the bank inventory by name.
    
    Args:
        item_name: Name of the item to find
        payload: Game state payload (optional)
        
    Returns:
        Item data if found, None otherwise
    """
    items = get_bank_inventory_items(payload)
    
    for item in items:
        if item.get("name", "").lower() == item_name.lower():
            return item
    
    return None


def has_bank_inventory_item(item_name: str, payload: dict | None = None) -> bool:
    """
    Check if a specific item exists in the bank inventory.
    
    Args:
        item_name: Name of the item to check for
        payload: Game state payload (optional)
        
    Returns:
        True if item exists, False otherwise
    """
    return find_bank_inventory_item(item_name, payload) is not None


def get_bank_inventory_count(payload: dict | None = None) -> int:
    """
    Get the total number of items in the bank inventory.
    
    Args:
        payload: Game state payload (optional)
        
    Returns:
        Number of items in bank inventory
    """
    items = get_bank_inventory_items(payload)
    return len(items)


def is_bank_inventory_open(payload: dict | None = None) -> bool:
    """
    Check if the bank inventory interface is open.
    
    Args:
        payload: Game state payload (optional)
        
    Returns:
        True if bank inventory is open, False otherwise
    """
    if payload is None:
        payload = get_payload()
    
    response = ipc_send({"cmd": "get_bank_inventory"}, payload)
    
    if not response or not response.get("ok"):
        return False
    
    return response.get("interfaceOpen", False)


def get_bank_inventory_item_bounds(item_name: str, payload: dict | None = None) -> dict | None:
    """
    Get the bounds of a specific bank inventory item.
    
    Args:
        item_name: Name of the item
        payload: Game state payload (optional)
        
    Returns:
        Bounds dictionary with x, y, width, height, or None if not found
    """
    item = find_bank_inventory_item(item_name, payload)
    
    if not item:
        return None
    
    return item.get("bounds")


def get_bank_inventory_item_canvas(item_name: str, payload: dict | None = None) -> dict | None:
    """
    Get the canvas coordinates of a specific bank inventory item.
    
    Args:
        item_name: Name of the item
        payload: Game state payload (optional)
        
    Returns:
        Canvas coordinates dictionary with x, y, or None if not found
    """
    item = find_bank_inventory_item(item_name, payload)
    
    if not item:
        return None
    
    return item.get("canvas")
