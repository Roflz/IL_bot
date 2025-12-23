"""
Bank Inventory Helper Functions
==============================

Helper functions for interacting with the bank inventory interface.
Similar to equipment inventory but for bank items.
"""

from helpers.runtime_utils import ipc


def get_bank_inventory_items() -> list[dict]:
    """
    Get all items from the bank inventory interface.
        
    Returns:
        List of bank inventory items with their details
    """
    # Send IPC command to get bank inventory items
    response = ipc.get_bank_inventory()
    
    if not response or not response.get("ok"):
        print(f"[BANK_INVENTORY] Failed to get bank inventory: {response.get('err', 'unknown error')}")
        return []
    
    items = response.get("items", [])
    print(f"[BANK_INVENTORY] Found {len(items)} bank inventory items")
    
    return items


def find_bank_inventory_item(item_name: str) -> dict | None:
    """
    Find a specific item in the bank inventory by name.
    
    Args:
        item_name: Name of the item to find
        
    Returns:
        Item data if found, None otherwise
    """
    items = get_bank_inventory_items()
    
    for item in items:
        if item.get("name", "").lower() == item_name.lower():
            return item
    
    return None


def has_bank_inventory_item(item_name: str) -> bool:
    """
    Check if a specific item exists in the bank inventory.
    
    Args:
        item_name: Name of the item to check for
        
    Returns:
        True if item exists, False otherwise
    """
    return find_bank_inventory_item(item_name) is not None


def get_bank_inventory_count() -> int:
    """
    Get the total number of items in the bank inventory.
        
    Returns:
        Number of items in bank inventory
    """
    items = get_bank_inventory_items()
    return len(items)


def is_bank_inventory_open() -> bool:
    """
    Check if the bank inventory interface is open.
        
    Returns:
        True if bank inventory is open, False otherwise
    """
    response = ipc.get_bank_inventory()
    
    if not response or not response.get("ok"):
        return False
    
    return response.get("interfaceOpen", False)


def get_bank_inventory_item_bounds(item_name: str) -> dict | None:
    """
    Get the bounds of a specific bank inventory item.
    
    Args:
        item_name: Name of the item
        
    Returns:
        Bounds dictionary with x, y, width, height, or None if not found
    """
    item = find_bank_inventory_item(item_name)
    
    if not item:
        return None
    
    return item.get("bounds")


def get_bank_inventory_item_canvas(item_name: str) -> dict | None:
    """
    Get the canvas coordinates of a specific bank inventory item.
    
    Args:
        item_name: Name of the item
        
    Returns:
        Canvas coordinates dictionary with x, y, or None if not found
    """
    item = find_bank_inventory_item(item_name)
    
    if not item:
        return None
    
    return item.get("canvas")
