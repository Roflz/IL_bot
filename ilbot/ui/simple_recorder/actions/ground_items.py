from __future__ import annotations
from typing import Optional
import time

from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.ipc import ipc_send
from ..services.camera_integration import dispatch_with_camera


def loot(item_name: str, radius: int = 10, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Search for a ground item around the player and pick it up.
    
    Args:
        item_name: Name of the item to look for (partial match allowed)
        radius: Search radius around player (default: 10 tiles)
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get fresh payload for ground items search
    fresh_payload = get_payload()
    
    # Search for ground items using IPC
    ground_items_resp = ipc_send({
        "cmd": "ground_items",
        "name": item_name,
        "radius": radius
    }, fresh_payload)
    
    if not ground_items_resp or not ground_items_resp.get("ok"):
        print(f"[LOOT] Failed to search for ground items: {item_name}")
        return None
    
    ground_items = ground_items_resp.get("items", [])
    if not ground_items:
        print(f"[LOOT] No ground items found with name containing '{item_name}'")
        return None
    
    # Find the closest ground item
    closest_item = None
    closest_distance = float('inf')
    
    for item in ground_items:
        distance = item.get("distance", float('inf'))
        if distance < closest_distance:
            closest_distance = distance
            closest_item = item
    
    if not closest_item:
        print(f"[LOOT] No valid ground items found")
        return None
    
    print(f"[LOOT] Found ground item: {closest_item.get('name')} at distance {closest_distance}")
    
    # Pick up the ground item
    result = _pickup_ground_item(closest_item, fresh_payload, ui)
    if result:
        print(f"[LOOT] Successfully picked up {closest_item.get('name')}")
    
    return result


def _pickup_ground_item(item: dict, payload: dict, ui) -> Optional[dict]:
    """
    Pick up a specific ground item.
    
    Args:
        item: Ground item data dictionary
        payload: Game state payload
        ui: UI instance
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    from ..helpers.rects import unwrap_rect, rect_center_xy
    
    # Get item coordinates and bounds
    rect = unwrap_rect(item.get("clickbox"))
    world_coords = {
        "x": item.get("world", {}).get("x"), 
        "y": item.get("world", {}).get("y"), 
        "p": item.get("world", {}).get("p", 0)
    }
    
    # Determine click coordinates
    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
        print(f"[LOOT] Using rect coordinates: ({cx}, {cy})")
    elif isinstance(item.get("canvas", {}).get("x"), (int, float)) and isinstance(item.get("canvas", {}).get("y"), (int, float)):
        cx, cy = int(item.get("canvas", {}).get("x")), int(item.get("canvas", {}).get("y"))
        anchor = {}
        point = {"x": cx, "y": cy}
        print(f"[LOOT] Using canvas coordinates: ({cx}, {cy})")
    else:
        print(f"[LOOT] No valid coordinates found for ground item")
        return None
    
    # Ground items always use "Take" action - use context menu with specific item name
    item_name = item.get("name", "Unknown")
    world_coords = {
        "x": item.get("world", {}).get("x"), 
        "y": item.get("world", {}).get("y"), 
        "p": item.get("world", {}).get("p", 0)
    }
    print(f"[LOOT] Using context menu with 'Take {item_name}' for ground item pickup")
    
    # Create the pickup step using context menu with "Take [ItemName]" action
    step = emit({
        "action": "ground-item-pickup-context",
        "click": {
            "type": "context-select",
            "option": f"Take",  # Use "Take [ItemName]" to specify which item
            "target": f"{item_name}",  # Use "Take [ItemName]" to specify which item
            "x": cx,
            "y": cy,
            "row_height": 16,
            "start_dy": 18,
            "open_delay_ms": 120
        },
        "target": {"domain": "ground-item", "name": item_name, "world": world_coords, **anchor} if rect else {"domain": "ground-item", "name": item_name, "world": world_coords},
        "anchor": point
    })
    
    # Execute the pickup with camera integration
    result = dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)
    
    return result


def _wait_for_inventory_increase(item_name: str, timeout_seconds: int = 3) -> bool:
    """
    Wait for the inventory count of a specific item to increase by 1.
    
    Args:
        item_name: Name of the item to check for
        timeout_seconds: Maximum time to wait in seconds
        
    Returns:
        True if inventory count increased, False if timeout
    """
    from .inventory import get_inventory_count
    
    # Get initial count
    initial_count = get_inventory_count(item_name)
    print(f"[LOOT] Initial inventory count for {item_name}: {initial_count}")
    
    # Wait for count to increase
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        current_count = get_inventory_count(item_name)
        if current_count > initial_count:
            print(f"[LOOT] Inventory count increased from {initial_count} to {current_count} for {item_name}")
            return True
        time.sleep(0.1)  # Check every 100ms
    
    print(f"[LOOT] Timeout waiting for inventory increase for {item_name}")
    return False
