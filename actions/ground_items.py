from __future__ import annotations
from typing import Optional

from helpers.runtime_utils import ipc, dispatch


def loot(item_name: str, radius: int = 10) -> Optional[dict]:
    """
    Search for a ground item around the player and pick it up.
    
    Args:
        item_name: Name of the item to look for (partial match allowed)
        radius: Search radius around player (default: 10 tiles)
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    # Search for ground items using IPC
    ground_items_resp = ipc.get_ground_items(item_name, radius)
    
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
    result = _pickup_ground_item(closest_item)
    if result:
        print(f"[LOOT] Successfully picked up {closest_item.get('name')}")
    
    return result


def _pickup_ground_item(item: dict) -> Optional[dict]:
    """
    Pick up a specific ground item.
    
    Args:
        item: Ground item data dictionary
        ui: UI instance
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    from ..helpers.rects import unwrap_rect
    from ..helpers.utils import rect_beta_xy, clean_rs
    
    # Inner attempt loop with fresh coordinate recalculation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Fresh coordinate recalculation
        # Get item coordinates and bounds
        rect = unwrap_rect(item.get("clickbox"))
        world_coords = {
            "x": item.get("world", {}).get("x"), 
            "y": item.get("world", {}).get("y"), 
            "p": item.get("world", {}).get("p", 0)
        }
        
        # Determine click coordinates
        if rect:
            cx, cy = rect_beta_xy((rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0),
                                   rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)), alpha=2.0, beta=2.0)
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
            continue
        
        # Ground items always use "Take" action - use context menu with specific item name
        item_name = item.get("name", "Unknown")
        world_coords = {
            "x": item.get("world", {}).get("x"), 
            "y": item.get("world", {}).get("y"), 
            "p": item.get("world", {}).get("p", 0)
        }
        print(f"[LOOT] Using context menu with 'Take {item_name}' for ground item pickup")
        
        # Create the pickup step using context menu with "Take [ItemName]" action
        step = {
            "action": "ground-item-pickup-context",
            "option": f"Take",  # Use "Take [ItemName]" to specify which item
            "click": {
                "type": "context-select",
                "target": f"{item_name}",  # Use "Take [ItemName]" to specify which item
                "x": cx,
                "y": cy,
                "row_height": 16,
                "start_dy": 18,
                "open_delay_ms": 120
            },
            "target": {"domain": "ground-item", "name": item_name, "world": world_coords, **anchor} if rect else {"domain": "ground-item", "name": item_name, "world": world_coords},
            "anchor": point
        }
        
        # Execute the pickup with camera integration
        result = dispatch(step)
        
        if result:
            # Check if the correct interaction was performed
            from ..helpers.ipc import get_last_interaction
            last_interaction = get_last_interaction()
            
            expected_action = "Take"
            expected_target = item_name
            
            if (last_interaction and 
                last_interaction.get("action") == expected_action and 
                clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()):
                print(f"[CLICK] {expected_target} - interaction verified")
                return result
            else:
                print(f"[CLICK] {expected_target} - incorrect interaction, retrying...")
                continue
        
        # If we get here, dispatch failed
        continue
    
    return None
