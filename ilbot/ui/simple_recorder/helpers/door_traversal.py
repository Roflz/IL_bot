import time
from typing import Dict, List, Optional, Tuple, Any
from .context import get_payload
from .ipc import ipc_send


def check_door_traversal(door_x: int, door_y: int, door_p: int, max_time: float = 3.0) -> Dict[str, Any]:
    """
    Check for door traversal by monitoring door tile and adjacent tiles for new wall objects.
    
    Args:
        door_x, door_y, door_p: Door coordinates
        max_time: Maximum time to wait in seconds
        
    Returns:
        Dictionary with traversal status and tile information
    """
    start_time = time.time()
    payload = get_payload()
    
    # Define tiles to check: door tile and adjacent tiles
    tiles_to_check = [
        (door_x - 1, door_y, door_p),  # before tile
        (door_x, door_y, door_p),      # door tile
        (door_x + 1, door_y, door_p),  # after tile
    ]
    
    print(f"[DOOR] Starting traversal check for door at ({door_x}, {door_y})")
    print(f"[DOOR] Checking tiles: {tiles_to_check}")
    
    while (time.time() - start_time) < max_time:
        tile_info = {}
        
        for i, (x, y, p) in enumerate(tiles_to_check):
            # Get wall object info for this tile
            resp = ipc_send({
                "cmd": "door_state",
                "door_x": x,
                "door_y": y, 
                "door_p": p
            }, payload)
            
            if resp and resp.get("ok"):
                tile_info[f"tile_{i}"] = {
                    "coordinates": (x, y, p),
                    "wall_object": resp.get("wall_object")
                }
            else:
                tile_info[f"tile_{i}"] = {
                    "coordinates": (x, y, p),
                    "wall_object": None,
                    "error": resp.get("err") if resp else "no_response"
                }
        
        print(f"[DOOR] Check at {time.time() - start_time:.2f}s:")
        for tile_name, info in tile_info.items():
            coords = info["coordinates"]
            wobj = info.get("wall_object")
            if wobj:
                print(f"  {tile_name} ({coords[0]}, {coords[1]}): ID={wobj.get('id')}, orientationA={wobj.get('orientationA')}, orientationB={wobj.get('orientationB')}")
            else:
                print(f"  {tile_name} ({coords[0]}, {coords[1]}): No wall object")
        
        # Check every 100ms
        time.sleep(0.1)
    
    print(f"[DOOR] Traversal check completed after {time.time() - start_time:.2f}s")
    
    return {
        "completed": True,
        "duration": time.time() - start_time,
        "final_tile_info": tile_info
    }
