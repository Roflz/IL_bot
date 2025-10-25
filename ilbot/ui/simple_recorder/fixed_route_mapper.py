#!/usr/bin/env python3
"""
Fixed route mapping script.
Properly handles scene boundaries and provides accurate collision data collection.
"""
import sys
import os
import time
import json
import socket
import argparse
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_ipc_port(start_port: int = 17000, end_port: int = 17020) -> int:
    """Find the IPC port by testing connectivity."""
    for port in range(start_port, end_port + 1):
        try:
            resp = ipc_send_direct({"cmd": "ping"}, port=port, timeout=0.5)
            if resp and resp.get("ok"):
                print(f"[PORT_DETECT] Found IPC plugin on port {port}")
                return port
        except Exception:
            continue
    return None

def ipc_send_direct(msg: dict, port: int = 17000, timeout: float = 0.35):
    """Send a message directly to the IPC plugin."""
    host = "127.0.0.1"
    t0 = time.time()
    try:
        line = json.dumps(msg, separators=(",", ":"))
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((line + "\n").encode("utf-8"))
            data = b""
            while True:
                ch = s.recv(1)
                if not ch or ch == b"\n":
                    break
                data += ch
        resp = json.loads(data.decode("utf-8")) if data else None
        return resp
    except Exception as e:
        print(f"[IPC ERR] {type(e).__name__}: {e}")
        return None


def get_player_position(port: int = 17000):
    """Get current player position."""
    resp = ipc_send_direct({"cmd": "get_player"}, port=port)
    if resp and resp.get("ok"):
        player = resp.get("player", {})
        return player.get("worldX"), player.get("worldY"), player.get("worldP")
    return None, None, None


def scan_current_scene(port: int = 17000):
    """Scan the current scene for collision data."""
    resp = ipc_send_direct({"cmd": "scan_scene"}, port=port)
    if resp and resp.get("ok"):
        return resp
    return None


def is_player_in_scene(player_x, player_y, player_p, scene_base_x, scene_base_y, scene_plane):
    """Check if player is within the current scene bounds."""
    if player_x is None or player_y is None or player_p is None:
        return False
    
    # Scene bounds (104x104 tiles)
    scene_min_x = scene_base_x
    scene_max_x = scene_base_x + 104
    scene_min_y = scene_base_y
    scene_max_y = scene_base_y + 104
    
    return (scene_min_x <= player_x < scene_max_x and 
            scene_min_y <= player_y < scene_max_y and 
            player_p == scene_plane)


# Removed auto-map generation to prevent cache corruption


def save_collision_data(data, cache_dir=""):
    """Save collision data to cache."""
    cache_path = Path(cache_dir) if cache_dir else Path(".")
    cache_dir_path = cache_path / "collision_cache"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Use main cache file
    cache_file = cache_dir_path / "collision_map_debug.json"
    
    # Load existing data from the cache file
    existing_data = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load existing cache: {e}")
    
    # Merge new data
    collision_data = existing_data.get("collision_data", {})
    water_data = existing_data.get("water_data", {})
    
    # Add new collision tiles
    new_tiles = 0
    for tile in data.get("collisionData", []):
        key = f"{tile['x']},{tile['y']},{tile['p']}"
        if key not in collision_data:
            # Store the new tile data verbatim (flags, door, ladderUp, ladderDown)
            collision_data[key] = tile
            new_tiles += 1
    
    # Only save if there are new tiles
    if new_tiles > 0:
        print(f"  [DEBUG] Saving to: {cache_file.absolute()}")
        print(f"  [DEBUG] Writing {len(collision_data)} collision tiles and {len(water_data)} water tiles")
        print(f"  [DEBUG] New tiles being added: {new_tiles}")
        
        # Atomic save: write to temp file first, then rename
        temp_file = cache_file.with_suffix('.tmp')
        try:
            data_to_write = {
                "collision_data": collision_data,
                "water_data": water_data
            }
            print(f"  [DEBUG] Writing data structure with keys: {list(data_to_write.keys())}")
            
            with open(temp_file, 'w') as f:
                json.dump(data_to_write, f, indent=2)
            
            print(f"  [DEBUG] Temp file written successfully")
            
            # Verify the temp file was written correctly
            print(f"  [DEBUG] Verifying temp file...")
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
                temp_tiles = len(temp_data.get("collision_data", {}))
                temp_water = len(temp_data.get("water_data", {}))
                print(f"  [DEBUG] Temp file contains: {temp_tiles} collision tiles, {temp_water} water tiles")
                if temp_tiles != len(collision_data):
                    raise Exception(f"Write verification failed: expected {len(collision_data)} tiles, got {temp_tiles}")
            
            # Atomic rename
            print(f"  [DEBUG] Renaming temp file to final cache file...")
            temp_file.replace(cache_file)
            print(f"  [DEBUG] Atomic rename completed")
            
            # Verify the final file was written correctly
            print(f"  [DEBUG] Verifying final cache file...")
            with open(cache_file, 'r') as f:
                final_data = json.load(f)
                final_tiles = len(final_data.get("collision_data", {}))
                final_water = len(final_data.get("water_data", {}))
                print(f"  [DEBUG] Final file contains: {final_tiles} collision tiles, {final_water} water tiles")
            
            print(f"  [SAVED] Added {new_tiles} new tiles to cache")
            print(f"  [VERIFY] In-memory: {len(collision_data)} tiles, File: {final_tiles} tiles")
            
        except Exception as e:
            print(f"  [ERROR] Failed to save cache: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            return len(collision_data), new_tiles
    else:
        print(f"  [SKIPPED] No new tiles to save")

    print(f"Current working directory: {os.getcwd()}")
    os.system("python visual_collision_map.py")
    
    return len(collision_data), new_tiles


def fixed_route_mapper(port: int = None):
    """
    Fixed route mapping - properly handles scene boundaries.
    """
    print("FIXED ROUTE MAPPER")
    print("=" * 50)
    print("This script will properly collect collision data as you walk around.")
    print("It handles scene boundaries correctly and provides accurate information.")
    print("Press Ctrl+C to stop when you reach your destination.")
    print()
    
    # Auto-detect port if not provided
    if port is None:
        print("Auto-detecting IPC port...")
        port = find_ipc_port()
        if port is None:
            print("ERROR: Could not find IPC plugin on any port (17000-17020). Make sure:")
            print("  1. RuneLite is running")
            print("  2. IPC plugin is enabled")
            print("  3. State Exporter plugin is enabled")
            print("  4. Try specifying a port manually: python fixed_route_mapper.py --port 17001")
            return False
    else:
        print(f"Using specified port: {port}")
    
    # Test connection
    print("Testing connection to IPC plugin...")
    ping_resp = ipc_send_direct({"cmd": "ping"}, port=port)
    
    if not ping_resp or not ping_resp.get("ok"):
        print("ERROR: Cannot connect to IPC plugin. Make sure:")
        print("  1. RuneLite is running")
        print("  2. IPC plugin is enabled")
        print("  3. State Exporter plugin is enabled")
        return False
    
    print("SUCCESS: Connected to IPC plugin")
    print()
    
    # Load existing cache
    cache_file = Path("collision_cache") / "collision_map_debug.json"
    collision_data = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            collision_data = data.get("collision_data", {})
            print(f"Loaded existing cache: {len(collision_data)} tiles")
        except Exception as e:
            print(f"[WARNING] Could not load existing cache: {e}")
    
    scan_count = 0
    last_scene = None
    total_tiles = len(collision_data)
    
    print(f"Starting with {total_tiles} cached tiles")
    print()
    
    try:
        while True:
            # Get player position
            player_x, player_y, player_p = get_player_position(port)
            
            # Scan current scene
            scene_data = scan_current_scene(port)
            if scene_data and scene_data.get("ok"):
                base_x = scene_data.get("baseX")
                base_y = scene_data.get("baseY")
                plane = scene_data.get("plane")
                collision_count = scene_data.get("count", 0)
                
                current_scene = (base_x, base_y, plane)
                
                # Check if player is in the current scene
                player_in_scene = is_player_in_scene(player_x, player_y, player_p, base_x, base_y, plane)
                
                if current_scene != last_scene:
                    print(f"New scene detected: ({base_x}, {base_y}, {plane})")
                    print(f"  Scene collision tiles: {collision_count}")
                    
                    if player_in_scene:
                        print(f"  [OK] Player is in this scene: ({player_x}, {player_y}, {player_p})")
                    else:
                        print(f"  [WARNING] Player is OUTSIDE this scene: ({player_x}, {player_y}, {player_p})")
                        print(f"    Scene bounds: ({base_x}-{base_x+104}, {base_y}-{base_y+104}, {plane})")
                    
                    # Add new tiles to cache
                    total_tiles, new_tiles = save_collision_data(scene_data)
                    
                    if new_tiles > 0:
                        print(f"  Added {new_tiles} new tiles to cache")
                        print(f"  Total cached tiles: {total_tiles}")
                        scan_count += 1
                    else:
                        print(f"  No new tiles (scene already cached)")
                    
                    print()
                    last_scene = current_scene
                else:
                    # Same scene
                    if player_in_scene:
                        print(f"Same scene ({base_x}, {base_y}, {plane}) - Player in scene, waiting for movement...")
                    else:
                        print(f"Same scene ({base_x}, {base_y}, {plane}) - Player outside scene, waiting for movement...")
            else:
                print("ERROR: Could not scan current scene")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\nRoute mapping stopped by user.")
        print(f"Collected data for {scan_count} scenes.")
        print(f"Total collision tiles cached: {total_tiles}")
        print("Collision data saved to collision_cache/collision_map_debug.json")
        
        # Note: Run 'python visual_collision_map.py' manually to generate the map
        print("To generate the map, run: python visual_collision_map.py")
    except Exception as e:
        print(f"\nAn error occurred during mapping: {e}")
        import traceback
        traceback.print_exc()
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Route Mapper - Collect collision data while walking")
    parser.add_argument("--port", type=int, help="IPC port number (auto-detected if not specified)")
    parser.add_argument("--start-port", type=int, default=17000, help="Start port for auto-detection (default: 17000)")
    parser.add_argument("--end-port", type=int, default=17020, help="End port for auto-detection (default: 17020)")
    
    args = parser.parse_args()
    
    print("FIXED ROUTE MAPPER")
    print("=" * 50)
    
    try:
        success = fixed_route_mapper(port=args.port)
        if success:
            print("\n[SUCCESS] Route mapping completed successfully!")
        else:
            print("\n[ERROR] Route mapping failed!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
