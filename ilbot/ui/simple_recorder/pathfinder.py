#!/usr/bin/env python3
"""
Pathfinding script for RuneScape collision map.
Generates a path from current player position to Grand Exchange and overlays it on the collision map PNG.
"""
import json
import math
import heapq
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import os

# Removed problematic import

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_collision_data():
    """Load collision data from cache."""
    script_dir = Path(__file__).parent
    cache_file = script_dir / "collision_cache" / "collision_map.json"
    print(f"[DEBUG] Loading collision data from: {cache_file.absolute()}")
    
    if not cache_file.exists():
        print("ERROR: No collision cache found. Run collision mapping first.")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded {len(data.get('collision_data', {}))} collision tiles")
        return data.get("collision_data", {})
    except Exception as e:
        print(f"ERROR: Could not load cache: {e}")
        return None


def get_walkable_tiles(collision_data):
    """Convert collision data to a set of walkable coordinates."""
    walkable = set()
    blocked = set()
    
    for tile_data in collision_data.values():
        x, y = tile_data['x'], tile_data['y']
        
        # Check if tile has a door object
        if tile_data.get('door'):
            door_info = tile_data.get('door', {})
            passable = door_info.get('passable', False)
            
            # Only walkable if the door is passable
            if passable:
                walkable.add((x, y))
            else:
                blocked.add((x, y))
        # For tiles without doors, use walkable flag
        elif tile_data.get('walkable', False):
            walkable.add((x, y))
        else:
            blocked.add((x, y))
    
    # Add all tiles that are NOT in collision_data as walkable (NO DATA = walkable)
    # We need to find the bounds of the collision data first
    if collision_data:
        all_x = [tile['x'] for tile in collision_data.values()]
        all_y = [tile['y'] for tile in collision_data.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add all tiles in the bounds that are not in collision_data as walkable
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) not in walkable and (x, y) not in blocked:
                    walkable.add((x, y))
    
    return walkable, blocked


def heuristic(a, b):
    """Calculate Euclidean distance between two points (for 8-directional movement)."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def get_neighbors(pos, walkable_tiles):
    """Get walkable neighbors of a position."""
    x, y = pos
    neighbors = []
    
    # 8-directional movement (including diagonals)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_pos = (x + dx, y + dy)
            # Only walkable tiles are allowed
            if new_pos in walkable_tiles:
                neighbors.append(new_pos)
    
    return neighbors




def find_closest_walkable_tile(goal, walkable_tiles, search_radius=50):
    """Find the closest walkable tile to the given goal position."""
    goal_x, goal_y = goal
    
    # Search in expanding squares around the goal
    for radius in range(1, search_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Only check tiles on the perimeter of the current square
                if abs(dx) == radius or abs(dy) == radius:
                    candidate = (goal_x + dx, goal_y + dy)
                    if candidate in walkable_tiles:
                        print(f"[DEBUG] Found closest walkable tile to {goal}: {candidate} (distance: {radius})")
                        return candidate
    
    print(f"[DEBUG] No walkable tile found within {search_radius} tiles of {goal}")
    return None


def astar_pathfinding(start, goal, walkable_tiles, max_iterations=20000):
    """Find path using A* algorithm with optimizations."""
    print(f"[DEBUG] Finding path from {start} to {goal}")
    print(f"[DEBUG] Walkable tiles: {len(walkable_tiles)}")
    
    if start not in walkable_tiles:
        print(f"ERROR: Start position {start} is not walkable")
        return None
    
    if goal not in walkable_tiles:
        print(f"[DEBUG] Goal position {goal} is not walkable, finding closest walkable tile...")
        closest_goal = find_closest_walkable_tile(goal, walkable_tiles)
        if closest_goal is None:
            print(f"ERROR: No walkable tile found near goal {goal}")
            return None
        goal = closest_goal
        print(f"[DEBUG] Using closest walkable goal: {goal}")
    
    # Early exit if start and goal are very close
    distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if distance <= 2:
        return [start, goal]
    
    # Priority queue: (f_score, tie_breaker, position)
    tie_breaker = 0
    open_set = [(0, tie_breaker, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()
    in_open_set = {start}
    
    iterations = 0
    max_queue_size = 1000  # Limit queue size for performance
    
    while open_set and iterations < max_iterations:
        iterations += 1
        if iterations % 2000 == 0:  # Less frequent debug output
            print(f"[DEBUG] Iteration {iterations}, visited {len(visited)} tiles, queue size {len(open_set)}")
            
        # Limit queue size to prevent memory issues
        if len(open_set) > max_queue_size:
            # Keep only the best candidates
            open_set = open_set[:max_queue_size]
            heapq.heapify(open_set)
            
        current = heapq.heappop(open_set)[2]
        in_open_set.discard(current)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print(f"[DEBUG] Path found with {len(path)} waypoints after {iterations} iterations")
            return path
        
        # Check if we're close enough to goal (within 5 tiles)
        if abs(current[0] - goal[0]) + abs(current[1] - goal[1]) <= 5:
            # Direct path to goal
            path = []
            temp_current = current
            while temp_current in came_from:
                path.append(temp_current)
                temp_current = came_from[temp_current]
            path.append(start)
            path.reverse()
            path.append(goal)
            print(f"[DEBUG] Early path found with {len(path)} waypoints after {iterations} iterations")
            return path
        
        for neighbor in get_neighbors(current, walkable_tiles):
            if neighbor in visited:
                continue
            
            # Calculate cost (diagonal movement costs more)
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])
            cost = 1.414 if dx == 1 and dy == 1 else 1.0
                
            tentative_g_score = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                # Only add to open set if not already there
                if neighbor not in in_open_set:
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score[neighbor], tie_breaker, neighbor))
                    in_open_set.add(neighbor)
    
    if iterations >= max_iterations:
        print(f"WARNING: Pathfinding stopped after {max_iterations} iterations")
        # Try to return a partial path to the closest point we found
        if visited:
            closest_to_goal = min(visited, key=lambda pos: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))
            path = []
            current = closest_to_goal
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print(f"[DEBUG] Partial path found with {len(path)} waypoints")
            return path
    
    print(f"ERROR: No path found after {iterations} iterations")
    print(f"Visited {len(visited)} tiles")
    return None


def find_closest_walkable(pos, walkable_tiles, max_distance=50):
    """Find the closest walkable tile to a given position."""
    x, y = pos
    best_pos = None
    best_distance = float('inf')
    
    for wx, wy in walkable_tiles:
        distance = math.sqrt((x - wx)**2 + (y - wy)**2)
        if distance < best_distance and distance <= max_distance:
            best_distance = distance
            best_pos = (wx, wy)
    
    return best_pos


def world_to_map_coords(world_x, world_y, min_x, min_y, max_y, tile_size=16, padding=20):
    """Convert world coordinates to map pixel coordinates."""
    # X coordinate: simple offset
    map_x = padding + (world_x - min_x) * tile_size
    
    # Y coordinate: inverted (like in visual_collision_map.py)
    map_height = (max_y - min_y + 1) * tile_size
    map_y = padding + map_height - ((world_y - min_y) * tile_size) - tile_size
    
    return int(map_x), int(map_y)


def draw_path_on_map(image_path, path, collision_data, output_path, start_pos=None, goal_pos=None):
    """Draw the path on the collision map image."""
    print(f"[DEBUG] Loading image from: {image_path}")
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    print(f"[DEBUG] Image size: {img.size}")
    
    # Get map bounds from collision data
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    print(f"[DEBUG] Map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    # Use provided positions or fallback to path start/end
    if start_pos:
        start_world = start_pos
    elif path and len(path) > 0:
        start_world = path[0]
    else:
        start_world = (3236, 3227)  # Fallback to Lumbridge
    
    if goal_pos:
        goal_world = goal_pos
    elif path and len(path) > 0:
        goal_world = path[-1]
    else:
        goal_world = (3164, 3487)  # Fallback to GE
    
    start_pixel = world_to_map_coords(start_world[0], start_world[1], min_x, min_y, max_y)
    goal_pixel = world_to_map_coords(goal_world[0], goal_world[1], min_x, min_y, max_y)
    
    print(f"[DEBUG] Start: World {start_world} -> Pixel {start_pixel}")
    print(f"[DEBUG] Goal: World {goal_world} -> Pixel {goal_pixel}")
    print(f"[DEBUG] Drawing path from {start_pixel} to {goal_pixel}")
    
    # Draw straight line first so you can see the actual start/end points
    print(f"[DEBUG] Drawing straight line from {start_pixel} to {goal_pixel}")
    draw.line([start_pixel, goal_pixel], fill=(255, 255, 0), width=4)  # Yellow straight line
    
    # Draw start and end points
    print(f"[DEBUG] Drawing start point at {start_pixel}")
    draw.ellipse([start_pixel[0]-10, start_pixel[1]-10, start_pixel[0]+10, start_pixel[1]+10], 
                fill=(0, 255, 0), outline=(255, 255, 255), width=3)  # Green start
    
    print(f"[DEBUG] Drawing end point at {goal_pixel}")
    draw.ellipse([goal_pixel[0]-10, goal_pixel[1]-10, goal_pixel[0]+10, goal_pixel[1]+10], 
                fill=(255, 0, 0), outline=(255, 255, 255), width=3)  # Red end
    
    # Draw path (if we have one)
    if path and len(path) > 1:
        print(f"[DEBUG] Drawing path with {len(path)} waypoints")
        print(f"[DEBUG] Path coordinates: {path[:5]}...{path[-5:] if len(path) > 5 else path}")
        
        # Convert path to map coordinates
        map_path = []
        for world_x, world_y in path:
            map_x, map_y = world_to_map_coords(world_x, world_y, min_x, min_y, max_y)
            map_path.append((map_x, map_y))
            print(f"[DEBUG] World ({world_x}, {world_y}) -> Map ({map_x}, {map_y})")
        
        print(f"[DEBUG] Map path coordinates: {map_path[:5]}...{map_path[-5:] if len(map_path) > 5 else map_path}")
        
        # Draw path lines with thicker lines
        for i in range(len(map_path) - 1):
            start = map_path[i]
            end = map_path[i + 1]
            print(f"[DEBUG] Drawing line from {start} to {end}")
            
            # Check if coordinates are within image bounds
            if (0 <= start[0] < img.size[0] and 0 <= start[1] < img.size[1] and
                0 <= end[0] < img.size[0] and 0 <= end[1] < img.size[1]):
                draw.line([start, end], fill=(0, 255, 255), width=6)  # Cyan path
                print(f"[DEBUG] Line drawn successfully")
            else:
                print(f"[WARNING] Line coordinates out of bounds! Image size: {img.size}")
                # Draw anyway but with a warning
                draw.line([start, end], fill=(0, 255, 255), width=6)
        
        print(f"[DEBUG] Path drawn from {start_pixel} to {goal_pixel}")
    else:
        print("WARNING: No path to draw")
    
    # Save the image
    print(f"[DEBUG] Saving image to: {output_path}")
    img.save(output_path)
    print(f"[SUCCESS] Path image saved to: {output_path}")


def simple_greedy_path(start, goal, walkable_tiles):
    """Simple greedy pathfinding that tries to get as close as possible to goal."""
    print(f"[DEBUG] Creating greedy path from {start} to {goal}")
    
    path = [start]
    current = start
    visited = set([start])
    
    max_steps = 5000  # Much higher limit
    step = 0
    
    while current != goal and step < max_steps:
        step += 1
        
        # Find the best next step towards goal
        best_next = None
        best_distance = float('inf')
        
        # Try all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                next_pos = (current[0] + dx, current[1] + dy)
                
                # Check if it's walkable and not visited
                if next_pos in walkable_tiles and next_pos not in visited:
                    # Calculate distance to goal
                    distance = ((goal[0] - next_pos[0])**2 + (goal[1] - next_pos[1])**2)**0.5
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_next = next_pos
        
        if best_next is None:
            # If no direct neighbors, try to find any nearby walkable tile
            print(f"[WARNING] No direct neighbors at {current}, searching for nearby tiles...")
            min_dist = float('inf')
            for tile in walkable_tiles:
                if tile not in visited:
                    dist = ((current[0] - tile[0])**2 + (current[1] - tile[1])**2)**0.5
                    if dist < min_dist and dist < 10:  # Within 10 tiles
                        min_dist = dist
                        best_next = tile
            
            if best_next is None:
                print(f"[ERROR] No nearby walkable tiles found at {current}")
                break
            
        current = best_next
        path.append(current)
        visited.add(current)
        
        if step % 100 == 0:
            print(f"[DEBUG] Step {step}, current: {current}, distance to goal: {best_distance}")
    
    print(f"[DEBUG] Greedy path created with {len(path)} waypoints")
    return path


def simple_path(start, goal, blocked_tiles):
    """Create a simple straight-line path, avoiding unwalkable areas."""
    print(f"[DEBUG] Creating simple path from {start} to {goal}")
    
    path = [start]
    current = start
    
    # Try to move towards goal step by step
    max_steps = 1000
    step = 0
    
    while current != goal and step < max_steps:
        step += 1
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        
        # Normalize direction
        if dx != 0:
            dx = 1 if dx > 0 else -1
        if dy != 0:
            dy = 1 if dy > 0 else -1
        
        # Try to move in the direction of the goal
        next_pos = (current[0] + dx, current[1] + dy)
        
        if next_pos not in blocked_tiles:
            current = next_pos
            path.append(current)
        else:
            # Try alternative directions if direct path is blocked
            alternatives = [
                (current[0] + dx, current[1]),  # Move only X
                (current[0], current[1] + dy),  # Move only Y
                (current[0] + 1, current[1]),   # Move right
                (current[0] - 1, current[1]),   # Move left
                (current[0], current[1] + 1),   # Move up
                (current[0], current[1] - 1),   # Move down
            ]
            
            found_alternative = False
            for alt in alternatives:
                if alt not in blocked_tiles and alt not in path:
                    current = alt
                    path.append(current)
                    found_alternative = True
                    break
            
            if not found_alternative:
                print(f"[WARNING] Stuck at {current}, cannot find path to goal")
                break
    
    print(f"[DEBUG] Simple path created with {len(path)} waypoints")
    return path


def get_current_player_position():
    """Get current player position using IPC."""
    print("[DEBUG] Getting current player position...")
    
    try:
        # Import the necessary modules
        from ilbot.ui.simple_recorder.services.ipc_client import RuneLiteIPC
        from ilbot.ui.simple_recorder.helpers.context import set_payload
        from ilbot.ui.simple_recorder.actions.player import get_x, get_y
        
        # Create IPC connection (try common ports)
        ipc = None
        for port in range(17000, 17021):  # Try ports 17000-17020
            try:
                print(f"[DEBUG] Trying IPC connection on port {port}...")
                ipc = RuneLiteIPC(port=port, pre_action_ms=120, timeout_s=0.5)  # Shorter timeout
                # Test the connection
                test_response = ipc._send({"cmd": "get_player"})
                if test_response and test_response.get("ok"):
                    print(f"[DEBUG] Successfully connected to RuneLite on port {port}")
                    break
                else:
                    ipc = None
            except Exception as e:
                # Don't print every failed port to avoid spam
                if port % 5 == 0:  # Print every 5th port
                    print(f"[DEBUG] Port {port} failed: {e}")
                ipc = None
                continue
        
        if not ipc:
            print("[ERROR] Could not connect to RuneLite on any port")
            return None
        
        # Create a payload with the IPC connection
        payload = {"__ipc": ipc, "__ipc_port": ipc.port}
        set_payload(payload)
        
        # Get player position using the IPC functions
        x = get_x(payload)
        y = get_y(payload)
        
        if x is not None and y is not None:
            position = (x, y)
            print(f"[DEBUG] Got player position: {position}")
            return position
        else:
            print("[ERROR] Could not get valid player coordinates")
            return None
            
    except Exception as e:
        print(f"[ERROR] Exception getting player position: {e}")
        return None


def main():
    """Main pathfinding function."""
    print("RUNESCAPE PATHFINDER - CURRENT POSITION TO GE")
    print("=" * 50)
    
    # Get current player position
    current_pos = get_current_player_position()
    if not current_pos:
        print("ERROR: Could not get current player position")
        return False
    
    # Load collision data first
    collision_data = load_collision_data()
    if not collision_data:
        return False
    
    # Get walkable tiles (only black and green tiles)
    walkable_tiles, blocked_tiles = get_walkable_tiles(collision_data)
    print(f"Walkable tiles: {len(walkable_tiles)}")
    print(f"Blocked tiles: {len(blocked_tiles)}")
    
    # GE coordinates from constants
    GE_TARGET = (3164, 3487)  # GE coordinates
    
    # Find closest walkable tile to current position
    current_walkable = []
    for x in range(current_pos[0] - 10, current_pos[0] + 11):  # Search around current position
        for y in range(current_pos[1] - 10, current_pos[1] + 11):
            if (x, y) in walkable_tiles:
                current_walkable.append((x, y))
    
    # Find closest walkable tile to GE
    ge_walkable = []
    for x in range(3155, 3175):  # Around GE
        for y in range(3480, 3495):
            if (x, y) in walkable_tiles:
                ge_walkable.append((x, y))
    
    if not current_walkable:
        print("ERROR: No walkable tiles found near current position")
        return False
    
    if not ge_walkable:
        print("ERROR: No walkable tiles found near GE")
        return False
    
    # Find closest tiles
    def find_closest(target, candidates):
        min_dist = float('inf')
        closest = None
        for candidate in candidates:
            dist = ((target[0] - candidate[0])**2 + (target[1] - candidate[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest = candidate
        return closest
    
    start = find_closest(current_pos, current_walkable)
    goal = find_closest(GE_TARGET, ge_walkable)
    
    print(f"Current position: {current_pos}")
    print(f"Start (closest walkable): {start}")
    print(f"Goal (GE): {goal}")
    print()
    
    # Calculate and show pixel coordinates
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    start_pixel = world_to_map_coords(start[0], start[1], min_x, min_y, max_y)
    goal_pixel = world_to_map_coords(goal[0], goal[1], min_x, min_y, max_y)
    
    print(f"Start pixel coordinates: {start_pixel}")
    print(f"Goal pixel coordinates: {goal_pixel}")
    print()
    
    # Find path using only walkable tiles (black and green)
    print("Finding path using only walkable tiles (black and green)...")
    path = astar_pathfinding(start, goal, walkable_tiles)
    
    if not path:
        print("A* failed, trying simple greedy pathfinding...")
        path = simple_greedy_path(start, goal, walkable_tiles)
    
    if not path:
        print("All pathfinding failed, falling back to straight line...")
        path = [start, goal]  # Fallback to straight line
    
    print(f"Path found with {len(path)} waypoints")
    print(f"Path length: {len(path)} tiles")
    print()
    
    # Draw path on map
    script_dir = Path(__file__).parent
    input_image = script_dir / "collision_cache" / "detailed_collision_map.png"
    output_image = script_dir / "collision_cache" / "path_current_to_ge.png"
    
    draw_path_on_map(input_image, path, collision_data, output_image, start_pos=current_pos, goal_pos=GE_TARGET)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n[SUCCESS] Pathfinding completed successfully!")
        else:
            print("\n[ERROR] Pathfinding failed!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
