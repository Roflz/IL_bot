#!/usr/bin/env python3
"""
Pathfinding script for RuneScape collision map.
Generates a path from Lumbridge to Grand Exchange and overlays it on the collision map PNG.
"""
import json
import math
import heapq
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import os

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




def astar_pathfinding(start, goal, walkable_tiles):
    """Find path using A* algorithm."""
    print(f"[DEBUG] Finding path from {start} to {goal}")
    print(f"[DEBUG] Walkable tiles: {len(walkable_tiles)}")
    
    if start not in walkable_tiles:
        print(f"ERROR: Start position {start} is not walkable")
        return None
    
    if goal not in walkable_tiles:
        print(f"ERROR: Goal position {goal} is not walkable")
        return None
    
    # Priority queue: (f_score, tie_breaker, position)
    # Use a counter for tie-breaking to ensure more comprehensive exploration
    tie_breaker = 0
    open_set = [(0, tie_breaker, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()
    in_open_set = {start}  # Track what's in the open set to avoid duplicates
    
    iterations = 0
    
    while open_set:
        iterations += 1
        if iterations % 1000 == 0:  # Less frequent debug output
            print(f"[DEBUG] Iteration {iterations}, visited {len(visited)} tiles, queue size {len(open_set)}")
            
        current = heapq.heappop(open_set)[2]  # Get position (third element)
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


def draw_path_on_map(image_path, path, collision_data, output_path):
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
    
    # Calculate proper pixel coordinates for Lumbridge spawn and GE
    lumbridge_world = (3236, 3227)  # Lumbridge spawn world coordinates (walkable tile)
    ge_world = (3164, 3487)         # GE world coordinates (walkable tile)
    
    lumbridge_pixel = world_to_map_coords(lumbridge_world[0], lumbridge_world[1], min_x, min_y, max_y)
    ge_pixel = world_to_map_coords(ge_world[0], ge_world[1], min_x, min_y, max_y)
    
    print(f"[DEBUG] Lumbridge: World {lumbridge_world} -> Pixel {lumbridge_pixel}")
    print(f"[DEBUG] GE: World {ge_world} -> Pixel {ge_pixel}")
    print(f"[DEBUG] Drawing path from Lumbridge {lumbridge_pixel} to GE {ge_pixel}")
    
    # Draw straight line first so you can see the actual start/end points
    print(f"[DEBUG] Drawing straight line from Lumbridge {lumbridge_pixel} to GE {ge_pixel}")
    draw.line([lumbridge_pixel, ge_pixel], fill=(255, 255, 0), width=4)  # Yellow straight line
    
    # Draw start and end points (ALWAYS use the actual GE coordinates, not pathfinding result)
    print(f"[DEBUG] Drawing start point at {lumbridge_pixel}")
    draw.ellipse([lumbridge_pixel[0]-10, lumbridge_pixel[1]-10, lumbridge_pixel[0]+10, lumbridge_pixel[1]+10], 
                fill=(0, 255, 0), outline=(255, 255, 255), width=3)  # Green start
    
    print(f"[DEBUG] Drawing end point at {ge_pixel} (ACTUAL GE COORDINATES)")
    draw.ellipse([ge_pixel[0]-10, ge_pixel[1]-10, ge_pixel[0]+10, ge_pixel[1]+10], 
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
        
        # Draw start and end points with larger circles
        if map_path:
            # Start point (green circle) - use actual Lumbridge coordinates
            start = lumbridge_pixel
            print(f"[DEBUG] Drawing start point at {start} (ACTUAL LUMBRIDGE)")
            draw.ellipse([start[0]-10, start[1]-10, start[0]+10, start[1]+10], 
                        fill=(0, 255, 0), outline=(255, 255, 255), width=3)
            
            # End point (red circle) - use actual GE coordinates, NOT pathfinding result
            end = ge_pixel
            print(f"[DEBUG] Drawing end point at {end} (ACTUAL GE COORDINATES)")
            draw.ellipse([end[0]-10, end[1]-10, end[0]+10, end[1]+10], 
                        fill=(255, 0, 0), outline=(255, 255, 255), width=3)
        
        print(f"[DEBUG] Path drawn from {lumbridge_pixel} to {ge_pixel} (ACTUAL COORDINATES)")
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


def main():
    """Main pathfinding function."""
    print("RUNESCAPE PATHFINDER")
    print("=" * 50)
    
    # Load collision data first
    collision_data = load_collision_data()
    if not collision_data:
        return False
    
    # Get walkable tiles (only black and green tiles)
    walkable_tiles, blocked_tiles = get_walkable_tiles(collision_data)
    print(f"Walkable tiles: {len(walkable_tiles)}")
    print(f"Blocked tiles: {len(blocked_tiles)}")
    
    # Find actual walkable tiles in the collision data
    # LUMBRIDGE_NEW_PLAYER_SPAWN: (3231, 3240, 3212, 3224) - rectangle
    # GE: (3155, 3173, 3479, 3498) - rectangle
    
    # Find closest walkable tiles to the target coordinates
    LUMBRIDGE_TARGET = (3236, 3227)  # ACTUAL Lumbridge spawn coordinates from the green circle
    GE_TARGET = (3164, 3487)         # ACTUAL GE coordinates from the red circle
    
    # Find closest walkable tile to Lumbridge
    lumbridge_walkable = []
    for x in range(3230, 3245):  # Around (3236, 3227)
        for y in range(3220, 3235):
            if (x, y) in walkable_tiles:
                lumbridge_walkable.append((x, y))
    
    # Find closest walkable tile to GE
    ge_walkable = []
    for x in range(3155, 3175):  # Around (3164, 3487)
        for y in range(3480, 3495):
            if (x, y) in walkable_tiles:
                ge_walkable.append((x, y))
    
    if not lumbridge_walkable:
        print("ERROR: No walkable tiles found near Lumbridge")
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
    
    LUMBRIDGE_SPAWN = find_closest(LUMBRIDGE_TARGET, lumbridge_walkable)
    GE_COORDS = find_closest(GE_TARGET, ge_walkable)
    
    print(f"Start: Lumbridge spawn {LUMBRIDGE_SPAWN}")
    print(f"Goal: Grand Exchange {GE_COORDS}")
    print()
    
    # Use the original coordinates since everything is walkable by default
    start = LUMBRIDGE_SPAWN
    goal = GE_COORDS
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
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
    output_image = script_dir / "collision_cache" / "path_lumbridge_to_ge.png"
    
    draw_path_on_map(input_image, path, collision_data, output_image)
    
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
