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

from ilbot.ui.simple_recorder.actions.player import get_player_position

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
    """Convert collision data to a set of walkable coordinates with wall orientation handling."""
    walkable = set()
    blocked = set()
    wall_orientations = {}  # Store wall orientation data for directional blocking
    
    for tile_data in collision_data.values():
        x, y = tile_data['x'], tile_data['y']
        
        # Check if tile has a door object
        if tile_data.get('door'):
            door_info = tile_data.get('door', {})
            door_id = door_info.get('id')
            passable = door_info.get('passable', False)
            
            # Special case: Bank doors and specific walkable walls are always passable
            if door_id in [11787, 11786, 23751, 23752, 23750]:
                walkable.add((x, y))
            # Only walkable if the door is passable
            elif passable:
                walkable.add((x, y))
            else:
                blocked.add((x, y))
        # Check for wall orientation data
        elif tile_data.get('wall_orientation'):
            # Wall tiles WITH orientation data are WALKABLE
            # The orientation determines which directions are blocked
            walkable.add((x, y))
            wall_orientations[(x, y)] = tile_data['wall_orientation']
            print(f"[DEBUG] Wall tile with orientation at ({x}, {y}): {tile_data['wall_orientation']}")
        # For tiles without doors or wall orientation, use walkable flag
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
    
    return walkable, blocked, wall_orientations


def heuristic(a, b):
    """Calculate Euclidean distance between two points (for 8-directional movement)."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def get_neighbors(pos, walkable_tiles, wall_orientations=None):
    """Get walkable neighbors of a position, respecting wall orientation blocking."""
    x, y = pos
    neighbors = []
    
    if wall_orientations is None:
        wall_orientations = {}
    
    # 8-directional movement (including diagonals)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_pos = (x + dx, y + dy)
            
            # Only walkable tiles are allowed
            if new_pos in walkable_tiles:
                # Check wall orientation blocking
                if not is_movement_blocked(pos, new_pos, wall_orientations):
                    # For diagonal movement, check corner-cutting prevention
                    if dx != 0 and dy != 0:  # Diagonal movement
                        # Check that both adjacent tiles are walkable to prevent corner-cutting
                        adj1 = (x + dx, y)  # Horizontal adjacent
                        adj2 = (x, y + dy)  # Vertical adjacent
                        
                        if (adj1 in walkable_tiles and adj2 in walkable_tiles and
                            not is_movement_blocked(pos, adj1, wall_orientations) and
                            not is_movement_blocked(pos, adj2, wall_orientations)):
                            neighbors.append(new_pos)
                        # If either adjacent tile is blocked, skip diagonal movement
                    else:
                        # Non-diagonal movement (horizontal/vertical) is allowed if not blocked by walls
                        neighbors.append(new_pos)
    
    return neighbors


def is_movement_blocked(from_pos, to_pos, wall_orientations):
    """Check if movement from one position to another is blocked by wall orientation."""
    from_x, from_y = from_pos
    to_x, to_y = to_pos
    
    # Calculate direction of movement
    dx = to_x - from_x
    dy = to_y - from_y
    
    # Check if the source tile has wall orientation that blocks this direction
    if from_pos in wall_orientations:
        orientation = wall_orientations[from_pos]
        if is_direction_blocked_by_orientation(dx, dy, orientation):
            print(f"[DEBUG] Movement from {from_pos} to {to_pos} blocked by wall orientation: {orientation}")
            return True
    
    # Check if the destination tile has wall orientation that blocks entry from this direction
    if to_pos in wall_orientations:
        orientation = wall_orientations[to_pos]
        # For entry blocking, we need to check if the wall blocks movement FROM the opposite direction
        if is_direction_blocked_by_orientation(-dx, -dy, orientation):
            print(f"[DEBUG] Movement from {from_pos} to {to_pos} blocked by destination wall orientation: {orientation}")
            return True
    
    return False


def is_direction_blocked_by_orientation(dx, dy, orientation):
    """Check if a movement direction is blocked by wall orientation data."""
    # Wall orientation typically indicates which sides have walls
    # This is a simplified interpretation - you may need to adjust based on your actual orientation data format
    
    # Common wall orientation values and their meanings:
    # 1 = North wall, 2 = East wall, 4 = South wall, 8 = West wall
    # Combinations: 3 = North+East, 5 = North+South, etc.
    
    if orientation is None:
        return False
    
    # Convert direction to orientation bit
    direction_bit = 0
    if dy == -1:  # Moving north
        direction_bit = 1
    elif dx == 1:  # Moving east
        direction_bit = 2
    elif dy == 1:  # Moving south
        direction_bit = 4
    elif dx == -1:  # Moving west
        direction_bit = 8
    
    # Check if the orientation has a wall in this direction
    if isinstance(orientation, (int, float)):
        return bool(int(orientation) & direction_bit)
    elif isinstance(orientation, dict):
        # Handle dictionary format if orientation is more complex
        return orientation.get('blocked_directions', 0) & direction_bit
    else:
        # Handle other formats as needed
        return False




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


def astar_pathfinding(start, goal, walkable_tiles, wall_orientations=None, max_iterations=20000):
    """Find path using A* algorithm with wall orientation support."""
    print(f"[DEBUG] Finding path from {start} to {goal}")
    print(f"[DEBUG] Walkable tiles: {len(walkable_tiles)}")
    print(f"[DEBUG] Wall orientations: {len(wall_orientations) if wall_orientations else 0}")
    
    if wall_orientations is None:
        wall_orientations = {}
    
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
    # Use deterministic tie-breaking for path consistency
    tie_breaker = 0
    open_set = [(0, tie_breaker, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()
    in_open_set = {start}
    
    iterations = 0
    max_queue_size = 1000  # Limit queue size for performance
    
    # Path consistency: prefer paths that go in a consistent direction
    def get_direction_consistency_score(pos):
        """Calculate a score based on how consistent the path direction is."""
        if pos == start:
            return 0
        
        # Calculate direction from start to current position
        dx = pos[0] - start[0]
        dy = pos[1] - start[1]
        
        # Calculate direction from current position to goal
        gx = goal[0] - pos[0]
        gy = goal[1] - pos[1]
        
        # Dot product to measure direction consistency
        if dx != 0 or dy != 0:
            dot_product = dx * gx + dy * gy
            start_length = (dx**2 + dy**2)**0.5
            goal_length = (gx**2 + gy**2)**0.5
            if start_length > 0 and goal_length > 0:
                return (dot_product / (start_length * goal_length)) * 2  # Small bonus for consistency
        return 0
    
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
        
        # Check if we're close enough to goal (within 2 tiles)
        if abs(current[0] - goal[0]) + abs(current[1] - goal[1]) <= 2:
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
        
        for neighbor in get_neighbors(current, walkable_tiles, wall_orientations):
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
                
                # Use direction consistency to break ties and prefer consistent paths
                direction_bonus = get_direction_consistency_score(neighbor)
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal) - direction_bonus
                
                # Only add to open set if not already there
                if neighbor not in in_open_set:
                    tie_breaker += 1
                    # Use deterministic tie-breaking based on position for consistency
                    position_tie_breaker = neighbor[0] * 1000 + neighbor[1]
                    heapq.heappush(open_set, (f_score[neighbor], position_tie_breaker, neighbor))
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


def simple_greedy_path(start, goal, walkable_tiles, wall_orientations=None):
    """Simple greedy pathfinding that tries to get as close as possible to goal."""
    print(f"[DEBUG] Creating greedy path from {start} to {goal}")
    
    if wall_orientations is None:
        wall_orientations = {}
    
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
                
                # Check if it's walkable, not visited, and not blocked by walls
                if (next_pos in walkable_tiles and next_pos not in visited and
                    not is_movement_blocked(current, next_pos, wall_orientations)):
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
                if tile not in visited and not is_movement_blocked(current, tile, wall_orientations):
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

def main(destination_x=None, destination_y=None):
    """Main pathfinding function."""
    print("RUNESCAPE PATHFINDER - CURRENT POSITION TO DESTINATION")
    print("=" * 50)
    print("Now with WALL ORIENTATION support!")
    print("- Wall tiles WITH orientation data are WALKABLE")
    print("- Wall tiles WITHOUT orientation data are BLOCKING")
    print("- Wall orientation determines directional movement restrictions")
    print("=" * 50)
    
    # Get current player position
    current_pos = get_player_position()
    if not current_pos:
        print("ERROR: Could not get current player position")
        return False
    
    # Load collision data first
    collision_data = load_collision_data()
    if not collision_data:
        return False
    
    # Get walkable tiles with wall orientation support
    walkable_tiles, blocked_tiles, wall_orientations = get_walkable_tiles(collision_data)
    print(f"Walkable tiles: {len(walkable_tiles)}")
    print(f"Blocked tiles: {len(blocked_tiles)}")
    print(f"Wall orientations: {len(wall_orientations)}")
    
    # Set destination - use provided coordinates or default to GE
    if destination_x is not None and destination_y is not None:
        DESTINATION_TARGET = (destination_x, destination_y)
        print(f"Using provided destination: {DESTINATION_TARGET}")
    else:
        DESTINATION_TARGET = (3164, 3487)  # Default to GE coordinates
        print(f"Using default destination (GE): {DESTINATION_TARGET}")
    
    # Find closest walkable tile to current position
    current_walkable = []
    for x in range(current_pos[0] - 10, current_pos[0] + 11):  # Search around current position
        for y in range(current_pos[1] - 10, current_pos[1] + 11):
            if (x, y) in walkable_tiles:
                current_walkable.append((x, y))
    
    # Find closest walkable tile to destination
    dest_walkable = []
    for x in range(DESTINATION_TARGET[0] - 10, DESTINATION_TARGET[0] + 11):  # Search around destination
        for y in range(DESTINATION_TARGET[1] - 10, DESTINATION_TARGET[1] + 11):
            if (x, y) in walkable_tiles:
                dest_walkable.append((x, y))
    
    if not current_walkable:
        print("ERROR: No walkable tiles found near current position")
        return False
    
    if not dest_walkable:
        print("ERROR: No walkable tiles found near destination")
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
    goal = find_closest(DESTINATION_TARGET, dest_walkable)
    
    print(f"Current position: {current_pos}")
    print(f"Start (closest walkable): {start}")
    print(f"Goal (destination): {goal}")
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
    
    # Find path using walkable tiles with wall orientation support
    print("Finding path using walkable tiles with wall orientation support...")
    path = astar_pathfinding(start, goal, walkable_tiles, wall_orientations)
    
    if not path:
        print("A* failed, trying simple greedy pathfinding...")
        path = simple_greedy_path(start, goal, walkable_tiles, wall_orientations)
    
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
    
    draw_path_on_map(input_image, path, collision_data, output_image, start_pos=current_pos, goal_pos=DESTINATION_TARGET)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    destination_x = None
    destination_y = None
    port = None
    
    if len(sys.argv) >= 3:
        try:
            destination_x = int(sys.argv[1])
            destination_y = int(sys.argv[2])
            print(f"Using command line destination: ({destination_x}, {destination_y})")
            
            # Check for port argument
            if len(sys.argv) >= 4:
                port = int(sys.argv[3])
                print(f"Using specified port: {port}")
        except ValueError:
            print("ERROR: Invalid coordinates provided. Usage: python pathfinder.py [x] [y] [port]")
            print("Examples:")
            print("  python pathfinder.py 3164 3487 17000  (Grand Exchange)")
            print("  python pathfinder.py 3200 3200 17000  (Lumbridge area - test wall navigation)")
            print("  python pathfinder.py 3100 3500 17000  (Falador area - test complex walls)")
            print("  python pathfinder.py 3000 3000 17000  (Edgeville area - test different wall types)")
            sys.exit(1)
    elif len(sys.argv) > 1:
        print("ERROR: Please provide both X and Y coordinates. Usage: python pathfinder.py [x] [y] [port]")
        print("Examples:")
        print("  python pathfinder.py 3164 3487 17000  (Grand Exchange)")
        print("  python pathfinder.py 3200 3200 17000  (Lumbridge area - test wall navigation)")
        print("  python pathfinder.py 3100 3500 17000  (Falador area - test complex walls)")
        print("  python pathfinder.py 3000 3000 17000  (Edgeville area - test different wall types)")
        sys.exit(1)
    else:
        print("No destination provided, using default (Grand Exchange)")
        print("Try these test destinations to see wall orientation differences:")
        print("  Lumbridge: 3200 3200")
        print("  Falador: 3100 3500") 
        print("  Edgeville: 3000 3000")
    
    try:
        success = main(destination_x, destination_y)
        if success:
            print("\n[SUCCESS] Pathfinding completed successfully!")
        else:
            print("\n[ERROR] Pathfinding failed!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
