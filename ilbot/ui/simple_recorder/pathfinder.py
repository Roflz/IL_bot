#!/usr/bin/env python3
"""
Pathfinding script for RuneScape collision map.
Generates a path from current player position to Grand Exchange and overlays it on the collision map PNG.
"""
import json
import math
import heapq
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import os

from ilbot.ui.simple_recorder.actions.player import get_player_position
from ilbot.ui.simple_recorder.helpers.ipc import IPCClient

# Removed problematic import

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_collision_data(start_coords=None, dest_coords=None, buffer=100):
    """Load collision data from cache, filtered by start/destination bounds."""
    script_dir = Path(__file__).parent
    cache_file = script_dir / "collision_cache" / "collision_map.json"
    print(f"[DEBUG] Loading collision data from: {cache_file.absolute()}")
    
    if not cache_file.exists():
        print("ERROR: No collision cache found. Run collision mapping first.")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        all_collision_data = data.get("collision_data", {})
        print(f"[DEBUG] Total collision tiles in cache: {len(all_collision_data)}")
        
        # If no start/dest coordinates provided, return all data
        if not start_coords or not dest_coords:
            print(f"[DEBUG] No start/dest coordinates provided, returning all collision data")
            return all_collision_data
        
        # Calculate bounds based on start and destination coordinates
        start_x, start_y = start_coords
        dest_x, dest_y = dest_coords
        
        min_x = min(start_x, dest_x) - buffer
        max_x = max(start_x, dest_x) + buffer
        min_y = min(start_y, dest_y) - buffer
        max_y = max(start_y, dest_y) + buffer
        
        print(f"[DEBUG] Filtering collision data to bounds: X({min_x} to {max_x}), Y({min_y} to {max_y})")
        
        # Filter collision data to only include tiles within bounds
        filtered_data = {}
        for tile_id, tile_data in all_collision_data.items():
            x, y = tile_data['x'], tile_data['y']
            if min_x <= x <= max_x and min_y <= y <= max_y:
                filtered_data[tile_id] = tile_data
        
        print(f"[DEBUG] Filtered collision data: {len(filtered_data)} tiles (from {len(all_collision_data)} total)")
        return filtered_data
        
    except Exception as e:
        print(f"ERROR: Could not load cache: {e}")
        return None


def get_walkable_tiles(collision_data):
    """Convert collision data to a set of walkable coordinates with wall orientation handling."""
    walkable = set()
    blocked = set()
    wall_masks = {}  # Store normalized orientation masks for drawing/debug
    orientation_blockers = set()  # Only tiles that are non-walkable or doors with passable == False
    
    for tile_data in collision_data.values():
        x, y = tile_data['x'], tile_data['y']
        
        # Check if tile has a door object
        if tile_data.get('door'):
            door_info = tile_data.get('door', {})
            door_id = door_info.get('id')
            passable = door_info.get('passable', False)
            
            # Extract wall orientation data from door object
            orientation_a = door_info.get('orientationA')
            orientation_b = door_info.get('orientationB')
            
            # Build normalized mask for any tile with orientation
            if orientation_a is not None or orientation_b is not None:
                mask = build_orientation_mask({'orientationA': orientation_a, 'orientationB': orientation_b})
                wall_masks[(x, y)] = mask
                print(f"[DEBUG] Door tile with orientation at ({x}, {y}): A={orientation_a}, B={orientation_b}, mask={mask}")
                # Tiles with orientation data are ALWAYS walkable - the orientation determines which directions are blocked
                walkable.add((x, y))
                # Add to orientation_blockers so we check wall blocking for this tile
                orientation_blockers.add((x, y))
            # Special case: Bank doors and specific walkable walls are always passable
            elif door_id in [11787, 11786, 23751, 23752, 23750]:
                walkable.add((x, y))
            # Only walkable if the door is passable
            elif passable:
                walkable.add((x, y))
            else:
                blocked.add((x, y))
        # Check for wall orientation data at top level (legacy support)
        elif tile_data.get('wall_orientation'):
            # Wall tiles WITH orientation data are WALKABLE
            # The orientation determines which directions are blocked
            walkable.add((x, y))
            mask = build_orientation_mask(tile_data['wall_orientation'])
            wall_masks[(x, y)] = mask
            print(f"[DEBUG] Wall tile with orientation at ({x}, {y}): {tile_data['wall_orientation']}, mask={mask}")
        # For tiles without doors or wall orientation, use walkable flag
        elif tile_data.get('walkable', False):
            walkable.add((x, y))
        else:
            blocked.add((x, y))
            # Add to orientation_blockers only if non-walkable
            orientation_blockers.add((x, y))
    
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
    
    print(f"[DEBUG] Loaded {len(wall_masks)} wall masks and {len(orientation_blockers)} orientation blockers")
    return walkable, blocked, wall_masks, orientation_blockers


def heuristic(a, b):
    """Calculate Euclidean distance between two points (for 8-directional movement)."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


# Wall orientation bit mapping constants
W, N, E, S = 1, 2, 4, 8
NW, NE, SE, SW = 16, 32, 64, 128


def build_orientation_mask(orient) -> int:
    """Build normalized orientation mask from orientation data."""
    if orient is None:
        return 0
    if isinstance(orient, int):
        return int(orient)
    a = int((orient.get('orientationA') or 0))
    b = int((orient.get('orientationB') or 0))
    return a | b


def exit_blocked_by_mask(dx: int, dy: int, m: int, use_diag_bars: bool = True) -> bool:
    """Check if movement is blocked by source tile orientation mask."""
    # cardinals (source side) - wall blocks exit in opposite direction
    if dx == 0 and dy == -1 and (m & S): return True  # Moving up blocked by south wall
    if dx == 1 and dy == 0  and (m & W): return True  # Moving right blocked by west wall
    if dx == 0 and dy == 1  and (m & N): return True  # Moving down blocked by north wall
    if dx == -1 and dy == 0 and (m & E): return True  # Moving left blocked by east wall
    if not use_diag_bars: return False
    # diagonals: only the corner bar (faces handled via two-leg rule)
    if dx == 1 and dy == -1 and (m & SW): return True  # Moving NE blocked by SW wall
    if dx == 1 and dy == 1  and (m & NW): return True  # Moving SE blocked by NW wall
    if dx == -1 and dy == -1 and (m & SE): return True # Moving NW blocked by SE wall
    if dx == -1 and dy == 1 and (m & NE): return True  # Moving SW blocked by NE wall
    return False


def entry_blocked_by_mask(dx: int, dy: int, m: int, use_diag_bars: bool = True) -> bool:
    """Check if movement is blocked by destination tile orientation mask."""
    # destination side - wall blocks entry from opposite direction
    if dx == 0 and dy == -1 and (m & N): return True  # Moving up blocked by north wall
    if dx == 1 and dy == 0  and (m & E): return True  # Moving right blocked by east wall
    if dx == 0 and dy == 1  and (m & S): return True  # Moving down blocked by south wall
    if dx == -1 and dy == 0 and (m & W): return True  # Moving left blocked by west wall
    if not use_diag_bars: return False
    # opposite corner bar on destination
    if dx == 1 and dy == -1 and (m & NE): return True  # Moving NE blocked by NE wall
    if dx == 1 and dy == 1  and (m & SE): return True  # Moving SE blocked by SE wall
    if dx == -1 and dy == -1 and (m & NW): return True # Moving NW blocked by NW wall
    if dx == -1 and dy == 1 and (m & SW): return True  # Moving SW blocked by SW wall
    return False


def is_movement_blocked(a, b, wall_masks, orientation_blockers, use_diag_bars: bool = True) -> bool:
    """Check if movement from a to b is blocked by wall orientation (gated by blocker set)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    
    # Check source tile (exit blocking)
    if a in orientation_blockers and exit_blocked_by_mask(dx, dy, wall_masks.get(a, 0), use_diag_bars):
        return True
    
    # Check destination tile (entry blocking)
    if b in orientation_blockers and entry_blocked_by_mask(dx, dy, wall_masks.get(b, 0), use_diag_bars):
        return True
    
    return False
    

def get_neighbors(pos, walkable_tiles, wall_masks, orientation_blockers):
    """Get walkable neighbors of a position, respecting wall orientation blocking."""
    x, y = pos
    out = []
    
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: 
                continue
            nxt = (x + dx, y + dy)
            if nxt not in walkable_tiles:
                continue
            if is_movement_blocked(pos, nxt, wall_masks, orientation_blockers):
                continue
            if dx != 0 and dy != 0:
                # Diagonal move: require both orthogonal legs to be possible
                a1 = (x + dx, y)      # horizontal leg
                a2 = (x, y + dy)      # vertical leg
                if (a1 in walkable_tiles and a2 in walkable_tiles and
                    not is_movement_blocked(pos, a1, wall_masks, orientation_blockers) and
                    not is_movement_blocked(a1, nxt, wall_masks, orientation_blockers) and
                    not is_movement_blocked(pos, a2, wall_masks, orientation_blockers) and
                    not is_movement_blocked(a2, nxt, wall_masks, orientation_blockers)):
                    out.append(nxt)
            else:
                # Orthogonal move allowed
                out.append(nxt)
    
    return out




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


def astar_pathfinding(start, goal, walkable_tiles, max_iterations=50000):
    """Find path using A* algorithm with wall orientation support."""
    print(f"[DEBUG] Finding path from {start} to {goal}")
    print(f"[DEBUG] Walkable tiles: {len(walkable_tiles)}")
    
    # Always load wall orientations automatically, filtered by start/goal coordinates
    print(f"[DEBUG] Loading wall orientations automatically...")
    collision_data = load_collision_data(start, goal)
    if collision_data:
        _, _, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
        print(f"[DEBUG] Loaded {len(wall_masks)} wall masks and {len(orientation_blockers)} orientation blockers")
    else:
        wall_masks = {}
        orientation_blockers = set()
        print(f"[DEBUG] Could not load collision data, using empty wall data")
    
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
    
    # Removed direction consistency function - using pure A* now
    
    while open_set and iterations < max_iterations:
        iterations += 1
        if iterations % 2000 == 0:  # Less frequent debug output
            print(f"[DEBUG] Iteration {iterations}, visited {len(visited)} tiles, queue size {len(open_set)}")
            
    # Don't trim the heap - this breaks A* optimality
    # if len(open_set) > max_queue_size:
    #     # Keep only the best candidates
    #     open_set = open_set[:max_queue_size]
    #     heapq.heapify(open_set)
            
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
        
        for neighbor in get_neighbors(current, walkable_tiles, wall_masks, orientation_blockers):
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
                
                # Use pure A*: f = g + h (no direction bonus)
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                # Always push improved nodes (lazy deletion of stale entries)
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


def draw_wall_lines(draw, x, y, tile_size, orientation_a, orientation_b):
    """Draw wall lines based on orientation."""
    # Wall orientation values:
    # 1 = West, 2 = North, 4 = East, 8 = South
    # 16 = North-west, 32 = North-east, 64 = South-east, 128 = South-west
    
    line_color = (255, 255, 255)  # White for wall lines
    line_width = 2  # Thinner lines for better visibility
    
    # Draw orientation A
    if orientation_a & 1:  # West
        draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)
    if orientation_a & 2:  # North
        draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)
    if orientation_a & 4:  # East
        draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    if orientation_a & 8:  # South
        draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    
    # Draw orientation B
    if orientation_b & 1:  # West
        draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)
    if orientation_b & 2:  # North
        draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)
    if orientation_b & 4:  # East
        draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    if orientation_b & 8:  # South
        draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)


def world_to_map_coords(world_x, world_y, min_x, min_y, max_y, tile_size=16, padding=20):
    """Convert world coordinates to map pixel coordinates."""
    # X coordinate: simple offset
    map_x = padding + (world_x - min_x) * tile_size
    
    # Y coordinate: inverted (like in visual_collision_map.py)
    map_height = (max_y - min_y + 1) * tile_size
    map_y = padding + map_height - ((world_y - min_y) * tile_size) - tile_size
    
    return int(map_x), int(map_y)


def draw_path_on_map(image_path, path, collision_data, output_path, start_pos=None, goal_pos=None):
    """Generate a filtered collision map image and draw the path on it."""
    print(f"[DEBUG] Generating filtered collision map from collision data")
    
    # Get map bounds from collision data (this is already filtered)
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1
    
    print(f"[DEBUG] Filtered map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"[DEBUG] Filtered map size: {width_tiles}x{height_tiles} tiles")
    
    # Create a new image for the filtered area
    tile_size = 16
    map_width = width_tiles * tile_size
    map_height = height_tiles * tile_size
    
    # Add padding and legend space
    padding = 20
    legend_width = 200
    img_width = map_width + legend_width + (padding * 3)
    img_height = max(map_height, 300) + (padding * 2)
    
    print(f"[DEBUG] Creating image: {img_width}x{img_height} pixels")
    
    # Create the image
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        small_font = ImageFont.truetype("arial.ttf", 8)
        legend_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        small_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw collision tiles
    map_start_x = padding
    map_start_y = padding
    
    # Collect wall tiles for orientation lines
    wall_tiles = []
    
    tiles_drawn = 0
    for key, tile_data in collision_data.items():
        x, y, p = tile_data['x'], tile_data['y'], tile_data['p']
        
        # Calculate position on image
        draw_x = map_start_x + (x - min_x) * tile_size
        draw_y = map_start_y + map_height - ((y - min_y) * tile_size) - tile_size  # Invert Y
        
        # Determine color and symbol
        color = (0, 0, 0)  # Default black
        symbol = ""
        
        if tile_data.get('door'):
            # Handle door/wall tiles
            door_info = tile_data.get('door', {})
            door_id = door_info.get('id')
            orientation_a = door_info.get('orientationA', 0)
            orientation_b = door_info.get('orientationB', 0)
            passable = door_info.get('passable', False)
            
            # Special case: Bank doors and specific walkable walls are always walkable
            if door_id in [11787, 11786, 23751, 23752, 23750]:
                color = (0, 255, 0)  # Green background
                symbol = "P"  # P for passable
            # Check if this wall has orientation data (standable)
            elif orientation_a > 0 or orientation_b > 0:
                color = (0, 0, 0)  # Black background
                symbol = "W"
                # Store for later drawing of orientation lines
                wall_tiles.append((draw_x, draw_y, tile_size, orientation_a, orientation_b))
            else:
                color = (128, 128, 128)  # Gray background
                symbol = "W"
        elif tile_data.get('solid', False):
            color = (255, 0, 0)    # Red
            symbol = "#"
        elif tile_data.get('object', False):
            color = (0, 0, 255)    # Blue
            symbol = "O"
        elif tile_data.get('walkable', False):
            color = (0, 255, 0)    # Green
            symbol = "."
        else:
            color = (0, 0, 0)  # Black for no data
            symbol = "?"
        
        # Draw tile background
        draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=color)
        
        # Add symbol
        if symbol:
            draw.text((draw_x + 1, draw_y + 1), symbol, fill=(255, 255, 255), font=small_font)
        
        tiles_drawn += 1
    
    # Draw wall orientation lines
    print(f"[DEBUG] Drawing {len(wall_tiles)} wall orientation lines...")
    for draw_x, draw_y, tile_size, orientation_a, orientation_b in wall_tiles:
        draw_wall_lines(draw, draw_x, draw_y, tile_size, orientation_a, orientation_b)
    
    # Draw border around map
    draw.rectangle([map_start_x - 2, map_start_y - 2, 
                   map_start_x + map_width + 2, map_start_y + map_height + 2], 
                  outline=(100, 100, 100), width=2)
    
    # Draw embedded legend
    legend_x = map_start_x + map_width + padding
    legend_y = map_start_y
    
    # Legend title
    draw.text((legend_x, legend_y), "Pathfinding Map", fill=(255, 255, 255), font=title_font)
    legend_y += 30
    
    # Legend items
    def add_legend_item(text, color, y):
        draw.rectangle([legend_x, y, legend_x + 20, y + 15], fill=color, outline=(255, 255, 255))
        draw.text((legend_x + 25, y + 2), text, fill=(255, 255, 255), font=legend_font)
        return y + 20
    
    legend_y = add_legend_item("Walkable (.)", (0, 255, 0), legend_y)
    legend_y = add_legend_item("Solid (#)", (255, 0, 0), legend_y)
    legend_y = add_legend_item("Object (O)", (0, 0, 255), legend_y)
    legend_y = add_legend_item("Wall (W) - Standable", (0, 0, 0), legend_y)
    legend_y = add_legend_item("Wall (W) - Non-standable", (128, 128, 128), legend_y)
    legend_y = add_legend_item("Unknown (?)", (128, 128, 128), legend_y)
    legend_y = add_legend_item("No Data", (0, 0, 0), legend_y)
    
    # Add statistics
    legend_y += 20
    draw.text((legend_x, legend_y), "Statistics:", fill=(255, 255, 255), font=legend_font)
    legend_y += 20
    draw.text((legend_x, legend_y), f"Tiles: {len(collision_data):,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Size: {width_tiles}x{height_tiles}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Path waypoints: {len(path) if path else 0}", fill=(255, 255, 255), font=legend_font)
    
    print(f"[DEBUG] Generated filtered collision map with {tiles_drawn} tiles")
    
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
        
        # Convert path to map coordinates
        map_path = []
        for world_x, world_y in path:
            map_x, map_y = world_to_map_coords(world_x, world_y, min_x, min_y, max_y)
            map_path.append((map_x, map_y))
        
        # Draw path lines
        for i in range(len(map_path) - 1):
            start = map_path[i]
            end = map_path[i + 1]
            
            # Check if coordinates are within image bounds
            if (0 <= start[0] < img.size[0] and 0 <= start[1] < img.size[1] and
                0 <= end[0] < img.size[0] and 0 <= end[1] < img.size[1]):
                draw.line([start, end], fill=(0, 255, 255), width=6)  # Cyan path
            else:
                print(f"[WARNING] Line coordinates out of bounds! Image size: {img.size}")
                draw.line([start, end], fill=(0, 255, 255), width=6)
        
        print(f"[DEBUG] Path drawn from {start_pixel} to {goal_pixel}")
    else:
        print("WARNING: No path to draw")
    
    # Save the image
    print(f"[DEBUG] Saving image to: {output_path}")
    img.save(output_path)
    print(f"[SUCCESS] Filtered collision map with path saved to: {output_path}")
    
    # Verify the file was created
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        print(f"[DEBUG] Image file created successfully, size: {file_size:,} bytes")
    else:
        print(f"[ERROR] Image file was not created!")
    
    return True


def simple_greedy_path(start, goal, walkable_tiles, wall_masks, orientation_blockers):
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
                
                # Check if it's walkable, not visited, and not blocked by walls
                if (next_pos in walkable_tiles and next_pos not in visited and
                    not is_movement_blocked(current, next_pos, wall_masks, orientation_blockers)):
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
                if tile not in visited and not is_movement_blocked(current, tile, wall_masks, orientation_blockers):
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

def main(destination_x=None, destination_y=None, port=17000):
    """Main pathfinding function."""
    print("RUNESCAPE PATHFINDER - CURRENT POSITION TO DESTINATION")
    print("=" * 50)
    print("Now with WALL ORIENTATION support!")
    print("- Wall tiles WITH orientation data are WALKABLE")
    print("- Wall tiles WITHOUT orientation data are BLOCKING")
    print("- Wall orientation determines directional movement restrictions")
    print("=" * 50)
    
    # Create IPC client for this session
    try:
        ipc = IPCClient(port=port)
        print(f"[DEBUG] Created IPC client on port {port}")
    except Exception as e:
        print(f"ERROR: Could not create IPC client on port {port}: {e}")
        return False
    
    # Get current player position using the IPC client
    try:
        player_resp = ipc.get_player()
        if not player_resp or not player_resp.get("ok"):
            print("ERROR: Could not get player data from IPC")
            return False
        
        player_data = player_resp.get("player", {})
        current_x = player_data.get("worldX")
        current_y = player_data.get("worldY")
        
        if not isinstance(current_x, int) or not isinstance(current_y, int):
            print("ERROR: Invalid player coordinates from IPC")
            return False
        
        current_pos = (current_x, current_y)
        print(f"[DEBUG] Current player position: {current_pos}")
    except Exception as e:
        print(f"ERROR: Could not get current player position: {e}")
        return False
    
    # Set destination target
    DESTINATION_TARGET = (destination_x, destination_y)
    print(f"[DEBUG] Destination target: {DESTINATION_TARGET}")
    
    # Load collision data first, filtered by start and destination coordinates
    collision_data = load_collision_data(current_pos, DESTINATION_TARGET)
    if not collision_data:
        return False
    
    # Get walkable tiles with wall orientation support
    walkable_tiles, blocked_tiles, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
    print(f"Walkable tiles: {len(walkable_tiles)}")
    print(f"Blocked tiles: {len(blocked_tiles)}")
    print(f"Wall masks: {len(wall_masks)}")
    print(f"Orientation blockers: {len(orientation_blockers)}")
    
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
    path = astar_pathfinding(start, goal, walkable_tiles)
    
    if not path:
        print("A* failed, trying simple greedy pathfinding...")
        # Load collision data for simple greedy path, filtered by start/goal coordinates
        collision_data = load_collision_data(start, goal)
        if collision_data:
            _, _, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
        else:
            wall_masks = {}
            orientation_blockers = set()
        path = simple_greedy_path(start, goal, walkable_tiles, wall_masks, orientation_blockers)
    
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
        success = main(destination_x, destination_y, port)
        if success:
            print("\n[SUCCESS] Pathfinding completed successfully!")
        else:
            print("\n[ERROR] Pathfinding failed!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
