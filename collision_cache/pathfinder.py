#!/usr/bin/env python3
"""
Pathfinding script for RuneScape collision map.
Generates a path from current player position to Grand Exchange and overlays it on the collision map PNG.
"""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import os

from helpers.ipc import IPCClient

# Removed problematic import

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_collision_data(start_coords=None, dest_coords=None, buffer=200):
    """Load collision data from cache, filtered by start/destination bounds."""
    script_dir = Path(__file__).parent
    cache_file = script_dir / "collision_cache" / "collision_map_debug.json"
    # Loading collision data
    
    if not cache_file.exists():
        print("ERROR: No collision cache found. Run collision mapping first.")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        all_collision_data = data.get("collision_data", {})
        # Total collision tiles in cache
        
        # If no start/dest coordinates provided, return all data
        if not start_coords or not dest_coords:
            # No start/dest coordinates provided, returning all collision data
            return all_collision_data
        
        # Calculate bounds based on start and destination coordinates
        start_x, start_y = start_coords
        dest_x, dest_y = dest_coords
        
        min_x = min(start_x, dest_x) - buffer
        max_x = max(start_x, dest_x) + buffer
        min_y = min(start_y, dest_y) - buffer
        max_y = max(start_y, dest_y) + buffer
        
        # Filtering collision data to bounds
        
        # Filter collision data to only include tiles within bounds
        filtered_data = {}
        for tile_id, tile_data in all_collision_data.items():
            x, y = tile_data['x'], tile_data['y']
            if min_x <= x <= max_x and min_y <= y <= max_y:
                filtered_data[tile_id] = tile_data
        
        # Filtered collision data
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
    
    # RuneLite CollisionDataFlag constants
    BLOCK_MOVEMENT_NORTH       = 2
    BLOCK_MOVEMENT_NORTH_EAST  = 4
    BLOCK_MOVEMENT_EAST        = 8
    BLOCK_MOVEMENT_SOUTH_EAST  = 16
    BLOCK_MOVEMENT_SOUTH       = 32
    BLOCK_MOVEMENT_SOUTH_WEST  = 64
    BLOCK_MOVEMENT_WEST        = 128
    BLOCK_MOVEMENT_NORTH_WEST  = 1

    BLOCK_MOVEMENT_OBJECT      = 256
    BLOCK_MOVEMENT_FLOOR_DECOR = 262144
    BLOCK_MOVEMENT_FLOOR       = 2097152
    BLOCK_MOVEMENT_FULL        = 2359552

    DECOR_BLOCKS = True
    
    # Internal mask bits for edges
    W, N, E, S = 1, 2, 4, 8
    NW, NE, SE, SW = 16, 32, 64, 128
    FULL = 1 << 20  # internal "center-blocked" marker
    
    def mask_from_flags(flags: int) -> int:
        m = 0
        if flags & BLOCK_MOVEMENT_NORTH:  m |= N
        if flags & BLOCK_MOVEMENT_EAST:  m |= E
        if flags & BLOCK_MOVEMENT_SOUTH:  m |= S
        if flags & BLOCK_MOVEMENT_WEST:  m |= W
        if flags & BLOCK_MOVEMENT_NORTH_EAST: m |= NE
        if flags & BLOCK_MOVEMENT_SOUTH_EAST: m |= SE
        if flags & BLOCK_MOVEMENT_SOUTH_WEST: m |= SW
        if flags & BLOCK_MOVEMENT_NORTH_WEST: m |= NW
        if flags & BLOCK_MOVEMENT_OBJECT:      m |= FULL
        if flags & BLOCK_MOVEMENT_FLOOR:       m |= FULL
        if flags & BLOCK_MOVEMENT_FULL:        m |= FULL
        if DECOR_BLOCKS and (flags & BLOCK_MOVEMENT_FLOOR_DECOR): m |= FULL
        return m
    
    # stash door edge clear requests; we'll merge after the first pass
    door_clears_here = {}      # (x,y) -> mask_to_clear_bits
    door_clears_neighbor = {}  # (nx,ny) -> mask_to_clear_bits

    # pass 1: scan tiles and record clears
    for tile_data in collision_data.values():
        x, y = tile_data['x'], tile_data['y']
        flags = tile_data.get('flags', 0)
        mask = mask_from_flags(flags)

        door_info = tile_data.get('door')
        if door_info and isinstance(door_info, dict):
            # Check if door has 'Open' action (regardless of passable status)
            door_actions = door_info.get('actions', [])
            if 'Open' in door_actions:
                a = int(door_info.get('orientationA') or 0)
                b = int(door_info.get('orientationB') or 0)

                # RuneLite door orientation mapping:
                # 1=West, 2=North, 4=East, 8=South, 16=NW, 32=NE, 64=SE, 128=SW

                # bits to clear on THIS tile
                clear_here = 0
                if a & 1:   clear_here |= W  # West edge open
                if a & 2:   clear_here |= N  # North edge open
                if a & 4:   clear_here |= E  # East edge open
                if a & 8:   clear_here |= S  # South edge open
                if b & 16:  clear_here |= NW # NW diagonal
                if b & 32:  clear_here |= NE # NE diagonal
                if b & 64:  clear_here |= SE # SE diagonal
                if b & 128: clear_here |= SW # SW diagonal
                if clear_here:
                    door_clears_here[(x, y)] = door_clears_here.get((x, y), 0) | clear_here

                # And clear the OPPOSITE edge on the NEIGHBOR tile
                def add_neighbor_clear(dx, dy, opp):
                    if dx == 0 and dy == 0:
                        return
                    key = (x + dx, y + dy)
                    door_clears_neighbor[key] = door_clears_neighbor.get(key, 0) | opp

                # NOTE: y+1 is NORTH in your grid; opposite edges are paired accordingly
                if a & 1:   add_neighbor_clear(-1, 0, E)  # WEST opens → neighbor EAST
                if a & 2:   add_neighbor_clear(0, +1, S)  # NORTH opens → neighbor SOUTH
                if a & 4:   add_neighbor_clear(+1, 0, W)  # EAST opens → neighbor WEST
                if a & 8:   add_neighbor_clear(0, -1, N)  # SOUTH opens → neighbor NORTH
                if b & 16:  add_neighbor_clear(-1, +1, SE) # NW diagonal
                if b & 32:  add_neighbor_clear(+1, +1, SW) # NE diagonal
                if b & 64:  add_neighbor_clear(+1, -1, NW) # SE diagonal
                if b & 128: add_neighbor_clear(-1, -1, NE) # SW diagonal

        # store raw masks for now
        wall_masks[(x, y)] = mask

    # pass 2: apply clears (on both the door tile and its opposite neighbor)
    for pos, clr in door_clears_here.items():
        if pos in wall_masks:
            wall_masks[pos] &= ~clr

    for pos, clr in door_clears_neighbor.items():
        if pos in wall_masks:
            wall_masks[pos] &= ~clr
        else:
            # neighbor exists but had no edges yet — ensure it exists so entry/exit checks see the cleared edge
            wall_masks[pos] = 0

    # now decide walkable vs blocked:
    for tile_data in collision_data.values():
        x, y = tile_data['x'], tile_data['y']
        flags = tile_data.get('flags', 0)
        mask = wall_masks.get((x, y), 0)

        if mask & FULL:
            blocked.add((x, y))
        else:
            walkable.add((x, y))
    
    # Add all tiles that are NOT in collision_data as walkable (NO DATA = walkable)
    if collision_data:
        all_x = [tile['x'] for tile in collision_data.values()]
        all_y = [tile['y'] for tile in collision_data.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) not in walkable and (x, y) not in blocked:
                    walkable.add((x, y))
    
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
    # Cardinals: only the matching edge blocks
    if dx == 1 and dy == 0 and (m & E): return True   # E
    if dx == -1 and dy == 0 and (m & W): return True  # W
    if dx == 0 and dy == 1 and (m & N): return True   # N
    if dx == 0 and dy == -1 and (m & S): return True  # S

    if not use_diag_bars:
        # With diagonal bars disabled, diagonals are blocked only by crossed source cardinals
        if dx == 1 and dy == 1:   return bool((m & N) or (m & E))  # NE
        if dx == -1 and dy == 1:  return bool((m & N) or (m & W))  # NW
        if dx == 1 and dy == -1:  return bool((m & S) or (m & E))  # SE
        if dx == -1 and dy == -1: return bool((m & S) or (m & W))  # SW
        return False

    # Diagonals: the diagonal bar itself blocks, OR either crossed source cardinals
    if dx == 1 and dy == 1:    # NE
        return bool((m & NE) or (m & N) or (m & E))
    if dx == -1 and dy == 1:   # NW
        return bool((m & NW) or (m & N) or (m & W))
    if dx == 1 and dy == -1:   # SE
        return bool((m & SE) or (m & S) or (m & E))
    if dx == -1 and dy == -1:  # SW
        return bool((m & SW) or (m & S) or (m & W))

    return False


def entry_blocked_by_mask(dx: int, dy: int, m: int, use_diag_bars: bool = True) -> bool:
    """
    Return True if the *destination* tile's mask `m` blocks entry for a move (dx, dy).

    Cardinals: check the edge you're ENTERING FROM (opposite of motion).
    Diagonals: check the opposite-corner bar *and* the two cardinal edges crossed.
    """

    # --- Cardinals ---
    if dx ==  1 and dy ==  0:  # moving East -> enter from West -> dest.W blocks
        return bool(m & W)
    if dx == -1 and dy ==  0:  # moving West -> enter from East -> dest.E blocks
        return bool(m & E)
    if dx ==  0 and dy ==  1:  # moving North -> enter from South -> dest.S blocks
        return bool(m & S)
    if dx ==  0 and dy == -1:  # moving South -> enter from North -> dest.N blocks
        return bool(m & N)

    # --- Diagonals ---
    if not use_diag_bars:
        # If diagonal bars disabled, still block if either cardinal edge blocks
        if dx == -1 and dy ==  1:  # NW -> cross dest.S + dest.E
            return bool((m & S) or (m & E))
        if dx ==  1 and dy ==  1:  # NE -> cross dest.S + dest.W
            return bool((m & S) or (m & W))
        if dx == -1 and dy == -1:  # SW -> cross dest.N + dest.E
            return bool((m & N) or (m & E))
        if dx ==  1 and dy == -1:  # SE -> cross dest.N + dest.W
            return bool((m & N) or (m & W))
        return False

    # With diagonal bars: check the correct diagonal corner + both cardinals
    if dx == -1 and dy ==  1:  # NW
        return bool((m & SE) or (m & S) or (m & E))
    if dx ==  1 and dy ==  1:  # NE
        return bool((m & SW) or (m & S) or (m & W))
    if dx == -1 and dy == -1:  # SW
        return bool((m & NE) or (m & N) or (m & E))
    if dx ==  1 and dy == -1:  # SE
        return bool((m & NW) or (m & N) or (m & W))

    return False


def is_movement_blocked(a, b, wall_masks, orientation_blockers, use_diag_bars: bool = True) -> bool:
    """Check if movement from a to b is blocked by wall orientation."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    
    # Check source tile (exit blocking)
    ma = wall_masks.get(a, 0)
    if ma and exit_blocked_by_mask(dx, dy, ma, use_diag_bars):
        return True

    # Check destination tile (entry blocking)
    mb = wall_masks.get(b, 0)
    if mb and entry_blocked_by_mask(dx, dy, mb, use_diag_bars):
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
                a = (x + dx, y)
                b = (x, y + dy)
                if (a not in walkable_tiles) or (b not in walkable_tiles):
                    continue
                if (is_movement_blocked(pos, a, wall_masks, orientation_blockers) or
                    is_movement_blocked(pos, b, wall_masks, orientation_blockers) or
                    is_movement_blocked(a, nxt, wall_masks, orientation_blockers) or
                    is_movement_blocked(b, nxt, wall_masks, orientation_blockers)):
                    continue

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


def astar_pathfinding(start, goal, walkable_tiles, max_iterations=100000):
    """Find path using A* algorithm with wall orientation support and partial fallback."""
    print(f"[DEBUG] Finding path from {start} to {goal}")
    print(f"[DEBUG] Walkable tiles: {len(walkable_tiles)}")

    collision_data = load_collision_data(start, goal)
    if collision_data:
        _, _, wall_masks, orientation_blockers = get_walkable_tiles(collision_data)
        # Loaded wall masks and orientation blockers
    else:
        wall_masks = {}
        orientation_blockers = set()
        print(f"[DEBUG] Could not load collision data, using empty wall data")

    if start not in walkable_tiles:
        print(f"ERROR: Start position {start} is not walkable")
        return None

    if goal not in walkable_tiles:
        print(f"[DEBUG] Goal {goal} not walkable; finding closest walkable...")
        cg = find_closest_walkable_tile(goal, walkable_tiles)
        if not cg:
            print(f"ERROR: No walkable tile near goal {goal}")
            return None
        goal = cg
        print(f"[DEBUG] Using closest walkable goal: {goal}")
    else:
        print(f"[DEBUG] Goal position {goal} is walkable")

    if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) <= 2:
        return [start, goal]

    import heapq
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    tie = 0

    best_node = start
    best_h = heuristic(start, goal)

    iterations = 0
    while open_set and iterations < max_iterations:
        iterations += 1
        if iterations % 2000 == 0:
            print(f"[DEBUG] Iteration {iterations}, visited {len(visited)} tiles, queue size {len(open_set)}")

        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        h_cur = heuristic(current, goal)
        if h_cur < best_h:
            best_h = h_cur
            best_node = current

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            print(f"[DEBUG] Path found with {len(path)} waypoints after {iterations} iterations")
            print(f"[DEBUG] Path waypoints: {path}")
            return path

        for nbr in get_neighbors(current, walkable_tiles, wall_masks, orientation_blockers):
            if nbr in visited:
                continue
            dx = abs(nbr[0] - current[0])
            dy = abs(nbr[1] - current[1])
            step_cost = 1.414 if dx and dy else 1.0
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nbr, float('inf')):
                came_from[nbr] = current
                g_score[nbr] = tentative
                tie += 1
                f = tentative + heuristic(nbr, goal)
                heapq.heappush(open_set, (f, tie, nbr))

    print(f"ERROR: No path found after {iterations} iterations (visited {len(visited)} tiles)")
    if best_node is not None and best_node != start:
        print(f"[DEBUG] Returning partial path to closest discovered node {best_node} (h={best_h:.3f})")
        path = [best_node]
        cur = best_node
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

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


def draw_wall_lines(draw, x, y, tile_size, flags, orientation_b, is_real_door=False):
    """Draw wall lines based on movement blocking flags."""
    if is_real_door:
        line_color = (0, 255, 255)  # Cyan for real door lines
    else:
        line_color = (255, 255, 255)  # White for solid wall lines
    line_width = 2  # Thinner lines for better visibility
    
    # Draw lines based on movement blocking flags
    # BLOCK_MOVEMENT_NORTH (2) - draw line on NORTH edge
    if flags & 2:  # North
        draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_EAST (8) - draw line on EAST edge  
    if flags & 8:  # East
        draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_SOUTH (32) - draw line on SOUTH edge
    if flags & 32:  # South
        draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_WEST (128) - draw line on WEST edge
    if flags & 128:  # West
        draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)
    
    # Draw diagonal lines for orientation B (using flags directly)
    # BLOCK_MOVEMENT_NORTH_EAST (4) - draw diagonal from NW to SE
    if flags & 4:  # North-east
        draw.line([x + tile_size, y, x, y + tile_size], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_SOUTH_EAST (16) - draw diagonal from SW to NE
    if flags & 16:  # South-east
        draw.line([x, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_SOUTH_WEST (64) - draw diagonal from SE to NW
    if flags & 64:  # South-west
        draw.line([x + tile_size, y + tile_size, x, y], fill=line_color, width=line_width)
    # BLOCK_MOVEMENT_NORTH_WEST (1) - draw diagonal from NE to SW
    if flags & 1:  # North-west
        draw.line([x, y + tile_size, x + tile_size, y], fill=line_color, width=line_width)


def world_to_map_coords(world_x, world_y, min_x, min_y, max_y, tile_size=16, padding=20):
    """Convert world coordinates to map pixel coordinates."""
    # X coordinate: simple offset
    map_x = padding + (world_x - min_x) * tile_size
    
    # Y coordinate: inverted (like in visual_collision_map.py)
    map_height = (max_y - min_y + 1) * tile_size
    map_y = padding + map_height - ((world_y - min_y) * tile_size) - tile_size
    
    return int(map_x), int(map_y)


def world_to_map_coords_center(world_x, world_y, min_x, min_y, max_y, tile_size=16, padding=20):
    """Convert world coordinates to map pixel coordinates at tile center."""
    # X coordinate: simple offset + half tile size for center
    map_x = padding + (world_x - min_x) * tile_size + (tile_size // 2)
    
    # Y coordinate: inverted (like in visual_collision_map.py) + half tile size for center
    map_height = (max_y - min_y + 1) * tile_size
    map_y = padding + map_height - ((world_y - min_y) * tile_size) - tile_size + (tile_size // 2)
    
    return int(map_x), int(map_y)


def draw_path_on_map_simple(image_path, path, collision_data, output_path, start_pos=None, goal_pos=None):
    """Load existing collision map and draw the path on it."""
    print(f"[DEBUG] Loading existing collision map: {image_path}")
    
    # Load the existing collision map image
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        print(f"[DEBUG] Loaded collision map: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"[ERROR] Could not load collision map: {e}")
        return False
    
    # Load fonts
    try:
        small_font = ImageFont.truetype("arial.ttf", 8)
        legend_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        small_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Get map bounds from collision data
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Map parameters (matching visual_collision_map.py)
    tile_size = 16
    padding = 20
    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1
    map_width = width_tiles * tile_size
    map_height = height_tiles * tile_size
    
    # Draw collision tiles
    map_start_x = padding
    map_start_y = padding
    
    # Collect wall tiles for orientation lines
    wall_tiles = []  # (draw_x, draw_y, tile_size, orientation_a, orientation_b, is_real_door)
    
    tiles_drawn = 0
    for key, tile_data in collision_data.items():
        x, y, p = tile_data['x'], tile_data['y'], tile_data['p']
        
        # Calculate position on image
        draw_x = map_start_x + (x - min_x) * tile_size
        draw_y = map_start_y + map_height - ((y - min_y) * tile_size) - tile_size  # Invert Y
        
        # Get flags and compute mask
        flags = tile_data.get('flags', 0)
        is_door = tile_data.get('door', False)
        is_ladder_up = tile_data.get('ladderUp', False)
        is_ladder_down = tile_data.get('ladderDown', False)
        
        # RuneLite CollisionDataFlag constants
        BLOCK_MOVEMENT_NORTH       = 2
        BLOCK_MOVEMENT_NORTH_EAST  = 4
        BLOCK_MOVEMENT_EAST        = 8
        BLOCK_MOVEMENT_SOUTH_EAST  = 16
        BLOCK_MOVEMENT_SOUTH       = 32
        BLOCK_MOVEMENT_SOUTH_WEST  = 64
        BLOCK_MOVEMENT_WEST        = 128
        BLOCK_MOVEMENT_NORTH_WEST  = 1

        BLOCK_MOVEMENT_OBJECT      = 256
        BLOCK_MOVEMENT_FLOOR_DECOR = 262144
        BLOCK_MOVEMENT_FLOOR       = 2097152
        BLOCK_MOVEMENT_FULL        = 2359552

        DECOR_BLOCKS = True
        
        # Internal mask bits for edges
        W, N, E, S = 1, 2, 4, 8
        NW, NE, SE, SW = 16, 32, 64, 128
        FULL = 1 << 20  # internal "center-blocked" marker
        
        def mask_from_flags(flags: int) -> int:
            m = 0
            if flags & BLOCK_MOVEMENT_NORTH:  m |= N
            if flags & BLOCK_MOVEMENT_EAST:  m |= E
            if flags & BLOCK_MOVEMENT_SOUTH:  m |= S
            if flags & BLOCK_MOVEMENT_WEST:  m |= W
            if flags & BLOCK_MOVEMENT_NORTH_EAST: m |= NE
            if flags & BLOCK_MOVEMENT_SOUTH_EAST: m |= SE
            if flags & BLOCK_MOVEMENT_SOUTH_WEST: m |= SW
            if flags & BLOCK_MOVEMENT_NORTH_WEST: m |= NW
            if flags & BLOCK_MOVEMENT_OBJECT:      m |= FULL
            if flags & BLOCK_MOVEMENT_FLOOR:       m |= FULL
            if flags & BLOCK_MOVEMENT_FULL:        m |= FULL
            if DECOR_BLOCKS and (flags & BLOCK_MOVEMENT_FLOOR_DECOR): m |= FULL
            return m
        
        mask = mask_from_flags(flags)
        
        # Determine color and symbol based on flags
        color = (0, 0, 0)  # Default black
        symbol = ""
        
        # Check for door/ladder markers first (these take priority)
        door_info = tile_data.get('door')
        is_real_door = door_info and isinstance(door_info, dict) and 'orientationA' in door_info
        if is_real_door:
            color = (0, 0, 0)  # Black background for doors
            symbol = "D"  # D for door
            # Store for wall line drawing if has orientation
            if mask != 0:
                wall_tiles.append((draw_x, draw_y, tile_size, flags, 0, True))  # True = is_real_door
        elif is_ladder_up:
            color = (0, 0, 0)  # Black background
            symbol = "U"  # U for ladder up
        elif is_ladder_down:
            color = (0, 0, 0)  # Black background
            symbol = "↓"  # ↓ for ladder down
        # Check for full tile blocked (after doors/ladders)
        elif mask & FULL:
            # Check if it's an object vs solid block
            if (flags & BLOCK_MOVEMENT_OBJECT) == BLOCK_MOVEMENT_OBJECT:
                color = (0, 0, 255)  # Blue background for objects
                symbol = "O"  # O for object
            else:
                color = (255, 0, 0)  # Red background for solid blocks
                symbol = "#"  # # for solid
        # Check for tiles with wall orientation data (but not full blocks)
        elif mask != 0:
            # Tiles with wall orientation are walkable but block movement
            color = (0, 0, 0)  # Black background
            symbol = "W"  # W for wall
            # Store for wall line drawing
            wall_tiles.append((draw_x, draw_y, tile_size, flags, 0, False))  # False = not real door
        # Tiles with no flags are walkable
        else:
            color = (0, 0, 0)  # Black background
            symbol = "."  # . for walkable
        
        # Draw tile background
        draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=color)
        
        # Add symbol
        if symbol:
            draw.text((draw_x + 1, draw_y + 1), symbol, fill=(255, 255, 255), font=small_font)
        
        tiles_drawn += 1
    
    # Draw wall orientation lines
    print(f"[DEBUG] Drawing {len(wall_tiles)} wall orientation lines...")
    for draw_x, draw_y, tile_size, flags, orientation_b, is_real_door in wall_tiles:
        draw_wall_lines(draw, draw_x, draw_y, tile_size, flags, orientation_b, is_real_door)
    
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
    legend_y = add_legend_item("Door (D) - Cyan lines", (0, 255, 255), legend_y)
    legend_y = add_legend_item("Wall (W) - White lines", (255, 255, 255), legend_y)
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
    
    start_pixel = world_to_map_coords_center(start_world[0], start_world[1], min_x, min_y, max_y)
    goal_pixel = world_to_map_coords_center(goal_world[0], goal_world[1], min_x, min_y, max_y)
    
    print(f"[DEBUG] Start: World {start_world} -> Pixel {start_pixel}")
    print(f"[DEBUG] Goal: World {goal_world} -> Pixel {goal_pixel}")
    
    # Draw straight line first so you can see the actual start/end points
    print(f"[DEBUG] Drawing straight line from {start_pixel} to {goal_pixel}")
    draw.line([start_pixel, goal_pixel], fill=(255, 255, 0), width=3)  # Yellow straight line
    
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
        
        # Convert path to map coordinates (using tile centers)
        map_path = []
        for world_x, world_y in path:
            map_x, map_y = world_to_map_coords_center(world_x, world_y, min_x, min_y, max_y)
            map_path.append((map_x, map_y))
        
        # Draw path lines
        for i in range(len(map_path) - 1):
            start = map_path[i]
            end = map_path[i + 1]
            
            # Check if coordinates are within image bounds
            if (0 <= start[0] < img.size[0] and 0 <= start[1] < img.size[1] and
                0 <= end[0] < img.size[0] and 0 <= end[1] < img.size[1]):
                draw.line([start, end], fill=(0, 255, 255), width=4)  # Cyan path (thicker for visibility)
            else:
                print(f"[WARNING] Line coordinates out of bounds! Image size: {img.size}")
                draw.line([start, end], fill=(0, 255, 255), width=4)
        
        # Draw small dots at each waypoint to show which tiles the path goes through
        for i, waypoint in enumerate(map_path):
            if (0 <= waypoint[0] < img.size[0] and 0 <= waypoint[1] < img.size[1]):
                # Different colors for start, end, and middle waypoints
                if i == 0:
                    # Start waypoint - green dot
                    draw.ellipse([waypoint[0]-3, waypoint[1]-3, waypoint[0]+3, waypoint[1]+3], 
                               fill=(0, 255, 0), outline=(255, 255, 255), width=1)
                elif i == len(map_path) - 1:
                    # End waypoint - red dot
                    draw.ellipse([waypoint[0]-3, waypoint[1]-3, waypoint[0]+3, waypoint[1]+3], 
                               fill=(255, 0, 0), outline=(255, 255, 255), width=1)
                else:
                    # Middle waypoints - white dots
                    draw.ellipse([waypoint[0]-2, waypoint[1]-2, waypoint[0]+2, waypoint[1]+2], 
                               fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        
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


def draw_path_on_map(image_path, path, collision_data, output_path, start_pos=None, goal_pos=None):
    """Load existing collision map and draw the path on it."""
    print(f"[DEBUG] Loading existing collision map: {image_path}")
    
    # Load the existing collision map image
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        print(f"[DEBUG] Loaded collision map: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"[ERROR] Could not load collision map: {e}")
        return False
    
    # Get map bounds from collision data to convert world coordinates to pixel coordinates
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Map parameters (matching visual_collision_map.py)
    tile_size = 16
    padding = 20
    
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
    
    # Convert world coordinates to pixel coordinates
    start_pixel = world_to_map_coords_center(start_world[0], start_world[1], min_x, min_y, max_y, tile_size, padding)
    goal_pixel = world_to_map_coords_center(goal_world[0], goal_world[1], min_x, min_y, max_y, tile_size, padding)
    
    print(f"[DEBUG] Start: World {start_world} -> Pixel {start_pixel}")
    print(f"[DEBUG] Goal: World {goal_world} -> Pixel {goal_pixel}")
    
    # Draw straight line first so you can see the actual start/end points
    print(f"[DEBUG] Drawing straight line from {start_pixel} to {goal_pixel}")
    draw.line([start_pixel, goal_pixel], fill=(255, 255, 0), width=3)  # Yellow straight line
    
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
        
        # Convert path to pixel coordinates
        path_pixels = []
        for world_pos in path:
            pixel_pos = world_to_map_coords_center(world_pos[0], world_pos[1], min_x, min_y, max_y, tile_size, padding)
            path_pixels.append(pixel_pos)
        
        # Draw path lines
        for i in range(len(path_pixels) - 1):
            start_pixel = path_pixels[i]
            end_pixel = path_pixels[i + 1]
            draw.line([start_pixel, end_pixel], fill=(255, 0, 255), width=2)  # Magenta path line
    
    # Save the image
    img.save(output_path)
    print(f"[SUCCESS] Path drawn on collision map saved to: {output_path}")
    
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

    max_steps = 5000
    step = 0

    stagnant_count = 0
    last_pos = current

    while current != goal and step < max_steps:
        step += 1

        best_next = None
        best_distance = float('inf')

        # Primary pass: 8-directional greedy step
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (current[0] + dx, current[1] + dy)
                if (nxt in walkable_tiles and
                    not is_movement_blocked(current, nxt, wall_masks, orientation_blockers)):
                    d = ((goal[0] - nxt[0])**2 + (goal[1] - nxt[1])**2) ** 0.5
                    if d < best_distance:
                        best_distance = d
                        best_next = nxt

        # Fallback: try a small neighborhood, but NEVER pick current
        if best_next is None:
            near_best = None
            near_best_d = float('inf')
            cx, cy = current
            for x in range(cx - 3, cx + 4):
                for y in range(cy - 3, cy + 4):
                    if (x, y) == current:
                        continue
                    if (x, y) in walkable_tiles and not is_movement_blocked(current, (x, y), wall_masks, orientation_blockers):
                        d = ((goal[0] - x)**2 + (goal[1] - y)**2) ** 0.5
                        if d < near_best_d:
                            near_best_d = d
                            near_best = (x, y)
            best_next = near_best

        if best_next is None or best_next == current:
            print(f"[ERROR] Greedy stuck at {current}; returning partial path ({len(path)} waypoints)")
            break

        current = best_next
        path.append(current)

        if current == last_pos:
            stagnant_count += 1
        else:
            stagnant_count = 0
            last_pos = current

        if step % 100 == 0:
            print(f"[DEBUG] Step {step}, current: {current}, distance to goal: {best_distance}")

        if stagnant_count >= 10:
            print(f"[WARNING] Greedy stagnant near {current}; returning partial path ({len(path)} waypoints)")
            break

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
    
    start_pixel = world_to_map_coords_center(start[0], start[1], min_x, min_y, max_y)
    goal_pixel = world_to_map_coords_center(goal[0], goal[1], min_x, min_y, max_y)
    
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
    input_image = script_dir / "collision_cache" / "detailed_collision_map_debug.png"
    output_image = script_dir / "collision_cache" / "path_current_to_ge_debug.png"
    
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
