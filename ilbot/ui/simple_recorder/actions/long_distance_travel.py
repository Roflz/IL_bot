# ilbot/ui/simple_recorder/actions/long_distance_travel.py
"""
Long-distance travel module that uses cached collision data for pathfinding.
Integrates with the pathfinder.py script to find routes across large distances.
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from ..helpers.navigation import get_nav_rect


def load_collision_data() -> Optional[Dict]:
    """Load collision data from cache."""
    script_dir = Path(__file__).parent.parent
    cache_file = script_dir / "collision_cache" / "collision_map.json"
    
    if not cache_file.exists():
        print(f"[LONG_DISTANCE] No collision cache found at {cache_file}")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        collision_data = data.get('collision_data', {})
        print(f"[LONG_DISTANCE] Loaded {len(collision_data)} collision tiles")
        return collision_data
    except Exception as e:
        print(f"[LONG_DISTANCE] Error loading collision cache: {e}")
        return None


def get_walkable_tiles(collision_data: Dict) -> Tuple[set, set]:
    """Convert collision data to walkable and blocked tile sets."""
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


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def get_neighbors(pos: Tuple[int, int], walkable_tiles: set) -> List[Tuple[int, int]]:
    """Get walkable neighbors of a position."""
    x, y = pos
    neighbors = []
    
    # 8-directional movement (including diagonals)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_pos = (x + dx, y + dy)
            if new_pos in walkable_tiles:
                neighbors.append(new_pos)
    
    return neighbors


def astar_pathfinding(start: Tuple[int, int], goal: Tuple[int, int], walkable_tiles: set) -> Optional[List[Tuple[int, int]]]:
    """Find path using A* algorithm."""
    import heapq
    
    if start not in walkable_tiles:
        print(f"[LONG_DISTANCE] Start position {start} is not walkable")
        return None
    
    if goal not in walkable_tiles:
        print(f"[LONG_DISTANCE] Goal position {goal} is not walkable")
        return None
    
    # Priority queue: (f_score, tie_breaker, position)
    tie_breaker = 0
    open_set = [(0, tie_breaker, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()
    in_open_set = {start}
    
    iterations = 0
    max_iterations = 50000  # Reasonable limit for long distances
    
    while open_set and iterations < max_iterations:
        iterations += 1
        if iterations % 5000 == 0:
            print(f"[LONG_DISTANCE] A* iteration {iterations}, visited {len(visited)} tiles")
            
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
            print(f"[LONG_DISTANCE] Path found with {len(path)} waypoints after {iterations} iterations")
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
                
                if neighbor not in in_open_set:
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score[neighbor], tie_breaker, neighbor))
                    in_open_set.add(neighbor)
    
    print(f"[LONG_DISTANCE] No path found after {iterations} iterations")
    return None


def find_closest_walkable(pos: Tuple[int, int], walkable_tiles: set, max_distance: int = 50) -> Optional[Tuple[int, int]]:
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


def get_target_coordinates(rect_or_key: Union[str, tuple, list]) -> Optional[Tuple[int, int]]:
    """Get target coordinates from rect_or_key."""
    if isinstance(rect_or_key, (tuple, list)) and len(rect_or_key) == 4:
        # Use center of rectangle
        min_x, max_x, min_y, max_y = rect_or_key
        return (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    else:
        # Get rectangle from key
        rect = get_nav_rect(str(rect_or_key))
        if rect and len(rect) == 4:
            min_x, max_x, min_y, max_y = rect
            return (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    
    return None


def sample_waypoints(path: List[Tuple[int, int]], sample_interval: int = 35) -> List[Tuple[int, int]]:
    """
    Sample waypoints from the path at regular intervals.
    
    Args:
        path: Full path from pathfinding
        sample_interval: Take every Nth waypoint (default 35)
        
    Returns:
        List of sampled waypoints
    """
    if not path:
        return []
    
    # Always include the first and last waypoints
    sampled = [path[0]]
    
    # Sample intermediate waypoints
    for i in range(sample_interval, len(path) - 1, sample_interval):
        sampled.append(path[i])
    
    # Always include the last waypoint if it's not already included
    if len(path) > 1 and path[-1] != sampled[-1]:
        sampled.append(path[-1])
    
    print(f"[LONG_DISTANCE] Sampled {len(sampled)} waypoints from {len(path)} total waypoints")
    return sampled


def convert_path_to_waypoints(path: List[Tuple[int, int]]) -> List[Dict]:
    """Convert pathfinder path to intermediate waypoints for local pathfinding."""
    # Sample waypoints from the full path
    sampled_path = sample_waypoints(path, sample_interval=35)
    
    waypoints = []
    for i, (world_x, world_y) in enumerate(sampled_path):
        # Create waypoint for local pathfinding
        waypoint = {
            "x": world_x,
            "y": world_y,
            "p": 0,  # Default plane
            "world": {
                "x": world_x,
                "y": world_y,
                "p": 0
            },
            "index": i,
            "is_intermediate": i > 0 and i < len(sampled_path) - 1  # Not start or end
        }
        waypoints.append(waypoint)
    
    return waypoints

