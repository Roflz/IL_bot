# collision_mapper.py
"""
Dynamic collision mapping service that learns the world as the bot moves around.
Collects collision data and builds a knowledge base for better pathing.
"""
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

from ilbot.ui.simple_recorder.helpers.ipc import ipc_send


class CollisionMapper:
    """
    Maps collision data as the bot moves around the world.
    Builds a knowledge base for better long-distance pathing.
    """
    
    def __init__(self, cache_dir: str = "collision_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory collision data
        self.collision_map = {}  # {(x, y, plane): collision_data}
        self.water_regions = set()  # {(x, y, plane)} - unwalkable water tiles
        self.door_map = {}  # {(x, y, plane): door_info}
        
        # Scene tracking
        self.scanned_scenes = set()  # {(baseX, baseY, plane)} - already scanned scenes
        self.last_scan_time = 0
        self.scan_interval = 2.0  # seconds between scans
        
        # Load existing data
        self._load_cached_data()
    
    def should_scan(self) -> bool:
        """Check if we should perform a new scan."""
        return time.time() - self.last_scan_time > self.scan_interval
    
    def scan_current_scene(self) -> bool:
        """
        Scan the current scene for collision data.
        
        Args:
            payload: Game state payload
            
        Returns:
            True if scan was successful, False otherwise
        """
        
        # Check if we should scan
        if not self.should_scan():
            return True
        
        print("[COLLISION_MAPPER] Scanning current scene...")
        
        # Get collision data from current scene
        collision_resp = ipc_send({"cmd": "scan_scene"})
        if not collision_resp or not collision_resp.get("ok"):
            print(f"[COLLISION_MAPPER] Failed to scan scene: {collision_resp}")
            return False
        
        # Process collision data
        base_x = collision_resp.get("baseX", 0)
        base_y = collision_resp.get("baseY", 0)
        plane = collision_resp.get("plane", 0)
        collision_data = collision_resp.get("collisionData", [])
        
        # Check if we've already scanned this scene
        scene_key = (base_x, base_y, plane)
        if scene_key in self.scanned_scenes:
            print(f"[COLLISION_MAPPER] Scene already scanned: {scene_key}")
            return True
        
        # Process collision data
        new_tiles = 0
        for tile_data in collision_data:
            x = tile_data.get("x")
            y = tile_data.get("y")
            p = tile_data.get("p", 0)
            walkable = tile_data.get("walkable", True)
            door_info = tile_data.get("door")
            
            tile_key = (x, y, p)
            
            # Store collision data
            self.collision_map[tile_key] = {
                "walkable": walkable,
                "door": door_info,
                "timestamp": time.time()
            }
            
            # Store door information
            if door_info:
                self.door_map[tile_key] = door_info
            
            new_tiles += 1
        
        # Get water regions
        water_resp = ipc_send({"cmd": "detect_water"})
        if water_resp and water_resp.get("ok"):
            water_regions = water_resp.get("waterRegions", [])
            for region_data in water_regions:
                region = region_data.get("region", [])
                for water_tile in region:
                    wx = water_tile.get("x")
                    wy = water_tile.get("y")
                    wp = water_tile.get("p", 0)
                    if isinstance(wx, int) and isinstance(wy, int):
                        self.water_regions.add((wx, wy, wp))
        
        # Mark scene as scanned
        self.scanned_scenes.add(scene_key)
        self.last_scan_time = time.time()
        
        print(f"[COLLISION_MAPPER] Scanned scene {scene_key}: {new_tiles} new tiles, {len(self.water_regions)} water tiles")
        
        # Save to cache
        self._save_cached_data()
        
        return True
    
    def is_walkable(self, x: int, y: int, plane: int = 0) -> bool:
        """
        Check if a tile is walkable based on our collision map.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            plane: World plane
            
        Returns:
            True if walkable, False if blocked or unknown
        """
        tile_key = (x, y, plane)
        
        # Check if we have data for this tile
        if tile_key in self.collision_map:
            return self.collision_map[tile_key]["walkable"]
        
        # Check if it's a known water tile
        if tile_key in self.water_regions:
            return False
        
        # If we don't have data, assume walkable (conservative approach)
        return True
    
    def has_door(self, x: int, y: int, plane: int = 0) -> Optional[Dict]:
        """
        Check if a tile has a door.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            plane: World plane
            
        Returns:
            Door information if present, None otherwise
        """
        tile_key = (x, y, plane)
        return self.door_map.get(tile_key)
    
    def get_known_tiles_in_radius(self, center_x: int, center_y: int, radius: int, plane: int = 0) -> List[Tuple[int, int, bool]]:
        """
        Get all known tiles within a radius of a center point.
        
        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
            radius: Search radius
            plane: World plane
            
        Returns:
            List of (x, y, walkable) tuples
        """
        known_tiles = []
        
        for x in range(center_x - radius, center_x + radius + 1):
            for y in range(center_y - radius, center_y + radius + 1):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    tile_key = (x, y, plane)
                    if tile_key in self.collision_map:
                        walkable = self.collision_map[tile_key]["walkable"]
                        known_tiles.append((x, y, walkable))
        
        return known_tiles
    
    def get_path_around_obstacles(self, start: Tuple[int, int], goal: Tuple[int, int], plane: int = 0) -> Optional[List[Tuple[int, int]]]:
        """
        Get a simple path around known obstacles.
        
        Args:
            start: (x, y) start coordinates
            goal: (x, y) goal coordinates
            plane: World plane
            
        Returns:
            List of waypoints or None if no path found
        """
        # Simple A* pathfinding using our collision data
        # This is a basic implementation - could be enhanced
        
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # For now, return a simple straight-line path if all tiles are walkable
        path = []
        dx = goal_x - start_x
        dy = goal_y - start_y
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return [start]
        
        for i in range(steps + 1):
            x = start_x + (dx * i) // steps
            y = start_y + (dy * i) // steps
            
            # Check if this tile is walkable
            if not self.is_walkable(x, y, plane):
                # If we hit an obstacle, try to go around it
                # This is a simple implementation - could be enhanced
                return None
            
            path.append((x, y))
        
        return path
    
    def _load_cached_data(self):
        """Load collision data from cache files."""
        try:
            # Load collision map
            collision_file = self.cache_dir / "collision_map.json"
            if collision_file.exists():
                with open(collision_file, 'r') as f:
                    data = json.load(f)
                    self.collision_map = {tuple(k): v for k, v in data.get("collision_map", {}).items()}
                    self.water_regions = {tuple(k) for k in data.get("water_regions", [])}
                    self.scanned_scenes = {tuple(k) for k in data.get("scanned_scenes", [])}
                    self.door_map = {tuple(k): v for k, v in data.get("door_map", {}).items()}
                print(f"[COLLISION_MAPPER] Loaded cached data: {len(self.collision_map)} tiles, {len(self.water_regions)} water tiles")
        except Exception as e:
            print(f"[COLLISION_MAPPER] Failed to load cached data: {e}")
    
    def _save_cached_data(self):
        """Save collision data to cache files."""
        try:
            data = {
                "collision_map": {list(k): v for k, v in self.collision_map.items()},
                "water_regions": [list(k) for k in self.water_regions],
                "scanned_scenes": [list(k) for k in self.scanned_scenes],
                "door_map": {list(k): v for k, v in self.door_map.items()},
                "timestamp": time.time()
            }
            
            collision_file = self.cache_dir / "collision_map.json"
            with open(collision_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[COLLISION_MAPPER] Saved collision data to cache")
        except Exception as e:
            print(f"[COLLISION_MAPPER] Failed to save cached data: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the collision map."""
        return {
            "total_tiles": len(self.collision_map),
            "water_tiles": len(self.water_regions),
            "doors": len(self.door_map),
            "scanned_scenes": len(self.scanned_scenes),
            "cache_dir": str(self.cache_dir)
        }


# Global instance
_collision_mapper = CollisionMapper()


def scan_current_scene() -> bool:
    """Scan the current scene for collision data."""
    return _collision_mapper.scan_current_scene()


def is_walkable(x: int, y: int, plane: int = 0) -> bool:
    """Check if a tile is walkable."""
    return _collision_mapper.is_walkable(x, y, plane)


def has_door(x: int, y: int, plane: int = 0) -> Optional[Dict]:
    """Check if a tile has a door."""
    return _collision_mapper.has_door(x, y, plane)


def get_collision_stats() -> Dict:
    """Get collision mapping statistics."""
    return _collision_mapper.get_stats()
