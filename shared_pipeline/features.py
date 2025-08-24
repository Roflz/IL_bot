"""
Feature Extraction Module for OSRS Bot Imitation Learning

This module extracts feature vectors from gamestates,
preserving the exact behavior and feature ordering from legacy extract_features.py.
The feature count is determined dynamically based on the extracted features.
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .feature_map import create_feature_mapping, get_feature_group_for_index


class FeatureExtractor:
    """
    Feature extractor that produces feature vectors from gamestates.
    
    This class preserves the exact feature extraction logic from legacy extract_features.py,
    including feature ordering, data types, and normalization behavior.
    The feature count is determined dynamically based on the extracted features.
    """
    
    def __init__(self):
        # Feature groups will be determined dynamically during extraction
        self.feature_groups = {}
        self.n_features = 0
        
        # Session timing for relative timestamps
        self.session_start_time = None
        self.session_start_time_initialized = False
        
        # Feature mapping storage
        self.feature_mappings = []
        
        # Automatic feature index counter
        self.current_feature_index = 0
        
        # ID and hash mapping tracking (exact from legacy code)
        self.id_mappings = {
            "Player": {
                "player_animation_ids": {},
                "player_movement_direction_hashes": {}
            },
            "Interaction": {
                "action_type_hashes": {},
                "item_name_hashes": {},
                "target_hashes": {}
            },
            "Inventory": {
                "item_ids": {},
                "empty_slot_ids": {}
            },
            "Bank": {
                "slot_ids": {},
                "boolean_states": {}
            },
            "Game Objects": {
                "object_ids": {}
            },
            "NPCs": {
                "npc_ids": {}
            },
            "Tabs": {
                "tab_ids": {}
            },
            "Phase Context": {
                "phase_type_hashes": {}
            },
            "Global": {
                "hash_mappings": {}
            }
        }
    
    def track_id_mapping(self, feature_group: str, mapping_type: str, id_value: Any, name_value: str):
        """Track ID mappings for items, NPCs, animations, objects, and hashes with proper feature isolation"""
        if feature_group not in self.id_mappings:
            return
            
        if mapping_type not in self.id_mappings[feature_group]:
            return
            
        if isinstance(id_value, (int, float)):
            self.id_mappings[feature_group][mapping_type][int(id_value)] = name_value
    
    def stable_hash(self, text: str, max_val: int = 100_000) -> int:
        """
        Deterministic cross-run hash (independent of PYTHONHASHSEED).
        """
        h = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8, person=b"osrs-map")
        return int.from_bytes(h.digest(), "big") % max_val
    
    def safe_float(self, value) -> float:
        """Safely convert value to float, returning 0.0 for invalid values."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def safe_int(self, val, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            if val == "" or val is None:
                return default
            return int(val)
        except Exception:
            return default
    
    def initialize_session_timing(self, gamestates: List[Dict]):
        """Initialize session start time from all gamestates for relative timestamp conversion."""
        if self.session_start_time_initialized:
            return
            
        print("Initializing session timing for relative timestamps...")
        
        # Find the earliest timestamp across all gamestates
        all_timestamps = []
        for gamestate in gamestates:
            timestamp = gamestate.get('timestamp', 0)
            if timestamp > 0:
                all_timestamps.append(timestamp)
        
        if all_timestamps:
            self.session_start_time = min(all_timestamps)
            print(f"Session start time: {self.session_start_time}")
        else:
            self.session_start_time = 0
            print("Warning: No valid timestamps found, using 0 as session start")
        
        self.session_start_time_initialized = True
    
    def to_relative_timestamp(self, absolute_timestamp: int) -> float:
        """Convert absolute Unix timestamp to milliseconds since session start."""
        if not self.session_start_time_initialized:
            raise ValueError("Session timing not initialized. Call initialize_session_timing() first.")
        
        if absolute_timestamp <= 0:
            return 0.0
        
        relative_ms = absolute_timestamp - self.session_start_time
        return float(relative_ms)
    
    def extract_player_features(self, player: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract player state features using OSRS IDs directly where possible."""
        features = []
        feature_mappings = []
        
        # Position (numerical - keep as-is)
        world_x = player.get('world_x', 0)
        world_y = player.get('world_y', 0)
        features.extend([
            self.safe_float(world_x),
            self.safe_float(world_y)
        ])
        
        # Store mappings for position features
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "player_world_x", None, None, None, "world_coordinate", "Player")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "player_world_y", None, None, None, "world_coordinate", "Player")
        )
        self.current_feature_index += 1
        
        # Animation ID (numerical - use OSRS ID directly)
        animation_id = player.get('animation_id', -1)
        
        # Track the animation ID mapping
        if animation_id == 899:
            self.track_id_mapping('Player', 'player_animation_ids', animation_id, 'crafting')
        elif animation_id >= 0:
            self.track_id_mapping('Player', 'player_animation_ids', animation_id, 'Unknown Animation')
        elif animation_id == -1:
            self.track_id_mapping('Player', 'player_animation_ids', -1, 'idle')
        
        features.append(float(animation_id))
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "player_animation_id", None, None, None, "animation_id", "Player")
        )
        self.current_feature_index += 1
        
        # Movement state (boolean - keep as-is)
        is_moving = player.get('is_moving', False)
        features.append(1.0 if is_moving else 0.0)
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "player_is_moving", None, None, None, "boolean", "Player")
        )
        self.current_feature_index += 1
        
        # Movement direction (categorical - hash this since it's dynamic text)
        direction = player.get('movement_direction', 'stationary')
        hashed_direction = self.stable_hash(direction)
        features.append(float(hashed_direction))
        # 1) Player movement direction
        self.track_id_mapping('Player', 'player_movement_direction_hashes', hashed_direction, direction)
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "player_movement_direction", None, None, None, "hashed_string", "Player")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_interaction_features(self, gamestate: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract interaction features as separate meaningful features instead of one hash."""
        features = []
        feature_mappings = []
        
        last_interaction = gamestate.get('last_interaction', {})
        gamestate_timestamp = gamestate.get('timestamp', 0)
        
        # Extract the actual data fields from last_interaction
        action = last_interaction.get('action', '')
        item_name = last_interaction.get('item_name', '')
        target = last_interaction.get('target', '')
        interaction_timestamp = last_interaction.get('timestamp', 0)
        
        # Feature 5: Action type (hash the action text since it's dynamic)
        action_hash = self.stable_hash(action)
        features.append(self.safe_float(action_hash))
        # 2) Interaction action type (hash of action string)
        self.track_id_mapping('Interaction', 'action_type_hashes', action_hash, action)
        
        # Feature 6: Item name (hash the item name since it's dynamic)
        item_hash = self.stable_hash(item_name)
        features.append(self.safe_float(item_hash))
        self.track_id_mapping('Interaction', 'item_name_hashes', item_hash, item_name)
        
        # Feature 7: Target (hash the target text since it's dynamic)
        target_hash = self.stable_hash(target)
        features.append(self.safe_float(target_hash))
        self.track_id_mapping('Interaction', 'target_hashes', target_hash, target)
        
        # Feature 8: Time since interaction (raw milliseconds)
        if interaction_timestamp > 0 and gamestate_timestamp > 0:
            time_since_interaction = gamestate_timestamp - interaction_timestamp
        else:
            time_since_interaction = 0.0
        features.append(time_since_interaction)
        
        # Store mappings for interaction features
        feature_mappings = []
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "action_type", None, None, None, "hashed_string", "Interaction")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "item_name", None, None, None, "hashed_string", "Interaction")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "target", None, None, None, "hashed_string", "Interaction")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "time_since_interaction", None, None, None, "time_ms", "Interaction")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_camera_features(self, state: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract camera position and orientation (numerical - keep as-is)."""
        features = []
        feature_mappings = []
        
        camera_x = state.get('camera_x', 0)
        camera_y = state.get('camera_y', 0)
        camera_z = state.get('camera_z', 0)
        camera_pitch = state.get('camera_pitch', 0)
        camera_yaw = state.get('camera_yaw', 0)
        
        features.extend([camera_x, camera_y, camera_z, camera_pitch, camera_yaw])
        
        # Store mappings for camera features
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "camera_x", None, None, None, "camera_coordinate", "Camera")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "camera_y", None, None, None, "camera_coordinate", "Camera")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "camera_z", None, None, None, "camera_coordinate", "Camera")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "camera_pitch", None, None, None, "angle_degrees", "Camera")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "camera_yaw", None, None, None, "angle_degrees", "Camera")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_inventory_features(self, inventory: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """Extract inventory features using item IDs directly instead of hashing."""
        features = []
        feature_mappings = []
        
        for i in range(28):
            if i < len(inventory) and inventory[i]:
                item = inventory[i]
                item_id = item.get('id', -1)
                item_name = item.get('name', 'Unknown')
                quantity = item.get('quantity', 1)
                
                # Track the item ID mapping
                if item_id >= 0:
                    self.track_id_mapping('Inventory', 'item_ids', item_id, item_name)
                
                features.append(self.safe_float(item_id))  # Use OSRS item ID directly
                
                # Store mapping for inventory feature
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"inventory_slot_{i}", 
                                          None, None, None, "item_id", "Inventory")
                )
                self.current_feature_index += 1
            else:
                features.append(-1.0)  # No item
                # Track empty slot mapping separately
                # 3) Inventory empty slot sentinel (-1 -> "empty_slot")
                self.track_id_mapping('Inventory', 'empty_slot_ids', -1, 'empty_slot')
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"inventory_slot_{i}", 
                                          None, None, None, "item_id", "Inventory")
                )
                self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_bank_features(self, gamestate: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract bank features as separate meaningful features for each material position."""
        features = []
        feature_mappings = []
        
        # Feature: Bank open status (boolean)
        bank_open = gamestate.get('bank_open', False)
        features.append(1.0 if bank_open else 0.0)
        

        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "bank_open", None, None, None, "boolean", "Bank")
        )
        self.current_feature_index += 1
        
        # Individual bank material features (20 total)
        # For each material type, create 5 features: existence, quantity, slot, x, y
        bank_item_positions = gamestate.get('bank_item_positions', {})
        
        # Define material types
        material_types = ['sapphires', 'gold_bars', 'rings', 'moulds']
        
        for material_type in material_types:
            items = bank_item_positions.get(material_type, [])
            
            if items:
                # Take the first item of this type
                item = items[0]
                
                # Feature 1: Existence (1.0 if item exists, 0.0 if not)
                features.append(1.0)
                

                
                # Feature 2: Quantity
                quantity = item.get('quantity', 0)
                features.append(self.safe_float(quantity))
                
                # Feature 3: Slot
                slot = item.get('slot', -1)
                features.append(self.safe_float(slot))
                
                # Feature 4: Canvas X coordinate
                canvas_x = item.get('canvas_x', -1)
                features.append(self.safe_float(canvas_x))
                
                # Feature 5: Canvas Y coordinate
                canvas_y = item.get('canvas_y', -1)
                features.append(self.safe_float(canvas_y))
            else:
                # Material doesn't exist - set all features to 0/-1
                features.extend([0.0, 0.0, -1.0, 0.0, 0.0])
                

        
        # Store mappings for all bank material features
        for material_type in material_types:
            items = bank_item_positions.get(material_type, [])
            
            if items:
                item = items[0]
                # Feature 1: Existence
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_exists", 
                                          None, None, None, "boolean", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 2: Quantity
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_quantity", 
                                          None, None, None, "count", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 3: Slot
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_slot", 
                                          None, None, None, "slot_id", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 4: Canvas X coordinate
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_x", 
                                          None, None, None, "screen_coordinate", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 5: Canvas Y coordinate
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_y", 
                                          None, None, None, "screen_coordinate", "Bank")
                )
                self.current_feature_index += 1
            else:
                # Material doesn't exist - set all features to 0/-1
                # Feature 1: Existence
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_exists", 
                                          None, None, None, "boolean", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 2: Quantity
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_quantity", 
                                          None, None, None, "count", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 3: Slot
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_slot", 
                                          None, None, None, "slot_id", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 4: Canvas X coordinate
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_x", 
                                          None, None, None, "screen_coordinate", "Bank")
                )
                self.current_feature_index += 1
                
                # Feature 5: Canvas Y coordinate
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_{material_type}_y", 
                                          None, None, None, "screen_coordinate", "Bank")
                )
                self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_phase_context_features(self, gamestate: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract phase context features as separate meaningful features - no normalization."""
        features = []
        feature_mappings = []
        
        # Get phase context data
        phase_context = gamestate.get('phase_context', {})
        phase_start_time = phase_context.get('phase_start_time', 0)
        cycle_phase = phase_context.get('cycle_phase', 'unknown')
        phase_duration_ms = phase_context.get('phase_duration_ms', 0)
        gamestates_in_phase = phase_context.get('gamestates_in_phase', 0)
        
        # Feature 57: Phase type (hash the phase type text since it's dynamic)
        cycle_phase = phase_context.get('cycle_phase', 'unknown')
        phase_type_hash = self.stable_hash(cycle_phase)
        features.append(self.safe_float(phase_type_hash))
        self.track_id_mapping('Phase Context', 'phase_type_hashes', phase_type_hash, cycle_phase)
        
        # Feature 58: Phase start time (relative to session start, milliseconds)
        relative_phase_start = self.to_relative_timestamp(phase_start_time)
        features.append(relative_phase_start)
        
        # Feature 60: Phase duration (convert to relative time for consistent normalization)
        # The phase duration should also be relative to session start for consistent scaling
        if phase_duration_ms > 0:
            # Convert to relative time by subtracting from current time
            current_time = gamestate.get('timestamp', 0)
            relative_phase_duration = current_time - phase_start_time
            features.append(float(relative_phase_duration))
        else:
            features.append(0.0)
        
        # Feature 61: Gamestates in phase (raw count)
        features.append(float(gamestates_in_phase))
        
        # Store mappings for phase context features
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "phase_type", None, None, None, "hashed_string", "Phase Context")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "phase_start_time", None, None, None, "time_ms", "Phase Context")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "phase_duration", None, None, None, "time_ms", "Phase Context")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "gamestates_in_phase", None, None, None, "count", "Phase Context")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_game_objects_features(self, state: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract game objects features broken down into individual features for each object."""
        features = []
        feature_mappings = []
        
        # Get all game objects, filter out null objects, and remove duplicates (same coordinates)
        all_objects = state.get('game_objects', [])
        
        # Filter out null objects and remove duplicates based on coordinates
        unique_objects = []
        seen_coords = set()
        for obj in all_objects:
            # Skip null objects
            if obj is None or obj.get('name') == 'null' or obj.get('id') is None:
                continue
            coords = (obj.get('x', 0), obj.get('y', 0))
            if coords not in seen_coords:
                unique_objects.append(obj)
                seen_coords.add(coords)
        
        # Sort by distance
        unique_objects.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        # 1. 10 closest game objects - each with 3 features (ID, x, y)
        for i in range(10):
            if i < len(unique_objects):
                obj = unique_objects[i]
                
                # Feature: Object ID (use OSRS object ID directly)
                obj_id = obj.get('id', 0)
                obj_name = obj.get('name', 'Unknown')
                
                # Track the object ID mapping
                if obj_id > 0:
                    self.track_id_mapping('Game Objects', 'object_ids', obj_id, obj_name)
                
                features.append(float(obj_id))
                
                # Feature: World X coordinate
                world_x = obj.get('x', 0)
                features.append(float(world_x))
                
                # Feature: World Y coordinate
                world_y = obj.get('y', 0)
                features.append(float(world_y))
                
                # Store mappings for game object features
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_id", 
                                          None, None, None, "object_id", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_x", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_y", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
            else:
                # No object at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_id", 
                                          None, None, None, "object_id", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_x", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_y", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
        
        # 2. 1 closest furnace - with 3 features (ID, x, y)
        furnaces = state.get('furnaces', [])
        if furnaces:
            # Sort by distance and take closest
            furnaces.sort(key=lambda f: f.get('distance', float('inf')))
            closest_furnace = furnaces[0]
            
            # Feature: Furnace ID
            furnace_id = closest_furnace.get('id', 0)
            features.append(float(furnace_id))
            
            # Feature: Furnace X coordinate
            furnace_x = closest_furnace.get('x', 0)
            features.append(float(furnace_x))
            
            # Feature: Furnace Y coordinate
            furnace_y = closest_furnace.get('y', 0)
            features.append(float(furnace_y))
            
            # Store mappings for furnace features
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_id", 
                                      None, None, None, "object_id", "Game Objects")
            )
            self.current_feature_index += 1
            
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_x", 
                                      None, None, None, "world_coordinate", "Game Objects")
            )
            self.current_feature_index += 1
            
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_y", 
                                      None, None, None, "world_coordinate", "Game Objects")
            )
            self.current_feature_index += 1
        else:
            # No furnace - fill with default values
            features.extend([0.0, 0.0, 0.0])  # id, x, y
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_id", 
                                      None, None, None, "object_id", "Game Objects")
            )
            self.current_feature_index += 1
            
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_x", 
                                      None, None, None, "world_coordinate", "Game Objects")
            )
            self.current_feature_index += 1
            
            feature_mappings.append(
                create_feature_mapping(self.current_feature_index, "closest_furnace_y", 
                                      None, None, None, "world_coordinate", "Game Objects")
            )
            self.current_feature_index += 1
        
        # 3. 3 closest bank booths - each with 3 features (ID, x, y)
        # Use game_objects section to get all bank booths in the game world
        actual_bank_booths = [obj for obj in all_objects if obj.get('name') == 'Bank booth']
        
        # Sort by distance to get closest bank booths
        actual_bank_booths.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        for i in range(3):
            if i < len(actual_bank_booths):
                booth = actual_bank_booths[i]
                
                # Feature: Booth ID
                booth_id = booth.get('id', 0)
                features.append(float(booth_id))
                
                # Feature: Booth X coordinate
                booth_x = booth.get('x', 0)
                features.append(float(booth_x))
                
                # Feature: Booth Y coordinate
                booth_y = booth.get('y', 0)
                features.append(float(booth_y))
                
                # Store mappings for bank booth features
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_id", 
                                          None, None, None, "object_id", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_x", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_y", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
            else:
                # No bank booth at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_id", 
                                          None, None, None, "object_id", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_x", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_y", 
                                          None, None, None, "world_coordinate", "Game Objects")
                )
                self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_npc_features(self, npcs: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """Extract NPC features broken down into individual features for each NPC."""
        features = []
        feature_mappings = []
        
        # Sort by distance and take closest 5
        npcs_sorted = sorted(npcs, key=lambda n: n.get('distance', float('inf')))
        
        for i in range(5):
            if i < len(npcs_sorted):
                npc = npcs_sorted[i]
                
                # Feature: NPC ID (use OSRS NPC ID directly)
                npc_id = npc.get('id', 0)
                npc_name = npc.get('name', 'Unknown')
                
                # Track the NPC ID mapping
                if npc_id > 0:
                    self.track_id_mapping('NPCs', 'npc_ids', npc_id, npc_name)
                
                features.append(float(npc_id))
                
                # Feature: NPC world X coordinate
                npc_x = npc.get('x', 0)
                features.append(float(npc_x))
                
                # Feature: NPC world Y coordinate
                npc_y = npc.get('y', 0)
                features.append(float(npc_y))
                
                # Store mappings for NPC features
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_id", 
                                          None, None, None, "npc_id", "NPCs")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_x", 
                                          None, None, None, "world_coordinate", "NPCs")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_y", 
                                          None, None, None, "world_coordinate", "NPCs")
                )
                self.current_feature_index += 1
            else:
                # No NPC at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_id", 
                                          None, None, None, "npc_id", "NPCs")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_x", 
                                          None, None, None, "world_coordinate", "NPCs")
                )
                self.current_feature_index += 1
                
                feature_mappings.append(
                    create_feature_mapping(self.current_feature_index, f"npc_{i+1}_y", 
                                          None, None, None, "world_coordinate", "NPCs")
                )
                self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_tabs_features(self, tabs: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract UI tabs features (current tab only)."""
        features = []
        feature_mappings = []
        
        # Feature 143: Current tab (raw tab ID value)
        current_tab = tabs.get('currentTab', 0)
        features.append(self.safe_float(current_tab))
        
        # Map tab IDs to actual tab names
        tab_names = {
            0: "Combat", 1: "Skills", 2: "Quests", 3: "Inventory", 4: "Equipment",
            5: "Prayer", 6: "Magic", 7: "Friends", 8: "Friends Chat", 9: "Clan Chat",
            10: "Settings", 11: "Emotes", 12: "Music", 13: "Logout"
        }
        tab_name = tab_names.get(current_tab, f"Unknown Tab {current_tab}")
        self.track_id_mapping('Tabs', 'tab_ids', current_tab, tab_name)
        
        # Store mapping for tab feature
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "current_tab", None, None, None, "tab_id", "Tabs")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_skills_features(self, skills: Dict) -> Tuple[List[float], List[Dict]]:
        """Extract skills features (numerical - keep as-is)."""
        features = []
        feature_mappings = []
        
        # Crafting level and XP (most relevant for ring crafting)
        crafting = skills.get('crafting', {})
        crafting_level = crafting.get('level', 0)
        crafting_xp = crafting.get('xp', 0)
        
        features.extend([crafting_level, crafting_xp])
        
        # Store mappings for skills features
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "crafting_level", None, None, None, "skill_level", "Skills")
        )
        self.current_feature_index += 1
        
        feature_mappings.append(
            create_feature_mapping(self.current_feature_index, "crafting_xp", None, None, None, "skill_xp", "Skills")
        )
        self.current_feature_index += 1
        
        return features, feature_mappings
    
    def extract_features_from_gamestate(self, gamestate: Dict) -> np.ndarray:
        """Extract all features from a single gamestate."""
        features = []
        
        # Reset feature index counter and mappings for this gamestate
        self.current_feature_index = 0
        self.feature_mappings = []
        
        # Extract features from each category
        player_features, player_mappings = self.extract_player_features(gamestate.get('player', {}))
        features.extend(player_features)
        self.feature_mappings.extend(player_mappings)
        
        interaction_features, interaction_mappings = self.extract_interaction_features(gamestate)
        features.extend(interaction_features)
        self.feature_mappings.extend(interaction_mappings)
        
        camera_features, camera_mappings = self.extract_camera_features(gamestate)
        features.extend(camera_features)
        self.feature_mappings.extend(camera_mappings)
        
        inventory = gamestate.get('inventory', [])
        inventory_features, inventory_mappings = self.extract_inventory_features(inventory)
        features.extend(inventory_features)
        self.feature_mappings.extend(inventory_mappings)
        
        bank_features, bank_mappings = self.extract_bank_features(gamestate)
        features.extend(bank_features)
        self.feature_mappings.extend(bank_mappings)
        
        phase_features, phase_mappings = self.extract_phase_context_features(gamestate)
        features.extend(phase_features)
        self.feature_mappings.extend(phase_mappings)
        
        game_object_features, game_object_mappings = self.extract_game_objects_features(gamestate)
        features.extend(game_object_features)
        self.feature_mappings.extend(game_object_mappings)
        
        npcs = gamestate.get('npcs', [])
        npc_features, npc_mappings = self.extract_npc_features(npcs)
        features.extend(npc_features)
        self.feature_mappings.extend(npc_mappings)
        
        tabs = gamestate.get('tabs', {})
        tabs_features, tabs_mappings = self.extract_tabs_features(tabs)
        features.extend(tabs_features)
        self.feature_mappings.extend(tabs_mappings)
        
        skills = gamestate.get('skills', {})
        skills_features, skills_mappings = self.extract_skills_features(skills)
        features.extend(skills_features)
        self.feature_mappings.extend(skills_mappings)
        
        # Feature 127: Timestamp (ms) relative to session start (used by normalizer)
        absolute_timestamp = gamestate.get('timestamp', 0)
        relative_ms = self.to_relative_timestamp(absolute_timestamp)  # uses extractor.session_start_time
        features.append(self.safe_float(relative_ms))
        self.feature_mappings.append(
            create_feature_mapping(
                self.current_feature_index,
                "timestamp",
                None, None, None,
                "time_ms",   # keep as time_ms so offline /180 normalization applies
                "System"
            )
        )
        self.current_feature_index += 1
        
        # Set the feature count based on what was actually extracted
        self.n_features = len(features)
        
        # Verify that feature indices are correct
        for i, mapping in enumerate(self.feature_mappings):
            if mapping['feature_index'] != i:
                raise ValueError(f"Feature mapping {i} has incorrect index: {mapping['feature_index']}")
        
        return np.array(features, dtype=np.float64)
    
    def get_feature_mappings(self) -> List[Dict]:
        """Get the feature mappings for this extractor."""
        return self.feature_mappings.copy()
    
    def get_id_mappings(self) -> Dict:
        """Get the ID mappings for this extractor."""
        return self.id_mappings.copy()
    
    def save_feature_mappings(self, output_file: str = "data/features/feature_mappings.json") -> None:
        """
        Save the automatically generated feature mappings to a JSON file.
        
        Args:
            output_file: Path to save the feature mappings JSON file
        """
        if not self.feature_mappings:
            raise ValueError("No feature mappings available. Extract features from a gamestate first.")
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save feature mappings
        with open(output_path, 'w') as f:
            json.dump(self.feature_mappings, f, indent=2)
        
        print(f"Feature mappings saved to {output_path}")
        print(f"Total features: {len(self.feature_mappings)}")
        
        # Print feature group summary
        group_counts = {}
        for mapping in self.feature_mappings:
            group = mapping['feature_group']
            group_counts[group] = group_counts.get(group, 0) + 1
        
        print("\nFeature Structure Summary:")
        print("=" * 50)
        for group, count in sorted(group_counts.items()):
            print(f"{group:20} : {count:3d} features")
        print(f"{'Total':20} : {len(self.feature_mappings):3d} features")
        print("=" * 50)

    def save_id_mappings(self, output_file: str = "data/05_mappings/live_id_mappings.json") -> None:
        """
        Save the automatically generated ID mappings to a JSON file.
        
        Args:
            output_file: Path to save the ID mappings JSON file
        """
        if not self.id_mappings:
            raise ValueError("No ID mappings available. Extract features from a gamestate first.")
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ID mappings
        with open(output_path, 'w') as f:
            json.dump(self.id_mappings, f, indent=2)
        



def extract_features_from_gamestate(gamestate: Dict, session_start_time: Optional[int] = None, 
                                  save_mappings: bool = True, mappings_file: str = "data/features/feature_mappings.json",
                                  save_id_mappings: bool = True, id_mappings_file: str = "data/05_mappings/live_id_mappings.json") -> Tuple[np.ndarray, List[Dict]]:
    """
    Convenience function to extract features from a single gamestate.
    
    Args:
        gamestate: Gamestate dictionary
        session_start_time: Optional session start time for relative timestamps
        save_mappings: Whether to automatically save feature mappings to file
        mappings_file: Path to save feature mappings (if save_mappings is True)
        save_id_mappings: Whether to automatically save ID mappings to file
        id_mappings_file: Path to save ID mappings (if save_id_mappings is True)
        
    Returns:
        Tuple of (features_array, feature_mappings)
    """
    extractor = FeatureExtractor()
    
    if session_start_time is not None:
        extractor.session_start_time = session_start_time
        extractor.session_start_time_initialized = True
    
    features = extractor.extract_features_from_gamestate(gamestate)
    mappings = extractor.get_feature_mappings()
    
    # Automatically save feature mappings if requested
    if save_mappings:
        extractor.save_feature_mappings(mappings_file)
    
    # Automatically save ID mappings if requested
    if save_id_mappings:
        extractor.save_id_mappings(id_mappings_file)
    
    return features, mappings
