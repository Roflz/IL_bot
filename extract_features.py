#!/usr/bin/env python3
"""
Comprehensive Feature Extraction for Enhanced OSRS Gamestate Data

This script extracts fixed-length feature vectors from the enhanced gamestate data
collected by the RuneLite plugin, including:
- Player state and interactions
- Inventory and bank contents
- Game world objects and NPCs
- Enhanced action tracking (last_interaction)
- Movement and positioning data

UPDATED: Uses OSRS IDs directly instead of hashing, separates categorical from numerical
features, and extracts meaningful temporal context for better model learning.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import hashlib
from tqdm import tqdm
import sys

# Add utils directory to path for key mapper
sys.path.append(str(Path(__file__).parent / "utils"))
from key_mapper import KeyboardKeyMapper

class FeatureExtractor:
    def __init__(self):
        # Data directory setup
        self.data_dir = Path("data")
        self.gamestates_dir = self.data_dir / "gamestates"
        self.main_gamestate = self.data_dir / "runelite_gamestate.json"
        self.actions_file = self.data_dir / "actions.csv"
        self.output_dir = self.data_dir / "features"
        self.output_dir.mkdir(exist_ok=True)
        
        # Session timing for relative timestamps
        self.session_start_time = None
        self.session_start_time_initialized = False
        
        # Feature groups with automatic indexing
        self.feature_groups = {
            "Player": (0, 5),                   # 5 features
            "Interaction": (5, 9),              # 4 features
            "Camera": (9, 14),
            "Inventory": (14, 42),
            "Bank": (42, 63),                   # 21 features (bank_open + 4 material types × 5 features each)
            "Phase Context": (63, 67),          # 4 features (phase_type, start_time, duration, gamestates_count)
            "Game Objects": (67, 109),          # 42 features (10 objects × 3 + 1 furnace × 3 + 3 booths × 3)
            "NPCs": (109, 124),                 # 15 features (5 NPCs × 3)
            "Tabs": (124, 125),                 # 1 feature (current tab)
            "Skills": (125, 127),               # 2 features (crafting level, xp)
            "Timestamp": (127, 128)             # 1 feature (relative to session start, milliseconds)
        }
        self.n_features = 128  # Updated: corrected feature count (distance features removed)
        
        # Feature mapping storage - stores original values for each feature
        self.feature_mappings = []
        
        # Automatic feature index counter
        self.current_feature_index = 0
        
        # ID and hash mapping tracking - feature-specific to avoid conflicts
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
    
    def track_id_mapping(self, feature_group: str, mapping_type: str, id_value: any, name_value: str):
        """Track ID mappings for items, NPCs, animations, objects, and hashes with proper feature isolation"""
        if feature_group not in self.id_mappings:
            return
            
        if mapping_type not in self.id_mappings[feature_group]:
            return
            
        if isinstance(id_value, (int, float)):
            self.id_mappings[feature_group][mapping_type][int(id_value)] = name_value
    
    def stable_hash(self, text: str, max_val: int = 100000) -> int:
        """Create a stable hash for text values that don't have built-in IDs."""
        if not text:
            return 0
        
        # Create a hash and track the mapping
        hash_value = hash(text) % max_val
        self.track_id_mapping('Global', 'hash_mappings', hash_value, text)
        
        return hash_value
    
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
            print(f"Session start time (human readable): {pd.to_datetime(self.session_start_time, unit='ms')}")
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
    
    def create_feature_mapping(self, feature_index: int, feature_name: str, 
                              original_value, processed_value: float, 
                              context: str, data_type: str) -> Dict:
        """Create a simplified feature mapping entry with only essential information."""
        return {
            'feature_index': feature_index,
            'feature_name': feature_name,
            'data_type': data_type,
            'feature_group': self.get_feature_group_for_index(feature_index)
        }
    
    def extract_player_features(self, player: Dict) -> List[float]:
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
        feature_mappings.extend([
            self.create_feature_mapping(self.current_feature_index, "player_world_x", None, None, None, "world_coordinate"),
            self.create_feature_mapping(self.current_feature_index + 1, "player_world_y", None, None, None, "world_coordinate")
        ])
        self.current_feature_index += 2
        
        # Animation ID (numerical - use OSRS ID directly)
        animation_id = player.get('animation_id', -1)
        
        # Track the animation ID mapping
        if animation_id == 899:
            self.track_id_mapping('Player', 'player_animation_ids', animation_id, 'crafting')
        elif animation_id >= 0:
            self.track_id_mapping('Player', 'player_animation_ids', animation_id, 'Unknown Animation')
        elif animation_id == -1:
            # Map common negative animation IDs - this is specifically for animations, not inventory
            self.track_id_mapping('Player', 'player_animation_ids', -1, 'idle')
        
        features.append(float(animation_id))
        feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "player_animation_id", None, None, None, "animation_id")
        )
        self.current_feature_index += 1
        
        # Movement state (boolean - keep as-is)
        is_moving = player.get('is_moving', False)
        features.append(1.0 if is_moving else 0.0)
        
        # Track the movement state mapping - automatically map boolean values
        self.track_id_mapping('Player', 'player_movement_direction_hashes', 1.0, 'true')
        self.track_id_mapping('Player', 'player_movement_direction_hashes', 0.0, 'false')
        
        feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "player_is_moving", None, None, None, "boolean")
        )
        self.current_feature_index += 1
        
        # Movement direction (categorical - hash this since it's dynamic text)
        direction = player.get('movement_direction', 'stationary')
        hashed_direction = self.stable_hash(direction)
        features.append(float(hashed_direction))
        feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "player_movement_direction", None, None, None, "hashed_string")
        )
        self.current_feature_index += 1
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_interaction_features(self, gamestate: Dict) -> List[float]:
        """Extract interaction features as separate meaningful features instead of one hash."""
        features = []
        
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
        self.feature_mappings.extend([
            self.create_feature_mapping(self.current_feature_index, "action_type", None, None, None, "hashed_string"),
            self.create_feature_mapping(self.current_feature_index + 1, "item_name", None, None, None, "hashed_string"),
            self.create_feature_mapping(self.current_feature_index + 2, "target", None, None, None, "hashed_string"),
            self.create_feature_mapping(self.current_feature_index + 3, "time_since_interaction", None, None, None, "time_ms")
        ])
        self.current_feature_index += 4
        
        return features
    
    def extract_camera_features(self, state: Dict) -> List[float]:
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
        feature_mappings.extend([
            self.create_feature_mapping(self.current_feature_index, "camera_x", None, None, None, "camera_coordinate"),
            self.create_feature_mapping(self.current_feature_index + 1, "camera_y", None, None, None, "camera_coordinate"),
            self.create_feature_mapping(self.current_feature_index + 2, "camera_z", None, None, None, "camera_coordinate"),
            self.create_feature_mapping(self.current_feature_index + 3, "camera_pitch", None, None, None, "angle_degrees"),
            self.create_feature_mapping(self.current_feature_index + 4, "camera_yaw", None, None, None, "angle_degrees")
        ])
        self.current_feature_index += 5
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_inventory_features(self, inventory: List[Dict]) -> List[float]:
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
                    self.create_feature_mapping(self.current_feature_index, f"inventory_slot_{i}", 
                                              None, None, None, "item_id")
                )
                self.current_feature_index += 1
            else:
                features.append(-1.0)  # No item
                # Track empty slot mapping separately to avoid conflict with animation idle
                self.track_id_mapping('Inventory', 'empty_slot_ids', -1, 'empty_slot')
                feature_mappings.append(
                    self.create_feature_mapping(self.current_feature_index, f"inventory_slot_{i}", 
                                              None, None, None, "item_id")
                )
                self.current_feature_index += 1
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_bank_features(self, gamestate: Dict) -> List[float]:
        """Extract bank features as separate meaningful features for each material position."""
        features = []
        feature_mappings = []
        
        # Feature: Bank open status (boolean)
        bank_open = gamestate.get('bank_open', False)
        features.append(1.0 if bank_open else 0.0)
        
        # Track the bank open state mapping - automatically map boolean values
        self.track_id_mapping('Bank', 'boolean_states', 1.0, 'true')
        self.track_id_mapping('Bank', 'boolean_states', 0.0, 'false')
        
        feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "bank_open", None, None, None, "boolean")
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
                
                # Track the existence mapping - automatically map boolean values
                self.track_id_mapping('Bank', 'boolean_states', 1.0, 'true')
                
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
                
                # Store mappings for this material's features
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index - 4, f"{material_type}_exists", None, None, None, "boolean"),
                    self.create_feature_mapping(self.current_feature_index - 3, f"{material_type}_quantity", None, None, None, "count"),
                    self.create_feature_mapping(self.current_feature_index - 2, f"{material_type}_slot", None, None, None, "slot_id"),
                    self.create_feature_mapping(self.current_feature_index - 1, f"{material_type}_x", None, None, None, "screen_coordinate"),
                    self.create_feature_mapping(self.current_feature_index, f"{material_type}_y", None, None, None, "screen_coordinate")
                ])
            else:
                # Material doesn't exist - set all features to 0/-1
                features.extend([0.0, 0.0, -1.0, 0.0, 0.0])
                
                # Track the non-existence mapping - automatically map boolean values
                self.track_id_mapping('Bank', 'boolean_states', 0.0, 'false')
        
        # Store mappings for all bank material features
        for material_type in material_types:
            items = bank_item_positions.get(material_type, [])
            
            if items:
                item = items[0]
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"bank_{material_type}_exists", 
                                              None, None, None, "boolean"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"bank_{material_type}_quantity", 
                                              None, None, None, "count"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"bank_{material_type}_slot", 
                                              None, None, None, "slot_id"),
                    self.create_feature_mapping(self.current_feature_index + 3, f"bank_{material_type}_x", 
                                              None, None, None, "screen_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 4, f"bank_{material_type}_y", 
                                              None, None, None, "screen_coordinate")
                ])
            else:
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"bank_{material_type}_exists", 
                                              None, None, None, "boolean"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"bank_{material_type}_quantity", 
                                              None, None, None, "count"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"bank_{material_type}_slot", 
                                              None, None, None, "slot_id"),
                    self.create_feature_mapping(self.current_feature_index + 3, f"bank_{material_type}_x", 
                                              None, None, None, "screen_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 4, f"bank_{material_type}_y", 
                                              None, None, None, "screen_coordinate")
                ])
            
            self.current_feature_index += 5
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_phase_context_features(self, gamestate: Dict) -> List[float]:
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
        
        # Feature 60: Phase duration (raw milliseconds)
        features.append(float(phase_duration_ms))
        
        # Feature 61: Gamestates in phase (raw count)
        features.append(float(gamestates_in_phase))
        
        # Store mappings for phase context features
        self.feature_mappings.extend([
            self.create_feature_mapping(self.current_feature_index, "phase_type", None, None, None, "hashed_string"),
            self.create_feature_mapping(self.current_feature_index + 1, "phase_start_time", None, None, None, "time_ms"),
            self.create_feature_mapping(self.current_feature_index + 2, "phase_duration", None, None, None, "time_ms"),
            self.create_feature_mapping(self.current_feature_index + 3, "gamestates_in_phase", None, None, None, "count")
        ])
        self.current_feature_index += 4
        
        return features
    
    def extract_widget_features(self, gamestate: Dict) -> List[float]:
        """Extract widget features - these are already handled by phase context now."""
        # Widget features are now part of phase context, so return empty list
        return []
    
    def extract_game_objects_features(self, state: Dict) -> List[float]:
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
        
        # 1. 10 closest game objects - each with 4 features (ID, distance, x, y)
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
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_id", 
                                              None, None, None, "object_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"game_object_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"game_object_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
            else:
                # No object at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"game_object_{i+1}_id", 
                                              None, None, None, "object_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"game_object_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"game_object_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
        
        # 2. 1 closest furnace - with 4 features (ID, distance, x, y)
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
            feature_mappings.extend([
                self.create_feature_mapping(self.current_feature_index, "closest_furnace_id", 
                                          None, None, None, "object_id"),
                self.create_feature_mapping(self.current_feature_index + 1, "closest_furnace_x", 
                                          None, None, None, "world_coordinate"),
                self.create_feature_mapping(self.current_feature_index + 2, "closest_furnace_y", 
                                          None, None, None, "world_coordinate")
            ])
            self.current_feature_index += 3
        else:
            # No furnace - fill with default values
            features.extend([0.0, 0.0, 0.0])  # id, x, y
            feature_mappings.extend([
                self.create_feature_mapping(self.current_feature_index, "closest_furnace_id", 
                                          None, None, None, "object_id"),
                self.create_feature_mapping(self.current_feature_index + 1, "closest_furnace_x", 
                                          None, None, None, "world_coordinate"),
                self.create_feature_mapping(self.current_feature_index + 2, "closest_furnace_y", 
                                          None, None, None, "world_coordinate")
            ])
            self.current_feature_index += 3
        
        # 3. 3 closest bank booths - each with 4 features (ID, distance, x, y)
        # Use game_objects section to get all bank booths in the game world
        all_game_objects = state.get('game_objects', [])
        actual_bank_booths = [obj for obj in all_game_objects if obj.get('name') == 'Bank booth']
        
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
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_id", 
                                              None, None, None, "object_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"bank_booth_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"bank_booth_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
            else:
                # No bank booth at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"bank_booth_{i+1}_id", 
                                              None, None, None, "object_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"bank_booth_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"bank_booth_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_npc_features(self, npcs: List[Dict]) -> List[float]:
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
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"npc_{i+1}_id", 
                                              None, None, None, "npc_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"npc_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"npc_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
            else:
                # No NPC at this position - fill with default values
                features.extend([0.0, 0.0, 0.0])  # id, x, y
                feature_mappings.extend([
                    self.create_feature_mapping(self.current_feature_index, f"npc_{i+1}_id", 
                                              None, None, None, "npc_id"),
                    self.create_feature_mapping(self.current_feature_index + 1, f"npc_{i+1}_x", 
                                              None, None, None, "world_coordinate"),
                    self.create_feature_mapping(self.current_feature_index + 2, f"npc_{i+1}_y", 
                                              None, None, None, "world_coordinate")
                ])
                self.current_feature_index += 3
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_tabs_features(self, tabs: Dict) -> List[float]:
        """Extract UI tabs features (current tab only)."""
        features = []
        feature_mappings = []
        
        # Feature 143: Current tab (raw tab ID value)
        current_tab = tabs.get('currentTab', 0)
        features.append(self.safe_float(current_tab))
        
        # Map tab IDs to actual tab names
        tab_names = {
            0: "Combat",
            1: "Skills", 
            2: "Quests",
            3: "Inventory",
            4: "Equipment",
            5: "Prayer",
            6: "Magic",
            7: "Friends",
            8: "Friends Chat",
            9: "Clan Chat",
            10: "Settings",
            11: "Emotes",
            12: "Music",
            13: "Logout"
        }
        tab_name = tab_names.get(current_tab, f"Unknown Tab {current_tab}")
        self.track_id_mapping('Tabs', 'tab_ids', current_tab, tab_name)
        
        # Store mapping for tab feature
        self.feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "current_tab", None, None, None, "tab_id")
        )
        self.current_feature_index += 1
        

        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_skills_features(self, skills: Dict) -> List[float]:
        """Extract skills features (numerical - keep as-is)."""
        features = []
        feature_mappings = []
        
        # Crafting level and XP (most relevant for ring crafting)
        crafting = skills.get('crafting', {})
        crafting_level = crafting.get('level', 0)
        crafting_xp = crafting.get('xp', 0)
        
        features.extend([crafting_level, crafting_xp])
        
        # Store mappings for skills features
        feature_mappings.extend([
            self.create_feature_mapping(self.current_feature_index, "crafting_level", None, None, None, "skill_level"),
            self.create_feature_mapping(self.current_feature_index + 1, "crafting_xp", None, None, None, "skill_xp")
        ])
        self.current_feature_index += 2
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_action_summary_features(self, gamestate: Dict) -> List[float]:
        """Extract action summary features for compatibility with existing pipeline."""
        features = []
        feature_mappings = []
        
        # Get timestamp from gamestate to find relevant actions
        gamestate_timestamp = gamestate.get('timestamp', 0)
        
        # Load actions data if available
        actions_data = []
        if self.actions_file.exists():
            try:
                # Read actions CSV and find actions within 600ms BEFORE gamestate
                actions_df = pd.read_csv(self.actions_file)
                time_window = 600  # 600ms BEFORE gamestate timestamp
                relevant_actions = actions_df[
                    (actions_df['timestamp'] >= gamestate_timestamp - time_window) & 
                    (actions_df['timestamp'] < gamestate_timestamp)  # BEFORE, not after
                ]
                actions_data = relevant_actions.to_dict('records')
            except Exception as e:
                print(f"Warning: Could not load actions data: {e}")
                actions_data = []
        
        # Action summary features removed - raw action data provides all necessary information
        # No features added for actions - they are handled separately in raw_action_data
        
        self.feature_mappings.extend(feature_mappings)
        return features
    
    def extract_raw_action_data(self, gamestate: Dict) -> Dict:
        """Extract raw action data from the last 600ms for Option 3 implementation."""
        gamestate_timestamp = gamestate.get('timestamp', 0)
        window_start = gamestate_timestamp - 600
        
        # Load actions from CSV
        actions_data = []
        if self.actions_file.exists():
            try:
                actions_df = pd.read_csv(self.actions_file)
                relevant_actions = actions_df[
                    (actions_df['timestamp'] >= window_start) & 
                    (actions_df['timestamp'] < gamestate_timestamp)
                ].sort_values('timestamp')
                actions_data = relevant_actions.to_dict('records')
            except Exception as e:
                print(f"Warning: Could not load actions data: {e}")
        
        # Separate by action type
        mouse_movements = []
        clicks = []
        key_presses = []
        key_releases = []
        scrolls = []
        
        for action in actions_data:
            action_type = action.get('event_type', '')
            absolute_action_timestamp = action.get('timestamp', 0)
            relative_action_timestamp = self.to_relative_timestamp(absolute_action_timestamp)
            
            if action_type == 'move':
                mouse_movements.append({
                    'timestamp': relative_action_timestamp,
                    'x': action.get('x_in_window', 0),
                    'y': action.get('y_in_window', 0)
                })
            elif action_type == 'click':
                clicks.append({
                    'timestamp': relative_action_timestamp,
                    'x': action.get('x_in_window', 0),
                    'y': action.get('y_in_window', 0),
                    'button': action.get('btn', '')
                })
            elif action_type == 'key_press':
                key_presses.append({
                    'timestamp': relative_action_timestamp,
                    'key': action.get('key', '')
                })
            elif action_type == 'key_release':
                key_releases.append({
                    'timestamp': relative_action_timestamp,
                    'key': action.get('key', '')
                })
            elif action_type == 'scroll':
                scrolls.append({
                    'timestamp': relative_action_timestamp,
                    'dx': action.get('scroll_dx', 0),
                    'dy': action.get('scroll_dy', 0)
                })
        
        return {
            'mouse_movements': mouse_movements,
            'clicks': clicks,
            'key_presses': key_presses,
            'key_releases': key_releases,
            'scrolls': scrolls
        }
    
    def extract_features_from_gamestate(self, gamestate: Dict) -> np.ndarray:
        """Extract all features from a single gamestate."""
        features = []
        
        # Reset feature index counter for this gamestate
        self.current_feature_index = 0
        self.feature_mappings = []
        
        # Extract features from each category
        player_features = self.extract_player_features(gamestate.get('player', {}))
        features.extend(player_features)
        
        interaction_features = self.extract_interaction_features(gamestate)
        features.extend(interaction_features)
        
        camera_features = self.extract_camera_features(gamestate)
        features.extend(camera_features)
        
        inventory = gamestate.get('inventory', [])
        inventory_features = self.extract_inventory_features(inventory)
        features.extend(inventory_features)
        
        bank_features = self.extract_bank_features(gamestate)
        features.extend(bank_features)
        
        phase_features = self.extract_phase_context_features(gamestate)
        features.extend(phase_features)
        
        game_object_features = self.extract_game_objects_features(gamestate)
        features.extend(game_object_features)
        
        npcs = gamestate.get('npcs', [])
        npc_features = self.extract_npc_features(npcs)
        features.extend(npc_features)
        
        tabs = gamestate.get('tabs', {})
        tabs_features = self.extract_tabs_features(tabs)
        features.extend(tabs_features)
        
        skills = gamestate.get('skills', {})
        skills_features = self.extract_skills_features(skills)
        features.extend(skills_features)
        
        action_summary_features = self.extract_action_summary_features(gamestate)
        features.extend(action_summary_features)
        
        # Add timestamp feature (relative to session start, milliseconds)
        absolute_timestamp = gamestate.get('timestamp', 0)
        relative_timestamp = self.to_relative_timestamp(absolute_timestamp)
        features.append(relative_timestamp)
        
        # Store mapping for timestamp feature
        self.feature_mappings.append(
            self.create_feature_mapping(self.current_feature_index, "timestamp", None, None, None, "time_ms")
        )
        self.current_feature_index += 1
        
        # Ensure we have exactly the right number of features
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
        
        return np.array(features, dtype=np.float64)
    
    def process_all_gamestates(self):
        """Process all gamestate files and extract features."""
        gamestate_files = list(self.gamestates_dir.glob("*.json"))
        
        if not gamestate_files:
            raise ValueError(f"No gamestate files found in {self.gamestates_dir}")
        
        print(f"Processing {len(gamestate_files)} gamestate files...")
        
        # First pass: load all gamestates to initialize session timing
        print("Loading gamestates for session timing initialization...")
        all_gamestates_temp = []
        for gamestate_file in tqdm(gamestate_files, desc="Loading gamestates"):
            try:
                with open(gamestate_file, 'r') as f:
                    gamestate = json.load(f)
                all_gamestates_temp.append(gamestate)
            except Exception as e:
                print(f"Error loading {gamestate_file}: {e}")
                continue
        
        # Initialize session timing
        self.initialize_session_timing(all_gamestates_temp)
        
        all_features = []
        all_gamestates = []
        all_action_data = []
        
        # Second pass: extract features with relative timestamps
        print("Extracting features with relative timestamps...")
        for gamestate_file in tqdm(gamestate_files, desc="Extracting features"):
            try:
                # Reset feature index counter for each gamestate
                self.current_feature_index = 0
                self.feature_mappings = []
                
                with open(gamestate_file, 'r') as f:
                    gamestate = json.load(f)
                
                features = self.extract_features_from_gamestate(gamestate)
                
                if len(features) != self.n_features:
                    raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
                
                # Extract raw action data for Option 3
                action_data = self.extract_raw_action_data(gamestate)
                
                all_features.append(features)
                all_gamestates.append(gamestate)
                all_action_data.append(action_data)
                
            except Exception as e:
                print(f"Error processing {gamestate_file}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features were successfully extracted")
        
        return all_features, all_gamestates, all_action_data
    
    def save_features(self, features: List[List[float]], gamestates: List[Dict], action_data: List[Dict] = None):
        """Save extracted features and essential metadata"""
        print("Saving features...")
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float64)
        
        # Save features - this is the main output needed for training
        np.save(self.output_dir / "state_features.npy", features_array)
        print("Features saved to data\\features\\state_features.npy")
        
        # Save feature mappings - this contains the feature structure and descriptions
        with open(self.output_dir / "feature_mappings.json", 'w') as f:
            json.dump(self.feature_mappings, f, indent=2)
        print("Feature mappings saved to data\\features\\feature_mappings.json")
        
        # Save comprehensive ID mappings - this maps all IDs and hashes to their real meanings
        with open(self.output_dir / "id_mappings.json", 'w') as f:
            json.dump(self.id_mappings, f, indent=2)
        print("ID mappings saved to data\\features\\id_mappings.json")
        
        # Save gamestates metadata - this contains timestamps and other metadata
        gamestates_metadata = []
        for i, gamestate in enumerate(gamestates):
            absolute_timestamp = gamestate.get('timestamp', 0)
            relative_timestamp = self.to_relative_timestamp(absolute_timestamp)
            metadata = {
                'index': i,
                'absolute_timestamp': absolute_timestamp,
                'relative_timestamp': relative_timestamp,
                'filename': f"gamestate_{i}.json"  # or extract actual filename
            }
            gamestates_metadata.append(metadata)
        
        with open(self.output_dir / "gamestates_metadata.json", 'w') as f:
            json.dump(gamestates_metadata, f, indent=2)
        print("Gamestates metadata saved to data\\features\\gamestates_metadata.json")
        
        # Save raw action data for Option 3 implementation
        if action_data:
            with open(self.output_dir / "raw_action_data.json", 'w') as f:
                json.dump(action_data, f, indent=2)
            print("Raw action data saved to data\\features\\raw_action_data.json")
    
    def get_feature_group_for_index(self, index: int) -> str:
        """Get the feature group name for a given feature index."""
        for group_name, (start, end) in self.feature_groups.items():
            if start <= index < end:
                return group_name
        return "Unknown"
    
    def run_extraction(self):
        """Run the complete feature extraction pipeline."""
        print("Starting enhanced feature extraction...")
        
        try:
            # Process all gamestates
            features, gamestates, action_data = self.process_all_gamestates()
            
            # Save features
            self.save_features(features, gamestates, action_data)
            
            print(f"Feature extraction completed successfully!")
            print(f"Extracted {len(features)} gamestates with {self.n_features} features each")
            print(f"Features saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise

    def get_id_mappings(self) -> Dict:
        """Return mappings for common ID values to readable names."""
        return {
            "animation_ids": {
                # Common animation IDs and their meanings
                -1: "IDLE",           # Fixed: -1 should map to idle
                0: "IDLE",
                808: "WALKING",
                819: "RUNNING",
                820: "CRAFTING",
                899: "CRAFTING",       # Fixed: 899 should map to crafting
                821: "SMITHING",
                822: "BANKING",
                823: "LOGGING_OUT",
                824: "EATING",
                825: "DRINKING",
                826: "FISHING",
                827: "WOODCUTTING",
                828: "MINING"
            },
            "item_ids": {
                # Common item IDs for sapphire ring crafting and general gameplay
                -1: "NO_ITEM", # Used when slot is empty in extract_features
                0: "EMPTY", # Used when slot is empty in analyze_feature_values
                1607: "SAPPHIRE",
                1637: "SAPPHIRE_RING",  # Fixed: Correct ID for sapphire rings
                1592: "RING_MOULD",      # Fixed: Correct ID for ring moulds
                2357: "GOLD_BAR",        # Fixed: Correct ID for gold bars
                2355: "BUCKET_OF_SAND",
                1511: "LOGS",
                1513: "OAK_LOGS",
                1515: "WILLOW_LOGS",
                1517: "MAPLE_LOGS",
                1519: "YEW_LOGS",
                1521: "MAGIC_LOGS",
                434: "COPPER_ORE",
                436: "TIN_ORE",
                438: "IRON_ORE",
                440: "SILVER_ORE",
                442: "COAL",
                444: "GOLD_ORE",
                446: "MITHRIL_ORE",
                448: "ADAMANTITE_ORE",
                450: "RUNITE_ORE",
                317: "COOKED_SHRIMP", 315: "RAW_SHRIMP",
                319: "COOKED_SARDINE", 321: "RAW_SARDINE",
                325: "COOKED_HERRING", 327: "RAW_HERRING",
                329: "COOKED_TROUT", 331: "RAW_TROUT",
                333: "COOKED_SALMON", 335: "RAW_SALMON",
                339: "COOKED_TUNA", 341: "RAW_TUNA",
                347: "COOKED_BASS", 349: "RAW_BASS",
                351: "COOKED_SWORDFISH", 353: "RAW_SWORDFISH",
                355: "COOKED_LOBSTER", 377: "RAW_LOBSTER",
                379: "COOKED_MONKFISH", 384: "RAW_MONKFISH",
                385: "COOKED_SHARK", 383: "RAW_SHARK"
            },
            "tab_ids": {
                # Interface tab IDs (from currentTab feature) - Updated to match RuneLite plugin
                0: "COMBAT", 1: "STATS", 2: "QUESTS", 3: "INVENTORY", 4: "EQUIPMENT",
                5: "PRAYER", 6: "MAGIC", 7: "CLAN", 8: "FRIENDS", 9: "IGNORES",
                10: "SETTINGS", 11: "EMOTES", 12: "MUSIC", 13: "LOGOUT"
            },
            "direction_values": {
                # Movement direction mappings (from player_movement_direction feature, which is hashed)
                # These are hashes, so they need to be consistent with stable_hash in extract_features.py
                # For now, we'll use the raw values as they appear in the gamestate before hashing.
                # The analyze script will need to map the *hashed* values back to these.
                # This is a more complex mapping as the analyze script only sees the hash.
                # For now, I will map the raw string values here, and the analyze script will need to handle the hash.
                # This is a point of friction. The user wants to map the *feature values* to names.
                # If the feature value is a hash, then the mapping needs to be from hash to name.
                # I will generate the hashes for these strings here and include them in the mapping.
                self.stable_hash("NORTH"): "NORTH",
                self.stable_hash("NORTHEAST"): "NORTHEAST",
                self.stable_hash("EAST"): "EAST",
                self.stable_hash("SOUTHEAST"): "SOUTHEAST",
                self.stable_hash("SOUTH"): "SOUTH",
                self.stable_hash("SOUTHWEST"): "SOUTHWEST",
                self.stable_hash("WEST"): "WEST",
                self.stable_hash("NORTHWEST"): "NORTHWEST",
                self.stable_hash("stationary"): "STATIONARY" # Default for player_movement_direction
            },
            "game_states": {
                # Common game state values for action_type, item_name, target features (which are hashed)
                # These are hashes of dynamic text, so hardcoding is difficult
                # The analyze script will need to map the *hashed* value back to a meaningful string.
                # This is where the "meaning" in feature_context is more useful than direct value mapping.
                # For now, I will include some generic states that might be represented by hashes.
                # This is a limitation of hashing dynamic text into features.
                # The user's request for "mapping ids or hashes to specific names" is challenging for dynamic text hashes.
                # I will add a note about this in the feature_metadata.
            },
            "event_types": {
                # Action event types from actions.csv - these are now numerical features
                # Mouse movement distance, direction, click coordinates, key timing, scroll intensity, action count
                # These are all normalized numerical values (0-1) that don't need ID mappings
            },
            "phase_types": {
                # Phase types from phase_type feature (which is hashed)
                # These are hashes of dynamic text like "crafting", "banking", "walking"
                # The analyze script will need to map the *hashed* value back to the phase name.
                # For now, I will include some common phase names that might be represented by hashes.
                self.stable_hash("banking"): "BANKING_PHASE",
                self.stable_hash("crafting"): "CRAFTING_PHASE",
                self.stable_hash("walking"): "WALKING_PHASE",
                self.stable_hash("unknown"): "UNKNOWN_PHASE"
            },
            "widget_states": {
                # Widget existence states - these are now part of phase context
                # Widget features have been removed and replaced with phase context features
                True: "WIDGET_EXISTS",
                False: "WIDGET_HIDDEN",
                -1: "WIDGET_INVALID_POSITION"
            },
            "button_types": {
                # Mouse button types - these are now numerical coordinates
                # Click coordinates are normalized 0-1 values, not button types
                "left": "LEFT_CLICK",
                "right": "RIGHT_CLICK",
                "middle": "MIDDLE_CLICK",
                "": "NO_BUTTON"
            },
            "common_keys": {
                # Common keyboard keys - these are now timing features
                # Key press/release are now timing features (0-1), not key codes
                "w": "W_KEY",
                "a": "A_KEY", 
                "s": "S_KEY",
                "d": "D_KEY",
                "e": "E_KEY",
                "r": "R_KEY",
                "f": "F_KEY",
                "space": "SPACE",
                "enter": "ENTER",
                "escape": "ESCAPE",
                "tab": "TAB",
                "shift": "SHIFT",
                "ctrl": "CTRL",
                "alt": "ALT",
                "": "NO_KEY"
            }
        }

def main():
    """Main function to run feature extraction."""
    extractor = FeatureExtractor()
    extractor.run_extraction()

if __name__ == "__main__":
    main()
