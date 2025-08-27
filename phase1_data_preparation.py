#!/usr/bin/env python3
"""
Phase 1: Data Preparation & Alignment for Multi-Action Sequence Imitation Learning

This script implements the sophisticated approach where the model predicts the next 600ms
of actions (multiple actions) based on gamestate context, rather than single actions.

Key Features:
- Extracts 600ms action windows from actions.csv
- Aligns with gamestate features by timestamp
- Creates temporal sequences (10 gamestates → next 600ms actions)
- Vectorizes action sequences into training targets
- Handles variable-length action sequences per window

UPDATED: Now handles 80 features instead of 73, with meaningful numerical features
instead of hashed combinations for better model learning.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys

# Add utils directory to path for key mapper
sys.path.append(str(Path(__file__).parent / "utils"))
from key_mapper import KeyboardKeyMapper

# RobustScaler no longer needed - only timestamps are normalized

class MultiActionSequencePreparer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.gamestates_dir = self.data_dir / "gamestates"
        self.screenshots_dir = self.data_dir / "runelite_screenshots"
        self.actions_file = self.data_dir / "actions.csv"
        self.output_dir = self.data_dir / "training_data"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create separate folder for final training data
        self.final_training_dir = self.data_dir / "final_training_data"
        self.final_training_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.sequence_length = 10  # Number of gamestates for context
        self.action_window_ms = 600  # Predict next 600ms of actions
        self.max_actions_per_window = None  # No artificial limit - use all actions
        
        # Load existing features
        self.features = None
        self.feature_mappings = None
        self.gamestates_metadata = None
        self.actions_df = None
        self.raw_action_data = None
        
    def load_existing_data(self):
        """Load the features and metadata already extracted by extract_features.py"""
        print("Loading existing extracted features...")
        
        # Load features array
        features_file = self.features_dir / "state_features.npy"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        self.features = np.load(features_file)
        print(f"Loaded features: {self.features.shape}")
        
        # Verify we have 155 features per gamestate (updated from 80)
        if self.features.shape[1] != 128:
            raise ValueError(f"Expected 128 features per gamestate, got {self.features.shape[1]}")
        
        # Load feature mappings
        mappings_file = self.features_dir / "feature_mappings.json"
        if not mappings_file.exists():
            raise FileNotFoundError(f"Feature mappings not found: {mappings_file}")
        
        with open(mappings_file, 'r') as f:
            self.feature_mappings = json.load(f)
        print(f"Loaded feature mappings for {len(self.feature_mappings)} gamestates")
        
        # Load gamestates metadata
        metadata_file = self.features_dir / "gamestates_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Gamestates metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.gamestates_metadata = json.load(f)
        print(f"Loaded metadata for {len(self.gamestates_metadata)} gamestates")
        
        # Load actions data
        if not self.actions_file.exists():
            raise FileNotFoundError(f"Actions file not found: {self.actions_file}")
        
        self.actions_df = pd.read_csv(self.actions_file)
        print(f"Loaded {len(self.actions_df)} action records")
        
        # Convert timestamp to numeric for easier processing
        self.actions_df['timestamp'] = pd.to_numeric(self.actions_df['timestamp'], errors='coerce')
        self.actions_df = self.actions_df.dropna(subset=['timestamp'])
        
        # Load raw action data for Option 3
        raw_action_file = self.features_dir / "raw_action_data.json"
        if raw_action_file.exists():
            with open(raw_action_file, 'r') as f:
                self.raw_action_data = json.load(f)
            print(f"Loaded raw action data for {len(self.raw_action_data)} gamestates")
        else:
            print("Warning: Raw action data not found, will use actions.csv directly")
            self.raw_action_data = None
        
        print("Data loading completed successfully!")
        print(f"Feature structure: {self.features.shape[0]} gamestates × {self.features.shape[1]} features")
        print("Feature breakdown:")
        print("  - Player state: 5 features (world_x, world_y, animation_id, is_moving, movement_direction)")
        print("  - Interaction context: 4 features (action_type, item_name, target, time_since_interaction)")
        print("  - Camera: 5 features (x, y, z, pitch, yaw)")
        print("  - Inventory: 28 features (using item IDs directly)")
        print("  - Bank: 21 features (bank_open + 4 materials × 5 features each)")
        print("  - Phase context: 4 features (type, start_time, duration, gamestates_count)")
        print("  - Game objects: 30 features (10 objects × 3) - REMOVED distance")
        print("  - NPCs: 15 features (5 NPCs × 3) - REMOVED distance")
        print("  - Bank booths: 9 features (3 booths × 3) - REMOVED distance")
        print("  - Furnace: 3 features (1 furnace × 3) - REMOVED distance")
        print("  - Tabs: 1 feature (current tab)")
        print("  - Skills: 2 features (crafting level, xp)")
        print("  - Timestamp: 1 feature (raw timestamp)")
        print("  - Raw Action Data: Separate JSON file with detailed action sequences")
    
    def trim_data(self):
        """Smart data trimming to remove initialization artifacts and session boundaries"""
        print("Applying smart data trimming...")
        
        original_shape = self.features.shape
        print(f"Original data shape: {original_shape}")
        
        # Always trim at least the first 5 timesteps as a safety buffer
        min_trim_start = 5
        
        # Find first meaningful interaction (first action_type > 0)
        # action_type is feature index 5 based on the feature breakdown
        action_type_idx = 5
        first_meaningful_idx = 0
        
        for i in range(len(self.features)):
            if self.features[i, action_type_idx] > 0:
                first_meaningful_idx = i
                break
        
        # Use whichever is later: minimum trim or first meaningful interaction
        actual_start_idx = max(min_trim_start, first_meaningful_idx)
        
        # Cut off last 20 timesteps to remove session boundary artifacts
        trim_end = 20
        valid_end = len(self.features) - trim_end
        
        if valid_end <= actual_start_idx:
            print("Warning: Not enough data after trimming, using minimal trimming")
            valid_end = len(self.features)
        
        # Apply trimming
        self.features = self.features[actual_start_idx:valid_end]
        
        # Update gamestates metadata to match
        if self.gamestates_metadata:
            self.gamestates_metadata = self.gamestates_metadata[actual_start_idx:valid_end]
        
        # Update raw action data if it exists
        if self.raw_action_data:
            self.raw_action_data = self.raw_action_data[actual_start_idx:valid_end]
        
        # Save trimmed action data to a new file
        if self.raw_action_data:
            trimmed_action_file = self.output_dir / "trimmed_action_data.json"
            with open(trimmed_action_file, 'w') as f:
                json.dump(self.raw_action_data, f, indent=2)
            print(f"  - Saved trimmed action data to: {trimmed_action_file}")
            print(f"  - Trimmed action data: {len(self.raw_action_data)} gamestates")
        
        print(f"Data trimmed:")
        print(f"  - Minimum trim: {min_trim_start} timesteps (safety buffer)")
        print(f"  - First meaningful interaction: {first_meaningful_idx} timesteps")
        print(f"  - Actual start: {actual_start_idx} timesteps (using max of above)")
        print(f"  - End: Removed last {trim_end} timesteps (session boundaries)")
        print(f"  - Final shape: {self.features.shape}")
        print(f"  - Kept {self.features.shape[0]} gamestates for training")
        
        # Verify we still have enough data for sequences
        if self.features.shape[0] < self.sequence_length + 1:
            raise ValueError(f"Not enough data after trimming: {self.features.shape[0]} < {self.features.shape[0]} < {self.sequence_length + 1}")
        
        return actual_start_idx, trim_end
    
    def convert_trimmed_actions_to_tensors(self):
        """Convert trimmed action data to action tensors in training format"""
        print("Converting trimmed actions to action tensors...")
        
        if not self.raw_action_data:
            print("Warning: No trimmed action data available")
            return []
        
        action_tensors = []
        
        for gamestate_idx, gamestate_actions in enumerate(tqdm(self.raw_action_data, desc="Converting to action tensors")):
            # Count total actions for this gamestate
            total_actions = (len(gamestate_actions.get('mouse_movements', [])) + 
                            len(gamestate_actions.get('clicks', [])) + 
                            len(gamestate_actions.get('key_presses', [])) + 
                            len(gamestate_actions.get('key_releases', [])) + 
                            len(gamestate_actions.get('scrolls', [])))
            
            # Start building the action tensor: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, timestamp2, type2, x2, y2, button2, key2, scroll_dx2, scroll_dy2, ...]
            action_tensor = [total_actions]
            
            # Collect all actions with their metadata
            all_actions = []
            
            # Process mouse movements
            for move in gamestate_actions.get('mouse_movements', []):
                all_actions.append({
                    'timestamp': move.get('timestamp', 0),
                    'type': 0,  # 0 = move
                    'x': move.get('x', 0),
                    'y': move.get('y', 0),
                    'button': 0,  # No button for moves
                    'key': 0,     # No key for moves
                    'scroll_dx': 0,  # No scroll for moves
                    'scroll_dy': 0
                })
            
            # Process clicks
            for click in gamestate_actions.get('clicks', []):
                button_map = {'left': 1, 'right': 2, 'middle': 3}
                button = button_map.get(click.get('button', 'left'), 1)
                all_actions.append({
                    'timestamp': click.get('timestamp', 0),
                    'type': 1,  # 1 = click
                    'x': click.get('x', 0),
                    'y': click.get('y', 0),
                    'button': button,
                    'key': 0,     # No key for clicks
                    'scroll_dx': 0,  # No scroll for clicks
                    'scroll_dy': 0
                })
            
            # Process key presses
            for key_press in gamestate_actions.get('key_presses', []):
                key = key_press.get('key', '')
                key_value = KeyboardKeyMapper.map_key_to_number(key)
                all_actions.append({
                    'timestamp': key_press.get('timestamp', 0),
                    'type': 2,  # 2 = key
                    'x': 0,     # No coordinates for keys
                    'y': 0,
                    'button': 0,  # No button for keys
                    'key': key_value,
                    'scroll_dx': 0,  # No scroll for keys
                    'scroll_dy': 0
                })
            
            # Process key releases
            for key_release in gamestate_actions.get('key_releases', []):
                key = key_release.get('key', '')
                key_value = KeyboardKeyMapper.map_key_to_number(key)
                all_actions.append({
                    'timestamp': key_release.get('timestamp', 0),
                    'type': 2,  # 2 = key
                    'x': 0,     # No coordinates for keys
                    'y': 0,
                    'button': 0,  # No button for keys
                    'key': key_value,
                    'scroll_dx': 0,  # No scroll for keys
                    'scroll_dy': 0
                })
            
            # Process scrolls
            for scroll in gamestate_actions.get('scrolls', []):
                dx = scroll.get('dx', 0)
                dy = scroll.get('dy', 0)
                all_actions.append({
                    'timestamp': scroll.get('timestamp', 0),
                    'type': 4,  # 4 = scroll
                    'x': 0,     # No coordinates for scrolls
                    'y': 0,
                    'button': 0,  # No button for scrolls
                    'key': 0,     # No key for scrolls
                    'scroll_dx': dx,
                    'scroll_dy': dy
                })
            
            # Sort actions by timestamp
            all_actions.sort(key=lambda x: x['timestamp'])
            
            # Convert each action to training format features
            for action in all_actions:
                # Add raw timestamp (no timing calculation)
                action_tensor.append(action['timestamp'])
                
                # Action type
                action_tensor.append(action['type'])
                
                # Coordinates
                action_tensor.append(action['x'])
                action_tensor.append(action['y'])
                
                # Button
                action_tensor.append(action['button'])
                
                # Key hash
                action_tensor.append(action['key'])
                
                # Scroll deltas (dx and dy)
                action_tensor.append(action['scroll_dx'])
                action_tensor.append(action['scroll_dy'])
            
            action_tensors.append(action_tensor)
        
        print(f"Converted {len(action_tensors)} gamestates to action tensors")
        print(f"Action tensor format: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, timestamp2, type2, x2, y2, button2, key2, scroll_dx2, scroll_dy2, ...]")
        
        # Save the raw action tensors (not yet normalized)
        raw_tensor_file = self.output_dir / "raw_action_tensors.json"
        with open(raw_tensor_file, 'w') as f:
            json.dump(action_tensors, f, indent=2)
        print(f"Saved raw action tensors to: {raw_tensor_file}")
        
        return action_tensors
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using coordinate system grouping strategy"""
        print("Normalizing features using coordinate system grouping...")
        
        # Load feature mappings to determine which features to normalize
        feature_mappings_file = self.features_dir / "feature_mappings.json"
        if not feature_mappings_file.exists():
            print("Warning: Feature mappings not found, using basic normalization")
            # Fallback: normalize all features
            scaler = RobustScaler()
            return scaler.fit_transform(features)
        
        with open(feature_mappings_file, 'r') as f:
            feature_mappings = json.load(f)
        
        # Categorize features by coordinate system
        world_coord_features = []      # Player X/Y, NPC X/Y, Object X/Y, Camera X/Y/Z
        camera_orient_features = []    # Camera pitch, yaw only
        screen_coord_features = []     # Bank material X/Y, Action data X/Y
        other_continuous_features = [] # Timestamps, durations, etc.
        categorical_features = []      # IDs, booleans, counts, slots, etc.
        
        for mapping in feature_mappings:
            feature_idx = mapping.get('feature_index')
            feature_name = mapping.get('feature_name', '')
            data_type = mapping.get('data_type', '')
            
            if feature_idx is None:
                continue
                
            # World coordinates (including camera position)
            if (data_type == 'world_coordinate' or 
                feature_name in ['camera_x', 'camera_y', 'camera_z']):
                world_coord_features.append(feature_idx)
            
            # Camera orientation (pitch and yaw only)
            elif feature_name in ['camera_pitch', 'camera_yaw']:
                camera_orient_features.append(feature_idx)
            
            # Screen coordinates
            elif data_type == 'screen_coordinate':
                screen_coord_features.append(feature_idx)
            
            # Other continuous features
            elif data_type in ['time_ms', 'duration_ms']:
                other_continuous_features.append(feature_idx)
            
            # Categorical features (IDs, booleans, counts, slots, etc.)
            else:
                categorical_features.append(feature_idx)
        
        # Create normalized features array
        normalized_features = np.zeros_like(features, dtype=np.float64)
        
        print(f"Feature grouping:")
        print(f"  - World coordinates: {len(world_coord_features)} features")
        print(f"  - Camera orientation: {len(camera_orient_features)} features")
        print(f"  - Screen coordinates: {len(screen_coord_features)} features")
        print(f"  - Other continuous: {len(other_continuous_features)} features")
        print(f"  - Categorical: {len(categorical_features)} features")
        
        # Group 1: Skip normalizing world coordinates (they already have small ranges)
        if world_coord_features:
            print("  Skipping normalization for world coordinates (already well-scaled)...")
            print(f"    Found {len(world_coord_features)} world coordinate features")
            print(f"    Keeping raw values to preserve spatial intuition")
            
            # Just copy the raw values without normalization
            for feature_idx in world_coord_features:
                normalized_features[:, feature_idx] = features[:, feature_idx]
        
        # Group 2: Preserve camera orientation (no normalization)
        if camera_orient_features:
            print("  Preserving camera orientation (no normalization)...")
            print(f"    Found {len(camera_orient_features)} camera orientation features")
            print(f"    Keeping raw values to preserve angular relationships")
            
            # Just copy the raw values without normalization
            for feature_idx in camera_orient_features:
                normalized_features[:, feature_idx] = features[:, feature_idx]
        
        # Group 3: Preserve screen coordinates (no normalization)
        if screen_coord_features:
            print("  Preserving screen coordinates (no normalization)...")
            print(f"    Found {len(screen_coord_features)} screen coordinate features")
            print(f"    Keeping raw values to preserve UI positioning")
            
            # Just copy the raw values without normalization
            for feature_idx in screen_coord_features:
                normalized_features[:, feature_idx] = features[:, feature_idx]
        
        # Time features: Scale to 0-10000 range for 30-minute session
        time_features = []
        for mapping in feature_mappings:
            if mapping.get('data_type') == 'time_ms':
                time_features.append(mapping.get('feature_index'))
        
        if time_features:
            print("  Scaling time features to 0-10000 range for 30-minute session...")
            # 30 minutes = 1,800,000 milliseconds
            # Scale to 0-10000 range
            MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
            SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
            
            for feature_idx in time_features:
                # Scale from ms to 0-10000 range
                time_values_ms = features[:, feature_idx]
                time_values_scaled = time_values_ms * SCALE_FACTOR
                normalized_features[:, feature_idx] = time_values_scaled
                
                # Find feature name for logging
                feature_name = "unknown"
                for mapping in feature_mappings:
                    if mapping.get('feature_index') == feature_idx:
                        feature_name = mapping.get('feature_name', 'unknown')
                        break
                print(f"    Feature {feature_idx}: {feature_name} scaled to 0-10000 range")
                print(f"      Raw range: {np.min(time_values_ms):.0f} to {np.max(time_values_ms):.0f} ms")
                print(f"      Scaled range: {np.min(time_values_scaled):.1f} to {np.max(time_values_scaled):.1f}")
        
        # Other continuous features (excluding time features)
        other_continuous_non_time = [f for f in other_continuous_features if f not in time_features]
        if other_continuous_non_time:
            print("  Normalizing other continuous features individually...")
            for feature_idx in other_continuous_non_time:
                feature_values = features[:, feature_idx]
                normalized_features[:, feature_idx] = self._robust_scale_with_fallback(feature_values)
        
        # Categorical features (no normalization)
        if categorical_features:
            print("  Preserving categorical features (no normalization)...")
            for feature_idx in categorical_features:
                normalized_features[:, feature_idx] = features[:, feature_idx]
        
        print("Feature normalization completed using coordinate system grouping!")
        return normalized_features
    
    def _robust_scale_with_fallback(self, feature_values: np.ndarray) -> np.ndarray:
        """Helper method for robust scaling with fallback to std normalization"""
        try:
            scaler = RobustScaler()
            normalized_values = scaler.fit_transform(feature_values.reshape(-1, 1)).flatten()
            # Remove aggressive clipping to preserve precision - only clip extreme outliers
            normalized_values = np.clip(normalized_values, -10, 10)
            return normalized_values
        except Exception as e:
            # Fallback to standard deviation normalization
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            if std_val > 0.001:
                return (feature_values - mean_val) / std_val
            else:
                return feature_values
    
    def normalize_input_sequences(self, input_sequences: np.ndarray) -> np.ndarray:
        """Normalize input sequences using the same normalization as features"""
        print("Normalizing input sequences...")
        
        # Reshape to 2D for normalization
        original_shape = input_sequences.shape
        sequences_2d = input_sequences.reshape(-1, input_sequences.shape[-1])
        
        # Apply the same normalization
        normalized_2d = self.normalize_features(sequences_2d)
        
        # Reshape back to 3D
        normalized_sequences = normalized_2d.reshape(original_shape)
        
        print(f"Input sequences normalized: {normalized_sequences.shape}")
        return normalized_sequences
    
    def normalize_action_data(self, raw_action_data: List[Dict]) -> List[Dict]:
        """Normalize action data (timestamps only, preserve all other action features)"""
        print("Normalizing action data (timestamps only)...")
        
        if not hasattr(self, 'normalized_features') or self.normalized_features is None:
            print("Warning: No normalized features available, returning raw action data")
            return raw_action_data
        
        normalized_action_data = []
        
        for gamestate_idx, action_data in enumerate(raw_action_data):
            normalized_gamestate = {}
            
            # Normalize mouse movements (timestamp only)
            normalized_movements = []
            for move in action_data.get('mouse_movements', []):
                normalized_move = move.copy()
                
                # Normalize timestamp: Scale to 0-10000 range (consistent with gamestate features)
                MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
                SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
                raw_timestamp_ms = move.get('timestamp', 0)
                normalized_timestamp_scaled = raw_timestamp_ms * SCALE_FACTOR
                normalized_move['timestamp'] = normalized_timestamp_scaled
                
                # Preserve original screen coordinates (no normalization)
                # x and y remain as original pixel values
                
                normalized_movements.append(normalized_move)
            
            normalized_gamestate['mouse_movements'] = normalized_movements
            
            # Normalize clicks (timestamp only)
            normalized_clicks = []
            for click in action_data.get('clicks', []):
                normalized_click = click.copy()
                
                # Normalize timestamp: Scale to 0-10000 range (consistent with gamestate features)
                MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
                SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
                raw_timestamp_ms = click.get('timestamp', 0)
                normalized_timestamp_scaled = raw_timestamp_ms * SCALE_FACTOR
                normalized_click['timestamp'] = normalized_timestamp_scaled
                
                # Preserve original screen coordinates (no normalization)
                # x and y remain as original pixel values
                
                normalized_clicks.append(normalized_click)
            
            normalized_gamestate['clicks'] = normalized_clicks
            
            # Normalize key presses (timestamp only, preserve key info)
            normalized_key_presses = []
            for key_press in action_data.get('key_presses', []):
                normalized_key = key_press.copy()
                
                # Normalize timestamp: Scale to 0-10000 range (consistent with gamestate features)
                MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
                SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
                raw_timestamp_ms = key_press.get('timestamp', 0)
                normalized_timestamp_scaled = raw_timestamp_ms * SCALE_FACTOR
                normalized_key['timestamp'] = normalized_timestamp_scaled
                
                # Keep key info as-is (no normalization)
                normalized_key_presses.append(normalized_key)
            
            normalized_gamestate['key_presses'] = normalized_key_presses
            
            # Normalize key releases (timestamp only, preserve key info)
            normalized_key_releases = []
            for key_release in action_data.get('key_releases', []):
                normalized_key = key_release.copy()
                
                # Normalize timestamp: Scale to 0-10000 range (consistent with gamestate features)
                MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
                SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
                raw_timestamp_ms = key_release.get('timestamp', 0)
                normalized_timestamp_scaled = raw_timestamp_ms * SCALE_FACTOR
                normalized_key['timestamp'] = normalized_timestamp_scaled
                
                # Keep key info as-is (no normalization)
                normalized_key_releases.append(normalized_key)
            
            normalized_gamestate['key_releases'] = normalized_key_releases
            
            # Normalize scrolls (timestamp only, preserve scroll deltas)
            normalized_scrolls = []
            for scroll in action_data.get('scrolls', []):
                normalized_scroll = scroll.copy()
                
                # Normalize timestamp: Scale to 0-10000 range (consistent with gamestate features)
                MS_PER_SESSION = 1800000  # 30 minutes in milliseconds
                SCALE_FACTOR = 10000 / MS_PER_SESSION  # Scale to 0-10000 range
                raw_timestamp_ms = scroll.get('timestamp', 0)
                normalized_timestamp_scaled = raw_timestamp_ms * SCALE_FACTOR
                normalized_scroll['timestamp'] = normalized_timestamp_scaled
                
                # Keep scroll deltas as-is (no normalization)
                normalized_scrolls.append(normalized_scroll)
            
            normalized_gamestate['scrolls'] = normalized_scrolls
            
            normalized_action_data.append(normalized_gamestate)
        
        print(f"Action data normalized for {len(normalized_action_data)} gamestates")
        print(f"  - Only timestamps normalized (scaled to 0-10000 range)")
        print(f"  - Screen coordinates preserved as original pixel values")
        print(f"  - Scroll deltas preserved as original pixel values")
        print(f"  - Key information mapped using comprehensive keyboard mapping system")
        print(f"  - Button information preserved as original values")
        return normalized_action_data
    
    def extract_action_sequences(self) -> List[Dict]:
        """Extract action sequences for each gamestate timestamp"""
        print("Extracting action sequences...")
        
        action_sequences = []
        
        # Use raw action data if available, otherwise fall back to actions.csv
        if self.raw_action_data and len(self.raw_action_data) == len(self.features):
            print("Using pre-extracted raw action data...")
            
            for i, action_data in enumerate(tqdm(self.raw_action_data, desc="Processing raw action data")):
                # Convert raw action data to action sequence format
                action_sequence = {
                    'gamestate_index': i,
                    'gamestate_timestamp': self.features[i, -1],  # Last feature is timestamp
                    'action_count': len(action_data.get('mouse_movements', [])) + len(action_data.get('clicks', [])) + 
                                  len(action_data.get('key_presses', [])) + len(action_data.get('key_releases', [])) + 
                                  len(action_data.get('scrolls', [])),
                    'actions': []
                }
                
                # Convert mouse movements
                for move in action_data.get('mouse_movements', []):
                    action_sequence['actions'].append({
                        'relative_timestamp': int(move.get('timestamp', 0)),  # Already relative to 600ms window
                        'event_type': 'move',
                        'x_in_window': move.get('x', 0),
                        'y_in_window': move.get('y', 0),
                        'btn': '',
                        'key': '',
                        'scroll_dx': 0,
                        'scroll_dy': 0
                    })
                
                # Convert clicks
                for click in action_data.get('clicks', []):
                    action_sequence['actions'].append({
                        'relative_timestamp': int(click.get('timestamp', 0)),
                        'event_type': 'click',
                        'x_in_window': click.get('x', 0),
                        'y_in_window': click.get('y', 0),
                        'btn': click.get('button', ''),
                        'key': '',
                        'scroll_dx': 0,
                        'scroll_dy': 0
                    })
                
                # Convert key presses
                for key_press in action_data.get('key_presses', []):
                    action_sequence['actions'].append({
                        'relative_timestamp': int(key_press.get('timestamp', 0)),
                        'event_type': 'key_press',
                        'x_in_window': 0,
                        'y_in_window': 0,
                        'btn': '',
                        'key': key_press.get('key', ''),
                        'scroll_dx': 0,
                        'scroll_dy': 0
                    })
                
                # Convert key releases
                for key_release in action_data.get('key_releases', []):
                    action_sequence['actions'].append({
                        'relative_timestamp': int(key_release.get('timestamp', 0)),
                        'event_type': 'key_release',
                        'x_in_window': 0,
                        'y_in_window': 0,
                        'btn': '',
                        'key': key_release.get('key', ''),
                        'scroll_dx': 0,
                        'scroll_dy': 0
                    })
                
                # Convert scrolls
                for scroll in action_data.get('scrolls', []):
                    action_sequence['actions'].append({
                        'relative_timestamp': int(scroll.get('timestamp', 0)),
                        'event_type': 'scroll',
                        'x_in_window': 0,
                        'y_in_window': 0,
                        'btn': '',
                        'key': '',
                        'scroll_dx': scroll.get('dx', 0),
                        'scroll_dy': scroll.get('dy', 0)
                    })
                
                # Sort actions by timestamp
                action_sequence['actions'].sort(key=lambda x: x['relative_timestamp'])
                action_sequences.append(action_sequence)
                
        else:
            print("Using actions.csv directly...")
            
            for i, gamestate_meta in enumerate(tqdm(self.gamestates_metadata, desc="Extracting action sequences")):
                # Use absolute timestamp for action window calculation since actions.csv has absolute timestamps
                gamestate_timestamp = gamestate_meta['absolute_timestamp']
                
                # Find actions in the 600ms window BEFORE this gamestate
                window_start = gamestate_timestamp - self.action_window_ms
                window_end = gamestate_timestamp
                
                # Get actions in this window
                window_actions = self.actions_df[
                    (self.actions_df['timestamp'] >= window_start) & 
                    (self.actions_df['timestamp'] <= window_end)
                ].copy()
                
                # Sort by timestamp
                window_actions = window_actions.sort_values('timestamp')
                
                # Convert to relative timestamps (ms from window start, where 0 = 600ms ago, 600 = now)
                window_actions['relative_timestamp'] = (
                    (window_actions['timestamp'] - window_start) * 1000
                ).astype(int)
                
                # Create action sequence
                action_sequence = {
                    'gamestate_index': i,
                    'gamestate_timestamp': gamestate_timestamp,
                    'action_count': len(window_actions),
                    'actions': []
                }
                
                for _, action in window_actions.iterrows():
                    action_data = {
                        'relative_timestamp': action['relative_timestamp'],
                        'event_type': action['event_type'],
                        'x_in_window': action.get('x_in_window', 0),
                        'y_in_window': action.get('y_in_window', 0),
                        'btn': action.get('btn', ''),
                        'key': action.get('key', ''),
                        'scroll_dx': action.get('scroll_dx', 0),
                        'scroll_dy': action.get('scroll_dy', 0)
                    }
                    action_sequence['actions'].append(action_data)
                
                action_sequences.append(action_sequence)
        
        print(f"Extracted {len(action_sequences)} action sequences")
        return action_sequences
    


    
    def create_temporal_sequences(self) -> Tuple[np.ndarray, List[List[float]], List[List[List[float]]]]:
        """Create temporal sequences for training with both gamestate and action inputs"""
        print("Creating temporal sequences...")
        
        n_samples = len(self.features)
        n_sequences = n_samples - self.sequence_length - 1  # -1 because we need next gamestate for target
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough samples for sequence length {self.sequence_length}")
        
        # Input sequences: [batch, sequence_length, features]
        input_sequences = np.zeros((n_sequences, self.sequence_length, self.features.shape[1]), dtype=np.float64)
        
        # Action input sequences: [batch, sequence_length, variable_length_actions]
        action_input_sequences = []
        
        # Target action sequences: variable length (no fixed size)
        target_sequences = []
        
        # Create sequences
        for i in range(n_sequences):
            # Input: gamestates from i to i+sequence_length-1
            input_sequences[i] = self.features[i:i+self.sequence_length]
            
            # Action inputs: action sequences from i to i+sequence_length-1
            action_inputs = []
            for j in range(self.sequence_length):
                action_idx = i + j
                if action_idx < len(self.action_targets):
                    action_inputs.append(self.action_targets[action_idx])
                else:
                    action_inputs.append([])  # Empty action sequence if out of bounds
            action_input_sequences.append(action_inputs)
            
            # Target: action sequence for the NEXT gamestate after the sequence
            target_sequences.append(self.action_targets[i+self.sequence_length])
        
        print(f"Created {n_sequences} training sequences")
        print(f"Input shape: {input_sequences.shape}")
        print(f"Action input sequences: {len(action_input_sequences)} (each with {self.sequence_length} timesteps)")
        print(f"Target sequences: {len(target_sequences)} (variable lengths)")
        
        return input_sequences, target_sequences, action_input_sequences
    
    def create_screenshot_paths(self) -> List[str]:
        """Create paths to screenshots for each gamestate"""
        print("Creating screenshot paths...")
        
        screenshot_paths = []
        
        for gamestate_meta in self.gamestates_metadata:
            # Use absolute timestamp for screenshot naming to maintain uniqueness
            timestamp = gamestate_meta['absolute_timestamp']
            screenshot_name = f"{timestamp}.png"
            screenshot_path = str(self.screenshots_dir / screenshot_name)
            
            # Check if screenshot exists
            if Path(screenshot_path).exists():
                screenshot_paths.append(screenshot_path)
            else:
                screenshot_paths.append("")  # Empty string if no screenshot
        
        print(f"Created {len(screenshot_paths)} screenshot paths")
        return screenshot_paths
    
    def save_training_data(self, input_sequences: np.ndarray, target_sequences: List[List[float]], 
                          action_input_sequences: List[List[List[float]]], screenshot_paths: List[str], 
                          normalized_features: np.ndarray, normalized_input_sequences: np.ndarray, 
                          normalized_action_data: List[Dict]):
        """Save all training data including normalized versions"""
        print("Saving training data...")
        
        # Save raw numpy arrays
        np.save(self.output_dir / "input_sequences.npy", input_sequences)
        
        # Save normalized numpy arrays
        np.save(self.output_dir / "normalized_features.npy", normalized_features)
        np.save(self.output_dir / "normalized_input_sequences.npy", normalized_input_sequences)
        
        # Save variable-length targets as JSON (can't save as numpy array due to variable lengths)
        with open(self.output_dir / "target_sequences.json", 'w') as f:
            json.dump(target_sequences, f, indent=2)
        
        # Save action input sequences (10 timesteps of action history for each training sequence)
        with open(self.output_dir / "action_input_sequences.json", 'w') as f:
            json.dump(action_input_sequences, f, indent=2)
        
        # Save screenshot paths
        with open(self.output_dir / "screenshot_paths.json", 'w') as f:
            json.dump(screenshot_paths, f, indent=2)
        

        
        # Save trimmed action data (this is what the browser should use as "raw" data)
        with open(self.output_dir / "raw_action_data.json", 'w') as f:
            json.dump(self.raw_action_data, f, indent=2)
        
        # Save normalized action data
        with open(self.output_dir / "normalized_action_data.json", 'w') as f:
            json.dump(normalized_action_data, f, indent=2)
        
        # NEW: Save action data in training format for both raw and normalized versions
        print("Converting action data to training format...")
        
        # Convert raw action data to training format
        raw_training_format = self.convert_action_data_to_training_format(self.raw_action_data)
        with open(self.output_dir / "raw_action_training_format.json", 'w') as f:
            json.dump(raw_training_format, f, indent=2)
        
        # Convert normalized action data to training format
        normalized_training_format = self.convert_action_data_to_training_format(normalized_action_data)
        with open(self.output_dir / "normalized_action_training_format.json", 'w') as f:
            json.dump(normalized_training_format, f, indent=2)
        
        print(f"Saved training format action data:")
        print(f"  - raw_action_training_format.json: {len(raw_training_format)} gamestates")
        print(f"  - normalized_action_training_format.json: {len(normalized_training_format)} gamestates")
        
        # Save training metadata
        target_lengths = [len(t) for t in target_sequences]
        training_metadata = {
            'n_sequences': len(input_sequences),
            'sequence_length': self.sequence_length,
            'action_window_ms': self.action_window_ms,
            'max_actions_per_window': None,  # No artificial limit
            'input_shape': input_sequences.shape,
            'target_info': {
                'n_targets': len(target_sequences),
                'min_length': min(target_lengths),
                'max_length': max(target_lengths),
                'avg_length': sum(target_lengths) / len(target_lengths)
            },
            'feature_dimensions': self.features.shape[1],
            'data_description': {
                'input': f"Temporal sequences of {self.sequence_length} gamestates + {self.sequence_length} action sequences, each with {self.features.shape[1]} meaningful features",
                'target': f"Variable-length action sequences for next {self.action_window_ms}ms (no artificial limits) - predicting t+1 from t-9 to t-0",
                'screenshots': "Paths to corresponding screenshots for visual context",
                'normalized_data': "All data normalized using robust scaling for continuous features, categorical features preserved"
            },
            'feature_structure': {
                'total_features': 128,
                'player_state': 5,      # world_x, world_y, animation_id, is_moving, movement_direction
                'interaction_context': 4, # action_type, item_name, target, time_since_interaction
                'camera': 5,            # x, y, z, pitch, yaw
                'inventory': 28,        # 28 slots using item IDs directly
                'bank': 21,             # bank_open + 4 materials × 5 features each
                'phase_context': 4,     # phase_type, start_time, duration, gamestates_count
                'game_objects': 30,     # 10 objects × 3 (REMOVED distance)
                'npcs': 15,             # 5 NPCs × 3 (REMOVED distance)
                'bank_booths': 9,       # 3 booths × 3 (REMOVED distance)
                'furnace': 3,           # 1 furnace × 3 (REMOVED distance)
                'tabs': 1,              # current tab
                'skills': 2,            # crafting level, xp
                'timestamp': 1          # raw timestamp value
            },
            'improvements': {
                'description': "Updated to 128 features with Option 3 implementation - clean separation of game state and actions, removed 19 redundant distance features",
                'benefits': [
                    "OSRS IDs used directly for items, NPCs, and objects (better generalization)",
                    "Separate numerical features for temporal context (better learning)",
                    "Clean feature set with no redundant action summary features",
                    "Raw action data stored separately for detailed action sequences",
                    "No artificial limits on action counts per window",
                    "Phase context as separate meaningful features (game state understanding)"
                ]
            },
            'action_data_formats': {
                'description': "Action data now saved in multiple formats for easy browsing",
                'formats': [
                    "raw_action_data.json - Structured action data (original format)",
                    "normalized_action_data.json - Normalized action data (timestamps and coordinates scaled)",
                    "raw_action_training_format.json - Raw actions converted to training format (flattened list)",
                    "normalized_action_training_format.json - Normalized actions converted to training format (flattened list)",
                    "action_input_sequences.json - 10 timesteps of action history for each training sequence (NEW!)"
                ]
            }
        }
        
        with open(self.output_dir / "training_metadata.json", 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        print(f"Training data saved to {self.output_dir}")
        
        # Save final training data to separate folder
        self.save_final_training_data(input_sequences, target_sequences, action_input_sequences, 
                                    normalized_features, normalized_input_sequences, normalized_action_data)
        
        print(f"\n🎯 NEW: Action Input Sequences Created:")
        print(f"  - action_input_sequences.json: {len(action_input_sequences)} training sequences")
        print(f"  - Each sequence has {len(action_input_sequences[0]) if action_input_sequences else 0} timesteps of action history")
        print(f"  - Each timestep contains the action sequence for that gamestate")
        print(f"  - Target: Predict actions for timestep t+1 (11th timestep)")
    
    def analyze_data_distribution_from_targets(self, action_targets: List[List[float]]):
        """Analyze the distribution of actions from the training format targets"""
        print("Analyzing data distribution from action targets...")
        
        # Action counts per window
        action_counts = [int(seq[0]) for seq in action_targets if len(seq) > 0]
        
        # Action types distribution (type is at index 2, 10, 18, etc.)
        action_types = defaultdict(int)
        for seq in action_targets:
            if len(seq) > 1:  # Has at least action count + 1 action
                for i in range(1, len(seq), 8):  # Skip action count, then every 8 features
                    if i + 1 < len(seq):  # Make sure we have the type
                        action_type = int(seq[i + 1])  # type is at index i+1
                        action_types[action_type] += 1
        
        # Timing distribution (timestamp is at index 1, 9, 17, etc.)
        timings = []
        for seq in action_targets:
            if len(seq) > 1:
                for i in range(1, len(seq), 8):  # Skip action count, then every 8 features
                    if i < len(seq):  # Make sure we have the timestamp
                        timings.append(seq[i])  # timestamp is at index i
        
        # Create analysis report
        analysis = {
            'action_count_stats': {
                'mean': float(np.mean(action_counts)) if action_counts else 0,
                'std': float(np.std(action_counts)) if action_counts else 0,
                'min': int(np.min(action_counts)) if action_counts else 0,
                'max': int(np.max(action_counts)) if action_counts else 0,
                'distribution': np.bincount(action_counts).tolist() if action_counts else []
            },
            'action_type_distribution': dict(action_types),
            'timing_stats': {
                'mean': float(np.mean(timings)) if timings else 0,
                'std': float(np.std(timings)) if timings else 0,
                'min': float(np.min(timings)) if timings else 0,
                'max': float(np.max(timings)) if timings else 0
            },
            'total_sequences': len(action_targets),
            'total_actions': int(sum(action_counts)) if action_counts else 0
        }
        
        # Save analysis
        with open(self.output_dir / "data_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("Data analysis completed and saved")
        return analysis
    
    def save_final_training_data(self, input_sequences: np.ndarray, target_sequences: List[List[float]], 
                                action_input_sequences: List[List[List[float]]], normalized_features: np.ndarray,
                                normalized_input_sequences: np.ndarray, normalized_action_data: List[Dict]):
        """Save clean final training data to separate folder"""
        print("\n" + "=" * 50)
        print("SAVING FINAL TRAINING DATA")
        print("=" * 50)
        
        # Save final training data (clean, no debug files)
        np.save(self.final_training_dir / "gamestate_sequences.npy", normalized_input_sequences)
        print(f"✓ Saved gamestate sequences: {normalized_input_sequences.shape}")
        
        with open(self.final_training_dir / "action_input_sequences.json", 'w') as f:
            json.dump(action_input_sequences, f, indent=2)
        print(f"✓ Saved action input sequences: {len(action_input_sequences)} sequences")
        
        with open(self.final_training_dir / "action_targets.json", 'w') as f:
            json.dump(target_sequences, f, indent=2)
        print(f"✓ Saved action targets: {len(target_sequences)} targets")
        
        # Create clean metadata
        final_metadata = {
            'description': 'Final training data for sequence-to-sequence action prediction',
            'data_structure': {
                'n_training_sequences': len(input_sequences),
                'sequence_length': 10,  # 10 timesteps of history
                'gamestate_features': normalized_input_sequences.shape[2],
                'prediction_target': 'actions for timestep t+1 (11th timestep)'
            },
            'input_sequences': {
                'gamestate_sequences': {
                    'file': 'gamestate_sequences.npy',
                    'shape': normalized_input_sequences.shape,
                    'description': '10 timesteps of gamestate features (t-9 to t-0)'
                },
                'action_input_sequences': {
                    'file': 'action_input_sequences.json',
                    'count': len(action_input_sequences),
                    'description': '10 timesteps of action history (t-9 to t-0)'
                }
            },
            'targets': {
                'action_targets': {
                    'file': 'action_targets.json',
                    'count': len(target_sequences),
                    'description': 'Action sequences to predict for timestep t+1'
                }
            },
            'training_pattern': {
                'input': 'Given 10 gamestates + 10 action sequences',
                'output': 'Predict actions for the next timestep',
                'sequence_length': 10,
                'total_sequences': len(input_sequences)
            },
            'feature_info': {
                'gamestate_features': 128,
                'action_features_per_action': 8,
                'action_features': ['timestamp', 'type', 'x', 'y', 'button', 'key', 'scroll_dx', 'scroll_dy']
            },
            'normalization': {
                'gamestate_features': 'Coordinate system grouping (preserves spatial relationships)',
                'action_features': 'Timestamps scaled to 0-10000 range, coordinates preserved',
                'note': 'All normalization pre-computed, no on-the-fly processing needed'
            }
        }
        
        with open(self.final_training_dir / "metadata.json", 'w') as f:
            json.dump(final_metadata, f, indent=2)
        print(f"✓ Saved training metadata")
        
        print(f"\n🎯 FINAL TRAINING DATA SAVED TO: {self.final_training_dir}")
        print(f"📁 Training data folder: {self.output_dir} (debug/audit files)")
        print(f"📁 Final training folder: {self.final_training_dir} (clean training data)")
        print("=" * 50)
    
    def convert_action_data_to_training_format(self, action_data: List[Dict]) -> List[List[float]]:
        """Convert action data to the same format as target_sequences.json for training"""
        print(f"Converting {len(action_data)} gamestates of action data to training format...")
        
        training_formats = []
        
        for gamestate_idx, gamestate_actions in enumerate(tqdm(action_data, desc="Converting to training format")):
            training_sequence = []
            
            # Count total actions
            total_actions = (len(gamestate_actions.get('mouse_movements', [])) + 
                            len(gamestate_actions.get('clicks', [])) + 
                            len(gamestate_actions.get('key_presses', [])) + 
                            len(gamestate_actions.get('key_releases', [])) + 
                            len(gamestate_actions.get('scrolls', [])))
            
            # Add action count as first element
            training_sequence.append(total_actions)
            
            # Process each action type and add to sequence
            all_actions = []
            
            # Mouse movements
            for move in gamestate_actions.get('mouse_movements', []):
                all_actions.append({
                    'timestamp': move.get('timestamp', 0),
                    'type': 0,  # 0 = move
                    'x': move.get('x', 0),
                    'y': move.get('y', 0),
                    'button': 0,  # No button for moves
                    'key': 0,     # No key for moves
                    'scroll': 0   # No scroll for moves
                })
            
            # Clicks
            for click in gamestate_actions.get('clicks', []):
                button_map = {'left': 1, 'right': 2, 'middle': 3}
                button = button_map.get(click.get('button', 'left'), 1)
                all_actions.append({
                    'timestamp': click.get('timestamp', 0),
                    'type': 1,  # 1 = click
                    'x': click.get('x', 0),
                    'y': click.get('y', 0),
                    'button': button,
                    'key': 0,     # No key for clicks
                    'scroll': 0   # No scroll for clicks
                })
            
            # Key presses
            for key_press in gamestate_actions.get('key_presses', []):
                key = key_press.get('key', '')
                key_value = KeyboardKeyMapper.map_key_to_number(key)
                
                all_actions.append({
                    'timestamp': key_press.get('timestamp', 0),
                    'type': 2,  # 2 = key
                    'x': 0,     # No coordinates for keys
                    'y': 0,
                    'button': 0,  # No button for keys
                    'key': key_value,
                    'scroll': 0   # No scroll for keys
                })
            
            # Key releases
            for key_release in gamestate_actions.get('key_releases', []):
                key = key_release.get('key', '')
                key_value = KeyboardKeyMapper.map_key_to_number(key)
                
                all_actions.append({
                    'timestamp': key_release.get('timestamp', 0),
                    'type': 2,  # 2 = key
                    'x': 0,     # No coordinates for keys
                    'y': 0,
                    'button': 0,  # No button for keys
                    'key': key_value,
                    'scroll': 0   # No scroll for keys
                })
            
            # Scrolls
            for scroll in gamestate_actions.get('scrolls', []):
                dx = scroll.get('dx', 0)
                dy = scroll.get('dy', 0)
                            all_actions.append({
                'timestamp': scroll.get('timestamp', 0),
                'type': 4,  # 4 = scroll
                'x': 0,     # No coordinates for scrolls
                'y': 0,
                'button': 0,  # No button for scrolls
                'key': 0,     # No key for scrolls
                'scroll': dx   # Keep original dx value
            })
            
            # Sort actions by timestamp
            all_actions.sort(key=lambda x: x['timestamp'])
            
            # Add each action's features to the training sequence
            for action in all_actions:
                # Timestamp (normalized to 0-10000 range by normalize_action_data)
                timestamp = action['timestamp']
                training_sequence.append(timestamp)
                
                # Action type
                training_sequence.append(action['type'])
                
                # Coordinates (preserved as original pixel values)
                x = action['x']
                y = action['y']
                training_sequence.append(x)
                training_sequence.append(y)
                
                # Button
                training_sequence.append(action['button'])
                
                # Key
                training_sequence.append(action['key'])
                
                # Scroll deltas (dx and dy)
                if action['type'] == 4:  # Scroll
                    # For scrolls, use the scroll value as dx, and 0 as dy
                    # (since we're only storing one scroll value in the current structure)
                    training_sequence.append(action['scroll'])  # dx
                    training_sequence.append(0)  # dy (placeholder)
                else:
                    # For non-scroll actions, use 0 for both dx and dy
                    training_sequence.append(0)  # dx
                    training_sequence.append(0)  # dy
            
            training_formats.append(training_sequence)
        
        print(f"Converted {len(training_formats)} gamestates to training format")
        print(f"Training format structure: [action_count, timing1, type1, x1, y1, button1, key1, timing2, type2, x2, y2, button2, key2, ...]")
        print(f"Note: Timestamps normalized, keys mapped using comprehensive keyboard mapping system, coordinates/scrolls preserved as original values")
        
        return training_formats
    
    def run_preparation(self):
        """Run the complete Phase 1 data preparation"""
        print("Starting Phase 1: Multi-Action Sequence Data Preparation")
        print("=" * 60)
        
        # Initialize and display key mapping system info
        print("🔑 Initializing comprehensive keyboard key mapping system...")
        KeyboardKeyMapper.print_mapping_summary()
        
        # Test some common key mappings
        test_keys = ['a', 'w', '1', 'f1', 'space', 'enter', 'shift', 'up']
        print("\n🔑 Sample key mappings:")
        for key in test_keys:
            value = KeyboardKeyMapper.map_key_to_number(key)
            print(f"  '{key}' -> {value}")
        print("=" * 60)
        
        try:
            # Step 1: Load existing data
            self.load_existing_data()
            
            # Step 2: Trim data to remove initialization artifacts and session boundaries
            first_meaningful_idx, trim_end = self.trim_data()
            
            # Step 3: Convert trimmed actions to action tensors
            print("\n" + "=" * 40)
            print("STEP 3: CONVERTING TRIMMED ACTIONS TO TENSORS")
            print("=" * 40)
            raw_action_tensors = self.convert_trimmed_actions_to_tensors()
            print("Action tensor conversion completed successfully!")
            print("=" * 40)
            
            # Step 4: Skip action sequence extraction (we'll use training format directly)
            print("Skipping action sequence extraction - using training format directly")
            
            # Step 5: Convert action data to training format (this will be our action targets)
            print("Converting action data to training format for targets...")
            raw_training_format = self.convert_action_data_to_training_format(self.raw_action_data)
            
            # Store raw training format as action targets for now (will be normalized later)
            self.action_targets = raw_training_format
            
            # Save action targets for debugging
            action_targets_file = self.output_dir / "action_targets.json"
            with open(action_targets_file, 'w') as f:
                json.dump(self.action_targets, f, indent=2)
            print(f"Saved action targets to: {action_targets_file}")
            print(f"Action targets format: {len(self.action_targets)} gamestates with 8 features per action")
            
            # Step 6: Normalize all data for training
            print("\n" + "=" * 40)
            print("STEP 6: DATA NORMALIZATION")
            print("=" * 40)
            
            # Normalize features
            normalized_features = self.normalize_features(self.features)
            
            # Save normalized features for debugging
            np.save(self.output_dir / "normalized_features_debug.npy", normalized_features)
            print(f"Saved normalized features debug file")
            
            # Store normalized features for action normalization
            self.normalized_features = normalized_features
            
            # Normalize action data
            normalized_action_data = self.normalize_action_data(self.raw_action_data)
            
            # Save normalized action data for debugging
            with open(self.output_dir / "normalized_action_data_debug.json", 'w') as f:
                json.dump(normalized_action_data, f, indent=2)
            print(f"Saved normalized action data debug file")
            
            # Update action targets with normalized data
            normalized_training_format = self.convert_action_data_to_training_format(normalized_action_data)
            self.action_targets = normalized_training_format
            
            # Update the saved action targets file with normalized data
            with open(action_targets_file, 'w') as f:
                json.dump(self.action_targets, f, indent=2)
            print(f"Updated action targets with normalized data")
            
            print("Data normalization completed successfully!")
            print("=" * 40)
            
            # Step 7: Create temporal sequences (AFTER normalization)
            input_sequences, target_sequences, action_input_sequences = self.create_temporal_sequences()
            
            # Now normalize the input sequences
            normalized_input_sequences = self.normalize_input_sequences(input_sequences)
            
            # Save normalized input sequences for debugging
            np.save(self.output_dir / "normalized_input_sequences_debug.npy", normalized_input_sequences)
            print(f"Saved normalized input sequences debug file")
            
            # Save temporal sequences for debugging
            temporal_sequences_file = self.output_dir / "temporal_sequences.json"
            with open(temporal_sequences_file, 'w') as f:
                json.dump({
                    'input_sequences_shape': input_sequences.shape,
                    'target_sequences_count': len(target_sequences),
                    'sample_target': target_sequences[0] if target_sequences else []
                }, f, indent=2)
            print(f"Saved temporal sequences info to: {temporal_sequences_file}")
            
            # Step 8: Create screenshot paths
            screenshot_paths = self.create_screenshot_paths()
            
            # Step 9: Analyze data distribution
            analysis = self.analyze_data_distribution_from_targets(self.action_targets)
            
            # Step 10: Save training data (including normalized versions)
            self.save_training_data(input_sequences, target_sequences, action_input_sequences, screenshot_paths,
                                 normalized_features, normalized_input_sequences, normalized_action_data)
            
            print("\n" + "=" * 60)
            print("Phase 1 completed successfully!")
            print(f"Created {len(input_sequences)} training sequences")
            print(f"Each sequence: {self.sequence_length} gamestates → next {self.action_window_ms}ms of actions")
            print(f"Input features: {input_sequences.shape[2]} clean game state features per gamestate")
            print(f"Target actions: Variable length (no artificial limits)")
            print(f"Average actions per window: {analysis['action_count_stats']['mean']:.1f}")
            print(f"\n📊 NORMALIZED DATA SAVED:")
            print(f"  - normalized_features.npy: {normalized_features.shape}")
            print(f"  - normalized_input_sequences.npy: {normalized_input_sequences.shape}")
            print(f"  - normalized_action_data.json: {len(normalized_action_data)} gamestates")
            print(f"\n🎯 ACTION DATA IN TRAINING FORMAT:")
            print(f"  - raw_action_training_format.json: {len(self.raw_action_data)} gamestates")
            print(f"  - normalized_action_training_format.json: {len(normalized_action_data)} gamestates")
            print(f"\n🔍 DEBUG FILES SAVED:")
            print(f"  - action_input_sequences.json: {len(action_input_sequences)} training sequences")
            print(f"  - action_targets.json: {len(self.action_targets)} gamestates")
            print(f"  - temporal_sequences.json: {len(target_sequences)} sequences")
            print(f"  - normalized_features_debug.npy: {normalized_features.shape}")
            print(f"  - normalized_input_sequences_debug.npy: {normalized_input_sequences.shape}")
            print(f"  - normalized_action_data_debug.json: {len(normalized_action_data)} gamestates")
            print("\nFeature Improvements:")
            print("✓ Updated to 128 features with Option 3 implementation - clean separation of game state and actions, removed 19 redundant distance features")
            print("✓ Smart data trimming: removed initialization artifacts and session boundaries")
            print("✓ OSRS IDs used directly (no more hashing of items/NPCs/objects)")
            print("✓ Separate numerical features for temporal context")
            print("✓ Mouse movements and clicks as normalized coordinates")
            print("✓ Key press/release timing as normalized values")
            print("✓ Phase context as separate meaningful features")
            print("✓ Action data now saved in multiple formats for easy browsing")
            print("\nYour model will now learn:")
            print("- Spatial relationships (mouse movements from A to B)")
            print("- Temporal patterns (when actions occur relative to game state)")
            print("- Game object relationships (using actual OSRS IDs)")
            print("- Phase-based behavior patterns")
            
        except Exception as e:
            print(f"Phase 1 failed: {e}")
            raise

def main():
    """Main function to run Phase 1 data preparation"""
    preparer = MultiActionSequencePreparer()
    preparer.run_preparation()

if __name__ == "__main__":
    main()
