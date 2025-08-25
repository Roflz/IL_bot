"""
Data Normalization Module for OSRS Bot Imitation Learning

This module handles normalization of features and action data,
preserving the exact behavior from legacy phase1_data_preparation.py.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def normalize_features(features: np.ndarray, feature_mappings_file: str = "data/features/feature_mappings.json") -> np.ndarray:
    """
    Normalize features using coordinate system grouping strategy.
    
    Args:
        features: Array of shape (n_gamestates, n_features) where n_features=128
        feature_mappings_file: Path to feature_mappings.json
        
    Returns:
        Normalized features array with same shape as input
    """
    print("Normalizing features using coordinate system grouping...")
    
    # Load feature mappings to determine which features to normalize
    mappings_path = Path(feature_mappings_file)
    if not mappings_path.exists():
        raise FileNotFoundError(f"Feature mappings file not found: {feature_mappings_file}. Cannot proceed with normalization.")
    
    with open(mappings_path, 'r') as f:
        feature_mappings = json.load(f)
    
    # Categorize features by coordinate system
    world_coord_features = []      # Player X/Y, NPC X/Y, Object X/Y, Camera X/Y/Z
    camera_orient_features = []    # Camera pitch, yaw only
    screen_coord_features = []     # Bank material X/Y, Action data X/Y
    categorical_features = []      # IDs, booleans, counts, slots, etc.
    time_features = []             # Timestamps, durations, etc.
    
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
        
        # Time features: All time-related features should be normalized consistently
        elif (data_type in ['time_ms', 'duration_ms'] or 
              feature_name in ['time_since_interaction', 'phase_start_time', 'phase_duration']):
            time_features.append(feature_idx)
        
        # Categorical features (IDs, booleans, counts, slots, etc.)
        else:
            categorical_features.append(feature_idx)
    
    # Create normalized features array
    normalized_features = np.zeros_like(features, dtype=np.float64)
    
    print(f"Feature grouping:")
    print(f"  - World coordinates: {len(world_coord_features)} features")
    print(f"  - Camera orientation: {len(camera_orient_features)} features")
    print(f"  - Screen coordinates: {len(screen_coord_features)} features")
    print(f"  - Categorical: {len(categorical_features)} features")
    print(f"  - Time features: {len(time_features)} features")
    
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
    
    # Time features: Values are already relative ms since session start. Scale by dividing by 180.
    if time_features:
        print("  Scaling time features (ms) by dividing by 180 (no re-zeroing)...")
        for feature_idx in time_features:
            raw_ms = features[:, feature_idx]
            scaled = raw_ms / 180.0
            normalized_features[:, feature_idx] = scaled
            # Log brief stats
            try:
                feature_name = next(
                    (m.get('feature_name', 'unknown') for m in feature_mappings if m.get('feature_index') == feature_idx),
                    'unknown'
                )
            except Exception:
                feature_name = 'unknown'
            print(f"    Feature {feature_idx}: {feature_name}")
            print(f"      Raw ms range: {np.min(raw_ms):.0f} to {np.max(raw_ms):.0f}")
            print(f"      Scaled (/180) range: {np.min(scaled):.6f} to {np.max(scaled):.6f}")
    
    # Categorical features (no normalization)
    if categorical_features:
        print("  Preserving categorical features (no normalization)...")
        for feature_idx in categorical_features:
            normalized_features[:, feature_idx] = features[:, feature_idx]
    
    print("Feature normalization completed using coordinate system grouping!")
    return normalized_features


def _robust_scale_with_fallback(feature_values: np.ndarray) -> np.ndarray:
    """
    Helper method for robust scaling with fallback to std normalization.
    
    Args:
        feature_values: 1D array of feature values
        
    Returns:
        Normalized feature values
    """
    try:
        # Try robust scaling (median-based)
        median_val = np.median(feature_values)
        mad_val = np.median(np.abs(feature_values - median_val))
        
        if mad_val > 1e-8:
            normalized_values = (feature_values - median_val) / mad_val
            # Remove aggressive clipping to preserve precision - only clip extreme outliers
            normalized_values = np.clip(normalized_values, -10, 10)
            return normalized_values
        else:
            # MAD too small, fall back to std normalization
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            if std_val > 0.001:
                return (feature_values - mean_val) / std_val
            else:
                return feature_values
                
    except Exception as e:
        # Fallback to standard deviation normalization
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        if std_val > 0.001:
            return (feature_values - mean_val) / std_val
        else:
            return feature_values


def normalize_input_sequences(input_sequences: np.ndarray, feature_mappings_file: str = "data/features/feature_mappings.json") -> np.ndarray:
    """
    Normalize input sequences using the same normalization as features.
    
    Args:
        input_sequences: Array of shape (n_sequences, sequence_length, n_features)
        feature_mappings_file: Path to feature_mappings.json
        
    Returns:
        Normalized input sequences with same shape
    """
    print("Normalizing input sequences...")
    
    # Reshape to 2D for normalization
    original_shape = input_sequences.shape
    sequences_2d = input_sequences.reshape(-1, input_sequences.shape[-1])
    
    # Apply the same normalization
    normalized_2d = normalize_features(sequences_2d, feature_mappings_file)
    
    # Reshape back to 3D
    normalized_sequences = normalized_2d.reshape(original_shape)
    
    print(f"Input sequences normalized: {normalized_sequences.shape}")
    return normalized_sequences


def normalize_action_data(raw_action_data: List[Dict], normalized_features: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Normalize action data (convert timestamps to relative and scale to 0-1000 range).
    
    Args:
        raw_action_data: List of raw action data dictionaries
        normalized_features: Optional normalized features for reference
        
    Returns:
        List of normalized action data dictionaries
    """
    print("Normalizing action data: converting absolute timestamps to session-relative and scaling by /180...")

    if normalized_features is None:
        print("Warning: No normalized features available, returning raw action data")
        return raw_action_data

    # Find the session start time from the first gamestate's actions
    session_start_time = None
    for action_data in raw_action_data:
        for action_type in ['mouse_movements', 'clicks', 'key_presses', 'key_releases', 'scrolls']:
            for action in action_data.get(action_type, []):
                timestamp = action.get('timestamp', 0)
                if timestamp > 0:
                    if session_start_time is None or timestamp < session_start_time:
                        session_start_time = timestamp
                    break
            if session_start_time is not None:
                break
        if session_start_time is not None:
            break
    
    if session_start_time is None:
        print("Warning: Could not determine session start time, using 0")
        session_start_time = 0

    normalized_action_data = []
    
    for gamestate_idx, action_data in enumerate(raw_action_data):
        normalized_gamestate = {}
        
        # Normalize mouse movements (divide relative ms by 180)
        normalized_movements = []
        for move in action_data.get('mouse_movements', []):
            normalized_move = move.copy()

            # Convert absolute timestamp to session-relative, then divide by 180
            absolute_timestamp = move.get('timestamp', 0)
            relative_timestamp = (absolute_timestamp - session_start_time) / 180.0
            normalized_move['timestamp'] = relative_timestamp
            
            # Preserve original screen coordinates (no normalization)
            # x and y remain as original pixel values
            
            normalized_movements.append(normalized_move)
        
        normalized_gamestate['mouse_movements'] = normalized_movements
        
        # Normalize clicks (divide relative ms by 180)
        normalized_clicks = []
        for click in action_data.get('clicks', []):
            normalized_click = click.copy()

            # Convert absolute timestamp to session-relative, then divide by 180
            absolute_timestamp = click.get('timestamp', 0)
            relative_timestamp = (absolute_timestamp - session_start_time) / 180.0
            normalized_click['timestamp'] = relative_timestamp
            
            # Preserve original screen coordinates (no normalization)
            # x and y remain as original pixel values
            
            normalized_clicks.append(normalized_click)
        
        normalized_gamestate['clicks'] = normalized_clicks
        
        # Normalize key presses (divide relative ms by 180)
        normalized_key_presses = []
        for key_press in action_data.get('key_presses', []):
            normalized_key = key_press.copy()

            # Convert absolute timestamp to session-relative, then divide by 180
            absolute_timestamp = key_press.get('timestamp', 0)
            relative_timestamp = (absolute_timestamp - session_start_time) / 180.0
            normalized_key['timestamp'] = relative_timestamp
            
            # Keep key info as-is (no normalization)
            normalized_key_presses.append(normalized_key)
        
        normalized_gamestate['key_presses'] = normalized_key_presses
        
        # Normalize key releases (divide relative ms by 180)
        normalized_key_releases = []
        for key_release in action_data.get('key_releases', []):
            normalized_key = key_release.copy()

            # Convert absolute timestamp to session-relative, then divide by 180
            absolute_timestamp = key_release.get('timestamp', 0)
            relative_timestamp = (absolute_timestamp - session_start_time) / 180.0
            normalized_key['timestamp'] = relative_timestamp
            
            # Keep key info as-is (no normalization)
            normalized_key_releases.append(normalized_key)
        
        normalized_gamestate['key_releases'] = normalized_key_releases
        
        # Normalize scrolls (divide relative ms by 180)
        normalized_scrolls = []
        for scroll in action_data.get('scrolls', []):
            normalized_scroll = scroll.copy()

            # Convert absolute timestamp to session-relative, then divide by 180
            absolute_timestamp = scroll.get('timestamp', 0)
            relative_timestamp = (absolute_timestamp - session_start_time) / 180.0
            normalized_scroll['timestamp'] = relative_timestamp
            
            # Keep scroll deltas as-is (no normalization)
            normalized_scrolls.append(normalized_scroll)
        
        normalized_gamestate['scrolls'] = normalized_scrolls
        
        normalized_action_data.append(normalized_gamestate)
    
    print(f"Action data normalized for {len(normalized_action_data)} gamestates")
    print(f"  - Timestamps converted from absolute to session-relative, then divided by 180")
    print(f"  - Screen coordinates preserved as original pixel values")
    print(f"  - Scroll deltas preserved as original pixel values")
    print(f"  - Key information preserved as original values")
    print(f"  - Button information preserved as original values")
    
    return normalized_action_data


def get_normalization_summary(features: np.ndarray, normalized_features: np.ndarray, 
                            feature_mappings_file: str = "data/features/feature_mappings.json") -> Dict:
    """
    Get a summary of the normalization process.
    
    Args:
        features: Original features array
        normalized_features: Normalized features array
        feature_mappings_file: Path to feature mappings
        
    Returns:
        Dictionary containing normalization summary
    """
    try:
        with open(feature_mappings_file, 'r') as f:
            feature_mappings = json.load(f)
    except:
        feature_mappings = []
    
    # Calculate statistics for each feature group
    feature_groups = {
        "Player": (0, 5),
        "Interaction": (5, 9),
        "Camera": (9, 14),
        "Inventory": (14, 42),
        "Bank": (42, 63),
        "Phase Context": (63, 67),
        "Game Objects": (67, 109),
        "NPCs": (109, 124),
        "Tabs": (124, 125),
        "Skills": (125, 127),
        "Timestamp": (127, 128)
    }
    
    group_stats = {}
    for group_name, (start, end) in feature_groups.items():
        if start < features.shape[1] and end <= features.shape[1]:
            orig_range = (np.min(features[:, start:end]), np.max(features[:, start:end]))
            norm_range = (np.min(normalized_features[:, start:end]), np.max(normalized_features[:, start:end]))
            
            group_stats[group_name] = {
                'original_range': orig_range,
                'normalized_range': norm_range,
                'feature_count': end - start
            }
    
    summary = {
        'normalization_strategy': 'Coordinate system grouping with selective normalization',
        'feature_groups': group_stats,
        'overall_stats': {
            'original_shape': features.shape,
            'normalized_shape': normalized_features.shape,
            'original_range': (np.min(features), np.max(features)),
            'normalized_range': (np.min(normalized_features), np.max(normalized_features))
        },
        'normalization_rules': {
            'world_coordinates': 'No normalization (preserve spatial relationships)',
            'camera_orientation': 'No normalization (preserve angular relationships)',
            'screen_coordinates': 'No normalization (preserve UI positioning)',
            'time_features': 'Convert to relative ms since session start, scale to 0-1000 range',
            'continuous_features': 'Robust scaling with fallback to z-score',
            'categorical_features': 'No normalization (preserve discrete values)'
        }
    }
    
    return summary
