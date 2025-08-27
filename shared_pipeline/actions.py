"""
Action Processing Module for OSRS Bot Imitation Learning

This module processes action sequences and converts them to training format,
preserving the exact behavior from legacy phase1_data_preparation.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .encodings import ActionEncoder


def extract_action_sequences(gamestates: List[Dict], actions_file: str = "data/actions.csv", 
                           action_window_ms: int = 600) -> List[Dict]:
    """
    Extract action sequences for each gamestate timestamp.
    
    Args:
        gamestates: List of gamestate dictionaries
        actions_file: Path to actions.csv file
        action_window_ms: Time window in milliseconds to look for actions
        
    Returns:
        List of action sequence dictionaries
    """
    print("Extracting action sequences...")
    
    # Load actions data
    actions_path = Path(actions_file)
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    
    actions_df = pd.read_csv(actions_path)
    print(f"Loaded {len(actions_df)} action records")
    
    # Convert timestamp to numeric for easier processing
    actions_df['timestamp'] = pd.to_numeric(actions_df['timestamp'], errors='coerce')
    actions_df = actions_df.dropna(subset=['timestamp'])
    
    action_sequences = []
    
    for i, gamestate in enumerate(gamestates):
        # Use absolute timestamp for action window calculation
        gamestate_timestamp = gamestate.get('timestamp', 0)
        
        # Find actions in the 600ms window BEFORE this gamestate
        window_start = gamestate_timestamp - action_window_ms
        window_end = gamestate_timestamp
        
        # Get actions in this window
        window_actions = actions_df[
            (actions_df['timestamp'] >= window_start) & 
            (actions_df['timestamp'] <= window_end)
        ].copy()
        
        # Sort by timestamp
        window_actions = window_actions.sort_values('timestamp')
        
        # Keep absolute timestamps instead of converting to relative
        # window_actions['relative_timestamp'] = (
        #     (window_actions['timestamp'] - window_start) * 1000
        # ).astype(int)
        
        # Create action sequence
        action_sequence = {
            'gamestate_index': i,
            'gamestate_timestamp': gamestate_timestamp,
            'action_count': len(window_actions),
            'actions': []
        }
        
        for _, action in window_actions.iterrows():
            action_data = {
                'timestamp': action['timestamp'],  # Use absolute timestamp
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


def flatten_action_window(actions: List[Dict], encoder: ActionEncoder) -> List[float]:
    """
    Flatten a window of actions to training format.
    
    Args:
        actions: List of action dictionaries
        encoder: ActionEncoder instance for consistent encoding
        
    Returns:
        Flattened action sequence in training format:
        [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
    """
    if not actions:
        return [0]  # No actions
    
    # Sort actions by timestamp
    sorted_actions = sorted(actions, key=lambda a: a.get('timestamp', 0))
    
    # Start with action count
    flattened_sequence = [len(sorted_actions)]
    
    # Encode each action
    for action in sorted_actions:
        # Timestamp (now absolute timestamp)
        flattened_sequence.append(float(action.get('timestamp', 0)))
        
        # Action type
        event_type = action.get('event_type', 'move')
        flattened_sequence.append(float(encoder.encode_action_type(event_type)))
        
        # Coordinates
        flattened_sequence.append(float(action.get('x_in_window', 0)))
        flattened_sequence.append(float(action.get('y_in_window', 0)))
        
        # Button
        button = action.get('btn', '')
        flattened_sequence.append(float(encoder.encode_button(button)))
        
        # Key
        key = action.get('key', '')
        flattened_sequence.append(float(encoder.encode_key(key)))
        
        # Scroll deltas
        flattened_sequence.append(float(action.get('scroll_dx', 0)))
        flattened_sequence.append(float(action.get('scroll_dy', 0)))
    
    return flattened_sequence


def convert_actions_to_training_format(action_sequences: List[Dict], encoder: ActionEncoder) -> List[List[float]]:
    """
    Convert action sequences to training format.
    
    Args:
        action_sequences: List of action sequence dictionaries
        encoder: ActionEncoder instance for consistent encoding
        
    Returns:
        List of training format action sequences
    """
    print("Converting action sequences to training format...")
    
    training_formats = []
    
    for gamestate_idx, action_sequence in enumerate(action_sequences):
        # Convert actions to training format
        training_sequence = flatten_action_window(action_sequence['actions'], encoder)
        training_formats.append(training_sequence)
    
    print(f"Converted {len(training_formats)} action sequences to training format")
    return training_formats


def extract_raw_action_data(gamestates: List[Dict], actions_file: str = "data/actions.csv") -> List[Dict]:
    """
    Extract raw action data from the last 600ms for each gamestate.
    
    Args:
        gamestates: List of gamestate dictionaries
        actions_file: Path to actions.csv file
        
    Returns:
        List of raw action data dictionaries with relative timestamps
    """
    print("Extracting raw action data...")
    
    # Load actions from CSV
    actions_path = Path(actions_file)
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    
    actions_df = pd.read_csv(actions_path)
    actions_df['timestamp'] = pd.to_numeric(actions_df['timestamp'], errors='coerce')
    actions_df = actions_df.dropna(subset=['timestamp'])
    
    # Find session start time (minimum timestamp across all actions)
    all_timestamps = actions_df['timestamp'].tolist()
    session_start_time = min(all_timestamps)
    print(f"  Session start time: {session_start_time}")
    
    raw_action_data = []
    
    for gamestate in gamestates:
        gamestate_timestamp = gamestate.get('timestamp', 0)
        window_start = gamestate_timestamp - 600
        
        # Get actions in 600ms window BEFORE gamestate
        relevant_actions = actions_df[
            (actions_df['timestamp'] >= window_start) & 
            (actions_df['timestamp'] < gamestate_timestamp)
        ].sort_values('timestamp')
        
        # Separate by action type
        mouse_movements = []
        clicks = []
        key_presses = []
        key_releases = []
        scrolls = []
        
        for _, action in relevant_actions.iterrows():
            action_type = action.get('event_type', '')
            absolute_action_timestamp = action.get('timestamp', 0)
            # Convert to relative milliseconds from session start
            relative_action_timestamp = absolute_action_timestamp - session_start_time
            
            if action_type == 'move':
                mouse_movements.append({
                    'timestamp': relative_action_timestamp,  # Use relative timestamp
                    'x': action.get('x_in_window', 0),
                    'y': action.get('y_in_window', 0)
                })
            elif action_type == 'click':
                clicks.append({
                    'timestamp': relative_action_timestamp,  # Use relative timestamp
                    'x': action.get('x_in_window', 0),
                    'y': action.get('y_in_window', 0),
                    'button': action.get('btn', '')
                })
            elif action_type == 'key_press':
                key_presses.append({
                    'timestamp': relative_action_timestamp,  # Use relative timestamp
                    'key': action.get('key', '')
                })
            elif action_type == 'key_release':
                key_releases.append({
                    'timestamp': relative_action_timestamp,  # Use relative timestamp
                    'key': action.get('key', '')
                })
            elif action_type == 'scroll':
                scrolls.append({
                    'timestamp': relative_action_timestamp,  # Use relative timestamp
                    'dx': action.get('scroll_dx', 0),
                    'dy': action.get('scroll_dy', 0)
                })
        
        raw_action_data.append({
            'mouse_movements': mouse_movements,
            'clicks': clicks,
            'key_presses': key_presses,
            'key_releases': key_releases,
            'scrolls': scrolls
        })
    
    print(f"Extracted raw action data for {len(raw_action_data)} gamestates")
    print(f"  - Timestamps converted to relative milliseconds from session start")
    return raw_action_data


def convert_raw_actions_to_tensors(raw_action_data: List[Dict], encoder: ActionEncoder) -> List[List[float]]:
    """
    Convert raw action data to action tensors in training format.
    
    Args:
        raw_action_data: List of raw action data dictionaries
        encoder: ActionEncoder instance for consistent encoding
        
    Returns:
        List of action tensors in training format
    """
    print("Converting raw actions to action tensors...")
    
    action_tensors = []
    
    for gamestate_idx, gamestate_actions in enumerate(raw_action_data):
        # Count total actions for this gamestate
        total_actions = (len(gamestate_actions.get('mouse_movements', [])) + 
                        len(gamestate_actions.get('clicks', [])) + 
                        len(gamestate_actions.get('key_presses', [])) + 
                        len(gamestate_actions.get('key_releases', [])) + 
                        len(gamestate_actions.get('scrolls', [])))
        
        # Start building the action tensor: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
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
            key_value = encoder.encode_key(key)
            all_actions.append({
                'timestamp': key_press.get('timestamp', 0),
                'type': 2,  # 2 = key_press
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
            key_value = encoder.encode_key(key)
            all_actions.append({
                'timestamp': key_release.get('timestamp', 0),
                'type': 3,  # 3 = key_release
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
    return action_tensors


def analyze_action_distribution(action_targets: List[List[float]]) -> Dict:
    """
    Analyze the distribution of actions from the training format targets.
    
    Args:
        action_targets: List of action target sequences in training format
        
    Returns:
        Dictionary containing analysis results
    """
    print("Analyzing action distribution from targets...")
    
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
    
    print("Action distribution analysis completed")
    return analysis


def convert_live_features_to_sequence_format(feature_window: np.ndarray) -> np.ndarray:
    """
    Convert live feature window from botgui to the format expected by create_temporal_sequences.
    
    Args:
        feature_window: Array of shape (10, 128) from live feature pipeline
        
    Returns:
        Array of shape (10, 128) ready for sequence processing
    """
    if feature_window.shape != (10, 128):
        raise ValueError(f"Expected feature window shape (10, 128), got {feature_window.shape}")
    
    # The live data is already in the right format, just ensure correct dtype
    return feature_window.astype(np.float64)


def convert_live_actions_to_raw_format(action_tensors: List[List[float]], 
                                     gamestate_timestamps: List[int]) -> List[Dict]:
    """
    Convert live action tensors from botgui to the raw format expected by convert_raw_actions_to_tensors.
    
    Args:
        action_tensors: List of 10 action tensors from live actions service
        gamestate_timestamps: List of 10 timestamps corresponding to each action tensor
        
    Returns:
        List of raw action data dictionaries in the format expected by existing pipeline methods
    """
    if len(action_tensors) != 10:
        raise ValueError(f"Expected 10 action tensors, got {len(action_tensors)}")
    
    if len(gamestate_timestamps) != 10:
        raise ValueError(f"Expected 10 timestamps, got {len(gamestate_timestamps)}")
    
    raw_action_data = []
    
    for i, (action_tensor, timestamp) in enumerate(zip(action_tensors, gamestate_timestamps)):
        if not action_tensor or len(action_tensor) == 0:
            # No actions in this timestep
            action_sequence = {
                'gamestate_index': i,
                'gamestate_timestamp': timestamp,
                'action_count': 0,
                'actions': []
            }
        else:
            # Parse the action tensor to extract individual actions
            actions = []
            action_count = int(action_tensor[0]) if len(action_tensor) > 0 else 0
            
            # The action tensor format is: [count, Δt_ms, type, x, y, button, key, scroll_dx, scroll_dy × count]
            if action_count > 0 and len(action_tensor) >= 8:
                # Extract the first action (simplified - you may need to adjust based on your actual tensor format)
                action_data = {
                    'timestamp': timestamp,
                    'event_type': _convert_action_type_code(int(action_tensor[2])),
                    'x_in_window': int(action_tensor[3]),
                    'y_in_window': int(action_tensor[4]),
                    'btn': _convert_button_code(int(action_tensor[5])),
                    'key': _convert_key_code(int(action_tensor[6])),
                    'scroll_dx': int(action_tensor[7]),
                    'scroll_dy': 0  # Assuming single scroll value
                }
                actions.append(action_data)
            
            action_sequence = {
                'gamestate_index': i,
                'gamestate_timestamp': timestamp,
                'action_count': action_count,
                'actions': actions
            }
        
        raw_action_data.append(action_sequence)
    
    return raw_action_data


def _convert_action_type_code(code: int) -> str:
    """Convert action type code to string"""
    action_types = {
        0: 'move',
        1: 'click', 
        2: 'key_press',
        3: 'key_release',
        4: 'scroll'
    }
    return action_types.get(code, 'unknown')


def _convert_button_code(code: int) -> str:
    """Convert button code to string"""
    button_types = {
        0: '',
        1: 'left',
        2: 'right',
        3: 'middle'
    }
    return button_types.get(code, '')


def _convert_key_code(code: int) -> str:
    """Convert key code to string"""
    if code == 0:
        return ''
    # You may want to expand this based on your actual key encoding
    return str(code)


def create_live_training_sequences(feature_window: np.ndarray, 
                                 action_tensors: List[List[float]],
                                 gamestate_timestamps: List[int]) -> Tuple[np.ndarray, List[List[float]], List[List[List[float]]]]:
    """
    Create training sequences directly from live data using existing pipeline methods.
    
    This is a convenience function that combines the conversion methods above with
    the existing sequence creation logic.
    
    Args:
        feature_window: Array of shape (10, 128) from live feature pipeline
        action_tensors: List of 10 action tensors from live actions service  
        gamestate_timestamps: List of 10 timestamps corresponding to each action tensor
        
    Returns:
        Tuple of (input_sequences, action_input_sequences, target_sequences) ready for training
    """
    # Convert live data to pipeline format
    features = convert_live_features_to_sequence_format(feature_window)
    raw_actions = convert_live_actions_to_raw_format(action_tensors, gamestate_timestamps)
    
    # Use existing pipeline methods
    from .encodings import derive_encodings_from_data
    
    # Note: You'll need to provide a path to existing training data to derive encodings
    # This is a limitation since the encoder needs to be trained on existing data
    try:
        encoder = derive_encodings_from_data("data/training_data/action_targets.json")
        action_targets = convert_raw_actions_to_tensors(raw_actions, encoder)
    except FileNotFoundError:
        # Fallback: create dummy action targets if no encoder available
        print("Warning: No action encoder found, using dummy targets")
        action_targets = [[] for _ in range(len(raw_actions))]
    
    # Create sequences using existing method
    input_sequences, action_input_sequences, target_sequences = create_temporal_sequences(
        features, action_targets, sequence_length=10
    )
    
    return input_sequences, action_input_sequences, target_sequences
