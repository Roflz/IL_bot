"""
Sequence Building Module for OSRS Bot Imitation Learning

This module builds temporal sequences (10-timestep rolling windows) from features
and actions, preserving the exact behavior from legacy phase1_data_preparation.py.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def create_temporal_sequences(features: np.ndarray, action_targets: List[List[float]], 
                            sequence_length: int = 10) -> Tuple[np.ndarray, List[List[float]], List[List[List[float]]]]:
    """
    Create temporal sequences for training with both gamestate and action inputs.
    
    Args:
        features: Array of shape (n_gamestates, n_features) where n_features=128
        action_targets: List of action target sequences in training format
        sequence_length: Number of timesteps for context (default: 10)
        
    Returns:
        Tuple of:
        - input_sequences: Array of shape (n_sequences, sequence_length, n_features)
        - target_sequences: List of action target sequences for next timestep
        - action_input_sequences: List of action history sequences for context
        
    Raises:
        ValueError: If not enough samples for sequence length
    """
    print("Creating temporal sequences...")
    
    n_samples = len(features)
    n_sequences = n_samples - sequence_length - 1  # -1 because we need next gamestate for target
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough samples for sequence length {sequence_length}. "
                       f"Need at least {sequence_length + 1} samples, got {n_samples}")
    
    # Input sequences: [batch, sequence_length, features]
    input_sequences = np.zeros((n_sequences, sequence_length, features.shape[1]), dtype=np.float64)
    
    # Action input sequences: [batch, sequence_length, variable_length_actions]
    action_input_sequences = []
    
    # Target action sequences: variable length (no fixed size)
    target_sequences = []
    
    # Create sequences
    for i in range(n_sequences):
        # Input: gamestates from i to i+sequence_length-1
        input_sequences[i] = features[i:i+sequence_length]
        
        # Action inputs: action sequences from i to i+sequence_length-1
        action_inputs = []
        for j in range(sequence_length):
            action_idx = i + j
            if action_idx < len(action_targets):
                action_inputs.append(action_targets[action_idx])
            else:
                action_inputs.append([])  # Empty action sequence if out of bounds
        action_input_sequences.append(action_inputs)
        
        # Target: action sequence for the NEXT gamestate after the sequence
        target_sequences.append(action_targets[i+sequence_length])
    
    print(f"Created {n_sequences} training sequences")
    print(f"Input shape: {input_sequences.shape}")
    print(f"Action input sequences: {len(action_input_sequences)} (each with {sequence_length} timesteps)")
    print(f"Target sequences: {len(target_sequences)} (variable lengths)")
    
    return input_sequences, target_sequences, action_input_sequences


def pad_sequence_window(sequence: List[Any], target_length: int, pad_value: Any = None) -> List[Any]:
    """
    Pad a sequence to a target length.
    
    Args:
        sequence: Input sequence
        target_length: Desired length
        pad_value: Value to use for padding (default: None)
        
    Returns:
        Padded sequence of length target_length
    """
    if len(sequence) >= target_length:
        return sequence[:target_length]
    
    # Left-pad the sequence
    padding_needed = target_length - len(sequence)
    padded_sequence = [pad_value] * padding_needed + sequence
    
    return padded_sequence


def create_rolling_windows(features: np.ndarray, window_size: int = 10, 
                          step_size: int = 1) -> np.ndarray:
    """
    Create rolling windows from a feature array.
    
    Args:
        features: Array of shape (n_timesteps, n_features)
        window_size: Size of each window
        step_size: Step size between windows
        
    Returns:
        Array of shape (n_windows, window_size, n_features)
    """
    n_timesteps, n_features = features.shape
    
    if window_size > n_timesteps:
        raise ValueError(f"Window size {window_size} cannot be larger than number of timesteps {n_timesteps}")
    
    n_windows = (n_timesteps - window_size) // step_size + 1
    
    windows = np.zeros((n_windows, window_size, n_features), dtype=features.dtype)
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = features[start_idx:end_idx]
    
    return windows


def validate_sequence_structure(input_sequences: np.ndarray, target_sequences: List[List[float]], 
                              action_input_sequences: List[List[List[float]]], 
                              expected_features: int = 128, expected_sequence_length: int = 10) -> bool:
    """
    Validate the structure of created sequences.
    
    Args:
        input_sequences: Input gamestate sequences
        target_sequences: Target action sequences
        action_input_sequences: Action input sequences for context
        expected_features: Expected number of features per timestep
        expected_sequence_length: Expected sequence length
        
    Returns:
        True if validation passes, False otherwise
    """
    print("Validating sequence structure...")
    
    # Check input sequences
    if input_sequences.ndim != 3:
        print(f"Error: input_sequences should be 3D, got {input_sequences.ndim}D")
        return False
    
    n_sequences, seq_len, n_features = input_sequences.shape
    
    if seq_len != expected_sequence_length:
        print(f"Error: Expected sequence length {expected_sequence_length}, got {seq_len}")
        return False
    
    if n_features != expected_features:
        print(f"Error: Expected {expected_features} features, got {n_features}")
        return False
    
    # Check target sequences
    if len(target_sequences) != n_sequences:
        print(f"Error: Number of target sequences ({len(target_sequences)}) "
              f"doesn't match input sequences ({n_sequences})")
        return False
    
    # Check action input sequences
    if len(action_input_sequences) != n_sequences:
        print(f"Error: Number of action input sequences ({len(action_input_sequences)}) "
              f"doesn't match input sequences ({n_sequences})")
        return False
    
    for i, action_inputs in enumerate(action_input_sequences):
        if len(action_inputs) != expected_sequence_length:
            print(f"Error: Action input sequence {i} has length {len(action_inputs)}, "
                  f"expected {expected_sequence_length}")
            return False
    
    print("Sequence structure validation passed ✓")
    print(f"  - Input sequences: {input_sequences.shape}")
    print(f"  - Target sequences: {len(target_sequences)}")
    print(f"  - Action input sequences: {len(action_input_sequences)} × {expected_sequence_length}")
    
    return True


def create_sequence_metadata(input_sequences: np.ndarray, target_sequences: List[List[float]], 
                           action_input_sequences: List[List[List[float]]]) -> Dict:
    """
    Create metadata about the created sequences.
    
    Args:
        input_sequences: Input gamestate sequences
        target_sequences: Target action sequences
        action_input_sequences: Action input sequences for context
        
    Returns:
        Dictionary containing sequence metadata
    """
    n_sequences, sequence_length, n_features = input_sequences.shape
    
    # Analyze target sequence lengths
    target_lengths = [len(t) for t in target_sequences]
    
    # Analyze action input sequences
    action_input_lengths = []
    for seq in action_input_sequences:
        for timestep in seq:
            action_input_lengths.append(len(timestep))
    
    metadata = {
        'n_sequences': n_sequences,
        'sequence_length': sequence_length,
        'n_features': n_features,
        'target_info': {
            'n_targets': len(target_sequences),
            'min_length': min(target_lengths) if target_lengths else 0,
            'max_length': max(target_lengths) if target_lengths else 0,
            'avg_length': sum(target_lengths) / len(target_lengths) if target_lengths else 0,
            'length_distribution': np.bincount(target_lengths).tolist() if target_lengths else []
        },
        'action_input_info': {
            'n_action_inputs': len(action_input_sequences),
            'min_length': min(action_input_lengths) if action_input_lengths else 0,
            'max_length': max(action_input_lengths) if action_input_lengths else 0,
            'avg_length': sum(action_input_lengths) / len(action_input_lengths) if action_input_lengths else 0
        },
        'training_pattern': {
            'input': f'Given {sequence_length} gamestates + {sequence_length} action sequences',
            'output': 'Predict actions for the next timestep',
            'sequence_length': sequence_length,
            'total_sequences': n_sequences
        }
    }
    
    return metadata


def trim_sequences(features: np.ndarray, action_targets: List[List[float]], 
                  min_trim_start: int = 5, trim_end: int = 20) -> Tuple[np.ndarray, List[List[float]], int, int]:
    """
    Smart data trimming to remove initialization artifacts and session boundaries.
    
    Args:
        features: Array of features
        action_targets: List of action targets
        min_trim_start: Minimum timesteps to trim from start
        trim_end: Timesteps to trim from end
        
    Returns:
        Tuple of (trimmed_features, trimmed_action_targets, actual_start_idx, trim_end)
    """
    print("Applying smart data trimming...")
    
    original_shape = features.shape
    print(f"Original data shape: {original_shape}")
    
    # Always trim at least the first 5 timesteps as a safety buffer
    min_trim_start = max(min_trim_start, 5)
    
    # Find first meaningful interaction (first action_type > 0)
    # action_type is feature index 5 based on the feature breakdown
    action_type_idx = 5
    first_meaningful_idx = 0
    
    for i in range(len(features)):
        if features[i, action_type_idx] > 0:
            first_meaningful_idx = i
            break
    
    # Use whichever is later: minimum trim or first meaningful interaction
    actual_start_idx = max(min_trim_start, first_meaningful_idx)
    
    # Cut off last timesteps to remove session boundary artifacts
    valid_end = len(features) - trim_end
    
    if valid_end <= actual_start_idx:
        print("Warning: Not enough data after trimming, using minimal trimming")
        valid_end = len(features)
    
    # Apply trimming
    trimmed_features = features[actual_start_idx:valid_end]
    trimmed_action_targets = action_targets[actual_start_idx:valid_end]
    
    print(f"Data trimmed:")
    print(f"  - Minimum trim: {min_trim_start} timesteps (safety buffer)")
    print(f"  - First meaningful interaction: {first_meaningful_idx} timesteps")
    print(f"  - Actual start: {actual_start_idx} timesteps (using max of above)")
    print(f"  - End: Removed last {trim_end} timesteps (session boundaries)")
    print(f"  - Final shape: {trimmed_features.shape}")
    print(f"  - Kept {trimmed_features.shape[0]} gamestates for training")
    
    # Verify we still have enough data for sequences
    if trimmed_features.shape[0] < 11:  # Need at least 10 + 1 for target
        raise ValueError(f"Not enough data after trimming: {trimmed_features.shape[0]} < 11")
    
    return trimmed_features, trimmed_action_targets, actual_start_idx, trim_end


def create_screenshot_paths(gamestates: List[Dict], screenshots_dir: str = "data/runelite_screenshots") -> List[str]:
    """
    Create paths to screenshots for each gamestate.
    
    Args:
        gamestates: List of gamestate dictionaries
        screenshots_dir: Directory containing screenshots
        
    Returns:
        List of screenshot file paths
    """
    print("Creating screenshot paths...")
    
    screenshots_path = Path(screenshots_dir)
    screenshot_paths = []
    
    for gamestate in gamestates:
        # Use absolute timestamp for screenshot naming to maintain uniqueness
        timestamp = gamestate.get('timestamp', 0)
        screenshot_name = f"{timestamp}.png"
        screenshot_path = str(screenshots_path / screenshot_name)
        
        # Check if screenshot exists
        if Path(screenshot_path).exists():
            screenshot_paths.append(screenshot_path)
        else:
            screenshot_paths.append("")  # Empty string if no screenshot
    
    print(f"Created {len(screenshot_paths)} screenshot paths")
    return screenshot_paths
