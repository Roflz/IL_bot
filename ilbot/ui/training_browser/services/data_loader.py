"""
Data Loading Service

Loads all training data artifacts from a data root directory.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ilbot.pipeline.shared_pipeline.io_offline import (
    load_feature_mappings, load_gamestates_metadata, 
    load_raw_action_data, load_action_targets
)


@dataclass
class LoadedData:
    """Container for all loaded training data."""
    
    # Core sequences
    input_sequences: np.ndarray
    target_sequences: List[List[float]]
    
    # Mappings
    feature_mappings: List[Dict[str, Any]]
    id_mappings: Dict[str, Any]
    
    # Optional sequences
    action_input_sequences: Optional[List[List[List[float]]]] = None
    normalized_input_sequences: Optional[np.ndarray] = None
    
    # Optional normalized data
    normalized_features: Optional[np.ndarray] = None
    normalized_action_data: Optional[List[Dict[str, Any]]] = None
    
    # Optional raw data
    state_features: Optional[np.ndarray] = None
    raw_action_data: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    gamestates_metadata: Optional[List[Dict[str, Any]]] = None
    screenshot_paths: Optional[List[str]] = None


def load_all(data_root: str) -> LoadedData:
    """
    Load all training data artifacts from the specified data root.
    
    Args:
        data_root: Path to the data directory
        
    Returns:
        LoadedData object containing all available artifacts
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data cannot be loaded
    """
    data_path = Path(data_root)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    print(f"Loading training data from: {data_root}")
    
    # Load required sequences
    input_sequences_path = data_path / "04_sequences" / "input_sequences.npy"
    if not input_sequences_path.exists():
        raise FileNotFoundError(f"Required file not found: {input_sequences_path}")
    
    input_sequences = np.load(str(input_sequences_path))
    print(f"✓ Loaded input sequences: {input_sequences.shape}")
    
    # Load target sequences
    target_sequences_path = data_path / "04_sequences" / "target_sequences.json"
    if not target_sequences_path.exists():
        raise FileNotFoundError(f"Required file not found: {target_sequences_path}")
    
    with open(target_sequences_path, 'r') as f:
        target_sequences = json.load(f)
    print(f"✓ Loaded target sequences: {len(target_sequences)}")
    
    # Load feature mappings
    feature_mappings_path = data_path / "05_mappings" / "feature_mappings.json"
    if not feature_mappings_path.exists():
        raise FileNotFoundError(f"Required file not found: {feature_mappings_path}")
    
    feature_mappings = load_feature_mappings(str(feature_mappings_path))
    print(f"✓ Loaded feature mappings for {len(feature_mappings)} features")
    
    # Load ID mappings
    id_mappings_path = data_path / "05_mappings" / "id_mappings.json"
    if not id_mappings_path.exists():
        print("⚠ Warning: ID mappings not found, using empty dict")
        id_mappings = {}
    else:
        with open(id_mappings_path, 'r') as f:
            id_mappings = json.load(f)
        print(f"✓ Loaded ID mappings for items, NPCs, objects, and hashes")
    
    # Load optional normalized input sequences
    normalized_input_sequences = None
    normalized_path = data_path / "04_sequences" / "normalized_input_sequences.npy"
    if normalized_path.exists():
        normalized_input_sequences = np.load(str(normalized_path))
        print(f"✓ Loaded normalized input sequences: {normalized_input_sequences.shape}")
    else:
        print("⚠ Note: Normalized input sequences not found")
    
    # Load optional action input sequences
    action_input_sequences = None
    action_sequences_path = data_path / "04_sequences" / "action_input_sequences.json"
    if action_sequences_path.exists():
        with open(action_sequences_path, 'r') as f:
            action_input_sequences = json.load(f)
        print(f"✓ Loaded action input sequences: {len(action_input_sequences)}")
    else:
        print("⚠ Note: Action input sequences not found")
    
    # Load optional normalized features
    normalized_features = None
    norm_features_path = data_path / "03_normalized_data" / "normalized_features.npy"
    if norm_features_path.exists():
        normalized_features = np.load(str(norm_features_path))
        print(f"✓ Loaded normalized features: {normalized_features.shape}")
    else:
        print("⚠ Note: Normalized features not found")
    
    # Load optional normalized action data
    normalized_action_data = None
    norm_action_path = data_path / "03_normalized_data" / "normalized_action_data.json"
    if norm_action_path.exists():
        with open(norm_action_path, 'r') as f:
            normalized_action_data = json.load(f)
        print(f"✓ Loaded normalized action data for {len(normalized_action_data)}")
    else:
        print("⚠ Note: Normalized action data not found")
    
    # Load optional raw features
    state_features = None
    raw_features_path = data_path / "01_raw_data" / "state_features.npy"
    if raw_features_path.exists():
        state_features = np.load(str(raw_features_path))
        print(f"✓ Loaded raw features: {state_features.shape}")
    else:
        print("⚠ Note: Raw features not found")
    
    # Load optional raw action data
    raw_action_data = None
    raw_action_path = data_path / "01_raw_data" / "raw_action_data.json"
    if raw_action_path.exists():
        raw_action_data = load_raw_action_data(str(raw_action_path))
        print(f"✓ Loaded raw action data for {len(raw_action_data)} gamestates")
    else:
        print("⚠ Note: Raw action data not found")
    
    # Load optional gamestates metadata
    gamestates_metadata = None
    metadata_path = data_path / "05_mappings" / "gamestates_metadata.json"
    if metadata_path.exists():
        gamestates_metadata = load_gamestates_metadata(str(metadata_path))
        print(f"✓ Loaded gamestates metadata: {len(gamestates_metadata)} gamestates")
    else:
        print("⚠ Note: Gamestates metadata not found")
    
    # Load optional screenshot paths
    screenshot_paths = None
    screenshots_path = data_path / "05_mappings" / "screenshot_paths.json"
    if screenshots_path.exists():
        with open(screenshots_path, 'r') as f:
            screenshot_paths = json.load(f)
        print(f"✓ Loaded screenshot paths: {len(screenshot_paths)} paths")
    else:
        print("⚠ Note: Screenshot paths not found")
    
    return LoadedData(
        input_sequences=input_sequences,
        target_sequences=target_sequences,
        action_input_sequences=action_input_sequences,
        normalized_input_sequences=normalized_input_sequences,
        normalized_features=normalized_features,
        normalized_action_data=normalized_action_data,
        state_features=state_features,
        raw_action_data=raw_action_data,
        feature_mappings=feature_mappings,
        id_mappings=id_mappings,
        gamestates_metadata=gamestates_metadata,
        screenshot_paths=screenshot_paths
    )
