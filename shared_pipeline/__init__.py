"""
Shared Pipeline for OSRS Bot Imitation Learning

This package provides modular, pure functions for:
- Feature extraction from gamestates (128-dimensional vectors)
- Action encoding and flattening (600ms windows)
- Sequence building (10-timestep rolling windows)
- Data normalization and preprocessing
- I/O operations for offline training data

All functions preserve exact behavior from legacy scripts.
"""

from .feature_map import load_feature_mappings, validate_feature_mappings
from .features import extract_features_from_gamestate, FeatureExtractor
from .actions import (
    extract_action_sequences, flatten_action_window, convert_actions_to_training_format,
    extract_raw_action_data, convert_raw_actions_to_tensors, analyze_action_distribution,
    convert_live_features_to_sequence_format, convert_live_actions_to_raw_format, create_live_training_sequences
)
from .sequences import (
    create_temporal_sequences, pad_sequence_window, create_rolling_windows,
    validate_sequence_structure, create_sequence_metadata, trim_sequences,
    create_screenshot_paths
)
from .normalize import normalize_features, normalize_action_data, normalize_input_sequences
from .encodings import ActionEncoder, derive_encodings_from_data, derive_encodings_from_raw_actions, validate_action_encodings
from .io_offline import (
    load_gamestates, load_actions, save_training_data, load_existing_features,
    load_action_targets, save_final_training_data, extract_features_from_gamestates,
    load_gamestates_metadata, load_raw_action_data, validate_data_files,
    save_organized_training_data
)

__version__ = "1.0.0"
__all__ = [
    # Feature mapping
    "load_feature_mappings",
    "validate_feature_mappings",
    
    # Feature extraction
    "extract_features_from_gamestate",
    "FeatureExtractor",
    
    # Action processing
    "extract_action_sequences",
    "flatten_action_window",
    "convert_actions_to_training_format",
    "extract_raw_action_data",
    "convert_raw_actions_to_tensors",
    "analyze_action_distribution",
    "convert_live_features_to_sequence_format",
    "convert_live_actions_to_raw_format", 
    "create_live_training_sequences",
    
    # Sequence building
    "create_temporal_sequences",
    "pad_sequence_window",
    "create_rolling_windows",
    "validate_sequence_structure",
    "create_sequence_metadata",
    "trim_sequences",
    "create_screenshot_paths",
    
    # Normalization
    "normalize_features",
    "normalize_action_data",
    "normalize_input_sequences",
    
    # Encodings
    "ActionEncoder",
    "derive_encodings_from_data",
    "derive_encodings_from_raw_actions",
    "validate_action_encodings",
    
    # I/O
    "load_gamestates",
    "load_actions",
    "save_training_data",
    "load_existing_features",
    "load_action_targets",
    "save_final_training_data",
    "extract_features_from_gamestates",
    "load_gamestates_metadata",
    "load_raw_action_data",
    "validate_data_files",
    "save_organized_training_data",
]
