# Shared Pipeline Package
# This package provides core functionality for OSRS bot imitation learning

from .io_offline import (
    load_gamestates,
    load_gamestates_sorted,
    load_actions,
    load_existing_features,
    load_feature_mappings,
    load_gamestates_metadata,
    load_raw_action_data,
    load_action_targets,
    build_gamestates_metadata,
    extract_features_from_gamestates,
    save_training_data,
    save_final_training_data,
    save_organized_training_data,
    validate_data_files
)

from .features import FeatureExtractor
from .actions import extract_action_sequences, convert_raw_actions_to_tensors, extract_raw_action_data
from .encodings import ActionEncoder
from .normalize import normalize_features, normalize_input_sequences, normalize_action_data

# Import additional functions that might be needed
try:
    from .sequences import trim_sequences, create_temporal_sequences, create_screenshot_paths
except ImportError:
    # These might not exist yet, create placeholder functions
    def trim_sequences(*args, **kwargs):
        raise NotImplementedError("trim_sequences not implemented yet")
    
    def create_temporal_sequences(*args, **kwargs):
        raise NotImplementedError("create_temporal_sequences not implemented yet")
    
    def create_screenshot_paths(*args, **kwargs):
        raise NotImplementedError("create_screenshot_paths not implemented yet")

# Placeholder functions for missing functionality
def derive_encodings_from_data(*args, **kwargs):
    raise NotImplementedError("derive_encodings_from_data not implemented yet")

def derive_encodings_from_raw_actions(*args, **kwargs):
    raise NotImplementedError("derive_encodings_from_raw_actions not implemented yet")

__all__ = [
    'load_gamestates',
    'load_gamestates_sorted', 
    'load_actions',
    'load_existing_features',
    'load_feature_mappings',
    'load_gamestates_metadata',
    'load_raw_action_data',
    'load_action_targets',
    'build_gamestates_metadata',
    'extract_features_from_gamestates',
    'extract_raw_action_data',
    'save_training_data',
    'save_final_training_data',
    'save_organized_training_data',
    'validate_data_files',
    'FeatureExtractor',
    'extract_action_sequences',
    'convert_raw_actions_to_tensors',
    'ActionEncoder',
    'normalize_features',
    'normalize_input_sequences',
    'normalize_action_data',
    'trim_sequences',
    'create_temporal_sequences',
    'create_screenshot_paths',
    'derive_encodings_from_data',
    'derive_encodings_from_raw_actions'
]
