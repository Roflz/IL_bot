"""
Offline I/O Module for OSRS Bot Imitation Learning

This module handles reading recorded gamestates and actions from files,
preserving the exact behavior from legacy extract_features.py and phase1_data_preparation.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from .features import FeatureExtractor

from .encodings import ActionEncoder


def _add_only_merge_id_maps(path: Path, new_maps: dict) -> None:
    """
    Read existing ground-truth id_mappings.json (if any), add ONLY new keys from new_maps,
    and write back. Existing keys/labels are never overwritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}

    merged = existing if isinstance(existing, dict) else {}

    for group, gmap in (new_maps or {}).items():
        if not isinstance(gmap, dict):
            continue
        dst_g = merged.setdefault(group, {})
        for mtype, table in gmap.items():
            if not isinstance(table, dict):
                continue
            dst_t = dst_g.setdefault(mtype, {})
            for k, v in table.items():
                k = str(k)
                if k not in dst_t:
                    dst_t[k] = v

    path.write_text(json.dumps(merged, indent=2))


def load_gamestates(gamestates_dir: str = "data/gamestates") -> List[Dict]:
    """
    Load gamestate JSONs from a directory and return a list of dicts.
    """
    gamestates_path = Path(gamestates_dir)
    if not gamestates_path.exists():
        raise FileNotFoundError(f"Gamestates directory not found: {gamestates_path}")
    
    # Deterministic order: sort by filename stem (epoch ms)
    gamestate_files = sorted(
        gamestates_path.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )
    
    if not gamestate_files:
        raise ValueError(f"No gamestate files found in {gamestates_path}")
    
    print(f"Loading {len(gamestate_files)} gamestate files...")
    
    gamestates = []
    for gamestate_file in tqdm(gamestate_files, desc="Loading gamestates"):
        try:
            with open(gamestate_file, 'r') as f:
                gamestate = json.load(f)
            gamestates.append(gamestate)
        except Exception as e:
            print(f"Error loading {gamestate_file}: {e}")
            continue
    
    if not gamestates:
        raise ValueError("No gamestates were successfully loaded")
    
    print(f"Successfully loaded {len(gamestates)} gamestates")
    return gamestates


def load_gamestates_sorted(gamestates_dir: str = "data/gamestates") -> List[Dict]:
    """
    Load all gamestate files from directory with deterministic sorting.
    This is an alias for load_gamestates to maintain compatibility.
    
    Args:
        gamestates_dir: Directory containing gamestate JSON files
        
    Returns:
        List of gamestate dictionaries sorted by filename stem
    """
    return load_gamestates(gamestates_dir)


def build_gamestates_metadata(gamestates_dir: str) -> List[Dict]:
    """
    Build per-session metadata directly from gamestate JSONs.
    Each entry: {'raw_index', 'absolute_timestamp', 'filename'}
    """
    path = Path(gamestates_dir)
    files = sorted(
        path.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )
    meta = []
    for i, fp in enumerate(files):
        try:
            obj = json.loads(fp.read_text())
            ts = int(obj.get("timestamp", 0))
        except Exception:
            ts = 0
        meta.append({
            "raw_index": i,
            "absolute_timestamp": ts,
            "filename": fp.name
        })
    return meta


def load_actions(actions_file: str = "data/actions.csv") -> pd.DataFrame:
    """
    Load actions data from CSV file.
    
    Args:
        actions_file: Path to actions.csv file
        
    Returns:
        Pandas DataFrame containing actions data
        
    Raises:
        FileNotFoundError: If actions file doesn't exist
    """
    actions_path = Path(actions_file)
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    
    print(f"Loading actions from {actions_path}")
    
    actions_df = pd.read_csv(actions_path)
    print(f"Loaded {len(actions_df)} action records")
    
    # Convert timestamp to numeric for easier processing
    actions_df['timestamp'] = pd.to_numeric(actions_df['timestamp'], errors='coerce')
    actions_df = actions_df.dropna(subset=['timestamp'])
    
    print(f"Valid timestamps: {len(actions_df)} records")
    return actions_df


def extract_features_from_gamestates(gamestates: List[Dict], 
                                   session_start_time: Optional[int] = None,
                                   save_mappings: bool = True,
                                   mappings_file: str = "data/features/feature_mappings.json") -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Extract features from a list of gamestates.
    
    Args:
        gamestates: List of gamestate dictionaries
        session_start_time: Optional session start time for relative timestamps
        save_mappings: Whether to automatically save feature mappings to file
        mappings_file: Path to save feature mappings (if save_mappings is True)
        
    Returns:
        Tuple of (features_array, feature_mappings, id_mappings)
    """
    print("Extracting features from gamestates...")
    
    extractor = FeatureExtractor()
    
    # Initialize session timing
    extractor.initialize_session_timing(gamestates)
    
    if session_start_time is not None:
        extractor.session_start_time = session_start_time
        extractor.session_start_time_initialized = True
    
    # Extract features from each gamestate
    all_features = []
    all_feature_mappings = []
    all_timestamps = []
    
    for gamestate in tqdm(gamestates, desc="Extracting features"):
        try:
            features = extractor.extract_features_from_gamestate(gamestate)
            mappings = extractor.get_feature_mappings()
            
            # Extract timestamp from gamestate
            timestamp = gamestate.get('timestamp', 0)
            
            all_features.append(features)
            all_feature_mappings.append(mappings)
            all_timestamps.append(timestamp)
            
        except Exception as e:
            print(f"Error extracting features from gamestate: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features were successfully extracted")
    
    # Convert to numpy array
    features_array = np.array(all_features, dtype=np.float64)
    
    print(f"Successfully extracted features: {features_array.shape}")
    print(f"Feature mappings: {len(all_feature_mappings)} gamestates")
    print(f"Timestamps: {len(all_timestamps)} gamestates")
    
    # Automatically save feature mappings if requested
    if save_mappings and all_feature_mappings:
        # Use the first set of mappings (they should all be identical)
        extractor.feature_mappings = all_feature_mappings[0]
        extractor.save_feature_mappings(mappings_file)

    # Collect id mappings from the extractor
    id_mappings = extractor.get_id_mappings()

    # Return features, a single feature mapping list, and the id mappings
    single_feature_mappings = all_feature_mappings[0] if all_feature_mappings else []
    return features_array, single_feature_mappings, id_mappings


def load_existing_features(features_file: str = "data/features/state_features.npy") -> np.ndarray:
    """
    Load existing extracted features from numpy file.
    
    Args:
        features_file: Path to state_features.npy file
        
    Returns:
        Features array of shape (n_gamestates, n_features)
        
    Raises:
        FileNotFoundError: If features file doesn't exist
    """
    features_path = Path(features_file)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    print(f"Loading existing features from {features_path}")
    
    features = np.load(features_file)
    print(f"Loaded features: {features.shape}")
    
    return features


def load_feature_mappings(mappings_file: str = "data/features/feature_mappings.json") -> List[Dict]:
    """
    Load feature mappings from JSON file.
    
    Args:
        mappings_file: Path to feature_mappings.json file
        
    Returns:
        List of feature mapping dictionaries
        
    Raises:
        FileNotFoundError: If mappings file doesn't exist
    """
    mappings_path = Path(mappings_file)
    if not mappings_path.exists():
        raise FileNotFoundError(f"Feature mappings file not found: {mappings_path}")
    
    print(f"Loading feature mappings from {mappings_path}")
    
    with open(mappings_path, 'r') as f:
        feature_mappings = json.load(f)
    
    print(f"Loaded feature mappings for {len(feature_mappings)} features")
    return feature_mappings


def load_gamestates_metadata(metadata_file: str = "data/features/gamestates_metadata.json") -> List[Dict]:
    """
    Load gamestates metadata from JSON file.
    
    Args:
        metadata_file: Path to gamestates_metadata.json file
        
    Returns:
        List of gamestate metadata dictionaries
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Gamestates metadata file not found: {metadata_path}")
    
    print(f"Loading gamestates metadata from {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        gamestates_metadata = json.load(f)
    
    print(f"Loaded metadata for {len(gamestates_metadata)} gamestates")
    return gamestates_metadata


def load_raw_action_data(raw_action_file: str = "data/features/raw_action_data.json") -> List[Dict]:
    """
    Load raw action data from JSON file.
    
    Args:
        raw_action_file: Path to raw_action_data.json file
        
    Returns:
        List of raw action data dictionaries
        
    Raises:
        FileNotFoundError: If raw action data file doesn't exist
    """
    raw_action_path = Path(raw_action_file)
    if not raw_action_path.exists():
        raise FileNotFoundError(f"Raw action data file not found: {raw_action_path}")
    
    print(f"Loading raw action data from {raw_action_path}")
    
    with open(raw_action_path, 'r') as f:
        raw_action_data = json.load(f)
    
    print(f"Loaded raw action data for {len(raw_action_data)} gamestates")
    return raw_action_data


def load_action_targets(action_targets_file: str = "data/training_data/action_targets.json") -> List[List[float]]:
    """
    Load action targets from JSON file.
    
    Args:
        action_targets_file: Path to action_targets.json file
        
    Returns:
        List of action target sequences in training format
        
    Raises:
        FileNotFoundError: If action targets file doesn't exist
    """
    targets_path = Path(action_targets_file)
    if not targets_path.exists():
        raise FileNotFoundError(f"Action targets file not found: {targets_path}")
    
    print(f"Loading action targets from {targets_path}")
    
    with open(targets_path, 'r') as f:
        action_targets = json.load(f)
    
    print(f"Loaded {len(action_targets)} action target sequences")
    return action_targets


def save_organized_training_data(raw_dir: str, trimmed_dir: str, normalized_dir: str, 
                                mappings_dir: str, final_dir: str, features: np.ndarray, 
                                trimmed_features: np.ndarray, normalized_features: np.ndarray,
                                input_sequences: np.ndarray, normalized_input_sequences: np.ndarray,
                                target_sequences: List[List[float]], action_input_sequences: List[List[List[float]]],
                                raw_action_data: List[Dict], trimmed_raw_action_data: List[Dict], 
                                normalized_action_data: List[Dict], feature_mappings: List[Dict],
                                id_mappings: Dict, gamestates_metadata: List[Dict], 
                                screenshot_paths: List[str], normalized_target_sequences: List[List[float]] = None,
                                normalized_action_input_sequences: List[List[List[float]]] = None) -> None:
    """
    Save training data in an organized folder structure.
    
    Args:
        raw_dir: Directory for raw extracted data
        trimmed_dir: Directory for trimmed data
        normalized_dir: Directory for normalized data
        # sequences_dir: Directory for sequence data (SKIPPED - only final training data needed)
        mappings_dir: Directory for mappings and metadata
        final_dir: Directory for final clean training data
        features: Raw extracted features
        trimmed_features: Features after trimming
        normalized_features: Normalized features
        input_sequences: Raw input sequences
        normalized_input_sequences: Normalized input sequences
        target_sequences: Target action sequences
        action_input_sequences: Action input sequences
        raw_action_data: Raw action data
        feature_mappings: Feature mappings
        id_mappings: ID mappings
        gamestates_metadata: Gamestates metadata
        screenshot_paths: Screenshot file paths
    """
    print("\n📁 Saving organized training data...")
    
    # Create directories (skip sequences_dir - only final training data needed)
    for directory in [raw_dir, trimmed_dir, normalized_dir, mappings_dir, final_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 1. RAW DATA (data/raw_data/)
    print("  📂 Raw data directory...")
    np.save(Path(raw_dir) / "state_features.npy", features)
    print(f"    ✓ state_features.npy: {features.shape}")
    
    with open(Path(raw_dir) / "raw_action_data.json", 'w') as f:
        json.dump(raw_action_data, f, indent=2)
    print(f"    ✓ raw_action_data.json: {len(raw_action_data)} gamestates")

    # Also emit raw action tensors for the browser (no fallbacks; no count)
    def _flatten_actions_no_count(actions_obj):
        # Returns flat [t, type, x, y, button, key, dx, dy] * N without a leading count
        all_rows = []
        for mv in actions_obj.get("mouse_movements", []):
            all_rows += [mv.get("timestamp", 0), 0, mv.get("x", 0), mv.get("y", 0), 0, 0, 0, 0]
        for c in actions_obj.get("clicks", []):
            # Convert button to numeric code if it's a string
            button_value = c.get("button", 0)
            if isinstance(button_value, str):
                button_map = {'left': 1, 'right': 2, 'middle': 3, 'none': 0}
                button_value = button_map.get(button_value.lower(), 0)
            all_rows += [c.get("timestamp", 0), 1, c.get("x", 0), c.get("y", 0), button_value, 0, 0, 0]
        for kp in actions_obj.get("key_presses", []):
            # Convert key to numeric code if it's a string
            key_value = kp.get("key", 0)
            if isinstance(key_value, str):
                key_map = {'w': 87, 'a': 65, 's': 83, 'd': 68, 'space': 32, 'enter': 13, 'escape': 27}
                key_value = key_map.get(key_value.lower(), 0)
            all_rows += [kp.get("timestamp", 0), 2, kp.get("x", 0), kp.get("y", 0), 0, key_value, 0, 0]
        for kr in actions_obj.get("key_releases", []):
            # Convert key to numeric code if it's a string
            key_value = kr.get("key", 0)
            if isinstance(key_value, str):
                key_map = {'w': 65, 's': 83, 'd': 68, 'space': 32, 'enter': 13, 'escape': 27}
                key_value = key_map.get(key_value.lower(), 0)
            all_rows += [kr.get("timestamp", 0), 3, kr.get("x", 0), kr.get("y", 0), 0, key_value, 0, 0]
        for sc in actions_obj.get("scrolls", []):
            all_rows += [sc.get("timestamp", 0), 4, sc.get("x", 0), sc.get("y", 0), 0, 0, sc.get("dx", 0), sc.get("dy", 0)]
        return all_rows

    raw_action_tensors = [_flatten_actions_no_count(gs) for gs in raw_action_data]
    with open(Path(raw_dir) / "raw_action_tensors.json", 'w') as f:
        json.dump(raw_action_tensors, f, indent=2)
    print(f"    ✓ raw_action_tensors.json: {len(raw_action_tensors)} gamestates")
    
    # 2. TRIMMED DATA (data/trimmed_data/)
    print("  📂 Trimmed data directory...")
    np.save(Path(trimmed_dir) / "trimmed_features.npy", trimmed_features)
    print(f"    ✓ trimmed_features.npy: {trimmed_features.shape}")
    
    # Save trimmed raw action data
    with open(Path(trimmed_dir) / "trimmed_raw_action_data.json", 'w') as f:
        json.dump(trimmed_raw_action_data, f, indent=2)
    print(f"    ✓ trimmed_raw_action_data.json: {len(trimmed_raw_action_data)} gamestates")

    # And trimmed tensors (no count) for symmetry with raw
    trimmed_action_tensors = [_flatten_actions_no_count(gs) for gs in trimmed_raw_action_data]
    with open(Path(trimmed_dir) / "trimmed_raw_action_tensors.json", 'w') as f:
        json.dump(trimmed_action_tensors, f, indent=2)
    print(f"    ✓ trimmed_raw_action_tensors.json: {len(trimmed_action_tensors)} gamestates")

    # 3. NORMALIZED DATA (data/normalized_data/)
    print("  📂 Normalized data directory...")
    np.save(Path(normalized_dir) / "normalized_features.npy", normalized_features)
    print(f"    ✓ normalized_features.npy: {normalized_features.shape}")
    
    # Save normalized action data if provided
    if normalized_action_data is not None:
        with open(Path(normalized_dir) / "normalized_action_data.json", 'w') as f:
            json.dump(normalized_action_data, f, indent=2)
        print(f"    ✓ normalized_action_data.json: {len(normalized_action_data)} gamestates")

        # Normalized tensors (no count) with consistent naming
        normalized_action_tensors = [_flatten_actions_no_count(gs) for gs in normalized_action_data]
        with open(Path(normalized_dir) / "normalized_action_tensors.json", 'w') as f:
            json.dump(normalized_action_tensors, f, indent=2)
        print(f"    ✓ normalized_action_tensors.json: {len(normalized_action_tensors)} gamestates")
    else:
        print("    ⚠ normalized_action_data.json: Skipped (not available)")
        print("    ⚠ normalized_action_tensors.json: Skipped (not available)")
    
    # 4. SEQUENCES (data/sequences/) - SKIPPED - only final training data needed
    print("  📂 Sequences directory... SKIPPED (only final training data needed)")
    
    # 5. MAPPINGS (data/mappings/)
    print("  📂 Mappings directory...")
    with open(Path(mappings_dir) / "feature_mappings.json", 'w') as f:
        json.dump(feature_mappings, f, indent=2)
    print(f"    ✓ feature_mappings.json: {len(feature_mappings)} mappings")
    
    _add_only_merge_id_maps(Path(mappings_dir) / "id_mappings.json", id_mappings)
    print("    ✓ id_mappings.json (add-only merged)")
    
    with open(Path(mappings_dir) / "gamestates_metadata.json", 'w') as f:
        json.dump(gamestates_metadata, f, indent=2)
    print(f"    ✓ gamestates_metadata.json: {len(gamestates_metadata)} gamestates")
    
    with open(Path(mappings_dir) / "screenshot_paths.json", 'w') as f:
        json.dump(screenshot_paths, f, indent=2)
    print(f"    ✓ screenshot_paths.json: {len(screenshot_paths)} paths")
    
    # 6. FINAL TRAINING DATA (data/final_training_data/)
    print("  📂 Final training data directory...")
    np.save(Path(final_dir) / "gamestate_sequences.npy", normalized_input_sequences)  # normalized sequences from normalized_features
    print(f"    ✓ gamestate_sequences.npy: {normalized_input_sequences.shape}")

    # Save normalized action sequences and targets for final training
    if normalized_action_input_sequences is not None:
        np.save(Path(final_dir) / "action_input_sequences.npy", normalized_action_input_sequences)
        print(f"    ✓ action_input_sequences.npy (normalized): {normalized_action_input_sequences.shape}")
    else:
        # Fallback to raw if normalized not available
        np.save(Path(final_dir) / "action_input_sequences.npy", action_input_sequences)
        print(f"    ✓ action_input_sequences.npy (raw): {action_input_sequences.shape}")

    # Note: action_targets.npy and action_targets_non_delta.npy are now saved separately
    # to avoid conflicts with the new naming scheme
    
    # Create clean metadata
    final_metadata = {
        'description': 'Final training data for sequence-to-sequence action prediction',
        'data_structure': {
            'n_training_sequences': len(input_sequences),
            'sequence_length': 10,  # 10 timesteps of history
            'gamestate_features': input_sequences.shape[2],
            'prediction_target': 'actions for timestep t+1 (11th timestep)'
        },
        'input_sequences': {
            'gamestate_sequences': {
                'file': 'gamestate_sequences.npy',
                'shape': input_sequences.shape,
                'description': '10 timesteps of gamestate features (t-9 to t-0)'
            },
            'action_input_sequences': {
                'file': 'action_input_sequences.npy',
                'count': len(action_input_sequences),
                'description': '10 timesteps of action history (t-9 to t-0)'
            }
        },
        'targets': {
                    'action_targets_non_delta': {
            'file': 'action_targets_non_delta.npy',
            'count': len(target_sequences),
            'description': 'Raw action sequences with actual timestamps to predict for timestep t+1'
        },
        'action_targets': {
            'file': 'action_targets.npy',
            'count': len(target_sequences),
            'description': 'Delta action sequences (time deltas) to predict for timestep t+1'
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
            'action_features_per_action': 7,
            'action_features': ['time', 'x', 'y', 'button', 'key_action', 'key_id', 'scroll_y']
        },
        'normalization': {
            'gamestate_features': 'Coordinate system grouping (preserves spatial relationships)',
            'action_features': 'Actual timestamps in seconds, coordinates preserved',
            'note': 'All normalization pre-computed, no on-the-fly processing needed'
        },
        'folder_structure': {
            '01_raw_data': 'Raw extracted features and actions (Step 1)',
            '02_trimmed_data': 'Data after trimming session boundaries (Step 2)',
            '03_normalized_data': 'Normalized features and actions (Step 3)',
            '04_sequences': 'Input sequences and targets (Step 4)',
            '05_mappings': 'Feature definitions and metadata (Step 5)',
            '06_final_training_data': 'Clean, ready-to-use training data (Step 6)'
        }
    }
    
    with open(Path(final_dir) / "metadata.json", 'w') as f:
        json.dump(final_metadata, f, indent=2)
    print(f"    ✓ metadata.json")
    
    print(f"\n🎯 ORGANIZED TRAINING DATA SAVED!")
    print("=" * 60)
    print("📁 Folder Structure:")
    print(f"  Raw data: {raw_dir}")
    print(f"  Trimmed data: {trimmed_dir}")
    print(f"  Normalized data: {normalized_dir}")
    print(f"  Sequences: SKIPPED (only final training data needed)")
    print(f"  Mappings: {mappings_dir}")
    print(f"  Final training data: {final_dir}")
    print("=" * 60)


def save_training_data(output_dir: str, input_sequences: np.ndarray, target_sequences: List[List[float]], 
                      action_input_sequences: List[List[List[float]]], screenshot_paths: List[str],
                      normalized_features: np.ndarray, normalized_input_sequences: np.ndarray,
                      normalized_action_data: List[Dict], feature_mappings: List[Dict],
                      id_mappings: Dict, gamestates_metadata: List[Dict]) -> None:
    """
    Save all training data including normalized versions.
    
    Args:
        output_dir: Directory to save training data
        input_sequences: Input gamestate sequences
        target_sequences: Target action sequences
        action_input_sequences: Action input sequences for context
        screenshot_paths: List of screenshot file paths
        normalized_features: Normalized features array
        normalized_input_sequences: Normalized input sequences array
        normalized_action_data: Normalized action data
        feature_mappings: Feature mappings
        id_mappings: ID mappings
        gamestates_metadata: Gamestates metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving training data to {output_path}")
    
    # Save raw numpy arrays
    np.save(output_path / "input_sequences.npy", input_sequences)
    print(f"Saved input_sequences.npy: {input_sequences.shape}")
    
    # Save normalized numpy arrays
    np.save(output_path / "normalized_features.npy", normalized_features)
    print(f"Saved normalized_features.npy: {normalized_features.shape}")
    
    np.save(output_path / "normalized_input_sequences.npy", normalized_input_sequences)
    print(f"Saved normalized_input_sequences.npy: {normalized_input_sequences.shape}")
    
    # Save variable-length targets as JSON
    with open(output_path / "target_sequences.json", 'w') as f:
        json.dump(target_sequences, f, indent=2)
    print(f"Saved target_sequences.json: {len(target_sequences)} sequences")
    
    # Save action input sequences
    with open(output_path / "action_input_sequences.json", 'w') as f:
        json.dump(action_input_sequences, f, indent=2)
    print(f"Saved action_input_sequences.json: {len(action_input_sequences)} sequences")
    
    # Save screenshot paths
    with open(output_path / "screenshot_paths.json", 'w') as f:
        json.dump(screenshot_paths, f, indent=2)
    print(f"Saved screenshot_paths.json: {len(screenshot_paths)} paths")
    
    # Save feature mappings
    with open(output_path / "feature_mappings.json", 'w') as f:
        json.dump(feature_mappings, f, indent=2)
    print(f"Saved feature_mappings.json: {len(feature_mappings)} mappings")
    
    # Save ID mappings
    _add_only_merge_id_maps(output_path / "id_mappings.json", id_mappings)
    print("Saved id_mappings.json (add-only merged)")
    
    # Save gamestates metadata
    with open(output_path / "gamestates_metadata.json", 'w') as f:
        json.dump(gamestates_metadata, f, indent=2)
    print(f"Saved gamestates_metadata.json: {len(gamestates_metadata)} gamestates")
    
    # Save normalized action data
    with open(output_path / "normalized_action_data.json", 'w') as f:
        json.dump(normalized_action_data, f, indent=2)
    print(f"Saved normalized_action_data.json: {len(normalized_action_data)} gamestates")
    
    print(f"Training data saved successfully to {output_path}")


def save_final_training_data(final_dir: str, normalized_input_sequences: np.ndarray,
                           action_input_sequences: List[List[List[float]]],
                           target_sequences: List[List[float]]) -> None:
    """
    Save clean final training data to separate folder.
    
    Args:
        final_dir: Directory to save final training data
        normalized_input_sequences: Normalized input sequences
        action_input_sequences: Action input sequences for context
        target_sequences: Target action sequences
    """
    final_path = Path(final_dir)
    final_path.mkdir(exist_ok=True)
    
    print(f"Saving final training data to {final_path}")
    
    # Save final training data with correct names and formats
    np.save(final_path / "gamestate_input_sequences.npy", normalized_input_sequences)
    print(f"✓ Saved gamestate_input_sequences.npy: {normalized_input_sequences.shape}")
    
    # Save action_input_sequences as numpy array
    np.save(final_path / "action_input_sequences.npy", action_input_sequences)
    print(f"✓ Saved action_input_sequences.npy: {action_input_sequences.shape}")
    
    # Save target_sequences as numpy array
    np.save(final_path / "action_target_tensors.npy", target_sequences)
    print(f"✓ Saved action_target_tensors.npy: {target_sequences.shape}")
    
    # Create clean metadata
    final_metadata = {
        'description': 'Final training data for sequence-to-sequence action prediction',
        'data_structure': {
            'n_training_sequences': len(normalized_input_sequences),
            'sequence_length': 10,  # 10 timesteps of history
            'gamestate_features': normalized_input_sequences.shape[2],
            'prediction_target': 'actions for timestep t+1 (11th timestep)'
        },
        'input_sequences': {
            'gamestate_sequences': {
                'file': 'gamestate_input_sequences.npy',
                'shape': normalized_input_sequences.shape,
                'description': '10 timesteps of normalized gamestate features (t-9 to t-0)'
            },
            'action_input_sequences': {
                'file': 'action_input_sequences.npy',
                'count': len(action_input_sequences),
                'description': '10 timesteps of normalized action history (t-9 to t-0)'
            }
        },
        'targets': {
            'action_targets': {
                'file': 'action_target_tensors.npy',
                'count': len(target_sequences),
                'description': 'Normalized action sequences to predict for timestep t+1'
            }
        },
        'training_pattern': {
            'input': 'Given 10 gamestates + 10 action sequences',
            'output': 'Predict actions for the next timestep',
            'sequence_length': 10,
            'total_sequences': len(normalized_input_sequences)
        },
        'feature_info': {
            'gamestate_features': 128,
            'action_features_per_action': 7,
            'action_features': ['time', 'x', 'y', 'button', 'key_action', 'key_id', 'scroll_y']
        },
        'normalization': {
            'gamestate_features': 'Coordinate system grouping (preserves spatial relationships)',
            'action_features': 'Actual timestamps in seconds, coordinates preserved',
            'note': 'All normalization pre-computed, no on-the-fly processing needed'
        }
    }
    
    with open(final_path / "metadata.json", 'w') as f:
        json.dump(final_metadata, f, indent=2)
    print(f"✓ Saved training metadata")
    
    print(f"\n🎯 FINAL TRAINING DATA SAVED TO: {final_path}")
    print(f"📁 Training data folder: {final_path} (clean training data)")
    print("=" * 50)


def validate_data_files(data_dir: str = "data") -> Dict[str, bool]:
    """
    Validate that all required data files exist.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary mapping file paths to existence status
    """
    data_path = Path(data_dir)
    
    required_files = {
        'gamestates_dir': data_path / "gamestates",
        'actions_file': data_path / "actions.csv",
        'features_file': data_path / "features" / "state_features.npy",
        'feature_mappings': data_path / "05_mappings" / "feature_mappings.json",
        'gamestates_metadata': data_path / "features" / "gamestates_metadata.json",
        'raw_action_data': data_path / "features" / "raw_action_data.json",
        'action_targets': data_path / "training_data" / "action_targets.json"
    }
    
    validation_results = {}
    
    for name, file_path in required_files.items():
        exists = file_path.exists()
        validation_results[name] = exists
        
        if exists:
            print(f"✓ {name}: {file_path}")
        else:
            print(f"✗ {name}: {file_path} (MISSING)")
    
    return validation_results
