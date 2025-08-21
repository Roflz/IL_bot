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
from .actions import extract_action_sequences, convert_raw_actions_to_tensors
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
    Load all gamestate files from directory.
    
    Args:
        gamestates_dir: Directory containing gamestate JSON files
        
    Returns:
        List of gamestate dictionaries
        
    Raises:
        FileNotFoundError: If gamestates directory doesn't exist
        ValueError: If no gamestate files found
    """
    gamestates_path = Path(gamestates_dir)
    if not gamestates_path.exists():
        raise FileNotFoundError(f"Gamestates directory not found: {gamestates_path}")
    
    gamestate_files = list(gamestates_path.glob("*.json"))
    
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
        Tuple of (features_array, feature_mappings)
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
    
    for gamestate in tqdm(gamestates, desc="Extracting features"):
        try:
            features = extractor.extract_features_from_gamestate(gamestate)
            mappings = extractor.get_feature_mappings()
            
            all_features.append(features)
            all_feature_mappings.append(mappings)
            
        except Exception as e:
            print(f"Error extracting features from gamestate: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features were successfully extracted")
    
    # Convert to numpy array
    features_array = np.array(all_features, dtype=np.float64)
    
    print(f"Successfully extracted features: {features_array.shape}")
    print(f"Feature mappings: {len(all_feature_mappings)} gamestates")
    
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


def save_organized_training_data(raw_dir: str, trimmed_dir: str, normalized_dir: str, sequences_dir: str, 
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
        sequences_dir: Directory for sequence data
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
    
    # Create directories
    for directory in [raw_dir, trimmed_dir, normalized_dir, sequences_dir, mappings_dir, final_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 1. RAW DATA (data/raw_data/)
    print("  📂 Raw data directory...")
    np.save(Path(raw_dir) / "state_features.npy", features)
    print(f"    ✓ state_features.npy: {features.shape}")
    
    with open(Path(raw_dir) / "raw_action_data.json", 'w') as f:
        json.dump(raw_action_data, f, indent=2)
    print(f"    ✓ raw_action_data.json: {len(raw_action_data)} gamestates")

    # Also emit raw action tensors for the browser (optional convenience)
    try:
        encoder = ActionEncoder()
        # Prefer deriving from targets if available; otherwise from raw actions
        targets_path = Path(sequences_dir) / "target_sequences.json"
        if targets_path.exists():
            encoder.derive_encodings_from_data(str(targets_path))
        else:
            encoder.derive_encodings_from_raw_actions(raw_action_data)
        raw_action_tensors = convert_raw_actions_to_tensors(raw_action_data, encoder)
        with open(Path(raw_dir) / "raw_action_tensors.json", 'w') as f:
            json.dump(raw_action_tensors, f, indent=2)
        print(f"    ✓ raw_action_tensors.json: {len(raw_action_tensors)} gamestates")
    except Exception as e:
        print(f"    ⚠ Could not generate raw_action_tensors.json: {e}")
    
    # 2. TRIMMED DATA (data/trimmed_data/)
    print("  📂 Trimmed data directory...")
    np.save(Path(trimmed_dir) / "trimmed_features.npy", trimmed_features)
    print(f"    ✓ trimmed_features.npy: {trimmed_features.shape}")
    
    # Save trimmed raw action data
    with open(Path(trimmed_dir) / "trimmed_raw_action_data.json", 'w') as f:
        json.dump(trimmed_raw_action_data, f, indent=2)
    print(f"    ✓ trimmed_raw_action_data.json: {len(trimmed_raw_action_data)} gamestates")

    # 3. NORMALIZED DATA (data/normalized_data/)
    print("  📂 Normalized data directory...")
    np.save(Path(normalized_dir) / "normalized_features.npy", normalized_features)
    print(f"    ✓ normalized_features.npy: {normalized_features.shape}")
    
    with open(Path(normalized_dir) / "normalized_action_data.json", 'w') as f:
        json.dump(normalized_action_data, f, indent=2)
    print(f"    ✓ normalized_action_data.json: {len(normalized_action_data)} gamestates")

    # Also emit normalized action tensors in training format for the browser
    try:
        encoder = ActionEncoder()
        # Prefer deriving from targets if available; otherwise from raw actions
        targets_path = Path(sequences_dir) / "target_sequences.json"
        if targets_path.exists():
            encoder.derive_encodings_from_data(str(targets_path))
        else:
            encoder.derive_encodings_from_raw_actions(raw_action_data)
        
        # IMPORTANT: normalized_action_data already has timestamps converted to relative + 0-1000 scale
        # We need to convert this back to the training tensor format WITHOUT further timestamp processing
        normalized_action_tensors = []
        for gamestate_idx, gamestate_actions in enumerate(normalized_action_data):
            # Count total actions for this gamestate
            total_actions = (len(gamestate_actions.get('mouse_movements', [])) + 
                            len(gamestate_actions.get('clicks', [])) + 
                            len(gamestate_actions.get('key_presses', [])) + 
                            len(gamestate_actions.get('key_releases', [])) + 
                            len(gamestate_actions.get('scrolls', [])))
            
            # Start building the action tensor: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
            action_tensor = [total_actions]
            
            # Collect all actions with their metadata (timestamps are already normalized to 0-1000)
            all_actions = []
            
            # Process mouse movements
            for move in gamestate_actions.get('mouse_movements', []):
                all_actions.append({
                    'timestamp': move.get('timestamp', 0),  # Already normalized to 0-10000
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
                all_actions.append({
                    'timestamp': click.get('timestamp', 0),  # Already normalized to 0-10000
                    'type': 1,  # 1 = click
                    'x': click.get('x', 0),
                    'y': click.get('y', 0),
                    'button': encoder.encode_button(click.get('button', '')),
                    'key': 0,  # No key for clicks
                    'scroll_dx': 0,  # No scroll for clicks
                    'scroll_dy': 0
                })
            
            # Process key presses
            for key_press in gamestate_actions.get('key_presses', []):
                all_actions.append({
                    'timestamp': key_press.get('timestamp', 0),  # Already normalized to 0-10000
                    'type': 2,  # 2 = key_press
                    'x': 0,  # No position for key presses
                    'y': 0,
                    'button': 0,  # No button for key presses
                    'key': encoder.encode_key(key_press.get('key', '')),
                    'scroll_dx': 0,  # No scroll for key presses
                    'scroll_dy': 0
                })
            
            # Process key releases
            for key_release in gamestate_actions.get('key_releases', []):
                all_actions.append({
                    'timestamp': key_release.get('timestamp', 0),  # Already normalized to 0-10000
                    'type': 3,  # 3 = key_release
                    'x': 0,  # No position for key releases
                    'y': 0,
                    'button': 0,  # No button for key releases
                    'key': encoder.encode_key(key_release.get('key', '')),
                    'scroll_dx': 0,  # No scroll for key releases
                    'scroll_dy': 0
                })
            
            # Process scrolls
            for scroll in gamestate_actions.get('scrolls', []):
                all_actions.append({
                    'timestamp': scroll.get('timestamp', 0),  # Already normalized to 0-10000
                    'type': 4,  # 4 = scroll
                    'x': 0,  # No position for scrolls
                    'y': 0,
                    'button': 0,  # No button for scrolls
                    'key': 0,  # No key for scrolls
                    'scroll_dx': scroll.get('dx', 0),
                    'scroll_dy': scroll.get('dy', 0)
                })
            
            # Sort actions by timestamp (already normalized to 0-1000)
            all_actions.sort(key=lambda a: a['timestamp'])
            
            # Flatten into tensor format
            for action in all_actions:
                action_tensor.extend([
                    action['timestamp'],  # Already normalized to 0-1000
                    action['type'],
                    action['x'],
                    action['y'],
                    action['button'],
                    action['key'],
                    action['scroll_dx'],
                    action['scroll_dy']
                ])
            
            normalized_action_tensors.append(action_tensor)
        
        with open(Path(normalized_dir) / "normalized_action_training_format.json", 'w') as f:
            json.dump(normalized_action_tensors, f, indent=2)
        print(f"    ✓ normalized_action_training_format.json: {len(normalized_action_tensors)} gamestates")
    except Exception as e:
        print(f"    ⚠ Could not generate normalized_action_training_format.json: {e}")
    
    # 4. SEQUENCES (data/sequences/)
    print("  📂 Sequences directory...")
    np.save(Path(sequences_dir) / "input_sequences.npy", input_sequences)
    print(f"    ✓ input_sequences.npy: {input_sequences.shape}")

    # Save normalized input sequences (same as input_sequences since they were created from normalized features)
    if normalized_input_sequences is not None:
        np.save(Path(sequences_dir) / "normalized_input_sequences.npy", normalized_input_sequences)
        print(f"    ✓ normalized_input_sequences.npy: {normalized_input_sequences.shape}")
    else:
        # Use input_sequences as they are already normalized
        np.save(Path(sequences_dir) / "normalized_input_sequences.npy", input_sequences)
        print(f"    ✓ normalized_input_sequences.npy: {input_sequences.shape} (from input_sequences)")

    with open(Path(sequences_dir) / "target_sequences.json", 'w') as f:
        json.dump(target_sequences, f, indent=2)
    print(f"    ✓ target_sequences.json: {len(target_sequences)} sequences")

    with open(Path(sequences_dir) / "action_input_sequences.json", 'w') as f:
        json.dump(action_input_sequences, f, indent=2)
    print(f"    ✓ action_input_sequences.json: {len(action_input_sequences)} sequences")
    
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
        with open(Path(final_dir) / "action_input_sequences.json", 'w') as f:
            json.dump(normalized_action_input_sequences, f, indent=2)
        print(f"    ✓ action_input_sequences.json (normalized): {len(normalized_action_input_sequences)} sequences")
    else:
        # Fallback to raw if normalized not available
        with open(Path(final_dir) / "action_input_sequences.json", 'w') as f:
            json.dump(action_input_sequences, f, indent=2)
        print(f"    ✓ action_input_sequences.json (raw): {len(action_input_sequences)} sequences")

    if normalized_target_sequences is not None:
        with open(Path(final_dir) / "action_targets.json", 'w') as f:
            json.dump(normalized_target_sequences, f, indent=2)
        print(f"    ✓ action_targets.json (normalized): {len(normalized_target_sequences)} targets")
    else:
        # Fallback to raw if normalized not available
        with open(Path(final_dir) / "action_targets.json", 'w') as f:
            json.dump(target_sequences, f, indent=2)
        print(f"    ✓ action_targets.json (raw): {len(target_sequences)} targets")
    
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
    print(f"  Sequences: {sequences_dir}")
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
    
    # Save final training data (clean, no debug files)
    np.save(final_path / "gamestate_sequences.npy", normalized_input_sequences)
    print(f"✓ Saved gamestate sequences: {normalized_input_sequences.shape}")
    
    with open(final_path / "action_input_sequences.json", 'w') as f:
        json.dump(action_input_sequences, f, indent=2)
    print(f"✓ Saved action input sequences: {len(action_input_sequences)} sequences")
    
    with open(final_path / "action_targets.json", 'w') as f:
        json.dump(target_sequences, f, indent=2)
    print(f"✓ Saved action targets: {len(target_sequences)} targets")
    
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
                'file': 'gamestate_sequences.npy',
                'shape': normalized_input_sequences.shape,
                'description': '10 timesteps of normalized gamestate features (t-9 to t-0)'
            },
            'action_input_sequences': {
                'file': 'action_input_sequences.json',
                'count': len(normalized_action_input_sequences) if normalized_action_input_sequences else len(action_input_sequences),
                'description': '10 timesteps of normalized action history (t-9 to t-0)'
            }
        },
        'targets': {
            'action_targets': {
                'file': 'action_targets.json',
                'count': len(normalized_target_sequences) if normalized_target_sequences else len(target_sequences),
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
            'action_features_per_action': 8,
            'action_features': ['timestamp', 'type', 'x', 'y', 'button', 'key', 'scroll_dx', 'scroll_dy']
        },
        'normalization': {
            'gamestate_features': 'Coordinate system grouping (preserves spatial relationships)',
            'action_features': 'Timestamps scaled to 0-10000 range, coordinates preserved',
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
