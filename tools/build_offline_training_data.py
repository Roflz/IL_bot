#!/usr/bin/env python3
"""
Build Offline Training Data CLI

This script reproduces exactly the artifacts produced by the legacy Phase 1 script,
using the modularized shared pipeline. It preserves byte-for-byte output where feasible.
"""

import sys
import argparse
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_pipeline import (
    load_gamestates, load_actions, load_existing_features, load_feature_mappings,
    load_gamestates_metadata, load_raw_action_data, load_action_targets,
    extract_features_from_gamestates, extract_raw_action_data, convert_raw_actions_to_tensors,
    create_temporal_sequences, trim_sequences, create_screenshot_paths,
    normalize_features, normalize_input_sequences, normalize_action_data,
    save_training_data, save_final_training_data, validate_data_files,
    derive_encodings_from_data, derive_encodings_from_raw_actions,
    save_organized_training_data
)

def _write_slice_info(path, start_idx_raw, end_idx_raw, count, first_ts=None, last_ts=None):
    info = {
        "start_idx_raw": int(start_idx_raw) if start_idx_raw is not None else None,
        "end_idx_raw": int(end_idx_raw) if end_idx_raw is not None else None,
        "count": int(count) if count is not None else 0,
    }
    if first_ts is not None: info["first_timestamp"] = int(first_ts)
    if last_ts  is not None: info["last_timestamp"]  = int(last_ts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(info, f, indent=2)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Build offline training data using shared pipeline modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing extracted features (recommended)
  python tools/build_offline_training_data.py --use-existing-features
  
  # Extract features from gamestates (slower)
  python tools/build_offline_training_data.py --extract-features
  
  # Custom data directory
  python tools/build_offline_training_data.py --data-dir /path/to/data --use-existing-features
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        default='data',
        help='Root data directory (default: data)'
    )
    
    parser.add_argument(
        '--use-existing-features',
        action='store_true',
        help='Use existing extracted features from data/features/state_features.npy'
    )
    
    parser.add_argument(
        '--extract-features',
        action='store_true',
        help='Extract features from gamestates (slower but more flexible)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/training_data',
        help='Output directory for training data (default: data/training_data)'
    )
    
    parser.add_argument(
        '--final-dir',
        default='data/06_final_training_data',
        help='Final training data directory (default: data/06_final_training_data)'
    )
    
    parser.add_argument(
        '--raw-dir',
        default='data/01_raw_data',
        help='Raw extracted data directory (default: data/01_raw_data)'
    )
    
    parser.add_argument(
        '--trimmed-dir',
        default='data/02_trimmed_data',
        help='Trimmed data directory (default: data/02_trimmed_data)'
    )
    
    parser.add_argument(
        '--normalized-dir',
        default='data/03_normalized_data',
        help='Normalized data directory (default: data/03_normalized_data)'
    )
    
    parser.add_argument(
        '--sequences-dir',
        default='data/04_sequences',
        help='Sequence data directory (default: data/04_sequences)'
    )
    
    parser.add_argument(
        '--mappings-dir',
        default='data/05_mappings',
        help='Mappings and metadata directory (default: data/05_mappings)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data files, don\'t process'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILD OFFLINE TRAINING DATA")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Raw data directory: {args.raw_dir}")
    print(f"Trimmed data directory: {args.trimmed_dir}")
    print(f"Normalized data directory: {args.normalized_dir}")
    print(f"Sequences directory: {args.sequences_dir}")
    print(f"Mappings directory: {args.mappings_dir}")
    print(f"Final training data: {args.final_dir}")
    print(f"Use existing features: {args.use_existing_features}")
    print(f"Extract features: {args.extract_features}")
    print("=" * 60)
    
    # Validate data files
    print("Validating data files...")
    validation_results = validate_data_files(args.data_dir)
    
    if not any(validation_results.values()):
        print("❌ No required data files found!")
        print("Please ensure you have the required data structure:")
        print("  data/")
        print("  ├── gamestates/          # JSON gamestate files")
        print("  ├── actions.csv          # Actions CSV file")
        print("  ├── features/            # Extracted features")
        print("  └── training_data/       # Output directory")
        return 1
    
    if args.validate_only:
        print("Validation complete. Exiting.")
        return 0
    
    # Determine processing mode
    if args.use_existing_features and args.extract_features:
        print("❌ Cannot use both --use-existing-features and --extract-features")
        return 1
    
    if not args.use_existing_features and not args.extract_features:
        print("⚠️  No processing mode specified, defaulting to --use-existing-features")
        args.use_existing_features = True
    
    try:
        if args.use_existing_features:
            print("\n🔍 Using existing extracted features...")
            process_with_existing_features(args)
        else:
            print("\n🔍 Extracting features from gamestates...")
            process_with_feature_extraction(args)
        
        print("\n✅ Training data build completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training data build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def process_with_existing_features(args):
    """Process using existing extracted features."""
    # Load existing data
    features = load_existing_features(f"{args.data_dir}/features/state_features.npy")
    feature_mappings = load_feature_mappings(f"{args.data_dir}/features/feature_mappings.json")
    gamestates_metadata = load_gamestates_metadata(f"{args.data_dir}/features/gamestates_metadata.json")
    
    # -------------- RAW SLICE INFO --------------
    raw_N = int(features.shape[0])  # or len(raw_action_data)
    _write_slice_info(os.path.join(args.data_dir, "01_raw_data", "slice_info.json"),
                      0, raw_N - 1, raw_N,
                      first_ts=gamestates_metadata[0]["absolute_timestamp"],
                      last_ts=gamestates_metadata[raw_N-1]["absolute_timestamp"])
    
    # Load or extract raw action data
    raw_action_file = f"{args.data_dir}/features/raw_action_data.json"
    if Path(raw_action_file).exists():
        print("Loading existing raw action data...")
        raw_action_data = load_raw_action_data(raw_action_file)
    else:
        print("Extracting raw action data from actions.csv...")
        gamestates = load_gamestates(f"{args.data_dir}/gamestates")
        raw_action_data = extract_raw_action_data(gamestates, f"{args.data_dir}/actions.csv")
    
    # Apply data trimming
    print("Applying data trimming...")
    trimmed_features, trimmed_action_data, start_idx, end_idx = trim_sequences(
        features, raw_action_data
    )
    
    # Update metadata to match trimmed data
    if gamestates_metadata:
        gamestates_metadata = gamestates_metadata[start_idx:len(features) - end_idx]
    
    # -------------- TRIMMED SLICE INFO --------------
    # Derive start/end by matching first/last trimmed action timestamps into full metadata
    def _find_index_by_ts(ts, meta):
        for i, m in enumerate(meta):
            if m.get("absolute_timestamp") == ts:
                return i
        return None
    trim_first_ts = trimmed_action_data[0]["gamestate_timestamp"]
    trim_last_ts  = trimmed_action_data[-1]["gamestate_timestamp"]
    trim_start = _find_index_by_ts(trim_first_ts, gamestates_metadata)
    trim_end   = _find_index_by_ts(trim_last_ts, gamestates_metadata)
    _write_slice_info(os.path.join(args.data_dir, "02_trimmed_data", "slice_info.json"),
                      trim_start, trim_end, (trim_end - trim_start + 1) if (trim_start is not None and trim_end is not None) else len(trimmed_features),
                      first_ts=trim_first_ts, last_ts=trim_last_ts)
    
    # Normalize features
    print("Normalizing features...")
    normalized_features = normalize_features(trimmed_features, f"{args.data_dir}/05_mappings/feature_mappings.json")
    
    # Create temporal sequences from raw (trimmed) features
    print("Creating temporal sequences from raw features...")
    raw_input_sequences, action_input_sequences, target_sequences = create_temporal_sequences(
        trimmed_features, raw_action_data
    )
    
    # Create temporal sequences from normalized features
    print("Creating temporal sequences from normalized features...")
    normalized_input_sequences, _, _ = create_temporal_sequences(
        normalized_features, raw_action_data
    )
    
    # Use raw sequences as the default input sequences
    input_sequences = raw_input_sequences
    
    # Normalize action data
    print("Normalizing action data...")
    normalized_action_data = normalize_action_data(raw_action_data, normalized_features)
    
    # Create screenshot paths
    print("Creating screenshot paths...")
    screenshot_paths = create_screenshot_paths(gamestates_metadata, f"{args.data_dir}/runelite_screenshots")
    
    # Load id_mappings if available (existing-features mode)
    id_mappings_path = Path(f"{args.data_dir}/features/id_mappings.json")
    if id_mappings_path.exists():
        try:
            import json
            with open(id_mappings_path, 'r') as f:
                id_mappings = json.load(f)
            print("Loaded id_mappings from existing features")
        except Exception as e:
            print(f"⚠ Warning: Failed to load id_mappings: {e}")
            id_mappings = {}
    else:
        id_mappings = {}
    
    # Save training data
    print("Saving training data...")
    save_organized_training_data(
        args.raw_dir,
        args.trimmed_dir,
        args.normalized_dir,
        args.mappings_dir,
        args.final_dir,
        features,
        trimmed_features,
        normalized_features,
        input_sequences,
        normalized_input_sequences,
        target_sequences,
        action_input_sequences,
        raw_action_data,
        raw_action_data,  # trimmed_raw_action_data (using raw for now)
        normalized_action_data,
        feature_mappings,
        id_mappings,
        gamestates_metadata,
        screenshot_paths
    )
    
    # Save final training data
    print("Saving final training data...")
    save_final_training_data(
        args.final_dir,
        normalized_input_sequences,
        action_input_sequences,
        target_sequences
    )


def process_with_feature_extraction(args):
    """Process by extracting features from gamestates."""
    # Load gamestates
    print("Loading gamestates...")
    gamestates = load_gamestates(f"{args.data_dir}/gamestates")
    
    # Extract features
    print("Extracting features from gamestates...")
    features, feature_mappings, id_mappings = extract_features_from_gamestates(gamestates)
    
    # Load actions
    print("Loading actions...")
    actions_df = load_actions(f"{args.data_dir}/actions.csv")
    
    # Extract raw action data
    print("Extracting raw action data...")
    raw_action_data = extract_raw_action_data(gamestates, f"{args.data_dir}/actions.csv")
    
    # Derive encodings from existing training targets (if available)
    action_targets_file = f"{args.data_dir}/training_data/action_targets.json"
    if Path(action_targets_file).exists():
        print("Deriving action encodings from existing training data...")
        action_encoder = derive_encodings_from_data(action_targets_file)
    else:
        print("Creating new action encoder...")
        action_encoder = derive_encodings_from_raw_actions(raw_action_data)
    
    # Convert raw actions to training format
    print("Converting raw actions to training format...")
    action_targets = convert_raw_actions_to_tensors(raw_action_data, action_encoder)
    
    # Apply data trimming
    print("Applying data trimming...")
    trimmed_features, trimmed_action_targets, start_idx, end_idx = trim_sequences(
        features, action_targets
    )
    
    # Also trim raw_action_data to match the trimmed features
    print("Trimming raw action data to match features...")
    trimmed_raw_action_data = raw_action_data[start_idx:len(raw_action_data) - end_idx]
    print(f"  Raw action data trimmed from {len(raw_action_data)} to {len(trimmed_raw_action_data)} gamestates")

    # Create gamestates metadata
    print("Creating gamestates metadata...")
    gamestates_metadata = []
    for i, gamestate in enumerate(gamestates[start_idx:len(gamestates) - end_idx]):
        absolute_timestamp = gamestate.get('timestamp', 0)
        relative_timestamp = absolute_timestamp - gamestates[0].get('timestamp', 0)
        metadata = {
            'index': i,
            'absolute_timestamp': absolute_timestamp,
            'relative_timestamp': relative_timestamp,
            'filename': f"gamestate_{i}.json"
        }
        gamestates_metadata.append(metadata)

    # Normalize features (features are already relative timestamps, just need scaling)
    print("Normalizing features...")
    # Use the global feature mappings that were just created, not session-specific ones
    global_mappings_file = "data/features/feature_mappings.json"
    if not Path(global_mappings_file).exists():
        raise FileNotFoundError(f"Feature mappings file not found: {global_mappings_file}. Feature extraction should have created this file.")
    normalized_features = normalize_features(trimmed_features, global_mappings_file)

    # Create temporal sequences from raw (trimmed) features
    print("Creating temporal sequences from raw features...")
    raw_input_sequences, action_input_sequences, target_sequences = create_temporal_sequences(
        trimmed_features, trimmed_raw_action_data
    )
    
    # Create temporal sequences from normalized features
    print("Creating temporal sequences from normalized features...")
    normalized_input_sequences, _, _ = create_temporal_sequences(
        normalized_features, trimmed_raw_action_data
    )
    
    # Use raw sequences as the default input sequences
    input_sequences = raw_input_sequences
    print(f"Raw sequences created from trimmed features: {raw_input_sequences.shape}")
    print(f"Normalized sequences created from normalized features: {normalized_input_sequences.shape}")

    # Normalize action data (using trimmed raw action data)
    print("Normalizing action data...")
    normalized_action_data = normalize_action_data(trimmed_raw_action_data, normalized_features)
    
    # Create normalized action sequences and targets
    print("Creating normalized action sequences and targets...")
    _, normalized_action_input_sequences, normalized_target_sequences = create_temporal_sequences(
        normalized_features, normalized_action_data
    )
    
    print(f"Normalized action targets created: {len(normalized_target_sequences)} sequences")
    print(f"Normalized action input sequences created: {len(normalized_action_input_sequences)} sequences")
    
    # Sanity checks before saving
    assert raw_input_sequences.ndim == 3 and raw_input_sequences.shape[1] == 10, "inputs must be (B, 10, 128)"
    assert action_input_sequences.ndim == 4 and action_input_sequences.shape[1] == 10 and action_input_sequences.shape[2] == 100 and action_input_sequences.shape[3] == 8, "action_input_sequences must be (B, 10, 100, 8)"
    assert target_sequences.ndim == 2+1 and target_sequences.shape[1] == 100 and target_sequences.shape[2] == 8, "action_targets must be (B, 100, 8)"
    
    # Create screenshot paths
    print("Creating screenshot paths...")
    screenshot_paths = create_screenshot_paths(gamestates_metadata, f"{args.data_dir}/runelite_screenshots")
    
    # Ensure the expected directory structure exists
    print("Creating expected directory structure...")
    Path(args.mappings_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy the global feature mappings to the expected location
    global_mappings_file = "data/features/feature_mappings.json"
    expected_mappings_file = f"{args.mappings_dir}/feature_mappings.json"
    if Path(global_mappings_file).exists():
        import shutil
        shutil.copy2(global_mappings_file, expected_mappings_file)
        print(f"Copied feature mappings to: {expected_mappings_file}")
    else:
        print(f"Warning: Global feature mappings not found at {global_mappings_file}")
    
    # Save training data
    print("Saving training data...")
    save_organized_training_data(
        args.raw_dir,
        args.trimmed_dir,
        args.normalized_dir,
        args.mappings_dir,
        args.final_dir,
        features,
        trimmed_features,
        normalized_features,
        input_sequences,  # raw sequences from trimmed_features
        normalized_input_sequences,  # normalized sequences from normalized_features
        target_sequences,  # raw targets from trimmed_action_targets
        action_input_sequences,  # raw action input sequences
        raw_action_data,
        trimmed_raw_action_data,
        normalized_action_data,
        feature_mappings,
        id_mappings,
        gamestates_metadata,
        screenshot_paths
    )
    
    # -------------- FINAL TRAINING SLICE INFO --------------
    seq_B = int(input_sequences.shape[0])      # Batches
    SEQ_LEN = 10                                   # keep in sync
    final_count = seq_B + SEQ_LEN + 1
    final_start = start_idx if start_idx is not None else 0
    final_end   = (final_start + final_count - 1) if final_start is not None else None
    first_ts = gamestates_metadata[0]["absolute_timestamp"] if gamestates_metadata else None
    last_ts  = gamestates_metadata[-1]["absolute_timestamp"] if gamestates_metadata else None
    _write_slice_info(os.path.join(args.data_dir, "06_final_training_data", "slice_info.json"),
                      final_start, final_end, final_count, first_ts=first_ts, last_ts=last_ts)


if __name__ == "__main__":
    sys.exit(main())
