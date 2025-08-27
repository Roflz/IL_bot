#!/usr/bin/env python3
"""
Process a specific recording session through the shared pipeline to generate training data.
This script is a wrapper around the shared pipeline components.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import pandas as pd # Added for timestamp synchronization

# Add the parent directory to the path so we can import shared_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from shared_pipeline.io_offline import load_gamestates, extract_features_from_gamestates
from shared_pipeline.actions import extract_raw_action_data, convert_raw_actions_to_tensors
from shared_pipeline.encodings import derive_encodings_from_raw_actions
from shared_pipeline.sequences import create_temporal_sequences
import numpy as np

def process_recording_session(gamestates_dir, actions_file, output_dir=None, visualize=False, visualize_mode="all"):
    """
    Process a recording session through the shared pipeline.
    
    Args:
        gamestates_dir: Path to directory containing gamestate JSON files
        actions_file: Path to actions.csv file
        output_dir: Output directory (defaults to gamestates_dir/processed)
        visualize: Whether to visualize the generated data
        visualize_mode: Which data to visualize ("all", "gamestates", "actions", "sequences")
    """
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(gamestates_dir), "processed")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing recording session...")
    print(f"Gamestates directory: {gamestates_dir}")
    print(f"Actions file: {actions_file}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load gamestates and extract features
    print("\n1. Loading gamestates and extracting features...")
    gamestates = load_gamestates(gamestates_dir)
    gamestate_features, feature_mappings, id_mappings, timestamps = extract_features_from_gamestates(gamestates)
    
    # Extract original Unix timestamps from gamestates before they get converted to relative
    original_unix_timestamps = []
    for gamestate in gamestates:
        original_unix_timestamps.append(gamestate.get('timestamp', 0))
    
    print(f"   Loaded {len(gamestates)} gamestates")
    print(f"   Extracted {len(gamestate_features)} gamestate feature vectors")
    print(f"   Feature vector shape: {gamestate_features.shape}")
    print(f"   Extracted {len(timestamps)} timestamps")
    print(f"   Original Unix timestamps range: {min(original_unix_timestamps)} to {max(original_unix_timestamps)}")
    
    # Step 1.5: Synchronize timestamps between gamestates and actions
    print("\n1.5. Synchronizing timestamps between gamestates and actions...")
    
    # Load actions to get their timestamps
    actions_df = pd.read_csv(actions_file)
    actions_df['timestamp'] = pd.to_numeric(actions_df['timestamp'], errors='coerce')
    actions_df = actions_df.dropna(subset=['timestamp'])
    
    # Extract timestamp features from the gamestate features (feature 127)
    # These are already relative timestamps in milliseconds since session start
    gamestate_timestamp_features = gamestate_features[:, 127]  # Feature 127 is the timestamp
    action_timestamps = actions_df['timestamp'].tolist()
    
    if len(gamestate_timestamp_features) > 0 and action_timestamps:
        earliest_gamestate = np.min(gamestate_timestamp_features)
        earliest_action = min(action_timestamps)
        
        print(f"   Earliest gamestate timestamp feature: {earliest_gamestate}")
        print(f"   Earliest action timestamp: {earliest_action}")
        
        # The gamestate timestamp features are already relative (starting at 0)
        # We need to normalize the action timestamps to the same scale
        if earliest_gamestate == 0.0:
            # Gamestates already start at 0, normalize actions to start at 0
            earliest_overall = earliest_action
            normalized_gamestate_timestamps = gamestate_timestamp_features  # Already normalized
        else:
            # Both need normalization
            earliest_overall = min(earliest_gamestate, earliest_action)
            normalized_gamestate_timestamps = gamestate_timestamp_features - earliest_overall
        
        print(f"   Normalizing action timestamps to start at 0...")
        
        # Normalize action timestamps to start at 0 (same scale as gamestates)
        actions_df['timestamp'] = actions_df['timestamp'] - earliest_overall
        
        # Save normalized actions CSV
        normalized_actions_file = Path(output_dir) / "normalized_actions.csv"
        actions_df.to_csv(normalized_actions_file, index=False)
        print(f"   Saved normalized actions CSV: {normalized_actions_file}")
        
        # Update the timestamps variable to use normalized values
        timestamps = normalized_gamestate_timestamps.tolist()
        
        # Also update the gamestate features with normalized timestamp features
        gamestate_features[:, 127] = normalized_gamestate_timestamps
        
        print(f"   Timestamps synchronized: gamestates and actions now start at 0")
        print(f"   Gamestate timestamp feature range: {normalized_gamestate_timestamps.min()} to {normalized_gamestate_timestamps.max()}")
        print(f"   Action range: {actions_df['timestamp'].min()} to {actions_df['timestamp'].max()}")
        
        # Find the actual overlap between gamestates and actions
        action_start_time = actions_df['timestamp'].min()
        action_end_time = actions_df['timestamp'].max()
        
        # Find gamestates that have actions in their 600ms window
        gamestates_with_actions = []
        for i, gamestate_time in enumerate(normalized_gamestate_timestamps):
            window_start = gamestate_time - 600
            window_end = gamestate_time
            # Check if there are any actions in this window
            actions_in_window = actions_df[(actions_df['timestamp'] >= window_start) & (actions_df['timestamp'] < window_end)]
            if len(actions_in_window) > 0:
                gamestates_with_actions.append(i)
        
        print(f"   Found {len(gamestates_with_actions)} gamestates with actions in their 600ms window")
        if gamestates_with_actions:
            print(f"   First gamestate with actions: {gamestates_with_actions[0]} at time {normalized_gamestate_timestamps[gamestates_with_actions[0]]}")
            print(f"   Last gamestate with actions: {gamestates_with_actions[-1]} at time {normalized_gamestate_timestamps[gamestates_with_actions[-1]]}")
    else:
        print(f"   Warning: Could not synchronize timestamps - missing data")
    
    # Step 2: Extract actions from gamestates and actions.csv
    print("\n2. Extracting actions from gamestates and actions.csv...")
    # Use the normalized actions CSV if we created one
    actions_file_to_use = str(normalized_actions_file) if 'normalized_actions_file' in locals() else actions_file
    
    # Create a modified gamestates list with synchronized timestamps for action extraction
    synchronized_gamestates = []
    for i, gamestate in enumerate(gamestates):
        # Create a copy of the gamestate with the synchronized timestamp
        synced_gamestate = gamestate.copy()
        if 'normalized_actions_file' in locals() and i < len(timestamps):
            synced_gamestate['timestamp'] = timestamps[i]
        synchronized_gamestates.append(synced_gamestate)
    
    raw_action_data = extract_raw_action_data(synchronized_gamestates, actions_file_to_use)
    encoder = derive_encodings_from_raw_actions(raw_action_data)
    action_targets = convert_raw_actions_to_tensors(raw_action_data, encoder)
    print(f"   Extracted {len(raw_action_data)} action sequences")
    print(f"   Converted to {len(action_targets)} action tensors")
    
    # Step 3: Create temporal sequences
    print("\n3. Creating temporal sequences...")
    gamestate_sequences, action_input_sequences, action_targets = create_temporal_sequences(
        gamestate_features, action_targets
    )
    
    print(f"   Gamestate sequences shape: {gamestate_sequences.shape}")
    print(f"   Action input sequences shape: {action_input_sequences.shape}")
    print(f"   Action targets shape: {action_targets.shape}")
    
    # Save the generated data
    print("\n4. Saving generated data...")
    
    # Convert output_dir to Path object if it's a string
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
    
    # Save gamestate sequences
    gamestate_sequences_file = output_path / "gamestate_sequences.npy"
    np.save(gamestate_sequences_file, gamestate_sequences)
    print(f"   Saved gamestate sequences: {gamestate_sequences_file}")
    
    # Save action input sequences
    action_input_sequences_file = output_path / "action_input_sequences.npy"
    np.save(action_input_sequences_file, action_input_sequences)
    print(f"   Saved action input sequences: {action_input_sequences_file}")
    
    # Save action targets
    action_targets_file = output_path / "action_targets.npy"
    np.save(action_targets_file, action_targets)
    print(f"   Saved action targets: {action_targets_file}")
    
    # Save timestamps
    timestamps_file = output_path / "timestamps.npy"
    np.save(timestamps_file, np.array(timestamps))
    print(f"   Saved timestamps: {timestamps_file}")
    
    # Save original Unix timestamps for display
    original_unix_timestamps_file = output_path / "original_unix_timestamps.npy"
    np.save(original_unix_timestamps_file, np.array(original_unix_timestamps))
    print(f"   Saved original Unix timestamps: {original_unix_timestamps_file}")
    
    # Save feature mappings and ID mappings
    feature_mappings_file = output_path / "feature_mappings.json"
    with open(feature_mappings_file, 'w') as f:
        json.dump(feature_mappings, f, indent=2)
    print(f"   Saved feature mappings: {feature_mappings_file}")
    
    id_mappings_file = output_path / "id_mappings.json"
    with open(id_mappings_file, 'w') as f:
        json.dump(id_mappings, f, indent=2)
    print(f"   Saved ID mappings: {id_mappings_file}")
    
    print(f"\nProcessing complete! Data saved to: {output_dir}")
    
    # Step 5: Visualize the data if requested
    if visualize:
        print("\n5. Visualizing generated data...")
        visualize_generated_data(output_dir, visualize_mode)
    
    return {
        'gamestate_sequences': gamestate_sequences,
        'action_input_sequences': action_input_sequences,
        'action_targets': action_targets,
        'output_dir': output_dir
    }

def visualize_generated_data(output_dir, visualize_mode="all"):
    """Visualize the generated training data using print_numpy_array.py"""
    print("\n5. Visualizing generated data...")
    
    # Find the print_numpy_array.py script
    script_dir = Path(__file__).parent
    print_numpy_script = script_dir / "print_numpy_array.py"
    
    if not print_numpy_script.exists():
        print(f"   Warning: print_numpy_array.py not found at {print_numpy_script}")
        return
    
    # Determine which files to visualize based on mode
    files_to_visualize = []
    
    if visualize_mode in ["all", "gamestates"]:
        gamestate_file = os.path.join(output_dir, "gamestate_sequences.npy")
        if os.path.exists(gamestate_file):
            files_to_visualize.append(gamestate_file)
    
    if visualize_mode in ["all", "actions"]:
        action_input_file = os.path.join(output_dir, "action_input_sequences.npy")
        if os.path.exists(action_input_file):
            files_to_visualize.append(action_input_file)
    
    if visualize_mode in ["all", "sequences"]:
        action_targets_file = os.path.join(output_dir, "action_targets.npy")
        if os.path.exists(action_targets_file):
            files_to_visualize.append(action_targets_file)
    
    # Always include timestamps file for timestamp display
    timestamps_file = os.path.join(output_dir, "timestamps.npy")
    if os.path.exists(timestamps_file):
        files_to_visualize.append(timestamps_file)
    
    # Always include original Unix timestamps file for display
    original_unix_timestamps_file = os.path.join(output_dir, "original_unix_timestamps.npy")
    if os.path.exists(original_unix_timestamps_file):
        files_to_visualize.append(original_unix_timestamps_file)
    
    if not files_to_visualize:
        print("   No data files found to visualize")
        return
    
    # Call print_numpy_array.py with all files
    print(f"   Opening unified visualization for {len(files_to_visualize)} data files...")
    
    try:
        # Pass all files to the unified visualization script
        cmd = [sys.executable, str(print_numpy_script)] + files_to_visualize
        subprocess.Popen(cmd)
        print(f"   Opened unified visualization with {len(files_to_visualize)} tabs")
        
    except Exception as e:
        print(f"   Error opening visualization: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Process a recording session through the shared pipeline to generate training data"
    )
    
    parser.add_argument(
        "--gamestates-dir",
        required=True,
        help="Path to directory containing gamestate JSON files"
    )
    
    parser.add_argument(
        "--actions-file",
        required=True,
        help="Path to actions.csv file"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to gamestates_dir/processed)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated data after processing"
    )
    
    parser.add_argument(
        "--visualize-mode",
        choices=["all", "gamestates", "actions", "sequences"],
        default="all",
        help="Which data to visualize (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.gamestates_dir):
        print(f"Error: Gamestates directory does not exist: {args.gamestates_dir}")
        sys.exit(1)
    
    if not os.path.isfile(args.actions_file):
        print(f"Error: Actions file does not exist: {args.actions_file}")
        sys.exit(1)
    
    # Process the recording session
    try:
        result = process_recording_session(
            args.gamestates_dir,
            args.actions_file,
            args.output_dir,
            args.visualize,
            args.visualize_mode
        )
        
        print("\nSummary:")
        print(f"  Gamestate sequences: {result['gamestate_sequences'].shape}")
        print(f"  Action input sequences: {result['action_input_sequences'].shape}")
        print(f"  Action targets: {result['action_targets'].shape}")
        print(f"  Output directory: {result['output_dir']}")
        
    except Exception as e:
        print(f"Error processing recording session: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
