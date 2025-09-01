#!/usr/bin/env python3
"""
Debug script to find valid action data
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load action targets
    action_targets = np.load(data_dir / "action_targets.npy")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    
    print("=== SEARCHING FOR VALID ACTION DATA ===")
    
    # Check all sequences for non-zero values
    print("Action targets - searching all sequences:")
    total_non_zero = 0
    for seq_idx in range(action_targets.shape[0]):
        seq = action_targets[seq_idx, :, :]
        non_zero_count = np.count_nonzero(seq)
        if non_zero_count > 0:
            print(f"  Sequence {seq_idx}: {non_zero_count} non-zero values")
            total_non_zero += non_zero_count
            # Show first few non-zero values
            for col in range(seq.shape[1]):
                non_zero = seq[:, col][seq[:, col] != 0]
                if len(non_zero) > 0:
                    print(f"    Column {col}: {non_zero[:3]}")
    
    print(f"Total non-zero values in action targets: {total_non_zero}")
    
    print("\nAction input sequences - searching all sequences:")
    total_non_zero_input = 0
    for seq_idx in range(action_input_sequences.shape[0]):
        for timestep in range(action_input_sequences.shape[1]):
            seq = action_input_sequences[seq_idx, timestep, :, :]
            non_zero_count = np.count_nonzero(seq)
            if non_zero_count > 0:
                print(f"  Sequence {seq_idx}, timestep {timestep}: {non_zero_count} non-zero values")
                total_non_zero_input += non_zero_count
                # Show first few non-zero values
                for col in range(seq.shape[1]):
                    non_zero = seq[:, col][seq[:, col] != 0]
                    if len(non_zero) > 0:
                        print(f"    Column {col}: {non_zero[:3]}")
                break  # Only show first timestep with data per sequence
    
    print(f"Total non-zero values in action input sequences: {total_non_zero_input}")
    
    # Check if the issue is with the data generation
    print("\n=== CHECKING DATA GENERATION ===")
    
    # Check if there are any sequences with valid data at all
    if total_non_zero == 0 and total_non_zero_input == 0:
        print("❌ No valid action data found in any sequence!")
        print("This suggests an issue with the data generation process.")
        
        # Check the original data files
        print("\nChecking if original data files exist:")
        original_files = [
            "action_targets_non_delta.npy",
            "gamestate_sequences.npy"
        ]
        
        for file in original_files:
            file_path = data_dir / file
            if file_path.exists():
                data = np.load(file_path)
                non_zero = np.count_nonzero(data)
                print(f"  {file}: shape={data.shape}, non-zero values={non_zero}")
            else:
                print(f"  {file}: NOT FOUND")

if __name__ == "__main__":
    main()
