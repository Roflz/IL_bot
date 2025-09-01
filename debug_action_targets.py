#!/usr/bin/env python3
"""
Debug script to check action target values
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load action targets
    action_targets = np.load(data_dir / "action_targets.npy")
    
    print("=== ACTION TARGETS ANALYSIS ===")
    print(f"Shape: {action_targets.shape}")
    
    # Check the first few sequences
    for seq_idx in range(min(3, action_targets.shape[0])):
        print(f"\nSequence {seq_idx}:")
        at = action_targets[seq_idx, :, :]  # All actions in this sequence
        
        # Check timestamp column (index 0)
        timestamps = at[:, 0]
        valid_timestamps = timestamps[timestamps > 0]
        
        if len(valid_timestamps) > 0:
            print(f"  Timestamps: min={valid_timestamps.min():.6f}, max={valid_timestamps.max():.6f}, mean={valid_timestamps.mean():.6f}")
            print(f"  First 5 valid timestamps: {valid_timestamps[:5]}")
        else:
            print("  No valid timestamps found")
        
        # Check x, y coordinates (indices 1, 2)
        x_coords = at[:, 1]
        y_coords = at[:, 2]
        valid_x = x_coords[x_coords > 0]
        valid_y = y_coords[y_coords > 0]
        
        if len(valid_x) > 0:
            print(f"  X coords: min={valid_x.min():.2f}, max={valid_x.max():.2f}, mean={valid_x.mean():.2f}")
        if len(valid_y) > 0:
            print(f"  Y coords: min={valid_y.min():.2f}, max={valid_y.max():.2f}, mean={valid_y.mean():.2f}")
    
    # Check action input sequences for comparison
    print("\n=== ACTION INPUT SEQUENCES ANALYSIS ===")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    print(f"Shape: {action_input_sequences.shape}")
    
    for seq_idx in range(min(2, action_input_sequences.shape[0])):
        print(f"\nSequence {seq_idx}:")
        ais = action_input_sequences[seq_idx, 0, :, :]  # First timestep, all actions
        
        # Check timestamp column (index 0)
        timestamps = ais[:, 0]
        valid_timestamps = timestamps[timestamps > 0]
        
        if len(valid_timestamps) > 0:
            print(f"  Input timestamps: min={valid_timestamps.min():.2f}, max={valid_timestamps.max():.2f}, mean={valid_timestamps.mean():.2f}")
            print(f"  First 5 valid timestamps: {valid_timestamps[:5]}")

if __name__ == "__main__":
    main()
