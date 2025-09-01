#!/usr/bin/env python3
"""
Debug script to check if data is normalized and examine sequence 5
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load the data
    gamestate_sequences = np.load(data_dir / "gamestate_sequences.npy")
    action_targets = np.load(data_dir / "action_targets.npy")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    
    print("=== CHECKING SEQUENCE 5 (FIRST NON-ZERO ACTIONS) ===")
    
    # Check sequence 5 specifically
    seq_idx = 5
    print(f"\nSequence {seq_idx} action targets:")
    at = action_targets[seq_idx, :, :]
    
    # Find non-zero actions
    non_zero_mask = np.any(at != 0, axis=1)
    non_zero_actions = at[non_zero_mask]
    
    if len(non_zero_actions) > 0:
        print(f"Found {len(non_zero_actions)} non-zero actions:")
        for i, action in enumerate(non_zero_actions[:5]):  # Show first 5
            print(f"  Action {i}: {action}")
        
        # Check timestamp values specifically
        timestamps = at[:, 0]
        valid_timestamps = timestamps[timestamps > 0]
        if len(valid_timestamps) > 0:
            print(f"\nTimestamp values: min={valid_timestamps.min():.6f}, max={valid_timestamps.max():.6f}")
            print(f"First 5 timestamps: {valid_timestamps[:5]}")
    else:
        print("No non-zero actions found in sequence 5")
    
    # Check action input sequences for sequence 5
    print(f"\nSequence {seq_idx} action input sequences:")
    ais = action_input_sequences[seq_idx, 0, :, :]  # First timestep
    
    # Find non-zero actions
    non_zero_mask_input = np.any(ais != 0, axis=1)
    non_zero_actions_input = ais[non_zero_mask_input]
    
    if len(non_zero_actions_input) > 0:
        print(f"Found {len(non_zero_actions_input)} non-zero input actions:")
        for i, action in enumerate(non_zero_actions_input[:5]):  # Show first 5
            print(f"  Action {i}: {action}")
        
        # Check timestamp values specifically
        timestamps_input = ais[:, 0]
        valid_timestamps_input = timestamps_input[timestamps_input > 0]
        if len(valid_timestamps_input) > 0:
            print(f"\nInput timestamp values: min={valid_timestamps_input.min():.6f}, max={valid_timestamps_input.max():.6f}")
            print(f"First 5 input timestamps: {valid_timestamps_input[:5]}")
    else:
        print("No non-zero input actions found in sequence 5")
    
    # Check if timestamps are normalized (should be small values if divided by 1000)
    print("\n=== NORMALIZATION CHECK ===")
    
    # Check if input timestamps look normalized (small values)
    if 'valid_timestamps_input' in locals() and len(valid_timestamps_input) > 0:
        if valid_timestamps_input.max() < 1000:
            print("✅ Input timestamps appear NORMALIZED (values < 1000)")
        else:
            print("❌ Input timestamps appear UNNORMALIZED (values >= 1000)")
            print(f"   Max value: {valid_timestamps_input.max():.2f}")
    else:
        print("❌ No valid input timestamps found")
    
    # Check if target timestamps look normalized
    if len(valid_timestamps) > 0:
        if valid_timestamps.max() < 1000:
            print("✅ Target timestamps appear NORMALIZED (values < 1000)")
        else:
            print("❌ Target timestamps appear UNNORMALIZED (values >= 1000)")
            print(f"   Max value: {valid_timestamps.max():.2f}")
    
    # Check the relationship between input and target timestamps
    print("\n=== TIMESTAMP RELATIONSHIP ===")
    if 'valid_timestamps_input' in locals() and len(valid_timestamps_input) > 0 and len(valid_timestamps) > 0:
        # Find matching actions
        min_len = min(len(valid_timestamps_input), len(valid_timestamps))
        if min_len > 0:
            input_vals = valid_timestamps_input[:min_len]
            target_vals = valid_timestamps[:min_len]
            
            print(f"Input vs Target comparison (first {min_len} values):")
            for i in range(min(5, min_len)):
                print(f"  {i}: Input={input_vals[i]:.6f}, Target={target_vals[i]:.6f}, Ratio={input_vals[i]/target_vals[i]:.2f}")
    else:
        print("Cannot compare - missing input timestamps")
    
    # Check gamestate timing features for sequence 5
    print(f"\n=== GAMESTATE TIMING FEATURES (Sequence {seq_idx}) ===")
    gs = gamestate_sequences[seq_idx, 0, :]  # First timestep
    
    timing_indices = [8, 65, 66, 128]
    timing_names = ["time_since_interaction", "phase_start_time", "phase_duration", "timestamp"]
    
    for idx, name in zip(timing_indices, timing_names):
        if idx < len(gs):
            value = gs[idx]
            print(f"{name} (idx {idx}): {value:.2f}")

if __name__ == "__main__":
    main()
