#!/usr/bin/env python3
"""
Debug script to investigate player position and timing issues
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load the data
    gamestate_sequences = np.load(data_dir / "gamestate_sequences.npy")
    action_targets = np.load(data_dir / "action_targets.npy")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    
    print("=== DATA SHAPES ===")
    print(f"Gamestate sequences: {gamestate_sequences.shape}")
    print(f"Action targets: {action_targets.shape}")
    print(f"Action input sequences: {action_input_sequences.shape}")
    
    # Check player position features (indices 0, 1 should be player_world_x, player_world_y)
    print("\n=== PLAYER POSITION ANALYSIS ===")
    print("First few gamestate sequences (first timestep):")
    for i in range(min(3, gamestate_sequences.shape[0])):
        gs = gamestate_sequences[i, 0, :]  # First timestep of sequence i
        print(f"Sequence {i}: player_x={gs[0]:.2f}, player_y={gs[1]:.2f}")
    
    print("\nPlayer position statistics across all data:")
    all_player_x = gamestate_sequences[:, :, 0].flatten()
    all_player_y = gamestate_sequences[:, :, 1].flatten()
    print(f"Player X - min: {all_player_x.min():.2f}, max: {all_player_x.max():.2f}, mean: {all_player_x.mean():.2f}")
    print(f"Player Y - min: {all_player_y.min():.2f}, max: {all_player_y.max():.2f}, mean: {all_player_y.mean():.2f}")
    
    # Check timing features
    print("\n=== TIMING ANALYSIS ===")
    print("Action input sequences timing (first few):")
    for i in range(min(3, action_input_sequences.shape[0])):
        ais = action_input_sequences[i, 0, :, 0]  # First timestep, all actions, timestamp column
        valid_timestamps = ais[ais > 0]  # Only valid (non-padding) timestamps
        if len(valid_timestamps) > 0:
            print(f"Sequence {i}: timestamps range {valid_timestamps.min():.2f} to {valid_timestamps.max():.2f}")
    
    print("\nAction targets timing (first few):")
    for i in range(min(3, action_targets.shape[0])):
        at = action_targets[i, :, 0]  # All actions, timestamp column
        valid_deltas = at[at > 0]  # Only valid (non-padding) deltas
        if len(valid_deltas) > 0:
            print(f"Sequence {i}: deltas range {valid_deltas.min():.2f} to {valid_deltas.max():.2f}")
    
    # Check gamestate timing features
    print("\nGamestate timing features:")
    # time_since_interaction (index 8), phase_start_time (65), phase_duration (66), timestamp (128)
    timing_indices = [8, 65, 66, 128]
    timing_names = ["time_since_interaction", "phase_start_time", "phase_duration", "timestamp"]
    
    for idx, name in zip(timing_indices, timing_names):
        if idx < gamestate_sequences.shape[2]:
            values = gamestate_sequences[:, :, idx].flatten()
            valid_values = values[values > 0]
            if len(valid_values) > 0:
                print(f"{name} (idx {idx}): min={valid_values.min():.2f}, max={valid_values.max():.2f}, mean={valid_values.mean():.2f}")
            else:
                print(f"{name} (idx {idx}): all zeros")
    
    # Check if there's a relationship between input timestamps and target deltas
    print("\n=== TIMESTAMP vs DELTA RELATIONSHIP ===")
    # Look at a few sequences to see the relationship
    for seq_idx in range(min(2, action_input_sequences.shape[0])):
        print(f"\nSequence {seq_idx}:")
        input_timestamps = action_input_sequences[seq_idx, 0, :, 0]  # First timestep, all actions
        target_deltas = action_targets[seq_idx, :, 0]  # All actions
        
        # Find valid pairs
        valid_mask = (input_timestamps > 0) & (target_deltas > 0)
        if valid_mask.sum() > 0:
            valid_inputs = input_timestamps[valid_mask]
            valid_targets = target_deltas[valid_mask]
            print(f"  Valid pairs: {valid_mask.sum()}")
            print(f"  Input timestamps: {valid_inputs[:5]}")
            print(f"  Target deltas: {valid_targets[:5]}")
            print(f"  Ratio (input/target): {(valid_inputs / valid_targets)[:5]}")

if __name__ == "__main__":
    main()
