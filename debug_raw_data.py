#!/usr/bin/env python3
"""
Debug script to check raw data values
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load action targets
    action_targets = np.load(data_dir / "action_targets.npy")
    
    print("=== ACTION TARGETS RAW VALUES ===")
    print(f"Shape: {action_targets.shape}")
    
    # Check the first sequence in detail
    seq_0 = action_targets[0, :, :]
    print(f"\nFirst sequence shape: {seq_0.shape}")
    print("First 10 actions (all columns):")
    for i in range(min(10, seq_0.shape[0])):
        print(f"  Action {i}: {seq_0[i, :]}")
    
    # Check for any non-zero values
    print(f"\nNon-zero values in first sequence:")
    for col in range(seq_0.shape[1]):
        non_zero = seq_0[:, col][seq_0[:, col] != 0]
        if len(non_zero) > 0:
            print(f"  Column {col}: {len(non_zero)} non-zero values, range: {non_zero.min():.6f} to {non_zero.max():.6f}")
        else:
            print(f"  Column {col}: all zeros")
    
    # Check action input sequences
    print("\n=== ACTION INPUT SEQUENCES RAW VALUES ===")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    print(f"Shape: {action_input_sequences.shape}")
    
    # Check first sequence, first timestep
    seq_0_t0 = action_input_sequences[0, 0, :, :]
    print(f"\nFirst sequence, first timestep shape: {seq_0_t0.shape}")
    print("First 10 actions (all columns):")
    for i in range(min(10, seq_0_t0.shape[0])):
        print(f"  Action {i}: {seq_0_t0[i, :]}")
    
    # Check for any non-zero values
    print(f"\nNon-zero values in first sequence, first timestep:")
    for col in range(seq_0_t0.shape[1]):
        non_zero = seq_0_t0[:, col][seq_0_t0[:, col] != 0]
        if len(non_zero) > 0:
            print(f"  Column {col}: {len(non_zero)} non-zero values, range: {non_zero.min():.6f} to {non_zero.max():.6f}")
        else:
            print(f"  Column {col}: all zeros")

if __name__ == "__main__":
    main()
