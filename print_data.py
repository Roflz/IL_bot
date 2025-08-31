#!/usr/bin/env python3
import numpy as np

# Load the data
action_targets = np.load("data/recording_sessions/20250831_113719/06_final_training_data/action_targets.npy")

print("First 10 rows of first 10 sequences:")
print("=" * 50)

for seq_idx in range(10):
    print(f"\nSequence {seq_idx}:")
    for row_idx in range(10):
        row = action_targets[seq_idx, row_idx]
        print(f"  Row {row_idx}: {row.tolist()}")
