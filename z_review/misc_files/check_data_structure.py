#!/usr/bin/env python3
"""Check the structure of training data files"""

import json
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/06_final_training_data")
    
    print("=== Training Data Structure Analysis ===\n")
    
    # Check gamestate sequences
    gamestate_file = data_dir / "gamestate_sequences.npy"
    if gamestate_file.exists():
        data = np.load(gamestate_file)
        print(f"Gamestate sequences:")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Sample values (first 5 features, first 3 timesteps):")
        print(f"    {data[0, :3, :5]}")
        print()
    
    # Check action input sequences
    action_input_file = data_dir / "action_input_sequences.json"
    if action_input_file.exists():
        with open(action_input_file, 'r') as f:
            data = json.load(f)
        print(f"Action input sequences:")
        print(f"  Count: {len(data)}")
        print(f"  First sequence type: {type(data[0])}")
        print(f"  First sequence length: {len(data[0])}")
        print(f"  Sample first sequence (first 100 elements):")
        print(f"    {data[0][:100]}")
        print()
    
    # Check action targets
    action_targets_file = data_dir / "action_targets.json"
    if action_targets_file.exists():
        with open(action_targets_file, 'r') as f:
            data = json.load(f)
        print(f"Action targets:")
        print(f"  Count: {len(data)}")
        print(f"  First target type: {type(data[0])}")
        print(f"  First target length: {len(data[0])}")
        print(f"  Sample first target (first 20 elements):")
        print(f"    {data[0][:20]}")
        print()

if __name__ == "__main__":
    main()
