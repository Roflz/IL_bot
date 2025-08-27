#!/usr/bin/env python3

import json
import numpy as np

print("=== Analyzing Training Data Structure ===")

# Load action input sequences
with open('data/06_final_training_data/action_input_sequences.json', 'r') as f:
    action_inputs = json.load(f)

print(f"Action Input Sequences:")
print(f"  - Number of sequences: {len(action_inputs)}")
print(f"  - Timesteps per sequence: {len(action_inputs[0])}")
print(f"  - Features per timestep: {len(action_inputs[0][0])}")
print(f"  - Shape: ({len(action_inputs)}, {len(action_inputs[0])}, {len(action_inputs[0][0])})")

# Check if this is flattened action data
first_timestep = action_inputs[0][0]
print(f"  - First timestep sample: {first_timestep[:16]}")

# Load gamestate sequences
gamestate_sequences = np.load('data/06_final_training_data/gamestate_sequences.npy')
print(f"\nGamestate Sequences:")
print(f"  - Shape: {gamestate_sequences.shape}")

# Load action targets
with open('data/06_final_training_data/action_targets.json', 'r') as f:
    action_targets = json.load(f)

print(f"\nAction Targets:")
print(f"  - Number of targets: {len(action_targets)}")
print(f"  - First target length: {len(action_targets[0])}")
print(f"  - First target sample: {action_targets[0][:16]}")

print(f"\n=== Model Expected vs Actual ===")
print(f"Model expects:")
print(f"  - Gamestate: (batch, 10, 128)")
print(f"  - Actions: (batch, 10, 101, 8)  # 101 = action_count + 100 actions, 8 features each")

print(f"\nActual data:")
print(f"  - Gamestate: {gamestate_sequences.shape}")
print(f"  - Actions: ({len(action_inputs)}, {len(action_inputs[0])}, {len(action_inputs[0][0])})")

# Check if action input is flattened
if len(action_inputs[0][0]) > 8:
    print(f"\n⚠️  MISMATCH DETECTED:")
    print(f"  Action inputs appear to be FLATTENED with {len(action_inputs[0][0])} features per timestep")
    print(f"  Model expects STRUCTURED actions: (num_actions, 8_features_each)")
    print(f"  Need to reshape from flat to structured format!")
    
    # Estimate structure
    features_per_timestep = len(action_inputs[0][0])
    if features_per_timestep % 8 == 1:  # Includes action count
        max_actions = (features_per_timestep - 1) // 8
        print(f"  Estimated: action_count + {max_actions} actions × 8 features = {features_per_timestep} total")
else:
    print(f"✅ Data structure looks correct")


