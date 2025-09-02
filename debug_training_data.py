#!/usr/bin/env python3
"""
Debug script to check what the actual training data looks like
"""

import torch
import numpy as np
from ilbot.training.setup import OSRSDataset
from torch.utils.data import DataLoader

def debug_training_data():
    print("🔍 Debugging Training Data")
    print("=" * 50)
    
    # Load the actual training data
    data_dir = "data/recording_sessions/20250831_113719/06_final_training_data"
    
    try:
        dataset = OSRSDataset(data_dir, targets_version="v1")
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample batch
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        
        # Check shapes
        temporal_sequence = sample['temporal_sequence']
        action_sequence = sample['action_sequence']
        action_target = sample['action_target']
        valid_mask = sample['valid_mask']
        
        print(f"\nData shapes:")
        print(f"  temporal_sequence: {temporal_sequence.shape}")
        print(f"  action_sequence: {action_sequence.shape}")
        print(f"  action_target: {action_target.shape}")
        print(f"  valid_mask: {valid_mask.shape}")
        
        # Check valid mask
        print(f"\nValid mask analysis:")
        print(f"  Total actions: {valid_mask.numel()}")
        print(f"  Valid actions: {valid_mask.sum().item()}")
        print(f"  Invalid actions: {(~valid_mask).sum().item()}")
        print(f"  Valid mask pattern: {valid_mask[:10].numpy()}")
        
        # Check action sequence content
        print(f"\nAction sequence analysis:")
        print(f"  Action sequence shape: {action_sequence.shape}")
        print(f"  First timestep actions (first 5):")
        for i in range(min(5, action_sequence.shape[1])):
            action = action_sequence[0, i, :]  # [timestep=0, action=i, features]
            print(f"    Action {i}: {action.numpy()}")
        
        # Check if action sequences have meaningful temporal data
        print(f"\nTemporal data analysis:")
        for t in range(min(3, action_sequence.shape[0])):  # Check first 3 timesteps
            timestep_actions = action_sequence[t, :, :]  # [actions, features]
            valid_actions = timestep_actions[valid_mask]
            if len(valid_actions) > 0:
                print(f"  Timestep {t}: {len(valid_actions)} valid actions")
                print(f"    First action timing: {valid_actions[0, 0].item():.6f}s")
                if len(valid_actions) > 1:
                    print(f"    Second action timing: {valid_actions[1, 0].item():.6f}s")
                    print(f"    Time difference: {valid_actions[1, 0].item() - valid_actions[0, 0].item():.6f}s")
        
        # Check action targets
        print(f"\nAction target analysis:")
        print(f"  Action target shape: {action_target.shape}")
        valid_targets = action_target[valid_mask]
        print(f"  Valid targets: {len(valid_targets)}")
        if len(valid_targets) > 0:
            print(f"  First target: {valid_targets[0].numpy()}")
            print(f"  Target event types: {np.unique(valid_targets[:, 0])}")
            print(f"  Target timing range: {valid_targets[:, 0].min().item():.6f}s - {valid_targets[:, 0].max().item():.6f}s")
        
        # Check if the data actually has multiple actions per gamestate
        print(f"\nActions per gamestep analysis:")
        actions_per_timestep = []
        for t in range(action_sequence.shape[0]):
            timestep_valid = valid_mask & (action_sequence[t, :, 0] != 0)  # Check if timing is non-zero
            actions_per_timestep.append(timestep_valid.sum().item())
        
        print(f"  Actions per timestep: {actions_per_timestep}")
        print(f"  Average actions per timestep: {np.mean(actions_per_timestep):.2f}")
        print(f"  Max actions per timestep: {np.max(actions_per_timestep)}")
        print(f"  Min actions per timestep: {np.min(actions_per_timestep)}")
        
        # Check if this matches the training metrics
        print(f"\n🔍 Comparison with training metrics:")
        print(f"  Training showed: 16.3 ± 17.6 actions per gamestate")
        print(f"  Actual data shows: {np.mean(actions_per_timestep):.1f} ± {np.std(actions_per_timestep):.1f}")
        
        if abs(np.mean(actions_per_timestep) - 16.3) < 1.0:
            print(f"  ✅ Data matches training metrics")
        else:
            print(f"  ❌ Data does NOT match training metrics!")
            print(f"     This suggests the metrics calculation is wrong!")
        
        # Check multiple samples to see if this is consistent
        print(f"\n🔍 Checking multiple samples:")
        total_actions = 0
        total_valid = 0
        for i in range(min(5, len(dataset))):
            sample_i = dataset[i]
            valid_count = sample_i['valid_mask'].sum().item()
            total_valid += valid_count
            total_actions += 100
            print(f"  Sample {i}: {valid_count} valid actions")
        
        print(f"  Average valid actions across samples: {total_valid / min(5, len(dataset)):.1f}")
        print(f"  This suggests the data has very few valid actions per sample")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_data()
