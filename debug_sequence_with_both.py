#!/usr/bin/env python3
"""
Debug script to find a sequence with both input and target data
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/recording_sessions/20250831_113719/06_final_training_data")
    
    # Load the data
    action_targets = np.load(data_dir / "action_targets.npy")
    action_input_sequences = np.load(data_dir / "action_input_sequences.npy")
    
    print("=== FINDING SEQUENCE WITH BOTH INPUT AND TARGET DATA ===")
    
    # Find sequences that have both input and target data
    for seq_idx in range(min(20, action_targets.shape[0])):
        # Check targets
        at = action_targets[seq_idx, :, :]
        target_non_zero = np.count_nonzero(at)
        
        # Check inputs (first timestep)
        ais = action_input_sequences[seq_idx, 0, :, :]
        input_non_zero = np.count_nonzero(ais)
        
        if target_non_zero > 0 and input_non_zero > 0:
            print(f"\n✅ Sequence {seq_idx}: {target_non_zero} target non-zero, {input_non_zero} input non-zero")
            
            # Show target timestamps
            target_timestamps = at[:, 0]
            valid_target_timestamps = target_timestamps[target_timestamps > 0]
            if len(valid_target_timestamps) > 0:
                print(f"  Target timestamps: {valid_target_timestamps[:5]}")
            
            # Show input timestamps
            input_timestamps = ais[:, 0]
            valid_input_timestamps = input_timestamps[input_timestamps > 0]
            if len(valid_input_timestamps) > 0:
                print(f"  Input timestamps: {valid_input_timestamps[:5]}")
            
            # Check if they're normalized
            if len(valid_target_timestamps) > 0:
                if valid_target_timestamps.max() < 1000:
                    print("  ✅ Target timestamps appear NORMALIZED")
                else:
                    print("  ❌ Target timestamps appear UNNORMALIZED")
            
            if len(valid_input_timestamps) > 0:
                if valid_input_timestamps.max() < 1000:
                    print("  ✅ Input timestamps appear NORMALIZED")
                else:
                    print("  ❌ Input timestamps appear UNNORMALIZED")
            
            # Show the relationship
            if len(valid_target_timestamps) > 0 and len(valid_input_timestamps) > 0:
                min_len = min(len(valid_target_timestamps), len(valid_input_timestamps))
                if min_len > 0:
                    print(f"  Comparison (first {min_len} values):")
                    for i in range(min(3, min_len)):
                        target_val = valid_target_timestamps[i]
                        input_val = valid_input_timestamps[i]
                        ratio = input_val / target_val if target_val != 0 else float('inf')
                        print(f"    {i}: Input={input_val:.2f}, Target={target_val:.2f}, Ratio={ratio:.2f}")
            
            break  # Found one, stop looking
    
    # Check what the training data looks like when loaded by the dataset
    print("\n=== CHECKING OSRSDataset NORMALIZATION ===")
    
    # Import and test the dataset
    try:
        from ilbot.training.setup import OSRSDataset
        
        dataset = OSRSDataset(data_dir, targets_version="v1")
        
        # Get a sample
        sample = dataset[5]  # Same sequence we looked at
        
        print(f"Dataset sample keys: {list(sample.keys())}")
        
        if 'action_targets' in sample:
            at_sample = sample['action_targets']
            print(f"Sample action targets shape: {at_sample.shape}")
            
            # Check timestamps in the sample
            timestamps = at_sample[:, 0]
            valid_timestamps = timestamps[timestamps > 0]
            if len(valid_timestamps) > 0:
                print(f"Sample target timestamps: {valid_timestamps[:5]}")
                if valid_timestamps.max() < 1000:
                    print("✅ Sample target timestamps appear NORMALIZED")
                else:
                    print("❌ Sample target timestamps appear UNNORMALIZED")
        
        if 'action_input_sequences' in sample:
            ais_sample = sample['action_input_sequences']
            print(f"Sample action input sequences shape: {ais_sample.shape}")
            
            # Check timestamps in the sample (first timestep)
            timestamps = ais_sample[0, :, 0]
            valid_timestamps = timestamps[timestamps > 0]
            if len(valid_timestamps) > 0:
                print(f"Sample input timestamps: {valid_timestamps[:5]}")
                if valid_timestamps.max() < 1000:
                    print("✅ Sample input timestamps appear NORMALIZED")
                else:
                    print("❌ Sample input timestamps appear UNNORMALIZED")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
