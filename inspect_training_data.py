#!/usr/bin/env python3
"""
Inspect the actual training data created by phase1_data_preparation.py
"""

import numpy as np
import json
from pathlib import Path

def inspect_training_data():
    """Inspect the training data files and show a summary."""
    print("🔍 INSPECTING TRAINING DATA CREATED BY PHASE1")
    print("=" * 60)
    
    training_dir = Path("data/training_data")
    
    # Check input sequences
    print("\n📊 INPUT SEQUENCES (input_sequences.npy):")
    input_file = training_dir / "input_sequences.npy"
    if input_file.exists():
        input_data = np.load(input_file)
        print(f"  Shape: {input_data.shape}")
        print(f"  Data type: {input_data.dtype}")
        print(f"  Memory usage: {input_data.nbytes / 1024:.1f} KB")
        
        # Show sample values for first sequence
        print(f"  First sequence shape: {input_data[0].shape}")
        print(f"  First sequence sample (first 3 features):")
        for i in range(3):
            print(f"    Feature {i}: {input_data[0, :, i]}")
        
        # Check for any NaN or infinite values
        has_nan = np.any(np.isnan(input_data))
        has_inf = np.any(np.isinf(input_data))
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        # Show feature value ranges
        print(f"  Feature value ranges:")
        for i in range(min(5, input_data.shape[2])):  # First 5 features
            feature_data = input_data[:, :, i]
            print(f"    Feature {i}: {np.min(feature_data):.3f} to {np.max(feature_data):.3f}")
    else:
        print("  ❌ File not found!")
    
    # Check target sequences
    print("\n🎯 TARGET SEQUENCES (target_sequences.npy):")
    target_file = training_dir / "target_sequences.npy"
    if target_file.exists():
        target_data = np.load(target_file)
        print(f"  Shape: {target_data.shape}")
        print(f"  Data type: {target_data.dtype}")
        print(f"  Memory usage: {target_data.nbytes / 1024:.1f} KB")
        
        # Show sample values
        print(f"  First target shape: {target_data[0].shape}")
        print(f"  First target sample (first 10 values): {target_data[0, :10]}")
        
        # Check action counts (first value should be action count)
        action_counts = target_data[:, 0]
        print(f"  Action count stats:")
        print(f"    Min: {np.min(action_counts)}")
        print(f"    Max: {np.max(action_counts)}")
        print(f"    Mean: {np.mean(action_counts):.2f}")
        print(f"    Std: {np.std(action_counts):.2f}")
        
        # Check for any NaN or infinite values
        has_nan = np.any(np.isnan(target_data))
        has_inf = np.any(np.isinf(target_data))
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
    else:
        print("  ❌ File not found!")
    
    # Check action sequences metadata
    print("\n📝 ACTION SEQUENCES METADATA (action_sequences.json):")
    action_file = training_dir / "action_sequences.json"
    if action_file.exists():
        with open(action_file, 'r') as f:
            action_data = json.load(f)
        print(f"  Total sequences: {len(action_data)}")
        
        # Show first few sequences
        print(f"  First sequence:")
        first_seq = action_data[0]
        print(f"    Gamestate index: {first_seq['gamestate_index']}")
        print(f"    Timestamp: {first_seq['gamestate_timestamp']}")
        print(f"    Action count: {first_seq['action_count']}")
        if first_seq['actions']:
            print(f"    First action: {first_seq['actions'][0]}")
        
        # Show action type distribution
        action_types = {}
        for seq in action_data:
            for action in seq['actions']:
                action_type = action['event_type']
                action_types[action_type] = action_types.get(action_type, 0) + 1
        
        print(f"  Action type distribution: {action_types}")
    else:
        print("  ❌ File not found!")
    
    # Check training metadata
    print("\n📋 TRAINING METADATA (training_metadata.json):")
    metadata_file = training_dir / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for key, value in metadata.items():
            if key != 'data_description':
                print(f"  {key}: {value}")
        
        print(f"  Data description:")
        for key, value in metadata['data_description'].items():
            print(f"    {key}: {value}")
    else:
        print("  ❌ File not found!")
    
    # Check data analysis
    print("\n📈 DATA ANALYSIS (data_analysis.json):")
    analysis_file = training_dir / "data_analysis.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        print(f"  Total sequences: {analysis['total_sequences']}")
        print(f"  Total actions: {analysis['total_actions']}")
        print(f"  Action count stats:")
        stats = analysis['action_count_stats']
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Std: {stats['std']:.2f}")
        print(f"    Min: {stats['min']}")
        print(f"    Max: {stats['max']}")
        
        print(f"  Action type distribution: {analysis['action_type_distribution']}")
    else:
        print("  ❌ File not found!")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING DATA INSPECTION COMPLETE")

if __name__ == "__main__":
    inspect_training_data()

