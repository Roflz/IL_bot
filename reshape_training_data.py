#!/usr/bin/env python3
"""
Reshape training data to match model expected input format.

Current: (111, 10, 313) flattened actions 
Target:  (111, 10, 101, 8) structured actions

The flattened format is: [action_count, action1_feat1, action1_feat2, ..., action1_feat8, action2_feat1, ...]
The structured format is: [[action_count, 0, 0, 0, 0, 0, 0, 0], [action1_feat1, action1_feat2, ..., action1_feat8], ...]
"""

import json
import numpy as np
from pathlib import Path

def reshape_flattened_to_structured(flattened_actions, max_actions=100):
    """
    Reshape flattened action sequences to structured format.
    
    Args:
        flattened_actions: List of sequences, each with shape (10, 313)
        max_actions: Maximum number of actions to support (default 100)
        
    Returns:
        Structured actions with shape (num_sequences, 10, max_actions+1, 8)
    """
    print(f"Reshaping {len(flattened_actions)} sequences...")
    
    structured_sequences = []
    
    for seq_idx, sequence in enumerate(flattened_actions):
        structured_timesteps = []
        
        for timestep_idx, timestep_data in enumerate(sequence):
            # Parse the flattened timestep data
            if len(timestep_data) < 1:
                # Empty timestep
                structured_timestep = np.zeros((max_actions + 1, 8))
                structured_timesteps.append(structured_timestep)
                continue
                
            action_count = int(timestep_data[0])
            
            # Create structured timestep: (max_actions+1, 8)
            structured_timestep = np.zeros((max_actions + 1, 8))
            
            # First row contains action count in first column, rest zeros
            structured_timestep[0, 0] = action_count
            
            # Parse individual actions
            if action_count > 0 and len(timestep_data) > 1:
                actions_data = timestep_data[1:]  # Skip the count
                
                # Each action has 8 features
                num_complete_actions = min(len(actions_data) // 8, max_actions)
                
                for action_idx in range(num_complete_actions):
                    start_idx = action_idx * 8
                    end_idx = start_idx + 8
                    
                    if end_idx <= len(actions_data):
                        action_features = actions_data[start_idx:end_idx]
                        structured_timestep[action_idx + 1, :] = action_features
            
            structured_timesteps.append(structured_timestep)
        
        structured_sequences.append(np.array(structured_timesteps))
    
    # Convert to numpy array
    structured_array = np.array(structured_sequences)
    print(f"Reshaped to: {structured_array.shape}")
    
    return structured_array

def analyze_action_distribution(flattened_actions):
    """Analyze the distribution of action counts in the data."""
    print("\n=== Action Distribution Analysis ===")
    
    action_counts = []
    max_actions_found = 0
    
    for sequence in flattened_actions:
        for timestep_data in sequence:
            if len(timestep_data) >= 1:
                action_count = int(timestep_data[0])
                action_counts.append(action_count)
                
                # Calculate actual number of actions from data length
                if action_count > 0:
                    available_features = len(timestep_data) - 1  # Subtract count
                    actual_actions = available_features // 8
                    max_actions_found = max(max_actions_found, actual_actions)
    
    action_counts = np.array(action_counts)
    
    print(f"Action count statistics:")
    print(f"  - Mean: {np.mean(action_counts):.2f}")
    print(f"  - Max: {np.max(action_counts)}")
    print(f"  - Min: {np.min(action_counts)}")
    print(f"  - 95th percentile: {np.percentile(action_counts, 95):.1f}")
    print(f"  - 99th percentile: {np.percentile(action_counts, 99):.1f}")
    print(f"  - Maximum actions found in data: {max_actions_found}")
    
    return max_actions_found

def main():
    print("=== Reshaping Training Data ===")
    
    # Load current training data
    print("Loading action input sequences...")
    with open('data/06_final_training_data/action_input_sequences.json', 'r') as f:
        action_inputs = json.load(f)
    
    print(f"Current shape: ({len(action_inputs)}, {len(action_inputs[0])}, {len(action_inputs[0][0])})")
    
    # Analyze action distribution
    max_actions_in_data = analyze_action_distribution(action_inputs)
    
    # Decide on max_actions parameter
    if max_actions_in_data <= 50:
        max_actions = 50
        print(f"\nUsing max_actions = {max_actions} (sufficient for data)")
    elif max_actions_in_data <= 100:
        max_actions = 100
        print(f"\nUsing max_actions = {max_actions} (model default)")
    else:
        max_actions = max_actions_in_data
        print(f"\nUsing max_actions = {max_actions} (data-driven)")
    
    # Reshape the data
    structured_actions = reshape_flattened_to_structured(action_inputs, max_actions)
    
    print(f"\n=== Results ===")
    print(f"Original: {len(action_inputs)} sequences × 10 timesteps × {len(action_inputs[0][0])} features")
    print(f"Reshaped: {structured_actions.shape}")
    print(f"Model expects: (batch, 10, {max_actions+1}, 8)")
    
    # Save reshaped data
    output_file = f'data/06_final_training_data/action_input_sequences_structured_{max_actions+1}x8.npy'
    np.save(output_file, structured_actions)
    print(f"\nSaved reshaped data to: {output_file}")
    
    # Update metadata
    metadata_file = 'data/06_final_training_data/metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Add reshape info
    metadata['reshaping'] = {
        'original_action_shape': [len(action_inputs), len(action_inputs[0]), len(action_inputs[0][0])],
        'structured_action_shape': structured_actions.shape.tolist(),
        'max_actions_supported': max_actions,
        'max_actions_in_data': max_actions_in_data,
        'structured_file': output_file.split('/')[-1]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata: {metadata_file}")
    
    # Test first sequence
    print(f"\n=== Sample Verification ===")
    print(f"First timestep of first sequence:")
    print(f"  Action count: {structured_actions[0, 0, 0, 0]}")
    print(f"  First action features: {structured_actions[0, 0, 1, :]}")
    
    return structured_actions

if __name__ == "__main__":
    structured_actions = main()


