#!/usr/bin/env python3
"""
Quick script to inspect numpy arrays in the final training data
"""

import numpy as np
from pathlib import Path

def inspect_numpy_file(file_path):
    """Print a numpy array in raw format"""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Set numpy to use regular decimal notation instead of scientific
        np.set_printoptions(suppress=True, precision=3)
        
        # Load the numpy array
        data = np.load(file_path)
        
        # Print basic info
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Size: {data.size}")
        print(f"Memory usage: {data.nbytes / 1024:.1f} KB")
        
        # Print the raw array with context for input sequences
        print(f"\nRAW ARRAY CONTENTS:")
        print(f"{'='*40}")
        
        # Add context for input sequences
        if len(data.shape) == 3 and data.shape[1] == 10 and data.shape[2] == 128:
            print("(This appears to be gamestate input sequences: N sequences × 10 timesteps × 128 features)")
            print(f"Showing all {data.shape[0]} sequences:")
            print("(Showing only player world positions and time variables for readability)")
            
            # For gamestate sequences, show a more readable format
            for seq_idx in range(min(3, data.shape[0])):  # Show first 3 sequences
                print(f"\nSequence {seq_idx}:")
                for timestep_idx in range(min(5, data.shape[1])):  # Show first 5 timesteps
                    features = data[seq_idx, timestep_idx]
                    # Extract key features: world_x, world_y, time features
                    world_x = features[0] if len(features) > 0 else 0
                    world_y = features[1] if len(features) > 1 else 0
                    time_8 = features[8] if len(features) > 8 else 0
                    time_64 = features[64] if len(features) > 64 else 0
                    time_65 = features[65] if len(features) > 65 else 0
                    time_127 = features[127] if len(features) > 127 else 0
                    print(f"  Timestep {timestep_idx}: world_x={world_x:8.1f}, world_y={world_y:8.1f}, time_8={time_8:8.1f}, time_64={time_64:8.1f}, time_65={time_65:8.1f}, time_127={time_127:8.1f}")
                if data.shape[1] > 5:
                    print(f"  ... and {data.shape[1] - 5} more timesteps")
                    
        elif len(data.shape) == 3 and data.shape[1] == 10 and data.shape[2] == 8:
            print("(This appears to be action input sequences: N sequences × 10 timesteps × 8 features)")
            print(f"Showing all {data.shape[0]} sequences:")
        elif len(data.shape) == 2 and data.shape[1] == 8:
            print("(This appears to be action targets: N sequences × 8 features)")
            print(f"Showing all {data.shape[0]} sequences:")
        
        print(data)
        print(f"{'='*40}")
        
        # If it's a large array, show first and last feature vectors or elements
        if data.size > 160:
            if len(data.shape) == 3 and data.shape[2] == 128:
                # For gamestate sequences, show first and last feature vectors
                print(f"\nFIRST FEATURE VECTOR (128 features):")
                print(f"{'='*60}")
                first_features = data[0, 0, :]  # First sequence, first timestep
                for i in range(0, 128, 8):
                    line_features = first_features[i:i+8]
                    line_str = " ".join(f"{f:8.3f}" for f in line_features)
                    print(line_str)
                
                print(f"\nLAST FEATURE VECTOR (128 features):")
                print(f"{'='*60}")
                last_features = data[-1, -1, :]  # Last sequence, last timestep
                for i in range(0, 128, 8):
                    line_features = last_features[i:i+8]
                    line_str = " ".join(f"{f:8.3f}" for f in line_features)
                    print(line_str)
                
            else:
                # For other data types, show first and last 160 elements
                print(f"\nFIRST 160 ELEMENTS (20 actions × 8 features):")
                print(f"{'='*60}")
                flat_data = data.flatten()
                
                # Print first 160 elements, 8 per line
                for i in range(0, 160, 8):
                    line_elements = flat_data[i:i+8]
                    line_str = " ".join(f"{f:8.3f}" for f in line_elements)
                    print(line_str)
                
                print(f"\nLAST 160 ELEMENTS (20 actions × 8 features):")
                print(f"{'='*60}")
                
                # Print last 160 elements, 8 per line
                for i in range(max(0, len(flat_data) - 160), len(flat_data), 8):
                    line_elements = flat_data[i:i+8]
                    line_str = " ".join(f"{f:8.3f}" for f in line_elements)
                    print(line_str)
            
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python inspect_numpy.py <filepath>")
        print("Example: python inspect_numpy.py data/06_final_training_data/gamestate_sequences.npy")
        return
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    if not file_path.suffix == '.npy':
        print(f"❌ Not a numpy file: {file_path}")
        print("Please provide a .npy file")
        return
    
    # Inspect the specified numpy file
    inspect_numpy_file(file_path)

if __name__ == "__main__":
    main()
