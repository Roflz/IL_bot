#!/usr/bin/env python3
"""
Inspect Numpy Array Tool

This tool allows you to examine the contents of numpy arrays (.npy files) to see
what data they actually contain. Useful for debugging data pipeline issues.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Inspect numpy array contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection of a numpy array
  python tools/inspect_numpy_array.py data/04_sequences/input_sequences.npy
  
  # Show first few values of specific features
  python tools/inspect_numpy_array.py data/04_sequences/input_sequences.npy --feature-indices 0,1,127
  
  # Show specific sequence and timestep
  python tools/inspect_numpy_array.py data/04_sequences/input_sequences.npy --sequence 0 --timestep 0
  
  # Compare two arrays
  python tools/inspect_numpy_array.py data/04_sequences/input_sequences.npy --compare data/04_sequences/normalized_input_sequences.npy
        """
    )
    
    parser.add_argument(
        'file_path',
        help='Path to the .npy file to inspect'
    )
    
    parser.add_argument(
        '--feature-indices',
        help='Comma-separated list of feature indices to show (e.g., "0,1,127")'
    )
    
    parser.add_argument(
        '--sequence',
        type=int,
        help='Specific sequence index to show'
    )
    
    parser.add_argument(
        '--timestep',
        type=int,
        help='Specific timestep to show'
    )
    
    parser.add_argument(
        '--compare',
        help='Path to another .npy file to compare with'
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=10,
        help='Maximum number of rows to show (default: 10)'
    )
    
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show statistical summary of the array'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        print(f"❌ File not found: {args.file_path}")
        return 1
    
    try:
        # Load the numpy array
        print(f"📁 Loading numpy array: {args.file_path}")
        array = np.load(args.file_path)
        
        # Basic array info
        print(f"\n📊 Array Information:")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Size: {array.size:,} elements")
        print(f"  Memory usage: {array.nbytes / 1024 / 1024:.2f} MB")
        
        # Show array structure
        if len(array.shape) == 3:
            print(f"  Structure: {array.shape[0]} sequences × {array.shape[1]} timesteps × {array.shape[2]} features")
        elif len(array.shape) == 2:
            print(f"  Structure: {array.shape[0]} samples × {array.shape[1]} features")
        elif len(array.shape) == 1:
            print(f"  Structure: {array.shape[0]} elements")
        
        # Show statistical summary if requested
        if args.show_stats:
            print(f"\n📈 Statistical Summary:")
            print(f"  Min value: {np.min(array):.6f}")
            print(f"  Max value: {np.max(array):.6f}")
            print(f"  Mean value: {np.mean(array):.6f}")
            print(f"  Std deviation: {np.std(array):.6f}")
            print(f"  Non-zero elements: {np.count_nonzero(array):,}")
        
        # Show specific feature indices if requested
        if args.feature_indices:
            feature_indices = [int(x.strip()) for x in args.feature_indices.split(',')]
            print(f"\n🔍 Showing specific features: {feature_indices}")
            
            if len(array.shape) == 3:
                # For 3D arrays (sequences × timesteps × features)
                for seq_idx in range(min(args.max_rows, array.shape[0])):
                    print(f"\n  Sequence {seq_idx}:")
                    for timestep in range(min(5, array.shape[1])):  # Show first 5 timesteps
                        values = [f"{array[seq_idx, timestep, feat_idx]:.6f}" for feat_idx in feature_indices]
                        print(f"    Timestep {timestep}: {values}")
            elif len(array.shape) == 2:
                # For 2D arrays (samples × features)
                for row_idx in range(min(args.max_rows, array.shape[0])):
                    values = [f"{array[row_idx, feat_idx]:.6f}" for feat_idx in feature_indices]
                    print(f"  Row {row_idx}: {values}")
        
        # Show specific sequence and timestep if requested
        elif args.sequence is not None and args.timestep is not None:
            if len(array.shape) == 3:
                if args.sequence < array.shape[0] and args.timestep < array.shape[1]:
                    print(f"\n🔍 Sequence {args.sequence}, Timestep {args.timestep}:")
                    features = array[args.sequence, args.timestep, :]
                    for feat_idx, value in enumerate(features):
                        print(f"  Feature {feat_idx}: {value:.6f}")
                else:
                    print(f"❌ Sequence {args.sequence} or timestep {args.timestep} out of range")
            else:
                print("❌ Array is not 3D (sequences × timesteps × features)")
        
        # Show first few values by default
        else:
            print(f"\n🔍 First few values:")
            if len(array.shape) == 3:
                # For 3D arrays, show first sequence, first few timesteps, first few features
                print(f"  Sequence 0, Timestep 0, Features 0-9:")
                features = array[0, 0, :10]
                for feat_idx, value in enumerate(features):
                    print(f"    Feature {feat_idx}: {value:.6f}")
                
                print(f"\n  Sequence 0, Timestep 0, Features 120-127 (including timestamp):")
                features = array[0, 0, 120:128]
                for feat_idx, value in enumerate(features):
                    print(f"    Feature {feat_idx}: {value:.6f}")
                
                print(f"\n  Sequence 0, Timestep 1, Features 120-127 (including timestamp):")
                features = array[0, 1, 120:128]
                for feat_idx, value in enumerate(features):
                    print(f"    Feature {feat_idx}: {value:.6f}")
                
            elif len(array.shape) == 2:
                # For 2D arrays, show first few rows and features
                for row_idx in range(min(args.max_rows, array.shape[0])):
                    features = array[row_idx, :10]  # First 10 features
                    values = [f"{x:.6f}" for x in features]
                    print(f"  Row {row_idx}: {values}")
            else:
                # For 1D arrays, show first few elements
                elements = array[:min(args.max_rows, array.shape[0])]
                values = [f"{x:.6f}" for x in elements]
                print(f"  Elements: {values}")
        
        # Compare with another array if requested
        if args.compare:
            if not Path(args.compare).exists():
                print(f"\n❌ Comparison file not found: {args.compare}")
            else:
                print(f"\n🔍 Comparing with: {args.compare}")
                compare_array = np.load(args.compare)
                
                if array.shape != compare_array.shape:
                    print(f"  ❌ Shapes differ: {array.shape} vs {compare_array.shape}")
                else:
                    # Check if arrays are identical
                    if np.array_equal(array, compare_array):
                        print(f"  ✅ Arrays are IDENTICAL (same data)")
                    else:
                        # Check if they're close (within numerical precision)
                        if np.allclose(array, compare_array, rtol=1e-10):
                            print(f"  ⚠️  Arrays are NUMERICALLY EQUAL (within precision)")
                        else:
                            # Find differences
                            diff_mask = ~np.isclose(array, compare_array, rtol=1e-10)
                            diff_count = np.sum(diff_mask)
                            print(f"  ❌ Arrays DIFFER: {diff_count:,} elements differ")
                            
                            # Show some differences
                            if diff_count > 0:
                                diff_indices = np.where(diff_mask)
                                print(f"  First few differences:")
                                for i in range(min(5, len(diff_indices[0]))):
                                    seq_idx = diff_indices[0][i]
                                    timestep_idx = diff_indices[1][i]
                                    feat_idx = diff_indices[2][i]
                                    val1 = array[seq_idx, timestep_idx, feat_idx]
                                    val2 = compare_array[seq_idx, timestep_idx, feat_idx]
                                    print(f"    [{seq_idx}, {timestep_idx}, {feat_idx}]: {val1:.6f} vs {val2:.6f}")
        
        print(f"\n✅ Inspection complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error inspecting array: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
