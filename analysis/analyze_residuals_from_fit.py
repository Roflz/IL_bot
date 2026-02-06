#!/usr/bin/env python3
"""
Analyze residuals from a camera model fitting run.

This script helps understand why the cost is high by analyzing the distribution
of residuals and identifying outliers.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.fit_camera_model import (
    load_calibration_files,
    prepare_training_data,
    reprojection_error_residuals
)


def analyze_residuals(residuals: np.ndarray, verbose: bool = True):
    """
    Analyze residual distribution and explain why cost is high.
    
    Args:
        residuals: Array of residuals (x errors and y errors interleaved)
        verbose: If True, print detailed analysis
    """
    n_residuals = len(residuals)
    n_points = n_residuals // 2
    
    # Calculate cost
    cost = 0.5 * np.sum(residuals**2)
    
    # Statistics on absolute residuals
    abs_residuals = np.abs(residuals)
    
    # Count by error magnitude
    small_errors = np.sum(abs_residuals < 100)
    medium_errors = np.sum((abs_residuals >= 100) & (abs_residuals < 500))
    large_errors = np.sum((abs_residuals >= 500) & (abs_residuals < 1000))
    huge_errors = np.sum(abs_residuals >= 1000)
    
    # Calculate contribution to cost by error magnitude
    small_cost = 0.5 * np.sum(residuals[abs_residuals < 100]**2)
    medium_cost = 0.5 * np.sum(residuals[(abs_residuals >= 100) & (abs_residuals < 500)]**2)
    large_cost = 0.5 * np.sum(residuals[(abs_residuals >= 500) & (abs_residuals < 1000)]**2)
    huge_cost = 0.5 * np.sum(residuals[abs_residuals >= 1000]**2)
    
    if verbose:
        print("=" * 80)
        print("RESIDUAL ANALYSIS")
        print("=" * 80)
        print(f"Total residuals: {n_residuals:,}")
        print(f"Total training points: {n_points:,}")
        print(f"Cost (0.5 * sum(residuals²)): {cost:.2e}")
        print()
        
        print("Absolute residual statistics:")
        print(f"  Mean: {np.mean(abs_residuals):.2f} pixels")
        print(f"  Median: {np.median(abs_residuals):.2f} pixels")
        print(f"  Std: {np.std(abs_residuals):.2f} pixels")
        print(f"  Min: {np.min(abs_residuals):.2f} pixels")
        print(f"  Max: {np.max(abs_residuals):.2f} pixels")
        print(f"  25th percentile: {np.percentile(abs_residuals, 25):.2f} pixels")
        print(f"  75th percentile: {np.percentile(abs_residuals, 75):.2f} pixels")
        print(f"  95th percentile: {np.percentile(abs_residuals, 95):.2f} pixels")
        print(f"  99th percentile: {np.percentile(abs_residuals, 99):.2f} pixels")
        print()
        
        print("Error distribution:")
        print(f"  < 100 pixels: {small_errors:,} ({100*small_errors/n_residuals:.1f}%)")
        print(f"  100-500 pixels: {medium_errors:,} ({100*medium_errors/n_residuals:.1f}%)")
        print(f"  500-1000 pixels: {large_errors:,} ({100*large_errors/n_residuals:.1f}%)")
        print(f"  >= 1000 pixels: {huge_errors:,} ({100*huge_errors/n_residuals:.1f}%)")
        print()
        
        print("Cost contribution by error magnitude:")
        print(f"  < 100 pixels: {small_cost:.2e} ({100*small_cost/cost:.1f}% of total cost)")
        print(f"  100-500 pixels: {medium_cost:.2e} ({100*medium_cost/cost:.1f}% of total cost)")
        print(f"  500-1000 pixels: {large_cost:.2e} ({100*large_cost/cost:.1f}% of total cost)")
        print(f"  >= 1000 pixels: {huge_cost:.2e} ({100*huge_cost/cost:.1f}% of total cost)")
        print()
        
        # Show examples of huge errors
        huge_indices = np.where(abs_residuals >= 1000)[0]
        if len(huge_indices) > 0:
            print(f"Examples of huge errors (>= 1000 pixels):")
            for idx in huge_indices[:10]:
                print(f"  Residual[{idx}] = {residuals[idx]:.2f} pixels "
                      f"(contributes {0.5*residuals[idx]**2:.2e} to cost)")
            if len(huge_indices) > 10:
                print(f"  ... and {len(huge_indices) - 10} more")
        print()
        
        # Explain why cost is high
        print("=" * 80)
        print("WHY THE COST IS HIGH")
        print("=" * 80)
        print(f"Even though most errors are reasonable (median = {np.median(abs_residuals):.1f} pixels),")
        print(f"the cost is high because outliers dominate the sum-of-squares cost function.")
        print()
        print(f"Key insight: Cost = 0.5 × sum(residuals²)")
        print(f"  - A 100-pixel error contributes: 0.5 × 100² = 5,000 to cost")
        print(f"  - A 1000-pixel error contributes: 0.5 × 1000² = 500,000 to cost")
        print(f"  - So a 1000-pixel error contributes 100× more than a 100-pixel error!")
        print()
        if huge_errors > 0:
            print(f"Your data has {huge_errors:,} residuals with errors >= 1000 pixels.")
            print(f"These {100*huge_errors/n_residuals:.1f}% of residuals contribute {100*huge_cost/cost:.1f}% of the cost!")
        print()
        print("RECOMMENDATION:")
        print("  Filter out outliers (errors > 500 or > 1000 pixels) before fitting.")
        print("  This will allow the optimizer to focus on the majority of well-behaved data.")
        print("=" * 80)
    
    return {
        "n_residuals": n_residuals,
        "n_points": n_points,
        "cost": cost,
        "mean_error": np.mean(abs_residuals),
        "median_error": np.median(abs_residuals),
        "max_error": np.max(abs_residuals),
        "small_errors": small_errors,
        "medium_errors": medium_errors,
        "large_errors": large_errors,
        "huge_errors": huge_errors,
        "huge_cost_pct": 100 * huge_cost / cost if cost > 0 else 0
    }


def main():
    """Main function to analyze residuals from a fitting run."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze residuals from camera model fitting")
    parser.add_argument("data_file", help="Path to calibration data JSONL file")
    parser.add_argument("--k", type=float, default=1000.0, help="Initial k value for focal length")
    parser.add_argument("--offset", type=float, default=0.0, help="Initial offset value (for shifted formula)")
    parser.add_argument("--formula", choices=["inverse", "direct", "shifted", "fov_based"], 
                       default="inverse", help="Focal length formula to use")
    parser.add_argument("--no-screen-filter", action="store_true", 
                       help="Don't filter objects outside screen bounds")
    parser.add_argument("--limit", type=int, help="Limit number of training points (for testing)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading data from: {args.data_file}")
    raw_data = load_calibration_files([args.data_file])
    
    # Prepare training data
    print(f"[INFO] Preparing training data...")
    training_data = prepare_training_data(raw_data, filter_screen_coords=not args.no_screen_filter)
    
    if args.limit:
        training_data = training_data[:args.limit]
        print(f"[INFO] Limited to {len(training_data)} training points")
    
    print(f"[INFO] Using {len(training_data)} training points")
    
    # Calculate residuals with initial parameters
    params = np.array([args.k, args.offset]) if args.formula == "shifted" else np.array([args.k])
    residuals = reprojection_error_residuals(params, training_data, args.formula, debug=False)
    
    # Analyze residuals
    analyze_residuals(np.array(residuals), verbose=True)


if __name__ == "__main__":
    main()


