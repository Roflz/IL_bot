"""
Estimate Initial k Values for Focal Length Formulas

This script analyzes a sample of calibration data to estimate good initial
guesses for the k parameter in each focal length formula.

Method:
1. Sample a subset of data points
2. For each formula, solve for k analytically using a few data points
3. Take median/average of solutions to get robust initial guess
"""

import json
import math
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.camera_projection import (
    world_to_camera_space,
    camera_to_screen,
    SCREEN_WIDTH,
    SCREEN_HEIGHT
)


def estimate_k_inverse(zoom: int, x_ndc: float, screen_x: float) -> Optional[float]:
    """
    Estimate k for inverse formula: focal = k / zoom
    
    From: screen_x = center_x + x_ndc * (k / zoom)
    Solve: k = (screen_x - center_x) * zoom / x_ndc
    """
    if abs(x_ndc) < 1e-6:
        return None
    k = (screen_x - SCREEN_WIDTH / 2) * zoom / x_ndc
    return k if k > 0 else None


def estimate_k_direct(zoom: int, x_ndc: float, screen_x: float) -> Optional[float]:
    """
    Estimate k for direct formula: focal = k * zoom
    
    From: screen_x = center_x + x_ndc * (k * zoom)
    Solve: k = (screen_x - center_x) / (x_ndc * zoom)
    """
    if abs(x_ndc * zoom) < 1e-6:
        return None
    k = (screen_x - SCREEN_WIDTH / 2) / (x_ndc * zoom)
    return k if k > 0 else None


def estimate_k_shifted(zoom: int, x_ndc: float, screen_x: float, offset: float = 0.0) -> Optional[float]:
    """
    Estimate k for shifted formula: focal = k / (zoom - offset)
    
    From: screen_x = center_x + x_ndc * (k / (zoom - offset))
    Solve: k = (screen_x - center_x) * (zoom - offset) / x_ndc
    """
    if abs(x_ndc) < 1e-6 or (zoom - offset) <= 0:
        return None
    k = (screen_x - SCREEN_WIDTH / 2) * (zoom - offset) / x_ndc
    return k if k > 0 else None


def analyze_sample_data(
    training_data: List[Dict[str, Any]],
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Analyze a sample of training data to estimate initial k values.
    
    Args:
        training_data: List of training data points
        sample_size: Number of points to sample (default: 100)
    
    Returns:
        Dictionary with estimated k values for each formula
    """
    # Sample data points
    if len(training_data) > sample_size:
        indices = np.random.choice(len(training_data), sample_size, replace=False)
        sample = [training_data[i] for i in indices]
    else:
        sample = training_data
    
    print(f"[ANALYSIS] Analyzing {len(sample)} sample points...")
    
    k_inverse_estimates = []
    k_direct_estimates = []
    k_shifted_estimates = []
    
    for point in sample:
        player_world = point["player"]
        obj_world = {"x": point["object"]["x"], "y": point["object"]["y"], "p": 0}
        camera = point["camera"]
        observed = point["observed_screen"]
        
        # Transform to camera space
        x_cam, y_cam, z_cam = world_to_camera_space(
            obj_world, player_world, camera["yaw"], camera["pitch"]
        )
        
        if z_cam <= 0:
            continue  # Object behind camera
        
        # Calculate normalized device coordinates
        x_ndc = x_cam / z_cam
        y_ndc = y_cam / z_cam
        
        # Estimate k for each formula
        # Use X coordinate for estimation
        k_inv = estimate_k_inverse(camera["zoom"], x_ndc, observed["x"])
        if k_inv:
            k_inverse_estimates.append(k_inv)
        
        k_dir = estimate_k_direct(camera["zoom"], x_ndc, observed["x"])
        if k_dir:
            k_direct_estimates.append(k_dir)
        
        # For shifted, try a few offset values
        for offset in [-100, 0, 100]:
            k_shift = estimate_k_shifted(camera["zoom"], x_ndc, observed["x"], offset)
            if k_shift:
                k_shifted_estimates.append(k_shift)
    
    results = {}
    
    if k_inverse_estimates:
        # Use median for robustness (less sensitive to outliers)
        k_inv_median = np.median(k_inverse_estimates)
        k_inv_mean = np.mean(k_inverse_estimates)
        results["inverse"] = {
            "median": float(k_inv_median),
            "mean": float(k_inv_mean),
            "std": float(np.std(k_inverse_estimates)),
            "min": float(np.min(k_inverse_estimates)),
            "max": float(np.max(k_inverse_estimates)),
            "count": len(k_inverse_estimates)
        }
        print(f"[ANALYSIS] Inverse formula: median k = {k_inv_median:.2f}, mean = {k_inv_mean:.2f}")
    
    if k_direct_estimates:
        k_dir_median = np.median(k_direct_estimates)
        k_dir_mean = np.mean(k_direct_estimates)
        results["direct"] = {
            "median": float(k_dir_median),
            "mean": float(k_dir_mean),
            "std": float(np.std(k_direct_estimates)),
            "min": float(np.min(k_direct_estimates)),
            "max": float(np.max(k_direct_estimates)),
            "count": len(k_direct_estimates)
        }
        print(f"[ANALYSIS] Direct formula: median k = {k_dir_median:.6f}, mean = {k_dir_mean:.6f}")
    
    if k_shifted_estimates:
        k_shift_median = np.median(k_shifted_estimates)
        k_shift_mean = np.mean(k_shifted_estimates)
        results["shifted"] = {
            "median": float(k_shift_median),
            "mean": float(k_shift_mean),
            "std": float(np.std(k_shifted_estimates)),
            "min": float(np.min(k_shifted_estimates)),
            "max": float(np.max(k_shifted_estimates)),
            "count": len(k_shifted_estimates)
        }
        print(f"[ANALYSIS] Shifted formula: median k = {k_shift_median:.2f}, mean = {k_shift_mean:.2f}")
    
    return results


if __name__ == "__main__":
    # Test with sample data
    import sys
    from fit_camera_model import load_calibration_files, prepare_training_data
    
    if len(sys.argv) < 2:
        print("Usage: python estimate_initial_k.py <calibration_file1.jsonl> [calibration_file2.jsonl ...]")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    data = load_calibration_files(file_paths)
    training_data = prepare_training_data(data)
    
    if not training_data:
        print("[ERROR] No training data available")
        sys.exit(1)
    
    print(f"[INFO] Analyzing {len(training_data)} training points")
    results = analyze_sample_data(training_data, sample_size=200)
    
    print("\n" + "=" * 60)
    print("RECOMMENDED INITIAL K VALUES")
    print("=" * 60)
    for formula, stats in results.items():
        print(f"\n{formula.upper()}:")
        print(f"  Recommended (median): {stats['median']:.6f}")
        print(f"  Alternative (mean):   {stats['mean']:.6f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  StdDev: {stats['std']:.2f}")
        print(f"  Valid estimates: {stats['count']}")

