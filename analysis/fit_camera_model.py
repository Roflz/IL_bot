#!/usr/bin/env python3
"""
Fit Camera Projection Model Parameters

This script:
1. Loads calibration data from JSONL files
2. Fits model parameters using scipy.optimize.least_squares
3. Saves fitted parameters to JSON file
4. Validates the model on a test set

Based on CAMERA_PROJECTION_IMPLEMENTATION_PLAN.md
"""

import json
import os
import sys
import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.camera_projection import (
    predict_screen_position,
    calculate_reprojection_error,
    world_to_camera_space,
    focal_length_from_zoom,
    YAW_SCALE,
    PITCH_SCALE,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_CENTER_X,
    SCREEN_CENTER_Y
)

try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not available. Install with: pip install scipy")


def load_calibration_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load calibration data from specified JSONL files.
    
    Args:
        file_paths: List of file paths to load
    
    Returns:
        List of data entries (one per line)
    """
    all_data = []
    
    for filepath in file_paths:
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found: {filepath}")
            continue
        
        print(f"[INFO] Loading: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        all_data.append(entry)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Failed to parse line in {filepath}: {e}")
            print(f"[INFO] Loaded {line_count} entries from {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    
    print(f"[INFO] Loaded {len(all_data)} total data entries")
    return all_data


def prepare_training_data(data: List[Dict[str, Any]], filter_screen_coords: bool = True) -> List[Dict[str, Any]]:
    """
    Prepare data for training by extracting valid data points.
    
    Filters out:
    - Entries with missing camera/player/object data
    - Objects with null screen positions (not visible)
    - Objects with screen coordinates outside ±1000 pixels of canvas bounds (if filter_screen_coords=True)
    - Objects that are likely behind camera (will be caught by z_cam <= 0 check)
    
    Args:
        data: Raw calibration data
        filter_screen_coords: If True, filter objects outside ±1000 pixels of canvas bounds (default True)
    
    Returns:
        List of prepared data points
    """
    training_points = []
    prep_start = time.time()
    last_progress = time.time()
    
    # Screen coordinate filter bounds (±1000 pixels from canvas)
    FILTER_MIN_X = -1000
    FILTER_MAX_X = SCREEN_WIDTH + 1000
    FILTER_MIN_Y = -1000
    FILTER_MAX_Y = SCREEN_HEIGHT + 1000
    
    total_entries = len(data)
    filtered_out_count = 0
    print(f"[PREP] Processing {total_entries} entries...")
    if filter_screen_coords:
        print(f"[PREP] Screen coordinate filter: ±1000 pixels from canvas bounds ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"[PREP] Filter range: X=[{FILTER_MIN_X}, {FILTER_MAX_X}], Y=[{FILTER_MIN_Y}, {FILTER_MAX_Y}]")
    
    for idx, entry in enumerate(data):
        player = entry.get("player", {})
        camera = entry.get("camera", {})
        objects = entry.get("objects", [])
        
        player_x = player.get("x")
        player_y = player.get("y")
        yaw = camera.get("yaw")
        pitch = camera.get("pitch")
        zoom = camera.get("zoom")
        
        if None in [player_x, player_y, yaw, pitch, zoom]:
            continue
        
        for obj in objects:
            world = obj.get("world", {})
            screen = obj.get("screen")
            
            if screen is None:  # Object not visible
                continue
            
            obj_x = world.get("x")
            obj_y = world.get("y")
            obj_plane = world.get("p")  # Plane field from world coordinates
            screen_x = screen.get("x")
            screen_y = screen.get("y")
            model_height = obj.get("modelHeight")  # Model height if available
            
            if None in [obj_x, obj_y, screen_x, screen_y]:
                continue
            
            # Apply screen coordinate filter if enabled
            if filter_screen_coords:
                x_in_range = FILTER_MIN_X <= screen_x <= FILTER_MAX_X
                y_in_range = FILTER_MIN_Y <= screen_y <= FILTER_MAX_Y
                if not (x_in_range and y_in_range):
                    filtered_out_count += 1
                    continue
            
            # Create training point with plane and modelHeight if available
            # Extract camera position (x, y, z) if available
            camera_dict = {"yaw": yaw, "pitch": pitch, "zoom": zoom}
            camera_x = camera.get("x")
            camera_y = camera.get("y")
            camera_z = camera.get("z")
            if camera_x is not None:
                camera_dict["x"] = camera_x
            if camera_y is not None:
                camera_dict["y"] = camera_y
            if camera_z is not None:
                camera_dict["z"] = camera_z
            
            training_point = {
                "player": {"x": player_x, "y": player_y},
                "object": {"x": obj_x, "y": obj_y},
                "camera": camera_dict,
                "observed_screen": {"x": screen_x, "y": screen_y}
            }
            
            # Add plane if available
            if obj_plane is not None:
                training_point["object"]["p"] = obj_plane
            
            # Add model height if available
            if model_height is not None:
                training_point["object"]["modelHeight"] = model_height
            
            training_points.append(training_point)
        
        # Progress update every 2 seconds or every 10% of entries
        current_time = time.time()
        if (current_time - last_progress >= 2.0) or ((idx + 1) % max(1, total_entries // 10) == 0):
            progress_pct = 100.0 * (idx + 1) / total_entries
            print(f"[PREP] Progress: {idx + 1}/{total_entries} entries ({progress_pct:.1f}%), {len(training_points)} training points")
            last_progress = current_time
    
    prep_elapsed = time.time() - prep_start
    print(f"[INFO] Prepared {len(training_points)} valid training points in {prep_elapsed:.1f}s")
    if filter_screen_coords and filtered_out_count > 0:
        print(f"[INFO] Filtered out {filtered_out_count} objects with screen coordinates outside ±1000 pixel range")
    return training_points


def debug_prediction_pipeline(
    point: Dict[str, Any],
    focal_formula: str,
    focal_k: float,
    focal_offset: float
):
    """
    Debug function to print all intermediate values in the prediction pipeline.
    """
    import math
    
    print("\n" + "="*80)
    print("DEBUG: Prediction Pipeline Values")
    print("="*80)
    
    # Extract input values
    obj_plane = point["object"].get("p", 0)
    obj_world = {"x": point["object"]["x"], "y": point["object"]["y"], "p": obj_plane}
    camera = point["camera"]
    observed = point["observed_screen"]
    model_height = point["object"].get("modelHeight")
    
    # Extract camera position (x, y, z)
    camera_world = {
        "x": camera.get("x", 0.0),
        "y": camera.get("y", 0.0),
        "z": camera.get("z", 0.0)
    }
    
    print("\n[INPUT VALUES FROM DATA]")
    print(f"  Camera world: x={camera_world.get('x')}, y={camera_world.get('y')}, z={camera_world.get('z')}")
    print(f"  Object world: x={obj_world.get('x')}, y={obj_world.get('y')}, p={obj_plane}")
    print(f"  Model height: {model_height if model_height is not None else 'None (defaulting to 0)'}")
    print(f"  Camera yaw: {camera.get('yaw')} (units, 0-2047)")
    print(f"  Camera pitch: {camera.get('pitch')} (units, 0-512)")
    print(f"  Camera zoom: {camera.get('zoom')} (units, 551-4409)")
    print(f"  Observed screen: x={observed.get('x')}, y={observed.get('y')}")
    print(f"  Focal formula: {focal_formula}, k={focal_k}, offset={focal_offset}")
    
    # Step 1: Translate relative to camera position
    obj_x = obj_world.get("x", 0.0)
    obj_y = obj_world.get("y", 0.0)
    camera_x = camera_world.get("x", 0.0)
    camera_y = camera_world.get("y", 0.0)
    camera_z = camera_world.get("z", 0.0)
    dx = obj_x - camera_x
    dy = obj_y - camera_y
    
    print("\n[STEP 1: TRANSLATE RELATIVE TO CAMERA POSITION]")
    print(f"  dx = obj_x - camera_x = {obj_x} - {camera_x} = {dx}")
    print(f"  dy = obj_y - camera_y = {obj_y} - {camera_y} = {dy}")
    
    # Calculate object Z and then relative to camera
    PLANE_HEIGHT_OFFSET = 640.0
    obj_z = obj_plane * PLANE_HEIGHT_OFFSET
    if model_height is not None:
        obj_z += model_height
    dz = obj_z - camera_z
    
    print("\n[STEP 2: CALCULATE Z OFFSET]")
    print(f"  obj_plane = {obj_plane}")
    print(f"  obj_z_base = obj_plane * 640 = {obj_plane} * 640 = {obj_plane * PLANE_HEIGHT_OFFSET}")
    print(f"  model_height = {model_height if model_height is not None else 0.0}")
    print(f"  obj_z = obj_z_base + model_height = {obj_plane * PLANE_HEIGHT_OFFSET} + {model_height if model_height is not None else 0.0} = {obj_z}")
    print(f"  camera_z = {camera_z}")
    print(f"  dz = obj_z - camera_z = {obj_z} - {camera_z} = {dz}")
    
    # Convert angles to radians
    yaw = camera.get("yaw")
    pitch = camera.get("pitch")
    yaw_rad = yaw * YAW_SCALE
    pitch_rad = pitch * PITCH_SCALE
    
    print("\n[STEP 3: CONVERT ANGLES TO RADIANS]")
    print(f"  yaw = {yaw} units")
    print(f"  yaw_rad = yaw * (2π/2048) = {yaw} * {YAW_SCALE:.6f} = {yaw_rad:.6f} rad ({math.degrees(yaw_rad):.2f}°)")
    print(f"  pitch = {pitch} units")
    print(f"  pitch_rad = pitch * (π/1024) = {pitch} * {PITCH_SCALE:.6f} = {pitch_rad:.6f} rad ({math.degrees(pitch_rad):.2f}°)")
    
    # Yaw rotation (around Z axis - vertical axis in OSRS)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x_rotated = dx * cos_yaw - dy * sin_yaw
    y_rotated = dx * sin_yaw + dy * cos_yaw
    z_rotated = dz  # Z doesn't change in yaw rotation (it's the rotation axis)
    
    print("\n[STEP 4: ROTATE BY YAW (Z-axis - vertical)]")
    print(f"  cos(yaw) = {cos_yaw:.6f}, sin(yaw) = {sin_yaw:.6f}")
    print(f"  x_rotated = dx*cos(yaw) - dy*sin(yaw) = {dx}*{cos_yaw:.6f} - {dy}*{sin_yaw:.6f} = {x_rotated:.6f}")
    print(f"  y_rotated = dx*sin(yaw) + dy*cos(yaw) = {dx}*{sin_yaw:.6f} + {dy}*{cos_yaw:.6f} = {y_rotated:.6f}")
    print(f"  z_rotated = dz = {z_rotated:.6f}")
    
    # Pitch rotation
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    x_cam = x_rotated
    y_cam = y_rotated * cos_pitch - z_rotated * sin_pitch
    z_cam = y_rotated * sin_pitch + z_rotated * cos_pitch
    
    print("\n[STEP 5: ROTATE BY PITCH (X-axis)]")
    print(f"  cos(pitch) = {cos_pitch:.6f}, sin(pitch) = {sin_pitch:.6f}")
    print(f"  x_cam = x_rotated = {x_cam:.6f}")
    print(f"  y_cam = y_rotated*cos(pitch) - z_rotated*sin(pitch) = {y_rotated}*{cos_pitch:.6f} - {z_rotated}*{sin_pitch:.6f} = {y_cam:.6f}")
    print(f"  z_cam = y_rotated*sin(pitch) + z_rotated*cos(pitch) = {y_rotated}*{sin_pitch:.6f} + {z_rotated}*{cos_pitch:.6f} = {z_cam:.6f}")
    
    # Focal length calculation
    zoom = camera.get("zoom")
    print("\n[STEP 6: CALCULATE FOCAL LENGTH FROM ZOOM]")
    print(f"  zoom = {zoom}")
    
    if focal_formula == "inverse":
        focal_length = focal_k / zoom if zoom > 0 else 0.0
        print(f"  formula: f = k / zoom")
        print(f"  focal_length = {focal_k} / {zoom} = {focal_length:.6f}")
    elif focal_formula == "direct":
        focal_length = focal_k * zoom
        print(f"  formula: f = k * zoom")
        print(f"  focal_length = {focal_k} * {zoom} = {focal_length:.6f}")
    elif focal_formula == "shifted":
        zoom_shifted = zoom - focal_offset
        focal_length = focal_k / zoom_shifted if zoom_shifted > 0 else 0.0
        print(f"  formula: f = k / (zoom - offset)")
        print(f"  zoom_shifted = {zoom} - {focal_offset} = {zoom_shifted}")
        print(f"  focal_length = {focal_k} / {zoom_shifted} = {focal_length:.6f}")
    elif focal_formula == "fov_based":
        fov_rad = (focal_k / zoom) if zoom > 0 else 0.0
        focal_length = (SCREEN_HEIGHT / 2) / math.tan(fov_rad / 2) if fov_rad > 0 and fov_rad < math.pi else 0.0
        print(f"  formula: fov = k/zoom, then f = (height/2) / tan(fov/2)")
        print(f"  fov_rad = {focal_k} / {zoom} = {fov_rad:.6f} rad ({math.degrees(fov_rad):.2f}°)")
        print(f"  focal_length = ({SCREEN_HEIGHT}/2) / tan({fov_rad:.6f}/2) = {SCREEN_HEIGHT/2} / {math.tan(fov_rad/2):.6f} = {focal_length:.6f}")
    
    # Perspective divide
    x_ndc = x_cam / z_cam
    y_ndc = y_cam / z_cam
    
    print("\n[STEP 7: PERSPECTIVE DIVIDE]")
    print(f"  x_ndc = x_cam / z_cam = {x_cam:.6f} / {z_cam:.6f} = {x_ndc:.6f}")
    print(f"  y_ndc = y_cam / z_cam = {y_cam:.6f} / {z_cam:.6f} = {y_ndc:.6f}")
    
    # Screen projection
    screen_x = SCREEN_WIDTH / 2 + x_ndc * focal_length
    screen_y = SCREEN_HEIGHT / 2 - y_ndc * focal_length
    
    print("\n[STEP 8: PROJECT TO SCREEN]")
    print(f"  screen_center_x = {SCREEN_WIDTH} / 2 = {SCREEN_WIDTH/2}")
    print(f"  screen_center_y = {SCREEN_HEIGHT} / 2 = {SCREEN_HEIGHT/2}")
    print(f"  screen_x = center_x + x_ndc * f = {SCREEN_WIDTH/2} + {x_ndc:.6f} * {focal_length:.6f} = {screen_x:.2f}")
    print(f"  screen_y = center_y - y_ndc * f = {SCREEN_HEIGHT/2} - {y_ndc:.6f} * {focal_length:.6f} = {screen_y:.2f}")
    
    # Compare with observed
    error_x = screen_x - observed.get("x")
    error_y = screen_y - observed.get("y")
    error_dist = math.sqrt(error_x**2 + error_y**2)
    
    print("\n[COMPARISON WITH OBSERVED]")
    print(f"  Predicted: x={screen_x:.2f}, y={screen_y:.2f}")
    print(f"  Observed:  x={observed.get('x')}, y={observed.get('y')}")
    print(f"  Error:     x={error_x:.2f}, y={error_y:.2f}, distance={error_dist:.2f} pixels")
    print("="*80 + "\n")


def reprojection_error_residuals(
    params: np.ndarray,
    training_data: List[Dict[str, Any]],
    focal_formula: str = "inverse",
    debug: bool = False,
    debug_count: int = 5
) -> np.ndarray:
    """
    Calculate reprojection error residuals for least-squares optimization.
    
    Args:
        params: Parameter array [focal_k, focal_offset] (offset only used for shifted formula)
        training_data: List of training data points
        focal_formula: Formula type ("inverse", "direct", "shifted", "fov_based")
    
    Returns:
        Array of residuals (predicted - observed) for x and y, flattened
    """
    focal_k = params[0]
    focal_offset = params[1] if len(params) > 1 else 0.0
    
    residuals = []
    debug_printed = 0
    huge_error_points = []  # Track points with huge errors for debugging
    
    for idx, point in enumerate(training_data):
        # Debug output for first N points
        if debug and debug_printed < debug_count:
            debug_prediction_pipeline(point, focal_formula, focal_k, focal_offset)
            debug_printed += 1
        # Use plane from object if available, otherwise default to 0
        obj_plane = point["object"].get("p", 0)
        obj_world = {"x": point["object"]["x"], "y": point["object"]["y"], "p": obj_plane}
        # Add modelHeight if available
        model_height = point["object"].get("modelHeight")
        if model_height is not None:
            obj_world["modelHeight"] = model_height
        
        camera = point["camera"]
        # Extract camera position (x, y, z) - use camera position, not player position!
        camera_world = {
            "x": camera.get("x", 0.0),
            "y": camera.get("y", 0.0),
            "z": camera.get("z", 0.0)
        }
        observed = point["observed_screen"]
        
        # Predict screen position
        predicted = predict_screen_position(
            obj_world,
            camera_world,
            camera["yaw"],
            camera["pitch"],
            camera["zoom"],
            focal_formula,
            focal_k,
            focal_offset,
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )
        
        if predicted is None:
            # This shouldn't happen, but track it
            huge_error_points.append({
                "idx": idx,
                "reason": "predicted is None",
                "point": point
            })
            continue
        
        pred_x = predicted["x"]
        pred_y = predicted["y"]
        
        # Calculate residuals
        error_x = pred_x - observed["x"]
        error_y = pred_y - observed["y"]
        error_distance = math.sqrt(error_x**2 + error_y**2)
        
        # Track points with huge errors (> 10000 pixels)
        if error_distance > 10000:
            # Calculate intermediate values for debugging
            x_cam, y_cam, z_cam = world_to_camera_space(obj_world, camera_world, camera["yaw"], camera["pitch"])
            focal_length = focal_length_from_zoom(camera["zoom"], focal_formula, focal_k, focal_offset)
            x_ndc = x_cam / z_cam if z_cam != 0 else float('inf')
            y_ndc = y_cam / z_cam if z_cam != 0 else float('inf')
            
            huge_error_points.append({
                "idx": idx,
                "error_distance": error_distance,
                "error_x": error_x,
                "error_y": error_y,
                "observed": observed,
                "predicted": {"x": pred_x, "y": pred_y},
                "obj_world": obj_world,
                "camera_world": camera_world,
                "camera": {"yaw": camera["yaw"], "pitch": camera["pitch"], "zoom": camera["zoom"]},
                "x_cam": x_cam,
                "y_cam": y_cam,
                "z_cam": z_cam,
                "focal_length": focal_length,
                "x_ndc": x_ndc,
                "y_ndc": y_ndc,
                "focal_k": focal_k,
                "focal_offset": focal_offset
            })
        
        residuals.append(error_x)
        residuals.append(error_y)
    
    # Print huge error points if any (limit to first 10 to avoid spam)
    if huge_error_points:
        print(f"\n[WARNING] Found {len(huge_error_points)} points with errors > 10000 pixels:")
        for i, err_point in enumerate(huge_error_points[:10]):
            print(f"\n  Point {i+1} (idx={err_point['idx']}):")
            print(f"    Error: {err_point['error_distance']:.1f} pixels (x={err_point['error_x']:.1f}, y={err_point['error_y']:.1f})")
            print(f"    Observed: {err_point['observed']}")
            print(f"    Predicted: {err_point['predicted']}")
            print(f"    Object world: {err_point['obj_world']}")
            print(f"    Camera world: {err_point['camera_world']}")
            print(f"    Camera state: yaw={err_point['camera']['yaw']}, pitch={err_point['camera']['pitch']}, zoom={err_point['camera']['zoom']}")
            print(f"    Camera space: x={err_point['x_cam']:.2f}, y={err_point['y_cam']:.2f}, z={err_point['z_cam']:.2f}")
            print(f"    Focal length: {err_point['focal_length']:.2f} (k={err_point['focal_k']:.2f}, offset={err_point['focal_offset']:.2f})")
            print(f"    NDC: x={err_point['x_ndc']:.2f}, y={err_point['y_ndc']:.2f}")
        if len(huge_error_points) > 10:
            print(f"    ... and {len(huge_error_points) - 10} more")
    
    return np.array(residuals)
    
    return np.array(residuals)


def fit_model_parameters(
    training_data: List[Dict[str, Any]],
    focal_formula: str = "inverse",
    initial_k: float = 100000.0,
    initial_offset: float = 0.0,
    test_multiple_k: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Fit model parameters using least-squares optimization.
    
    Args:
        training_data: Prepared training data points
        focal_formula: Formula type to fit
        initial_k: Initial guess for focal_k
        initial_offset: Initial guess for focal_offset (only for shifted formula)
    
    Returns:
        Dictionary with fitted parameters and optimization results
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available"}
    
    print(f"\n[FITTING] Fitting parameters for formula: {focal_formula}")
    print(f"[FITTING] Using {len(training_data)} training points")
    
    # Test multiple initial k values if requested
    if test_multiple_k:
        print(f"[FITTING] Testing multiple initial k values...")
        k_values_to_test = [
            1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 
            200000.0, 500000.0, 1000000.0, 2000000.0
        ]
        best_result = None
        best_cost = float('inf')
        best_k = initial_k
        
        for test_k in k_values_to_test:
            print(f"\n[FITTING] Testing initial k = {test_k:.0f}...")
            
            # Calculate initial cost BEFORE optimization (scipy uses 0.5 * sum(residuals^2))
            initial_residuals = reprojection_error_residuals([test_k], training_data[:10000], focal_formula, debug=False)
            initial_cost = 0.5 * np.sum(initial_residuals**2)
            print(f"  Initial cost (before optimization): {initial_cost:.2e}")
            
            if focal_formula == "shifted":
                x0_test = np.array([test_k, initial_offset])
                bounds = ([0.0, -1000.0], [1e6, 1000.0])
            else:
                x0_test = np.array([test_k])
                bounds = ([0.0], [1e6])
            
            # Quick test with limited evaluations
            test_state = {"last_k": test_k, "count": 0}
            
            def test_residual_with_progress(params):
                """Track k changes during quick test."""
                test_state["count"] += 1
                current_k = params[0]
                if test_state["count"] <= 3 or abs(current_k - test_state["last_k"]) > 0.01 * abs(test_state["last_k"]):
                    print(f"    Eval {test_state['count']}: k={current_k:.2f} (from {test_k:.0f})")
                    test_state["last_k"] = current_k
                return reprojection_error_residuals(params, training_data[:10000], focal_formula, debug=False)
            
            # Quick test with limited evaluations
            try:
                result_test = least_squares(
                    test_residual_with_progress,
                    x0_test,
                    bounds=bounds,
                    method='trf',
                    verbose=0,
                    max_nfev=20  # Just a few evaluations to test
                )
                test_cost = result_test.cost
                final_test_k = result_test.x[0]
                k_change = final_test_k - test_k
                print(f"  Result: k optimized from {test_k:.0f} → {final_test_k:.2f} (Δ{k_change:+.2f}), cost={test_cost:.2e}")
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_k = test_k
                    print(f"  ✓ New best initial k: {best_k:.0f} (final k after optimization: {final_test_k:.2f})")
            except Exception as e:
                print(f"  ✗ Error with k={test_k:.0f}: {e}")
                continue
        
        print(f"\n[FITTING] Best initial k: {best_k:.0f} (cost: {best_cost:.2e})")
        initial_k = best_k
    
    # Initial parameter guess
    if focal_formula == "shifted":
        x0 = np.array([initial_k, initial_offset])
        bounds = ([0.0, -1000.0], [1e6, 1000.0])  # Reasonable bounds
    else:
        x0 = np.array([initial_k])
        bounds = ([0.0], [1e6])  # Only k parameter
    
    # Run optimization
    print(f"[FITTING] Starting optimization...")
    print(f"[FITTING] Initial guess: k={initial_k}, offset={initial_offset if focal_formula == 'shifted' else 'N/A'}")
    
    # Progress tracking - wrap the residual function
    fit_start_time = time.time()
    progress_state = {
        "last_time": time.time(), 
        "count": 0,
        "initial_k": initial_k,
        "initial_offset": initial_offset,
        "last_k": initial_k,
        "last_offset": initial_offset
    }
    
    def residual_with_progress(params):
        """Wrapper around reprojection_error_residuals that tracks progress."""
        progress_state["count"] += 1
        current_time = time.time()
        elapsed = current_time - fit_start_time
        
        # Extract current parameter values
        current_k = params[0]
        current_offset = params[1] if len(params) > 1 else 0.0
        
        # Only debug on first evaluation to avoid spam
        debug_this_eval = debug and progress_state["count"] == 1
        
        # Calculate residuals
        residuals = reprojection_error_residuals(params, training_data, focal_formula, debug=debug_this_eval)
        
        # Print progress every 5 seconds or every 10 evaluations, or if parameters changed significantly
        k_changed = abs(current_k - progress_state["last_k"]) > 0.01 * abs(progress_state["last_k"])
        offset_changed = (focal_formula == "shifted" and 
                         abs(current_offset - progress_state["last_offset"]) > 0.01 * abs(progress_state["last_offset"]))
        
        should_print = ((current_time - progress_state["last_time"] >= 5.0) or 
                        (progress_state["count"] % 10 == 0) or
                        (progress_state["count"] <= 5) or  # Always print first 5
                        k_changed or offset_changed)
        
        if should_print:
            # Calculate cost using scipy's convention: 0.5 * sum(residuals^2)
            current_cost = 0.5 * np.sum(residuals**2) if len(residuals) > 0 else 0.0
            
            # Calculate diagnostic statistics
            residual_array = np.array(residuals)
            abs_residuals = np.abs(residual_array)
            
            # Count errors by magnitude
            small_errors = np.sum(abs_residuals < 100)
            medium_errors = np.sum((abs_residuals >= 100) & (abs_residuals < 500))
            large_errors = np.sum((abs_residuals >= 500) & (abs_residuals < 1000))
            very_large_errors = np.sum((abs_residuals >= 1000) & (abs_residuals <= 10000))
            extreme_errors = np.sum(abs_residuals > 10000)
            
            # Calculate stats on reasonable residuals (<= 10000 pixels)
            reasonable_residuals = abs_residuals[abs_residuals <= 10000]
            
            avg_time_per_eval = elapsed / progress_state["count"] if progress_state["count"] > 0 else 0
            remaining_evals = max(0, 1000 - progress_state["count"])
            est_remaining = avg_time_per_eval * remaining_evals
            
            # Calculate parameter changes
            k_change = current_k - progress_state["initial_k"]
            k_change_pct = 100.0 * k_change / progress_state["initial_k"] if progress_state["initial_k"] != 0 else 0.0
            
            # Build diagnostic string
            diag_parts = []
            if len(reasonable_residuals) > 0:
                diag_parts.append(f"errors: mean={np.mean(reasonable_residuals):.1f}, median={np.median(reasonable_residuals):.1f}, max={np.max(reasonable_residuals):.1f}")
            
            # Show error distribution
            error_dist = []
            if small_errors > 0:
                error_dist.append(f"<100px: {small_errors}")
            if medium_errors > 0:
                error_dist.append(f"100-500px: {medium_errors}")
            if large_errors > 0:
                error_dist.append(f"500-1kpx: {large_errors}")
            if very_large_errors > 0:
                error_dist.append(f"1k-10kpx: {very_large_errors}")
            if extreme_errors > 0:
                extreme_max = np.max(abs_residuals) if len(abs_residuals) > 0 else 0
                error_dist.append(f">10kpx: {extreme_errors} (max={extreme_max:.1f})")
            
            if error_dist:
                diag_parts.append(f"distribution: {', '.join(error_dist)}")
            
            diag_str = ", " + ", ".join(diag_parts) if diag_parts else ""
            
            if focal_formula == "shifted":
                offset_change = current_offset - progress_state["initial_offset"]
                print(f"[FITTING] Eval {progress_state['count']}/1000: k={current_k:.2f} (Δ{k_change:+.2f}, {k_change_pct:+.2f}%), "
                      f"offset={current_offset:.2f} (Δ{offset_change:+.2f}), cost={current_cost:.2e}{diag_str}, "
                      f"elapsed={elapsed:.1f}s, est. remaining={est_remaining:.1f}s")
            else:
                print(f"[FITTING] Eval {progress_state['count']}/1000: k={current_k:.2f} (Δ{k_change:+.2f}, {k_change_pct:+.2f}%), "
                      f"cost={current_cost:.2e}{diag_str}, elapsed={elapsed:.1f}s, est. remaining={est_remaining:.1f}s")
            
            progress_state["last_time"] = current_time
            progress_state["last_k"] = current_k
            progress_state["last_offset"] = current_offset
        
        return residuals
    
    try:
        result = least_squares(
            residual_with_progress,
            x0,
            bounds=bounds,
            method='trf',  # Trust Region Reflective (supports bounds)
            verbose=0,  # We'll use our own progress reporting
            max_nfev=1000  # Max function evaluations
        )
        
        fit_elapsed = time.time() - fit_start_time
        print(f"[FITTING] Optimization completed in {fit_elapsed:.1f}s ({progress_state['count']} evaluations)")
        
        if result.success:
            fitted_k = result.x[0]
            fitted_offset = result.x[1] if len(result.x) > 1 else 0.0
            
            # Print parameter change summary
            k_change = fitted_k - progress_state["initial_k"]
            k_change_pct = 100.0 * k_change / progress_state["initial_k"] if progress_state["initial_k"] != 0 else 0.0
            print(f"\n[FITTING] Parameter Optimization Summary:")
            print(f"  Initial k: {progress_state['initial_k']:.2f}")
            print(f"  Final k:   {fitted_k:.2f}")
            print(f"  Change:    {k_change:+.2f} ({k_change_pct:+.2f}%)")
            if focal_formula == "shifted":
                offset_change = fitted_offset - progress_state["initial_offset"]
                print(f"  Initial offset: {progress_state['initial_offset']:.2f}")
                print(f"  Final offset:   {fitted_offset:.2f}")
                print(f"  Change:         {offset_change:+.2f}")
            
            # Calculate final error statistics
            final_residuals = reprojection_error_residuals(
                result.x, training_data, focal_formula, debug=False
            )
            
            # Calculate error metrics
            error_distances = []
            for i in range(0, len(final_residuals), 2):
                error_x = final_residuals[i]
                error_y = final_residuals[i + 1]
                distance = math.sqrt(error_x**2 + error_y**2)
                error_distances.append(distance)
            
            return {
                "success": True,
                "formula": focal_formula,
                "focal_k": float(fitted_k),
                "focal_offset": float(fitted_offset),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "num_evaluations": int(result.nfev),
                "error_stats": {
                    "mean_error": float(np.mean(error_distances)),
                    "median_error": float(np.median(error_distances)),
                    "std_error": float(np.std(error_distances)),
                    "max_error": float(np.max(error_distances)),
                    "p95_error": float(np.percentile(error_distances, 95))
                }
            }
        else:
            return {
                "success": False,
                "error": result.message,
                "cost": float(result.cost) if hasattr(result, 'cost') else None
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def test_multiple_formulas(training_data: List[Dict[str, Any]], test_multiple_k: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Test multiple focal length formulas and return the best one.
    
    Args:
        training_data: Prepared training data points
        test_multiple_k: If True, test multiple initial k values for each formula
        debug: If True, print detailed debug output for first few predictions
    
    Returns:
        Dictionary with results for each formula
    """
    formulas_to_test = ["inverse", "direct", "shifted", "fov_based"]
    results = {}
    
    for formula in formulas_to_test:
        print(f"\n{'='*60}")
        print(f"Testing formula: {formula}")
        print(f"{'='*60}")
        
        result = fit_model_parameters(
            training_data,
            focal_formula=formula,
            initial_k=100000.0 if formula != "fov_based" else 1.0,
            initial_offset=0.0,
            test_multiple_k=test_multiple_k,
            debug=debug
        )
        
        results[formula] = result
    
    # Find best formula (lowest cost/error)
    best_formula = None
    best_cost = float('inf')
    
    for formula, result in results.items():
        if result.get("success") and result.get("cost", float('inf')) < best_cost:
            best_cost = result["cost"]
            best_formula = formula
    
    return {
        "results": results,
        "best_formula": best_formula,
        "best_cost": best_cost
    }


def save_fitted_parameters(
    params: Dict[str, Any],
    output_file: str = "camera_model_params.json"
):
    """
    Save fitted parameters to JSON file.
    
    Args:
        params: Dictionary with fitted parameters
        output_file: Output file path
    """
    # Prepare parameters dict for saving
    params_to_save = {
        "yaw_scale": YAW_SCALE,
        "pitch_scale": PITCH_SCALE,
        "pitch_offset": 0.0,  # Confirmed
        "focal_formula": params.get("formula", "inverse"),
        "focal_k": params.get("focal_k", 100000.0),
        "focal_offset": params.get("focal_offset", 0.0),
        "screen_width": SCREEN_WIDTH,
        "screen_height": SCREEN_HEIGHT,
        "screen_center_x": SCREEN_CENTER_X,
        "screen_center_y": SCREEN_CENTER_Y,
        "aspect_ratio": SCREEN_WIDTH / SCREEN_HEIGHT,
        "fitted_date": params.get("fitted_date", ""),
        "error_stats": params.get("error_stats", {})
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(params_to_save, f, indent=2)
        print(f"\n[INFO] Saved fitted parameters to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save parameters: {e}")


def main():
    """Main fitting function."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Fit Camera Projection Model')
    parser.add_argument(
        '--files',
        nargs='+',
        default=[
            'screen_projection_calibration_20251228_031854.jsonl'
        ],
        help='Calibration data files to use'
    )
    parser.add_argument(
        '--formula',
        choices=['inverse', 'direct', 'shifted', 'fov_based', 'all'],
        default='all',
        help='Focal length formula to fit (default: test all)'
    )
    parser.add_argument(
        '--output',
        default='camera_model_params.json',
        help='Output file for fitted parameters (default: camera_model_params.json)'
    )
    parser.add_argument(
        '--no-screen-filter',
        action='store_true',
        help='Disable screen coordinate filtering (include all objects, even far off-screen)'
    )
    parser.add_argument(
        '--test-initial-k',
        action='store_true',
        help='Test multiple initial k values to find best starting point'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print detailed debug output showing all intermediate calculation values for first 5 training points'
    )
    
    args = parser.parse_args()
    
    # Resolve file paths (relative to script directory or current directory)
    script_dir = Path(__file__).parent.parent
    file_paths = []
    for f in args.files:
        full_path = script_dir / f
        if full_path.exists():
            file_paths.append(str(full_path))
        elif os.path.exists(f):
            file_paths.append(f)
        else:
            print(f"[WARNING] File not found: {f}")
    
    if not file_paths:
        print("[ERROR] No valid calibration files found")
        return
    
    print("=" * 60)
    print("CAMERA PROJECTION MODEL FITTING")
    print("=" * 60)
    print(f"Files: {file_paths}")
    print(f"Formula: {args.formula}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Load data
    print("\n[STEP 1] Loading calibration data...")
    data = load_calibration_files(file_paths)
    
    if not data:
        print("[ERROR] No data loaded")
        return
    
    # Prepare training data
    print("\n[STEP 2] Preparing training data...")
    training_data = prepare_training_data(data, filter_screen_coords=not args.no_screen_filter)
    
    if not training_data:
        print("[ERROR] No valid training points found")
        return
    
    # Split into training/validation (80/20)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    print(f"[INFO] Training set: {len(train_data)} points")
    print(f"[INFO] Validation set: {len(val_data)} points")
    
    # Fit model
    print("\n[STEP 3] Fitting model parameters...")
    
    if args.formula == 'all':
        # Test all formulas
        results = test_multiple_formulas(train_data, test_multiple_k=args.test_initial_k, debug=args.debug)
        
        print("\n" + "=" * 60)
        print("FITTING RESULTS SUMMARY")
        print("=" * 60)
        
        for formula, result in results["results"].items():
            print(f"\n{formula.upper()}:")
            if result.get("success"):
                print(f"  Success: Yes")
                print(f"  Focal k: {result.get('focal_k', 'N/A'):.2f}")
                if result.get('focal_offset'):
                    print(f"  Focal offset: {result.get('focal_offset', 0):.2f}")
                print(f"  Cost: {result.get('cost', 'N/A'):.2f}")
                error_stats = result.get('error_stats', {})
                if error_stats:
                    print(f"  Mean error: {error_stats.get('mean_error', 0):.2f} pixels")
                    print(f"  Median error: {error_stats.get('median_error', 0):.2f} pixels")
                    print(f"  Max error: {error_stats.get('max_error', 0):.2f} pixels")
            else:
                print(f"  Success: No")
                print(f"  Error: {result.get('error', 'Unknown')}")
        
        print(f"\nBest formula: {results['best_formula']}")
        print(f"Best cost: {results['best_cost']:.2f}")
        
        # Save best result
        if results['best_formula']:
            best_result = results["results"][results['best_formula']]
            best_result["fitted_date"] = datetime.now().isoformat()
            save_fitted_parameters(best_result, args.output)
    else:
        # Fit single formula
        result = fit_model_parameters(
            train_data,
            focal_formula=args.formula,
            initial_k=100000.0,
            initial_offset=0.0,
            test_multiple_k=args.test_initial_k,
            debug=args.debug
        )
        
        if result.get("success"):
            result["fitted_date"] = datetime.now().isoformat()
            save_fitted_parameters(result, args.output)
            
            print("\n" + "=" * 60)
            print("FITTING RESULTS")
            print("=" * 60)
            print(f"Formula: {result['formula']}")
            print(f"Focal k: {result['focal_k']:.2f}")
            if result.get('focal_offset'):
                print(f"Focal offset: {result['focal_offset']:.2f}")
            print(f"Cost: {result['cost']:.2f}")
            error_stats = result.get('error_stats', {})
            if error_stats:
                print(f"\nError Statistics:")
                print(f"  Mean: {error_stats.get('mean_error', 0):.2f} pixels")
                print(f"  Median: {error_stats.get('median_error', 0):.2f} pixels")
                print(f"  StdDev: {error_stats.get('std_error', 0):.2f} pixels")
                print(f"  Max: {error_stats.get('max_error', 0):.2f} pixels")
                print(f"  95th percentile: {error_stats.get('p95_error', 0):.2f} pixels")
        else:
            print(f"\n[ERROR] Fitting failed: {result.get('error', 'Unknown error')}")
    
    print("\n[INFO] Fitting complete!")


if __name__ == "__main__":
    main()

