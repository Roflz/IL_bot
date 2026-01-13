#!/usr/bin/env python3
"""
Suggest and execute camera movement (Δyaw, Δpitch, optional Δzoom) using a *local Jacobian*
estimated from your logged calibration data (JSONL).

Automatically retrieves player position, camera state, and finds target object by name.
Uses live projection for current screen position.

JSONL format assumed (like your sample):
{
  "player": {"x":..., "y":...},
  "camera": {"yaw":..., "pitch":..., "zoom":...},
  "objects": [{"world":{"x":...,"y":...},"screen":{"x":...,"y":...}}, ...]
}

Usage:
    python suggest_camera_move_jacobian.py data.jsonl [--target TARGET_NAME] [--port PORT] [--use-zoom]
    
    --target TARGET     Object name to target (default: "Oak tree")
    --port PORT        IPC port number (default: 17002)
    --use-zoom         Solve for Δzoom too (3 vars instead of 2)
"""

import sys
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import after path setup
from helpers.ipc import IPCClient
from helpers.runtime_utils import set_ipc
from services.camera_integration import (
    _ensure_camera_thread,
    stop_camera_thread,
    _camera_movement_queue,
    YAW_MS_PER_UNIT,
    PITCH_MS_PER_UNIT
)
from services.camera_jacobian import (
    aim_camera_at_target_iterative,
    get_screen_position_preset
)
from actions import player

# IPC will be initialized in main()
ipc = None


def wait_for_camera_stable(measurement_type="yaw", max_wait=3.0, check_interval=0.1):
    """Wait for camera to stabilize by checking values multiple times."""
    last_value = None
    stable_count = 0
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        camera_data = ipc.get_camera()
        if not camera_data:
            time.sleep(check_interval)
            continue
        
        if measurement_type == "yaw":
            current_value = camera_data.get('yaw', 0)
        elif measurement_type == "pitch":
            current_value = camera_data.get('pitch', 0)
        elif measurement_type == "zoom":
            current_value = camera_data.get('scale', 0)
        else:
            return None
        
        if last_value is None:
            last_value = current_value
            stable_count = 0
        elif current_value == last_value:
            stable_count += 1
            if stable_count >= 2:  # Same value seen 3 times total
                return current_value
        else:
            last_value = current_value
            stable_count = 0
        
        time.sleep(check_interval)
    
    return last_value

# Default target object
DEFAULT_TARGET = "Oak tree"

# Target screen position (center X, 25% from top Y)
TARGET_Y_FRACTION = 0.25  # 25% from top

# Camera state bounds
YAW_MOD = 2048
PITCH_MIN = 128
PITCH_MAX = 383
ZOOM_MIN = 551
ZOOM_MAX = 4409

# Global pre-computed model (loaded from file if available)
_global_model = None
_global_model_file = None

# Jacobian cache (for performance optimization)
# Key: (dx, dy, yaw_bin, pitch_bin, zoom_bin) -> (bx, by, J)
# Bins are used to allow reuse for similar states
_jacobian_cache = {}
_jacobian_cache_size = 100  # Max cache entries
# Bin sizes for caching (larger = more reuse, less accuracy)
JACOBIAN_CACHE_YAW_BIN = 64   # ~32 bins across 2048 yaw range
JACOBIAN_CACHE_PITCH_BIN = 16  # ~16 bins across 256 pitch range
JACOBIAN_CACHE_ZOOM_BIN = 200  # ~20 bins across zoom range


def wrap_yaw_diff(a, b, mod=2048):
    """Smallest signed diff a-b in yaw units (wraparound)."""
    d = (a - b) % mod
    if d > mod / 2:
        d -= mod
    return d


def load_rows(jsonl_path):
    """
    Returns list of rows:
      (dx, dy, yaw, pitch, zoom, sx, sy)
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            player = rec.get("player") or {}
            cam = rec.get("camera") or {}
            px, py = player.get("x"), player.get("y")
            yaw, pitch, zoom = cam.get("yaw"), cam.get("pitch"), cam.get("zoom")

            if None in (px, py, yaw, pitch, zoom):
                continue

            for obj in rec.get("objects") or []:
                w = (obj or {}).get("world")
                s = (obj or {}).get("screen")
                if not w or not s:
                    continue
                if s.get("x") is None or s.get("y") is None:
                    continue
                ox, oy = w.get("x"), w.get("y")
                if ox is None or oy is None:
                    continue

                dx = int(ox) - int(px)
                dy = int(oy) - int(py)
                sx = float(s["x"])
                sy = float(s["y"])

                rows.append((dx, dy, int(yaw), int(pitch), int(zoom), sx, sy))
    return rows


def pick_neighbors(rows, dx0, dy0, yaw0, pitch0, zoom0, k=500, max_dxdy=3):
    """
    Find a local neighborhood:
    - prefer same (dx,dy) within a small window (max_dxdy)
    - then rank by distance in (dx,dy,pitch,zoom,yawwrap)
    """
    candidates = []
    for (dx, dy, yaw, pitch, zoom, sx, sy) in rows:
        if abs(dx - dx0) > max_dxdy or abs(dy - dy0) > max_dxdy:
            continue
        dyaw = wrap_yaw_diff(yaw, yaw0)
        dp = pitch - pitch0
        dz = zoom - zoom0
        # distance weights: tune if needed
        dist2 = (dx - dx0) ** 2 + (dy - dy0) ** 2 + (dyaw / 32.0) ** 2 + (dp / 8.0) ** 2 + (dz / 256.0) ** 2
        candidates.append((dist2, dx, dy, dyaw, pitch, zoom, sx, sy))

    candidates.sort(key=lambda t: t[0])
    return candidates[:k]


def get_jacobian_cache_key(dx0, dy0, yaw0, pitch0, zoom0):
    """Generate cache key for Jacobian lookup."""
    yaw_bin = int(yaw0 // JACOBIAN_CACHE_YAW_BIN)
    pitch_bin = int((pitch0 - PITCH_MIN) // JACOBIAN_CACHE_PITCH_BIN)
    zoom_bin = int((zoom0 - ZOOM_MIN) // JACOBIAN_CACHE_ZOOM_BIN)
    return (dx0, dy0, yaw_bin, pitch_bin, zoom_bin)


def fit_global_polynomial_model(rows, degree=2, use_zoom=False):
    """
    Fit a global polynomial model: sx = f(dx, dy, yaw, pitch, zoom), sy = g(...)
    
    This pre-computes a model from ALL calibration data, allowing instant Jacobian evaluation.
    
    Args:
        rows: List of (dx, dy, yaw, pitch, zoom, sx, sy) tuples
        degree: Polynomial degree (1=linear, 2=quadratic, etc.)
        use_zoom: Whether to include zoom in the model
    
    Returns:
        Dictionary with model coefficients and metadata, or None if fitting fails
    """
    if len(rows) < 100:
        print(f"[WARNING] Not enough data points ({len(rows)}) for global model")
        return None
    
    print(f"\n[INFO] Fitting global polynomial model (degree={degree}) from {len(rows)} data points...")
    
    # Prepare feature matrix X and targets Yx, Yy
    X_features = []
    Yx = []
    Yy = []
    
    for (dx, dy, yaw, pitch, zoom, sx, sy) in rows:
        # Normalize inputs for better conditioning
        # dx, dy: already in reasonable range
        # yaw: normalize to [0, 1] (yaw / YAW_MOD)
        # pitch: normalize to [0, 1] ((pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN))
        # zoom: normalize to [0, 1] ((zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN))
        
        yaw_norm = yaw / YAW_MOD
        pitch_norm = (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
        zoom_norm = (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) if use_zoom else 0.0
        
        # Build polynomial features
        features = []
        if degree >= 1:
            features.extend([1.0, dx, dy, yaw_norm, pitch_norm])
            if use_zoom:
                features.append(zoom_norm)
        if degree >= 2:
            # Quadratic terms
            features.extend([
                dx*dx, dy*dy, yaw_norm*yaw_norm, pitch_norm*pitch_norm,
                dx*dy, dx*yaw_norm, dx*pitch_norm, dy*yaw_norm, dy*pitch_norm, yaw_norm*pitch_norm
            ])
            if use_zoom:
                features.extend([
                    zoom_norm*zoom_norm,
                    dx*zoom_norm, dy*zoom_norm, yaw_norm*zoom_norm, pitch_norm*zoom_norm
                ])
        
        X_features.append(features)
        Yx.append(float(sx))
        Yy.append(float(sy))
    
    X = np.asarray(X_features, dtype=float)
    Yx = np.asarray(Yx, dtype=float)
    Yy = np.asarray(Yy, dtype=float)
    
    # Fit using least squares
    try:
        bx, *_ = np.linalg.lstsq(X, Yx, rcond=None)
        by, *_ = np.linalg.lstsq(X, Yy, rcond=None)
        
        # Compute R² for validation
        sx_pred = X @ bx
        sy_pred = X @ by
        r2_x = 1 - np.sum((Yx - sx_pred)**2) / np.sum((Yx - np.mean(Yx))**2)
        r2_y = 1 - np.sum((Yy - sy_pred)**2) / np.sum((Yy - np.mean(Yy))**2)
        
        print(f"  Model fitted: R² for sx={r2_x:.4f}, R² for sy={r2_y:.4f}")
        print(f"  Feature count: {len(bx)} coefficients")
        
        return {
            'bx': bx.tolist(),
            'by': by.tolist(),
            'degree': degree,
            'use_zoom': use_zoom,
            'r2_x': float(r2_x),
            'r2_y': float(r2_y),
            'feature_count': len(bx)
        }
    except Exception as e:
        print(f"[ERROR] Failed to fit global model: {e}")
        return None


def compute_jacobian_from_global_model(model, dx0, dy0, yaw0, pitch0, zoom0, use_zoom):
    """
    Compute Jacobian analytically from pre-fitted global polynomial model.
    
    Returns:
        (bx_coeffs, by_coeffs, J) where J is the Jacobian matrix
    """
    bx = np.array(model['bx'])
    by = np.array(model['by'])
    degree = model['degree']
    
    # Normalize inputs
    yaw_norm = yaw0 / YAW_MOD
    pitch_norm = (pitch0 - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
    zoom_norm = (zoom0 - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) if use_zoom else 0.0
    
    # Compute partial derivatives analytically
    # For sx = f(dx, dy, yaw, pitch, zoom), we need:
    # ∂sx/∂yaw, ∂sx/∂pitch, (∂sx/∂zoom)
    # ∂sy/∂yaw, ∂sy/∂pitch, (∂sy/∂zoom)
    
    # Build feature vector and compute derivatives
    # This is simplified - for a full implementation, we'd need to track which
    # coefficient corresponds to which feature and compute derivatives properly
    
    # For now, use numerical differentiation (small perturbation)
    eps = 1e-6
    
    # Compute base prediction
    def predict_screen(dx, dy, yaw, pitch, zoom):
        yaw_n = yaw / YAW_MOD
        pitch_n = (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
        zoom_n = (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) if use_zoom else 0.0
        
        features = []
        if degree >= 1:
            features.extend([1.0, dx, dy, yaw_n, pitch_n])
            if use_zoom:
                features.append(zoom_n)
        if degree >= 2:
            features.extend([
                dx*dx, dy*dy, yaw_n*yaw_n, pitch_n*pitch_n,
                dx*dy, dx*yaw_n, dx*pitch_n, dy*yaw_n, dy*pitch_n, yaw_n*pitch_n
            ])
            if use_zoom:
                features.extend([
                    zoom_n*zoom_n,
                    dx*zoom_n, dy*zoom_n, yaw_n*zoom_n, pitch_n*zoom_n
                ])
        
        X = np.array([features])
        sx = (X @ bx)[0]
        sy = (X @ by)[0]
        return sx, sy
    
    sx0, sy0 = predict_screen(dx0, dy0, yaw0, pitch0, zoom0)
    
    # Numerical derivatives
    sx_yaw, sy_yaw = predict_screen(dx0, dy0, yaw0 + eps*YAW_MOD, pitch0, zoom0)
    sx_pitch, sy_pitch = predict_screen(dx0, dy0, yaw0, pitch0 + eps*(PITCH_MAX-PITCH_MIN), zoom0)
    
    dx_dyaw = (sx_yaw - sx0) / (eps * YAW_MOD)
    dy_dyaw = (sy_yaw - sy0) / (eps * YAW_MOD)
    dx_dpitch = (sx_pitch - sx0) / (eps * (PITCH_MAX - PITCH_MIN))
    dy_dpitch = (sy_pitch - sy0) / (eps * (PITCH_MAX - PITCH_MIN))
    
    if use_zoom:
        sx_zoom, sy_zoom = predict_screen(dx0, dy0, yaw0, pitch0, zoom0 + eps*(ZOOM_MAX-ZOOM_MIN))
        dx_dzoom = (sx_zoom - sx0) / (eps * (ZOOM_MAX - ZOOM_MIN))
        dy_dzoom = (sy_zoom - sy0) / (eps * (ZOOM_MAX - ZOOM_MIN))
        J = np.array([
            [dx_dyaw, dx_dpitch, dx_dzoom],
            [dy_dyaw, dy_dpitch, dy_dzoom]
        ])
    else:
        J = np.array([
            [dx_dyaw, dx_dpitch],
            [dy_dyaw, dy_dpitch]
        ])
    
    # Return coefficients for compatibility (though we compute J directly)
    return bx, by, J


def fit_local_linear(neigh, yaw0, pitch0, zoom0, use_zoom):
    """
    Fit:
      sx ≈ ax0 + ax1*dyaw + ax2*dpitch (+ ax3*dzoom)
      sy ≈ ay0 + ay1*dyaw + ay2*dpitch (+ ay3*dzoom)

    Using relative variables improves conditioning and handles yaw wrap.
    """
    if len(neigh) < (8 if not use_zoom else 12):
        return None

    X = []
    Yx = []
    Yy = []

    for (_, _dx, _dy, dyaw, pitch, zoom, sx, sy) in neigh:
        dp = pitch - pitch0
        dz = zoom - zoom0
        if use_zoom:
            X.append([1.0, float(dyaw), float(dp), float(dz)])
        else:
            X.append([1.0, float(dyaw), float(dp)])
        Yx.append(float(sx))
        Yy.append(float(sy))

    X = np.asarray(X, dtype=float)
    Yx = np.asarray(Yx, dtype=float)
    Yy = np.asarray(Yy, dtype=float)

    # Least squares
    bx, *_ = np.linalg.lstsq(X, Yx, rcond=None)
    by, *_ = np.linalg.lstsq(X, Yy, rcond=None)

    return bx, by  # coefficient vectors


def solve_delta(bx, by, sx0, sy0, tx, ty, use_zoom, yaw0=None, pitch0=None, zoom0=None):
    """
    Build J and solve J Δ ≈ -(p0 - t) with bounds constraints.
    
    Enforces pitch bounds during the solve itself, not just after.
    """
    e = np.array([sx0 - tx, sy0 - ty], dtype=float)

    if use_zoom:
        # bx = [b0, b_dyaw, b_dpitch, b_dzoom]
        J = np.array([
            [bx[1], bx[2], bx[3]],
            [by[1], by[2], by[3]],
        ], dtype=float)
    else:
        J = np.array([
            [bx[1], bx[2]],
            [by[1], by[2]],
        ], dtype=float)

    rhs = -e
    
    # Use constrained least squares if bounds are provided
    if pitch0 is not None:
        if use_zoom:
            # Bounds: [yaw_low, pitch_low, zoom_low], [yaw_high, pitch_high, zoom_high]
            # For yaw: allow full range (will wrap), but limit step size
            # For pitch: enforce absolute bounds
            # For zoom: enforce absolute bounds
            bounds_low = np.array([-np.inf, PITCH_MIN - pitch0, ZOOM_MIN - zoom0])
            bounds_high = np.array([np.inf, PITCH_MAX - pitch0, ZOOM_MAX - zoom0])
        else:
            # Bounds: [yaw_low, pitch_low], [yaw_high, pitch_high]
            bounds_low = np.array([-np.inf, PITCH_MIN - pitch0])
            bounds_high = np.array([np.inf, PITCH_MAX - pitch0])
        
        # Use scipy's bounded least squares solver
        result = lsq_linear(J, rhs, bounds=(bounds_low, bounds_high), method='trf')
        delta = result.x
    else:
        # Fallback to unconstrained least squares if bounds not provided
        delta, *_ = np.linalg.lstsq(J, rhs, rcond=None)
    
    return e, J, delta


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_target_object(target_name):
    """Find the closest target object to the player."""
    try:
        player_x = player.get_x()
        player_y = player.get_y()
        
        if player_x is None or player_y is None:
            return None
        
        all_objects = ipc.get_objects()
        
        # Handle different response formats
        if isinstance(all_objects, dict):
            all_objects = all_objects.get("objects", [])
        elif not isinstance(all_objects, list):
            all_objects = []
        
        # Filter to only dict objects
        all_objects = [o for o in all_objects if isinstance(o, dict)]
        
        # Find matching objects
        matching_objs = [o for o in all_objects if o.get("name") == target_name]
        
        if not matching_objs:
            return None
        
        # Get closest object
        closest_obj = None
        min_dist = float('inf')
        for obj in matching_objs:
            obj_world = obj.get("world")
            if not obj_world:
                continue
            dist = math.sqrt(
                (obj_world["x"] - player_x) ** 2 + 
                (obj_world["y"] - player_y) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        return closest_obj
        
    except Exception as e:
        print(f"[ERROR] Error getting target object: {e}")
        return None


def get_current_screen_position(world_x, world_y):
    """Get current screen position using live projection."""
    try:
        proj = ipc.project_world_tile(world_x, world_y)
        if not proj or not proj.get("ok"):
            return None
        
        canvas = proj.get("canvas", {})
        if not canvas:
            return None
        
        return (float(canvas.get("x", 0)), float(canvas.get("y", 0)))
    except Exception as e:
        print(f"[ERROR] Error getting screen position: {e}")
        return None


def execute_camera_movement(yaw0, pitch0, zoom0, yaw_target, pitch_target, zoom_target):
    """Execute camera movement to reach target state."""
    # Calculate differences
    # Use wrap_yaw_diff helper for consistent yaw wrapping
    yaw_diff = wrap_yaw_diff(yaw_target, yaw0, mod=YAW_MOD)
    
    pitch_diff = pitch_target - pitch0
    zoom_diff = zoom_target - zoom0
    
    print(f"\n=== Executing Camera Movement ===")
    print(f"Yaw: {yaw0} -> {yaw_target} (diff: {yaw_diff})")
    print(f"Pitch: {pitch0} -> {pitch_target} (diff: {pitch_diff})")
    print(f"Zoom: {zoom0} -> {zoom_target} (diff: {zoom_diff})")
    
    movements_queued = False
    
    # Yaw movement
    if abs(yaw_diff) > 2:
        yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
        yaw_hold_ms = int(abs(yaw_diff) * YAW_MS_PER_UNIT)
        yaw_hold_ms = min(yaw_hold_ms, 2000)
        if yaw_hold_ms > 0:
            yaw_movement = {
                "type": "key_hold",
                "key": yaw_key,
                "duration_ms": yaw_hold_ms,
                "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
            }
            _camera_movement_queue.put(yaw_movement)
            movements_queued = True
            print(f"  Queued yaw: {yaw_key} for {yaw_hold_ms}ms")
    
    # Pitch movement
    if abs(pitch_diff) > 2:
        pitch_key = "UP" if pitch_diff > 0 else "DOWN"
        pitch_hold_ms = int(abs(pitch_diff) * PITCH_MS_PER_UNIT)
        pitch_hold_ms = min(pitch_hold_ms, 2000)
        if pitch_hold_ms > 0:
            pitch_movement = {
                "type": "key_hold",
                "key": pitch_key,
                "duration_ms": pitch_hold_ms,
                "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
            }
            _camera_movement_queue.put(pitch_movement)
            movements_queued = True
            print(f"  Queued pitch: {pitch_key} for {pitch_hold_ms}ms")
    
    # Zoom movement (if needed)
    if abs(zoom_diff) > 10:
        from services.camera_integration import calculate_zoom_scroll_count
        scroll_amount = 1 if zoom_diff > 0 else -1
        scroll_count = calculate_zoom_scroll_count(zoom_diff, zoom0)
        if scroll_count > 0:
            zoom_movement = {
                "type": "scroll",
                "amount": scroll_amount,
                "count": scroll_count
            }
            _camera_movement_queue.put(zoom_movement)
            movements_queued = True
            print(f"  Queued zoom: {scroll_amount} x {scroll_count}")
    
    if not movements_queued:
        print("  No movements needed")
        return
    
    # Wait for movements to complete
    print("\nWaiting for camera movement to complete...")
    max_wait_time = 3.0
    if abs(yaw_diff) > 2:
        max_wait_time = max(max_wait_time, (yaw_hold_ms / 1000.0) + 0.5)
    if abs(pitch_diff) > 2:
        max_wait_time = max(max_wait_time, (pitch_hold_ms / 1000.0) + 0.5)
    time.sleep(max_wait_time)
    
    # Wait for camera to stabilize
    wait_for_camera_stable("yaw", max_wait=2.0)
    wait_for_camera_stable("pitch", max_wait=2.0)
    wait_for_camera_stable("zoom", max_wait=2.0)
    time.sleep(0.3)
    
    print("Camera movement complete")


def save_global_model(model, jsonl_path, degree=2, use_zoom=False, player_pos=None):
    """
    Save global model to models/camera_jacobian/ folder with unique filename.
    
    Filename format: {base_name}_deg{degree}_zoom{use_zoom}_player{px}_{py}_{timestamp}.global_model.json
    """
    # Create models/camera_jacobian/ folder
    models_dir = Path(__file__).parent / "models" / "camera_jacobian"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name from JSONL file
    jsonl_path_obj = Path(jsonl_path)
    base_name = jsonl_path_obj.stem  # filename without extension
    
    # Build filename components
    zoom_str = "yes" if use_zoom else "no"
    player_str = ""
    if player_pos:
        px, py = player_pos
        player_str = f"_player{px}_{py}"
    
    # Add timestamp for uniqueness
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Build full filename
    model_filename = f"{base_name}_deg{degree}_zoom{zoom_str}{player_str}_{timestamp}.global_model.json"
    model_path = models_dir / model_filename
    
    # Save model
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model,
            'source_file': str(jsonl_path),
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'degree': degree,
                'use_zoom': use_zoom,
                'player_position': player_pos
            }
        }, f, indent=2)
    
    print(f"  Saved global model to: {model_path}")
    return model_path


def load_global_model(jsonl_path):
    """Load global model if it exists."""
    model_path = Path(jsonl_path).with_suffix('.global_model.json')
    if not model_path.exists():
        return None
    
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Loaded pre-computed global model from: {model_path}")
        return data['model']
    except Exception as e:
        print(f"[WARNING] Failed to load global model: {e}")
        return None


def main():
    global ipc, _global_model, _global_model_file
    
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Calibration JSONL path")
    ap.add_argument("--target", type=str, default=None, help=f"Target object name (default: {DEFAULT_TARGET} if --tile not provided)")
    ap.add_argument("--tile", type=str, default=None, help="Target tile coordinates as 'x,y' (e.g., '3013,3355'). If provided, uses tile instead of object")
    ap.add_argument("--port", type=int, default=17002, help="IPC port number (default: 17002)")
    ap.add_argument("--use-zoom", action="store_true", help="Solve for Δzoom too (3 vars)")
    ap.add_argument("--k", type=int, default=500, help="Neighbor count for local fit")
    ap.add_argument("--max-dxdy", type=int, default=3, help="Tile window around (dx,dy)")
    ap.add_argument("--step-scale", type=float, default=1.0, help="Damping factor on suggested delta (1.0 = no damping)")
    ap.add_argument("--max-yaw-step", type=float, default=9999.0, help="Maximum yaw step (9999.0 = no clamping)")
    ap.add_argument("--max-pitch-step", type=float, default=9999.0, help="Maximum pitch step (9999.0 = no clamping)")
    ap.add_argument("--max-zoom-step", type=float, default=512.0)
    ap.add_argument("--build-global-model", action="store_true", help="Pre-compute global polynomial model from calibration data")
    ap.add_argument("--polynomial-degree", type=int, default=2, help="Polynomial degree for global model (default: 2)")
    ap.add_argument("--use-local-fit", action="store_true", default=False, help="Force local fit instead of global model (enables LM+backtracking for better large error handling)")
    ap.add_argument("--no-local-fit", dest="use_local_fit", action="store_false", help="Use global model if available (default)")
    ap.add_argument("--iterative", action="store_true", default=True, help="Use iterative refinement (default: True)")
    ap.add_argument("--no-iterative", dest="iterative", action="store_false", help="Disable iterative refinement (use one-shot)")
    ap.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations for iterative refinement (default: 10)")
    ap.add_argument("--error-threshold", type=float, default=100.0, help="Stop when error is below this (pixels, default: 100)")
    ap.add_argument("--max-initial-error", type=float, default=500.0, help="If initial error > this, use smaller step_scale (default: 500)")
    ap.add_argument("--turbo", action="store_true", default=True, help="Turbo mode: no wait times, maximum speed (default: True)")
    ap.add_argument("--no-turbo", dest="turbo", action="store_false", help="Disable turbo mode (use normal wait times)")
    ap.add_argument("--manual", action="store_true", default=True, help="Manual mode: step through each iteration (press ENTER to continue, default: True)")
    ap.add_argument("--auto", dest="manual", action="store_false", help="Auto mode: run all iterations automatically")
    ap.add_argument("--smooth-steps", type=int, default=0, help="Number of intermediate steps for smooth movement (0 = disabled, default: 0, 5-10 recommended if enabled)")
    ap.add_argument("--smooth-all-iterations", action="store_true", default=False, help="Apply smooth movement to all iterations (default: only first iteration)")
    # LM solver parameters (for local fit only)
    ap.add_argument("--max-lm-iterations", type=int, default=10, help="Max inner LM iterations (default: 10, higher = more refinement)")
    ap.add_argument("--lm-tolerance", type=float, default=5.0, help="Stop inner LM loop if error < this (pixels, default: 5.0)")
    ap.add_argument("--lm-initial-lambda", type=float, default=5.0, help="Initial LM damping parameter (default: 5.0, lower = more aggressive)")
    ap.add_argument("--lm-lambda-reduction", type=float, default=2.5, help="LM lambda reduction factor on success (default: 2.5, higher = faster reduction)")
    ap.add_argument("--lm-allow-overshoot", action="store_true", default=False, help="Allow LM to test step sizes > 1.0 (overshooting)")
    # Jacobian computation method
    ap.add_argument("--jacobian", type=str, default="finite_diff", choices=["neighbors", "finite_diff", "mlp"],
                    help="Jacobian computation method: 'finite_diff' (default, uses oracle projector), 'neighbors' (uses regression), 'mlp' (future)")
    ap.add_argument("--finite-diff-dyaw", type=float, default=0.0, help="Yaw perturbation for finite-difference Jacobian (0.0 = auto-adaptive based on error, default: 0.0)")
    ap.add_argument("--finite-diff-dpitch", type=float, default=0.0, help="Pitch perturbation for finite-difference Jacobian (0.0 = auto-adaptive based on error, default: 0.0)")
    args = ap.parse_args()

    # Initialize IPC early (needed for both model building and normal operation)
    try:
        ipc = IPCClient(port=args.port)
        set_ipc(ipc)
        ipc_available = True
    except Exception as e:
        ipc_available = False
        if not args.build_global_model:
            # Only error if we need IPC for normal operation
            print(f"[ERROR] Could not initialize IPC on port {args.port}: {e}")
            return
        else:
            # For model building, IPC is optional (just for player position in filename)
            print(f"[INFO] IPC not available on port {args.port} (will skip player position in filename): {e}")

    # Handle global model building
    if args.build_global_model:
        print("\n" + "=" * 60)
        print("BUILDING GLOBAL POLYNOMIAL MODEL")
        print("=" * 60)
        print(f"\n[INFO] Loading calibration data from: {args.jsonl}")
        rows = load_rows(args.jsonl)
        if not rows:
            print("[ERROR] No usable rows loaded")
            return
        print(f"Loaded {len(rows)} data points")
        
        # Try to get player position from IPC if available
        player_pos = None
        if ipc_available:
            try:
                player_x = player.get_x()
                player_y = player.get_y()
                if player_x is not None and player_y is not None:
                    player_pos = (player_x, player_y)
                    print(f"[INFO] Detected player position: {player_pos}")
            except Exception as e:
                print(f"[INFO] Could not get player position: {e}, model will not include player position in filename")
        else:
            print(f"[INFO] IPC not available, model will not include player position in filename")
        
        global_model = fit_global_polynomial_model(rows, degree=args.polynomial_degree, use_zoom=args.use_zoom)
        if global_model:
            model_path = save_global_model(global_model, args.jsonl, degree=args.polynomial_degree, use_zoom=args.use_zoom, player_pos=player_pos)
            print("\n[SUCCESS] Global model built and saved!")
            print(f"Model saved to: {model_path}")
            print("You can now use this model by specifying the calibration_data_path in your code.")
        else:
            print("\n[ERROR] Failed to build global model")
        return
    
    # Ensure camera thread is running
    _ensure_camera_thread()
    
    # Try to load pre-computed global model (only once)
    # Skip if --use-local-fit is specified (forces LM+backtracking)
    global _global_model, _global_model_file
    if args.use_local_fit:
        _global_model = None
        _global_model_file = None
        print("[INFO] Using local fit (LM+backtracking enabled for better large error handling)")
    else:
        _global_model = load_global_model(args.jsonl)
        _global_model_file = args.jsonl if _global_model else None
        if _global_model:
            print("[INFO] Using global model (faster, but LM+backtracking only available with local fit)")
    
    # Load calibration data if using local fit (only once)
    rows = None
    if not _global_model:
        rows = load_rows(args.jsonl)
        if not rows:
            print("[ERROR] No usable rows loaded (check JSONL format / screen nulls).")
            return
    
    # Initialize parameters (can be adjusted interactively)
    # Defaults: smooth_steps=8, error_threshold=100.0
    current_params = {
        'step_scale': args.step_scale,
        'max_yaw_step': args.max_yaw_step,
        'max_pitch_step': args.max_pitch_step,
        'max_iterations': args.max_iterations,
        'error_threshold': args.error_threshold,  # Default: 100.0
        'min_wait_time': 0.1 if args.turbo else 0.2,
        'manual_mode': args.manual,
        'smooth_steps': args.smooth_steps  # Default: 0
    }
    
    # Loop for multiple movements
    movement_count = 0
    while True:
        movement_count += 1
        print("\n" + "=" * 60)
        print(f"JACOBIAN-BASED CAMERA MOVEMENT (Movement #{movement_count})")
        print("=" * 60)
        
        # Get player position
        print("\n[INFO] Retrieving player position...")
        px = player.get_x()
        py = player.get_y()
        if px is None or py is None:
            print("[ERROR] Could not get player position")
            break
        print(f"  Player position: ({px}, {py})")
        
        # Get camera state
        print("\n[INFO] Retrieving camera state...")
        camera_data = ipc.get_camera()
        if not camera_data:
            print("[ERROR] Could not get camera data")
            break
        yaw0 = camera_data.get("yaw", 0)
        pitch0 = camera_data.get("pitch", 256)
        zoom0 = camera_data.get("scale", 512)
        print(f"  Camera: yaw={yaw0}, pitch={pitch0}, zoom={zoom0}")
        
        # Get target coordinates (from tile or object)
        ox = None
        oy = None
        
        if args.tile:
            # Parse tile coordinates
            try:
                parts = args.tile.split(',')
                if len(parts) != 2:
                    raise ValueError("Tile must be in format 'x,y'")
                ox = int(parts[0].strip())
                oy = int(parts[1].strip())
                print(f"\n[INFO] Using tile coordinates: ({ox}, {oy})")
            except (ValueError, IndexError) as e:
                print(f"[ERROR] Invalid tile format: {args.tile}. Expected 'x,y' (e.g., '3013,3355')")
                print("Press ENTER to try again, or 'Q' to quit...")
                user_input = input().strip().upper()
                if user_input == 'Q':
                    break
                continue
        else:
            # Get target object
            target_name = args.target if args.target else DEFAULT_TARGET
            print(f"\n[INFO] Finding target object: {target_name}...")
            target_obj = get_target_object(target_name)
            if not target_obj:
                print(f"[ERROR] Could not find target object: {target_name}")
                print("Press ENTER to try again, or 'Q' to quit...")
                user_input = input().strip().upper()
                if user_input == 'Q':
                    break
                continue
            
            obj_world = target_obj.get("world")
            if not obj_world:
                print(f"[ERROR] Target object has no world coordinates")
                print("Press ENTER to try again, or 'Q' to quit...")
                user_input = input().strip().upper()
                if user_input == 'Q':
                    break
                continue
            
            ox = obj_world["x"]
            oy = obj_world["y"]
            print(f"  Object world position: ({ox}, {oy})")
        
        if ox is None or oy is None:
            print("[ERROR] Could not determine target coordinates")
            print("Press ENTER to try again, or 'Q' to quit...")
            user_input = input().strip().upper()
            if user_input == 'Q':
                break
            continue
        
        dx0 = ox - px
        dy0 = oy - py
        print(f"  dx,dy (object-player): ({dx0}, {dy0})")
        
        # Get screen dimensions and calculate target position
        where = ipc.where() or {}
        screen_width = int(where.get("w", 0))
        screen_height = int(where.get("h", 0))
        if screen_width == 0 or screen_height == 0:
            print("[ERROR] Could not get screen dimensions")
            print("Press ENTER to try again, or 'Q' to quit...")
            user_input = input().strip().upper()
            if user_input == 'Q':
                break
            continue
        
        # Determine target screen position
        target_screen = get_screen_position_preset("center_top", screen_width, screen_height)
        tx = target_screen["x"]
        ty = target_screen["y"]
        print(f"\n[INFO] Target screen position: ({tx}, {ty}) (center X, 25% from top Y)")
        
        # Get current screen position for display
        screen_pos = get_current_screen_position(ox, oy)
        if screen_pos:
            sx0, sy0 = screen_pos
            initial_error = math.hypot(sx0 - tx, sy0 - ty)
            print(f"[INFO] Current screen position: ({sx0:.1f}, {sy0:.1f})")
            print(f"[INFO] Initial error: {initial_error:.1f} pixels")
        else:
            print("[WARNING] Could not get current screen position (target may be off-screen)")
        
        # Interactive parameter adjustment menu
        while True:
            # Show current values
            print("\n" + "=" * 60)
            print("CURRENT PARAMETERS:")
            print("=" * 60)
            print(f"  [D] Dampening (step_scale):     {current_params['step_scale']:.2f}")
            print(f"  [Y] Max Yaw Step:               {current_params['max_yaw_step']:.1f}")
            print(f"  [P] Max Pitch Step:              {current_params['max_pitch_step']:.1f}")
            print(f"  [I] Max Iterations:             {current_params['max_iterations']}")
            print(f"  [E] Error Threshold:            {current_params['error_threshold']:.1f}px")
            print(f"  [W] Wait Time (seconds):        {current_params['min_wait_time']:.2f}s")
            print(f"  [M] Manual/Auto Mode:           {'MANUAL' if current_params['manual_mode'] else 'AUTO'}")
            print(f"  [S] Smooth Steps:                {current_params['smooth_steps']} (0 = disabled)")
            print("=" * 60)
            print("\nOptions:")
            print("  Press ENTER to execute with current parameters")
            print("  Press [D/Y/P/I/E/W/M/S] to adjust a parameter")
            print("  Press 'N' for next movement")
            print("  Press 'Q' to quit")
            print("\nEnter command: ", end="")
            
            user_input = input().strip().upper()
            
            if user_input == '' or user_input == 'ENTER':
                break
            elif user_input == 'Q':
                print("Quitting...")
                return
            elif user_input == 'N':
                print("Skipping this movement, calculating next...")
                continue
            elif user_input == 'D':
                try:
                    new_val = float(input(f"  Enter new step_scale (current: {current_params['step_scale']:.2f}): "))
                    if 0.0 <= new_val <= 2.0:
                        current_params['step_scale'] = new_val
                        print(f"  ✓ Step scale set to {new_val:.2f}")
                    else:
                        print(f"  ✗ Value must be between 0.0 and 2.0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'Y':
                try:
                    new_val = float(input(f"  Enter new max_yaw_step (current: {current_params['max_yaw_step']:.1f}): "))
                    if new_val > 0:
                        current_params['max_yaw_step'] = new_val
                        print(f"  ✓ Max yaw step set to {new_val:.1f}")
                    else:
                        print(f"  ✗ Value must be > 0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'P':
                try:
                    new_val = float(input(f"  Enter new max_pitch_step (current: {current_params['max_pitch_step']:.1f}): "))
                    if new_val > 0:
                        current_params['max_pitch_step'] = new_val
                        print(f"  ✓ Max pitch step set to {new_val:.1f}")
                    else:
                        print(f"  ✗ Value must be > 0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'I':
                try:
                    new_val = int(input(f"  Enter new max_iterations (current: {current_params['max_iterations']}): "))
                    if new_val > 0:
                        current_params['max_iterations'] = new_val
                        print(f"  ✓ Max iterations set to {new_val}")
                    else:
                        print(f"  ✗ Value must be > 0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'E':
                try:
                    new_val = float(input(f"  Enter new error_threshold (current: {current_params['error_threshold']:.1f}px): "))
                    if new_val > 0:
                        current_params['error_threshold'] = new_val
                        print(f"  ✓ Error threshold set to {new_val:.1f}px")
                    else:
                        print(f"  ✗ Value must be > 0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'W':
                try:
                    new_val = float(input(f"  Enter new wait_time in seconds (current: {current_params['min_wait_time']:.2f}s): "))
                    if new_val >= 0:
                        current_params['min_wait_time'] = new_val
                        print(f"  ✓ Wait time set to {new_val:.2f}s")
                    else:
                        print(f"  ✗ Value must be >= 0")
                except ValueError:
                    print("  ✗ Invalid number")
            elif user_input == 'M':
                current_params['manual_mode'] = not current_params['manual_mode']
                mode_str = 'MANUAL' if current_params['manual_mode'] else 'AUTO'
                print(f"  ✓ Mode switched to {mode_str}")
            elif user_input == 'S':
                try:
                    new_val = int(input(f"  Enter new smooth_steps (current: {current_params['smooth_steps']}, 0 = disabled, 5-10 recommended): "))
                    if new_val >= 0:
                        current_params['smooth_steps'] = new_val
                        print(f"  ✓ Smooth steps set to {new_val}")
                    else:
                        print(f"  ✗ Value must be >= 0")
                except ValueError:
                    print("  ✗ Invalid number")
            else:
                print(f"  ✗ Unknown command: '{user_input}'. Try D, Y, P, I, E, W, M, S, N, Q, or ENTER")
        
        # Use iterative refinement if enabled, otherwise use one-shot
        if args.iterative:
            print("\n" + "=" * 60)
            print("ITERATIVE CAMERA REFINEMENT")
            print("=" * 60)
            mode_str = "MANUAL" if current_params['manual_mode'] else "AUTO"
            print(f"Mode: {mode_str} | Max iterations: {current_params['max_iterations']} | Threshold: {current_params['error_threshold']:.1f}px")
            print(f"Step scale: {current_params['step_scale']:.2f} | Wait time: {current_params['min_wait_time']:.2f}s")
            if current_params['manual_mode']:
                print("You will be prompted before each iteration (press ENTER to continue)")
            
            result = aim_camera_at_target_iterative(
                world_coords={"x": ox, "y": oy},
                target_screen=target_screen,
                calibration_data_path=args.jsonl,
                max_iterations=current_params['max_iterations'],
                error_threshold=current_params['error_threshold'],
                max_initial_error=args.max_initial_error,
                use_global_model=_global_model is not None,
                use_zoom=args.use_zoom,
                step_scale=current_params['step_scale'],
                max_yaw_step=current_params['max_yaw_step'],
                max_pitch_step=current_params['max_pitch_step'],
                fast_mode=args.turbo,  # Turbo mode: no wait times, maximum speed
                min_wait_time=current_params['min_wait_time'],  # User-adjustable wait time
                manual_mode=current_params['manual_mode'],  # User-adjustable manual mode
                smooth_steps=current_params['smooth_steps'],  # User-adjustable smooth steps
                # LM solver parameters
                max_lm_iterations=args.max_lm_iterations,
                lm_tolerance=args.lm_tolerance,
                lm_initial_lambda=args.lm_initial_lambda,
                lm_lambda_reduction=args.lm_lambda_reduction,
                lm_allow_overshoot=args.lm_allow_overshoot,
                # Jacobian method
                jacobian_method=args.jacobian,
                finite_diff_dyaw=args.finite_diff_dyaw,
                finite_diff_dpitch=args.finite_diff_dpitch
            )
            
            print("\n" + "=" * 60)
            print("ITERATIVE REFINEMENT RESULTS")
            print("=" * 60)
            
            if result.get("success"):
                print(f"\n✓ {result.get('message')}")
                print(f"  Final error: {result.get('final_error', 0):.1f} pixels")
            else:
                print(f"\n✗ {result.get('message')}")
            
            iterations = result.get("iterations", [])
            if iterations:
                print(f"\nIteration details:")
                for it in iterations:
                    print(f"  Iteration {it['iteration']}: {it['error_before']:.1f}px -> {it['error_after']:.1f}px "
                          f"(improvement: {it['improvement']:.1f}px, step_scale: {it['step_scale']:.2f})")
            
            print("=" * 60 + "\n")
        else:
            # One-shot mode (original behavior) - keep existing logic if needed
            print("\n[INFO] One-shot mode not implemented - use --iterative (default)")
            continue
        
        # Ask if user wants to continue
        print("Press ENTER to do another movement, or 'Q' to quit...")
        user_input = input().strip().upper()
        if user_input == 'Q':
            break
    
    stop_camera_thread()


if __name__ == "__main__":
    main()
