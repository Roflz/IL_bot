"""
Jacobian-based camera movement system.

This module provides functions to calculate and execute precise camera movements
using a Jacobian matrix derived from calibration data. It can position objects
at exact screen coordinates using a single calculation instead of iterative loops.
"""

import json
import math
import time
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

# ANSI color codes
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_CYAN = '\033[96m'
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_RED = '\033[91m'
COLOR_MAGENTA = '\033[95m'


def _get_error_color(error_pixels):
    """Get color based on error magnitude."""
    if error_pixels < 50:
        return COLOR_GREEN
    elif error_pixels < 150:
        return COLOR_YELLOW
    else:
        return COLOR_RED


def _format_table_row(label, value, color=COLOR_RESET):
    """Format a table row with label and value."""
    return f"  {COLOR_BLUE}{label:20s}{COLOR_RESET} │ {color}{value}{COLOR_RESET}"


def _print_table_header(title):
    """Print a formatted table header."""
    print(f"\n{COLOR_CYAN}{COLOR_BOLD}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{COLOR_RESET}")


def _print_table_separator():
    """Print a table separator line."""
    print(f"  {COLOR_BLUE}{'─'*20}┼{'─'*38}{COLOR_RESET}")

from helpers.runtime_utils import ipc
from services.camera_integration import (
    _ensure_camera_thread,
    _camera_movement_queue,
    YAW_MS_PER_UNIT,
    PITCH_MS_PER_UNIT,
    calculate_zoom_scroll_count
)

# Camera state bounds
YAW_MOD = 2048
PITCH_MIN = 128
PITCH_MAX = 383
ZOOM_MIN = 551
ZOOM_MAX = 4409

# Jacobian cache (for performance optimization)
# Key: (dx, dy, yaw_bin, pitch_bin, zoom_bin) -> (bx, by, J)
_jacobian_cache = {}
_jacobian_cache_size = 100  # Max cache entries
# Bin sizes for caching (larger = more reuse, less accuracy)
JACOBIAN_CACHE_YAW_BIN = 64   # ~32 bins across 2048 yaw range
JACOBIAN_CACHE_PITCH_BIN = 16  # ~16 bins across 256 pitch range
JACOBIAN_CACHE_ZOOM_BIN = 200  # ~20 bins across zoom range

# Global model cache (per calibration file)
_global_model_cache = {}


def wrap_yaw_diff(a, b, mod=2048):
    """Smallest signed diff a-b in yaw units (wraparound)."""
    d = (a - b) % mod
    if d > mod / 2:
        d -= mod
    return d


def load_rows(jsonl_path):
    """
    Load calibration data from JSONL file.
    
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
        logging.warning(f"Not enough data points ({len(rows)}) for global model")
        return None
    
    logging.info(f"Fitting global polynomial model (degree={degree}) from {len(rows)} data points...")
    
    # Prepare feature matrix X and targets Yx, Yy
    X_features = []
    Yx = []
    Yy = []
    
    for (dx, dy, yaw, pitch, zoom, sx, sy) in rows:
        # Normalize inputs for better conditioning
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
        
        logging.info(f"Model fitted: R² for sx={r2_x:.4f}, R² for sy={r2_y:.4f}")
        
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
        logging.error(f"Failed to fit global model: {e}")
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
    
    # Use numerical differentiation (small perturbation)
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
    # Use wrapped yaw to handle wrap-around correctly
    yaw_perturbed = (yaw0 + eps*YAW_MOD) % YAW_MOD
    sx_yaw, sy_yaw = predict_screen(dx0, dy0, yaw_perturbed, pitch0, zoom0)
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
    
    return bx, by, J


def predict_screen_from_model(model, dx0, dy0, yaw, pitch, zoom, use_zoom):
    """Predict screen position using the global model."""
    bx = np.array(model['bx'])
    by = np.array(model['by'])
    degree = model['degree']
    
    # Normalize inputs
    yaw_n = yaw / YAW_MOD
    pitch_n = (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
    zoom_n = (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) if use_zoom else 0.0
    
    features = []
    if degree >= 1:
        features.extend([1.0, dx0, dy0, yaw_n, pitch_n])
        if use_zoom:
            features.append(zoom_n)
    if degree >= 2:
        features.extend([
            dx0*dx0, dy0*dy0, yaw_n*yaw_n, pitch_n*pitch_n,
            dx0*dy0, dx0*yaw_n, dx0*pitch_n, dy0*yaw_n, dy0*pitch_n, yaw_n*pitch_n
        ])
        if use_zoom:
            features.extend([
                zoom_n*zoom_n,
                dx0*zoom_n, dy0*zoom_n, yaw_n*zoom_n, pitch_n*zoom_n
            ])
    
    X = np.array([features])
    sx = (X @ bx)[0]
    sy = (X @ by)[0]
    return sx, sy


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


def predict_screen_from_delta(sx0, sy0, ax, ay, dyaw, dpitch):
    """
    Predict screen position from camera delta using local linear model.
    
    Args:
        sx0, sy0: Current screen position
        ax, ay: Coefficient vectors from local fit [ax0, ax1, ax2] (no zoom)
        dyaw, dpitch: Camera deltas
    
    Returns:
        (predicted_sx, predicted_sy): Predicted screen position
    """
    # sx ≈ ax0 + ax1*dyaw + ax2*dpitch
    # But we want delta from current, so: sx_pred = sx0 + ax1*dyaw + ax2*dpitch
    predicted_sx = sx0 + ax[1] * dyaw + ax[2] * dpitch
    predicted_sy = sy0 + ay[1] * dyaw + ay[2] * dpitch
    return predicted_sx, predicted_sy


def project_screen_xy_oracle(
    camera_yaw: int,
    camera_pitch: int,
    camera_zoom: int,
    player_pos: dict,
    object_world_pos: dict,
    global_model: dict = None,
    calibration_data_path: str = None,
    use_zoom: bool = False
) -> tuple[float, float] | None:
    """
    Oracle projector: Returns object's screen xy in pixels for a given camera state.
    Uses the global polynomial model as a pure function (no camera movement).
    
    Args:
        camera_yaw: Camera yaw in units (0-2047)
        camera_pitch: Camera pitch in units (128-383)
        camera_zoom: Camera zoom in units (551-4409)
        player_pos: Player position {"x": int, "y": int}
        object_world_pos: Object world position {"x": int, "y": int}
        global_model: Pre-loaded global model (optional)
        calibration_data_path: Path to calibration data (if model not provided)
        use_zoom: Whether to use zoom in the model
    
    Returns:
        Tuple of (screen_x, screen_y) in pixels, or None if projection fails
    """
    try:
        # Calculate relative position
        dx0 = object_world_pos["x"] - player_pos["x"]
        dy0 = object_world_pos["y"] - player_pos["y"]
        
        # Load model if not provided
        if global_model is None:
            if calibration_data_path is None:
                return None
            global_model = load_global_model(calibration_data_path)
            if global_model is None:
                return None
        
        # Use global model to predict screen position
        sx, sy = predict_screen_from_model(
            global_model, dx0, dy0, camera_yaw, camera_pitch, camera_zoom, use_zoom
        )
        
        return (sx, sy)
    except Exception as e:
        logging.debug(f"Oracle projection failed: {e}")
        return None


def jacobian_finite_diff(
    project_fn: callable,
    yaw0: int,
    pitch0: int,
    zoom0: int,
    player_pos: dict,
    obj_pos: dict,
    dyaw: float = None,  # Auto-adaptive if None
    dpitch: float = None,  # Auto-adaptive if None
    use_zoom: bool = False,
    error_pixels: float = None,  # For adaptive perturbation sizing
    use_central_diff: bool = True  # Use central differences (yaw±δ, pitch±δ) for better accuracy
) -> tuple[tuple[float, float], np.ndarray] | None:
    """
    Compute Jacobian using finite differences with an oracle projector.
    Supports adaptive perturbation sizing and central differences for better accuracy.
    
    Args:
        project_fn: Oracle projection function: (yaw, pitch, zoom, player_pos, obj_pos) -> (sx, sy) | None
        yaw0: Current yaw
        pitch0: Current pitch
        zoom0: Current zoom
        player_pos: Player position {"x": int, "y": int}
        obj_pos: Object world position {"x": int, "y": int}
        dyaw: Yaw perturbation (None = auto-adaptive based on error)
        dpitch: Pitch perturbation (None = auto-adaptive based on error)
        use_zoom: Whether to compute zoom derivatives
        error_pixels: Current error in pixels (for adaptive sizing)
        use_central_diff: If True, use central differences (yaw±δ) for better accuracy
    
    Returns:
        Tuple of ((sx0, sy0), J) where J is the 2x2 (or 2x3) Jacobian matrix, or None if projection fails
    """
    # Adaptive perturbation sizing based on error magnitude
    if dyaw is None:
        if error_pixels is not None:
            # Larger errors need larger perturbations to get accurate derivatives
            if error_pixels > 500:
                dyaw = 128.0  # ~22.5 degrees for very large errors
            elif error_pixels > 200:
                dyaw = 64.0   # ~11.2 degrees for large errors
            elif error_pixels > 100:
                dyaw = 48.0   # ~8.4 degrees for medium errors
            else:
                dyaw = 32.0   # ~5.6 degrees for small errors
        else:
            dyaw = 64.0  # Default: ~11.2 degrees
    
    if dpitch is None:
        if error_pixels is not None:
            if error_pixels > 500:
                dpitch = 32.0  # ~11.3 degrees for very large errors
            elif error_pixels > 200:
                dpitch = 24.0  # ~8.5 degrees for large errors
            elif error_pixels > 100:
                dpitch = 16.0  # ~5.7 degrees for medium errors
            else:
                dpitch = 12.0  # ~4.2 degrees for small errors
        else:
            dpitch = 16.0  # Default: ~5.7 degrees
    
    # Project at current state
    p0 = project_fn(yaw0, pitch0, zoom0, player_pos, obj_pos)
    if p0 is None:
        return None
    x0, y0 = p0
    
    if use_central_diff:
        # Central differences: (f(x+δ) - f(x-δ)) / (2δ) - more accurate than forward differences
        # Project with yaw +δ
        yaw_plus = int((yaw0 + dyaw) % YAW_MOD)
        p_yaw_plus = project_fn(yaw_plus, pitch0, zoom0, player_pos, obj_pos)
        if p_yaw_plus is None:
            return None
        x1_plus, y1_plus = p_yaw_plus
        
        # Project with yaw -δ (handle wrapping)
        yaw_minus = int((yaw0 - dyaw) % YAW_MOD)
        p_yaw_minus = project_fn(yaw_minus, pitch0, zoom0, player_pos, obj_pos)
        if p_yaw_minus is None:
            return None
        x1_minus, y1_minus = p_yaw_minus
        
        # Project with pitch +δ
        pitch_plus = int(max(PITCH_MIN, min(PITCH_MAX, pitch0 + dpitch)))
        p_pitch_plus = project_fn(yaw0, pitch_plus, zoom0, player_pos, obj_pos)
        if p_pitch_plus is None:
            return None
        x2_plus, y2_plus = p_pitch_plus
        
        # Project with pitch -δ
        pitch_minus = int(max(PITCH_MIN, min(PITCH_MAX, pitch0 - dpitch)))
        p_pitch_minus = project_fn(yaw0, pitch_minus, zoom0, player_pos, obj_pos)
        if p_pitch_minus is None:
            return None
        x2_minus, y2_minus = p_pitch_minus
        
        # Compute central differences
        actual_dyaw = wrap_yaw_diff(yaw_plus, yaw_minus, mod=YAW_MOD)
        if abs(actual_dyaw) < 0.1:
            actual_dyaw = 2.0 * dyaw  # Fallback to 2*dyaw if wrap calculation fails
        else:
            actual_dyaw = abs(actual_dyaw)
        
        actual_dpitch = pitch_plus - pitch_minus
        if actual_dpitch < 0.1:
            actual_dpitch = 2.0 * dpitch
        
        # Central difference: (f(x+δ) - f(x-δ)) / (2δ)
        J = np.array([
            [(x1_plus - x1_minus) / actual_dyaw, (x2_plus - x2_minus) / actual_dpitch],
            [(y1_plus - y1_minus) / actual_dyaw, (y2_plus - y2_minus) / actual_dpitch],
        ], dtype=float)
    else:
        # Forward differences (original method)
        # Project with yaw perturbation (handle wrapping)
        yaw_perturbed = int((yaw0 + dyaw) % YAW_MOD)
        p_yaw = project_fn(yaw_perturbed, pitch0, zoom0, player_pos, obj_pos)
        if p_yaw is None:
            return None
        x1, y1 = p_yaw
        
        # Project with pitch perturbation (handle clamping)
        pitch_perturbed = int(max(PITCH_MIN, min(PITCH_MAX, pitch0 + dpitch)))
        p_pitch = project_fn(yaw0, pitch_perturbed, zoom0, player_pos, obj_pos)
        if p_pitch is None:
            return None
        x2, y2 = p_pitch
        
        # Compute finite differences
        # Handle yaw wrapping: use wrapped difference
        actual_dyaw = wrap_yaw_diff(yaw_perturbed, yaw0, mod=YAW_MOD)
        if abs(actual_dyaw) < 0.1:  # Too small, use original dyaw
            actual_dyaw = dyaw
        
        actual_dpitch = pitch_perturbed - pitch0
        
        # Build Jacobian: J[i,j] = ∂screen_i / ∂camera_j
        # J[0,0] = ∂screen_x / ∂yaw
        # J[0,1] = ∂screen_x / ∂pitch
        # J[1,0] = ∂screen_y / ∂yaw
        # J[1,1] = ∂screen_y / ∂pitch
        J = np.array([
            [(x1 - x0) / actual_dyaw, (x2 - x0) / actual_dpitch],
            [(y1 - y0) / actual_dyaw, (y2 - y0) / actual_dpitch],
        ], dtype=float)
    
    if use_zoom:
        # Project with zoom perturbation
        zoom_perturbed = int(max(ZOOM_MIN, min(ZOOM_MAX, zoom0 + 50)))  # Small zoom delta
        p_zoom = project_fn(yaw0, pitch0, zoom_perturbed, player_pos, obj_pos)
        if p_zoom is None:
            return None
        x3, y3 = p_zoom
        actual_dzoom = zoom_perturbed - zoom0
        
        # Add zoom column to Jacobian
        J = np.array([
            [(x1 - x0) / actual_dyaw, (x2 - x0) / actual_dpitch, (x3 - x0) / actual_dzoom],
            [(y1 - y0) / actual_dyaw, (y2 - y0) / actual_dpitch, (y3 - y0) / actual_dzoom],
        ], dtype=float)
    
    return (x0, y0), J


def solve_lm(J, e, lam):
    """
    Solve Levenberg-Marquardt step: (J^T J + λI) Δ = -J^T e
    
    Args:
        J: 2x2 Jacobian matrix
        e: 2-element error vector
        lam: Damping parameter λ
    
    Returns:
        delta: 2-element solution vector [dyaw, dpitch]
    """
    # J is 2x2, e is (2,)
    A = J.T @ J + lam * np.eye(J.shape[1])
    b = -J.T @ e
    return np.linalg.solve(A, b)


def solve_delta(bx, by, sx0, sy0, tx, ty, use_zoom, yaw0=None, pitch0=None, zoom0=None):
    """
    Build J and solve J Δ ≈ -(p0 - t) with bounds constraints.
    
    Enforces pitch bounds during the solve itself, not just after.
    """
    e = np.array([sx0 - tx, sy0 - ty], dtype=float)

    if use_zoom:
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
            bounds_low = np.array([-np.inf, PITCH_MIN - pitch0, ZOOM_MIN - zoom0])
            bounds_high = np.array([np.inf, PITCH_MAX - pitch0, ZOOM_MAX - zoom0])
        else:
            bounds_low = np.array([-np.inf, PITCH_MIN - pitch0])
            bounds_high = np.array([np.inf, PITCH_MAX - pitch0])
        
        result = lsq_linear(J, rhs, bounds=(bounds_low, bounds_high), method='trf')
        delta = result.x
    else:
        # Fallback to unconstrained least squares if bounds not provided
        delta, *_ = np.linalg.lstsq(J, rhs, rcond=None)
    
    return e, J, delta


def load_global_model(jsonl_path):
    """
    Load global model if it exists (with caching).
    
    Searches in this order:
    1. models/camera_jacobian/ folder (new location) - finds most recent matching model
    2. Old location (next to JSONL file) - for backward compatibility
    """
    # Check cache first
    if jsonl_path in _global_model_cache:
        return _global_model_cache[jsonl_path]
    
    jsonl_path_obj = Path(jsonl_path)
    base_name = jsonl_path_obj.stem
    
    # Try new location first: models/camera_jacobian/
    models_dir = Path(__file__).parent.parent / "models" / "camera_jacobian"
    if models_dir.exists():
        # Find all models matching the base name
        matching_models = list(models_dir.glob(f"{base_name}_deg*_zoom*_player*_*.global_model.json"))
        if not matching_models:
            # Try without player position
            matching_models = list(models_dir.glob(f"{base_name}_deg*_zoom*_*.global_model.json"))
        
        if matching_models:
            # Sort by modification time (most recent first)
            matching_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = matching_models[0]
            logging.debug(f"Found model in models/camera_jacobian/: {model_path.name}")
        else:
            model_path = None
    else:
        model_path = None
    
    # Fall back to old location (next to JSONL file)
    if model_path is None:
        old_model_path = jsonl_path_obj.with_suffix('.global_model.json')
        if old_model_path.exists():
            model_path = old_model_path
            logging.debug(f"Using model from old location: {model_path}")
        else:
            logging.debug(f"No global model found for {base_name}")
            return None
    
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = data['model']
        # Cache it
        _global_model_cache[jsonl_path] = model
        logging.debug(f"Loaded pre-computed global model from: {model_path}")
        return model
    except Exception as e:
        logging.warning(f"Failed to load global model: {e}")
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
        logging.error(f"Error getting screen position: {e}")
        return None


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


def get_screen_position_preset(preset_name: str, screen_width: int, screen_height: int) -> dict:
    """
    Get target screen coordinates for a preset.
    
    Args:
        preset_name: One of: "center", "center_top", "center_bottom", "sweet_spot", 
                    "top_left", "top_right", "bottom_left", "bottom_right"
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
    
    Returns:
        {"x": int, "y": int} - Target screen coordinates
    """
    presets = {
        "center": (0.5, 0.5),
        "center_top": (0.5, 0.25),
        "center_bottom": (0.5, 0.75),
        "sweet_spot": (0.5, 0.67),
        "top_left": (0.25, 0.25),
        "top_right": (0.75, 0.25),
        "bottom_left": (0.25, 0.75),
        "bottom_right": (0.75, 0.75),
    }
    
    if preset_name not in presets:
        logging.warning(f"Unknown preset '{preset_name}', using 'center'")
        preset_name = "center"
    
    x_ratio, y_ratio = presets[preset_name]
    return {
        "x": int(screen_width * x_ratio),
        "y": int(screen_height * y_ratio)
    }


def calculate_camera_movement_to_screen_position(
    object_world_coords: dict,  # {"x": int, "y": int}
    target_screen_coords: dict,  # {"x": int, "y": int} or {"x_ratio": float, "y_ratio": float}
    calibration_data_path: str,
    use_global_model: bool = True,
    use_zoom: bool = False,
    step_scale: float = 0.7,
    max_yaw_step: float = 128.0,
    max_pitch_step: float = 48.0,
    k: int = 500,
    max_dxdy: int = 3,
    pitch_min: int = None,  # Custom pitch minimum (defaults to PITCH_MIN)
    pitch_max: int = None,  # Custom pitch maximum (defaults to PITCH_MAX)
    preloaded_global_model: dict = None,  # Pre-loaded global model (avoids reloading)
    preloaded_rows: list = None,  # Pre-loaded calibration rows (avoids reloading)
    # LM solver parameters (for local fit only)
    max_lm_iterations: int = 5,  # Max inner LM iterations
    lm_tolerance: float = 10.0,  # Stop inner loop if error < this (pixels)
    lm_initial_lambda: float = 10.0,  # Initial damping parameter (lower = more aggressive)
    lm_lambda_reduction: float = 2.0,  # Lambda reduction factor on success (higher = faster reduction)
    lm_alpha_range: list = None,  # Step sizes to test [1.0, 0.5, ...] (None = default)
    lm_allow_overshoot: bool = False,  # Allow alpha > 1.0 (overshooting)
    # Jacobian computation method
    jacobian_method: str = "finite_diff",  # ONLY "finite_diff" is supported (no fallbacks)
    finite_diff_dyaw: float = 0.0,  # Yaw perturbation (0.0 = auto-adaptive based on error)
    finite_diff_dpitch: float = 0.0,  # Pitch perturbation (0.0 = auto-adaptive based on error)
    error_pixels: float = None  # Current error for adaptive perturbation sizing (optional)
) -> dict:
    """
    Calculate camera movement needed to position object at target screen position.
    
    Args:
        object_world_coords: World coordinates of object {"x": int, "y": int}
        target_screen_coords: Target screen coordinates. Can be:
            - {"x": int, "y": int} - Absolute pixel coordinates
            - {"x_ratio": float, "y_ratio": float} - Ratios (0.0-1.0)
        calibration_data_path: Path to calibration JSONL file
        use_global_model: Whether to use pre-computed global model (faster)
        use_zoom: Whether to solve for zoom changes too
        step_scale: Damping factor (0.0-1.0, lower = more conservative)
        max_yaw_step: Maximum yaw change per step
        max_pitch_step: Maximum pitch change per step
        k: Number of neighbors for local fit (if not using global model)
        max_dxdy: Maximum dx/dy difference for neighbors
    
    Returns:
        {
            "success": bool,
            "delta_yaw": float,
            "delta_pitch": float,
            "delta_zoom": float,
            "target_yaw": int,
            "target_pitch": int,
            "target_zoom": int,
            "error_pixels": float,
            "jacobian": np.array,
            "message": str
        }
    """
    from actions import player
    
    try:
        # Get player position
        px = player.get_x()
        py = player.get_y()
        if px is None or py is None:
            return {"success": False, "message": "Could not get player position"}
        
        ox = object_world_coords.get("x")
        oy = object_world_coords.get("y")
        if not isinstance(ox, int) or not isinstance(oy, int):
            return {"success": False, "message": "Invalid object world coordinates"}
        
        # Calculate dx, dy
        dx0 = ox - px
        dy0 = oy - py
        
        # Get current camera state
        camera_data = ipc.get_camera()
        if not camera_data:
            return {"success": False, "message": "Could not get camera data"}
        
        yaw0 = camera_data.get("yaw", 0)
        pitch0 = camera_data.get("pitch", 256)
        zoom0 = camera_data.get("scale", 512)
        
        # Get screen dimensions
        where = ipc.where() or {}
        screen_width = int(where.get("w", 0))
        screen_height = int(where.get("h", 0))
        if screen_width == 0 or screen_height == 0:
            return {"success": False, "message": "Could not get screen dimensions"}
        
        # Resolve target screen coordinates
        if "x_ratio" in target_screen_coords:
            # Ratios provided
            tx = int(screen_width * target_screen_coords["x_ratio"])
            ty = int(screen_height * target_screen_coords["y_ratio"])
        else:
            # Absolute coordinates provided
            tx = int(target_screen_coords.get("x", screen_width // 2))
            ty = int(target_screen_coords.get("y", screen_height // 2))
        
        # Get current screen position
        screen_pos = get_current_screen_position(ox, oy)
        if screen_pos is None:
            return {"success": False, "message": "Object is off-screen or projection failed"}
        
        sx0, sy0 = screen_pos
        error_x = sx0 - tx
        error_y = sy0 - ty
        error_pixels_pre = math.hypot(error_x, error_y)
        error_color_pre = _get_error_color(error_pixels_pre)
        
        # Debug: Print formatted table
        _print_table_header("CALCULATE CAMERA MOVEMENT")
        print(_format_table_row("Camera Target Tile", f"({ox}, {oy})"))
        print(_format_table_row("Player Position", f"({px}, {py})"))
        print(_format_table_row("Relative (dx, dy)", f"({dx0}, {dy0})"))
        _print_table_separator()
        print(_format_table_row("Starting Yaw", f"{yaw0}"))
        print(_format_table_row("Starting Pitch", f"{pitch0}"))
        print(_format_table_row("Starting Zoom", f"{zoom0}"))
        _print_table_separator()
        print(_format_table_row("Current Screen", f"({sx0:.1f}, {sy0:.1f})"))
        print(_format_table_row("Target Screen", f"({tx}, {ty})"))
        print(_format_table_row("Screen Error (X)", f"{error_x:+.1f} px", error_color_pre))
        print(_format_table_row("Screen Error (Y)", f"{error_y:+.1f} px", error_color_pre))
        print(_format_table_row("Error Distance", f"{error_pixels_pre:.1f} px", error_color_pre))
        
        # Warn for large errors - Jacobian linearization may be inaccurate
        if error_pixels_pre > 200:
            print(f"\n{COLOR_YELLOW}{COLOR_BOLD}⚠ WARNING: Large error detected ({error_pixels_pre:.1f}px){COLOR_RESET}")
            print(f"{COLOR_YELLOW}  The Jacobian is a local linearization and may be inaccurate for large movements.{COLOR_RESET}")
            print(f"{COLOR_YELLOW}  Consider using iterative refinement or local fit method for better accuracy.{COLOR_RESET}\n")
        
        # Use custom pitch bounds if provided, otherwise use defaults
        effective_pitch_min = pitch_min if pitch_min is not None else PITCH_MIN
        effective_pitch_max = pitch_max if pitch_max is not None else PITCH_MAX
        
        # Initialize variables
        J = None
        bx = None
        by = None
        neigh = None
        global_model = None
        jacobian_computed = False
        
        # Finite-difference Jacobian - REQUIRED, NO FALLBACKS
        if jacobian_method != "finite_diff":
            return {"success": False, "message": f"Only finite_diff method is supported. Got: {jacobian_method}"}
        
        # Load global model if needed
        if preloaded_global_model is not None:
            global_model = preloaded_global_model
        else:
            global_model = load_global_model(calibration_data_path)
        
        if global_model is None:
            return {"success": False, "message": "Finite-diff requires global model. Please build a global model first."}
        
        # Create oracle projector function
        def oracle_projector(yaw, pitch, zoom, player_pos, obj_pos):
            return project_screen_xy_oracle(
                yaw, pitch, zoom, player_pos, obj_pos,
                global_model=global_model,
                use_zoom=use_zoom
            )
        
        # Compute finite-difference Jacobian
        player_pos = {"x": px, "y": py}
        obj_pos = {"x": ox, "y": oy}
        
        result = jacobian_finite_diff(
            oracle_projector, yaw0, pitch0, zoom0,
            player_pos, obj_pos,
            dyaw=finite_diff_dyaw if finite_diff_dyaw > 0 else None,  # None = auto-adaptive
            dpitch=finite_diff_dpitch if finite_diff_dpitch > 0 else None,  # None = auto-adaptive
            use_zoom=use_zoom,
            error_pixels=error_pixels_pre,  # Pass error for adaptive sizing
            use_central_diff=True  # Use central differences for better accuracy
        )
        
        if result is None:
            return {"success": False, "message": "Finite-diff Jacobian computation failed. Check calibration data and global model."}
        
        (sx0_oracle, sy0_oracle), J = result
        
        # DEBUG: Compare real vs oracle screen positions
        sx0_real = sx0  # Save real position before overwrite
        sy0_real = sy0
        oracle_error = math.hypot(sx0_oracle - sx0_real, sy0_oracle - sy0_real)
        if oracle_error > 50:  # Significant mismatch
            logging.warning(f"[JACOBIAN] Oracle vs Real screen position mismatch: "
                          f"Real=({sx0_real:.1f}, {sy0_real:.1f}), "
                          f"Oracle=({sx0_oracle:.1f}, {sy0_oracle:.1f}), "
                          f"Difference={oracle_error:.1f}px")
            print(f"\n{COLOR_RED}{COLOR_BOLD}⚠ ORACLE MISMATCH DETECTED:{COLOR_RESET}")
            print(f"  Real screen position:    ({sx0_real:.1f}, {sy0_real:.1f})")
            print(f"  Oracle prediction:       ({sx0_oracle:.1f}, {sy0_oracle:.1f})")
            print(f"  Difference:              {oracle_error:.1f}px")
            print(f"  This may indicate model inaccuracy or coordinate system mismatch.{COLOR_RESET}\n")
        
        # Update screen position and error with oracle's prediction
        # NOTE: We use oracle's prediction because the Jacobian was computed using it
        # If oracle is wrong, the Jacobian will be wrong, leading to incorrect movement
        sx0 = sx0_oracle
        sy0 = sy0_oracle
        error_x = sx0 - tx
        error_y = sy0 - ty
        error_pixels_pre = math.hypot(error_x, error_y)
        jacobian_computed = True
        
        # Determine actual perturbations used (may be adaptive)
        actual_dyaw = finite_diff_dyaw if finite_diff_dyaw > 0 else "auto"
        actual_dpitch = finite_diff_dpitch if finite_diff_dpitch > 0 else "auto"
        logging.info(f"[JACOBIAN] Using finite-difference method (dyaw={actual_dyaw}, dpitch={actual_dpitch}, error={error_pixels_pre:.1f}px)")
        if error_pixels_pre > 200:
            print(f"\n{COLOR_BLUE}Finite-Diff Jacobian:{COLOR_RESET}")
            print(f"  Error: {error_pixels_pre:.1f}px")
            print(f"  Perturbations: dyaw={actual_dyaw}, dpitch={actual_dpitch}")
            print(f"  J = [[{J[0,0]:+.4f}, {J[0,1]:+.4f}],")
            print(f"       [{J[1,0]:+.4f}, {J[1,1]:+.4f}]]")
        
        # Finite-diff path: solve using the computed Jacobian
        if not jacobian_computed or J is None:
            return {"success": False, "message": "Failed to compute finite-diff Jacobian"}
        
        # Solve using the pre-computed Jacobian with bounds
        e = np.array([sx0 - tx, sy0 - ty], dtype=float)
        
        # Debug: Print Jacobian signs for troubleshooting
        if error_pixels_pre > 200:
            print(f"\n{COLOR_BLUE}Jacobian Debug:{COLOR_RESET}")
            print(f"  Method: finite_diff")
            print(f"  Error vector: e = [{e[0]:+.1f}, {e[1]:+.1f}]")
            print(f"  Jacobian J = [[{J[0,0]:+.4f}, {J[0,1]:+.4f}],")
            print(f"                [{J[1,0]:+.4f}, {J[1,1]:+.4f}]]")
            print(f"  J[0,0] (dx/dyaw): {J[0,0]:+.4f} - how screen X changes when yaw increases")
            print(f"  J[1,0] (dy/dyaw): {J[1,0]:+.4f} - how screen Y changes when yaw increases")
        
        # Solve using the pre-computed Jacobian with bounds
        rhs = -e
        if use_zoom:
            bounds_low = np.array([-np.inf, effective_pitch_min - pitch0, ZOOM_MIN - zoom0])
            bounds_high = np.array([np.inf, effective_pitch_max - pitch0, ZOOM_MAX - zoom0])
        else:
            bounds_low = np.array([-np.inf, effective_pitch_min - pitch0])
            bounds_high = np.array([np.inf, effective_pitch_max - pitch0])
        
        result = lsq_linear(J, rhs, bounds=(bounds_low, bounds_high), method='trf')
        delta_raw = result.x
        
        # Debug: Check for potential sign errors and yaw inversion
        if error_pixels_pre > 200:
            print(f"  Solution: delta_yaw = {delta_raw[0]:+.2f}, delta_pitch = {delta_raw[1]:+.2f}")
            print(f"  Error vector: e = [{e[0]:+.1f}, {e[1]:+.1f}]")
            print(f"  Expected: if object is RIGHT of target (e[0] > 0), need to move LEFT (negative yaw)")
            print(f"            if object is LEFT of target (e[0] < 0), need to move RIGHT (positive yaw)")
            print(f"            if dx/dyaw > 0 (yaw+ moves object RIGHT), then delta_yaw should be NEGATIVE when e[0] > 0")
            print(f"            if dx/dyaw < 0 (yaw+ moves object LEFT), then delta_yaw should be POSITIVE when e[0] > 0")
            print(f"  Actual:   delta_yaw = {delta_raw[0]:+.2f}, dx/dyaw = {J[0,0]:+.4f}, e[0] = {e[0]:+.1f}")
            
            # Check if yaw sign is inverted
            # If object is LEFT of target (e[0] < 0), we need positive yaw to move RIGHT
            # If object is RIGHT of target (e[0] > 0), we need negative yaw to move LEFT
            yaw_sign_correct = True
            if e[0] < 0:  # Object LEFT of target
                if J[0,0] > 0 and delta_raw[0] < 0:  # yaw+ moves RIGHT, but we're moving LEFT
                    yaw_sign_correct = False
                elif J[0,0] < 0 and delta_raw[0] > 0:  # yaw+ moves LEFT, but we're moving RIGHT
                    yaw_sign_correct = False
            elif e[0] > 0:  # Object RIGHT of target
                if J[0,0] > 0 and delta_raw[0] > 0:  # yaw+ moves RIGHT, but we're moving RIGHT (wrong!)
                    yaw_sign_correct = False
                elif J[0,0] < 0 and delta_raw[0] < 0:  # yaw+ moves LEFT, but we're moving LEFT (wrong!)
                    yaw_sign_correct = False
            
            if not yaw_sign_correct:
                print(f"\n{COLOR_RED}{COLOR_BOLD}⚠ YAW SIGN INVERTED DETECTED!{COLOR_RESET}")
                print(f"  Inverting yaw sign: {delta_raw[0]:+.2f} -> {-delta_raw[0]:+.2f}")
                delta_raw[0] = -delta_raw[0]  # Flip yaw sign
                print(f"{COLOR_RESET}")
            elif (e[0] > 0 and J[0,0] > 0 and delta_raw[0] > 0) or (e[0] < 0 and J[0,0] < 0 and delta_raw[0] < 0):
                print(f"\n{COLOR_YELLOW}{COLOR_BOLD}⚠ POTENTIAL SIGN ERROR: Direction may be inverted!{COLOR_RESET}\n")
        
        # For large errors, validate the solution by predicting the result
        # Also check if yaw sign needs to be inverted
        if error_pixels_pre > 200:
            try:
                # Calculate predicted camera state after movement
                if use_zoom:
                    pred_yaw = (yaw0 + delta_raw[0]) % YAW_MOD
                    pred_pitch = max(effective_pitch_min, min(effective_pitch_max, pitch0 + delta_raw[1]))
                    pred_zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom0 + delta_raw[2]))
                else:
                    pred_yaw = (yaw0 + delta_raw[0]) % YAW_MOD
                    pred_pitch = max(effective_pitch_min, min(effective_pitch_max, pitch0 + delta_raw[1]))
                    pred_zoom = zoom0
                
                # Predict screen position with new camera state
                pred_sx, pred_sy = predict_screen_from_model(global_model, dx0, dy0, pred_yaw, pred_pitch, pred_zoom, use_zoom)
                pred_error = math.hypot(pred_sx - tx, pred_sy - ty)
                
                # If predicted error is worse or still very large, try inverting yaw
                if pred_error > error_pixels_pre * 0.5 or (pred_error > 500 and abs(delta_raw[0]) > 100):
                    # Try with inverted yaw
                    if use_zoom:
                        pred_yaw_inv = (yaw0 - delta_raw[0]) % YAW_MOD
                        pred_pitch_inv = max(effective_pitch_min, min(effective_pitch_max, pitch0 + delta_raw[1]))
                        pred_zoom_inv = max(ZOOM_MIN, min(ZOOM_MAX, zoom0 + delta_raw[2]))
                    else:
                        pred_yaw_inv = (yaw0 - delta_raw[0]) % YAW_MOD
                        pred_pitch_inv = max(effective_pitch_min, min(effective_pitch_max, pitch0 + delta_raw[1]))
                        pred_zoom_inv = zoom0
                    
                    pred_sx_inv, pred_sy_inv = predict_screen_from_model(global_model, dx0, dy0, pred_yaw_inv, pred_pitch_inv, pred_zoom_inv, use_zoom)
                    pred_error_inv = math.hypot(pred_sx_inv - tx, pred_sy_inv - ty)
                    
                    # If inverted yaw gives better prediction, use it
                    if pred_error_inv < pred_error:
                        print(f"\n{COLOR_GREEN}{COLOR_BOLD}✓ YAW SIGN INVERTED (better prediction):{COLOR_RESET}")
                        print(f"  Original:  delta_yaw={delta_raw[0]:+.2f}, pred_error={pred_error:.1f}px")
                        print(f"  Inverted:  delta_yaw={-delta_raw[0]:+.2f}, pred_error={pred_error_inv:.1f}px")
                        print(f"{COLOR_RESET}")
                        delta_raw[0] = -delta_raw[0]  # Flip yaw sign
                        pred_error = pred_error_inv
                        pred_sx, pred_sy = pred_sx_inv, pred_sy_inv
                
                # If predicted error is still large, warn
                if pred_error > error_pixels_pre * 0.5:  # Less than 50% improvement
                    print(f"\n{COLOR_RED}{COLOR_BOLD}⚠ WARNING: Predicted error after movement: {pred_error:.1f}px{COLOR_RESET}")
                    print(f"{COLOR_RED}  Jacobian linearization may be inaccurate for this large movement.{COLOR_RESET}")
                    print(f"{COLOR_RED}  Consider using iterative refinement or reducing step_scale.{COLOR_RESET}\n")
            except Exception as e:
                logging.debug(f"Could not validate Jacobian prediction: {e}")
        
        # Extract deltas from solution
        if use_zoom:
            dyaw_raw = float(delta_raw[0])
            dpitch_raw = float(delta_raw[1])
            dzoom_raw = float(delta_raw[2])
        else:
            dyaw_raw = float(delta_raw[0])
            dpitch_raw = float(delta_raw[1])
            dzoom_raw = 0.0
        
        # Apply step_scale damping (finite_diff always uses direct solve)
        delta_damped = np.array([dyaw_raw, dpitch_raw, dzoom_raw if use_zoom else 0.0]) * float(step_scale)
        if use_zoom:
            dyaw_damped = float(delta_damped[0])
            dpitch_damped = float(delta_damped[1])
            dzoom_damped = float(delta_damped[2])
        else:
            dyaw_damped = float(delta_damped[0])
            dpitch_damped = float(delta_damped[1])
            dzoom_damped = 0.0
        
        # Clamp by max step size
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))
        
        if use_zoom:
            dyaw = float(clamp(dyaw_damped, -max_yaw_step, max_yaw_step))
            dpitch = float(clamp(dpitch_damped, -max_pitch_step, max_pitch_step))
            dzoom = float(clamp(dzoom_damped, -512.0, 512.0))
        else:
            dyaw = float(clamp(dyaw_damped, -max_yaw_step, max_yaw_step))
            dpitch = float(clamp(dpitch_damped, -max_pitch_step, max_pitch_step))
            dzoom = 0.0
        
        # Use custom pitch bounds if provided, otherwise use defaults
        effective_pitch_min = pitch_min if pitch_min is not None else PITCH_MIN
        effective_pitch_max = pitch_max if pitch_max is not None else PITCH_MAX
        
        # Enforce absolute pitch bounds at the delta level
        dpitch = float(clamp(dpitch, effective_pitch_min - pitch0, effective_pitch_max - pitch0))
        
        # Enforce absolute zoom bounds at the delta level (if using zoom)
        if use_zoom:
            dzoom = float(clamp(dzoom, ZOOM_MIN - zoom0, ZOOM_MAX - zoom0))
        
        # Convert to absolute suggestions
        # For yaw, ensure we use the shortest wrapped path
        # Calculate raw target first
        yaw_target_raw = (yaw0 + round(dyaw)) % YAW_MOD
        # Find the shortest wrapped delta from current yaw to raw target
        # This ensures we always take the shortest path (e.g., if going from 0 to 1900,
        # we should go -148 instead of +1900)
        dyaw_wrapped = wrap_yaw_diff(yaw_target_raw, yaw0, mod=YAW_MOD)
        # Calculate final target using the wrapped delta
        yaw_target = int((yaw0 + dyaw_wrapped) % YAW_MOD)
        pitch_target = int(clamp(pitch0 + round(dpitch), effective_pitch_min, effective_pitch_max))
        zoom_target = int(clamp(zoom0 + round(dzoom), ZOOM_MIN, ZOOM_MAX))
        
        # Calculate error
        error_pixels = math.hypot(e[0], e[1])
        error_color = _get_error_color(error_pixels)
        
        # Debug: Print calculated movement table
        print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}Calculated Movement:{COLOR_RESET}")
        print(_format_table_row("Delta Yaw", f"{dyaw_wrapped:+.2f} (raw: {dyaw_raw:+.2f}, clamped: {dyaw:+.2f})"))
        print(_format_table_row("Delta Pitch", f"{dpitch:+.2f} (raw: {dpitch_raw:+.2f})"))
        print(_format_table_row("Delta Zoom", f"{dzoom:+.2f} (raw: {dzoom_raw:+.2f})"))
        
        print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}Target Camera State:{COLOR_RESET}")
        print(_format_table_row("Target Yaw", f"{yaw_target} (from {yaw0}, wrapped delta: {dyaw_wrapped:+.2f})"))
        print(_format_table_row("Target Pitch", f"{pitch_target} (from {pitch0}, min={effective_pitch_min})"))
        print(_format_table_row("Target Zoom", f"{zoom_target} (from {zoom0})"))
        
        print(f"\n{_format_table_row('Final Error', f'{error_pixels:.1f} pixels', error_color)}")
        print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}\n")
        
        return {
            "success": True,
            "delta_yaw": dyaw_wrapped,  # Use wrapped delta (shortest path)
            "delta_pitch": dpitch,
            "delta_zoom": dzoom,
            "target_yaw": yaw_target,
            "target_pitch": pitch_target,
            "target_zoom": zoom_target,
            "error_pixels": error_pixels,
            "jacobian": J,
            "current_screen": (sx0, sy0),
            "target_screen": (tx, ty),
            "calc_starting_yaw": yaw0,  # Store starting state for validation
            "calc_starting_pitch": pitch0,
            "calc_starting_zoom": zoom0,
            "message": "Success"
        }
        
    except Exception as e:
        logging.error(f"Error calculating camera movement: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


def aim_camera_at_target_iterative(
    world_coords: dict,
    target_screen: dict,
    calibration_data_path: str = None,
    max_iterations: int = 5,
    error_threshold: float = 20.0,
    max_initial_error: float = 500.0,
    use_global_model: bool = True,
    use_zoom: bool = False,
    pitch_min: int = None,
    pitch_max: int = None,
    step_scale: float = 1.0,
    max_yaw_step: float = 9999.0,
    max_pitch_step: float = 9999.0,
    fast_mode: bool = True,
    min_wait_time: float = 0.1,
    manual_mode: bool = True,
    prompt_callback: callable = None,
    smooth_steps: int = 0,  # Number of intermediate steps for smooth movement (0 = disabled)
    smooth_all_iterations: bool = False,  # If True, apply smooth to all iterations; if False, only first
    # LM solver parameters (for local fit only)
    max_lm_iterations: int = 10,  # Max inner LM iterations (increased from 5)
    lm_tolerance: float = 5.0,  # Stop inner loop if error < this (lowered from 10.0)
    lm_initial_lambda: float = 5.0,  # Initial damping (lowered from 10.0 for more aggressive start)
    lm_lambda_reduction: float = 2.5,  # Lambda reduction factor (increased from 2.0 for faster reduction)
    lm_alpha_range: list = None,  # Step sizes to test (None = default)
    lm_allow_overshoot: bool = False,  # Allow alpha > 1.0
    # Jacobian computation method
    jacobian_method: str = "finite_diff",  # ONLY "finite_diff" is supported (no fallbacks)
    finite_diff_dyaw: float = 0.0,  # Yaw perturbation (0.0 = auto-adaptive based on error)
    finite_diff_dpitch: float = 0.0,  # Pitch perturbation (0.0 = auto-adaptive based on error)
    error_pixels: float = None  # Current error for adaptive perturbation sizing (optional)
) -> dict:
    """
    Iteratively refine camera position using Jacobian until error is small enough.
    
    This automatically does multiple smaller movements instead of one large movement,
    which works much better because the Jacobian is a local linearization.
    
    Args:
        world_coords: World coordinates {"x": int, "y": int}
        target_screen: Target screen position {"x": int, "y": int} or preset name
        calibration_data_path: Path to calibration JSONL file
        max_iterations: Maximum number of iterations to try
        error_threshold: Stop when error is below this (pixels)
        max_initial_error: If initial error > this, use smaller step_scale
        use_global_model: Whether to use global model (faster) or local fit
        use_zoom: Whether to solve for zoom changes
        pitch_min: Minimum pitch value
        pitch_max: Maximum pitch value
        step_scale: Scaling factor for movements (1.0 = full, 0.5 = half)
        max_yaw_step: Maximum yaw step size (unused if step_scale handles it)
        max_pitch_step: Maximum pitch step size (unused if step_scale handles it)
    
    Returns:
        Dictionary with success status, final error, and iteration info
    """
    from helpers.runtime_utils import get_ipc
    ipc = get_ipc()
    if not ipc:
        return {"success": False, "message": "IPC not available"}
    
    world_x = world_coords.get("x")
    world_y = world_coords.get("y")
    if not isinstance(world_x, int) or not isinstance(world_y, int):
        return {"success": False, "message": "Invalid world coordinates"}
    
    # Get target screen position
    if isinstance(target_screen, str):
        where = ipc.where() or {}
        screen_width = int(where.get("w", 0))
        screen_height = int(where.get("h", 0))
        if screen_width == 0 or screen_height == 0:
            return {"success": False, "message": "Could not get screen dimensions"}
        target_screen = get_screen_position_preset(target_screen, screen_width, screen_height)
    
    target_x = target_screen.get("x")
    target_y = target_screen.get("y")
    if target_x is None or target_y is None:
        return {"success": False, "message": "Invalid target screen position"}
    
    iterations = []
    current_step_scale = step_scale
    
    # Load model/data once at the start (not each iteration) for performance
    global_model_loaded = None
    rows_loaded = None
    if use_global_model:
        global_model_loaded = load_global_model(calibration_data_path)
        if global_model_loaded:
            logging.debug(f"[ITERATIVE] Loaded global model once (will reuse for all iterations)")
    else:
        # Load calibration data once for local fit
        rows_loaded = load_rows(calibration_data_path)
        if rows_loaded:
            logging.debug(f"[ITERATIVE] Loaded {len(rows_loaded)} calibration rows once (will reuse for all iterations)")
    
    for iteration in range(max_iterations):
        # Get current screen position
        proj = ipc.project_world_tile(world_x, world_y) or {}
        if not proj.get("ok"):
            return {
                "success": False,
                "message": f"Target is off-screen at iteration {iteration + 1}",
                "iterations": iterations
            }
        
        current_screen = proj.get("canvas", {})
        current_x = float(current_screen.get("x", 0))
        current_y = float(current_screen.get("y", 0))
        
        # Calculate current error
        error_x = current_x - target_x
        error_y = current_y - target_y
        error_pixels = math.hypot(error_x, error_y)
        
        # Check if we're done
        if error_pixels <= error_threshold:
            return {
                "success": True,
                "message": f"Target reached in {iteration + 1} iterations",
                "final_error": error_pixels,
                "iterations": iterations
            }
        
        # Adaptive step sizing based on error magnitude
        # For large errors, use larger steps (but still safe)
        # For small errors, use smaller steps for precision
        if iteration == 0:
            if error_pixels > max_initial_error:
                # Very large error - use conservative step
                current_step_scale = min(step_scale, max_initial_error / error_pixels * step_scale)
                logging.info(f"[ITERATIVE] Large initial error ({error_pixels:.1f}px), using step_scale={current_step_scale:.2f}")
            elif error_pixels > 200:
                # Large error - use full step scale
                current_step_scale = step_scale
            elif error_pixels > 100:
                # Medium error - use slightly reduced step for accuracy
                current_step_scale = step_scale * 0.9
            else:
                # Small error - use smaller steps for precision
                current_step_scale = step_scale * 0.7
        else:
            # After first iteration, adapt based on previous improvement
            if iterations and iterations[-1].get("improvement", 0) > 50:
                # Good progress - can use larger steps
                current_step_scale = min(step_scale, current_step_scale * 1.1)
            elif iterations and iterations[-1].get("improvement", 0) < 10:
                # Poor progress - reduce step size
                current_step_scale = max(0.3, current_step_scale * 0.8)
        
        # Calculate movement (reuse pre-loaded model/data for speed)
        movement_result = calculate_camera_movement_to_screen_position(
            object_world_coords=world_coords,
            target_screen_coords=target_screen,
            calibration_data_path=calibration_data_path,
            use_global_model=use_global_model,
            use_zoom=use_zoom,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            step_scale=current_step_scale,
            max_yaw_step=max_yaw_step,
            max_pitch_step=max_pitch_step,
            preloaded_global_model=global_model_loaded,  # Reuse pre-loaded model
            preloaded_rows=rows_loaded,  # Reuse pre-loaded rows
            # LM solver parameters
            max_lm_iterations=max_lm_iterations,
            lm_tolerance=lm_tolerance,
            lm_initial_lambda=lm_initial_lambda,
            lm_lambda_reduction=lm_lambda_reduction,
            lm_alpha_range=lm_alpha_range,
            lm_allow_overshoot=lm_allow_overshoot,
            # Jacobian method
            jacobian_method=jacobian_method,
            finite_diff_dyaw=finite_diff_dyaw,
            finite_diff_dpitch=finite_diff_dpitch,
            error_pixels=error_pixels  # Pass current error for adaptive perturbations
        )
        
        if not movement_result.get("success"):
            return {
                "success": False,
                "message": f"Movement calculation failed at iteration {iteration + 1}: {movement_result.get('message')}",
                "iterations": iterations
            }
        
        # Execute movement with smooth interpolation if enabled
        # Apply smooth if: (1) smooth_steps > 0 AND (2) first iteration OR smooth_all_iterations is True
        use_smooth = smooth_steps > 0 and (iteration == 0 or smooth_all_iterations)
        if use_smooth:
            # For smooth movement, use a single movement with longer duration
            # This creates smoother motion than multiple queued movements
            logging.info(f"[SMOOTH] Using smooth movement with {smooth_steps} steps (iteration {iteration + 1})")
            camera_data = ipc.get_camera()
            yaw_start = camera_data.get("yaw", 0)
            pitch_start = camera_data.get("pitch", 256)
            zoom_start = camera_data.get("scale", 512)
            
            yaw_target = movement_result.get("target_yaw", yaw_start)
            pitch_target = movement_result.get("target_pitch", pitch_start)
            zoom_target = movement_result.get("target_zoom", zoom_start)
            
            yaw_diff = wrap_yaw_diff(yaw_target, yaw_start, mod=YAW_MOD)
            pitch_diff = pitch_target - pitch_start
            zoom_diff = zoom_target - zoom_start
            
            # Calculate base duration, then extend it for smoothness
            # Smooth factor: multiply duration by smooth_steps to make it slower/smoother
            smooth_factor = 1.0 + (smooth_steps * 0.3)  # 1.0 -> 3.4 for 8 steps
            
            # Queue smooth movements with extended duration
            if abs(yaw_diff) > 2:
                yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
                yaw_hold_ms = int(abs(yaw_diff) * YAW_MS_PER_UNIT * smooth_factor)
                yaw_hold_ms = min(yaw_hold_ms, 3000)  # Max 3 seconds for smooth movement
                if yaw_hold_ms > 0:
                    _camera_movement_queue.put({
                        "type": "key_hold",
                        "key": yaw_key,
                        "duration_ms": yaw_hold_ms,
                        "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                    })
            
            if abs(pitch_diff) > 2:
                pitch_key = "UP" if pitch_diff > 0 else "DOWN"
                pitch_hold_ms = int(abs(pitch_diff) * PITCH_MS_PER_UNIT * smooth_factor)
                pitch_hold_ms = min(pitch_hold_ms, 3000)  # Max 3 seconds
                if pitch_hold_ms > 0:
                    _camera_movement_queue.put({
                        "type": "key_hold",
                        "key": pitch_key,
                        "duration_ms": pitch_hold_ms,
                        "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
                    })
            
            if use_zoom and abs(zoom_diff) > 10:
                scroll_amount = 1 if zoom_diff > 0 else -1
                scroll_count = calculate_zoom_scroll_count(zoom_diff, zoom_start)
                if scroll_count > 0:
                    # For zoom, spread scrolls over time for smoothness
                    import time
                    for i in range(scroll_count):
                        _camera_movement_queue.put({
                            "type": "scroll",
                            "amount": scroll_amount,
                            "count": 1
                        })
                        if i < scroll_count - 1:
                            time.sleep(0.05)  # 50ms between scrolls
            
            # Wait for smooth movement to complete (longer wait for smooth movement)
            import time
            max_duration = max(
                int(abs(yaw_diff) * YAW_MS_PER_UNIT * smooth_factor) if abs(yaw_diff) > 2 else 0,
                int(abs(pitch_diff) * PITCH_MS_PER_UNIT * smooth_factor) if abs(pitch_diff) > 2 else 0
            )
            wait_time = (max_duration / 1000.0) + 0.2  # Add 200ms buffer
            logging.info(f"[SMOOTH] Smooth movement queued, waiting {wait_time:.2f}s for completion")
            time.sleep(wait_time)
            execute_success = True
        elif fast_mode:
            # Turbo mode: don't wait for stability, just queue the movement
            wait_for_stable = False
            execute_success = execute_jacobian_camera_movement(
                movement_result, 
                wait_for_stable=False,
                max_wait_time=0.0  # No wait time
            )
            # Small delay to let the movement execute and settle slightly
            import time
            time.sleep(min_wait_time)  # Use the min_wait_time parameter (default 0.1s)
        else:
            wait_for_stable = True
            execute_success = execute_jacobian_camera_movement(
                movement_result, 
                wait_for_stable=True,
                max_wait_time=3.0
            )
        
        if not execute_success:
            return {
                "success": False,
                "message": f"Movement execution failed at iteration {iteration + 1}",
                "iterations": iterations
            }
        
        # Wait for camera to stabilize before getting "after" coordinates
        # This ensures we get accurate screen position after movement completes
        import time
        if not use_smooth:  # Smooth mode already waited
            if fast_mode:
                # In fast mode, wait a bit longer to ensure camera has moved
                time.sleep(max(min_wait_time, 0.15))  # At least 150ms
            else:
                # In normal mode, wait_for_stable was already called, but ensure it's done
                # Wait a bit more to ensure camera is fully settled
                time.sleep(0.1)
        
        # Wait for camera stability (check yaw/pitch values)
        camera_stable = False
        stable_checks = 0
        max_stable_checks = 20  # Max 2 seconds (20 * 0.1s)
        last_yaw = None
        last_pitch = None
        
        for check in range(max_stable_checks):
            camera_data = ipc.get_camera()
            if camera_data:
                current_yaw = camera_data.get("yaw", 0)
                current_pitch = camera_data.get("pitch", 256)
                
                # Check if yaw and pitch have stabilized (same value twice in a row)
                if last_yaw == current_yaw and last_pitch == current_pitch:
                    stable_checks += 1
                    if stable_checks >= 2:  # Stable for 2 consecutive checks
                        camera_stable = True
                        break
                else:
                    stable_checks = 0
                
                last_yaw = current_yaw
                last_pitch = current_pitch
            
            time.sleep(0.1)  # Check every 100ms
        
        # Get new screen position after movement (camera should be stable now)
        proj_after = ipc.project_world_tile(world_x, world_y) or {}
        if not proj_after.get("ok"):
            return {
                "success": False,
                "message": f"Target went off-screen after iteration {iteration + 1}",
                "iterations": iterations
            }
        
        screen_after = proj_after.get("canvas", {})
        after_x = float(screen_after.get("x", 0))
        after_y = float(screen_after.get("y", 0))
        error_after = math.hypot(after_x - target_x, after_y - target_y)
        
        # Record iteration info
        iterations.append({
            "iteration": iteration + 1,
            "error_before": error_pixels,
            "error_after": error_after,
            "improvement": error_pixels - error_after,
            "step_scale": current_step_scale
        })
        
        # In manual mode, show detailed debug info
        if manual_mode:
            # Use current_x and current_y from start of iteration as "before" values
            before_x = current_x
            before_y = current_y
            
            # Print formatted debug table
            print(f"\n{COLOR_CYAN}{COLOR_BOLD}{'='*70}")
            print(f"  ITERATION {iteration + 1} - MANUAL MODE")
            print(f"{'='*70}{COLOR_RESET}")
            
            print(f"\n{COLOR_BLUE}Screen Coordinates:{COLOR_RESET}")
            print(f"  {COLOR_GREEN}Target:{COLOR_RESET}     ({target_x:7.1f}, {target_y:7.1f})")
            print(f"  {COLOR_YELLOW}Before:{COLOR_RESET}    ({before_x:7.1f}, {before_y:7.1f})")
            print(f"  {COLOR_MAGENTA}After:{COLOR_RESET}     ({after_x:7.1f}, {after_y:7.1f})")
            
            # Calculate movement in screen space
            screen_delta_x = after_x - before_x
            screen_delta_y = after_y - before_y
            print(f"  {COLOR_BLUE}Movement:{COLOR_RESET}   ({screen_delta_x:+.1f}, {screen_delta_y:+.1f}) px")
            
            print(f"\n{COLOR_BLUE}Error Analysis:{COLOR_RESET}")
            error_color_before = _get_error_color(error_pixels)
            error_color_after = _get_error_color(error_after)
            print(f"  {COLOR_YELLOW}Error Before:{COLOR_RESET} {error_color_before}{error_pixels:7.1f} px{COLOR_RESET}")
            print(f"  {COLOR_MAGENTA}Error After:{COLOR_RESET}  {error_color_after}{error_after:7.1f} px{COLOR_RESET}")
            
            improvement = error_pixels - error_after
            improvement_pct = (improvement / error_pixels * 100) if error_pixels > 0 else 0
            improvement_color = COLOR_GREEN if improvement > 0 else COLOR_RED
            print(f"  {COLOR_BLUE}Improvement:{COLOR_RESET}   {improvement_color}{improvement:+.1f} px ({improvement_pct:+.1f}%){COLOR_RESET}")
            
            print(f"\n{COLOR_BLUE}Camera Movement:{COLOR_RESET}")
            delta_yaw = movement_result.get("delta_yaw", 0)
            delta_pitch = movement_result.get("delta_pitch", 0)
            print(f"  ΔYaw:   {delta_yaw:+.2f}")
            print(f"  ΔPitch: {delta_pitch:+.2f}")
            
            print(f"\n{COLOR_BLUE}Status:{COLOR_RESET}")
            if error_after <= error_threshold:
                print(f"  {COLOR_GREEN}✓ Target reached! (error < {error_threshold}px){COLOR_RESET}")
            else:
                print(f"  {COLOR_YELLOW}→ Continuing... (target: <{error_threshold}px){COLOR_RESET}")
            
            print(f"\n{COLOR_CYAN}{'='*70}{COLOR_RESET}\n")
            
            if prompt_callback:
                user_input = prompt_callback(iteration + 1, error_after, error_threshold)
            else:
                print("Press ENTER to continue, 'S' to skip remaining iterations, or 'Q' to quit...")
                user_input = input().strip().upper()
            
            if user_input == 'Q':
                return {
                    "success": False,
                    "message": f"Stopped by user at iteration {iteration + 1}",
                    "final_error": error_after,
                    "iterations": iterations
                }
            elif user_input == 'S':
                # Skip remaining iterations but return current result
                return {
                    "success": error_after <= error_threshold,
                    "message": f"Stopped early at iteration {iteration + 1} (error: {error_after:.1f}px)",
                    "final_error": error_after,
                    "iterations": iterations
                }
        
        # Check if we're making progress
        if error_after >= error_pixels * 0.95:  # Less than 5% improvement
            logging.warning(f"[ITERATIVE] Iteration {iteration + 1} made little progress ({error_pixels:.1f} -> {error_after:.1f}px)")
            # Reduce step scale for next iteration
            current_step_scale *= 0.7
            if current_step_scale < 0.1:
                return {
                    "success": False,
                    "message": f"Convergence failed - no progress after {iteration + 1} iterations",
                    "iterations": iterations
                }
    
    # Max iterations reached
    return {
        "success": False,
        "message": f"Max iterations ({max_iterations}) reached",
        "final_error": error_after,
        "iterations": iterations
    }


def execute_jacobian_camera_movement(
    movement_result: dict,
    wait_for_stable: bool = True,
    max_wait_time: float = 3.0
) -> bool:
    """
    Execute calculated camera movement using existing thread queue.
    
    Args:
        movement_result: Result from calculate_camera_movement_to_screen_position()
        wait_for_stable: Whether to wait for camera to stabilize after movement
        max_wait_time: Maximum time to wait for movement completion
    
    Returns:
        True if movement was executed successfully, False otherwise
    """
    if not movement_result.get("success"):
        logging.warning(f"Movement result indicates failure: {movement_result.get('message')}")
        return False
    
    try:
        # Ensure camera thread is running
        _ensure_camera_thread()
        
        # Get current camera state
        camera_data = ipc.get_camera()
        if not camera_data:
            return False
        
        yaw0 = camera_data.get("yaw", 0)
        pitch0 = camera_data.get("pitch", 256)
        zoom0 = camera_data.get("scale", 512)
        
        # Get target values from calculation result
        yaw_target = movement_result.get("target_yaw", yaw0)
        pitch_target = movement_result.get("target_pitch", pitch0)
        zoom_target = movement_result.get("target_zoom", zoom0)
        
        # Check if camera state changed significantly since calculation
        # (This can happen if there's a delay or previous camera movement)
        calc_yaw = movement_result.get("calc_starting_yaw", None)
        calc_pitch = movement_result.get("calc_starting_pitch", None)
        if calc_yaw is not None and calc_pitch is not None:
            yaw_change = abs(wrap_yaw_diff(yaw0, calc_yaw, mod=YAW_MOD))
            pitch_change = abs(pitch0 - calc_pitch)
            if yaw_change > 50 or pitch_change > 20:
                print(f"\n{COLOR_YELLOW}{COLOR_BOLD}⚠ WARNING: Camera state changed since calculation{COLOR_RESET}")
                print(f"{COLOR_YELLOW}  Calculation: yaw={calc_yaw}, pitch={calc_pitch}{COLOR_RESET}")
                print(f"{COLOR_YELLOW}  Execution:   yaw={yaw0}, pitch={pitch0} (Δyaw={yaw_change:.1f}, Δpitch={pitch_change:.1f}){COLOR_RESET}")
                print(f"{COLOR_YELLOW}  Movement may be inaccurate due to state mismatch.{COLOR_RESET}\n")
        
        # Calculate differences
        yaw_diff = wrap_yaw_diff(yaw_target, yaw0, mod=YAW_MOD)
        pitch_diff = pitch_target - pitch0
        zoom_diff = zoom_target - zoom0
        
        # Debug: Print formatted table
        _print_table_header("EXECUTE CAMERA MOVEMENT")
        print(_format_table_row("Starting Yaw", f"{yaw0}"))
        print(_format_table_row("Starting Pitch", f"{pitch0}"))
        print(_format_table_row("Starting Zoom", f"{zoom0}"))
        _print_table_separator()
        print(_format_table_row("Target Yaw", f"{yaw_target}"))
        print(_format_table_row("Target Pitch", f"{pitch_target}"))
        print(_format_table_row("Target Zoom", f"{zoom_target}"))
        _print_table_separator()
        print(_format_table_row("Delta Yaw", f"{yaw_diff:+.2f}"))
        print(_format_table_row("Delta Pitch", f"{pitch_diff:+.2f}"))
        print(_format_table_row("Delta Zoom", f"{zoom_diff:+.2f}"))
        
        movements_queued = False
        
        # Yaw movement
        if abs(yaw_diff) > 2:
            yaw_key = "RIGHT" if yaw_diff > 0 else "LEFT"
            yaw_hold_ms = int(abs(yaw_diff) * YAW_MS_PER_UNIT)
            yaw_hold_ms = min(yaw_hold_ms, 2000)  # Max 2 seconds
            if yaw_hold_ms > 0:
                yaw_movement = {
                    "type": "key_hold",
                    "key": yaw_key,
                    "duration_ms": yaw_hold_ms,
                    "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
                }
                _camera_movement_queue.put(yaw_movement)
                movements_queued = True
        
        # Pitch movement
        if abs(pitch_diff) > 2:
            pitch_key = "UP" if pitch_diff > 0 else "DOWN"
            pitch_hold_ms = int(abs(pitch_diff) * PITCH_MS_PER_UNIT)
            pitch_hold_ms = min(pitch_hold_ms, 2000)  # Max 2 seconds
            if pitch_hold_ms > 0:
                pitch_movement = {
                    "type": "key_hold",
                    "key": pitch_key,
                    "duration_ms": pitch_hold_ms,
                    "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
                }
                _camera_movement_queue.put(pitch_movement)
                movements_queued = True
        
        # Zoom movement (if needed)
        if abs(zoom_diff) > 10:
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
        
        if not movements_queued:
            logging.debug("No camera movements needed")
            return True
        
        # Wait for camera to stabilize - check only values that are changing
        if wait_for_stable:
            start_time = time.time()
            max_wait = 2.0  # 2 second timeout
            check_interval = 0.01  # Check every 50ms
            
            # Track which values we're checking
            check_yaw = abs(yaw_diff) > 2
            check_pitch = abs(pitch_diff) > 2
            check_zoom = abs(zoom_diff) > 10
            
            # Track last values and stability
            last_yaw = None
            last_pitch = None
            last_zoom = None
            yaw_stable = not check_yaw  # If not checking, consider stable
            pitch_stable = not check_pitch
            zoom_stable = not check_zoom
            
            while (time.time() - start_time) < max_wait:
                # Check if all values we care about are stable
                if yaw_stable and pitch_stable and zoom_stable:
                    break
                
                # Get current camera state
                camera_data = ipc.get_camera()
                if not camera_data:
                    time.sleep(check_interval)
                    continue
                
                # Check yaw if needed
                if check_yaw and not yaw_stable:
                    current_yaw = camera_data.get('yaw', 0)
                    if last_yaw is not None and current_yaw == last_yaw:
                        yaw_stable = True
                    else:
                        last_yaw = current_yaw
                
                # Check pitch if needed
                if check_pitch and not pitch_stable:
                    current_pitch = camera_data.get('pitch', 0)
                    if last_pitch is not None and current_pitch == last_pitch:
                        pitch_stable = True
                    else:
                        last_pitch = current_pitch
                
                # Check zoom if needed
                if check_zoom and not zoom_stable:
                    current_zoom = camera_data.get('scale', 0)
                    if last_zoom is not None and current_zoom == last_zoom:
                        zoom_stable = True
                    else:
                        last_zoom = current_zoom
                
                time.sleep(check_interval)
        
        # Get ending camera state
        camera_data_after = ipc.get_camera()
        if camera_data_after:
            yaw_after = camera_data_after.get("yaw", 0)
            pitch_after = camera_data_after.get("pitch", 256)
            zoom_after = camera_data_after.get("scale", 512)
            
            actual_yaw = wrap_yaw_diff(yaw_after, yaw0, mod=YAW_MOD)
            actual_pitch = pitch_after - pitch0
            actual_zoom = zoom_after - zoom0
            
            yaw_accuracy = abs(wrap_yaw_diff(yaw_after, yaw_target, mod=YAW_MOD))
            pitch_accuracy = abs(pitch_after - pitch_target)
            zoom_accuracy = abs(zoom_after - zoom_target)
            
            # Color code accuracy (green < 5, yellow < 20, red >= 20)
            yaw_color = COLOR_GREEN if yaw_accuracy < 5 else (COLOR_YELLOW if yaw_accuracy < 20 else COLOR_RED)
            pitch_color = COLOR_GREEN if pitch_accuracy < 5 else (COLOR_YELLOW if pitch_accuracy < 20 else COLOR_RED)
            zoom_color = COLOR_GREEN if zoom_accuracy < 5 else (COLOR_YELLOW if zoom_accuracy < 20 else COLOR_RED)
            
            print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}Ending Camera State:{COLOR_RESET}")
            print(_format_table_row("Ending Yaw", f"{yaw_after}"))
            print(_format_table_row("Ending Pitch", f"{pitch_after}"))
            print(_format_table_row("Ending Zoom", f"{zoom_after}"))
            
            print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}Actual Movement:{COLOR_RESET}")
            print(_format_table_row("Actual Yaw Δ", f"{actual_yaw:+.2f}"))
            print(_format_table_row("Actual Pitch Δ", f"{actual_pitch:+.2f}"))
            print(_format_table_row("Actual Zoom Δ", f"{actual_zoom:+.2f}"))
            
            print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}Movement Accuracy:{COLOR_RESET}")
            print(_format_table_row("Yaw Error", f"{yaw_accuracy:.2f}", yaw_color))
            print(_format_table_row("Pitch Error", f"{pitch_accuracy:.2f}", pitch_color))
            print(_format_table_row("Zoom Error", f"{zoom_accuracy:.2f}", zoom_color))
            print(f"{COLOR_CYAN}{'='*60}{COLOR_RESET}\n")
        else:
            logging.warning(f"[CAMERA_JACOBIAN] Could not get ending camera state")
        
        return True
        
    except Exception as e:
        logging.error(f"Error executing camera movement: {e}")
        return False

