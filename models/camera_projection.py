"""
Camera Projection Model - Phase 1 Core Functions

Implements the Model-View-Projection (MVP) pipeline for OSRS camera projection.
Converts 3D world coordinates to 2D screen coordinates using camera state.

Confirmed parameters:
- Yaw: 0-2047 units = 0-360° → yaw_rad = yaw * 2π / 2048
- Pitch: 0-512 units where 0=horizontal, 512=straight down → pitch_rad = pitch * π / 1024
- Screen: 1647 x 1017 pixels, center at (823.5, 508.5)
- Aspect ratio: 1647/1017 ≈ 1.62

TODO: Determine zoom-to-focal length relationship from calibration data.
"""
import math
from typing import Dict, Tuple, Optional


# Confirmed constants
YAW_SCALE = 2 * math.pi / 2048  # 2048 units = 360°
PITCH_SCALE = math.pi / 1024  # 0-512 units = 0-90°
SCREEN_WIDTH = 1647
SCREEN_HEIGHT = 1017
SCREEN_CENTER_X = 823.5  # SCREEN_WIDTH / 2
SCREEN_CENTER_Y = 508.5  # SCREEN_HEIGHT / 2
ASPECT_RATIO = 1647 / 1017  # ≈ 1.62


def world_to_camera_space(
    obj_world: Dict[str, float],
    camera_world: Dict[str, float],
    yaw: int,
    pitch: int
) -> Tuple[float, float, float]:
    """
    Transform world coordinates to camera space.
    
    This implements the Model and View transformations:
    1. Translate relative to camera position
    2. Rotate by yaw (around Y axis)
    3. Rotate by pitch (around X axis)
    
    Args:
        obj_world: Object world coordinates {"x": float, "y": float, "p": int}
        camera_world: Camera world coordinates {"x": float, "y": float, "z": float}
        yaw: Camera yaw in units (0-2047)
        pitch: Camera pitch in units (0-512, where 0=horizontal, 512=straight down)
    
    Returns:
        Tuple of (x_cam, y_cam, z_cam) in camera space
    """
    # Extract coordinates
    obj_x = obj_world.get("x", 0.0)
    obj_y = obj_world.get("y", 0.0)
    camera_x = camera_world.get("x", 0.0)
    camera_y = camera_world.get("y", 0.0)
    camera_z = camera_world.get("z", 0.0)
    
    # Step 1: Translate relative to camera position
    dx = obj_x - camera_x
    dy = obj_y - camera_y
    
    # Calculate object Z coordinate based on plane and model height
    obj_plane = obj_world.get("p", 0)
    # Each plane is approximately 640 units apart in RuneLite's coordinate system
    PLANE_HEIGHT_OFFSET = 640.0
    obj_z = obj_plane * PLANE_HEIGHT_OFFSET
    
    # Add model height if available (in world units, typically much smaller than plane offset)
    model_height = obj_world.get("modelHeight")
    if model_height is not None:
        obj_z += model_height
    
    # Calculate Z offset relative to camera
    dz = obj_z - camera_z
    
    # Step 2: Convert angles to radians
    yaw_rad = yaw * YAW_SCALE
    pitch_rad = pitch * PITCH_SCALE
    
    # Step 3: Apply yaw rotation (around Z axis - vertical axis in OSRS)
    # In OSRS: X and Y are horizontal (tiles), Z is vertical (height)
    # Yaw rotates horizontally around the vertical Z-axis
    # yaw=0 typically points north, increasing clockwise
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    
    # Rotate around Z axis (yaw) - standard 2D rotation in XY plane
    x_rotated = dx * cos_yaw - dy * sin_yaw
    y_rotated = dx * sin_yaw + dy * cos_yaw
    z_rotated = dz  # Z doesn't change in yaw rotation (it's the rotation axis)
    
    # Step 4: Apply pitch rotation (around X axis)
    # Pitch: 0 = horizontal, 512 = straight down (90°)
    # Positive pitch means looking down
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    
    # Rotate around X axis (pitch)
    # After yaw: x_rotated=left/right, y_rotated=forward/backward, z_rotated=up/down
    # After pitch: x_cam=left/right, y_cam=up/down, z_cam=forward/backward (depth)
    # Pitch tilts the forward direction (y_rotated) and vertical (z_rotated)
    x_cam = x_rotated  # X doesn't change in pitch rotation
    # y_cam combines forward (y_rotated) and up (z_rotated) to get up/down
    y_cam = y_rotated * sin_pitch + z_rotated * cos_pitch
    # z_cam combines forward (y_rotated) and up (z_rotated) to get forward/backward (depth)
    z_cam = y_rotated * cos_pitch - z_rotated * sin_pitch
    
    return (x_cam, y_cam, z_cam)


def camera_to_screen(
    x_cam: float,
    y_cam: float,
    z_cam: float,
    zoom: int,
    focal_length: float,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT
) -> Tuple[float, float]:
    """
    Project camera-space coordinates to screen coordinates using perspective projection.
    
    This implements the Projection transformation:
    1. Perspective divide (x/z, y/z)
    2. Scale by focal length
    3. Convert to screen pixel coordinates
    
    Args:
        x_cam: Camera-space X coordinate
        y_cam: Camera-space Y coordinate
        z_cam: Camera-space Z coordinate (depth)
        zoom: Camera zoom value (551-4409) - used for logging/debugging
        focal_length: Focal length in pixels (determined from zoom via formula)
        screen_width: Screen width in pixels (default: 1647)
        screen_height: Screen height in pixels (default: 1017)
    
    Returns:
        Tuple of (screen_x, screen_y) in pixels
    """
    # Step 1: Perspective divide (normalized device coordinates)
    # z_cam is always negative (camera is above ground), so we divide by it directly
    x_ndc = x_cam / z_cam
    y_ndc = y_cam / z_cam
    
    # Step 2: Scale by focal length and convert to screen pixels
    # Focal length determines how much the view is "zoomed in"
    screen_x = screen_width / 2 + x_ndc * focal_length
    screen_y = screen_height / 2 - y_ndc * focal_length  # Y inversion (screen Y increases downward)
    
    return (screen_x, screen_y)


def focal_length_from_zoom(
    zoom: int,
    formula: str = "inverse",
    k: float = 100000.0,
    offset: float = 0.0
) -> float:
    """
    Calculate focal length from zoom value.
    
    This function implements different formulas for the zoom-to-focal relationship.
    The exact formula will be determined from calibration data.
    
    Args:
        zoom: Camera zoom value (551-4409)
        formula: Formula type - "inverse", "direct", "shifted", or "fov_based"
        k: Constant parameter (to be fitted from data)
        offset: Offset parameter for shifted formulas (to be fitted from data)
    
    Returns:
        Focal length in pixels
    """
    if formula == "inverse":
        # f = k / zoom
        # Lower zoom → higher focal → more zoomed in
        return k / zoom if zoom > 0 else 0.0
    
    elif formula == "direct":
        # f = k * zoom
        # Higher zoom → higher focal → more zoomed in
        return k * zoom
    
    elif formula == "shifted":
        # f = k / (zoom - offset)
        # Accounts for non-zero offset in zoom scale
        return k / (zoom - offset) if (zoom - offset) > 0 else 0.0
    
    elif formula == "fov_based":
        # FOV = k / zoom, then f = (height/2) / tan(FOV/2)
        # Converts zoom to FOV first, then FOV to focal length
        fov_rad = (k / zoom) if zoom > 0 else 0.0
        if fov_rad <= 0 or fov_rad >= math.pi:
            return 0.0
        return (SCREEN_HEIGHT / 2) / math.tan(fov_rad / 2)
    
    else:
        raise ValueError(f"Unknown formula type: {formula}")


def predict_screen_position(
    obj_world: Dict[str, float],
    camera_world: Dict[str, float],
    yaw: int,
    pitch: int,
    zoom: int,
    focal_formula: str = "inverse",
    focal_k: float = 100000.0,
    focal_offset: float = 0.0,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT
) -> Optional[Dict[str, float]]:
    """
    Predict screen position of an object given world coordinates and camera state.
    
    This is the main end-to-end function that combines world_to_camera_space
    and camera_to_screen.
    
    Args:
        obj_world: Object world coordinates {"x": float, "y": float, "p": int}
        camera_world: Camera world coordinates {"x": float, "y": float, "z": float}
        yaw: Camera yaw in units (0-2047)
        pitch: Camera pitch in units (0-512)
        zoom: Camera zoom value (551-4409)
        focal_formula: Formula type for zoom-to-focal conversion
        focal_k: Constant parameter for focal length calculation
        focal_offset: Offset parameter for shifted formulas
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
    
    Returns:
        Dictionary with {"x": float, "y": float} screen coordinates, or None if not visible
    """
    # Step 1: Transform to camera space
    x_cam, y_cam, z_cam = world_to_camera_space(obj_world, camera_world, yaw, pitch)
    
    # Step 2: Calculate focal length from zoom
    focal_length = focal_length_from_zoom(zoom, focal_formula, focal_k, focal_offset)
    
    if focal_length <= 0:
        return None  # Invalid focal length
    
    # Step 3: Project to screen
    screen_x, screen_y = camera_to_screen(
        x_cam, y_cam, z_cam, zoom, focal_length, screen_width, screen_height
    )
    
    return {"x": screen_x, "y": screen_y}


def calculate_reprojection_error(
    predicted: Dict[str, float],
    observed: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate reprojection error between predicted and observed screen positions.
    
    Args:
        predicted: Predicted screen coordinates {"x": float, "y": float}
        observed: Observed screen coordinates {"x": float, "y": float}
    
    Returns:
        Dictionary with error metrics:
        - "error_x": Error in X direction (pixels)
        - "error_y": Error in Y direction (pixels)
        - "error_distance": Euclidean distance error (pixels)
        - "error_manhattan": Manhattan distance error (pixels)
    """
    dx = predicted["x"] - observed["x"]
    dy = predicted["y"] - observed["y"]
    
    error_distance = math.sqrt(dx * dx + dy * dy)
    error_manhattan = abs(dx) + abs(dy)
    
    return {
        "error_x": dx,
        "error_y": dy,
        "error_distance": error_distance,
        "error_manhattan": error_manhattan
    }

