# Camera Projection Models

This directory contains the core mathematical functions for the OSRS camera projection system.

## Files

- `camera_projection.py` - Core projection functions implementing the MVP pipeline

## Usage

```python
from models.camera_projection import predict_screen_position

# Predict where an object will appear on screen
obj_world = {"x": 3200, "y": 3200, "p": 0}
player_world = {"x": 3200, "y": 3201}
yaw = 1024  # 180 degrees
pitch = 256  # 45 degrees down
zoom = 1000

predicted = predict_screen_position(
    obj_world,
    player_world,
    yaw,
    pitch,
    zoom,
    focal_formula="inverse",  # To be determined from calibration
    focal_k=100000.0  # To be fitted from data
)

if predicted:
    print(f"Predicted screen position: ({predicted['x']}, {predicted['y']})")
```

## Status

✅ Phase 1 Complete:
- `world_to_camera_space()` - Transforms world coords to camera space
- `camera_to_screen()` - Projects camera space to screen pixels
- `focal_length_from_zoom()` - Converts zoom to focal length (formula TBD)
- `predict_screen_position()` - End-to-end prediction function
- `calculate_reprojection_error()` - Error calculation

⚠️ Pending:
- Zoom-to-focal length relationship needs to be determined from calibration data
- Parameter fitting (Phase 3)


