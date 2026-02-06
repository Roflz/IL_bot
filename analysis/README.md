# Camera Projection Analysis Tools

This directory contains tools for analyzing calibration data and fitting the camera projection model.

## Files

- `analyze_screen_projection_data.py` - Analyzes collected calibration data

## Usage

```bash
# Analyze all calibration files
python analysis/analyze_screen_projection_data.py
```

This script will:
1. Load all `screen_projection_calibration_*.jsonl` files
2. Analyze data distribution (camera states, object positions)
3. Determine zoom-to-focal length relationship
4. Calculate initial error statistics

## Output

The script prints a comprehensive report including:
- Data distribution statistics
- Zoom-to-focal relationship analysis
- Initial error metrics (before parameter fitting)

## Next Steps

After running the analysis:
1. Review the zoom-to-focal relationship determination
2. Use the results to inform parameter fitting (Phase 3)
3. Iterate on model parameters to minimize reprojection error


