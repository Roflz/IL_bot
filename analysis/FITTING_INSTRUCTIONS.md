# Camera Model Fitting Instructions

## Overview

This directory contains scripts to fit the camera projection model parameters from your calibration data.

## Files

- **`fit_camera_model.py`**: Main script to fit model parameters from calibration data
- **`analyze_screen_projection_data.py`**: Script to analyze calibration data (already exists)

## Quick Start

### Step 1: Fit Model Parameters

Run the fitting script with your calibration data files:

```bash
cd D:\repos\bot_runelite_IL
python analysis/fit_camera_model.py --files screen_projection_calibration_20251227_202250.jsonl screen_projection_calibration_20251227_193615.jsonl
```

This will:
1. Load the calibration data from both files
2. Test all focal length formulas (inverse, direct, shifted, fov_based)
3. Fit parameters for each formula
4. Select the best formula based on lowest error
5. Save fitted parameters to `camera_model_params.json`

### Step 2: Test the Model

Once parameters are fitted, the forward model in `test_camera_movement.py` will automatically use them:

1. Run `test_camera_movement.py`
2. Press 'T' to test camera optimization (uses forward model)
3. Press 'M' to test single-object prediction (uses inverse model - not yet implemented)

## Command Line Options

```bash
python analysis/fit_camera_model.py [OPTIONS]

Options:
  --files FILES          Calibration data files (space-separated)
                        Default: screen_projection_calibration_20251227_202250.jsonl 
                                 screen_projection_calibration_20251227_193615.jsonl
  
  --formula FORMULA     Formula to fit: inverse, direct, shifted, fov_based, or all
                        Default: all (tests all formulas)
  
  --output OUTPUT       Output file for parameters
                        Default: camera_model_params.json
```

## Example Usage

### Test all formulas (recommended):
```bash
python analysis/fit_camera_model.py
```

### Fit only inverse formula:
```bash
python analysis/fit_camera_model.py --formula inverse
```

### Use different files:
```bash
python analysis/fit_camera_model.py --files file1.jsonl file2.jsonl file3.jsonl
```

### Custom output file:
```bash
python analysis/fit_camera_model.py --output my_params.json
```

## Output

The script will:
1. Print progress during optimization
2. Show results for each formula tested
3. Indicate which formula performed best
4. Save the best parameters to `camera_model_params.json`

The output JSON file contains:
- `focal_formula`: Best formula type
- `focal_k`: Fitted constant parameter
- `focal_offset`: Fitted offset (if using shifted formula)
- `error_stats`: Error statistics (mean, median, max, etc.)
- Other confirmed constants (yaw_scale, pitch_scale, etc.)

## Next Steps

After fitting:
1. The forward model in `test_camera_movement.py` will automatically use the fitted parameters
2. Test the 'T' key to see optimization in action
3. If errors are high, collect more calibration data and re-fit
4. Once satisfied, the model can be integrated into the main camera control system

## Troubleshooting

**"scipy not available"**: Install scipy: `pip install scipy`

**"No valid training points found"**: Check that your calibration data has objects with valid screen positions (not null)

**High errors**: 
- Collect more calibration data
- Ensure data covers full range of yaw/pitch/zoom
- Check for outliers in the data

**Optimization fails**: 
- Try different initial guesses for `focal_k`
- Try a different formula type
- Check that data is valid (no NaN or invalid values)



