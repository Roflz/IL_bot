# Normalization Audit Report

## Executive Summary

This audit examines data normalization throughout the training pipeline, from raw inputs to model outputs. **The coordinate normalization is working correctly**, but **input data normalization is missing** and could significantly impact training performance.

## Key Findings

### ✅ Working Correctly

1. **Coordinate Normalization**
   - Raw coordinates: X (0-1708), Y (0-860) pixels
   - Normalized to [0,1] by dividing by 1920x1080
   - Model outputs: `torch.sigmoid()` applied to ensure [0,1] range
   - Loss computation: MSE between normalized coordinates
   - Behavioral metrics: Proper denormalization for display

2. **Categorical Variables**
   - Button (0-2), key_action (0-2), key_id (0-87): No normalization needed
   - Scroll_y (-1, 1): Binary values, no normalization needed

3. **Event Classification**
   - Model outputs raw logits
   - Softmax applied in loss function and training loop
   - Cross-entropy loss (naturally normalized)

### ❌ Issues Found

1. **Input Data Not Normalized**
   - Gamestate sequences: Range -1 to 96,848 (mean: 5,051, std: 12,750)
   - Action input sequences: Range -1 to 398,147 (mean: 5,193, std: 36,212)
   - **Impact**: Large variance can cause training instability and slow convergence

2. **Time Values Unclear**
   - Action target column 0: Range 0 to 4,991
   - **Impact**: Unclear what these represent and how they should be normalized

## Data Flow Analysis

### Raw Data → Dataset
```
Raw coordinates (0-1708, 0-860) → Normalized (0-1) ✅
Raw gamestate (-1 to 96,848) → No normalization ❌
Raw action input (-1 to 398,147) → No normalization ❌
```

### Dataset → Model
```
Normalized coordinates (0-1) → Model receives correctly ✅
Unnormalized gamestate → Model receives large values ❌
Unnormalized action input → Model receives large values ❌
```

### Model → Outputs
```
Raw coordinate logits → torch.sigmoid() → [0,1] ✅
Raw event logits → torch.softmax() → probabilities ✅
Raw time outputs → No normalization ❓
```

### Outputs → Loss
```
Normalized coordinates → MSE loss ✅
Event probabilities → Cross-entropy loss ✅
Time outputs → MSE loss (unclear normalization) ❓
```

## Recommendations

### High Priority

1. **Add Input Normalization**
   ```python
   # In OSRSDataset.__getitem__()
   temporal_sequence = (temporal_sequence - temporal_sequence.mean()) / temporal_sequence.std()
   action_sequence = (action_sequence - action_sequence.mean()) / action_sequence.std()
   ```

2. **Investigate Time Normalization**
   - Determine what time values represent (milliseconds, seconds, etc.)
   - Apply appropriate normalization (e.g., divide by 1000 for milliseconds)

### Medium Priority

3. **Add Data Preprocessing Pipeline**
   - Compute normalization statistics on training data
   - Store mean/std for consistent normalization
   - Apply same normalization to validation/test data

4. **Model Architecture Review**
   - Consider if large input ranges affect gradient flow
   - Evaluate if batch normalization helps with unnormalized inputs

### Low Priority

5. **Documentation**
   - Document all normalization decisions
   - Add comments explaining coordinate scaling
   - Create data format specification

## Implementation Plan

### Phase 1: Input Normalization
1. Modify `OSRSDataset.__getitem__()` to normalize inputs
2. Test training with normalized inputs
3. Compare convergence speed and stability

### Phase 2: Time Normalization
1. Investigate time value meaning
2. Apply appropriate normalization
3. Update loss function if needed

### Phase 3: Preprocessing Pipeline
1. Create data preprocessing utilities
2. Compute and store normalization statistics
3. Ensure consistent normalization across splits

## Expected Impact

- **Training Stability**: Reduced input variance should improve gradient flow
- **Convergence Speed**: Normalized inputs typically train faster
- **Model Performance**: Better numerical stability may improve final accuracy
- **Debugging**: Consistent normalization makes debugging easier

## Files Modified

- `ilbot/training/setup.py`: Add input normalization
- `ilbot/training/train_loop.py`: Coordinate normalization (already fixed)
- `ilbot/training/simplified_behavioral_metrics.py`: Denormalization (already working)

## Testing

After implementing changes:
1. Run training with normalized inputs
2. Compare loss curves with previous runs
3. Verify coordinate predictions are still accurate
4. Check behavioral metrics display correctly
