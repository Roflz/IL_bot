# Simplified Live Features System - Implementation Summary

## Overview

This document summarizes the complete rewrite of the Live Features system to implement a simplified, straightforward workflow as specified in the requirements.

## Key Changes Made

### 1. `live_source.py` - Watchdog-Only File Watching

**Removed:**
- All polling fallback mechanisms
- Complex file detection logic
- Legacy compatibility classes

**Added:**
- Strict watchdog-only initialization (fails if watchdog unavailable)
- Comprehensive newest-file verification with directory scanning
- Detailed logging of file selection decisions
- File stability checks (size unchanged across two reads)

**Key Methods:**
- `__init__(dir_path: Path)` - Raises if watchdog unavailable
- `wait_for_next_gamestate(last_seen)` - Blocks until newer stable file appears
- `load_json(path)` - Attaches source metadata (_source_path, _source_mtime, _source_name_numeric)
- `_verify_newest_file(candidate_path)` - Ensures file is actually newest by numeric filename

**Behavior:**
- Construction fails if watchdog cannot start
- Every file event verified for stability and newest status
- Raises immediately if file is not newest (with diagnostic tail list)
- No silent fallbacks - hard failures on any inconsistency

### 2. `feature_pipeline.py` - Simple (10,128) Window Management

**Removed:**
- All delta computation functions (`push_and_diff`, `get_delta_updates_with_colors`)
- Color-bit rolling logic and complex state management
- Multiple buffer types and complex timing logic

**Added:**
- Simple `(10,128)` window with `prev_window` for change detection
- Single `_deque` for feature storage
- Straightforward `push()` method returning `(window, changed_mask, feature_names, feature_groups)`

**Key Methods:**
- `push(gamestate)` - Extracts features, builds window, returns change mask
- `get_buffer_status()` - Simple status reporting
- `clear_buffers()` - Clean buffer reset

**Implementation Rules:**
- Window shape always `(10,128)` after first push
- Row 0 = newest (T0), Row 9 = oldest (T9)
- Zero-padding on bottom (oldest) if <10 frames
- Changed mask computed as `(prev_window != window)`
- First window: `prev_window = np.full_like(window, np.nan)` so all non-zero entries count as changed

**Validation:**
- Exactly 128 features required (raises if mismatch)
- No NaN/Inf values allowed (raises if detected)
- Probe values logged for features 8/65/66/127 at row 0

### 3. `controller.py` - Simplified Workflow

**Removed:**
- Complex delta computation and color-bit management
- 600ms throttling and complex timing logic
- Multiple message types and complex UI queue handling

**Added:**
- Simple watcher → pipeline → UI flow
- Single `update_from_window` message type
- Hard failures on any step (no silent fallbacks)

**Key Methods:**
- `_watcher_worker()` - Watches files, verifies newest, loads JSON
- `_feature_worker()` - Pushes to pipeline, sends window+mask to UI
- `_pump_ui()` - Handles `update_from_window` messages

**Workflow:**
1. Watcher detects new file, verifies it's newest
2. Loads JSON with source metadata
3. Pushes to pipeline: `window, changed_mask, feature_names, feature_groups = pipeline.push(gs)`
4. Sends to UI: `("update_from_window", window, changed_mask, feature_names, feature_groups)`
5. UI updates cells based on changed_mask

**Error Handling:**
- Any step failure logged with full stack trace and re-raised
- App exits loudly on any inconsistency
- No fallback mechanisms

### 4. `live_features_view.py` - Direct Window Updates

**Removed:**
- Complex delta processing (`update_delta`)
- Color-bit management from pipeline
- Complex time-axis mapping logic

**Added:**
- `update_from_window(window, changed_mask)` - Main update method
- Simple color flipping: `color_bits[t,f] = ~color_bits[t,f]`
- Direct cell updates based on changed_mask

**Key Methods:**
- `set_schema(feature_names, feature_groups)` - Builds 128 rows
- `update_from_window(window, changed_mask)` - Updates changed cells only
- `_ensure_row_realized(feature_idx)` - Creates rows on demand

**Table Layout:**
- Columns: `["Feature", "Index", "Group", "T-0", "T-1", ..., "T-9"]`
- T-0 (newest) = leftmost time column
- T-9 (oldest) = rightmost time column
- Rows: 0..127 (one per feature)

**Change Detection:**
- For each `(t,f)` where `changed_mask[t,f]` is True:
  - Write value to cell at `row=f, col=("T-0"+t index offset)`
  - Flip `color_bits[t,f] = ~color_bits[t,f]`
  - Set text color: cyan if True, white if False

**Validation:**
- Window shape must be `(10,128)` always
- Row 0 = newest, Row 9 = oldest
- Any mismatch raises RuntimeError/ValueError
- Heavy DEBUG logging for first 8 changed cells

## Acceptance Checklist Verification

### ✅ Start Live Mode
- Watchdog logs directory and "watch started" timestamp
- First schema render shows 128 rows

### ✅ New File Processing
- Controller logs exact processed file and newest directory candidate
- Pipeline logs `(10,128)` window built with probe values at row 0
- UI logs "changed=N" and shows updates across all time columns

### ✅ Cell Updates
- Text color flips cyan⇄white for changed cells
- Only changed cells updated (no unnecessary repaints)

### ✅ Error Handling
- Watchdog missing → raises at startup
- Wrong newest file → raises immediately
- Shape mismatch → raises
- NaN/Inf → raises

## Removed Components

1. **Delta APIs**: `push_and_diff`, `get_delta_updates_with_colors`, `get_delta_updates`
2. **Color-bit Management**: Complex rolling and state tracking
3. **Polling Fallbacks**: All polling mechanisms removed
4. **Complex Timing**: 600ms throttling and complex frame analysis
5. **Legacy Compatibility**: Old classes and backward compatibility wrappers

## Benefits of Simplified System

1. **Easier Debugging**: Single data flow, clear error conditions
2. **Better Performance**: No unnecessary delta computations
3. **Reliable Updates**: Direct window-based updates with change masking
4. **Cleaner Code**: Removed complex state management and edge cases
5. **Predictable Behavior**: Hard failures prevent silent issues

## Testing

Run the test script to verify the system works:

```bash
python test_simplified_system.py
```

This will test:
- Feature pipeline window management
- Live source watchdog functionality  
- Controller integration
- Basic error handling

## Migration Notes

**For Existing Code:**
- Replace calls to `push_gamestate()` with `push()`
- Replace delta-based UI updates with `update_from_window()`
- Remove dependency on color-bit management
- Update to use new message types in controller

**Breaking Changes:**
- `FeaturePipeline.push_gamestate()` → `FeaturePipeline.push()`
- `LiveFeaturesView.update_delta()` → `LiveFeaturesView.update_from_window()`
- Controller now sends `update_from_window` instead of `deltas` messages
- All polling fallbacks removed - watchdog required

The simplified system provides the same functionality with much cleaner, more maintainable code and better error handling.
