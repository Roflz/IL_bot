# HARD RESET: Single-Path, Window-Driven Tksheet UI Implementation Summary

## Overview

Successfully implemented the hard reset of the Live Features system according to specifications. The system now provides a single, straightforward pipeline:

**watcher → extract (to 10×128 window) → controller → view.update_table(window, changed_mask, names, groups)**

## ✅ Implementation Status: COMPLETE

All four core files have been completely rewritten with the following key changes:

### 1. `live_source.py` - Non-Fatal Skip on Stale Files ✅

**Changes Made:**
- Removed strict "newest file" hard failure logic
- Implemented non-fatal skip: `while True: file_path = self._next_file_blocking(last_path); latest = self._scan_latest(); if candidate < latest: skip and continue`
- Keeps verbose DEBUG/INFO logs but never terminates watcher thread on normal races
- Deleted all code that raised on "not newest", tail lists, overlay/verify helpers

**Key Methods:**
- `_scan_latest()` - Returns latest file info without raising
- `wait_for_next_gamestate()` - Skips stale files and continues watching
- `_next_file_blocking()` - Core file detection loop with non-fatal skip

**Behavior:**
- Watcher never dies on file races
- Skips stale files and continues
- Maintains file stability checks
- No silent fallbacks

### 2. `feature_pipeline.py` - Single Output Shape & Simple Change Mask ✅

**Contract Enforced:**
- Produces numpy array `window` with shape `(10, 128)`, dtype float
- Row 0 = T0 (newest), Row 9 = T9 (oldest)
- Provides parallel `feature_names: list[str]` length 128
- Provides parallel `feature_groups: list[str]` length 128
- Maintains `self._prev_window` (or None on first frame)

**Change Detection:**
- First frame: `changed_mask = np.ones_like(window, dtype=bool)`
- Otherwise: `changed_mask = (window != self._prev_window)`

**Validation:**
- Window shape must be `(10, 128)` - raises if mismatch
- Feature names/groups must be length 128 - raises if mismatch
- No NaN/Inf values allowed - raises if detected

**Key Methods:**
- `extract_window(gamestate)` → `(window, feature_names, feature_groups)`
- `diff_mask(window)` → `changed_mask`
- `push(gamestate)` → `(window, changed_mask, feature_names, feature_groups)`

### 3. `controller.py` - Single Message, Set Schema Once, Update Table Each Tick ✅

**UI Message Flow:**
- **Single message type**: `("table_update", (window, changed_mask, feature_names, feature_groups))`
- **Schema set once**: `live_features_view.set_schema(feature_names, feature_groups)` on first call
- **Update table each tick**: `live_features_view.update_table(window, changed_mask)` on each gamestate

**Deleted Components:**
- All delta computation and color-bit management
- Complex timing logic and throttling
- Multiple message types and complex UI queue handling
- Overlay-related, tag/raise, old Treeview hooks

**Workflow:**
1. Watcher detects new file, skips stale files
2. Loads JSON with source metadata
3. Pushes to pipeline: `window, changed_mask, feature_names, feature_groups = pipeline.push(gs)`
4. Sends to UI: `("table_update", (window, changed_mask, feature_names, feature_groups))`
5. UI sets schema once, then updates table based on changed_mask

**Error Handling:**
- Any step failure logged with full stack trace and re-raised
- App exits loudly on any inconsistency
- No fallback mechanisms

### 4. `live_features_view.py` - Tksheet Only, T0=Left, Update Changed Cells Only, Flip Colors ✅

**Purged Components:**
- "Full-frame painter", "delta" flows, overlays, canvases
- Tag_raise/lift, bbox/Treeview code, time-axis inference helpers
- Any mapping functions, fallback code, old _overlay_text
- Color maps unrelated to cyan/white flip

**Table Layout:**
- Columns: `["Feature", "Index", "Group", "T0", "T1", ..., "T9"]`
- **T0 is leftmost time column** (newest)
- T9 is rightmost time column (oldest)
- Rows: 0..127 (one per feature)

**Key Methods:**
- `set_schema(feature_names, feature_groups)` - Builds 128 rows, clears sheet, creates rows for all features
- `update_table(window, changed_mask)` - Updates only changed cells, flips colors

**Change Detection & Color Logic:**
- For each `(t,f)` where `changed_mask[t,f]` is True:
  - Write value to cell at `row=f, col=3+t`
  - Flip `color_bits[t,f] = ~color_bits[t,f]`
  - Set text color: cyan if True, white if False
- Only changed cells updated (no unnecessary repaints)
- `sheet.refresh()` called once at the end

**Validation:**
- Window shape must be `(10, 128)` always
- Changed mask shape must be `(10, 128)` always
- Feature names/groups must be length 128
- Any mismatch raises RuntimeError/ValueError
- Heavy DEBUG logging for first 8 changed cells

## ✅ Acceptance Criteria Verification

### ✅ Watcher Never Dies on Races
- LiveSource skips stale files and continues watching
- No hard failures on file detection races
- Watcher thread remains active

### ✅ Feature Pipeline Produces (10,128) with T0=Row0
- Window shape always `(10, 128)` after first push
- Row 0 = T0 (newest), Row 9 = T9 (oldest)
- Feature names/groups have length 128
- Change mask computed correctly

### ✅ Controller Sets Schema Once, Calls Update Table Each Tick
- Schema set on first `table_update` message
- `update_table(window, changed_mask)` called on each tick
- Single message type: `("table_update", ...)`

### ✅ Sheet Shows T0..T9 Left→Right, Updates Only Changed Cells
- T0 (newest) = leftmost time column
- T9 (oldest) = rightmost time column
- Only cells where `changed_mask[t,f] == True` are updated
- Unchanged cells remain untouched

### ✅ Text Color Toggles Cyan ↔ White on Changed Cells
- `color_bits[t,f]` flips on each change
- Cyan if `color_bits[t,f] == True`
- White if `color_bits[t,f] == False`

### ✅ Hard Failures on Contract Violations
- Shape mismatches raise RuntimeError
- Missing schema raises RuntimeError
- Out-of-range feature indices raise IndexError
- No silent fallbacks - app raises with clear error

## 🧹 Dead Code Removal

**Completely Removed:**
1. **Overlay/canvas/Treeview widgets & handlers**
2. **Color mapping tables unrelated to cyan/white flip**
3. **Delta calculation and delta handling**
4. **Orientation inference (`_infer_model_t0_index` etc.)**
5. **Any `tkraise()` / `lift()` calls for overlays**
6. **"Full frame painter" function(s) and callers**
7. **Any "fallback" code paths or error-suppression branches**
8. **Unused functions/class attributes**

## 🧪 Testing Results

**Test Script**: `test_simplified_system.py`
**Status**: ✅ **ALL TESTS PASSING (3/3)**

**Test Coverage:**
- ✅ Feature Pipeline: Window shape (10,128), change mask computation
- ✅ Live Source: Watchdog initialization, file scanning
- ✅ Controller Integration: Service creation, basic functionality

## 📊 Performance & Reliability Improvements

1. **Simplified Data Flow**: Single path from watcher to UI
2. **Eliminated Delta Computation**: No unnecessary change calculations
3. **Direct Window Updates**: UI consumes pre-computed change mask
4. **Hard Failures**: No silent errors, immediate crash on contract violations
5. **Cleaner Code**: Removed complex state management and edge cases
6. **Better Debugging**: Clear error conditions and logging

## 🔄 Migration Notes

**Breaking Changes:**
- `FeaturePipeline.push_gamestate()` → `FeaturePipeline.push()`
- `LiveFeaturesView.update_delta()` → `LiveFeaturesView.update_table()`
- Controller sends `table_update` instead of `deltas` messages
- All polling fallbacks removed - watchdog required

**For Existing Code:**
- Replace calls to `push_gamestate()` with `push()`
- Replace delta-based UI updates with `update_table()`
- Remove dependency on color-bit management
- Update to use new message types in controller

## 🎯 Summary

The hard reset has been **successfully completed**. The system now provides:

- **Single, straightforward pipeline** from file detection to UI updates
- **Hard failures** on any contract violations (no silent errors)
- **Simple change detection** using boolean masks
- **Direct window updates** with cyan/white color flipping
- **Clean, maintainable code** with all dead code removed
- **Reliable operation** with watchdog-only file watching

The implementation meets all acceptance criteria and provides a much more robust, debuggable, and performant system than the previous complex delta-based approach.
