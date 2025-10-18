# Go_to() Timing Instrumentation

This document describes the timing instrumentation added to the `go_to()` function in `travel.py` to measure exactly where time is spent in the travel pipeline.

## Overview

The instrumentation adds high-resolution timing measurements to track the performance of different segments within the `go_to()` function, including:

- **Rect lookup** - Resolving region keys to coordinates
- **Player position** - Fetching current player coordinates  
- **Long distance path selection** - Waypoint selection and remaining count calculation
- **Path generation** - Both short and long distance pathfinding
- **Projection and merging** - Converting waypoints to screen coordinates
- **Door handling** - Door detection, path generation, and opening
- **Ground clicking** - Final click execution

## Files Modified

### `actions/travel.py`
- Added timing instrumentation to `go_to()` function
- Added `_GO_TO_TIMING_ENABLED = True` constant
- Added inline timing helper functions `_mark()` and `_emit()`
- Preserved all existing logging and functionality

### `scripts/summarize_go_to_timing.py`
- Analysis script to process timing data
- Calculates median, p90, p95 for each segment
- Separates analysis by LONG_DISTANCE vs SHORT_DISTANCE branches
- Identifies top 10 slowest calls and dominant segments

### `scripts/test_go_to_timing.py`
- Test script to generate timing data
- Tests both short and long distance destinations
- Performs rapid calls to generate more data points

## Timing Segments

The instrumentation measures these segments (in milliseconds):

### Common Segments
- `rect_lookup` - Region key resolution
- `player_pos` - Player position fetch
- `click_ground` - Final ground click execution
- `total` - Total function execution time

### Long Distance Branch
- `ldp_select` - Long distance waypoint selection
- `ldp_wp_path_gen` - Waypoint path generation
- `ldp_wp_project_merge` - Waypoint projection and door merging
- `ldp_door_path_gen` - Door check path generation
- `ldp_door_open` - Door opening execution

### Short Distance Branch  
- `short_path_gen` - Short distance path generation
- `short_project_merge` - Projection and door merging
- `short_door_path_gen` - Door check path generation
- `short_door_open` - Door opening execution

## Output Format

Timing data is emitted as JSONL (one JSON object per line) with this structure:

```json
{
  "phase": "go_to_timing",
  "branch": "LONG_DISTANCE",
  "ok": true,
  "dur_ms": {
    "rect_lookup": 2.1,
    "player_pos": 1.0,
    "ldp_select": 3.8,
    "ldp_wp_path_gen": 6.7,
    "ldp_wp_project_merge": 4.2,
    "ldp_door_path_gen": 5.3,
    "ldp_door_open": 420.5,
    "click_ground": 38.9,
    "total": 622.4
  },
  "context": {
    "waypoints_remaining": 17,
    "target_waypoint": {"x": 3208, "y": 3421},
    "wps_count": 24,
    "door_attempted": true,
    "door_success": true
  },
  "ts": "2025-01-10T21:05:00.123Z"
}
```

## Usage

### 1. Enable Timing
Timing is enabled by default with `_GO_TO_TIMING_ENABLED = True` in `travel.py`.

### 2. Generate Timing Data
Run the test script to generate timing data:

```bash
python scripts/test_go_to_timing.py
```

### 3. Analyze Results
Run the summarizer to analyze the timing data:

```bash
python scripts/summarize_go_to_timing.py
```

### 4. View Raw Data
Timing data is written to `travel.go_to.timing.jsonl` in the same directory as `travel.py`.

## Sample Output

The summarizer produces output like:

```
Go_to() Timing Analysis
==================================================
Total go_to() calls: 200
LONG_DISTANCE calls: 120
SHORT_DISTANCE calls: 80

=== LONG_DISTANCE BRANCH ANALYSIS ===
Calls: 120
Segment              Median   P90      P95      Samples 
------------------------------------------------------------
rect_lookup          2.1      3.2      4.1      120     
player_pos           1.0      1.5      2.0      120     
ldp_select           3.8      5.2      6.1      120     
ldp_wp_path_gen      6.7      12.3     15.8     120     
ldp_wp_project_merge 4.2      7.1      9.3      120     
ldp_door_path_gen    5.3      8.9      12.1     45      
ldp_door_open        420.5    850.2    1200.1   45      
click_ground         38.9     65.2     89.7     120     
total                622.4    1250.8   1800.3   120     

Top 10 slowest calls:
Rank Total (ms)    Dominant Segment    Door Attempted  Error              
--------------------------------------------------------------------------------
1    1800.3        ldp_door_open      True            Door opening failed
2    1650.2        ldp_door_open      True                            
3    1420.1        ldp_door_open      True                            
...
```

## Key Insights

The timing data helps identify:

1. **Bottlenecks** - Which segments take the most time
2. **Branch differences** - Performance differences between long/short distance
3. **Door impact** - How much time door handling adds
4. **Error patterns** - Which segments fail most often
5. **Optimization targets** - Where to focus performance improvements

## Implementation Notes

- Uses `time.perf_counter_ns()` for high-resolution timing
- All timing is in milliseconds for readability
- Preserves all existing functionality and logging
- Minimal code changes - only adds timing markers
- Handles both success and error cases
- Uses try/finally blocks to ensure timing data is always emitted

