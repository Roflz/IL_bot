# GUI Modularization - Extraction Status

## ✅ Completed Modules

1. **`gui/plan_editor.py`** - Plan editing with rules and parameters ✅
2. **`gui/config_manager.py`** - Configuration loading/saving/auto-detect ✅
3. **`gui/launcher.py`** - RuneLite instance launching and credential management ✅
4. **`gui/client_detector.py`** - Automatic client detection ✅
5. **`gui/instance_manager.py`** - Instance tab management and plan execution ✅
6. **`gui/statistics.py`** - Statistics display and monitoring ✅
7. **`gui/widgets.py`** - Widget creation helpers and styling ✅
8. **`gui/logging_utils.py`** - Centralized logging ✅
9. **`gui/main_window.py`** - Main window orchestrating all components ✅
10. **`gui.py`** - Simple entry point ✅

## ⚠️ Remaining Work

### 1. `create_instance_tab` Full Implementation
**Location**: `gui/instance_manager.py` line 76
**Status**: Placeholder exists, full implementation needed
**Source**: `gui.py` lines 1893-2410 (~520 lines)

**What it creates**:
- Main instance tab with sub-notebook (Plan Runner, Output, Statistics)
- Plan selection UI (available/selected plans with controls)
- Plan details UI (rules/parameters editing with inline widgets)
- Statistics display sections (skills/inventory/equipment)
- Control buttons (start/stop/pause)
- Log output area
- Statistics tab content

**Dependencies** (helper methods that need to be implemented or passed as callbacks):
- `browse_directory_for_instance` - Browse for session directory
- `add_plan_to_selection` - Add plan from available to selected
- `remove_plan_from_selection` - Remove plan from selected
- `move_plan_up` / `move_plan_down` - Reorder plans
- `clear_selected_plans` - Clear all selected plans
- `update_plan_details_inline` - Update plan details display
- `update_parameter_widgets` - Update parameter editing widgets
- `add_rule_inline_advanced` - Add rule with advanced widgets
- `clear_plan_parameters` / `clear_plan_rules` - Clear plan data
- `populate_sequences_list` - Populate saved sequences listbox
- `save_sequence_for_instance` - Save plan sequence to file
- `load_sequence_from_list` - Load plan sequence from file
- `delete_sequence_from_list` - Delete saved sequence
- `start_stats_monitor` - Start statistics monitoring (already in statistics module)
- `update_stats_text` - Update stats display (already in statistics module)

### 2. Launcher Tab UI Creation
**Location**: `gui/main_window.py` line 127 (`_create_launcher_tab`)
**Status**: Placeholder exists, full implementation needed
**Source**: `gui.py` lines 811-1020 (~210 lines)

**What it creates**:
- Setup & Configuration section (paths, auto-detect)
- Launch configuration (base port, delay, instance count)
- Credential selection UI (available/selected listboxes)
- Launch controls (build Maven checkbox, launch/stop buttons)
- Client detection controls (start/stop/test detection buttons)

### 3. Statistics Display Full Implementation
**Location**: `gui/statistics.py` line 89 (`update_stats_text`)
**Status**: Core structure exists, full display implementation needed
**Source**: `gui.py` lines 1332-1575 (~243 lines)

**What it displays**:
- Skills section (3 columns with icons and levels)
- Inventory section (all items with quantities)
- Equipment section (equipped items by slot)
- Key items totals (when bank is open)

## 📝 Implementation Notes

### Helper Methods Location
Most helper methods are currently in the original `gui.py`:
- Lines 2478-2528: Instance management helpers
- Lines 3480-4520: Plan selection and editing helpers
- Lines 4254-4360: Sequence management helpers

These should be:
1. Extracted to `instance_manager.py` as methods, OR
2. Passed as callbacks from `main_window.py` to `instance_manager`

### Integration Approach
The current structure uses callbacks for flexibility. To complete the extraction:

1. **Option A**: Extract all helper methods to `instance_manager.py`
   - Pros: Self-contained, easier to maintain
   - Cons: Large file, tight coupling

2. **Option B**: Keep callbacks, implement helpers in `main_window.py`
   - Pros: Flexible, smaller modules
   - Cons: More complex callback management

3. **Option C**: Create a separate `plan_management.py` module
   - Pros: Better separation of concerns
   - Cons: More modules to manage

## 🎯 Next Steps

1. Extract full `create_instance_tab` implementation
2. Extract launcher tab UI creation
3. Complete statistics display implementation
4. Test integrated GUI
5. Fix any integration issues
6. Remove old code from `gui.py` (keep only entry point)

## 📊 Progress

- **Modules Created**: 10/10 ✅
- **Core Functionality**: 8/10 ⚠️
- **Full UI Implementation**: 7/10 ⚠️
- **Integration**: 9/10 ✅

Overall: **~85% Complete**
