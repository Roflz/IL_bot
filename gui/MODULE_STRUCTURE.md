# GUI Module Structure

This document describes the modular structure of the GUI components.

## Overview

The GUI has been split into logical modules to improve maintainability and organization. The original `gui.py` (4525 lines) has been broken down into focused, single-responsibility modules.

## Module Descriptions

### `main_window.py`
**Purpose**: Main orchestrator for the GUI application
**Responsibilities**:
- Initialize all component managers
- Create main window structure
- Coordinate between modules
- Handle window lifecycle

**Key Classes**:
- `SimpleRecorderGUI`: Main application class

### `plan_editor.py`
**Purpose**: Modal dialog for editing plan rules and parameters
**Responsibilities**:
- Plan rule editing (max time, stop skill, stop items, etc.)
- Plan parameter editing (GE items, generic params, etc.)
- Validation and data collection

**Key Classes**:
- `PlanEditor`: Modal editor window
- `PlanEntry`: TypedDict for plan data structure

**Status**: ✅ **COMPLETE** - Fully extracted from original gui.py

### `config_manager.py`
**Purpose**: Configuration file management
**Responsibilities**:
- Load/save configuration from JSON
- Auto-detect RuneLite paths
- Manage configuration variables
- Browse for paths/files

**Key Classes**:
- `ConfigManager`: Configuration management

**Methods to implement**:
- `load_config()`: Load from launch-config.json
- `save_config()`: Save to launch-config.json
- `auto_detect_paths()`: Auto-detect RuneLite installation paths
- `toggle_auto_detect()`: Enable/disable auto-detect mode
- `browse_path()`: Open file/directory browser

### `launcher.py`
**Purpose**: RuneLite instance launching
**Responsibilities**:
- Launch RuneLite instances
- Manage credentials selection
- Handle Maven builds
- Stop running instances

**Key Classes**:
- `RuneLiteLauncher`: Instance launcher

**Methods to implement**:
- `setup_dependencies()`: Run setup script
- `populate_credentials()`: Load credentials from directory
- `refresh_credentials()`: Refresh credential list
- `add_credential()`: Add credential to launch list
- `remove_credential()`: Remove from launch list
- `launch_runelite()`: Launch instances
- `stop_runelite()`: Stop all instances

### `client_detector.py`
**Purpose**: Automatic RuneLite client detection
**Responsibilities**:
- Detect running RuneLite clients
- Auto-create instance tabs
- Monitor client lifecycle
- Test detection functionality

**Key Classes**:
- `ClientDetector`: Client detection manager

**Methods to implement**:
- `start_client_detection()`: Start auto-detection loop
- `stop_client_detection()`: Stop auto-detection
- `detect_running_clients()`: Scan for running clients
- `_client_detection_loop()`: Internal detection loop
- `test_client_detection()`: Debug/test method
- `remove_instance_tab()`: Clean up removed instances

### `instance_manager.py`
**Purpose**: Instance tab and plan execution management
**Responsibilities**:
- Create/manage instance tabs
- Handle plan selection and ordering
- Execute plans for instances
- Monitor plan execution

**Key Classes**:
- `InstanceManager`: Instance lifecycle manager

**Methods to implement**:
- `create_instance_tab()`: Create new instance tab
- `add_plan_to_instance()`: Add plan to instance queue
- `remove_plan_from_instance()`: Remove plan from queue
- `start_plans_for_instance()`: Start plan execution
- `stop_plans_for_instance()`: Stop plan execution
- `execute_plan_for_instance()`: Execute single plan
- `on_instance_tab_changed()`: Handle tab switching
- `_get_completion_patterns_for_plan()`: Get completion detection patterns
- `_write_rule_params_to_file()`: Write rules for StatsMonitor

### `statistics.py`
**Purpose**: Statistics display and monitoring
**Responsibilities**:
- Load and display character stats
- Manage skill icons
- Update statistics displays
- Start/stop stats monitoring

**Key Classes**:
- `StatisticsDisplay`: Statistics display manager

**Methods to implement**:
- `load_skill_icons()`: Load skill icon images
- `load_character_stats()`: Load from CSV
- `update_stats_text()`: Update display with current stats
- `start_stats_monitor()`: Start StatsMonitor
- `stop_stats_monitor()`: Stop StatsMonitor
- `update_statistics_display()`: Refresh display
- `start_statistics_timer()`: Start periodic updates
- `estimate_item_value()`: Estimate item prices

### `widgets.py`
**Purpose**: Reusable widget factories and style configuration
**Responsibilities**:
- Configure ttk styles
- Create common widget patterns
- Provide utility functions

**Key Classes**:
- `WidgetFactory`: Widget creation utilities

**Methods to implement**:
- `setup_styles()`: ✅ **COMPLETE** - Style configuration
- `create_scrollable_frame()`: Create scrollable frame widget
- `center_window()`: Center window on screen

### `logging_utils.py`
**Purpose**: Logging utilities for GUI text widgets
**Responsibilities**:
- Format log messages
- Apply color coding
- Handle timestamps

**Key Classes**:
- `LoggingUtils`: Logging utilities (static methods)

**Methods to implement**:
- `log_message()`: ✅ **COMPLETE** - Log to text widget
- `log_message_to_instance()`: ✅ **COMPLETE** - Log to instance

## Migration Status

### ✅ Completed
- Module structure created
- `plan_editor.py` fully extracted
- `widgets.py` style setup complete
- `logging_utils.py` complete

### 🚧 In Progress
- Module stubs created (ready for implementation)

### 📋 TODO
- Extract methods from `gui.py` to respective modules
- Update `main_window.py` to use component managers
- Update main `gui.py` to import from modules
- Test all functionality

## Next Steps

1. **Fill in module implementations** - Start with most self-contained modules:
   - `logging_utils.py` ✅ (already complete)
   - `widgets.py` (mostly complete, add helper methods)
   - `config_manager.py` (straightforward file I/O)
   - `statistics.py` (self-contained display logic)

2. **Extract complex modules** - These have more dependencies:
   - `launcher.py` (subprocess management)
   - `client_detector.py` (process detection)
   - `instance_manager.py` (plan execution, most complex)

3. **Update main_window.py** - Wire everything together

4. **Update gui.py** - Make it a simple entry point that imports from modules

## File Size Reduction

- **Original**: `gui.py` - 4525 lines
- **Target**: 
  - `main_window.py` - ~500 lines (orchestration)
  - `plan_editor.py` - 689 lines ✅
  - `config_manager.py` - ~200 lines
  - `launcher.py` - ~400 lines
  - `client_detector.py` - ~200 lines
  - `instance_manager.py` - ~800 lines
  - `statistics.py` - ~400 lines
  - `widgets.py` - ~100 lines
  - `logging_utils.py` - ~50 lines

**Total**: ~3339 lines (26% reduction + better organization)
