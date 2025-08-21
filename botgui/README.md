# BotGUI Package

A modular, maintainable GUI application for controlling and monitoring the RuneLite bot, built with Tkinter and following MVC-like architecture.

## Features

### 🎯 **Live Feature Tracking**
- **Rolling 10x128 Feature Window**: Displays the last 10 timesteps of 128-dimensional feature vectors
- **Smart Translations**: Toggle between raw values and human-readable labels using `id_mappings.json`
- **Group Filtering**: Filter features by logical groups (Player, Environment, etc.)
- **Search Functionality**: Find specific features by name
- **Real-time Updates**: Features update automatically when live mode is active

### 🤖 **Model Predictions**
- **Action Frame Display**: Shows predicted flattened 600ms action frames
- **Consistent Labels**: Uses exact encodings from `shared_pipeline.encodings`
- **Prediction Controls**: Enable/disable predictions, track user input
- **Export Functionality**: Save predictions to CSV format
- **Buffer Status**: Visual indication of when buffers are "warm" (10/10 features + actions)

### 📊 **Live View & Monitoring**
- **Screenshot Capture**: Real-time game window capture and display
- **Region Selection**: Manual region selection for focused monitoring
- **Status Monitoring**: Comprehensive logging and status displays
- **Buffer Management**: Clear and manage feature/action buffers

### 🏗️ **Architecture**
- **MVC-like Pattern**: Clear separation between Model (services), View (UI), and Controller
- **Thread Safety**: No Tk calls from worker threads; uses queues + `root.after()`
- **Modular Design**: Small, focused packages for maintainability
- **Service Layer**: Clean abstraction for data processing and model inference

## Package Structure

```
botgui/
├── __init__.py              # Package initialization and public API
├── app.py                   # Main application entry point
├── controller.py            # Main controller orchestrating services + views
├── state.py                 # Dataclasses for UI + runtime state
├── README.md                # This file
├── services/                # Business logic and data processing
│   ├── __init__.py
│   ├── feature_pipeline.py  # Processes gamestate → features using shared_pipeline
│   ├── live_source.py       # Abstract interface for live data sources
│   ├── mapping_service.py   # Hash/ID translations using id_mappings.json
│   ├── predictor.py         # Wraps trained model for inference
│   └── window_finder.py     # Game window detection and management
├── ui/                      # User interface components
│   ├── __init__.py
│   ├── main_window.py       # Main window with menu, toolbar, notebook
│   ├── styles.py            # Consistent theming and styling
│   ├── views/               # Individual tab implementations
│   │   ├── __init__.py
│   │   ├── live_features_view.py    # Live Feature Tracking tab
│   │   ├── predictions_view.py      # Predictions tab
│   │   ├── logs_view.py             # Bot Logs tab
│   │   └── live_view.py             # Live View tab
│   └── widgets/             # Reusable UI components
│       ├── __init__.py
│       └── tree_with_scrollbars.py  # Treeview with integrated scrollbars
└── util/                    # Utility functions
    ├── __init__.py
    ├── formatting.py        # Value formatting and display utilities
    └── queues.py            # Thread-safe queues and message dispatcher
```

## Key Components

### Controller (`controller.py`)
- **Orchestrates** all services and views
- **Manages** worker threads for live data and predictions
- **Handles** UI updates via message queues
- **Provides** public API for view interactions

### Services
- **FeaturePipeline**: Uses `shared_pipeline` modules to process gamestate data
- **LiveSource**: Abstract interface for file polling, plugin callbacks, or mock data
- **MappingService**: Translates raw feature values to human-readable labels
- **PredictorService**: Wraps trained model for real-time inference
- **WindowFinder**: Detects and manages game windows

### Views
- **LiveFeaturesView**: Displays rolling 10x128 feature window with filtering
- **PredictionsView**: Shows predicted action frames with formatting
- **LogsView**: Comprehensive logging and status monitoring
- **LiveView**: Screenshot capture and region preview

### Utilities
- **Formatting**: Consistent value display and translation
- **Queues**: Thread-safe message passing between workers and UI

## Usage

### Running the GUI
```bash
# From project root
python tools/bot_controller_gui.py

# Or directly
python -m botgui.app
```

### Testing the Package
```bash
# Run basic functionality tests
python test_botgui.py
```

### Key Controls
- **Ctrl+L**: Start live mode
- **Ctrl+S**: Stop live mode  
- **Ctrl+P**: Toggle predictions
- **Ctrl+R**: Refresh all views
- **Ctrl+M**: Load model

## Data Flow

### Live Mode
1. **LiveSource** polls for gamestate/actions (2 Hz)
2. **FeaturePipeline** processes raw data using `shared_pipeline` modules
3. **Queues** pass updates to UI thread
4. **Views** update displays via `root.after()` callbacks

### Predictions
1. **FeaturePipeline** maintains rolling 10x128 + 10 action frames
2. **PredictorService** runs inference when buffers are warm (600ms interval)
3. **Results** queued to UI thread for display
4. **PredictionsView** formats and displays action frames

## Configuration

### Feature Mappings
- Loaded from `data/05_mappings/feature_mappings.json`
- Defines 128 features with names, groups, and descriptions
- Used for display labels and filtering

### ID Mappings
- Loaded from `data/05_mappings/id_mappings.json`
- Provides hash/ID → label translations
- Supports "Global" and group-specific mappings

### Model Loading
- Supports PyTorch `.pth` files
- Automatically detects CUDA/CPU availability
- Loads `ImitationHybridModel` architecture

## Threading Model

### Worker Threads
- **LiveSourceWorker**: Polls data sources (2 Hz)
- **PredictorWorker**: Runs model inference (600ms interval)

### UI Updates
- **No direct Tk calls** from worker threads
- **Message queues** pass data to UI thread
- **`root.after()`** processes updates every 100ms
- **Coalescing** prevents UI spam from rapid updates

## Dependencies

### Required
- Python 3.7+
- tkinter (usually included with Python)
- numpy
- pathlib (Python 3.4+)

### Optional
- torch (for model inference)
- cv2 (for screenshot capture)
- PIL (for image processing)
- pygetwindow (for window detection)

## Development

### Adding New Views
1. Create view class inheriting from `ttk.Frame`
2. Implement required methods (`update_data`, `clear`, etc.)
3. Add to `MainWindow._build_notebook()`
4. Register with controller if needed

### Adding New Services
1. Create service class with clear interface
2. Add to `Controller._init_services()`
3. Wire into appropriate worker threads
4. Update state management as needed

### Styling
- Use `botgui.ui.styles` for consistent theming
- Follow established color palette and font schemes
- Apply alternating row colors to tables

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure you're running from project root
- **Missing Dependencies**: Install required packages
- **File Not Found**: Check data directory structure
- **UI Freezes**: Verify no Tk calls from worker threads

### Debug Mode
- Check `bot_controller.log` for detailed logging
- Enable debug logging in `app.py`
- Use status bar indicators for system state

## Migration from Old GUI

The new `botgui` package replaces the previous `bot_controller_gui.py` implementation:

- **Clean Architecture**: Modular, maintainable code structure
- **Thread Safety**: Proper separation of UI and worker threads
- **Feature Parity**: All functionality preserved and enhanced
- **Better UX**: Improved filtering, search, and display options

The old GUI file is now a thin shim that imports and runs the new implementation.

## Future Enhancements

- **Plugin System**: Support for custom data sources and processors
- **Advanced Filtering**: Regex search, value range filters
- **Data Export**: Additional formats (JSON, Excel)
- **Performance Monitoring**: CPU/GPU usage, memory tracking
- **Configuration UI**: Settings panel for customization
- **Theme Support**: Light/dark mode, custom color schemes
