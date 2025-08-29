# Training Browser - Modular Architecture

This is the modularized version of the Training Data Browser GUI, refactored for better separation of concerns and maintainability.

## Architecture Overview

The application follows a modular architecture with clear separation between:

- **Services**: Business logic and data operations
- **UI Components**: View widgets and user interface
- **Controller**: Application state and coordination
- **Utilities**: Helper functions and common operations

## Directory Structure

```
training_browser/
├── __init__.py              # Package initialization
├── app.py                   # Main application entry point
├── controller.py            # Application controller
├── state.py                 # UI state management
├── services/                # Business logic services
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and management
│   ├── mapping_service.py   # Hash/ID translation
│   ├── normalization_service.py # Normalization handling
│   ├── action_decoder.py    # Action data decoding
│   └── feature_catalog.py   # Feature metadata management
├── ui/                      # User interface components
│   ├── __init__.py
│   ├── main_window.py       # Main window and tab management
│   ├── styles.py            # UI styling and colors
│   ├── dialogs.py           # Dialog boxes
│   └── widgets/             # Reusable UI widgets
│       ├── __init__.py
│       ├── scrollable_frame.py # Scrollable frame widget
│       ├── feature_table.py     # Feature data table
│       ├── target_view.py       # Target actions view
│       ├── action_tensors_view.py # Action tensors view
│       ├── sequence_alignment_view.py # Sequence alignment
│       ├── feature_analysis_view.py # Feature analysis
│       └── normalization_view.py # Normalization analysis
└── util/                    # Utility functions
    ├── __init__.py
    ├── formatting.py        # Value formatting and export
    ├── filters.py           # Feature filtering logic
    └── tooltips.py          # Tooltip management
```

## Key Components

### Services Layer

- **DataLoader**: Loads training data artifacts from organized directories
- **MappingService**: Translates hash/ID values to human-readable labels
- **NormalizationService**: Handles normalized data views and fallback computation
- **ActionDecoder**: Decodes action tensors into readable formats
- **FeatureCatalog**: Manages feature metadata and grouping

### UI Layer

- **MainWindow**: Orchestrates all tabs and navigation
- **FeatureTableView**: Displays 10×128 feature data with filtering
- **TargetView**: Shows flattened action targets
- **ActionTensorsView**: Displays action tensor data
- **Supporting Views**: Additional analysis and comparison views

### Controller Layer

- **Controller**: Manages application state and coordinates between services and views
- **UIState**: Dataclass containing all UI state variables

## Data Flow

1. **Data Loading**: `DataLoader.load_all()` loads all artifacts from data root
2. **Service Initialization**: Controller initializes services with loaded data
3. **View Registration**: Views register with controller for updates
4. **User Interaction**: Controller handles events and updates state
5. **View Updates**: Views refresh based on state changes

## Usage

### Running the Application

```bash
python tools/browse_training_data.py
```

### Data Directory Structure

The application expects data in the following organized structure:

```
data/
├── 01_raw_data/            # Raw features and action data
├── 02_trimmed_data/        # Trimmed sequences
├── 03_normalized_data/     # Normalized features and actions
├── 04_sequences/           # Input sequences and targets
└── 05_mappings/            # Feature and ID mappings
```

### Key Features

- **Feature Browsing**: View 10×128 feature sequences with filtering
- **Action Analysis**: Examine target actions and action tensors
- **Data Comparison**: Compare raw vs. normalized data
- **Export Capabilities**: Export data to CSV and other formats
- **Flexible Data Roots**: Change data source directories dynamically

## Testing

Run the test suite:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python tests/test_loader_smoke.py
python tests/test_mapping_translate.py
python tests/test_formatting.py
python tests/test_feature_table_adapter.py
```

## Development

### Adding New Views

1. Create a new view class in `ui/widgets/`
2. Implement `refresh()` and `update()` methods
3. Register the view in `MainWindow._initialize_views()`
4. Add any necessary controller methods

### Adding New Services

1. Create a new service class in `services/`
2. Implement the service interface
3. Initialize the service in `Controller._initialize_services()`
4. Add getter methods to the controller

### Styling

- Colors and fonts are defined in `ui/styles.py`
- Use the provided style functions for consistent appearance
- Feature group colors are automatically applied

## Migration Notes

This modular version maintains exact behavioral parity with the original monolithic `browse_training_data.py`:

- Same file formats and data structures
- Identical feature ordering and encodings
- Same normalization behavior
- Compatible with existing training data

The main differences are:
- Better code organization and maintainability
- Cleaner separation of concerns
- More testable architecture
- Easier to extend with new features

## Dependencies

- **Core**: tkinter, numpy, pathlib
- **Shared Pipeline**: Uses existing `shared_pipeline/` modules
- **Testing**: unittest, tempfile, shutil
- **Optional**: pyperclip (for clipboard operations)
