# Shared Pipeline for OSRS Bot Imitation Learning

This directory contains the modularized, reusable components extracted from the legacy `extract_features.py` and `phase1_data_preparation.py` scripts. All functions preserve exact behavior from the original code.

## Overview

The shared pipeline provides pure, reusable functions for:
- **Feature Extraction**: Convert gamestates to 128-dimensional feature vectors
- **Action Processing**: Encode and flatten action sequences for training
- **Sequence Building**: Create 10-timestep rolling windows
- **Data Normalization**: Apply coordinate system-aware normalization
- **I/O Operations**: Load and save training data artifacts

## Core Contracts

The shared pipeline maintains strict contracts to ensure compatibility:

- **Feature Vectors**: Dynamic feature count determined by gamestate structure
- **Sequence Windows**: 10 timesteps (6 seconds of context) 
- **Action Frames**: 600ms flattened vector with [count, Δt_ms, type, x, y, button, key, scroll_dx, scroll_dy × count] format
- **Data Types**: float64 for features, int for discrete values
- **Feature Mappings**: Automatically generated and saved to `feature_mappings.json`

## Module Structure

### `feature_map.py`
Handles loading and validation of `feature_mappings.json`:
```python
from shared_pipeline import load_feature_mappings, validate_feature_mappings

mappings = load_feature_mappings("data/features/feature_mappings.json")
validate_feature_mappings(mappings)
```

### `features.py`
Extracts 128-dimensional features from gamestates:
```python
from shared_pipeline import extract_features_from_gamestate, FeatureExtractor

# Single gamestate
features, mappings = extract_features_from_gamestate(gamestate)

# Multiple gamestates
extractor = FeatureExtractor()
extractor.initialize_session_timing(gamestates)
features_array = np.array([extractor.extract_features_from_gamestate(gs) for gs in gamestates])
```

### `encodings.py`
Derives action encodings from existing training data:
```python
from shared_pipeline import derive_encodings_from_data, ActionEncoder

encoder = derive_encodings_from_data("data/training_data/action_targets.json")
encoded_actions = encoder.encode_action_sequence(actions)
```

### `actions.py`
Processes action sequences and converts to training format:
```python
from shared_pipeline import extract_raw_action_data, convert_raw_actions_to_tensors

raw_actions = extract_raw_action_data(gamestates, "data/actions.csv")
action_targets = convert_raw_actions_to_tensors(raw_actions, encoder)
```

### `sequences.py`
Builds temporal sequences for training:
```python
from shared_pipeline import create_temporal_sequences, validate_sequence_structure

input_sequences, target_sequences, action_inputs = create_temporal_sequences(
    features, action_targets, sequence_length=10
)

is_valid = validate_sequence_structure(input_sequences, target_sequences, action_inputs)
```

### `normalize.py`
Applies coordinate system-aware normalization:
```python
from shared_pipeline import normalize_features, normalize_input_sequences

normalized_features = normalize_features(features, "data/features/feature_mappings.json")
normalized_sequences = normalize_input_sequences(sequences, "data/features/feature_mappings.json")
```

### `io_offline.py`
Handles loading and saving of training data:
```python
from shared_pipeline import (
    load_gamestates, load_existing_features, save_training_data, save_final_training_data
)

gamestates = load_gamestates("data/gamestates")
features = load_existing_features("data/features/state_features.npy")

save_training_data("output_dir", input_sequences, target_sequences, ...)
save_final_training_data("final_dir", normalized_sequences, action_inputs, targets)
```

## Usage Examples

### Basic Feature Extraction
```python
from shared_pipeline import extract_features_from_gamestate

# Extract features from a single gamestate
features, mappings = extract_features_from_gamestate(gamestate)
print(f"Features shape: {features.shape}")  # (128,)
```

### Complete Pipeline
```python
from shared_pipeline import (
    load_existing_features, load_action_targets, normalize_features,
    create_temporal_sequences, save_final_training_data
)

# Load data
features = load_existing_features("data/features/state_features.npy")
action_targets = load_action_targets("data/training_data/action_targets.json")

# Normalize features
normalized_features = normalize_features(features, "data/features/feature_mappings.json")

# Create sequences
input_sequences, target_sequences, action_inputs = create_temporal_sequences(
    normalized_features, action_targets
)

# Save final training data
save_final_training_data(
    "data/final_training_data",
    input_sequences,
    action_inputs,
    target_sequences
)
```

### Action Encoding
```python
from shared_pipeline import derive_encodings_from_data, extract_raw_action_data

# Derive encodings from existing data
encoder = derive_encodings_from_data("data/training_data/action_targets.json")

# Extract and encode new actions
gamestates = load_gamestates("data/gamestates")
raw_actions = extract_raw_action_data(gamestates, "data/actions.csv")
encoded_actions = convert_raw_actions_to_tensors(raw_actions, encoder)
```

## CLI Tools

### Build Training Data
```bash
# Use existing features (recommended)
python tools/build_offline_training_data.py --use-existing-features

# Extract features from gamestates
python tools/build_offline_training_data.py --extract-features

# Custom data directory
python tools/build_offline_training_data.py --data-dir /path/to/data --use-existing-features
```

### Simulate Live Execution
```bash
# Basic simulation
python tools/simulate_live_from_recording.py

# Faster simulation
python tools/simulate_live_from_recording.py --speedup 2.0

# Custom sequence length
python tools/simulate_live_from_recording.py --sequence-length 15
```

## Testing

Run all tests to verify the pipeline works correctly:
```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python -m unittest tests.test_parity_phase1_vs_shared

# Run with coverage
python -m pytest tests/ --cov=shared_pipeline
```

## Data Flow

1. **Input**: Gamestates (JSON) + Actions (CSV)
2. **Feature Extraction**: Gamestates → 128-D features
3. **Action Processing**: Actions → Training format targets
4. **Normalization**: Features → Normalized features
5. **Sequence Building**: Features → (N, 10, 128) sequences
6. **Output**: Training data artifacts

## File Formats

### Input
- **Gamestates**: JSON files in `data/gamestates/`
- **Actions**: CSV file at `data/actions.csv`
- **Feature Mappings**: JSON at `data/features/feature_mappings.json`

### Output
- **Training Data**: `data/training_data/` (debug artifacts)
- **Final Data**: `data/final_training_data/` (clean training data)

## Dependencies

- **NumPy**: For numerical arrays
- **Pandas**: For CSV processing
- **Pathlib**: For file operations
- **JSON**: For configuration files

## Notes

- All functions preserve exact behavior from legacy scripts
- Feature ordering and normalization match legacy exactly
- Action encodings are derived from existing training data
- Sequences require minimum 11 samples for 10-timestep windows
- Normalization preserves spatial relationships for coordinates
