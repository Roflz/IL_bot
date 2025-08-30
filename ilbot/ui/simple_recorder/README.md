# Simple Bot Recorder

A clean, simple GUI for recording bot training data without the complexity of the full botgui system.

## Features

- **Window Detection**: Automatically detects Runelite windows
- **Session Management**: Creates timestamped recording sessions
- **Input Recording**: Captures mouse movements, clicks, key presses/releases, and scrolls
- **Coordinate Tracking**: Records mouse coordinates for ALL events (including key presses)
- **Countdown Timer**: 5-second countdown before recording starts
- **Pause/Resume**: Pause and resume recording during sessions
- **Clean CSV Output**: Generates actions.csv files in the same format as the original system

## Usage

1. **Launch the GUI**:
   ```bash
   python run_simple_recorder.py
   ```

2. **Detect Window**: Click "Detect Runelite Window" to find your Runelite client

3. **Create Session**: Click "Create Session" to create a new recording session folder

4. **Start Recording**: Click "Start Recording" for a 5-second countdown, then recording begins

5. **Control Recording**: Use "Pause Recording" to pause/resume and "End Session" to stop

## File Structure

```
data/recording_sessions/
└── YYYYMMDD_HHMMSS/
    ├── gamestates/          # For future gamestate files
    └── actions.csv          # Recorded input data
```

## CSV Format

The actions.csv file contains:
- `timestamp`: Unix timestamp in milliseconds
- `event_type`: move, click, key_press, key_release, scroll
- `x_in_window`, `y_in_window`: Mouse coordinates relative to window
- `btn`: Mouse button (left, right, middle)
- `key`: Key pressed/released
- `scroll_dx`, `scroll_dy`: Scroll deltas
- `modifiers`: Modifier keys (future use)
- `active_keys`: Currently pressed keys (future use)

## Dependencies

- `tkinter`: Built-in Python GUI framework
- `pynput`: Input monitoring (mouse/keyboard)
- `pywin32`: Windows API access for window detection

## Key Differences from Original

- **Simplified**: No complex feature pipelines or live sources
- **Focused**: Only handles input recording and CSV generation
- **Clean**: No debug spam or complex threading
- **Reliable**: Direct input capture without intermediate services
- **Coordinates**: ALL events include mouse coordinates (solves the original issue)
