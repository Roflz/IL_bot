# 🎮 OSRS Mouse & Keyboard Recorder

A Python application that records mouse movements, clicks, scrolls, and keyboard events for OSRS bot training data. **This recorder exactly matches the behavior of the original `gui_minimal.py`**.

## ✨ Features

- **Mouse Recording**: Movements (throttled to 10ms), clicks (press only), and scrolls
- **Keyboard Recording**: Both key presses and key releases
- **Window Detection**: Automatically finds and tracks Runelite windows
- **Focus-Based Recording**: Only records when Runelite window is focused
- **Relative Coordinates**: Mouse coordinates are relative to the Runelite window
- **Real-Time CSV Writing**: Actions are written immediately as they happen
- **Modifier Tracking**: Records shift, ctrl, alt, cmd key states
- **Active Key Tracking**: Tracks currently held keys
- **Unix Timestamps**: Uses millisecond timestamps since Unix epoch

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `pynput>=1.7.0` - Mouse and keyboard input monitoring
- `pyautogui>=0.9.0` - GUI automation
- `pygetwindow>=0.0.9` - Window detection and management

## 🚀 Usage

1. **Run the recorder**:
   ```bash
   python mouse_recorder.py
   ```

2. **Detect Runelite window**:
   - Click "🔍 Detect Runelite Window"
   - Make sure Runelite is running and visible

3. **Start recording**:
   - Click "▶ Start Recording"
   - Focus on your Runelite window

4. **Perform actions**:
   - Move mouse, click, type, scroll in Runelite
   - Only actions in the focused Runelite window are recorded

5. **Stop recording**:
   - Click "⏹ Stop Recording"
   - Actions are automatically saved to `data/actions.csv`

## 📊 Output Format

The recorder creates `data/actions.csv` with the exact same format as the original GUI:

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Unix timestamp (milliseconds) | `1755783809786` |
| `event_type` | Type of event | `move`, `click`, `scroll`, `key_press`, `key_release` |
| `x_in_window` | X coordinate relative to window | `1456` |
| `y_in_window` | Y coordinate relative to window | `885` |
| `btn` | Mouse button (for clicks) | `left`, `right` |
| `key` | Key pressed/released | `a`, `shift`, `enter` |
| `scroll_dx` | Horizontal scroll delta | `0` |
| `scroll_dy` | Vertical scroll delta | `1` |
| `modifiers` | Active modifier keys | `shift+ctrl` |
| `active_keys` | Currently held keys | `w+a` |

## 🎯 Event Types

### Mouse Events
- **`move`**: Mouse movement (throttled to 10ms intervals)
- **`click`**: Mouse button press (no release events recorded)
- **`scroll`**: Mouse wheel scroll

### Keyboard Events  
- **`key_press`**: Key press down
- **`key_release`**: Key release up

## 🔧 Technical Details

### Recording Behavior
- **Mouse movements**: Throttled to 10ms to avoid flooding
- **Mouse clicks**: Only button presses are recorded (no releases)
- **Mouse scrolls**: Recorded immediately when they occur
- **Keyboard events**: Both press and release are recorded
- **Window focus**: Only records when Runelite window is active

### Coordinate System
- All mouse coordinates are converted to be relative to the Runelite window's top-left corner
- This ensures coordinates are consistent regardless of window position

### CSV Writing
- Actions are written to CSV immediately as they occur
- Uses `csvf.flush()` to ensure data is written to disk
- File is opened in append mode to preserve existing data

## 🐛 Troubleshooting

### Window Detection Issues
1. **Runelite not found**: Make sure Runelite is running and visible
2. **Wrong window detected**: Use the "🧪 Test Window Detection" button for debugging
3. **Focus issues**: Ensure Runelite window is active/focused

### Recording Issues
1. **No actions recorded**: Check if Runelite window is focused
2. **CSV file errors**: Ensure `data/` directory exists and is writable
3. **Performance issues**: Mouse movement throttling is set to 10ms

### Common Runelite Window Titles
- `Runelite - username`
- `RuneLite - username` 
- `RuneLite` (no username)
- `Runelite` (no username)
- Any title containing `runelite` or `runescape`

## 📁 File Structure

```
bot_runelite_IL/
├── mouse_recorder.py          # Main recorder application
├── mouse_recorder_README.md   # This file
├── data/
│   └── actions.csv           # Recorded actions (created automatically)
└── requirements.txt           # Python dependencies
```

## 🔄 Differences from Original

This recorder is **functionally identical** to the original `gui_minimal.py`:

✅ **Same event types**: `move`, `click`, `scroll`, `key_press`, `key_release`  
✅ **Same CSV format**: 10 columns including `modifiers` and `active_keys`  
✅ **Same recording behavior**: Mouse moves throttled, clicks press-only, keys both ways  
✅ **Same window focus logic**: Only records when Runelite is focused  
✅ **Same coordinate system**: Relative to Runelite window  
✅ **Same real-time writing**: Immediate CSV updates with `flush()`  

## 🎯 Use Cases

- **Bot Training Data**: Record human gameplay for imitation learning
- **Input Analysis**: Analyze player interaction patterns
- **Testing**: Verify bot behavior against human input
- **Research**: Study OSRS player behavior and timing

## 📝 Notes

- The recorder automatically creates the `data/` directory if it doesn't exist
- Actions are appended to existing CSV files (doesn't overwrite)
- All timestamps use Unix milliseconds for consistency
- Window detection supports multiple Runelite instances
- The GUI updates action counts in real-time during recording
