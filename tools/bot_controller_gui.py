#!/usr/bin/env python3
"""
Bot Controller GUI - Thin Shim

This file is now a thin shim that imports and runs the new modular GUI
from the botgui package. The new GUI provides:

- Live Feature Tracking: Shows rolling 10x128 features with translations
- Predictions: Displays predicted action frames with consistent enum labels
- Modular architecture with proper separation of concerns
- Thread-safe UI updates using queues and dispatchers
- Clean, maintainable code structure

To run the GUI:
    python tools/bot_controller_gui.py

The new implementation is located in the botgui/ package and follows
MVC-like architecture with proper threading discipline.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point - import and run the new GUI"""
    try:
        # Import the new GUI
        from botgui.app import main as run_gui
        
        # Run the GUI
        run_gui()
        
    except ImportError as e:
        print(f"Error: Failed to import the new GUI: {e}")
        print("Make sure you're running from the project root directory.")
        print("Current directory:", Path.cwd())
        sys.exit(1)
        
    except Exception as e:
        print(f"Error running the GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
