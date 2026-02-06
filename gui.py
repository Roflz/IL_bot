#!/usr/bin/env python3
"""
Simple Recorder GUI
==================

A minimal GUI for running plans with the simple_recorder system.
Allows selecting plans, running multiple plans in sequence, and pausing between plans.

This file is now a simple entry point that imports the main GUI from gui.main_window.
"""

import tkinter as tk
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main GUI class from the modular structure
from gui.main_window import SimpleRecorderGUI

# Re-export PlanEntry for backwards compatibility
from gui.plan_editor import PlanEntry

# Keep the old PlanEditor class for backwards compatibility
# (It was moved to gui.plan_editor, but we keep a reference here)
from gui.plan_editor import PlanEditor


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = SimpleRecorderGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()


if __name__ == "__main__":
    main()
