#!/usr/bin/env python3
"""
Simple Recorder GUI Launcher
============================

Launches the Simple Recorder GUI for plan management.
"""

import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui import main

if __name__ == "__main__":
    main()
