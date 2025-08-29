#!/usr/bin/env python3
"""
Training Data Browser - Thin Wrapper

This is a thin wrapper that calls the modularized training browser application.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import training_browser
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ilbot.ui.training_browser.app import main

if __name__ == "__main__":
    main()
