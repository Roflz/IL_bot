#!/usr/bin/env python3
"""
Standalone script to run the simple recorder GUI.
This script can be run from the project root directory.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the simple recorder
from ilbot.ui.simple_recorder.app import run_simple_recorder

if __name__ == "__main__":
    run_simple_recorder()
