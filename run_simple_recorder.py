#!/usr/bin/env python3
"""Launcher script for the simple recorder GUI."""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ilbot.ui.simple_recorder.app import run_simple_recorder

if __name__ == "__main__":
    run_simple_recorder()
