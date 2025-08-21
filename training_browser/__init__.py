"""
Training Browser Package

A modular GUI for browsing training data with behavior-preserving refactoring.
"""

__version__ = "1.0.0"
__author__ = "Training Browser Team"

from .app import main
from .controller import Controller
from .state import UIState

__all__ = [
    "main",
    "Controller", 
    "UIState"
]
