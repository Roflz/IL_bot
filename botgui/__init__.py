#!/usr/bin/env python3
"""Bot Controller GUI Package

A modular GUI application for controlling and monitoring the RuneLite bot,
built with Tkinter and following MVC-like architecture.
"""

__version__ = "1.0.0"
__author__ = "Bot Development Team"

# Public API
from .app import main
from .controller import BotController
from .state import UIState, RuntimeState

__all__ = [
    'main',
    'BotController',
    'UIState',
    'RuntimeState'
]
