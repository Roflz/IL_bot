"""
GUI Module
==========

Modular GUI components for the Simple Recorder Plan Runner.

Module Structure:
- main_window.py: Main GUI window orchestrator
- plan_editor.py: Plan parameter editing dialog
- config_manager.py: Configuration loading/saving
- launcher.py: RuneLite instance launcher
- client_detector.py: Automatic client detection
- instance_manager.py: Instance tab and plan execution management
- statistics.py: Statistics display and monitoring
- widgets.py: Reusable widget factories and styles
- logging_utils.py: Logging utilities for GUI
"""

from gui.main_window import SimpleRecorderGUI
from gui.plan_editor import PlanEditor, PlanEntry
from gui.config_manager import ConfigManager
from gui.launcher import RuneLiteLauncher
from gui.client_detector import ClientDetector
from gui.instance_manager import InstanceManager
from gui.statistics import StatisticsDisplay
from gui.widgets import WidgetFactory
from gui.logging_utils import LoggingUtils

__all__ = [
    'SimpleRecorderGUI',
    'PlanEditor',
    'PlanEntry',
    'ConfigManager',
    'RuneLiteLauncher',
    'ClientDetector',
    'InstanceManager',
    'StatisticsDisplay',
    'WidgetFactory',
    'LoggingUtils',
]
