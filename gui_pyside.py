#!/usr/bin/env python3
"""
Simple Recorder GUI (PySide6)
==============================

Entry point for the PySide6 GUI. Normally run via the flez-bot root launcher
(launcher.py), which handles updates and readiness before starting this.
For development you can run this file directly from bot_runelite_IL.
"""

import sys
from pathlib import Path

# Ensure bot_runelite_IL is on the path so "from gui.main_window_pyside" works
_bot_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_bot_root))

from PySide6.QtWidgets import QApplication, QStyleFactory
from PySide6.QtGui import QPalette, QColor

from gui.main_window_pyside import SimpleRecorderGUI


def apply_dark_theme(app: QApplication):
    """Apply a dark theme to the application using QPalette."""
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(130, 42, 218))
    app.setPalette(palette)


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    app.setApplicationName("Simple Recorder")
    app.setOrganizationName("Simple Recorder")
    window = SimpleRecorderGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
