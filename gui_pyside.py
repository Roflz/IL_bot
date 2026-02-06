#!/usr/bin/env python3
"""
Simple Recorder GUI (PySide6)
==============================

Entry point for PySide6 version of the GUI.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QStyleFactory
from PySide6.QtGui import QPalette, QColor

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main GUI class from the PySide6 version
from gui.main_window_pyside import SimpleRecorderGUI


def apply_dark_theme(app: QApplication):
    """Apply a dark theme to the application using QPalette."""
    # Use Fusion style (cross-platform, supports custom palettes well)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Create dark palette
    palette = QPalette()
    
    # Window (background)
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    
    # Base (input fields background)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    
    # Text
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    
    # Button
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    
    # Highlight (selected items)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Tooltip
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    
    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(127, 127, 127))
    
    # Link
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(130, 42, 218))
    
    # Apply the palette
    app.setPalette(palette)


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Set application properties
    app.setApplicationName("Simple Recorder")
    app.setOrganizationName("Simple Recorder")
    
    # Create and show main window
    window = SimpleRecorderGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
