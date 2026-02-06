"""
Logging Utilities Module (PySide6)
===================================

Centralized logging within the GUI using PySide6.
"""

from PySide6.QtWidgets import QTextEdit, QApplication
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor, QPalette
import time
from typing import Optional


class LoggingUtils:
    """Utility class for logging messages to text widgets."""
    
    @staticmethod
    def _get_theme_aware_colors():
        """
        Get theme-aware colors based on current application palette.
        Returns colors that have good contrast with the current theme.
        """
        app = QApplication.instance()
        if not app:
            # Fallback colors if no app instance
            return {
                'info': QColor(200, 200, 200),  # Light gray for dark theme
                'success': QColor(100, 255, 100),  # Bright green
                'error': QColor(255, 100, 100),  # Bright red
                'warning': QColor(255, 200, 100)  # Bright orange/yellow
            }
        
        palette = app.palette()
        window_bg = palette.color(QPalette.ColorRole.Window)
        window_text = palette.color(QPalette.ColorRole.WindowText)
        
        # Determine if we're in dark or light theme
        is_dark = window_bg.lightness() < 128
        
        if is_dark:
            # Dark theme - use bright colors
            return {
                'info': QColor(220, 220, 220),  # Light gray/white for readability
                'success': QColor(100, 255, 100),  # Bright green
                'error': QColor(255, 100, 100),  # Bright red
                'warning': QColor(255, 200, 100)  # Bright orange/yellow
            }
        else:
            # Light theme - use darker colors
            return {
                'info': QColor(30, 30, 30),  # Dark gray/black for readability
                'success': QColor(0, 150, 0),  # Dark green
                'error': QColor(200, 0, 0),  # Dark red
                'warning': QColor(200, 120, 0)  # Dark orange
            }
    
    @staticmethod
    def log_message(text_widget: QTextEdit, message: str, level: str = 'info'):
        """
        Log a message to a text widget with theme-aware color coding.
        
        Args:
            text_widget: QTextEdit widget to log to
            message: Message to log
            level: Log level ('info', 'success', 'error', 'warning')
        """
        if not text_widget:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Get theme-aware colors
        color_map = LoggingUtils._get_theme_aware_colors()
        color = color_map.get(level, color_map['info'])
        
        # Create format
        format = QTextCharFormat()
        format.setForeground(color)
        
        # Move cursor to end and insert text
        cursor = text_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        text_widget.setTextCursor(cursor)
        text_widget.setCurrentCharFormat(format)
        text_widget.insertPlainText(formatted_message)
        
        # Auto-scroll to bottom
        cursor = text_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        text_widget.setTextCursor(cursor)
    
    @staticmethod
    def log_message_to_instance(instance_name: str, text_widget: QTextEdit, message: str, level: str = 'info'):
        """Log a message to an instance's text widget."""
        LoggingUtils.log_message(text_widget, f"[{instance_name}] {message}", level)
