"""
Logging Utilities Module
=========================

Handles logging messages to GUI text widgets.
"""

import tkinter as tk
import time
from typing import Optional


class LoggingUtils:
    """Utilities for logging messages to GUI."""
    
    @staticmethod
    def log_message(log_text: tk.Text, message: str, level: str = 'info'):
        """
        Add a message to the log output.
        
        Args:
            log_text: Text widget to log to
            message: Message to log
            level: Log level ('info', 'error', 'success', 'warning')
        """
        log_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding based on level
        colors = {
            'error': '#e74c3c',
            'success': '#27ae60',
            'warning': '#f39c12',
            'info': '#34495e'
        }
        color = colors.get(level, '#34495e')
        
        # Insert message
        log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Apply color to the last line
        start_line = log_text.index("end-2l")
        end_line = log_text.index("end-1l")
        log_text.tag_add(level, start_line, end_line)
        log_text.tag_config(level, foreground=color)
        
        log_text.config(state=tk.DISABLED)
        log_text.see(tk.END)
    
    @staticmethod
    def log_message_to_instance(instance_tab, message: str, level: str = 'info'):
        """
        Add a message to a specific instance's log output.
        
        Args:
            instance_tab: Instance tab object with log_text attribute
            message: Message to log
            level: Log level
        """
        if not hasattr(instance_tab, 'log_text'):
            return
        
        LoggingUtils.log_message(instance_tab.log_text, message, level)
