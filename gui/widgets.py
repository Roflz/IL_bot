"""
Widget Factory Module
=====================

Reusable widget creation helpers and style configuration.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


class WidgetFactory:
    """Factory for creating common GUI widgets."""
    
    @staticmethod
    def setup_styles():
        """Configure the GUI styles and colors."""
        style = ttk.Style()
        
        # Configure colors - gentle, comfortable scheme
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 16, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 10, 'bold'),
                       foreground='#34495e')
        
        style.configure('Info.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#7f8c8d')
        
        style.configure('Success.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#27ae60')
        
        style.configure('Error.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#e74c3c')
        
        # Configure buttons
        style.configure('Action.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Danger.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))
    
    @staticmethod
    def create_scrollable_frame(parent, width: Optional[int] = None, 
                                height: Optional[int] = None):
        """
        Create a scrollable frame with canvas and scrollbar.
        
        Returns:
            Tuple of (canvas, scrollable_frame, scrollbar)
        """
        import tkinter as tk
        from tkinter import ttk
        
        # Create canvas
        canvas = tk.Canvas(parent, width=width, height=height)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        return canvas, scrollable_frame, scrollbar
    
    @staticmethod
    def center_window(window):
        """Center a window on the screen."""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')
