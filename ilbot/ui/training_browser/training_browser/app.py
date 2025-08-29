"""
Main Application

Entry point for the training browser application.
"""

import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

from .controller import Controller
from .ui.main_window import MainWindow


def main():
    """Main entry point for the training browser application."""
    try:
        # Create root window
        root = tk.Tk()
        root.title("Training Browser")
        root.geometry("1200x800")
        
        # Set application icon if available
        try:
            # You can add an icon file here if desired
            pass
        except Exception:
            pass
        
        # Create controller
        controller = Controller(root)
        
        # Create main window
        main_window = MainWindow(root, controller)
        
        # Register views with controller
        controller.register_view('main_window', main_window)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Start main loop
        print("Starting Training Browser...")
        root.mainloop()
        
    except Exception as e:
        print(f"Fatal error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
