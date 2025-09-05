"""Simple recorder GUI application."""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from datetime import datetime
try:
    from .main_window import SimpleRecorderWindow
except ImportError:
    # Fallback for when running as script directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from ilbot.ui.simple_recorder.main_window import SimpleRecorderWindow


def run_simple_recorder():
    """Launch the simple recorder GUI."""
    root = tk.Tk()
    root.title("Simple Bot Recorder")
    root.geometry("600x400")
    
    app = SimpleRecorderWindow(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    run_simple_recorder()
