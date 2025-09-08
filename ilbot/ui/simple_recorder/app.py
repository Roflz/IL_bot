"""Simple recorder GUI application (multi-instance host)."""
import tkinter as tk

try:
    from .instances_manager import MultiInstanceHost
except ImportError:
    # Fallback for when running as script directly
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from ilbot.ui.simple_recorder.instances_manager import MultiInstanceHost


def run_simple_recorder():
    """Launch the simple recorder GUI."""
    root = tk.Tk()
    root.title("Simple Bot Recorder — Multi-Instance")
    root.geometry("1100x700")

    # Host notebook with an Add Instance button
    app = MultiInstanceHost(root)

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    run_simple_recorder()
