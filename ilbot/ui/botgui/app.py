#!/usr/bin/env python3
"""Main application entry point for Bot Controller GUI"""

import tkinter as tk
from tkinter import ttk
import logging
import sys
from pathlib import Path

# Import the controller and main window
from .controller import BotController
from .ui.main_window import MainWindow

# Import styles
from .ui.styles import apply_dark_theme

# Import logging setup
from .logging_setup import init_logging

LOG = logging.getLogger(__name__)


def create_root_window() -> tk.Tk:
    """Create and configure the root Tkinter window"""
    try:
        # Initialize logging before building any UI
        init_logging()
        LOG = logging.getLogger(__name__)
        LOG.debug("app: logging initialized")
        
        # Create root window
        root = tk.Tk()
        
        # Configure window properties
        root.title("Bot Controller GUI")
        root.iconname("Bot Controller")
        
        # Set window icon if available
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                root.iconbitmap(str(icon_path))
        except Exception:
            pass  # Icon not critical
        
        # Configure window
        root.minsize(800, 600)
        root.resizable(True, True)
        
        # Apply dark theme immediately after creating root
        try:
            apply_dark_theme(root)
            LOG.info("Dark theme applied successfully")
        except Exception as e:
            LOG.exception("Failed to apply dark theme")
            raise RuntimeError(f"Dark theme application failed: {e}") from e
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        LOG.info("Root window created successfully")
        return root
        
    except Exception as e:
        LOG.exception("Failed to create root window")
        raise


def setup_exception_handling(root: tk.Tk):
    """Setup global exception handling for the GUI"""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle unhandled exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Handle Ctrl+C gracefully
            LOG.info("Application interrupted by user")
            root.quit()
            return
        
        # Log the exception
        LOG.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Show error dialog
        try:
            from tkinter import messagebox
            messagebox.showerror(
                "Error",
                f"An unexpected error occurred:\n\n{exc_type.__name__}: {exc_value}\n\n"
                "Please check the logs for more details.",
                parent=root
            )
        except Exception:
            # If we can't show the dialog, just print to console
            print(f"Error: {exc_type.__name__}: {exc_value}")
    
    # Set the exception handler
    sys.excepthook = handle_exception
    
    # Also handle Tkinter errors - use the local logger reference
    def handle_tk_error(exc_type, exc_value, exc_traceback):
        try:
            import traceback as tb
            # Get the full traceback as a string
            tb_lines = tb.format_exception(exc_type, exc_value, exc_traceback)
            tb_str = ''.join(tb_lines)
            
            LOG.error(f"Tkinter error: {exc_type.__name__}: {exc_value}")
            LOG.error(f"Full traceback:\n{tb_str}")
            
            # Also print to console for immediate visibility
            print(f"TKINTER ERROR: {exc_type.__name__}: {exc_value}")
            print(f"FULL TRACEBACK:\n{tb_str}")
            
        except Exception as debug_e:
            # Fallback if anything goes wrong with debug logging
            print(f"ERROR IN ERROR HANDLER: {debug_e}")
            print(f"Original Tkinter error: {exc_type.__name__}: {exc_value}")
            print(f"Original Traceback object: {exc_traceback}")
    
    root.report_callback_exception = handle_tk_error
    
    LOG.info("Exception handling configured")


def create_controller(root: tk.Tk) -> BotController:
    """Create and initialize the main controller"""
    
    try:
        controller = BotController(root)
        LOG.info("Controller created successfully")
        return controller
        
    except Exception as e:
        LOG.exception("Failed to create controller")
        raise


def create_main_window(root: tk.Tk, controller: BotController) -> MainWindow:
    """Create and initialize the main window"""
    
    try:
        main_window = MainWindow(root, controller)
        LOG.info("Main window created successfully")
        return main_window
        
    except Exception as e:
        LOG.exception("Failed to create main window")
        raise


def setup_cleanup(root: tk.Tk, controller: BotController):
    """Setup cleanup handlers for graceful shutdown"""
    
    def on_closing():
        """Handle window closing"""
        try:
            LOG.info("Application shutting down...")
            
            # Shutdown controller
            controller.shutdown()
            
            # Destroy window
            root.destroy()
            
            LOG.info("Application shutdown complete")
            
        except Exception as e:
            LOG.exception("Error during shutdown")
            root.destroy()
    
    # Bind closing event
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Handle Ctrl+C
    def handle_sigint(signum, frame):
        LOG.info("Received interrupt signal")
        on_closing()
    
    try:
        import signal
        signal.signal(signal.SIGINT, handle_sigint)
    except Exception:
        pass  # Signal handling not available on Windows
    
    LOG.info("Cleanup handlers configured")


def run_application():
    """Run the main application"""
    
    try:
        LOG.info("Starting Bot Controller GUI...")
        
        # Create root window
        root = create_root_window()
        
        # Setup exception handling
        setup_exception_handling(root)
        
        # Create controller
        controller = create_controller(root)
        
        # Create main window
        main_window = create_main_window(root, controller)
        
        # Setup cleanup
        setup_cleanup(root, controller)
        
        # Start the main loop
        LOG.info("Starting main event loop...")
        root.mainloop()
        
        LOG.info("Application exited normally")
        
    except Exception as e:
        LOG.exception("Failed to start application")
        
        # Show error dialog if possible
        try:
            from tkinter import messagebox
            messagebox.showerror(
                "Startup Error",
                f"Failed to start the application:\n\n{str(e)}\n\n"
                "Please check the logs for more details.",
                parent=root
            )
        except Exception:
            print(f"Startup Error: {e}")
        
        sys.exit(1)


def main():
    """Main entry point"""
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            print("Error: Python 3.7 or higher is required")
            sys.exit(1)
        
        # Note: shared_pipeline is now under ilbot/pipeline/shared_pipeline after reorganization
        # No need to check for it in the root directory
        
        # Run the application
        run_application()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
