#!/usr/bin/env python3
"""Main window for the Bot Controller GUI"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional
import logging
from .styles import create_dark_stringvar, create_dark_booleanvar

LOG = logging.getLogger(__name__)

# Import views
from .views.logs_view import LogsView
from .views.live_features_view import LiveFeaturesView
from .views.predictions_view import PredictionsView
from .views.live_view import LiveView



class MainWindow:
    """Main window that builds the UI and connects to the controller"""

    def __init__(self, root: tk.Tk, controller):
        self.root = root
        self.controller = controller

        # Configure the root window
        self._configure_root()

        # Build the UI
        self._build_menu()
        self._build_main_pane()
        self._build_status_bar()
        


        # Initialize view state
        self.live_view_visible = True  # Track live view visibility
        
        # Bind window events
        self._bind_events()

        # Set initial state
        self._update_ui_state()
        
        # Initialize missing variables that were in the controls panel
        self.region_var = create_dark_stringvar(self.root, value="800x600")
        self.window_info_label = ttk.Label(self.root, text="Window not detected", foreground="red", font=("TkDefaultFont", 9))

    def _configure_root(self):
        """Configure the root window"""
        # Set window size and position
        window_width = 1200
        window_height = 800
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(800, 600)

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)  # Main pane gets most space

    def _build_menu(self):
        """Build the main menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="Load Model...", command=self._load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data...", command=self._export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_application)

        # Control menu
        control_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Control", menu=control_menu)
        
        control_menu.add_command(label="Start Live Mode", command=self._start_live_mode)
        control_menu.add_command(label="Stop Live Mode", command=self._stop_live_mode)
        control_menu.add_separator()
        control_menu.add_command(label="Clear Buffers", command=self._clear_buffers)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # View toggle
        view_menu.add_command(label="Toggle Live View", command=self._toggle_live_view)
        view_menu.add_separator()
        
        # Create BooleanVar for Show Translations with proper master
        self.show_translations_var = create_dark_booleanvar(self.root, value=self.controller.ui_state.show_translations)
        view_menu.add_checkbutton(label="Show Translations", 
                                command=self._toggle_translations,
                                variable=self.show_translations_var)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Documentation", command=self._show_documentation)

    def _build_main_pane(self):
        """Build main UI: Live View dominates top; bottom = notebook only"""
        # Root container
        container = ttk.Frame(self.root)
        container.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        container.grid_rowconfigure(0, weight=1)     # PanedWindow expands
        container.grid_columnconfigure(0, weight=1)

        # Vertical paned window: top = live, bottom = notebook
        self.paned = ttk.PanedWindow(container, orient="vertical")
        self.paned.grid(row=0, column=0, sticky="nsew")

        # ---- Top: Live View ----
        self.live_frame = ttk.LabelFrame(self.paned, text="Live View", padding=(6, 6))
        self.live_frame.grid_rowconfigure(0, weight=1)
        self.live_frame.grid_columnconfigure(0, weight=1)

        # Single LiveView instance without toolbar
        self.views = {}
        self.views['live'] = LiveView(self.live_frame, controller=self.controller, show_toolbar=False)
        self.views['live'].grid(row=0, column=0, sticky="nsew")

        # ---- Bottom: Notebook frame only ----
        self.notebook_frame = ttk.Frame(self.paned)
        self.notebook_frame.grid_rowconfigure(0, weight=1)
        self.notebook_frame.grid_columnconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(self.notebook_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.views['logs'] = LogsView(self.notebook, controller=self.controller)
        self.notebook.add(self.views['logs'], text="Bot Logs")

        self.views['live_features'] = LiveFeaturesView(self.notebook, controller=self.controller)
        self.notebook.add(self.views['live_features'], text="Live Features")

        self.views['predictions'] = PredictionsView(self.notebook, controller=self.controller)
        self.notebook.add(self.views['predictions'], text="Predictions")

        # Add panes to vertical PanedWindow (top:bottom ≈ 70:30 for more bottom space)
        self.paned.add(self.live_frame, weight=7)
        self.paned.add(self.notebook_frame, weight=3)
        
        # Verify panes were added correctly
        panes = self.paned.panes()
        print(f"🔍 DEBUG [main_window.py:155] Panes after addition: {panes}")
        print(f"🔍 DEBUG [main_window.py:156] live_frame: {self.live_frame}")
        print(f"🔍 DEBUG [main_window.py:157] notebook_frame: {self.notebook_frame}")
        
        # Set minimum sizes after adding panes
        try:
            self.paned.paneconfig(self.live_frame, minsize=220)
            self.paned.paneconfig(self.notebook_frame, minsize=180)
            print("🔍 DEBUG [main_window.py:162] Pane configuration successful")
        except Exception as e:
            print(f"⚠️ WARNING [main_window.py:165] Pane configuration failed: {e}")
            pass

        def _set_vertical_sash():
            try:
                # Ensure geometry is realized before measuring
                self.root.update_idletasks()
                # Prefer paned height; fall back to parent/root if needed
                h = self.paned.winfo_height() or container.winfo_height() or self.root.winfo_height()
                # Put ~65% to the top pane; never let top be below 220px
                pos = max(220, int(h * 0.65))
                self.paned.sashpos(0, pos)
            except Exception:
                # Be robust: don't crash layout on startup
                pass

        # Run when idle and once more shortly after to catch late geometry changes
        self.root.after_idle(_set_vertical_sash)
        self.root.after(300, _set_vertical_sash)



        # Register views with controller
        self.controller.bind_views(
            live_view=self.views['live'],
            logs_view=self.views['logs'],
            features_view=self.views['live_features'],
            predictions_view=self.views['predictions'],
        )

        # Wire encoder to predictions view if available
        if hasattr(self.controller.feature_pipeline, 'action_encoder'):
            self.views['predictions'].set_action_encoder(
                self.controller.feature_pipeline.action_encoder
            )

        # Handle tab changes
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_status_bar(self):
        """Build the status bar at the bottom"""
        status_bar = ttk.Frame(self.root)
        status_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        status_bar.grid_columnconfigure(1, weight=1)

        # Left side - buffer status
        self.buffer_status_label = ttk.Label(status_bar, text="Buffers: 0/10 features, 0/10 actions")
        self.buffer_status_label.grid(row=0, column=0, sticky="w")

        # Center - current mode and status
        status_frame = ttk.Frame(status_bar)
        status_frame.grid(row=0, column=1, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        
        # Status text: "Status: Ready | Region: 800x600 | Position: (100, 200)"
        self.status_text = ttk.Label(status_frame, text="Status: Ready | Region: 800x600 | Position: (0, 0)")
        self.status_text.grid(row=0, column=0, sticky="w")
        
        # Detect Window button to the right of status
        self.detect_button = ttk.Button(status_frame, text="🔍 Detect Window", command=self._auto_detect_runelite)
        self.detect_button.grid(row=0, column=1, sticky="e", padx=(8, 0))

        # Right side - timestamp
        self.timestamp_label = ttk.Label(status_bar, text="Ready")
        self.timestamp_label.grid(row=0, column=2, sticky="e")

    def _bind_events(self):
        """Bind window events"""
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Keyboard shortcuts
        self.root.bind('<Control-l>', lambda e: self._start_live_mode())
        self.root.bind('<Control-s>', lambda e: self._stop_live_mode())
        self.root.bind('<Control-m>', lambda e: self._load_model())

    def _on_closing(self):
        """Handle window closing"""
        try:
            # Shutdown controller
            self.controller.shutdown()
            
            # Destroy window
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.root.destroy()

    def _on_tab_changed(self, event):
        """Handle notebook tab changes"""
        try:
            current_tab = self.notebook.select()
            tab_id = self.notebook.index(current_tab)
            tab_name = self.notebook.tab(tab_id, "text")
            
            # Update status
            self.mode_label.config(text=f"Current Tab: {tab_name}")
        except Exception:
            # Handle case where no tab is selected
            pass

    def _on_predictions_toggle(self):
        """Handle predictions toggle - placeholder since predictions are removed"""
        pass

    # Menu command handlers

    def _load_model(self):
        """Load a trained model"""
        try:
            filename = filedialog.askopenfilename(
                parent=self.root,
                title="Load Trained Model",
                filetypes=[
                    ("PyTorch models", "*.pth"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                model_path = Path(filename)
                success = self.controller.load_model(model_path)
                
                if success:
                    # Model loaded successfully, no need to update status text here
                    pass
                else:
                    messagebox.showerror("Error", f"Failed to load model: {model_path}", parent=self.root)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}", parent=self.root)

    def _export_data(self):
        """Export data to file"""
        try:
            filename = filedialog.asksaveasfilename(
                parent=self.root,
                title="Export Data",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                # TODO: Implement data export
                messagebox.showinfo("Info", "Data export not yet implemented.", parent=self.root)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {e}", parent=self.root)

    def _exit_application(self):
        """Exit the application"""
        self._on_closing()

    def _start_live_mode(self):
        """Start live mode"""
        try:
            self.controller.start_live_mode()
            self._update_ui_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live mode: {e}", parent=self.root)

    def _stop_live_mode(self):
        """Stop live mode"""
        try:
            self.controller.stop_live_mode()
            self._update_ui_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop live mode: {e}", parent=self.root)

    def _clear_buffers(self):
        """Clear all buffers"""
        try:
            self.controller.clear_buffers()
            messagebox.showinfo("Info", "Buffers cleared successfully", parent=self.root)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear buffers: {e}", parent=self.root)

    def _auto_detect_runelite(self):
        """Auto-detect Runelite window and update region"""
        try:
            logger.info("Starting Runelite window auto-detection...")
            logger.info("Searching for windows that start with 'RuneLite' (capital R and L)")
            
            # Use the window finder to detect Runelite windows
            print(f"🔍 MAIN WINDOW: Using window finder to detect Runelite windows...")
            runelite_windows = self.controller.window_finder.find_runelite_windows()
            print(f"🔍 MAIN WINDOW: Window finder returned {len(runelite_windows)} Runelite windows")
            logger.info(f"Window finder returned {len(runelite_windows)} Runelite windows")
            
            if not runelite_windows:
                logger.warning("No Runelite windows detected by window finder")
                logger.info("Make sure RuneLite is running and visible on your screen")
                
                # Update window info display
                self.window_info_label.config(text="Window not detected", foreground="red")
                
                # Update status text
                self.status_text.config(text="Status: Window not detected | Region: 800x600 | Position: (0, 0)")
                
                # Log to bot logs
                if 'logs' in self.views:
                    self.views['logs'].add_log_message("WARNING: No RuneLite window detected. Make sure RuneLite is running and visible on your screen.", "warning")
                
                messagebox.showwarning(
                    "No Runelite Window Found", 
                    "No Runelite windows detected.\n\nMake sure RuneLite is running and visible on your screen.",
                    parent=self.root
                )
                return
            
            # Get the best window (active, then non-minimized, then first available)
            print(f"🔍 MAIN WINDOW: Getting best Runelite window...")
            logger.info("Getting best Runelite window...")
            best_window = self.controller.window_finder.get_active_runelite_window()
            
            if not best_window:
                logger.error("Failed to get best Runelite window")
                
                # Update window info display
                self.window_info_label.config(text="Window not detected", foreground="red")
                
                # Log to bot logs
                if 'logs' in self.views:
                    self.views['logs'].add_log_message("ERROR: Failed to get Runelite window information.", "error")
                
                messagebox.showerror(
                    "Error", 
                    "Failed to get Runelite window information.",
                    parent=self.root
                )
                return
            
            logger.info(f"Selected Runelite window: {best_window['title']} ({best_window['width']}x{best_window['height']})")
            
            # Update the region display
            width = best_window['width']
            height = best_window['height']
            self.region_var.set(f"{width}x{height}")
            logger.info(f"Updated region display to {width}x{height}")
            
            # Update the LiveView region if it has one
            if 'live' in self.views and hasattr(self.views['live'], 'region'):
                self.views['live'].region = self.controller.window_finder.get_window_region(best_window)
                logger.info("Updated LiveView region")
                
                # Start live streaming by triggering a refresh after window is ready
                if hasattr(self.views['live'], '_refresh_display'):
                    logger.info("LiveView has _refresh_display method, scheduling refresh...")
                    
                    # Don't test the method call directly - it's too early
                    # Just schedule the delayed call to ensure root is ready
                    logger.info("Scheduling LiveView refresh to start streaming (delayed)")
                    self.root.after(500, self.views['live']._refresh_display)
                    logger.info("Scheduled LiveView refresh to start streaming (500ms delay)")
                else:
                    logger.error("LiveView missing _refresh_display method!")
                    logger.info(f"LiveView methods: {[m for m in dir(self.views['live']) if not m.startswith('_')]}")
            else:
                logger.warning("LiveView not available or missing region attribute")
            
            # Update window info display with success
            window_info_text = f"Found: {best_window['title']}\nSize: {width}x{height}\nPosition: ({best_window['left']}, {best_window['top']})"
            self.window_info_label.config(text=window_info_text, foreground="green")
            
            # Update status text
            status_text = f"Status: Window Detected | Region: {width}x{height} | Position: ({best_window['left']}, {best_window['top']})"
            self.status_text.config(text=status_text)
            
            # Log success to bot logs
            if 'logs' in self.views:
                self.views['logs'].add_log_message(
                    f"SUCCESS: RuneLite window detected!\n"
                    f"Title: {best_window['title']}\n"
                    f"Size: {width}x{height}\n"
                    f"Position: ({best_window['left']}, {best_window['top']})", 
                    "info"
                )
            
            # Show success message
            messagebox.showinfo(
                "Runelite Window Detected", 
                f"Found Runelite window: {best_window['title']}\n"
                f"Region: {width}x{height}\n"
                f"Position: ({best_window['left']}, {best_window['top']})",
                parent=self.root
            )
            
            # Start live streaming automatically
            try:
                if not self.controller.runtime_state.live_mode:
                    logger.info("Starting live mode automatically after window detection...")
                    self.controller.start_live_mode()
                    self._update_ui_state()
                    
                    # Log to bot logs
                    if 'logs' in self.views:
                        self.views['logs'].add_log_message("Live streaming started automatically after window detection.", "info")
                else:
                    logger.info("Live mode already running, no need to start")
            except Exception as e:
                logger.error(f"Failed to start live mode automatically: {e}")
                if 'logs' in self.views:
                    self.views['logs'].add_log_message(f"WARNING: Failed to start live mode automatically: {e}", "warning")
            
        except Exception as e:
            logger.error(f"Auto-detect Runelite failed: {e}", exc_info=True)
            print(f"ERROR: Auto-detect Runelite failed: {e}")
            
            # Update window info display
            self.window_info_label.config(text="Window not detected", foreground="red")
            
            # Log error to bot logs
            if 'logs' in self.views:
                self.views['logs'].add_log_message(f"ERROR: Auto-detect failed: {e}", "error")
            
            messagebox.showerror(
                "Error", 
                f"Failed to auto-detect Runelite window: {e}",
                parent=self.root
            )
            
            # Re-raise the exception to stop execution and show the real error
            raise

    def _try_auto_detect_on_startup(self):
        """Try to auto-detect Runelite window on startup (only once)"""
        # Only try once to avoid repeated attempts
        if hasattr(self, '_auto_detect_attempted'):
            return
        
        self._auto_detect_attempted = True
        logger.info("Attempting auto-detect on startup...")
        
        try:
            # Check if Runelite windows are available
            if self.controller.window_finder.is_runelite_window_available():
                logger.info("Runelite windows available, attempting auto-detect...")
                # Auto-detect without showing message boxes
                best_window = self.controller.window_finder.get_active_runelite_window()
                if best_window:
                    width = best_window['width']
                    height = best_window['height']
                    self.region_var.set(f"{width}x{height}")
                    logger.info(f"Startup auto-detect successful: {width}x{height}")
                    
                    # Update the LiveView region if it has one
                    if 'live' in self.views and hasattr(self.views['live'], 'region'):
                        self.views['live'].region = self.controller.window_finder.get_window_region(best_window)
                        logger.info("Updated LiveView region on startup")
                        
                        # Start live streaming by triggering a refresh after window is ready
                        if hasattr(self.views['live'], '_refresh_display'):
                            logger.info("Startup: LiveView has _refresh_display method, scheduling refresh...")
                            
                            # Don't test the method call directly - it's too early
                            # Just schedule the delayed call to ensure root is ready
                            logger.info("Startup: Scheduling LiveView refresh to start streaming (delayed)")
                            self.root.after(500, self.views['live']._refresh_display)
                            logger.info("Scheduled LiveView refresh on startup to start streaming (delayed)")
                        else:
                            logger.error("Startup: LiveView missing _refresh_display method!")
                            logger.info(f"Startup: LiveView methods: {[m for m in dir(self.views['live']) if not m.startswith('_')]}")
                    else:
                        logger.warning("LiveView not available or missing region attribute on startup")
                    
                    # Update window info display
                    window_info_text = f"Found: {best_window['title']}\nSize: {width}x{height}\nPosition: ({best_window['left']}, {best_window['top']})"
                    self.window_info_label.config(text=window_info_text, foreground="green")
                    
                    # Update status
                    status_text = f"Status: Auto-detected | Region: {width}x{height} | Position: ({best_window['left']}, {best_window['top']})"
                    self.status_text.config(text=status_text)
                    logger.info(f"Auto-detected Runelite window on startup: {best_window['title']} ({width}x{height})")
                else:
                    logger.warning("Startup auto-detect: best window is None")
                    self.window_info_label.config(text="Window not detected", foreground="red")
            else:
                logger.info("No Runelite windows available on startup")
                self.window_info_label.config(text="Window not detected", foreground="red")
                    
        except Exception as e:
            logger.error(f"Auto-detect on startup failed: {e}", exc_info=True)
            print(f"ERROR: Auto-detect on startup failed: {e}")
            self.window_info_label.config(text="Window not detected", foreground="red")
            
            # Re-raise the exception to stop execution and show the real error
            raise

    def _toggle_translations(self):
        """Toggle translations display"""
        # Update the controller's UI state
        self.controller.ui_state.show_translations = self.show_translations_var.get()
        
        # Update all views that need to refresh
        if 'live_features' in self.views:
            self.views['live_features'].update_translations_state(self.show_translations_var.get())

    def _toggle_live_view(self):
        """Toggle the live view visibility and auto-stretch tabs when hidden"""
        try:
            print(f"🔍 DEBUG [main_window.py:565] _toggle_live_view called - current state: {self.live_view_visible}")
            print(f"🔍 DEBUG [main_window.py:570] About to check live_view_visible: {self.live_view_visible}")
            
            # First, validate that we have a proper PanedWindow and panes
            print("🔍 DEBUG [main_window.py:575] Validating PanedWindow setup...")
            if not hasattr(self, 'paned') or not self.paned:
                error_msg = "PanedWindow not initialized"
                print(f"❌ ERROR [main_window.py:580] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Check if frames are actually added as panes
            print("🔍 DEBUG [main_window.py:585] Checking if frames are panes...")
            panes = self.paned.panes()
            print(f"🔍 DEBUG [main_window.py:590] Current panes: {panes}")
            print(f"🔍 DEBUG [main_window.py:591] live_frame: {self.live_frame}")
            print(f"🔍 DEBUG [main_window.py:592] notebook_frame: {self.notebook_frame}")
            print(f"🔍 DEBUG [main_window.py:593] live_frame in panes: {self.live_frame in panes}")
            print(f"🔍 DEBUG [main_window.py:594] notebook_frame in panes: {self.notebook_frame in panes}")
            
            if self.live_frame not in panes:
                error_msg = f"live_frame not found in panes: {panes}"
                print(f"❌ ERROR [main_window.py:595] {error_msg}")
                raise RuntimeError(error_msg)
                
            if self.notebook_frame not in panes:
                error_msg = f"notebook_frame not found in panes: {panes}"
                print(f"❌ ERROR [main_window.py:600] {error_msg}")
                raise RuntimeError(error_msg)
            
            print("🔍 DEBUG [main_window.py:605] Pane validation passed")
            
            if self.live_view_visible:
                print("🔍 DEBUG [main_window.py:610] Hiding live view...")
                
                # Hide the live view by setting its weight to 0
                print("🔍 DEBUG [main_window.py:615] About to set live_frame weight to 0...")
                self.paned.paneconfig(self.live_frame, weight=0)
                print("🔍 DEBUG [main_window.py:620] About to hide live_frame with grid_remove...")
                self.live_frame.grid_remove()  # Hide the frame
                print("🔍 DEBUG [main_window.py:625] About to set live_view_visible to False...")
                self.live_view_visible = False
                
                # Make the notebook frame fill the entire paned window
                print("🔍 DEBUG [main_window.py:630] About to set notebook_frame weight to 1...")
                self.paned.paneconfig(self.notebook_frame, weight=1)
                
                print("🔍 DEBUG [main_window.py:635] About to update status...")
                self.update_status("Live view hidden - tabs expanded", "info")
                print("🔽 Live view hidden - tabs now fill the entire window")
                print("🔍 DEBUG [main_window.py:640] Live view hide completed successfully")
                
            else:
                print("🔍 DEBUG [main_window.py:645] Showing live view...")
                
                # Show the live view by restoring its weight and making it visible
                print("🔍 DEBUG [main_window.py:650] About to show live_frame with grid()...")
                self.live_frame.grid()  # Show the frame
                print("🔍 DEBUG [main_window.py:655] About to set live_frame weight to 7...")
                self.paned.paneconfig(self.live_frame, weight=7)
                print("🔍 DEBUG [main_window.py:660] About to set notebook_frame weight to 3...")
                self.paned.paneconfig(self.notebook_frame, weight=3)
                
                # Restore minimum sizes
                print("🔍 DEBUG [main_window.py:665] About to restore minimum sizes...")
                try:
                    self.paned.paneconfig(self.live_frame, minsize=220)
                    self.paned.paneconfig(self.notebook_frame, minsize=180)
                    print("🔍 DEBUG [main_window.py:670] Minimum sizes restored successfully")
                except Exception as e:
                    print(f"⚠️ WARNING [main_window.py:675] Failed to restore minimum sizes: {e}")
                
                # Restore the sash position
                print("🔍 DEBUG [main_window.py:680] About to schedule sash position restoration...")
                self.root.after_idle(self._restore_sash_position)
                print("🔍 DEBUG [main_window.py:685] Sash position restoration scheduled")
                
                print("🔍 DEBUG [main_window.py:690] About to set live_view_visible to True...")
                self.live_view_visible = True
                print("🔍 DEBUG [main_window.py:695] About to update status...")
                self.update_status("Live view restored", "info")
                print("🔼 Live view restored - normal layout")
                print("🔍 DEBUG [main_window.py:700] Live view show completed successfully")
                
        except Exception as e:
            print(f"❌ ERROR [main_window.py:705] in _toggle_live_view: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: try to restore normal state
            print("🔍 DEBUG [main_window.py:710] Attempting fallback recovery...")
            try:
                self.live_frame.grid()
                self.paned.paneconfig(self.live_frame, weight=7)
                self.paned.paneconfig(self.notebook_frame, weight=3)
                self.live_view_visible = True
                print("🔍 DEBUG [main_window.py:715] Fallback recovery completed")
            except Exception as fallback_e:
                print(f"❌ CRITICAL ERROR [main_window.py:720] Fallback recovery failed: {fallback_e}")
                import traceback
                traceback.print_exc()
                # CRASH IMMEDIATELY - UI layout errors are critical
                raise RuntimeError(f"Failed to toggle live view and fallback recovery failed: {e} -> {fallback_e}") from e
    
    def _restore_sash_position(self):
        """Restore the sash position when showing live view"""
        try:
            print("🔍 DEBUG [main_window.py:610] _restore_sash_position called")
            print("🔍 DEBUG [main_window.py:615] About to call root.update_idletasks()...")
            self.root.update_idletasks()
            print("🔍 DEBUG [main_window.py:620] About to get paned window height...")
            h = self.paned.winfo_height()
            print(f"🔍 DEBUG [main_window.py:625] Paned window height: {h}")
            
            if h > 0:
                pos = max(220, int(h * 0.65))
                print(f"🔍 DEBUG [main_window.py:630] About to set sash position to: {pos}")
                self.paned.sashpos(0, pos)
                print("🔍 DEBUG [main_window.py:635] Sash position restored successfully")
            else:
                print("🔍 DEBUG [main_window.py:640] Paned window height is 0, skipping sash position restore")
                
        except Exception as e:
            print(f"❌ ERROR [main_window.py:645] in _restore_sash_position: {e}")
            import traceback
            traceback.print_exc()
            # CRASH IMMEDIATELY - sash position errors can cause UI freezes
            raise RuntimeError(f"Failed to restore sash position: {e}") from e

    def _show_about(self):
        """Show about dialog"""
        about_text = """Bot Controller GUI v1.0.0

A modular GUI application for controlling and monitoring the RuneLite bot.

Features:
• Live feature tracking with 10x128 rolling window
• Real-time model predictions
• Live screenshot capture and display
• Comprehensive logging and status monitoring

Built with Tkinter and following MVC architecture."""
        
        messagebox.showinfo("About", about_text, parent=self.root)

    def _show_documentation(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "Documentation not yet available.", parent=self.root)

    def _update_ui_state(self):
        """Update UI state based on controller state"""
        try:
            # Update status based on live mode
            if self.controller.runtime_state.live_mode:
                # Update status text to show live mode
                current_status = self.status_text.cget("text")
                if "Status:" in current_status:
                    # Extract region and position from current status
                    parts = current_status.split(" | ")
                    if len(parts) >= 3:
                        region = parts[1] if "Region:" in parts[1] else "Region: 800x600"
                        position = parts[2] if "Position:" in parts[2] else "Position: (0, 0)"
                        new_status = f"Status: Live Mode Active | {region} | {position}"
                        self.status_text.config(text=new_status)
            else:
                # Update status text to show ready
                current_status = self.status_text.cget("text")
                if "Status:" in current_status:
                    # Extract region and position from current status
                    parts = current_status.split(" | ")
                    if len(parts) >= 3:
                        region = parts[1] if "Region:" in parts[1] else "Region: 800x600"
                        position = parts[2] if "Position:" in parts[2] else "Position: (0, 0)"
                        new_status = f"Status: Ready | {region} | {position}"
                        self.status_text.config(text=new_status)
                
        except Exception as e:
            logger.error(f"Error updating UI state: {e}")
            # Update status text to show error
            self.status_text.config(text="Status: Error | Region: 800x600 | Position: (0, 0)")

    def update_status(self, message: str, level: str = "info"):
        """Update status display"""
        try:
            print(f"🔍 DEBUG [main_window.py:720] update_status called - message: '{message}', level: {level}")
            
            # Extract current region and position from status text
            print("🔍 DEBUG [main_window.py:725] About to get current status text...")
            current_status = self.status_text.cget("text")
            print(f"🔍 DEBUG [main_window.py:730] Current status text: '{current_status}'")
            
            parts = current_status.split(" | ")
            region = parts[1] if len(parts) > 1 else "Region: 800x600"
            position = parts[2] if len(parts) > 2 else "Position: (0, 0)"
            print(f"🔍 DEBUG [main_window.py:735] Extracted - region: '{region}', position: '{position}'")
            
            new_status = f"Status: {message} | {region} | {position}"
            print(f"🔍 DEBUG [main_window.py:740] About to set new status: '{new_status}'")
            
            self.status_text.config(text=new_status)
            print("🔍 DEBUG [main_window.py:745] Status updated successfully")

        except Exception as e:
            print(f"❌ ERROR [main_window.py:750] in update_status: {e}")
            import traceback
            traceback.print_exc()
            # CRASH IMMEDIATELY - status update errors can indicate UI problems
            raise RuntimeError(f"Failed to update status: {e}") from e

    def get_view(self, view_name: str):
        """Get a view by name"""
        return self.views.get(view_name)
