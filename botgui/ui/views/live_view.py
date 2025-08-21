#!/usr/bin/env python3
"""Live View - displays live screenshot and region preview"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Tuple
import time
import logging
from ..styles import create_dark_canvas, create_dark_stringvar, create_dark_booleanvar
from ...services.window_finder import WindowFinder

logger = logging.getLogger(__name__)


class LiveView(ttk.Frame):
    """View for displaying live screenshots and region preview"""
    
    def __init__(self, parent, controller, show_toolbar: bool = True):
        super().__init__(parent)
        self.controller = controller
        self.show_toolbar = show_toolbar
        
        # Window detection
        self.window_finder = WindowFinder()
        self.detected_window = None
        
        # Image state
        self.current_image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.region: Tuple[int, int, int, int] = (0, 0, 800, 600)
        
        # Display state
        self.auto_refresh = True
        self.refresh_interval = 33  # ms (30 FPS instead of 10 FPS)
        self.last_refresh = 0
        
        # Performance optimizations
        self.skip_frames = 0  # Skip every Nth frame for performance
        self.frame_count = 0
        
        # UI elements
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[ttk.Label] = None
        self.window_status_label: Optional[ttk.Label] = None
        
        self._setup_ui()
        self._bind_events()
        
        # Debug logging
        # logger.info(f"LiveView initialized with show_toolbar={self.show_toolbar}")
        # logger.info(f"Initial region: {self.region}")
        # logger.info(f"Auto-refresh: {self.auto_refresh}")
        
        # Don't auto-detect window - wait for user to click Detect Window button
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Configure grid weights: window detection (0), status (1), canvas (2)
        self.grid_rowconfigure(0, weight=0)  # window detection - no expansion
        self.grid_rowconfigure(1, weight=0)  # status line - no expansion
        self.grid_rowconfigure(2, weight=1)  # canvas - expands to fill
        self.grid_columnconfigure(0, weight=1)  # full width
        
        # Determine row positions based on toolbar setting
        if self.show_toolbar:
            # With toolbar: toolbar(0), window detection(1), status(2), canvas(3)
            toolbar_row = 0
            window_detection_row = 1
            status_row = 2
            canvas_row = 3
            self.grid_rowconfigure(3, weight=1)  # canvas expands
            
            # Build toolbar frame
            toolbar_frame = ttk.Frame(self)
            toolbar_frame.grid(row=toolbar_row, column=0, sticky="ew", padx=8, pady=(0, 4))
            toolbar_frame.grid_columnconfigure(2, weight=1)
            
            # Left controls
            ttk.Button(toolbar_frame, text="📷 Capture", 
                      command=self._capture_screenshot).grid(row=0, column=0, padx=(0, 6))
            ttk.Button(toolbar_frame, text="💾 Save Image", 
                      command=self._save_image).grid(row=0, column=1, padx=(0, 12))
            
            # Center controls
            ttk.Label(toolbar_frame, text="Region:").grid(row=0, column=2, padx=(0, 4))
            self.region_var = create_dark_stringvar(self, value="800x600")
            self.region_entry = ttk.Entry(toolbar_frame, textvariable=self.region_var, width=10)
            self.region_entry.grid(row=0, column=3, padx=(0, 12))
            
            # Right controls
            self.auto_refresh_var = create_dark_booleanvar(self, value=True)
            ttk.Checkbutton(toolbar_frame, text="Auto-refresh", 
                           variable=self.auto_refresh_var).grid(row=0, column=4)
            
            # FPS controls
            ttk.Label(toolbar_frame, text="FPS:").grid(row=0, column=5, padx=(12, 4))
            self.fps_var = create_dark_stringvar(self, value="30")
            fps_combo = ttk.Combobox(toolbar_frame, textvariable=self.fps_var, 
                                   values=["15", "30", "60"], width=5, state="readonly")
            fps_combo.grid(row=0, column=6, padx=(0, 8))
            fps_combo.bind("<<ComboboxSelected>>", self._on_fps_changed)
        else:
            # No toolbar: window detection(0), status(1), canvas(2)
            window_detection_row = 0
            status_row = 1
            canvas_row = 2
            # No toolbar frame created at all
        
        # Window detection frame - always present
        window_detection_frame = ttk.Frame(self)
        window_detection_frame.grid(row=window_detection_row, column=0, sticky="ew", padx=8, pady=(0, 4))
        window_detection_frame.grid_columnconfigure(1, weight=1)
        
        # Detect window button
        ttk.Button(window_detection_frame, text="🔍 Detect Window", 
                  command=self._detect_runelite_window).grid(row=0, column=0, padx=(0, 8))
        
        # Window status display
        self.window_status_label = ttk.Label(window_detection_frame, text="Status: No window detected", 
                                           font=("Arial", 9))
        self.window_status_label.grid(row=0, column=1, sticky="w")
        
        # Status line - always present
        self.status_label = ttk.Label(self, text="Status: Ready | Region: 800x600 | FPS: 0", 
                                    font=("Arial", 9))
        self.status_label.grid(row=status_row, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Canvas for image display - always at canvas_row, expands to fill
        self.canvas = create_dark_canvas(self, bg="#202225", relief="sunken", bd=1, highlightthickness=0)
        self.canvas.grid(row=canvas_row, column=0, sticky="nsew", padx=8, pady=(0, 8))
        
        # Ensure canvas fills available space completely
        self.canvas.config(width=800, height=400)  # Set minimum size
        
        # Show initial placeholder
        self._show_placeholder()
        
        # Bind canvas events
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
    
    def _bind_events(self):
        """Bind UI events"""
        if self.show_toolbar:
            self.region_var.trace("w", self._on_region_change)
            self.auto_refresh_var.trace("w", self._on_auto_refresh_change)
    
    def _on_region_change(self, *args):
        """Handle region change"""
        try:
            # Parse region from string (e.g., "800x600")
            region_str = self.region_var.get()
            if 'x' in region_str:
                width, height = map(int, region_str.split('x'))
                self.region = (0, 0, width, height)
                self._update_status()
        except Exception:
            pass
    
    def _on_auto_refresh_change(self, *args):
        """Handle auto-refresh toggle change"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self._schedule_refresh()
    
    def _show_placeholder(self):
        """Show placeholder text when no window is detected"""
        if not self.canvas or not self.canvas.winfo_exists():
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not sized yet, use default
            width = 800
            height = 400
        
        # Draw placeholder text
        self.canvas.create_text(
            width // 2, height // 2,
            text="No window detected\nClick '🔍 Detect Window' to find Runelite windows",
            font=("Arial", 14),
            fill="#888888",
            justify="center"
        )
        
        # Draw a border around the canvas
        self.canvas.create_rectangle(
            2, 2, width-2, height-2,
            outline="#444444",
            width=2
        )
    
    def _detect_runelite_window(self):
        """Detect Runelite windows and update the region"""
        try:
            # Find Runelite windows
            runelite_windows = self.window_finder.find_runelite_windows()
            
            if not runelite_windows:
                self.window_status_label.config(text="Status: No Runelite windows found")
                self.detected_window = None
                logger.info("No Runelite windows detected")
                return
            
            # Get the active window or first available
            active_window = self.window_finder.get_active_runelite_window()
            
            if active_window:
                self.detected_window = active_window
                
                # Update region to match the detected window
                left = active_window['left']
                top = active_window['top']
                width = active_window['width']
                height = active_window['height']
                
                # Calculate aspect ratio
                aspect_ratio = width / height if height > 0 else 0
                
                # Update the region
                self.region = (left, top, left + width, top + height)
                
                # Update UI
                if hasattr(self, 'region_var'):
                    self.region_var.set(f"{width}x{height}")
                
                # Update status display
                status_text = f"Status: {active_window['title']} | Pos: ({left}, {top}) | Size: {width}x{height} | Aspect: {aspect_ratio:.2f}"
                self.window_status_label.config(text=status_text)
                
                # Auto-capture screenshot if auto-refresh is enabled
                if self.auto_refresh:
                    # Clear placeholder and start live capture
                    self.canvas.delete("all")
                    self._capture_screenshot()
                    self._schedule_refresh()
            else:
                self.window_status_label.config(text="Status: No active Runelite windows")
                self.detected_window = None
                # logger.warning("No active Runelite windows found")
                
        except Exception as e:
            error_msg = f"Failed to detect Runelite window: {e}"
            self.window_status_label.config(text=f"Status: Error - {error_msg}")
            logger.error(error_msg)
    
    def _update_window_status(self):
        """Update the window status display"""
        if self.detected_window:
            left = self.detected_window['left']
            top = self.detected_window['top']
            width = self.detected_window['width']
            height = self.detected_window['height']
            aspect_ratio = width / height if height > 0 else 0
            
            status_text = f"Status: {self.detected_window['title']} | Pos: ({left}, {top}) | Size: {width}x{height} | Aspect: {aspect_ratio:.2f}"
            self.window_status_label.config(text=status_text)
        else:
            self.window_status_label.config(text="Status: No window detected")
    
    def _capture_screenshot(self):
        """Capture a screenshot of the current region"""

        try:
            import pyautogui
            
            # Check if the window is ready
            if not self.canvas or not self.canvas.winfo_exists():
                error_msg = "Canvas not ready for screenshot capture"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Use detected window region if available, otherwise use default
            if self.detected_window:
                left = self.detected_window['left']
                top = self.detected_window['top']
                width = self.detected_window['width']
                height = self.detected_window['height']
                region = (left, top, left + width, top + height)
            else:
                # Use the stored region
                left, top, right, bottom = self.region
                width = right - left
                height = bottom - top
                region = (left, top, width, height)

            # logger.debug(f"Capturing screenshot for region: {region}")
            
            # Validate region dimensions
            if width <= 0 or height <= 0:
                error_msg = f"Invalid region dimensions: {width}x{height}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Take screenshot
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            # logger.debug(f"Screenshot captured successfully: {screenshot.size}")
            
            # Convert to numpy array
            self.current_image = np.array(screenshot)
            
            # Convert BGR to RGB for OpenCV
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            
            # Update display
            self._update_display()
            
            # Update status
            self._update_status()
            self._update_window_status()
            
            # logger.debug("Screenshot processing completed")
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise  # Re-raise the exception to stop execution
        

    
    def _update_display(self):
        """Update the canvas display with current image"""
        if self.current_image is None:
            return
        
        try:
            # Check if canvas is ready
            if not self.canvas or not self.canvas.winfo_exists():
                error_msg = "Canvas not ready for display update"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Check if the root window is ready for image creation
            try:
                # Test if we can create a PhotoImage by trying to access the root
                root = self.winfo_toplevel()
                if not root or not root.winfo_exists():
                    error_msg = "Root window not ready for image creation"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Root window not accessible: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Resize image to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized - show error and fail
                error_msg = f"Canvas dimensions too small: {canvas_width}x{canvas_height}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Calculate scaling
            img_height, img_width = self.current_image.shape[:2]
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y)
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            resized_image = cv2.resize(self.current_image, (new_width, new_height))
            
            # Convert to PIL Image
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Convert to PhotoImage with explicit master
            try:
                self.photo_image = ImageTk.PhotoImage(pil_image, master=root)
            except Exception as e:
                error_msg = f"Failed to create PhotoImage: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.config(width=new_width, height=new_height)
            
            # Center the image
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.photo_image)
            
        except Exception as e:
            logger.error(f"Failed to update display: {e}")
            raise  # Re-raise the exception to stop execution
    
    def _update_status(self):
        """Update the status label"""
        if self.current_image is None:
            status = "Status: Ready | Region: 800x600 | FPS: 0"
        else:
            height, width = self.current_image.shape[:2]
            fps = self._calculate_fps()
            status = f"Status: Active | Region: {width}x{height} | FPS: {fps:.1f}"
        
        if self.status_label:
            self.status_label.config(text=status)
        
        # Also update window status
        self._update_window_status()
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        if self.last_refresh > 0:
            fps = 1.0 / (current_time - self.last_refresh)
            self.last_refresh = current_time
            return min(fps, 60.0)  # Cap at 60 FPS
        else:
            self.last_refresh = current_time
            return 0.0
    
    def _optimize_refresh_rate(self):
        """Dynamically optimize refresh rate based on performance"""
        if not self.detected_window:
            return
        
        # Calculate current FPS
        current_fps = self._calculate_fps()
        
        # Adjust refresh interval based on performance
        if current_fps < 15:  # If we're getting less than 15 FPS
            self.refresh_interval = min(100, self.refresh_interval + 10)  # Slow down
            # logger.debug(f"Performance low ({current_fps:.1f} FPS), slowing down to {self.refresh_interval}ms")
        elif current_fps > 25 and self.refresh_interval > 33:  # If we're getting good FPS
            self.refresh_interval = max(33, self.refresh_interval - 5)  # Speed up
            # logger.debug(f"Performance good ({current_fps:.1f} FPS), speeding up to {self.refresh_interval}ms")
        
        # Skip frames if performance is still poor
        if current_fps < 10:
            self.skip_frames = 2  # Skip every 3rd frame
        elif current_fps < 20:
            self.skip_frames = 1  # Skip every 2nd frame
        else:
            self.skip_frames = 0  # No frame skipping
    
    def _schedule_refresh(self):
        """Schedule the next refresh if auto-refresh is enabled"""
        if self.auto_refresh:
            self.after(self.refresh_interval, self._refresh_display)
            
            # Also refresh window detection every 5 seconds
            self.after(5000, self._refresh_window_detection)
    
    def _refresh_window_detection(self):
        """Periodically refresh window detection"""
        if self.auto_refresh and self.detected_window:
            # Check if the window still exists and update if needed
            try:
                current_windows = self.window_finder.find_runelite_windows()
                window_still_exists = any(
                    w['title'] == self.detected_window['title'] 
                    for w in current_windows
                )
                
                if not window_still_exists:
                    # logger.info("Previously detected window no longer exists, re-detecting...")
                    self._detect_runelite_window()
                else:
                    # Update window status (position might have changed)
                    self._update_window_status()
            except Exception as e:
                logger.debug(f"Window detection refresh failed: {e}")
            
            # Schedule next refresh
            self.after(5000, self._refresh_window_detection)
    
    def _refresh_display(self):
        """Refresh the display"""
        # logger.info(f"_refresh_display called, auto_refresh: {self.auto_refresh}")
        
        if self.auto_refresh:
            # Check if the window is ready before attempting to capture
            try:
                # First check if the root window is ready
                try:
                    root = self.winfo_toplevel()
                    if not root or not root.winfo_exists():
                        error_msg = "Root window not ready for refresh"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                except Exception as e:
                    error_msg = f"Root window not accessible: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Test if we can access the canvas dimensions
                if self.canvas and self.canvas.winfo_exists():
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    # logger.debug(f"Canvas dimensions: {canvas_width}x{canvas_height}")
                    
                    # Only proceed if canvas has valid dimensions
                    if canvas_width > 1 and canvas_height > 1:
                        # logger.debug("Canvas ready, capturing screenshot...")
                        
                        # Only capture if we have a detected window
                        if self.detected_window:
                            # Implement frame skipping for performance
                            if self.frame_count % (self.skip_frames + 1) == 0:
                                self._capture_screenshot()
                                self._schedule_refresh()
                                
                                # Optimize refresh rate every 10 frames
                                if self.frame_count % 10 == 0:
                                    self._optimize_refresh_rate()
                            else:
                                # Skip this frame, but still schedule the next
                                self._schedule_refresh()
                            
                            self.frame_count += 1
                        else:
                            # Show placeholder text when no window is detected
                            self._show_placeholder()
                    else:
                        # Canvas not ready - show error and fail
                        error_msg = f"Canvas dimensions invalid: {canvas_width}x{canvas_height}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                else:
                    # Canvas doesn't exist - show error and fail
                    error_msg = "Canvas does not exist or is not ready"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Canvas refresh failed: {e}")
                raise
        else:
            # logger.debug("Auto-refresh is disabled, not refreshing")
            pass
        

    
    def _save_image(self):
        """Save the current image to a file"""
        if self.current_image is None:
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return
            
            # Save image
            cv2.imwrite(filename, self.current_image)
            
            # Update status
            self._update_status()
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
    
    def _on_canvas_click(self, event):
        """Handle canvas click"""
        # TODO: Implement region selection
        pass
    
    def _on_canvas_drag(self, event):
        """Handle canvas drag"""
        # TODO: Implement region selection
        pass
    
    def _on_canvas_release(self, event):
        """Handle canvas release"""
        # TODO: Implement region selection
        pass
    
    def _on_fps_changed(self, event=None):
        """Handle FPS selection change"""
        try:
            fps = int(self.fps_var.get())
            self.refresh_interval = int(1000 / fps)  # Convert FPS to milliseconds
            logger.info(f"FPS changed to {fps} ({self.refresh_interval}ms interval)")
            
            # Reset frame skipping
            self.skip_frames = 0
            self.frame_count = 0
            
        except ValueError:
            logger.error(f"Invalid FPS value: {self.fps_var.get()}")
    
    def set_fps(self, fps: int):
        """Set the target FPS manually"""
        self.fps_var.set(str(fps))
        self._on_fps_changed()
    
    def set_region(self, region: Tuple[int, int, int, int]):
        """Set the capture region"""
        self.region = region
        width = region[2] - region[0]
        height = region[3] - region[1]
        if hasattr(self, 'region_var'):
            self.region_var.set(f"{width}x{height}")
        self._update_status()
        
        # Start live streaming when region is set
        self._start_live_streaming()
    
    def _start_live_streaming(self):
        """Start continuous live streaming"""
        logger.info(f"Attempting to start live streaming for region: {self.region}")
        logger.info(f"Auto-refresh enabled: {self.auto_refresh}")
        
        # Ensure auto_refresh is enabled for live streaming
        self.auto_refresh = True
        
        if self.auto_refresh:
            self._schedule_refresh()
            logger.info(f"Successfully started live streaming for region: {self.region}")
        else:
            logger.warning("Failed to start live streaming - auto_refresh is disabled")
    
    def get_region(self) -> Tuple[int, int, int, int]:
        """Get the current capture region"""
        return self.region
    
    def test_method(self):
        """Test method to verify LiveView is working"""
        logger.info("LiveView test_method called successfully!")
        logger.info(f"Current region: {self.region}")
        logger.info(f"Auto-refresh: {self.auto_refresh}")
        logger.info(f"Canvas exists: {self.canvas is not None}")
        if self.canvas:
            logger.info(f"Canvas dimensions: {self.canvas.winfo_width()}x{self.canvas.winfo_height()}")
        return "LiveView is working!"
    
    def update_image(self, image: np.ndarray):
        """Update the view with a new image"""
        self.current_image = image.copy()
        self._update_display()
        self._update_status()
    
    def clear(self):
        """Clear the current image"""
        self.current_image = None
        self.photo_image = None
        self._show_placeholder()
        self._update_status()
