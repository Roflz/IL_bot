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

logger = logging.getLogger(__name__)


class LiveView(ttk.Frame):
    """View for displaying live screenshots and region preview"""
    
    def __init__(self, parent, controller, show_toolbar: bool = True):
        super().__init__(parent)
        self.controller = controller
        self.show_toolbar = show_toolbar
        
        # Image state
        self.current_image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.region: Tuple[int, int, int, int] = (0, 0, 800, 600)
        
        # Display state
        self.auto_refresh = True
        self.refresh_interval = 100  # ms
        self.last_refresh = 0
        
        # UI elements
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[ttk.Label] = None
        
        self._setup_ui()
        self._bind_events()
        
        # Debug logging
        logger.info(f"LiveView initialized with show_toolbar={self.show_toolbar}")
        logger.info(f"Initial region: {self.region}")
        logger.info(f"Auto-refresh: {self.auto_refresh}")
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Configure grid weights: status (0), canvas (1)
        self.grid_rowconfigure(0, weight=0)  # status line - no expansion
        self.grid_rowconfigure(1, weight=1)  # canvas - expands to fill
        self.grid_columnconfigure(0, weight=1)  # full width
        
        # Determine row positions based on toolbar setting
        if self.show_toolbar:
            # With toolbar: toolbar(0), status(1), canvas(2)
            toolbar_row = 0
            status_row = 1
            canvas_row = 2
            self.grid_rowconfigure(2, weight=1)  # canvas expands
            
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
        else:
            # No toolbar: status(0), canvas(1)
            status_row = 0
            canvas_row = 1
            # No toolbar frame created at all
        
        # Status line - always present
        self.status_label = ttk.Label(self, text="Status: Ready | Region: 800x600 | FPS: 0", 
                                    font=("Arial", 9))
        self.status_label.grid(row=status_row, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Canvas for image display - always at canvas_row, expands to fill
        self.canvas = create_dark_canvas(self, bg="#202225", relief="sunken", bd=1, highlightthickness=0)
        self.canvas.grid(row=canvas_row, column=0, sticky="nsew", padx=8, pady=(0, 8))
        
        # Ensure canvas fills available space completely
        self.canvas.config(width=800, height=400)  # Set minimum size
        
        # Bind canvas events
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
        
        # Initial display
        self._show_placeholder()
    
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
        """Show placeholder when no image is available"""
        if self.canvas:
            # Create a simple placeholder
            width, height = 400, 300
            self.canvas.config(width=width, height=height)
            
            # Draw placeholder text
            self.canvas.create_text(
                width // 2, height // 2,
                text="No image available\nClick 'Capture' to take a screenshot",
                font=("Arial", 12),
                fill="gray",
                justify="center"
            )
    
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
            
            # Get region coordinates
            left, top, right, bottom = self.region
            width = right - left
            height = bottom - top
            

            logger.debug(f"Capturing screenshot for region: {left}, {top}, {width}x{height}")
            
            # Validate region dimensions
            if width <= 0 or height <= 0:
                error_msg = f"Invalid region dimensions: {width}x{height}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Take screenshot
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            logger.debug(f"Screenshot captured successfully: {screenshot.size}")
            
            # Convert to numpy array
            self.current_image = np.array(screenshot)
            
            # Convert BGR to RGB for OpenCV
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            
            # Update display
            self._update_display()
            
            # Update status
            self._update_status()
            
            logger.debug("Screenshot processing completed")
            
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
    
    def _schedule_refresh(self):
        """Schedule the next refresh if auto-refresh is enabled"""
        if self.auto_refresh:
            self.after(self.refresh_interval, self._refresh_display)
    
    def _refresh_display(self):
        """Refresh the display"""
        logger.info(f"_refresh_display called, auto_refresh: {self.auto_refresh}")
        
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
                    logger.debug(f"Canvas dimensions: {canvas_width}x{canvas_height}")
                    
                    # Only proceed if canvas has valid dimensions
                    if canvas_width > 1 and canvas_height > 1:
                        logger.debug("Canvas ready, capturing screenshot...")
                        self._capture_screenshot()
                        self._schedule_refresh()
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
            logger.debug("Auto-refresh is disabled, not refreshing")
        

    
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
