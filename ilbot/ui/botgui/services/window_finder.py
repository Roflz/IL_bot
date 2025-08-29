#!/usr/bin/env python3
"""Window finder service for detecting Runelite windows specifically"""

import logging
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import pygetwindow
try:
    import pygetwindow as gw
    PYWINDOW_AVAILABLE = True
except ImportError:
    gw = None
    PYWINDOW_AVAILABLE = False


class WindowFinder:
    """Service for finding and managing Runelite windows specifically"""
    
    def __init__(self):
        self.pywindow_available = PYWINDOW_AVAILABLE
        if not self.pywindow_available:
            logger.warning("pygetwindow not available, Runelite window detection disabled")
    
    def find_runelite_windows(self) -> List[dict]:
        """Find windows that start with 'RuneLite' (capital R and L) - no fallbacks"""
        if not self.pywindow_available:
            logger.warning("pygetwindow not available, cannot detect Runelite windows")
            return []
        
        try:
            # Get all windows
            all_windows = gw.getAllWindows()
            # logger.info(f"Total windows found: {len(all_windows)}")
            
            # Log all window titles for debugging
            # for i, window in enumerate(all_windows):
            #     logger.info(f"Window {i}: '{window.title}' (active: {window.isActive}, minimized: {window.isMinimized})")
            
            runelite_windows = []
            search_term = "RuneLite"  # Capital R and L
            
            for window in all_windows:
                title = window.title
                
                # Only accept windows that start with "RuneLite" (capital R and L)
                if title.startswith(search_term):
                    runelite_windows.append({
                        'title': window.title,
                        'left': window.left,
                        'top': window.top,
                        'width': window.width,
                        'height': window.height,
                        'is_active': window.isActive,
                        'is_minimized': window.isMinimized
                    })
                    # logger.info(f"Found Runelite window: {title}")
                else:
                    # Log filtered windows for debugging
                    # logger.debug(f"Filtered out non-Runelite window: {title}")
                    pass
            
            print(f"🔍 RUNELITE WINDOW DETECTION: Found {len(runelite_windows)} Runelite window(s)")
            if runelite_windows:
                for i, window in enumerate(runelite_windows):
                    print(f"   Window {i+1}: '{window['title']}' at ({window['left']}, {window['top']}) {window['width']}x{window['height']}")
            else:
                print("   No Runelite windows found - make sure RuneLite is running")
            
            return runelite_windows
            
        except Exception as e:
            logger.error(f"Failed to find Runelite windows: {e}")
            return []
    
    def get_active_runelite_window(self) -> Optional[dict]:
        """Get the currently active Runelite window"""
        windows = self.find_runelite_windows()
        
        if not windows:
            logger.warning("No Runelite windows found")
            return None
        
        # Return the first active window
        for window in windows:
            if window['is_active'] and not window['is_minimized']:
                logger.info(f"Found active Runelite window: {window['title']}")
                return window
        
        # Return the first non-minimized window
        for window in windows:
            if not window['is_minimized']:
                logger.info(f"Found non-minimized Runelite window: {window['title']}")
                return window
        
        # Return the first available window
        logger.info(f"Using first available Runelite window: {windows[0]['title']}")
        return windows[0]
    
    def get_window_region(self, window: dict) -> Tuple[int, int, int, int]:
        """Get the region coordinates for a window (left, top, right, bottom)"""
        return (
            window['left'],
            window['top'],
            window['left'] + window['width'],
            window['top'] + window['height']
        )
    
    def validate_region(self, region: Tuple[int, int, int, int]) -> bool:
        """Validate that a region is reasonable"""
        left, top, right, bottom = region
        
        # Check for reasonable dimensions
        if right <= left or bottom <= top:
            return False
        
        # Check for reasonable size (not too small, not too large)
        width = right - left
        height = bottom - top
        
        if width < 100 or height < 100:
            return False
        
        if width > 4000 or height > 3000:
            return False
        
        return True
    
    def get_default_region(self) -> Tuple[int, int, int, int]:
        """Get a default region for testing when no Runelite window is found"""
        logger.warning("No Runelite window found, using default region")
        return (0, 0, 800, 600)
    
    def is_runelite_window_available(self) -> bool:
        """Check if any Runelite windows are available"""
        return len(self.find_runelite_windows()) > 0
    
    # Legacy method names for backward compatibility
    def find_game_windows(self, game_title: str = "Runelite") -> List[dict]:
        """Legacy method - now redirects to find_runelite_windows"""
        logger.warning("find_game_windows is deprecated, use find_runelite_windows instead")
        return self.find_runelite_windows()
    
    def get_active_game_window(self, game_title: str = "Runelite") -> Optional[dict]:
        """Legacy method - now redirects to get_active_runelite_window"""
        logger.warning("get_active_game_window is deprecated, use get_active_runelite_window instead")
        return self.get_active_runelite_window()
    
    def is_window_available(self) -> bool:
        """Legacy method - now redirects to is_runelite_window_available"""
        logger.warning("is_window_available is deprecated, use is_runelite_window_available instead")
        return self.is_runelite_window_available()
