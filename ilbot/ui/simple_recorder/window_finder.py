"""Window finder for detecting Runelite windows."""
import win32gui
import win32con


class WindowFinder:
    def __init__(self):
        self.runelite_windows = []
        
    def find_runelite_window(self):
        """Find the first available Runelite window."""
        self.runelite_windows = []
        
        # Enumerate all windows
        win32gui.EnumWindows(self._enum_windows_callback, None)
        
        if self.runelite_windows:
            # Return the first (most recently used) Runelite window
            return self.runelite_windows[0]
        else:
            return None
            
    def _enum_windows_callback(self, hwnd, extra):
        """Callback for window enumeration."""
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            
            # Check if this is a Runelite window
            if self._is_runelite_window(window_text):
                # Get window position and size
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                width = right - x
                height = bottom - y
                
                # Get client area (excluding title bar, borders, etc.)
                client_rect = win32gui.GetClientRect(hwnd)
                client_width = client_rect[2]
                client_height = client_rect[3]
                
                window_info = {
                    'hwnd': hwnd,
                    'title': window_text,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'client_width': client_width,
                    'client_height': client_height
                }
                
                self.runelite_windows.append(window_info)
                
    def _is_runelite_window(self, window_text):
        """Check if a window title indicates it's a Runelite window."""
        if not window_text:
            return False
            
        # Common Runelite window title patterns
        runelite_indicators = [
            'RuneLite -'
        ]
        
        return any(indicator in window_text for indicator in runelite_indicators)
        
    def get_all_runelite_windows(self):
        """Get all available Runelite windows."""
        self.runelite_windows = []
        win32gui.EnumWindows(self._enum_windows_callback, None)
        return self.runelite_windows
