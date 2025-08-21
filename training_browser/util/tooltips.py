"""
Tooltip Utilities

Tooltip lifecycle management and helpers.
"""

import tkinter as tk
from typing import Optional, Callable


class Tooltip:
    """Simple tooltip widget."""
    
    def __init__(self, widget, text: str, delay: int = 1000):
        """
        Initialize tooltip.
        
        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
            delay: Delay before showing tooltip (ms)
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.timer_id = None
        
        # Bind events
        self.widget.bind('<Enter>', self.on_enter)
        self.widget.bind('<Leave>', self.on_leave)
        self.widget.bind('<Button-1>', self.on_click)
    
    def on_enter(self, event):
        """Handle mouse enter event."""
        self.schedule_tooltip()
    
    def on_leave(self, event):
        """Handle mouse leave event."""
        self.unschedule_tooltip()
        self.hide_tooltip()
    
    def on_click(self, event):
        """Handle click event."""
        self.unschedule_tooltip()
        self.hide_tooltip()
    
    def schedule_tooltip(self):
        """Schedule tooltip to appear after delay."""
        self.unschedule_tooltip()
        self.timer_id = self.widget.after(self.delay, self.show_tooltip)
    
    def unschedule_tooltip(self):
        """Unschedule tooltip."""
        if self.timer_id:
            self.widget.after_cancel(self.timer_id)
            self.timer_id = None
    
    def show_tooltip(self):
        """Show the tooltip window."""
        if self.tooltip_window:
            return
        
        # Get widget position - handle different widget types
        try:
            if hasattr(self.widget, 'bbox') and callable(self.widget.bbox):
                # For treeview and other widgets with bbox
                bbox = self.widget.bbox("insert")
                if bbox:
                    x, y, _, _ = bbox
                else:
                    # Fallback to cursor position
                    x, y = self.widget.winfo_pointerx(), self.widget.winfo_pointery()
            else:
                # Fallback to cursor position
                x, y = self.widget.winfo_pointerx(), self.widget.winfo_pointery()
        except:
            # Final fallback
            x, y = self.widget.winfo_pointerx(), self.widget.winfo_pointery()
        
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Create label
        label = tk.Label(self.tooltip_window, text=self.text, 
                        justify=tk.LEFT, background="#ffffe0", 
                        relief=tk.SOLID, borderwidth=1,
                        font=("Tahoma", "8", "normal"))
        label.pack()
        
        # Make sure tooltip is on top
        self.tooltip_window.lift()
        self.tooltip_window.attributes('-topmost', True)
    
    def hide_tooltip(self):
        """Hide the tooltip window."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def destroy(self):
        """Clean up tooltip."""
        self.unschedule_tooltip()
        self.hide_tooltip()


class TooltipManager:
    """Manager for multiple tooltips."""
    
    def __init__(self):
        """Initialize tooltip manager."""
        self.tooltips = []
    
    def add_tooltip(self, widget, text: str, delay: int = 1000) -> Tooltip:
        """
        Add a tooltip to a widget.
        
        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
            delay: Delay before showing tooltip (ms)
            
        Returns:
            Created Tooltip instance
        """
        tooltip = Tooltip(widget, text, delay)
        self.tooltips.append(tooltip)
        return tooltip
    
    def clear_all(self):
        """Clear all tooltips."""
        for tooltip in self.tooltips:
            tooltip.destroy()
        self.tooltips.clear()
    
    def destroy(self):
        """Destroy tooltip manager."""
        self.clear_all()


def create_feature_tooltip_text(feature_idx: int, feature_name: str, 
                               feature_group: str, feature_type: str) -> str:
    """
    Create tooltip text for a feature.
    
    Args:
        feature_idx: Feature index
        feature_name: Feature name
        feature_group: Feature group
        feature_type: Feature data type
        
    Returns:
        Formatted tooltip text
    """
    lines = [
        f"Feature {feature_idx}: {feature_name}",
        f"Group: {feature_group}",
        f"Type: {feature_type}"
    ]
    
    # Add specific information based on feature type
    if feature_type == 'time_ms':
        lines.append("Time in milliseconds")
    elif feature_type == 'duration_ms':
        lines.append("Duration in milliseconds")
    elif feature_type == 'world_coordinate':
        lines.append("World coordinate value")
    elif feature_type == 'screen_coordinate':
        lines.append("Screen coordinate value")
    elif feature_type == 'boolean':
        lines.append("Boolean value (0/1)")
    elif feature_type == 'integer':
        lines.append("Integer value")
    elif feature_type == 'float':
        lines.append("Float value")
    
    return "\n".join(lines)


def create_action_tooltip_text(action_type: str, x: int, y: int, 
                              button: str, key: str) -> str:
    """
    Create tooltip text for an action.
    
    Args:
        action_type: Action type
        x: X coordinate
        y: Y coordinate
        button: Button type
        key: Key pressed
        
    Returns:
        Formatted tooltip text
    """
    lines = [f"Action: {action_type}"]
    
    if x != 0 or y != 0:
        lines.append(f"Position: ({x}, {y})")
    
    if button and button != "Button_0":
        lines.append(f"Button: {button}")
    
    if key and key != "Key_0":
        lines.append(f"Key: {key}")
    
    return "\n".join(lines)
