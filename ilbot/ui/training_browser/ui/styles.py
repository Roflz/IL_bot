"""
UI Styles

Centralized styling for colors, fonts, and column widths.
"""

from typing import Dict, Any


# Color scheme for feature groups
FEATURE_GROUP_COLORS = {
    "Player": "#e6f3ff",           # Light blue
    "Interaction": "#fff2e6",      # Light orange
    "Camera": "#f0f8ff",           # Light cyan
    "Inventory": "#f0fff0",        # Light green
    "Bank": "#fff0f5",             # Light pink
    "Phase Context": "#f8f8ff",    # Light lavender
    "Game Objects": "#fffacd",     # Light yellow
    "NPCs": "#ffe4e1",             # Light red
    "Tabs": "#f5f5dc",             # Light beige
    "Skills": "#e6e6fa",           # Light purple
    "Timestamp": "#f0f0f0",        # Light gray
    "other": "#ffffff"             # White (default)
}

# Feature type colors
FEATURE_TYPE_COLORS = {
    "time_ms": "#ffeb3b",          # Yellow
    "duration_ms": "#ff9800",      # Orange
    "world_coordinate": "#4caf50", # Green
    "screen_coordinate": "#2196f3", # Blue
    "boolean": "#9c27b0",          # Purple
    "integer": "#607d8b",          # Blue gray
    "float": "#795548",            # Brown
    "unknown": "#ffffff"           # White
}

# General UI colors
UI_COLORS = {
    "background": "#f5f5f5",
    "foreground": "#333333",
    "accent": "#2196f3",
    "success": "#4caf50",
    "warning": "#ff9800",
    "error": "#f44336",
    "border": "#cccccc",
    "highlight": "#e3f2fd"
}

# Font configurations
FONTS = {
    "default": ("Tahoma", 9),
    "heading": ("Tahoma", 10, "bold"),
    "monospace": ("Consolas", 9),
    "small": ("Tahoma", 8),
    "large": ("Tahoma", 11)
}

# Column widths for tables
COLUMN_WIDTHS = {
    "feature_index": 60,
    "feature_name": 200,
    "feature_group": 120,
    "feature_type": 100,
    "timestep_0": 80,
    "timestep_1": 80,
    "timestep_2": 80,
    "timestep_3": 80,
    "timestep_4": 80,
    "timestep_5": 80,
    "timestep_6": 80,
    "timestep_7": 80,
    "timestep_8": 80,
    "timestep_9": 80,
    "action_count": 80,
    "action_type": 120,
    "action_position": 100,
    "action_button": 80,
    "action_key": 80,
    "action_scroll": 100
}

# Table styling
TABLE_STYLES = {
    "alternate_row_color": "#f9f9f9",
    "selected_row_color": "#e3f2fd",
    "header_background": "#e0e0e0",
    "header_foreground": "#333333",
    "border_color": "#cccccc",
    "grid_color": "#e0e0e0"
}

# Button styling
BUTTON_STYLES = {
    "primary": {
        "background": "#2196f3",
        "foreground": "white",
        "activebackground": "#1976d2",
        "activeforeground": "white",
        "relief": "flat",
        "borderwidth": 1
    },
    "secondary": {
        "background": "#f5f5f5",
        "foreground": "#333333",
        "activebackground": "#e0e0e0",
        "activeforeground": "#333333",
        "relief": "flat",
        "borderwidth": 1
    },
    "success": {
        "background": "#4caf50",
        "foreground": "white",
        "activebackground": "#388e3c",
        "activeforeground": "white",
        "relief": "flat",
        "borderwidth": 1
    },
    "warning": {
        "background": "#ff9800",
        "foreground": "white",
        "activebackground": "#f57c00",
        "activeforeground": "white",
        "relief": "flat",
        "borderwidth": 1
    }
}

# Entry field styling
ENTRY_STYLES = {
    "default": {
        "relief": "solid",
        "borderwidth": 1,
        "background": "white",
        "foreground": "#333333"
    },
    "readonly": {
        "relief": "flat",
        "borderwidth": 1,
        "background": "#f5f5f5",
        "foreground": "#666666",
        "state": "readonly"
    }
}

# Label styling
LABEL_STYLES = {
    "heading": {
        "font": FONTS["heading"],
        "foreground": "#333333",
        "background": UI_COLORS["background"]
    },
    "subheading": {
        "font": FONTS["default"],
        "foreground": "#666666",
        "background": UI_COLORS["background"]
    },
    "info": {
        "font": FONTS["small"],
        "foreground": "#888888",
        "background": UI_COLORS["background"]
    }
}

# Frame styling
FRAME_STYLES = {
    "main": {
        "background": UI_COLORS["background"],
        "relief": "flat",
        "borderwidth": 0
    },
    "section": {
        "background": "white",
        "relief": "solid",
        "borderwidth": 1,
        "bd": 1
    },
    "subsection": {
        "background": "#fafafa",
        "relief": "flat",
        "borderwidth": 0
    }
}


def get_feature_group_color(group_name: str) -> str:
    """
    Get color for a feature group.
    
    Args:
        group_name: Name of the feature group
        
    Returns:
        Hex color string
    """
    return FEATURE_GROUP_COLORS.get(group_name, FEATURE_GROUP_COLORS["other"])


def get_feature_type_color(type_name: str) -> str:
    """
    Get color for a feature type.
    
    Args:
        type_name: Name of the feature type
        
    Returns:
        Hex color string
    """
    return FEATURE_TYPE_COLORS.get(type_name, FEATURE_TYPE_COLORS["unknown"])


def get_column_width(column_name: str) -> int:
    """
    Get width for a table column.
    
    Args:
        column_name: Name of the column
        
    Returns:
        Column width in pixels
    """
    return COLUMN_WIDTHS.get(column_name, 100)


def get_button_style(style_name: str) -> Dict[str, Any]:
    """
    Get button style configuration.
    
    Args:
        style_name: Name of the button style
        
    Returns:
        Dictionary of style properties
    """
    return BUTTON_STYLES.get(style_name, BUTTON_STYLES["secondary"])


def get_entry_style(style_name: str) -> Dict[str, Any]:
    """
    Get entry field style configuration.
    
    Args:
        style_name: Name of the entry style
        
    Returns:
        Dictionary of style properties
    """
    return ENTRY_STYLES.get(style_name, ENTRY_STYLES["default"])


def get_label_style(style_name: str) -> Dict[str, Any]:
    """
    Get label style configuration.
    
    Args:
        style_name: Name of the label style
        
    Returns:
        Dictionary of style properties
    """
    return LABEL_STYLES.get(style_name, LABEL_STYLES["info"])


def get_frame_style(style_name: str) -> Dict[str, Any]:
    """
    Get frame style configuration.
    
    Args:
        style_name: Name of the frame style
        
    Returns:
        Dictionary of style properties
    """
    return FRAME_STYLES.get(style_name, FRAME_STYLES["main"])
