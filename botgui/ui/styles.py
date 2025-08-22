#!/usr/bin/env python3
"""Dark theme implementation with strict error handling - no silent fallbacks"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple


# Dark theme color palette
PALETTE = {
    "bg": "#1e1f22",           # Main background
    "surface": "#2b2d31",      # Surface/panel background
    "sunken": "#202225",       # Sunken/input background
    "border": "#3b3f45",       # Border color
    "text": "#e6e6e6",         # Primary text
    "muted": "#a9b2bd",        # Muted/secondary text
    "accent": "#4e8ef7",       # Primary accent
    "accent_hover": "#6ca0ff", # Accent hover state
    "accent_active": "#4078d0", # Accent active state
    "sel_bg": "#314b7d",       # Selection background
    "sel_fg": "#ffffff",       # Selection foreground
    "menubar": "#3a3d42"       # Menu bar background (distinct from surface)
}


def _map(style: ttk.Style, widget: str, option: str, pairs: List[Tuple[str, str]]) -> None:
    """Apply style mapping with error checking"""
    try:
        style.map(widget, **{option: pairs})
    except Exception as e:
        raise RuntimeError(f"Failed to apply style mapping for {widget}.{option}: {e}") from e


def apply_dark_theme(root: tk.Tk, *, accent: str = PALETTE["accent"]) -> None:
    """
    Apply dark theme to the root window with strict error handling.
    
    Args:
        root: The Tk root window
        accent: Custom accent color (optional)
    
    Raises:
        RuntimeError: If theme cannot be applied, with clear error message
    """
    # Validate root parameter
    if not isinstance(root, tk.Misc) or not str(root).startswith("."):
        raise RuntimeError("apply_dark_theme must be called with a real Tk root.")
    
    # Get style object
    try:
        style = ttk.Style(root)
    except Exception as e:
        raise RuntimeError(f"Failed to get ttk.Style from root: {e}") from e
    
    # Validate base theme availability
    try:
        names = style.theme_names()
    except Exception as e:
        raise RuntimeError(f"Failed to get available themes: {e}") from e
    
    if "clam" not in names:
        available_themes = ", ".join(names) if names else "none"
        raise RuntimeError(
            f"Dark theme requires ttk base theme 'clam', which is not available. "
            f"Available themes: {available_themes}"
        )
    
    # Apply base theme
    try:
        style.theme_use("clam")
    except Exception as e:
        raise RuntimeError(f"Failed to switch to 'clam' theme: {e}") from e
    
    # Configure root background
    try:
        root.configure(bg=PALETTE["bg"])
    except Exception as e:
        raise RuntimeError(f"Failed to set root background: {e}") from e
    
    # Apply menu options (raise on any failure)
    try:
        root.option_add("*TearOff", False)
        # Menu bar (the File, Control, View, Help bar)
        root.option_add("*Menu.background", PALETTE["menubar"])
        root.option_add("*Menu.foreground", PALETTE["text"])
        root.option_add("*Menu.activeBackground", PALETTE["sel_bg"])
        root.option_add("*Menu.activeForeground", PALETTE["sel_fg"])
        root.option_add("*Menu.relief", "flat")
        root.option_add("*Menu.borderWidth", 1)
        root.option_add("*Menu.activeBorderWidth", 1)
        # Specific menubar styling
        root.option_add("*Menubar.background", PALETTE["menubar"])
        root.option_add("*Menubar.foreground", PALETTE["text"])
        root.option_add("*Menubar.relief", "raised")
        root.option_add("*Menubar.borderWidth", 1)
        # Alternative patterns for Windows
        root.configure(bg=PALETTE["menubar"])
        
        # Setup larger fonts globally
        setup_styles()
        
    except tk.TclError as e:
        raise RuntimeError(f"Failed to apply dark menu options: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error applying menu options: {e}") from e
    
    # Configure base styles
    try:
        style.configure(".",
            background=PALETTE["surface"],
            foreground=PALETTE["text"],
            fieldbackground=PALETTE["sunken"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            troughcolor=PALETTE["sunken"],
            focuscolor=accent,
            highlightthickness=0
        )
    except Exception as e:
        raise RuntimeError(f"Failed to configure base style: {e}") from e
    
    # Configure frame styles
    try:
        style.configure("TFrame", background=PALETTE["surface"])
        style.configure("TLabelframe", background=PALETTE["surface"])
        style.configure("TLabelframe.Label", 
                      background=PALETTE["surface"], 
                      foreground=PALETTE["muted"])
    except Exception as e:
        raise RuntimeError(f"Failed to configure frame styles: {e}") from e
    
    # Configure button styles
    try:
        style.configure("TButton", 
                       padding=6, 
                       background=PALETTE["surface"], 
                       foreground=PALETTE["text"])
        _map(style, "TButton", "background", [
            ("active", PALETTE["accent_hover"]), 
            ("pressed", PALETTE["accent_active"])
        ])
        _map(style, "TButton", "foreground", [("disabled", "#737b86")])
    except Exception as e:
        raise RuntimeError(f"Failed to configure button styles: {e}") from e
    
    # Configure checkbox and radio button styles
    try:
        style.configure("TCheckbutton", 
                       background=PALETTE["surface"], 
                       foreground=PALETTE["text"])
        style.configure("TRadiobutton", 
                       background=PALETTE["surface"], 
                       foreground=PALETTE["text"])
    except Exception as e:
        raise RuntimeError(f"Failed to configure checkbox/radio styles: {e}") from e
    
    # Configure input styles
    try:
        style.configure("TEntry", 
                       fieldbackground=PALETTE["sunken"], 
                       foreground=PALETTE["text"], 
                       bordercolor=PALETTE["border"])
        style.configure("TCombobox",
                       fieldbackground=PALETTE["sunken"], 
                       foreground=PALETTE["text"],
                       selectforeground=PALETTE["text"], 
                       selectbackground=PALETTE["sel_bg"])
    except Exception as e:
        raise RuntimeError(f"Failed to configure input styles: {e}") from e
    
    # Configure notebook styles
    try:
        style.configure("TNotebook", 
                       background=PALETTE["surface"], 
                       tabmargins=[6, 3, 6, 0])
        style.configure("TNotebook.Tab", 
                       background=PALETTE["bg"], 
                       foreground=PALETTE["muted"], 
                       padding=[10, 4])
        _map(style, "TNotebook.Tab", "background", [("selected", PALETTE["surface"])])
        _map(style, "TNotebook.Tab", "foreground", [("selected", PALETTE["text"])])
    except Exception as e:
        raise RuntimeError(f"Failed to configure notebook styles: {e}") from e
    
    # Configure treeview styles
    try:
        style.configure("Treeview",
                       background=PALETTE["sunken"], 
                       fieldbackground=PALETTE["sunken"],
                       foreground=PALETTE["text"], 
                       rowheight=24, 
                       bordercolor=PALETTE["border"])
        style.configure("Treeview.Heading", 
                       background=PALETTE["surface"], 
                       foreground=PALETTE["text"], 
                       bordercolor=PALETTE["border"])
        _map(style, "Treeview", "background", [("selected", PALETTE["sel_bg"])])
        _map(style, "Treeview", "foreground", [("selected", PALETTE["sel_fg"])])
    except Exception as e:
        raise RuntimeError(f"Failed to configure treeview styles: {e}") from e
    
    # Configure scrollbar and paned window styles
    try:
        style.configure("Vertical.TScrollbar", 
                       background=PALETTE["surface"], 
                       troughcolor=PALETTE["sunken"])
        style.configure("Horizontal.TScrollbar", 
                       background=PALETTE["surface"], 
                       troughcolor=PALETTE["sunken"])
        style.configure("TPanedwindow", background=PALETTE["surface"])
    except Exception as e:
        raise RuntimeError(f"Failed to configure scrollbar/paned window styles: {e}") from e
    
    # Configure toolbar style - distinct from other frames
    try:
        style.configure("Toolbar.TFrame", 
                       background="#3a3d42",  # Slightly lighter than surface, darker than border
                       relief="raised",
                       borderwidth=1)
    except Exception as e:
        raise RuntimeError(f"Failed to configure toolbar style: {e}") from e
    
    # Configure menubar style - distinct from other elements
    try:
        style.configure("Menubar.TFrame", 
                       background=PALETTE["menubar"],
                       relief="flat",
                       borderwidth=0)
    except Exception as e:
        raise RuntimeError(f"Failed to configure menubar style: {e}") from e
    
    # Sanity check: ensure base color was actually applied
    try:
        applied_bg = style.lookup("TFrame", "background")
        if applied_bg not in (PALETTE["surface"],):
            raise RuntimeError(
                f"Dark theme sanity check failed: TFrame background is '{applied_bg}' "
                f"instead of expected '{PALETTE['surface']}'"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to verify theme application: {e}") from e
    
    # Notify views that rely on this event; fail loudly if Tk refuses
    try:
        root.event_generate("<<DarkThemeApplied>>", when="tail")
    except tk.TclError as e:
        raise RuntimeError(f"Failed to emit <<DarkThemeApplied>> event: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error emitting <<DarkThemeApplied>> event: {e}") from e


# Helper factory functions for widgets that require explicit dark colors
def create_dark_text(parent, **kw) -> tk.Text:
    """
    Create a Text widget with required dark theme colors.
    
    Args:
        parent: Parent widget
        **kw: Additional Text widget options
    
    Returns:
        Configured Text widget
        
    Raises:
        RuntimeError: If dark colors are not provided
    """
    if "bg" not in kw or "fg" not in kw:
        raise RuntimeError(
            "Text widget must be created with explicit 'bg' and 'fg' colors "
            "after dark theme is active. Use create_dark_text(parent, bg='#202225', fg='#e6e6e6', ...)"
        )
    
    try:
        return tk.Text(parent, **kw)
    except Exception as e:
        raise RuntimeError(f"Failed to create dark Text widget: {e}") from e


def create_dark_canvas(parent, **kw) -> tk.Canvas:
    """
    Create a Canvas widget with required dark theme colors.
    
    Args:
        parent: Parent widget
        **kw: Additional Canvas widget options
    
    Returns:
        Configured Canvas widget
        
    Raises:
        RuntimeError: If dark colors are not provided
    """
    if "bg" not in kw:
        raise RuntimeError(
            "Canvas widget must be created with explicit 'bg' color "
            "after dark theme is active. Use create_dark_canvas(parent, bg='#202225', ...)"
        )
    
    try:
        return tk.Canvas(parent, **kw)
    except Exception as e:
        raise RuntimeError(f"Failed to create dark Canvas widget: {e}") from e


def create_dark_stringvar(parent, **kw) -> tk.StringVar:
    """
    Create a StringVar with explicit master to prevent implicit root creation.
    
    Args:
        parent: Parent widget (required for master parameter)
        **kw: Additional StringVar options
    
    Returns:
        Configured StringVar
        
    Raises:
        RuntimeError: If parent is not provided
    """
    if not parent:
        raise RuntimeError(
            "StringVar must be created with explicit parent/master to prevent "
            "implicit root creation. Use create_dark_stringvar(parent, ...)"
        )
    
    try:
        return tk.StringVar(master=parent, **kw)
    except Exception as e:
        raise RuntimeError(f"Failed to create dark StringVar: {e}") from e


def create_dark_booleanvar(parent, **kw) -> tk.BooleanVar:
    """
    Create a BooleanVar with explicit master to prevent implicit root creation.
    
    Args:
        parent: Parent widget (required for master parameter)
        **kw: Additional BooleanVar options
    
    Returns:
        Configured BooleanVar
        
    Raises:
        RuntimeError: If parent is not provided
    """
    if not parent:
        raise RuntimeError(
            "BooleanVar must be created with explicit parent/master to prevent "
            "implicit root creation. Use create_dark_booleanvar(parent, ...)"
        )
    
    try:
        return tk.BooleanVar(master=parent, **kw)
    except Exception as e:
        raise RuntimeError(f"Failed to create dark BooleanVar: {e}") from e


def create_dark_intvar(parent, **kw) -> tk.IntVar:
    """
    Create an IntVar with explicit master to prevent implicit root creation.
    
    Args:
        parent: Parent widget (required for master parameter)
        **kw: Additional IntVar options
    
    Returns:
        Configured IntVar
        
    Raises:
        RuntimeError: If parent is not provided
    """
    if not parent:
        raise RuntimeError(
            "IntVar must be created with explicit parent/master to prevent "
            "implicit root creation. Use create_dark_intvar(parent, ...)"
        )
    
    try:
        return tk.IntVar(master=parent, **kw)
    except Exception as e:
        raise RuntimeError(f"Failed to create dark IntVar: {e}") from e


def setup_styles():
    """Setup custom ttk styles"""
    style = ttk.Style()
    
    # Increase default font sizes globally
    default_font = ("Segoe UI", 13)  # Increased from default
    large_font = ("Segoe UI", 15)    # Increased from default
    header_font = ("Segoe UI", 17, "bold")  # Increased from default
    
    # Configure default fonts for all widgets
    style.configure(".", font=default_font)
    
    # Specific widget styles with larger fonts
    style.configure("TLabel", font=default_font)
    style.configure("TButton", font=default_font)
    style.configure("TEntry", font=default_font)
    style.configure("TCombobox", font=default_font)
    style.configure("Treeview", font=default_font)
    style.configure("Treeview.Heading", font=default_font)
    
    # Header styles
    style.configure("Header.TLabel", font=header_font)
    style.configure("Title.TLabel", font=large_font)
    
    # Button styles
    style.configure("Accent.TButton", font=default_font)
    style.configure("Primary.TButton", font=default_font)
    
    # Frame styles
    style.configure("Toolbar.TFrame", background="#3a3d42", relief="raised", borderwidth=1)
    
    # Status and info styles
    style.configure("Status.TLabel", font=default_font)
    style.configure("Info.TLabel", font=default_font)
