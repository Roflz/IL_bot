#!/usr/bin/env python3
"""Treeview widget with integrated scrollbars using grid layout"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Tuple


class TreeWithScrollbars:
    """Treeview with integrated scrollbars using grid layout"""
    
    def __init__(self, parent, columns: List[Tuple[str, str, int]], height: int = 20, **kwargs):
        """
        Initialize the tree with scrollbars.
        
        Args:
            parent: Parent widget
            columns: List of (column_id, heading_text, width) tuples
            height: Tree height in rows
            **kwargs: Additional arguments for Treeview
        """
        self.parent = parent
        self.columns = columns
        
        # Create the main frame
        self.frame = ttk.Frame(parent)
        
        # Create the treeview
        self.tree = ttk.Treeview(self.frame, columns=[col[0] for col in columns], 
                                show="headings", height=height, **kwargs)
        
        # Configure columns
        for col_id, heading, width in columns:
            self.tree.heading(col_id, text=heading)
            self.tree.column(col_id, width=width, anchor="center")
        
        # Create scrollbars
        self.vsb = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
        
        # Configure tree scrolling
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        
        # Grid layout - all widgets in the same parent frame
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        
        # Tree takes most space
        self.tree.grid(row=0, column=0, sticky="nsew")
        
        # Vertical scrollbar on the right
        self.vsb.grid(row=0, column=1, sticky="ns")
        
        # Horizontal scrollbar on the bottom
        self.hsb.grid(row=1, column=0, sticky="ew")
        
        # Bind events for better UX
        self._bind_events()
    
    def _bind_events(self):
        """Bind events for better user experience"""
        # Auto-scroll to selection
        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        
        # Mouse wheel scrolling
        self.tree.bind('<MouseWheel>', self._on_mousewheel)
        self.tree.bind('<Shift-MouseWheel>', self._on_shift_mousewheel)
    
    def _on_select(self, event):
        """Handle selection events"""
        # Ensure selected item is visible
        selection = self.tree.selection()
        if selection:
            self.tree.see(selection[0])
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_shift_mousewheel(self, event):
        """Handle shift + mouse wheel for horizontal scrolling"""
        self.tree.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def pack(self, **kwargs):
        """Pack the frame"""
        return self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame"""
        return self.frame.grid(**kwargs)
    
    def place(self, **kwargs):
        """Place the frame"""
        return self.frame.place(**kwargs)
    
    def clear(self):
        """Clear all items from the tree"""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def insert(self, parent="", index="end", **kwargs):
        """Insert an item into the tree"""
        return self.tree.insert(parent, index, **kwargs)
    
    def delete(self, item):
        """Delete an item from the tree"""
        self.tree.delete(item)
    
    def selection(self):
        """Get selected items"""
        return self.tree.selection()
    
    def see(self, item):
        """Ensure an item is visible"""
        self.tree.see(item)
    
    def get_children(self, item=""):
        """Get children of an item"""
        return self.tree.get_children(item)
    
    def item(self, item, option=None, **kwargs):
        """Get or set item options"""
        if option is None and not kwargs:
            return self.tree.item(item)
        return self.tree.item(item, option, **kwargs)
    
    def set(self, item, column, value):
        """Set a column value for an item"""
        self.tree.set(item, column, value)
    
    def get(self, item, column):
        """Get a column value for an item"""
        return self.tree.get(item, column)
    
    def heading(self, column, option=None, **kwargs):
        """Get or set column heading options"""
        if option is None and not kwargs:
            return self.tree.heading(column)
        return self.tree.heading(column, option, **kwargs)
    
    def column(self, column, option=None, **kwargs):
        """Get or set column options"""
        if option is None and not kwargs:
            return self.tree.column(column)
        return self.tree.column(column, option, **kwargs)
    
    def bind(self, sequence, func, add=None):
        """Bind an event to the tree"""
        return self.tree.bind(sequence, func, add)
    
    def configure(self, **kwargs):
        """Configure the tree"""
        return self.tree.configure(**kwargs)
    
    def config(self, **kwargs):
        """Configure the tree (alias for configure)"""
        return self.tree.config(**kwargs)
    
    def focus(self, item=None):
        """Set or get focus"""
        if item is None:
            return self.tree.focus()
        return self.tree.focus(item)
    
    def identify(self, component, x, y):
        """Identify component at position"""
        return self.tree.identify(component, x, y)
    
    def index(self, item):
        """Get index of item"""
        return self.tree.index(item)
    
    def next(self, item):
        """Get next item"""
        return self.tree.next(item)
    
    def parent(self, item):
        """Get parent of item"""
        return self.tree.parent(item)
    
    def prev(self, item):
        """Get previous item"""
        return self.tree.prev(item)
    
    def set_theme(self, theme):
        """Set the ttk theme for the tree"""
        style = ttk.Style()
        style.theme_use(theme)
    
    def set_alternating_colors(self, color1=None, color2=None):
        """Set alternating row colors with dark theme defaults"""
        # Use dark theme appropriate colors if none provided
        if color1 is None:
            color1 = "#2b2d31"  # Dark surface color
        if color2 is None:
            color2 = "#202225"  # Slightly darker for contrast
        
        self.tree.tag_configure("oddrow", background=color1)
        self.tree.tag_configure("evenrow", background=color2)
    
    def apply_alternating_colors(self):
        """Apply alternating colors to existing rows"""
        children = self.tree.get_children()
        for i, child in enumerate(children):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.tree.item(child, tags=(tag,))
