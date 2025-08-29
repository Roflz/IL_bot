"""
Feature Table View

Treeview table for displaying 10×128 feature data with coloring and tooltips.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Optional, Dict, Any

from ...util.formatting import format_value_for_display
from ...util.tooltips import TooltipManager, create_feature_tooltip_text
from ...ui.styles import get_feature_group_color, get_column_width


class FeatureTableView(ttk.Frame):
    """Feature table view for displaying gamestate features."""
    
    def __init__(self, parent, controller):
        """
        Initialize the feature table view.
        
        Args:
            parent: Parent widget
            controller: Application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.tooltip_manager = TooltipManager()
        
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
    
    def _create_widgets(self):
        """Create the feature table widgets."""
        # Control frame
        self.control_frame = ttk.Frame(self)
        
        # Normalized data toggle
        self.normalized_var = tk.BooleanVar(value=False)
        self.normalized_check = ttk.Checkbutton(
            self.control_frame,
            text="Show Normalized Data",
            variable=self.normalized_var,
            command=self._on_normalized_toggle
        )
        
        # Feature group filter
        ttk.Label(self.control_frame, text="Filter by group:").pack(side="left", padx=(20, 5))
        self.filter_var = tk.StringVar(value="All")
        self.filter_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.filter_var,
            values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", 
                   "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
            state="readonly",
            width=15
        )
        
        # Export button
        self.export_button = ttk.Button(
            self.control_frame,
            text="Export to CSV",
            command=self._on_export
        )
        
        # Create treeview first
        self._create_treeview()
        
        # Then create scrollbars that reference the tree
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
    
    def _create_treeview(self):
        """Create the treeview widget."""
        # Define columns
        columns = ["Feature", "Index", "Group", "Type"]
        for i in range(10):
            columns.append(f"T{i}")
        
        # Create treeview
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=20)
        
        # Configure column headings
        self.tree.heading("Feature", text="Feature Name")
        self.tree.heading("Index", text="Idx")
        self.tree.heading("Group", text="Group")
        self.tree.heading("Type", text="Type")
        
        for i in range(10):
            self.tree.heading(f"T{i}", text=f"T{i}")
        
        # Configure column widths
        self.tree.column("Feature", width=get_column_width("feature_name"))
        self.tree.column("Index", width=get_column_width("feature_index"))
        self.tree.column("Group", width=get_column_width("feature_group"))
        self.tree.column("Type", width=get_column_width("feature_type"))
        
        for i in range(10):
            self.tree.column(f"T{i}", width=get_column_width(f"timestep_{i}"))
    
    def _setup_layout(self):
        """Setup the widget layout."""
        # Control frame
        self.control_frame.pack(fill="x", pady=(0, 10))
        self.normalized_check.pack(side="left")
        self.filter_combo.pack(side="left", padx=(20, 0))
        self.export_button.pack(side="right")
        
        # Treeview and scrollbars - use pack for consistency
        # Pack scrollbars first
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)
    
    def _bind_events(self):
        """Bind widget events."""
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)
        
        # Bind treeview selection
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_change)
    
    def _on_normalized_toggle(self):
        """Handle normalized data toggle."""
        self.refresh()
    
    def _on_filter_change(self, event=None):
        """Handle feature group filter change."""
        filter_group = self.filter_var.get()
        self.controller.set_feature_group_filter(filter_group)
        self.refresh()
    
    def _on_selection_change(self, event):
        """Handle treeview selection change."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            if values and len(values) > 1:
                feature_idx = int(values[1])
                self._show_feature_details(feature_idx)
    
    def _on_export(self):
        """Handle export to CSV."""
        # This would integrate with the export dialog
        print("Export to CSV functionality would be implemented here")
    
    def _show_feature_details(self, feature_idx: int):
        """Show detailed information for a selected feature."""
        feature_catalog = self.controller.get_feature_catalog()
        if not feature_catalog:
            return
        
        feature_name = feature_catalog.get_feature_name(feature_idx)
        feature_group = feature_catalog.get_feature_group(feature_idx)
        feature_type = feature_catalog.get_feature_type(feature_idx)
        
        # Update status or show in a separate panel
        print(f"Selected feature {feature_idx}: {feature_name} ({feature_group}, {feature_type})")
    
    def refresh(self):
        """Refresh the feature table display."""
        self._populate_table()
    
    def _clear_table(self):
        """Clear all items from the treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def _populate_table(self):
        """Populate the treeview with feature data."""
        if not self.controller.is_data_loaded():
            return
        
        # Clear existing items FIRST to prevent accumulation
        self._clear_table()
        
        # Get current sequence data
        sequence_data = self.controller.get_current_sequence_data()
        if not sequence_data:
            return
        
        # Get services
        feature_catalog = self.controller.get_feature_catalog()
        mapping_service = self.controller.get_mapping_service()
        normalization_service = self.controller.get_normalization_service()
        
        if not feature_catalog:
            return
        
        # Determine which sequence to show
        show_normalized = self.normalized_var.get()
        if show_normalized and sequence_data.get('normalized_sequence') is not None:
            sequence = sequence_data['normalized_sequence']
        else:
            sequence = sequence_data['input_sequence']
        
        # Get filter
        filter_group = self.filter_var.get()
        
        # Populate table
        rows_inserted = 0
        for feature_idx in range(128):
            feature_name = feature_catalog.get_feature_name(feature_idx)
            feature_group = feature_catalog.get_feature_group(feature_idx)
            feature_type = feature_catalog.get_feature_type(feature_idx)
            
            # Apply filter
            if filter_group != "All" and feature_group != filter_group:
                continue
            
            # Get values for all timesteps
            values = [feature_name, feature_idx, feature_group, feature_type]
            
            for timestep in range(10):
                raw_value = sequence[timestep, feature_idx]
                
                # Format value
                show_translations = self.controller.get_state().show_translations
                mapping_service = self.controller.get_mapping_service()
                translate_func = mapping_service.translate if mapping_service else None
                formatted_value = format_value_for_display(
                    raw_value, feature_idx, 
                    show_translations,
                    translate_func
                )
                values.append(formatted_value)
            
            # Insert row
            item = self.tree.insert("", "end", values=values)
            rows_inserted += 1
            
            # Set row color based on feature group
            group_color = get_feature_group_color(feature_group)
            self.tree.tag_configure(f"group_{feature_group}", background=group_color)
            self.tree.item(item, tags=(f"group_{feature_group}",))
            
            # Add tooltip
            tooltip_text = create_feature_tooltip_text(
                feature_idx, feature_name, feature_group, feature_type
            )
            self.tooltip_manager.add_tooltip(
                self.tree, tooltip_text, delay=1000
            )
        

    
    def update(self):
        """Update the feature table display."""
        self._update_sequence_display()
    
    def _update_sequence_display(self):
        """Update display for current sequence."""
        if not self.controller.is_data_loaded():
            return
        
        # Update sequence number in window title or status
        current_seq = self.controller.get_state().current_sequence
        total_seqs = self.controller.get_sequence_count()
        
        # Update the display
        self.refresh()
    
    def destroy(self):
        """Clean up resources."""
        self.tooltip_manager.destroy()
        super().destroy()
