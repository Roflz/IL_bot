"""
Feature Analysis View

View for feature analysis and statistics.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, Dict, Any, List

class FeatureAnalysisView(ttk.Frame):
    """Feature Analysis view with full interactive functionality"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.feature_analysis_tree = None
        self.feature_analysis_tooltip = None
        self.feature_analysis_tooltip_text = ""
        
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the widgets for the feature analysis view"""
        # Create a canvas and scrollbar for the entire frame
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Info frame
        self.info_frame = ttk.Frame(self.scrollable_frame)
        self.feature_analysis_info_label = ttk.Label(self.info_frame, text="", font=("Arial", 10))
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.scrollable_frame)
        
        # Show translations checkbox
        self.show_analysis_translations = tk.BooleanVar(value=True)
        self.analysis_view_toggle = ttk.Checkbutton(
            self.controls_frame, 
            text="🔍 Show Hash Translations", 
            variable=self.show_analysis_translations,
            command=self._update_feature_analysis_display
        )
        
        # Normalization toggle for analysis
        self.show_analysis_normalized = tk.BooleanVar(value=False)
        self.analysis_normalization_toggle = ttk.Checkbutton(
            self.controls_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_analysis_normalized,
            command=self._update_feature_analysis_display
        )
        
        # Refresh button
        self.refresh_button = ttk.Button(self.controls_frame, text="🔄 Refresh Analysis", command=self._refresh_feature_analysis)
        
        # Feature group filter for analysis
        self.analysis_feature_group_filter = tk.StringVar(value="All")
        self.analysis_filter_combo = ttk.Combobox(
            self.controls_frame, 
            textvariable=self.analysis_feature_group_filter,
            values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
            state="readonly",
            width=20
        )
        self.analysis_filter_combo.bind('<<ComboboxSelected>>', self._on_analysis_feature_group_filter_changed)
        
        # Export button
        self.export_button = ttk.Button(self.controls_frame, text="💾 Export Analysis", command=self._export_feature_analysis)
        
        # Feature structure summary frame
        self.summary_frame = ttk.LabelFrame(self.scrollable_frame, text="Feature Structure Summary", padding="5")
        
        # Create summary labels in a grid
        self._create_feature_summary_grid()
        
        # Create Treeview for feature analysis
        self._create_feature_analysis_table()
        
        # Feature details panel for selected features
        self._create_feature_details_panel()
        
    def _setup_layout(self):
        """Set up the layout of all widgets"""
        # Info frame
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.feature_analysis_info_label.pack(side=tk.LEFT)
        
        # Controls frame
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Pack controls in order
        self.analysis_view_toggle.pack(side=tk.LEFT, padx=(0, 20))
        self.analysis_normalization_toggle.pack(side=tk.LEFT, padx=(0, 20))
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(self.controls_frame, text="Filter by Feature Group:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 10))
        self.analysis_filter_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.export_button.pack(side=tk.LEFT)
        
        # Summary frame
        self.summary_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Pack the canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize feature analysis data
        self._analyze_features()
        
    def _create_feature_summary_grid(self):
        """Create summary labels in a grid layout"""
        # Create a frame for the grid
        grid_frame = ttk.Frame(self.summary_frame)
        grid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Feature group labels
        feature_groups = ["Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", 
                         "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"]
        
        # Create labels in a grid (3 columns)
        for i, group in enumerate(feature_groups):
            row = i // 3
            col = i % 3
            
            # Group name label
            group_label = ttk.Label(grid_frame, text=f"{group}:", font=("Arial", 9, "bold"))
            group_label.grid(row=row, column=col*2, sticky="w", padx=(0, 5), pady=2)
            
            # Count label (will be updated)
            count_label = ttk.Label(grid_frame, text="0", font=("Arial", 9))
            count_label.grid(row=row, column=col*2+1, sticky="w", padx=(0, 20), pady=2)
            
            # Store reference for updating
            setattr(self, f"{group.lower().replace(' ', '_')}_count_label", count_label)
    
    def _create_feature_analysis_table(self):
        """Create the feature analysis table with Treeview"""
        # Create frame for the table
        table_frame = ttk.Frame(self.scrollable_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview
        columns = ['Feature', 'Index', 'Group', 'Type', 'Min', 'Max', 'Mean', 'Std', 'Most Common', 'Description']
        self.feature_analysis_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.feature_analysis_tree.heading(col, text=col)
            if col == 'Feature':
                self.feature_analysis_tree.column(col, width=150, minwidth=120)
            elif col == 'Index':
                self.feature_analysis_tree.column(col, width=50, minwidth=50)
            elif col == 'Group':
                self.feature_analysis_tree.column(col, width=100, minwidth=80)
            elif col == 'Type':
                self.feature_analysis_tree.column(col, width=80, minwidth=60)
            elif col in ['Min', 'Max', 'Mean', 'Std']:
                self.feature_analysis_tree.column(col, width=80, minwidth=60)
            elif col == 'Most Common':
                self.feature_analysis_tree.column(col, width=100, minwidth=80)
            else:  # Description
                self.feature_analysis_tree.column(col, width=200, minwidth=150)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.feature_analysis_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.feature_analysis_tree.xview)
        self.feature_analysis_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.feature_analysis_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Bind tooltip events
        self.feature_analysis_tree.bind('<Motion>', self._on_analysis_table_motion)
        self.feature_analysis_tree.bind('<Leave>', self._on_analysis_table_leave)
        self.feature_analysis_tree.bind('<<TreeviewSelect>>', self._on_feature_selected)
    
    def _create_feature_details_panel(self):
        """Create the feature details panel for selected features"""
        # Create frame for details
        details_frame = ttk.LabelFrame(self.scrollable_frame, text="Feature Details", padding="5")
        details_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Create buttons frame
        buttons_frame = ttk.Frame(details_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Feature details buttons
        self.show_values_button = ttk.Button(buttons_frame, text="📊 Show All Values", command=self._show_feature_all_values)
        self.show_timeline_button = ttk.Button(buttons_frame, text="📈 Show Timeline", command=self._show_feature_timeline)
        self.show_distribution_button = ttk.Button(buttons_frame, text="📊 Show Distribution", command=self._show_feature_distribution)
        
        # Pack buttons
        self.show_values_button.pack(side=tk.LEFT, padx=(0, 10))
        self.show_timeline_button.pack(side=tk.LEFT, padx=(0, 10))
        self.show_distribution_button.pack(side=tk.LEFT)
        
        # Details text area
        self.details_text = tk.Text(details_frame, height=8, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        # Pack text area
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Make text read-only
        self.details_text.configure(state="disabled")
    
    def _analyze_features(self):
        """Analyze features and populate the analysis table"""
        try:
            # Clear existing items
            for item in self.feature_analysis_tree.get_children():
                self.feature_analysis_tree.delete(item)
            
            # Get current sequence data
            sequence_data = self.controller.get_current_sequence_data()
            if not sequence_data:
                return
            
            input_sequence = sequence_data['input_sequence']
            if input_sequence is None:
                return
            
            # Get feature catalog
            feature_catalog = self.controller.get_feature_catalog()
            if not feature_catalog:
                return
            
            # Analyze each feature
            for feature_idx in range(128):  # Assuming 128 features
                try:
                    # Get feature info
                    feature_name = feature_catalog.get_feature_name(feature_idx)
                    feature_group = feature_catalog.get_feature_group(feature_idx)
                    feature_type = feature_catalog.get_feature_type(feature_idx)
                    
                    # Apply filter
                    if (self.analysis_feature_group_filter.get() != "All" and 
                        feature_group != self.analysis_feature_group_filter.get()):
                        continue
                    
                    # Extract feature values across timesteps
                    if len(input_sequence.shape) > 1:
                        values = input_sequence[:, feature_idx]
                    else:
                        values = [input_sequence[feature_idx]]
                    
                    # Calculate statistics
                    values_array = np.array(values)
                    min_val = float(values_array.min())
                    max_val = float(values_array.max())
                    mean_val = float(values_array.mean())
                    std_val = float(values_array.std())
                    
                    # Determine most common value
                    most_common = self._get_most_common_value(values, feature_idx)
                    
                    # Get feature description
                    description = self._get_feature_description(feature_idx, feature_name, feature_group)
                    
                    # Insert into table
                    row_values = [
                        feature_name,
                        feature_idx,
                        feature_group,
                        feature_type,
                        f"{min_val:.4f}",
                        f"{max_val:.4f}",
                        f"{mean_val:.4f}",
                        f"{std_val:.4f}",
                        str(most_common),
                        description
                    ]
                    
                    item = self.feature_analysis_tree.insert("", "end", values=row_values)
                    
                    # Apply row coloring based on feature group
                    self._apply_analysis_row_coloring(item, feature_group)
                    
                except Exception as e:
                    print(f"Error analyzing feature {feature_idx}: {e}")
                    continue
            
            # Update summary grid
            self._update_summary_grid()
            
        except Exception as e:
            print(f"Error in feature analysis: {e}")
    
    def _update_summary_grid(self):
        """Update the feature summary grid with current counts"""
        try:
            # Count features by group
            feature_groups = ["Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", 
                             "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"]
            
            group_counts = {}
            for item in self.feature_analysis_tree.get_children():
                values = self.feature_analysis_tree.item(item)['values']
                if len(values) >= 3:
                    group = values[2]  # Group column
                    group_counts[group] = group_counts.get(group, 0) + 1
            
            # Update labels
            for group in feature_groups:
                count = group_counts.get(group, 0)
                label_name = f"{group.lower().replace(' ', '_')}_count_label"
                if hasattr(self, label_name):
                    label = getattr(self, label_name)
                    label.config(text=str(count))
                    
        except Exception as e:
            print(f"Error updating summary grid: {e}")
    
    def _get_most_common_value(self, values, feature_idx):
        """Get the most common value for a feature"""
        try:
            # Convert to list if numpy array
            if hasattr(values, 'tolist'):
                values = values.tolist()
            
            # Count occurrences
            from collections import Counter
            counter = Counter(values)
            most_common = counter.most_common(1)
            
            if most_common:
                value, count = most_common[0]
                return f"{value} ({count}x)"
            else:
                return "N/A"
                
        except Exception as e:
            return "Error"
    
    def _get_feature_description(self, feature_idx, feature_name, category):
        """Get a description for a feature"""
        try:
            # Basic descriptions based on category
            descriptions = {
                "Player": "Player state and position information",
                "Interaction": "Player interaction with game objects",
                "Camera": "Camera position and orientation",
                "Inventory": "Inventory item states and quantities",
                "Bank": "Bank item states and quantities",
                "Phase Context": "Current game phase and context",
                "Game Objects": "Game object states and positions",
                "NPCs": "NPC states and positions",
                "Tabs": "Interface tab states",
                "Skills": "Player skill levels and experience",
                "Timestamp": "Temporal information"
            }
            
            base_desc = descriptions.get(category, "Game state feature")
            
            # Add specific info for certain features
            if "timestamp" in feature_name.lower():
                return f"{base_desc} - Time since session start (ms)"
            elif "x" in feature_name.lower() and "y" in feature_name.lower():
                return f"{base_desc} - Coordinate pair"
            elif "count" in feature_name.lower():
                return f"{base_desc} - Count or quantity"
            else:
                return base_desc
                
        except Exception as e:
            return "Description unavailable"
    
    def _apply_analysis_row_coloring(self, item, category):
        """Apply row coloring based on feature category"""
        try:
            # Color scheme for different feature groups
            colors = {
                "Player": "#e6f3ff",
                "Interaction": "#fff2e6",
                "Camera": "#f0e6ff",
                "Inventory": "#e6ffe6",
                "Bank": "#ffe6e6",
                "Phase Context": "#ffffe6",
                "Game Objects": "#ffe6f0",
                "NPCs": "#e6f0ff",
                "Tabs": "#f0ffe6",
                "Skills": "#ffe6ff",
                "Timestamp": "#f6f6f6"
            }
            
            bg_color = colors.get(category, "#ffffff")
            self.feature_analysis_tree.tag_configure(f"bg_{category}", background=bg_color)
            self.feature_analysis_tree.item(item, tags=(f"bg_{category}",))
            
        except Exception as e:
            print(f"Error applying row coloring: {e}")
    
    def _update_feature_analysis_display(self):
        """Update the feature analysis display"""
        self._analyze_features()
    
    def _refresh_feature_analysis(self):
        """Refresh the feature analysis"""
        self._analyze_features()
    
    def _export_feature_analysis(self):
        """Export feature analysis to CSV"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Get all items from the table
                items = self.feature_analysis_tree.get_children()
                if not items:
                    messagebox.showwarning("No Data", "No data to export!")
                    return
                
                # Write CSV file
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write headers
                    columns = self.feature_analysis_tree["columns"]
                    writer.writerow(columns)
                    
                    # Write data rows
                    for item in items:
                        values = self.feature_analysis_tree.item(item)['values']
                        writer.writerow(values)
                
                messagebox.showinfo("Success", f"Feature analysis exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def _on_analysis_feature_group_filter_changed(self, event=None):
        """Handle feature group filter change"""
        self._analyze_features()
    
    def _on_feature_selected(self, event):
        """Handle feature selection in the analysis table"""
        try:
            selection = self.feature_analysis_tree.selection()
            if selection:
                item = selection[0]
                values = self.feature_analysis_tree.item(item)['values']
                
                if len(values) >= 3:
                    feature_name = values[0]
                    feature_idx = values[1]
                    feature_group = values[2]
                    
                    # Update details text
                    self.details_text.configure(state="normal")
                    self.details_text.delete(1.0, tk.END)
                    
                    details_text = f"Selected Feature: {feature_name}\n"
                    details_text += f"Index: {feature_idx}\n"
                    details_text += f"Group: {feature_group}\n"
                    details_text += f"Type: {values[3] if len(values) > 3 else 'N/A'}\n\n"
                    
                    details_text += f"Statistics:\n"
                    details_text += f"• Min: {values[4] if len(values) > 4 else 'N/A'}\n"
                    details_text += f"• Max: {values[5] if len(values) > 5 else 'N/A'}\n"
                    details_text += f"• Mean: {values[6] if len(values) > 6 else 'N/A'}\n"
                    details_text += f"• Std: {values[7] if len(values) > 7 else 'N/A'}\n"
                    details_text += f"• Most Common: {values[8] if len(values) > 8 else 'N/A'}\n\n"
                    
                    details_text += f"Description: {values[9] if len(values) > 9 else 'N/A'}"
                    
                    self.details_text.insert(tk.END, details_text)
                    self.details_text.configure(state="disabled")
                    
        except Exception as e:
            print(f"Error handling feature selection: {e}")
    
    def _show_feature_all_values(self):
        """Show all values for the selected feature"""
        messagebox.showinfo("Info", "Show all values functionality would be implemented here")
    
    def _show_feature_timeline(self):
        """Show timeline for the selected feature"""
        messagebox.showinfo("Info", "Show timeline functionality would be implemented here")
    
    def _show_feature_distribution(self):
        """Show distribution for the selected feature"""
        messagebox.showinfo("Info", "Show distribution functionality would be implemented here")
    
    def _on_analysis_table_motion(self, event):
        """Handle mouse motion over analysis table for tooltips"""
        # Get the item under cursor
        item = self.feature_analysis_tree.identify_row(event.y)
        if item:
            # Get column under cursor
            column = self.feature_analysis_tree.identify_column(event.x)
            if column:
                # Get item values
                values = self.feature_analysis_tree.item(item)['values']
                if values:
                    col_idx = int(column[1]) - 1  # Convert #1, #2, etc. to 0, 1, etc.
                    if col_idx < len(values):
                        value = values[col_idx]
                        tooltip_text = f"Column {column}: {value}"
                        
                        if tooltip_text != self.feature_analysis_tooltip_text:
                            self.feature_analysis_tooltip_text = tooltip_text
                            self._show_analysis_tooltip(event.x_root, event.y_root, tooltip_text)
    
    def _on_analysis_table_leave(self, event):
        """Handle mouse leave from analysis table"""
        self._hide_analysis_tooltip()
    
    def _show_analysis_tooltip(self, x, y, text):
        """Show tooltip for analysis table"""
        self._hide_analysis_tooltip()
        
        self.feature_analysis_tooltip = tk.Toplevel()
        self.feature_analysis_tooltip.wm_overrideredirect(True)
        self.feature_analysis_tooltip.wm_geometry(f"+{x+10}+{y+10}")
        
        label = ttk.Label(self.feature_analysis_tooltip, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()
    
    def _hide_analysis_tooltip(self):
        """Hide analysis tooltip"""
        if self.feature_analysis_tooltip:
            self.feature_analysis_tooltip.destroy()
            self.feature_analysis_tooltip = None
    
    def refresh(self):
        """Refresh the view with current data"""
        self._analyze_features()
