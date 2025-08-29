"""
Sequence Alignment View

View for sequence alignment analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, Dict, Any, List

class SequenceAlignmentView(ttk.Frame):
    """Sequence Alignment view with full interactive functionality"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.sequence_canvas = None
        self.sequence_tooltip = None
        self.sequence_tooltip_text = ""
        
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the widgets for the sequence alignment view"""
        # Info frame
        self.info_frame = ttk.Frame(self)
        self.sequence_info_label = ttk.Label(self.info_frame, text="", font=("Arial", 10))
        
        # Controls frame
        self.controls_frame = ttk.Frame(self)
        
        # Show normalized toggle
        self.show_sequence_normalized = tk.BooleanVar(value=False)
        self.sequence_normalization_toggle = ttk.Checkbutton(
            self.controls_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_sequence_normalized,
            command=self._update_sequence_display
        )
        
        # Feature group filter
        ttk.Label(self.controls_frame, text="Filter by Feature Group:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 10))
        self.sequence_feature_group_filter = tk.StringVar(value="All")
        self.sequence_filter_combo = ttk.Combobox(
            self.controls_frame, 
            textvariable=self.sequence_feature_group_filter,
            values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
            state="readonly",
            width=20
        )
        self.sequence_filter_combo.bind('<<ComboboxSelected>>', self._on_sequence_feature_group_filter_changed)
        
        # Export button
        self.export_button = ttk.Button(self.controls_frame, text="💾 Export Alignment", command=self._export_sequence_alignment)
        
        # Main display frame
        self.display_frame = ttk.Frame(self)
        
        # Create notebook for different views
        self.sequence_notebook = ttk.Notebook(self.display_frame)
        
        # Sequence overview tab
        self.sequence_overview_tab = ttk.Frame(self.sequence_notebook)
        self.sequence_notebook.add(self.sequence_overview_tab, text="Sequence Overview")
        self._create_sequence_overview_tab()
        
        # Feature visualization tab
        self.feature_visualization_tab = ttk.Frame(self.sequence_notebook)
        self.sequence_notebook.add(self.feature_visualization_tab, text="Feature Visualization")
        self._create_feature_visualization_tab()
        
        # Sequence comparison tab
        self.sequence_comparison_tab = ttk.Frame(self.sequence_notebook)
        self.sequence_notebook.add(self.sequence_comparison_tab, text="Sequence Comparison")
        self._create_sequence_comparison_tab()
        
    def _setup_layout(self):
        """Set up the layout of all widgets"""
        # Info frame
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.sequence_info_label.pack(side=tk.LEFT)
        
        # Controls frame
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Pack controls in order
        self.sequence_normalization_toggle.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(self.controls_frame, text="Filter by Feature Group:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 10))
        self.sequence_filter_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.export_button.pack(side=tk.LEFT)
        
        # Main display
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.sequence_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Initial update
        self._update_sequence_display()
        
    def _create_sequence_overview_tab(self):
        """Create the sequence overview tab"""
        # Create scrollable frame
        overview_frame = ttk.Frame(self.sequence_overview_tab)
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Sequence info section
        info_section = ttk.LabelFrame(overview_frame, text="Sequence Information", padding="5")
        info_section.pack(fill=tk.X, pady=(0, 10))
        
        # Info grid
        info_grid = ttk.Frame(info_section)
        info_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Row 1
        ttk.Label(info_grid, text="Sequence Index:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=2)
        self.seq_index_label = ttk.Label(info_grid, text="N/A")
        self.seq_index_label.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=2)
        
        ttk.Label(info_grid, text="Input Features Shape:", font=("Arial", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 10), pady=2)
        self.seq_shape_label = ttk.Label(info_grid, text="N/A")
        self.seq_shape_label.grid(row=0, column=3, sticky="w", pady=2)
        
        # Row 2
        ttk.Label(info_grid, text="Target Actions Count:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=2)
        self.target_count_label = ttk.Label(info_grid, text="N/A")
        self.target_count_label.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=2)
        
        ttk.Label(info_grid, text="Normalized Features:", font=("Arial", 9, "bold")).grid(row=1, column=2, sticky="w", padx=(0, 10), pady=2)
        self.norm_shape_label = ttk.Label(info_grid, text="N/A")
        self.norm_shape_label.grid(row=1, column=3, sticky="w", pady=2)
        
        # Feature summary section
        summary_section = ttk.LabelFrame(overview_frame, text="Feature Summary", padding="5")
        summary_section.pack(fill=tk.X, pady=(0, 10))
        
        # Feature summary grid
        self.feature_summary_grid = ttk.Frame(summary_section)
        self.feature_summary_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Create feature summary labels
        self._create_feature_summary_labels()
        
        # Sample values section
        sample_section = ttk.LabelFrame(overview_frame, text="Sample Feature Values (First Timestep)", padding="5")
        sample_section.pack(fill=tk.X, pady=(0, 10))
        
        # Sample values text
        self.sample_values_text = tk.Text(sample_section, height=8, wrap=tk.WORD, font=("Consolas", 9))
        sample_scrollbar = ttk.Scrollbar(sample_section, orient="vertical", command=self.sample_values_text.yview)
        self.sample_values_text.configure(yscrollcommand=sample_scrollbar.set)
        
        self.sample_values_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        sample_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Make text read-only
        self.sample_values_text.configure(state="disabled")
        
    def _create_feature_visualization_tab(self):
        """Create the feature visualization tab"""
        # Create frame for visualization
        viz_frame = ttk.Frame(self.feature_visualization_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Controls for visualization
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(viz_controls, text="Feature Index:").pack(side=tk.LEFT, padx=(0, 5))
        self.feature_index_var = tk.StringVar(value="0")
        self.feature_index_entry = ttk.Entry(viz_controls, textvariable=self.feature_index_var, width=10)
        self.feature_index_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(viz_controls, text="Show Feature", command=self._show_feature_visualization).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(viz_controls, text="Show All Features", command=self._show_all_features_visualization).pack(side=tk.LEFT)
        
        # Visualization area
        self.viz_canvas = tk.Canvas(viz_frame, bg="white", height=300)
        viz_scrollbar = ttk.Scrollbar(viz_frame, orient="vertical", command=self.viz_canvas.yview)
        self.viz_canvas.configure(yscrollcommand=viz_scrollbar.set)
        
        self.viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        viz_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_sequence_comparison_tab(self):
        """Create the sequence comparison tab"""
        # Create frame for comparison
        comp_frame = ttk.Frame(self.sequence_comparison_tab)
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Controls for comparison
        comp_controls = ttk.Frame(comp_frame)
        comp_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(comp_controls, text="Compare with Sequence:").pack(side=tk.LEFT, padx=(0, 5))
        self.compare_sequence_var = tk.StringVar(value="1")
        self.compare_sequence_entry = ttk.Entry(comp_controls, textvariable=self.compare_sequence_var, width=10)
        self.compare_sequence_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(comp_controls, text="Compare", command=self._compare_sequences).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(comp_controls, text="Show Differences", command=self._show_sequence_differences).pack(side=tk.LEFT)
        
        # Comparison area
        self.comp_text = tk.Text(comp_frame, height=15, wrap=tk.WORD, font=("Consolas", 9))
        comp_scrollbar = ttk.Scrollbar(comp_frame, orient="vertical", command=self.comp_text.yview)
        self.comp_text.configure(yscrollcommand=comp_scrollbar.set)
        
        self.comp_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        comp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Make text read-only
        self.comp_text.configure(state="disabled")
        
    def _create_feature_summary_labels(self):
        """Create feature summary labels in a grid"""
        # Feature groups
        feature_groups = ["Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", 
                         "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"]
        
        # Create labels in a grid (3 columns)
        for i, group in enumerate(feature_groups):
            row = i // 3
            col = i % 3
            
            # Group name label
            group_label = ttk.Label(self.feature_summary_grid, text=f"{group}:", font=("Arial", 9, "bold"))
            group_label.grid(row=row, column=col*2, sticky="w", padx=(0, 5), pady=2)
            
            # Count label (will be updated)
            count_label = ttk.Label(self.feature_summary_grid, text="0", font=("Arial", 9))
            count_label.grid(row=row, column=col*2+1, sticky="w", padx=(0, 20), pady=2)
            
            # Store reference for updating
            setattr(self, f"seq_{group.lower().replace(' ', '_')}_count_label", count_label)
    
    def _update_sequence_display(self):
        """Update the sequence alignment display"""
        try:
            # Get current sequence data
            sequence_data = self.controller.get_current_sequence_data()
            if not sequence_data:
                return
            
            # Update sequence info
            self.seq_index_label.config(text=str(sequence_data['sequence_idx']))
            
            input_sequence = sequence_data['input_sequence']
            if input_sequence is not None:
                self.seq_shape_label.config(text=str(input_sequence.shape))
            else:
                self.seq_shape_label.config(text="N/A")
            
            target_sequence = sequence_data['target_sequence']
            if target_sequence is not None:
                self.target_count_label.config(text=str(len(target_sequence)))
            else:
                self.target_count_label.config(text="N/A")
            
            normalized_sequence = sequence_data.get('normalized_sequence')
            if normalized_sequence is not None:
                self.norm_shape_label.config(text=str(normalized_sequence.shape))
            else:
                self.norm_shape_label.config(text="N/A")
            
            # Update feature summary
            self._update_feature_summary()
            
            # Update sample values
            self._update_sample_values()
            
        except Exception as e:
            print(f"Error updating sequence display: {e}")
    
    def _update_feature_summary(self):
        """Update the feature summary grid"""
        try:
            # Get feature catalog
            feature_catalog = self.controller.get_feature_catalog()
            if not feature_catalog:
                return
            
            # Count features by group
            feature_groups = ["Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", 
                             "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"]
            
            group_counts = {}
            for i in range(128):  # Assuming 128 features
                group = feature_catalog.get_feature_group(i)
                group_counts[group] = group_counts.get(group, 0) + 1
            
            # Update labels
            for group in feature_groups:
                count = group_counts.get(group, 0)
                label_name = f"seq_{group.lower().replace(' ', '_')}_count_label"
                if hasattr(self, label_name):
                    label = getattr(self, label_name)
                    label.config(text=str(count))
                    
        except Exception as e:
            print(f"Error updating feature summary: {e}")
    
    def _update_sample_values(self):
        """Update the sample values text"""
        try:
            # Get current sequence data
            sequence_data = self.controller.get_current_sequence_data()
            if not sequence_data:
                return
            
            input_sequence = sequence_data['input_sequence']
            if input_sequence is None:
                return
            
            # Clear text
            self.sample_values_text.configure(state="normal")
            self.sample_values_text.delete(1.0, tk.END)
            
            # Get feature catalog
            feature_catalog = self.controller.get_feature_catalog()
            if not feature_catalog:
                return
            
            # Show first few features
            if len(input_sequence.shape) > 1:
                first_timestep = input_sequence[0]
                for i in range(min(10, len(first_timestep))):
                    feature_name = feature_catalog.get_feature_name(i)
                    value = first_timestep[i]
                    self.sample_values_text.insert(tk.END, f"Feature {i:3d} ({feature_name}): {value:8.4f}\n")
            else:
                for i in range(min(10, len(input_sequence))):
                    feature_name = feature_catalog.get_feature_name(i)
                    value = input_sequence[i]
                    self.sample_values_text.insert(tk.END, f"Feature {i:3d} ({feature_name}): {value:8.4f}\n")
            
            # Make text read-only
            self.sample_values_text.configure(state="disabled")
            
        except Exception as e:
            print(f"Error updating sample values: {e}")
    
    def _on_sequence_feature_group_filter_changed(self, event=None):
        """Handle feature group filter change"""
        self._update_sequence_display()
    
    def _show_feature_visualization(self):
        """Show visualization for a specific feature"""
        try:
            feature_idx = int(self.feature_index_var.get())
            if feature_idx < 0 or feature_idx >= 128:
                messagebox.showerror("Error", "Feature index must be between 0 and 127")
                return
            
            # Get current sequence data
            sequence_data = self.controller.get_current_sequence_data()
            if not sequence_data:
                return
            
            input_sequence = sequence_data['input_sequence']
            if input_sequence is None:
                return
            
            # Extract feature values across timesteps
            if len(input_sequence.shape) > 1:
                values = input_sequence[:, feature_idx]
            else:
                values = [input_sequence[feature_idx]]
            
            # Create simple visualization
            self._draw_feature_visualization(feature_idx, values)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid feature index")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show feature visualization: {e}")
    
    def _show_all_features_visualization(self):
        """Show visualization for all features"""
        try:
            # Get current sequence data
            sequence_data = self.controller.get_current_sequence_data()
            if not sequence_data:
                return
            
            input_sequence = sequence_data['input_sequence']
            if input_sequence is None:
                return
            
            # Create overview visualization
            self._draw_all_features_visualization(input_sequence)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show all features visualization: {e}")
    
    def _draw_feature_visualization(self, feature_idx, values):
        """Draw visualization for a single feature"""
        # Clear canvas
        self.viz_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.viz_canvas.winfo_width()
        canvas_height = self.viz_canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet sized
            canvas_width = 600
            canvas_height = 300
        
        # Draw title
        self.viz_canvas.create_text(canvas_width//2, 20, text=f"Feature {feature_idx} Values Across Timesteps", 
                                   font=("Arial", 12, "bold"))
        
        # Draw values as bars
        if len(values) > 0:
            bar_width = (canvas_width - 100) / len(values)
            max_val = max(abs(v) for v in values) if values else 1
            
            for i, value in enumerate(values):
                x = 50 + i * bar_width + bar_width/2
                height = (abs(value) / max_val) * (canvas_height - 100) if max_val > 0 else 0
                y = canvas_height - 50 - height
                
                # Draw bar
                color = "blue" if value >= 0 else "red"
                self.viz_canvas.create_rectangle(x - bar_width/3, y, x + bar_width/3, canvas_height - 50, 
                                               fill=color, outline="black")
                
                # Draw value label
                self.viz_canvas.create_text(x, y - 10, text=f"{value:.2f}", font=("Arial", 8))
                
                # Draw timestep label
                self.viz_canvas.create_text(x, canvas_height - 30, text=f"T{i}", font=("Arial", 8))
    
    def _draw_all_features_visualization(self, input_sequence):
        """Draw visualization for all features"""
        # Clear canvas
        self.viz_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.viz_canvas.winfo_width()
        canvas_height = self.viz_canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet sized
            canvas_width = 600
            canvas_height = 300
        
        # Draw title
        self.viz_canvas.create_text(canvas_width//2, 20, text="All Features Overview (First Timestep)", 
                                   font=("Arial", 12, "bold"))
        
        # Draw heatmap-like visualization
        if len(input_sequence.shape) > 1:
            features = input_sequence[0]  # First timestep
        else:
            features = input_sequence
        
        if len(features) > 0:
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(len(features))))
            cell_width = (canvas_width - 100) / grid_size
            cell_height = (canvas_height - 100) / grid_size
            
            # Find value range for normalization
            max_val = max(abs(v) for v in features) if features else 1
            
            for i, value in enumerate(features):
                row = i // grid_size
                col = i % grid_size
                
                x = 50 + col * cell_width
                y = 50 + row * cell_height
                
                # Normalize value to 0-1 range for color intensity
                intensity = abs(value) / max_val if max_val > 0 else 0
                
                # Create color (red for negative, blue for positive)
                if value >= 0:
                    color = f"#{int(255 * (1-intensity)):02x}{int(255 * (1-intensity)):02x}ff"
                else:
                    color = f"#ff{int(255 * (1-intensity)):02x}{int(255 * (1-intensity)):02x}"
                
                # Draw cell
                self.viz_canvas.create_rectangle(x, y, x + cell_width, y + cell_height, 
                                               fill=color, outline="gray")
                
                # Draw feature index
                self.viz_canvas.create_text(x + cell_width/2, y + cell_height/2, text=str(i), 
                                           font=("Arial", 6))
    
    def _compare_sequences(self):
        """Compare current sequence with another sequence"""
        try:
            compare_idx = int(self.compare_sequence_var.get())
            if compare_idx < 0:
                messagebox.showerror("Error", "Sequence index must be non-negative")
                return
            
            # Get current and compare sequence data
            current_data = self.controller.get_current_sequence_data()
            if not current_data:
                return
            
            # Get compare sequence data (this would need to be implemented in controller)
            # For now, show a placeholder message
            messagebox.showinfo("Info", f"Sequence comparison with sequence {compare_idx} would be implemented here")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid sequence index")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare sequences: {e}")
    
    def _show_sequence_differences(self):
        """Show differences between sequences"""
        messagebox.showinfo("Info", "Show sequence differences functionality would be implemented here")
    
    def _export_sequence_alignment(self):
        """Export sequence alignment data"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Get current sequence data
                sequence_data = self.controller.get_current_sequence_data()
                if not sequence_data:
                    messagebox.showwarning("No Data", "No sequence data to export!")
                    return
                
                # Write CSV file
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write sequence info
                    writer.writerow(["Sequence Information"])
                    writer.writerow(["Sequence Index", sequence_data['sequence_idx']])
                    writer.writerow(["Input Features Shape", str(sequence_data['input_sequence'].shape) if sequence_data['input_sequence'] is not None else "N/A"])
                    writer.writerow(["Target Actions Count", len(sequence_data['target_sequence']) if sequence_data['target_sequence'] else "N/A"])
                    writer.writerow(["Normalized Features Shape", str(sequence_data.get('normalized_sequence', {}).shape) if sequence_data.get('normalized_sequence') is not None else "N/A"])
                    writer.writerow([])
                    
                    # Write feature values
                    if sequence_data['input_sequence'] is not None:
                        writer.writerow(["Feature Values (First Timestep)"])
                        if len(sequence_data['input_sequence'].shape) > 1:
                            features = sequence_data['input_sequence'][0]
                        else:
                            features = sequence_data['input_sequence']
                        
                        for i, value in enumerate(features):
                            writer.writerow([f"Feature {i}", value])
                
                messagebox.showinfo("Success", f"Sequence alignment exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def refresh(self):
        """Refresh the view with current data"""
        self._update_sequence_display()
