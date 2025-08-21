"""
Action Tensors View

View for displaying action tensor data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

class ActionTensorsView(ttk.Frame):
    """Action Tensors view with full interactive functionality"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.current_action_gamestate = 0
        self.action_tensor_tree = None
        self.action_tensor_tooltip = None
        self.action_tensor_tooltip_text = ""
        
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the widgets for the action tensors view"""
        # Info frame
        self.info_frame = ttk.Frame(self)
        self.action_info_label = ttk.Label(self.info_frame, text="", font=("Arial", 10))
        
        # Controls frame
        self.controls_frame = ttk.Frame(self)
        
        # Gamestate selector
        ttk.Label(self.controls_frame, text="Gamestate:").pack(side=tk.LEFT)
        self.action_gamestate_var = tk.StringVar()
        self.action_gamestate_spinbox = ttk.Spinbox(
            self.controls_frame,
            from_=0,
            to=1000,  # Will be updated dynamically
            textvariable=self.action_gamestate_var,
            width=10,
            command=self._on_action_gamestate_change
        )
        
        # Navigation buttons
        self.prev_button = ttk.Button(self.controls_frame, text="◀ Previous", command=self._previous_action_gamestate)
        self.next_button = ttk.Button(self.controls_frame, text="Next ▶", command=self._next_action_gamestate)
        
        # Action type filter
        ttk.Label(self.controls_frame, text="Filter by Action Type:").pack(side=tk.LEFT)
        self.action_type_filter = tk.StringVar(value="All")
        self.action_filter_combo = ttk.Combobox(
            self.controls_frame,
            textvariable=self.action_type_filter,
            values=["All", "mouse_movements", "clicks", "key_presses", "key_releases", "scrolls"],
            state="readonly",
            width=15
        )
        self.action_filter_combo.bind('<<ComboboxSelected>>', self._on_action_type_filter_changed)
        
        # Export buttons
        self.copy_button = ttk.Button(self.controls_frame, text="📋 Copy to Clipboard", command=self._copy_action_data_to_clipboard)
        self.export_json_button = ttk.Button(self.controls_frame, text="💾 Export JSON", command=self._export_action_data_json)
        
        # Help text
        self.help_label = ttk.Label(self.controls_frame, text="(Normalizes timestamps and coordinates using same scaling as main features)", 
                                   font=("Arial", 8), foreground="gray")
        
        # Main display frame
        self.json_frame = ttk.LabelFrame(self, text="Action Tensor Table", padding="5")
        
        # Description
        self.flow_description = ttk.Label(
            self.json_frame,
            text="Action tensor table: Action Count (first element only) + 8 features per action across all timesteps",
            font=("Consolas", 9),
            foreground="blue"
        )
        
        # Create action tensor table
        self._create_action_tensor_table()
        
    def _setup_layout(self):
        """Set up the layout of all widgets"""
        # Info frame
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.action_info_label.pack(side=tk.LEFT)
        
        # Controls frame
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Pack controls in order
        ttk.Label(self.controls_frame, text="Gamestate:").pack(side=tk.LEFT)
        self.action_gamestate_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        self.next_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(self.controls_frame, text="Filter by Action Type:").pack(side=tk.LEFT)
        self.action_filter_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        self.copy_button.pack(side=tk.LEFT, padx=(0, 10))
        self.export_json_button.pack(side=tk.LEFT, padx=(0, 10))
        self.help_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Main display
        self.json_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.flow_description.pack(pady=(0, 5))
        
        # Update gamestate range
        self._update_gamestate_range()
        
    def _create_action_tensor_table(self):
        """Create the action tensor table with Treeview"""
        # Info frame for table
        self.table_info_frame = ttk.Frame(self.json_frame)
        self.table_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.action_tensor_info_label = ttk.Label(self.table_info_frame, text="", font=("Arial", 10))
        self.action_tensor_info_label.pack(side=tk.LEFT)
        
        # Export frame
        export_frame = ttk.Frame(self.json_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Button(export_frame, text="📋 Copy Table to Clipboard", command=self._copy_action_tensor_table_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="💾 Export to CSV", command=self._export_action_tensor_table_to_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        # Normalization toggle
        self.show_action_normalized = tk.BooleanVar(value=False)
        self.action_normalization_toggle = ttk.Checkbutton(
            export_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_action_normalized,
            command=self._update_action_tensor_table
        )
        self.action_normalization_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        # Data source info
        self.action_data_info_label = ttk.Label(export_frame, text="📊 Showing Trimmed Action Data")
        self.action_data_info_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Table frame
        table_frame = ttk.Frame(self.json_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview with scrollbars
        self.action_tensor_tree = ttk.Treeview(table_frame, show="headings", height=20)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.action_tensor_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.action_tensor_tree.xview)
        self.action_tensor_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.action_tensor_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Initial columns
        self.action_tensor_tree["columns"] = ["Feature", "Index"]
        
        # Set initial column headings
        for col in self.action_tensor_tree["columns"]:
            self.action_tensor_tree.heading(col, text=col)
            if col == "Feature":
                self.action_tensor_tree.column(col, width=200, minwidth=150)
            elif col == "Index":
                self.action_tensor_tree.column(col, width=50, minwidth=50)
        
        # Bind tooltip events
        self.action_tensor_tree.bind('<Motion>', self._on_action_tensor_table_motion)
        self.action_tensor_tree.bind('<Leave>', self._on_action_tensor_table_leave)
        
        # Initial update
        self._update_action_tensor_table()
        
    def _update_gamestate_range(self):
        """Update the gamestate spinbox range based on available data"""
        loaded_data = self.controller.get_loaded_data()
        if loaded_data and loaded_data.raw_action_data:
            max_gamestate = len(loaded_data.raw_action_data) - 1
            self.action_gamestate_spinbox.configure(to=max_gamestate)
            if self.current_action_gamestate > max_gamestate:
                self.current_action_gamestate = max_gamestate
                self.action_gamestate_var.set(str(self.current_action_gamestate))
        else:
            self.action_gamestate_spinbox.configure(to=0)
            self.current_action_gamestate = 0
            self.action_gamestate_var.set("0")
    
    def _on_action_gamestate_change(self):
        """Handle gamestate spinbox change"""
        try:
            new_gamestate = int(self.action_gamestate_var.get())
            if new_gamestate != self.current_action_gamestate:
                self.current_action_gamestate = new_gamestate
                self._update_action_tensor_table()
        except ValueError:
            pass
    
    def _previous_action_gamestate(self):
        """Navigate to previous gamestate"""
        if self.current_action_gamestate > 0:
            self.current_action_gamestate -= 1
            self.action_gamestate_var.set(str(self.current_action_gamestate))
            self._update_action_tensor_table()
    
    def _next_action_gamestate(self):
        """Navigate to next gamestate"""
        loaded_data = self.controller.get_loaded_data()
        if loaded_data and loaded_data.raw_action_data:
            max_gamestate = len(loaded_data.raw_action_data) - 1
            if self.current_action_gamestate < max_gamestate:
                self.current_action_gamestate += 1
                self.action_gamestate_var.set(str(self.current_action_gamestate))
                self._update_action_tensor_table()
    
    def _on_action_type_filter_changed(self, event=None):
        """Handle action type filter change"""
        self._update_action_tensor_table()
    
    def _convert_action_data_to_tensors(self, action_data):
        """Convert trimmed action data to tensor format for display"""
        if not action_data:
            return []
        
        tensors = []
        for gamestate_actions in action_data:
            # Count total actions for this gamestate
            total_actions = (len(gamestate_actions.get('mouse_movements', [])) + 
                            len(gamestate_actions.get('clicks', [])) + 
                            len(gamestate_actions.get('key_presses', [])) + 
                            len(gamestate_actions.get('key_releases', [])) + 
                            len(gamestate_actions.get('scrolls', [])))
            
            # Start building the action tensor: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
            action_tensor = [total_actions]
            
            # Collect all actions with their metadata
            all_actions = []
            
            # Process mouse movements
            for move in gamestate_actions.get('mouse_movements', []):
                all_actions.append({
                    'timestamp': move.get('timestamp', 0),
                    'type': 0,  # 0 = move
                    'x': move.get('x', 0),
                    'y': move.get('y', 0),
                    'button': 0,  # No button for moves
                    'key': 0,     # No key for moves
                    'scroll_dx': 0,  # No scroll for moves
                    'scroll_dy': 0
                })
            
            # Process clicks
            for click in gamestate_actions.get('clicks', []):
                all_actions.append({
                    'timestamp': click.get('timestamp', 0),
                    'type': 1,  # 1 = click
                    'x': click.get('x', 0),
                    'y': click.get('y', 0),
                    'button': click.get('button', 0),
                    'key': 0,     # No key for clicks
                    'scroll_dx': 0,  # No scroll for clicks
                    'scroll_dy': 0
                })
            
            # Process key presses
            for key_press in gamestate_actions.get('key_presses', []):
                all_actions.append({
                    'timestamp': key_press.get('timestamp', 0),
                    'type': 2,  # 2 = key press
                    'x': 0,     # No coordinates for key presses
                    'y': 0,
                    'button': 0,     # No button for key presses
                    'key': key_press.get('key', 0),
                    'scroll_dx': 0,  # No scroll for key presses
                    'scroll_dy': 0
                })
            
            # Process key releases
            for key_release in gamestate_actions.get('key_releases', []):
                all_actions.append({
                    'timestamp': key_release.get('timestamp', 0),
                    'type': 3,  # 3 = key release
                    'x': 0,     # No coordinates for key releases
                    'y': 0,
                    'button': 0,     # No button for key releases
                    'key': key_release.get('key', 0),
                    'scroll_dx': 0,  # No scroll for key releases
                    'scroll_dy': 0
                })
            
            # Process scrolls
            for scroll in gamestate_actions.get('scrolls', []):
                all_actions.append({
                    'timestamp': scroll.get('timestamp', 0),
                    'type': 4,  # 4 = scroll
                    'x': 0,     # No coordinates for scrolls
                    'y': 0,
                    'button': 0,     # No button for scrolls
                    'key': 0,     # No key for scrolls
                    'scroll_dx': scroll.get('dx', 0),
                    'scroll_dy': scroll.get('dy', 0)
                })
            
            # Sort actions by timestamp
            all_actions.sort(key=lambda x: x['timestamp'])
            
            # Flatten all actions into the tensor
            for action in all_actions:
                action_tensor.extend([
                    action['timestamp'],
                    action['type'],
                    action['x'],
                    action['y'],
                    action['button'],
                    action['key'],
                    action['scroll_dx'],
                    action['scroll_dy']
                ])
            
            tensors.append(action_tensor)
        
        return tensors
    
    def _update_action_tensor_table(self):
        """Update the action tensor table with current gamestate data"""
        try:
            # Clear existing items
            for item in self.action_tensor_tree.get_children():
                self.action_tensor_tree.delete(item)
            
            # Get current gamestate data
            gamestate_idx = self.current_action_gamestate
            
            # Load action tensors (trimmed). When normalized toggle is ON, use prebuilt normalized tensors
            action_tensors = None
            data_source = ""
            normalization_status = ""

            if self.show_action_normalized.get():
                # Normalized (trimmed) action tensors
                data_root = self.controller.get_data_root()
                normalized_path = f"{data_root}/03_normalized_data/normalized_action_training_format.json"
                try:
                    with open(normalized_path, 'r') as f:
                        action_tensors = json.load(f)
                    data_source = "normalized_action_training_format.json"
                    normalization_status = "Normalized (timestamps ÷180; coordinates preserved)"
                    self.action_data_info_label.config(text="📊 Showing Trimmed Normalized Action Data")
                except FileNotFoundError:
                    action_tensors = None
            else:
                # Unnormalized trimmed action data → convert to tensor format for display
                data_root = self.controller.get_data_root()
                trimmed_path = f"{data_root}/02_trimmed_data/trimmed_raw_action_data.json"
                try:
                    with open(trimmed_path, 'r') as f:
                        trimmed_data = json.load(f)
                    action_tensors = self._convert_action_data_to_tensors(trimmed_data)
                    data_source = "trimmed_raw_action_data.json (converted to tensors)"
                    normalization_status = "Trimmed (timestamps in ms; coordinates as-is)"
                    self.action_data_info_label.config(text="📊 Showing Trimmed Action Data")
                except FileNotFoundError:
                    action_tensors = None

            if action_tensors is None:
                # No action tensors found - show empty state
                self.action_tensor_tree.delete(*self.action_tensor_tree.get_children())
                self.action_tensor_info_label.config(text="No action tensor files found. Run tools/build_offline_training_data.py to generate them.")
                return
            
            if gamestate_idx >= len(action_tensors):
                self.action_tensor_info_label.config(text="Gamestate index out of range")
                return
            
            # Get the action tensor for current gamestate
            action_tensor = action_tensors[gamestate_idx]
            action_count = int(action_tensor[0]) if action_tensor else 0
            
            # Update info label
            self.action_tensor_info_label.config(
                text=f"Gamestate {gamestate_idx} | {action_count} actions | {data_source} | {normalization_status} | Action Count: First element only"
            )
            
            # Dynamically configure columns for all timesteps
            columns = ["Feature", "Index"]
            for timestep in range(action_count):
                columns.append(f"Timestep {timestep}")
            
            # Update treeview columns
            self.action_tensor_tree["columns"] = columns
            
            # Configure column headings and widths
            for col in columns:
                self.action_tensor_tree.heading(col, text=col)
                if col == "Feature":
                    self.action_tensor_tree.column(col, width=200, minwidth=150)
                elif col == "Index":
                    self.action_tensor_tree.column(col, width=50, minwidth=50)
                else:
                    self.action_tensor_tree.column(col, width=100, minwidth=80)
            
            # Create feature rows for the table
            # Each action has 8 features: timestamp, type, x, y, button, key, scroll_dx, scroll_dy
            feature_names = ["Action Count", "Timestamp", "Action Type", "Mouse X", "Mouse Y", "Button", "Key", "Scroll DX", "Scroll DY"]
            
            # For each feature, create a row showing values across all timesteps
            for feature_idx, feature_name in enumerate(feature_names):
                if feature_idx == 0:  # Action Count
                    # Action count is a single value at the beginning of the tensor
                    # Show it only in the first column, with "N/A" for other timesteps
                    values = [action_count] + ["N/A"] * (action_count - 1)
                else:
                    # For other features, we need to extract values from the flattened tensor
                    values = []
                    for timestep in range(action_count):
                        # Calculate position in flattened tensor: 1 + timestep * 8 + (feature_idx - 1)
                        tensor_idx = 1 + timestep * 8 + (feature_idx - 1)
                        if tensor_idx < len(action_tensor):
                            value = action_tensor[tensor_idx]
                            values.append(value)
                        else:
                            values.append("N/A")
                
                # Insert row into table
                row_values = [feature_name, feature_idx] + values
                self.action_tensor_tree.insert("", "end", values=row_values)
                
        except FileNotFoundError as e:
            self.action_tensor_tree.delete(*self.action_tensor_tree.get_children())
            self.action_tensor_info_label.config(text="No action tensor files found. Run tools/build_offline_training_data.py to generate them.")
        except Exception as e:
            self.action_tensor_tree.delete(*self.action_tensor_tree.get_children())
            self.action_tensor_info_label.config(text=f"Error loading data: {e}")
    
    def _copy_action_data_to_clipboard(self):
        """Copy action data to clipboard"""
        # This would copy the current gamestate's action data
        messagebox.showinfo("Info", "Copy action data functionality would be implemented here")
    
    def _export_action_data_json(self):
        """Export action data to JSON"""
        # This would export the current gamestate's action data
        messagebox.showinfo("Info", "Export action data functionality would be implemented here")
    
    def _copy_action_tensor_table_to_clipboard(self):
        """Copy action tensor table data to clipboard"""
        try:
            # Get all items from the table
            items = self.action_tensor_tree.get_children()
            if not items:
                messagebox.showwarning("No Data", "No data to copy!")
                return
            
            # Build CSV-like string with dynamic headers
            columns = self.action_tensor_tree["columns"]
            clipboard_text = ",".join(columns) + "\n"
            
            for item in items:
                values = self.action_tensor_tree.item(item)['values']
                clipboard_text += ",".join(str(v) for v in values) + "\n"
            
            # Copy to clipboard (requires pyperclip)
            try:
                import pyperclip
                pyperclip.copy(clipboard_text)
                messagebox.showinfo("Success", "Action tensor table copied to clipboard!")
            except ImportError:
                messagebox.showerror("Error", "pyperclip not available. Cannot copy to clipboard.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy data: {e}")
    
    def _export_action_tensor_table_to_csv(self):
        """Export action tensor table to CSV file"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Get all items from the table
                items = self.action_tensor_tree.get_children()
                if not items:
                    messagebox.showwarning("No Data", "No data to export!")
                    return
                
                # Write CSV file
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write headers
                    columns = self.action_tensor_tree["columns"]
                    writer.writerow(columns)
                    
                    # Write data rows
                    for item in items:
                        values = self.action_tensor_tree.item(item)['values']
                        writer.writerow(values)
                
                messagebox.showinfo("Success", f"Action tensor table exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def _on_action_tensor_table_motion(self, event):
        """Handle mouse motion over action tensor table for tooltips"""
        # Get the item under cursor
        item = self.action_tensor_tree.identify_row(event.y)
        if item:
            # Get column under cursor
            column = self.action_tensor_tree.identify_column(event.x)
            if column:
                # Get item values
                values = self.action_tensor_tree.item(item)['values']
                if values:
                    col_idx = int(column[1]) - 1  # Convert #1, #2, etc. to 0, 1, etc.
                    if col_idx < len(values):
                        value = values[col_idx]
                        tooltip_text = f"Column {column}: {value}"
                        
                        if tooltip_text != self.action_tensor_tooltip_text:
                            self.action_tensor_tooltip_text = tooltip_text
                            self._show_action_tensor_tooltip(event.x_root, event.y_root, tooltip_text)
    
    def _on_action_tensor_table_leave(self, event):
        """Handle mouse leave from action tensor table"""
        self._hide_action_tensor_tooltip()
    
    def _show_action_tensor_tooltip(self, x, y, text):
        """Show tooltip for action tensor table"""
        self._hide_action_tensor_tooltip()
        
        self.action_tensor_tooltip = tk.Toplevel()
        self.action_tensor_tooltip.wm_overrideredirect(True)
        self.action_tensor_tooltip.wm_geometry(f"+{x+10}+{y+10}")
        
        label = ttk.Label(self.action_tensor_tooltip, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()
    
    def _hide_action_tensor_tooltip(self):
        """Hide action tensor tooltip"""
        if self.action_tensor_tooltip:
            self.action_tensor_tooltip.destroy()
            self.action_tensor_tooltip = None
    
    def refresh(self):
        """Refresh the view with current data"""
        self._update_gamestate_range()
        self._update_action_tensor_table()
