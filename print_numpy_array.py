#!/usr/bin/env python3
"""
Simple script to display numpy arrays, JSON files, and CSV files in a table format
"""

import numpy as np
import json
import sys
import tkinter as tk
from tkinter import ttk

# Try to import pandas, but handle gracefully if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. CSV files cannot be loaded.")
    print("Install pandas with: pip install pandas")

def load_data(filepath):
    """Load data from either numpy array, JSON file, or CSV file"""
    try:
        if filepath.endswith('.npy'):
            # Load numpy array
            data = np.load(filepath)
            data_type = "numpy_array"
        elif filepath.endswith('.json'):
            # Load JSON file
            with open(filepath, 'r') as f:
                data = json.load(f)
            data_type = "json"
        elif filepath.endswith('.csv'):
            # Load CSV file
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required to load CSV files")
            
            # Try different encodings for better compatibility
            try:
                data = pd.read_csv(filepath)
            except UnicodeDecodeError:
                # Try with different encoding if default fails
                data = pd.read_csv(filepath, encoding='latin-1')
            
            data_type = "csv"
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        return data, data_type
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

def display_data(filepath):
    """Load and display data (numpy array or JSON) in a tkinter window"""
    data, data_type = load_data(filepath)
    if data is None:
        return
    
    print(f"Data type: {data_type}")
    if data_type == "numpy_array":
        print(f"Array shape: {data.shape}")
        print(f"Array dtype: {data.dtype}")
        # Add summary for numpy arrays
        total_elements = data.size
        non_zero_count = np.count_nonzero(data)
        zero_count = total_elements - non_zero_count
        print(f"Total elements: {total_elements:,}")
        print(f"Non-zero elements: {non_zero_count:,} ({100 * non_zero_count / total_elements:.1f}%)")
        print(f"Zero elements: {zero_count:,} ({100 * zero_count / total_elements:.1f}%)")
        if non_zero_count > 0:
            print(f"Value range: {data.min():.2f} to {data.max():.2f}")
    elif data_type == "json":
        print(f"JSON structure: {type(data).__name__}")
        if isinstance(data, list):
            print(f"List length: {len(data)}")
        elif isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
    elif data_type == "csv":
        print(f"CSV DataFrame shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data types: {dict(data.dtypes)}")
        print(f"First few rows:")
        print(data.head())
    
    # Create window
    root = tk.Tk()
    root.title(f"Data Viewer: {data_type} - {filepath}")
    root.geometry("1400x900")
    
        # Add navigation when we have 3D+ numpy arrays (or 3D JSON)
    is_numpy_3d_plus = (data_type == "numpy_array" and len(getattr(data, "shape", ())) >= 3)
    if is_numpy_3d_plus or \
        (data_type == "json" and isinstance(data, list) and data and 
         isinstance(data[0], list) and data[0] and isinstance(data[0][0], list)):
        
        nav_frame = ttk.Frame(root)
        nav_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        # Navigation controls in a compact row
        ttk.Label(nav_frame, text="Slice:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        # Previous button
        prev_btn = ttk.Button(nav_frame, text="◀", width=3, command=lambda: change_slice(-1))
        prev_btn.pack(side=tk.LEFT, padx=(0, 2))
        
        # Current slice display
        slice_var = tk.IntVar(value=0)
        slice_label = ttk.Label(nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
        slice_label.pack(side=tk.LEFT, padx=2)
        
        # Next button
        next_btn = ttk.Button(nav_frame, text="▶", width=3, command=lambda: change_slice(1))
        next_btn.pack(side=tk.LEFT, padx=(2, 10))
        
        # Slice info
        if data_type == "numpy_array":
            total_slices = data.shape[0]
        else:  # JSON 3D array
            total_slices = len(data)
        ttk.Label(nav_frame, text=f"of {total_slices} total slices", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Add slice statistics display
        stats_label = ttk.Label(nav_frame, text="", font=("Arial", 9), foreground="blue")
        stats_label.pack(side=tk.LEFT, padx=(20, 0))
        nav_frame.stats_label = stats_label  # Store reference for updates
        
        # Jump to specific slice
        ttk.Label(nav_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        jump_var = tk.StringVar(value="0")
        jump_entry = ttk.Entry(nav_frame, textvariable=jump_var, width=6)
        jump_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        jump_btn = ttk.Button(nav_frame, text="Go", command=lambda: jump_to_slice())
        jump_btn.pack(side=tk.LEFT)
        
        # Bind Enter key to jump
        jump_entry.bind('<Return>', lambda e: jump_to_slice())

        # If numpy 4D (e.g., actions: S x 10 x 101 x 8), add a TIME selector for axis-1
        time_var = tk.IntVar(value=0)
        if data_type == "numpy_array" and len(data.shape) == 4:
            ttk.Label(nav_frame, text="   Time:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 5))
            time_prev = ttk.Button(nav_frame, text="◀", width=3, command=lambda: change_time(-1))
            time_prev.pack(side=tk.LEFT, padx=(0, 2))
            time_label = ttk.Label(nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
            time_label.pack(side=tk.LEFT, padx=2)
            time_next = ttk.Button(nav_frame, text="▶", width=3, command=lambda: change_time(1))
            time_next.pack(side=tk.LEFT, padx=(2, 10))
            ttk.Label(nav_frame, text=f"of {data.shape[1]} timesteps", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Create main frame for the table
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Add legend for color coding
    legend_frame = ttk.Frame(main_frame)
    legend_frame.pack(fill=tk.X, pady=(0, 5))
    
    # Create legend labels
    legend_label = ttk.Label(legend_frame, text="Legend:", font=("Arial", 9, "bold"))
    legend_label.pack(side=tk.LEFT, padx=(0, 10))
    
    # Non-zero value indicator
    nonzero_indicator = tk.Label(legend_frame, text="  Non-zero values  ", 
                                background="#e6f3ff", relief=tk.RAISED, borderwidth=1)
    nonzero_indicator.pack(side=tk.LEFT, padx=(0, 20))
    
    # Zero value indicator
    zero_indicator = tk.Label(legend_frame, text="  0  ", 
                             background="white", relief=tk.RAISED, borderwidth=1)
    zero_indicator.pack(side=tk.LEFT, padx=(0, 20))
    
    # Info about clean formatting and row coloring
    info_label = ttk.Label(legend_frame, text="Values without trailing zeros | Rows with non-zero values highlighted", 
                           font=("Arial", 8), foreground="gray")
    info_label.pack(side=tk.LEFT)
    
    # Container for tree + scrollbars
    tree_frame = ttk.Frame(main_frame)
    tree_frame.pack(fill=tk.BOTH, expand=True)
    
    # CREATE TREE *INSIDE* tree_frame (not main_frame)
    tree = ttk.Treeview(tree_frame)
    
    # Configure columns based on data type and structure
    if data_type == "numpy_array":
        configure_numpy_tree(tree, data)
    elif data_type == "json":
        configure_json_tree(tree, data)
    elif data_type == "csv":
        configure_csv_tree(tree, data)
    
    # Scrollbars in the same parent
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Use grid so nothing steals space
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    
    # Make the tree cell expandable
    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    
    # 3D navigation functions (for numpy arrays and 3D JSON)
    if is_numpy_3d_plus or \
        (data_type == "json" and isinstance(data, list) and data and 
         isinstance(data[0], list) and data[0] and isinstance(data[0][0], list)):
        
        def change_slice(delta):
            current = slice_var.get()
            if data_type == "numpy_array":
                total_slices = data.shape[0]
            else:  # JSON 3D array
                total_slices = len(data)
            
            new_slice = current + delta
            if 0 <= new_slice < total_slices:
                slice_var.set(new_slice)
                update_3d_slice()
        
        def jump_to_slice():
            try:
                new_slice = int(jump_var.get())
                if data_type == "numpy_array":
                    total_slices = data.shape[0]
                else:  # JSON 3D array
                    total_slices = len(data)
                
                if 0 <= new_slice < total_slices:
                    slice_var.set(new_slice)
                    update_3d_slice()
                else:
                    jump_var.set(str(slice_var.get()))  # Reset to current
            except ValueError:
                jump_var.set(str(slice_var.get()))  # Reset to current

        def change_time(delta):
            # Only meaningful for 4D numpy arrays
            if not (data_type == "numpy_array" and len(data.shape) == 4):
                return
            current = time_var.get()
            new_t = current + delta
            if 0 <= new_t < data.shape[1]:
                time_var.set(new_t)
                update_3d_slice()
        
        def update_3d_slice():
            slice_idx = slice_var.get()
            if data_type == "numpy_array":
                if 0 <= slice_idx < data.shape[0]:
                    # Update label
                    slice_label.config(text=str(slice_idx))
                    jump_var.set(str(slice_idx))
                    
                    # Update slice statistics
                    if hasattr(nav_frame, 'stats_label'):
                        stats_text = get_slice_statistics(data, slice_idx)
                        nav_frame.stats_label.config(text=stats_text)
                    
                    # Clear existing items
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # Render depending on dimensionality
                    ndim = len(data.shape)
                    if ndim == 3:
                        # (A, R, C): rows R, cols C
                        rows = data.shape[1]
                        cols = data.shape[2]
                        _ensure_numpy_columns(tree, cols)
                        for r in range(rows):
                            row_vals = np.asarray(data[slice_idx, r, :]).ravel()
                            values = [format_value_clean(float(v)) for v in row_vals]
                            
                            # Check if this row has any non-zero values and prepare text accordingly
                            has_nonzero = np.any(np.abs(row_vals) > 1e-10)
                            if has_nonzero:
                                # Create a tag for this row
                                row_tag = f"nonzero_row_{r}"
                                tree.tag_configure(row_tag, background="#e6f3ff", foreground="black")
                                
                                # Add info about which columns have non-zero values
                                nonzero_cols = [f"col_{i}" for i, v in enumerate(row_vals) if abs(v) > 1e-10]
                                if len(nonzero_cols) <= 3:  # Show specific columns if few
                                    item_text = f"Row {r} (non-zero: {', '.join(nonzero_cols)})"
                                else:
                                    item_text = f"Row {r} ({len(nonzero_cols)} non-zero cols)"
                                
                                item = tree.insert("", tk.END, text=item_text, values=values, tags=(row_tag,))
                            else:
                                item = tree.insert("", tk.END, text=f"Row {r}", values=values)
                    elif ndim == 4:
                        # (S, T, E, F): events E as rows, features F as cols; choose T with time_var
                        t_idx = int(time_var.get())
                        t_idx = max(0, min(t_idx, data.shape[1] - 1))
                        # Update time label if present
                        try:
                            time_label.config(text=str(t_idx))
                        except Exception:
                            pass
                        
                        # Update slice statistics for 4D data
                        if hasattr(nav_frame, 'stats_label'):
                            stats_text = get_slice_statistics(data, slice_idx, t_idx)
                            nav_frame.stats_label.config(text=stats_text)
                        
                        events = data.shape[2]
                        feats = data.shape[3]
                        _ensure_numpy_columns(tree, feats)
                        for e in range(events):
                            row_vals = np.asarray(data[slice_idx, t_idx, e, :]).ravel()
                            values = [format_value_clean(float(v)) for v in row_vals]
                            
                            # Check if this row has any non-zero values and prepare text accordingly
                            has_nonzero = np.any(np.abs(row_vals) > 1e-10)
                            if has_nonzero:
                                # Create a tag for this row
                                row_tag = f"nonzero_row_{e}"
                                tree.tag_configure(row_tag, background="#e6f3ff", foreground="black")
                                
                                # Add info about which columns have non-zero values
                                nonzero_cols = [f"col_{i}" for i, v in enumerate(row_vals) if abs(v) > 1e-10]
                                if len(nonzero_cols) <= 3:  # Show specific columns if few
                                    item_text = f"Evt {e} (non-zero: {', '.join(nonzero_cols)})"
                                else:
                                    item_text = f"Evt {e} ({len(nonzero_cols)} non-zero cols)"
                                
                                item = tree.insert("", tk.END, text=item_text, values=values, tags=(row_tag,))
                            else:
                                item = tree.insert("", tk.END, text=f"Evt {e}", values=values)
            else:  # JSON 3D array
                if 0 <= slice_idx < len(data):
                    # Update label
                    slice_label.config(text=str(slice_idx))
                    jump_var.set(str(slice_idx))
                    
                    # Update JSON slice
                    update_3d_json_slice(tree, slice_idx)
        
        # Initial slice
        update_3d_slice()
    
    root.mainloop()

def configure_numpy_tree(tree, data):
    """Configure tree for numpy array data"""
    if len(data.shape) == 1:
        tree["columns"] = ("index", "value")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("index", anchor=tk.CENTER, width=100)
        tree.column("value", anchor=tk.CENTER, width=150)
        tree.heading("#0", text="")
        tree.heading("index", text="Index")
        tree.heading("value", text="Value")
        
        for i, val in enumerate(data):
            formatted_val = format_value_clean(val)
            item = tree.insert("", tk.END, values=(i, formatted_val))
            if abs(val) > 1e-10:  # Non-zero value
                # Create unique tag for this cell
                cell_tag = f"nonzero_cell_{i}"
                tree.tag_configure(cell_tag, background="#e6f3ff", foreground="black")
                tree.item(item, tags=(cell_tag,))
            
    elif len(data.shape) == 2:
        tree["columns"] = tuple([f"col_{i}" for i in range(data.shape[1])])
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Row")
        
        for i in range(data.shape[1]):
            tree.column(f"col_{i}", anchor=tk.CENTER, width=120)  # Consistent width
            tree.heading(f"col_{i}", text=f"Col {i}")
        
        for i in range(data.shape[0]):
            values = [format_value_clean(val) for val in data[i]]
            
            # Check if this row has any non-zero values and prepare text accordingly
            has_nonzero = np.any(np.abs(data[i]) > 1e-10)
            if has_nonzero:
                # Create a tag for this row
                row_tag = f"nonzero_row_{i}"
                tree.tag_configure(row_tag, background="#e6f3ff", foreground="black")
                
                # Add info about which columns have non-zero values
                nonzero_cols = [f"col_{j}" for j, v in enumerate(data[i]) if abs(v) > 1e-10]
                if len(nonzero_cols) <= 3:  # Show specific columns if few
                    item_text = f"Row {i} (non-zero: {', '.join(nonzero_cols)})"
                else:
                    item_text = f"Row {i} ({len(nonzero_cols)} non-zero cols)"
                
                item = tree.insert("", tk.END, text=item_text, values=values, tags=(row_tag,))
            else:
                item = tree.insert("", tk.END, text=f"Row {i}", values=values)
            
    else:
        # For 3D+: columns = last axis size
        last_axis = data.shape[-1]
        tree["columns"] = tuple([f"col_{i}" for i in range(last_axis)])
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Row")
        for i in range(last_axis):
            tree.column(f"col_{i}", anchor=tk.CENTER, width=120)
            tree.heading(f"col_{i}", text=f"Col {i}")

def format_value_clean(val):
    """Format value with trailing zeros removed and special handling for zeros"""
    if abs(val) < 1e-10:  # Essentially zero
        return "0"
    elif val == int(val):  # Integer value
        return str(int(val))
    else:  # Float value - remove trailing zeros
        formatted = f"{val:.6f}".rstrip('0').rstrip('.')
        return formatted

def get_slice_statistics(data, slice_idx, time_idx=None):
    """Get statistics for a specific slice of the data"""
    if len(data.shape) == 3:
        slice_data = data[slice_idx]
        slice_name = f"Sequence {slice_idx}"
    elif len(data.shape) == 4 and time_idx is not None:
        slice_data = data[slice_idx, time_idx]
        slice_name = f"Sequence {slice_idx}, Time {time_idx}"
    else:
        return "No slice data available"
    
    total_elements = slice_data.size
    non_zero_count = np.count_nonzero(slice_data)
    zero_count = total_elements - non_zero_count
    
    if non_zero_count == 0:
        return f"{slice_name}: All zeros ({total_elements:,} elements)"
    
    non_zero_values = slice_data[slice_data != 0]
    value_ranges = [
        (0, 1, "0-1"),
        (1, 10, "1-10"), 
        (10, 100, "10-100"),
        (100, 1000, "100-1000"),
        (1000, float('inf'), "1000+")
    ]
    
    range_counts = []
    for min_val, max_val, label in value_ranges:
        if max_val == float('inf'):
            count = np.sum(slice_data >= min_val)
        else:
            count = np.sum((slice_data >= min_val) & (slice_data < max_val))
        range_counts.append(f"{label}: {count}")
    
    return (f"{slice_name}: {non_zero_count:,}/{total_elements:,} non-zero "
            f"({100 * non_zero_count / total_elements:.1f}%) | "
            f"Range: {slice_data.min():.2f} to {slice_data.max():.2f} | "
            f"Values: {', '.join(range_counts)}")

def _ensure_numpy_columns(tree, ncols: int):
    """Ensure the tree has exactly ncols numerical columns, recreating if needed."""
    existing = list(tree["columns"]) if "columns" in tree.__dict__ else []
    desired = [f"col_{i}" for i in range(ncols)]
    if tuple(existing) != tuple(desired):
        tree["columns"] = tuple(desired)
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Row")
        for i in range(ncols):
            tree.column(f"col_{i}", anchor=tk.CENTER, width=120)
            tree.heading(f"col_{i}", text=f"Col {i}")

def configure_json_tree(tree, data):
    """Configure tree for JSON data"""
    if isinstance(data, list):
        # Check if this is a 3D array structure (list of lists of lists)
        if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            # 3D array structure - add slice navigation
            configure_3d_json_tree(tree, data)
            return
        
        # Handle list of dictionaries
        if data and isinstance(data[0], dict):
            # Get all unique keys from all dictionaries
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            # Sort keys for consistent ordering
            columns = sorted(all_keys)
            tree["columns"] = columns
            
            # Configure columns
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Index")
            
            for col in columns:
                tree.column(col, anchor=tk.CENTER, width=120)
                tree.heading(col, text=col)
            
            # Add data rows
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    values = [str(item.get(col, "")) for col in columns]
                    tree.insert("", tk.END, text=f"Row {i}", values=values)
                else:
                    tree.insert("", tk.END, text=f"Row {i}", values=[str(item)])
        
        else:
            # Simple list
            tree["columns"] = ("index", "value")
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("index", anchor=tk.CENTER, width=100)
            tree.column("value", anchor=tk.CENTER, width=300)
            tree.heading("#0", text="")
            tree.heading("index", text="Index")
            tree.heading("value", text="Value")
            
            for i, val in enumerate(data):
                tree.insert("", tk.END, text=f"Row {i}", values=format_json_value(val))
    
    elif isinstance(data, dict):
        # Handle dictionary
        keys = list(data.keys())
        tree["columns"] = ("key", "value")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("key", anchor=tk.WEST, width=200)
        tree.column("value", anchor=tk.WEST, width=400)
        tree.heading("#0", text="")
        tree.heading("key", text="Key")
        tree.heading("value", text="Value")
        
        for key, value in data.items():
            tree.insert("", tk.END, values=(key, format_json_value(value)))
    
    else:
        # Single value
        tree["columns"] = ("value",)
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("value", anchor=tk.CENTER, width=400)
        tree.heading("#0", text="")
        tree.heading("value", text="Value")
        
        tree.insert("", tk.END, values=(format_json_value(data),))

def configure_3d_json_tree(tree, data):
    """Configure tree for 3D JSON array data (list of lists of lists)"""
    # Configure columns for the 3D structure
    # Each row will represent a feature (1-8), columns will be actions
    max_actions = max(len(seq) if seq else 0 for seq in data)
    
    tree["columns"] = tuple([f"A{i+1}" for i in range(max_actions)])
    tree.column("#0", width=100, stretch=tk.NO)
    tree.heading("#0", text="Feature")
    
    # Set consistent column widths to match NumPy display
    for i in range(max_actions):
        tree.column(f"A{i+1}", anchor=tk.CENTER, width=120)  # Increased width for better alignment
        tree.heading(f"A{i+1}", text=f"A{i+1}")
    
    # Store the 3D data for slice navigation
    tree._3d_data = data
    tree._current_slice = 0
    
    # Load initial data
    update_3d_json_slice(tree, 0)

def format_json_value(value):
    """Format JSON values for better display"""
    if isinstance(value, list):
        if len(value) <= 5:  # Short lists
            return str(value)
        else:  # Long lists - truncate
            return f"[{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] ({len(value)} items)"
    elif isinstance(value, dict):
        return f"{{...}} ({len(value)} keys)"
    else:
        return str(value)

def update_3d_json_slice(tree, slice_idx):
    """Update the 3D JSON tree view for a specific slice"""
    if not hasattr(tree, '_3d_data') or not tree._3d_data:
        return
    
    data = tree._3d_data
    if not (0 <= slice_idx < len(data)):
        return
    
    # Clear existing items
    for item in tree.get_children():
        tree.delete(item)
    
    # Get the current slice (sequence)
    current_sequence = data[slice_idx]
    if not current_sequence:
        return
    
    # Add feature rows (1-8)
    for feature_idx in range(8):  # Features 1-8
        values = []
        for action_idx, action_data in enumerate(current_sequence):
            if isinstance(action_data, list) and feature_idx < len(action_data):
                val = action_data[feature_idx]
                if isinstance(val, (float, int)):
                    # Use clean formatting for numbers
                    values.append(format_value_clean(val))
                else:
                    values.append(str(val))
            else:
                values.append("")
        
        tree.insert("", tk.END, text=f"F{feature_idx+1}", values=values)

def configure_csv_tree(tree, data):
    """Configure tree for CSV data"""
    if isinstance(data, pd.DataFrame):
        # Get column names
        columns = data.columns.tolist()
        tree["columns"] = columns
        
        # Configure columns with better width handling
        tree.column("#0", width=80, stretch=tk.NO)
        tree.heading("#0", text="Row")
        
        # Calculate optimal column widths based on content
        for col in columns:
            # Get max length of column name and sample values
            col_width = len(str(col)) * 10  # Base width on column name
            
            # Sample some values to estimate content width
            sample_values = data[col].head(100).astype(str)
            max_val_len = max(len(str(val)) for val in sample_values) if len(sample_values) > 0 else 10
            
            # Set column width (minimum 80, maximum 200)
            col_width = max(80, min(200, max(col_width, max_val_len * 8)))
            
            tree.column(col, anchor=tk.CENTER, width=col_width)
            tree.heading(col, text=col)
        
        # Store the DataFrame for pagination
        tree._csv_data = data
        tree._current_page = 0
        tree._rows_per_page = 1000  # Show 1000 rows per page
        
        # Add pagination controls if dataset is large
        if len(data) > tree._rows_per_page:
            add_csv_pagination_controls(tree, data)
        
        # Load initial page
        load_csv_page(tree, 0)
    else:
        # Handle non-DataFrame CSV data (e.g., a single row or a single value)
        if isinstance(data, list) and len(data) == 1:
            # Single row (list of values)
            tree["columns"] = ("index", "value")
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("index", anchor=tk.CENTER, width=100)
            tree.column("value", anchor=tk.CENTER, width=300)
            tree.heading("#0", text="")
            tree.heading("index", text="Index")
            tree.heading("value", text="Value")
            for i, val in enumerate(data[0]):
                tree.insert("", tk.END, text=f"Row {i}", values=(i, val))
        elif not isinstance(data, pd.DataFrame): # Handle single value
            tree["columns"] = ("value",)
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("value", anchor=tk.CENTER, width=400)
            tree.heading("#0", text="")
            tree.heading("value", text="Value")
            tree.insert("", tk.END, values=(str(data),))

def add_csv_pagination_controls(tree, data):
    """Add pagination controls for large CSV datasets"""
    # Find the main window (root) from the tree widget
    root = tree.winfo_toplevel()
    
    # Create pagination frame at the bottom
    pagination_frame = ttk.Frame(root)
    pagination_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
    
    # Calculate total pages
    total_pages = (len(data) + tree._rows_per_page - 1) // tree._rows_per_page
    
    # Navigation controls
    ttk.Label(pagination_frame, text="Page:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Previous page button
    prev_btn = ttk.Button(pagination_frame, text="◀", width=3, 
                          command=lambda: change_csv_page(tree, -1))
    prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current page display
    page_var = tk.IntVar(value=1)
    page_label = ttk.Label(pagination_frame, text="1", font=("Arial", 10, "bold"), width=6)
    page_label.pack(side=tk.LEFT, padx=2)
    
    # Next page button
    next_btn = ttk.Button(pagination_frame, text="▶", width=3, 
                          command=lambda: change_csv_page(tree, 1))
    next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Page info
    ttk.Label(pagination_frame, text=f"of {total_pages} total pages", 
              font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Jump to specific page
    ttk.Label(pagination_frame, text="Jump to page:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
    jump_var = tk.StringVar(value="1")
    jump_entry = ttk.Entry(pagination_frame, textvariable=jump_var, width=6)
    jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    jump_btn = ttk.Button(pagination_frame, text="Go", 
                          command=lambda: jump_to_csv_page(tree, jump_var))
    jump_btn.pack(side=tk.LEFT)
    
    # Rows per page selector
    ttk.Label(pagination_frame, text="Rows per page:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
    rows_var = tk.StringVar(value=str(tree._rows_per_page))
    rows_combo = ttk.Combobox(pagination_frame, textvariable=rows_var, 
                              values=["100", "500", "1000", "2000", "5000"], width=8)
    rows_combo.pack(side=tk.LEFT, padx=(0, 5))
    rows_combo.bind('<<ComboboxSelected>>', lambda e: change_rows_per_page(tree, rows_var))
    
    # Store references for navigation functions
    tree._page_var = page_var
    tree._page_label = page_label
    tree._jump_var = jump_var
    tree._total_pages = total_pages
    tree._pagination_frame = pagination_frame
    
    # Bind Enter key to jump
    jump_entry.bind('<Return>', lambda e: jump_to_csv_page(tree, jump_var))

def change_csv_page(tree, delta):
    """Change to the next/previous page"""
    current = tree._current_page
    new_page = current + delta
    
    if 0 <= new_page < tree._total_pages:
        tree._current_page = new_page
        tree._page_var.set(new_page + 1)
        tree._page_label.config(text=str(new_page + 1))
        load_csv_page(tree, new_page)

def jump_to_csv_page(tree, jump_var):
    """Jump to a specific page"""
    try:
        new_page = int(jump_var.get()) - 1  # Convert to 0-based index
        if 0 <= new_page < tree._total_pages:
            tree._current_page = new_page
            tree._page_var.set(new_page + 1)
            tree._page_label.config(text=str(new_page + 1))
            load_csv_page(tree, new_page)
        else:
            jump_var.set(str(tree._page_var.get()))  # Reset to current
    except ValueError:
        jump_var.set(str(tree._page_var.get()))  # Reset to current

def change_rows_per_page(tree, rows_var):
    """Change the number of rows displayed per page"""
    try:
        new_rows = int(rows_var.get())
        tree._rows_per_page = new_rows
        tree._total_pages = (len(tree._csv_data) + new_rows - 1) // new_rows
        tree._current_page = 0  # Reset to first page
        tree._page_var.set(1)
        tree._page_label.config(text="1")
        tree._jump_var.set("1")
        load_csv_page(tree, 0)
    except ValueError:
        rows_var.set(str(tree._rows_per_page))

def load_csv_page(tree, page):
    """Load a specific page of CSV data"""
    if not hasattr(tree, '_csv_data'):
        return
    
    data = tree._csv_data
    start_idx = page * tree._rows_per_page
    end_idx = min(start_idx + tree._rows_per_page, len(data))
    
    # Clear existing items
    for item in tree.get_children():
        tree.delete(item)
    
    # Add page data
    for i in range(start_idx, end_idx):
        row = data.iloc[i]
        values = [str(row[col]) for col in data.columns]
        tree.insert("", tk.END, text=f"Row {i}", values=values)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_numpy_array.py <file_path>")
        print("Supported formats: .npy (NumPy arrays), .json (JSON files), .csv (CSV files)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    display_data(filepath)
