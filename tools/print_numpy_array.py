#!/usr/bin/env python3
"""
Simple script to display numpy arrays, JSON files, and CSV files in a table format
"""

import numpy as np
import json
import sys
import tkinter as tk
from tkinter import ttk
import os

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

def display_data(filepaths):
    """Load and display multiple data files in a unified tabbed interface"""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    # Load all data files
    data_files = []
    for filepath in filepaths:
        data, data_type = load_data(filepath)
        if data is not None:
            data_files.append({
                'filepath': filepath,
                'data': data,
                'data_type': data_type,
                'filename': os.path.basename(filepath)
            })
            print(f"Loaded: {os.path.basename(filepath)} - {data_type}")
            if data_type == "numpy_array":
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
            elif data_type == "json":
                print(f"  Structure: {type(data).__name__}")
                if isinstance(data, list):
                    print(f"  Length: {len(data)}")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
            elif data_type == "csv":
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
        else:
            print(f"Failed to load: {filepath}")
    
    if not data_files:
        print("No data files could be loaded!")
        return
    
    # Create unified window
    root = tk.Tk()
    root.title(f"Unified Data Viewer - {len(data_files)} files")
    root.geometry("1600x1000")
    
    # Create global navigation frame at the top
    nav_frame = ttk.Frame(root)
    nav_frame.pack(fill=tk.X, padx=10, pady=5)
    nav_frame.configure(relief="solid", borderwidth=2)
    
    # Create timestamp display at the very top
    timestamp_frame = ttk.Frame(nav_frame)
    timestamp_frame.pack(fill=tk.X, padx=5, pady=2)
    
    timestamp_label = ttk.Label(timestamp_frame, text="Current Batch Timestamp: ", font=("Arial", 10, "bold"))
    timestamp_label.pack(side=tk.LEFT)
    
    timestamp_value = ttk.Label(timestamp_frame, text="Loading...", font=("Arial", 10), foreground="blue")
    timestamp_value.pack(side=tk.LEFT, padx=(5, 0))
    
    # Store timestamp display in nav_frame for access by update functions
    nav_frame.timestamp_value = timestamp_value
    
    # Add color legend for action types
    legend_frame = ttk.Frame(nav_frame)
    legend_frame.pack(fill=tk.X, padx=5, pady=2)
    
    legend_label = ttk.Label(legend_frame, text="Action Types: ", font=("Arial", 9, "bold"))
    legend_label.pack(side=tk.LEFT)
    
    # Create colored legend items
    legend_items = [
        ("Move (count>0)", "lightblue"),
        ("Click (count>0)", "lightgreen"), 
        ("Key Press/Release (count>0)", "lightyellow"),
        ("Scroll (count>0)", "lightcoral"),
        ("No Action (count=0)", "white")
    ]
    
    for text, color in legend_items:
        legend_item_frame = ttk.Frame(legend_frame)
        legend_item_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        # Color indicator
        color_indicator = tk.Label(legend_item_frame, text="  ", background=color, relief="solid", borderwidth=1)
        color_indicator.pack(side=tk.LEFT)
        
        # Text label
        text_label = ttk.Label(legend_item_frame, text=text, font=("Arial", 8))
        text_label.pack(side=tk.LEFT, padx=(2, 0))
    
    # Determine what type of navigation to show based on the data
    has_4d = any(data_file['data_type'] == "numpy_array" and len(data_file['data'].shape) == 4 for data_file in data_files)
    has_3d = any(data_file['data_type'] == "numpy_array" and len(data_file['data'].shape) == 3 for data_file in data_files)
    
    if has_4d:
        # Find the first 4D array to configure navigation
        for data_file in data_files:
            if data_file['data_type'] == "numpy_array" and len(data_file['data'].shape) == 4:
                configure_4d_navigation(nav_frame, data_file['data'], data_files)
                break
    elif has_3d:
        # Find the first 3D array to configure navigation
        for data_file in data_files:
            if data_file['data_type'] == "numpy_array" and len(data_file['data'].shape) == 3:
                configure_3d_navigation(nav_frame, data_file['data'], data_file['data_type'], data_files)
                break
    else:
        # Basic navigation for 1D/2D arrays
        for data_file in data_files:
            if data_file['data_type'] == "numpy_array":
                configure_basic_navigation(nav_frame, data_file['data'], data_file['data_type'])
                break
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Store notebook reference in nav_frame for update functions
    nav_frame.notebook = notebook
    
    # Create tabs for each data file
    tabs = {}
    
    for i, data_file in enumerate(data_files):
        # Create tab frame
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=data_file['filename'])
        tabs[data_file['filename']] = tab_frame
        
        # Create content for this tab
        create_data_tab(tab_frame, data_file)
    
    # Force layout update after all tabs are created
    root.update_idletasks()
    print("All tabs created and navigation configured")
    
    root.mainloop()

def create_data_tab(parent, data_file):
    """Create content for a single data tab"""
    data = data_file['data']
    data_type = data_file['data_type']
    
    # Create tree view
    tree_frame = ttk.Frame(parent)
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    tree = ttk.Treeview(tree_frame)
    parent._tree = tree
    parent._data = data
    parent._data_type = data_type
    
    # Configure columns based on data type and structure
    if data_type == "numpy_array":
        configure_numpy_tree(tree, data)
    elif data_type == "json":
        configure_json_tree(tree, data)
    elif data_type == "csv":
        configure_csv_tree(tree, data)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    
    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    
    # Load initial data for 3D+ arrays
    if (data_type == "numpy_array" and len(data.shape) >= 3) or \
       (data_type == "json" and isinstance(data, list) and data and 
        isinstance(data[0], list) and data[0] and isinstance(data[0][0], list)):
        
        if data_type == "numpy_array" and len(data.shape) == 4:
            # 4D array: show initial slice
            update_4d_slice_for_tab(parent, 0, 0)
        else:
            # 3D array: show initial slice
            update_3d_slice_for_tab(parent, 0)
    
    print(f"  Created data tab for {data_file['filename']} with tree: {tree}")
    print(f"  Tree frame packed with fill=tk.BOTH, expand=True")
    
    # Ensure the tree frame doesn't cover navigation controls
    tree_frame.pack_configure(before=None)  # Ensure proper packing order

def configure_4d_navigation_for_tab(tab_frame, data, nav_frame):
    """Configure dual-level navigation for a specific 4D array tab"""
    # Create a navigation frame for this tab
    tab_nav_frame = ttk.Frame(tab_frame)
    tab_nav_frame.pack(fill=tk.X, padx=5, pady=(5, 0), side=tk.TOP)
    
    # Add a border to make the navigation frame visible
    tab_nav_frame.configure(relief="solid", borderwidth=2)
    
    print(f"  Creating 4D navigation for tab with shape: {data.shape}")
    print(f"  Tab frame: {tab_frame}")
    print(f"  Navigation frame: {tab_nav_frame}")
    
    # First level: Batch navigation
    ttk.Label(tab_nav_frame, text="Batch:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Batch previous button
    batch_prev_btn = ttk.Button(tab_nav_frame, text="◀", width=3)
    batch_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current batch display
    batch_var = tk.IntVar(value=0)
    batch_label = ttk.Label(tab_nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    batch_label.pack(side=tk.LEFT, padx=2)
    
    # Batch next button
    batch_next_btn = ttk.Button(tab_nav_frame, text="▶", width=3)
    batch_next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Batch info
    total_batches = data.shape[0]
    ttk.Label(tab_nav_frame, text=f"of {total_batches} total batches", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Add some spacing before timestep controls
    ttk.Label(tab_nav_frame, text="", width=10).pack(side=tk.LEFT)
    
    # Second level: Timestep navigation
    ttk.Label(tab_nav_frame, text="Timestep:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Timestep previous button
    timestep_prev_btn = ttk.Button(tab_nav_frame, text="◀", width=3)
    timestep_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current timestep display
    timestep_var = tk.IntVar(value=0)
    timestep_label = ttk.Label(tab_nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    timestep_label.pack(side=tk.LEFT, padx=2)
    
    # Timestep next button
    timestep_next_btn = ttk.Button(tab_nav_frame, text="▶", width=3)
    timestep_next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Timestep info
    total_timesteps = data.shape[1]
    ttk.Label(tab_nav_frame, text=f"of {total_timesteps} total timesteps", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Add some spacing before jump controls
    ttk.Label(tab_nav_frame, text="", width=10).pack(side=tk.LEFT)
    
    # Jump to specific batch/timestep
    ttk.Label(tab_nav_frame, text="Jump to batch:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
    batch_jump_var = tk.StringVar(value="0")
    batch_jump_entry = ttk.Entry(tab_nav_frame, textvariable=batch_jump_var, width=6)
    batch_jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    ttk.Label(tab_nav_frame, text="timestep:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
    timestep_jump_var = tk.StringVar(value="0")
    timestep_jump_entry = ttk.Entry(tab_nav_frame, textvariable=timestep_jump_var, width=6)
    timestep_jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    jump_btn = ttk.Button(tab_nav_frame, text="Go")
    jump_btn.pack(side=tk.LEFT)
    
    print(f"  Created 4D navigation with {total_batches} batches and {total_timesteps} timesteps")
    print(f"  Navigation frame packed and configured")
    print(f"  Timestep controls should be visible: Timestep: ◀ 0 ▶ of 10 total timesteps")
    
    # Store navigation state in tab_frame for access by other functions
    tab_frame.batch_var = batch_var
    tab_frame.batch_label = batch_label
    tab_frame.timestep_var = timestep_var
    tab_frame.timestep_label = timestep_label
    tab_frame.batch_jump_var = batch_jump_var
    tab_frame.timestep_jump_var = timestep_jump_var
    
    # Navigation functions for this specific tab
    def change_batch(delta):
        current = batch_var.get()
        new_batch = current + delta
        if 0 <= new_batch < total_batches:
            batch_var.set(new_batch)
            batch_label.config(text=str(new_batch))
            batch_jump_var.set(str(new_batch))
            # Update this specific tab
            update_4d_slice_for_tab(tab_frame, new_batch, timestep_var.get())
    
    def change_timestep(delta):
        current = timestep_var.get()
        new_timestep = current + delta
        if 0 <= new_timestep < total_timesteps:
            timestep_var.set(new_timestep)
            timestep_label.config(text=str(new_timestep))
            timestep_jump_var.set(str(new_timestep))
            # Update this specific tab
            update_4d_slice_for_tab(tab_frame, batch_var.get(), new_timestep)
    
    def jump_to_4d_slice():
        try:
            new_batch = int(batch_jump_var.get())
            new_timestep = int(timestep_jump_var.get())
            if 0 <= new_batch < total_batches and 0 <= new_timestep < total_timesteps:
                batch_var.set(new_batch)
                timestep_var.set(new_timestep)
                batch_label.config(text=str(new_batch))
                timestep_label.config(text=str(new_timestep))
                # Update this specific tab
                update_4d_slice_for_tab(tab_frame, new_batch, new_timestep)
            else:
                batch_jump_var.set(str(batch_var.get()))
                timestep_jump_var.set(str(timestep_var.get()))
        except ValueError:
            batch_jump_var.set(str(batch_var.get()))
            timestep_jump_var.set(str(timestep_var.get()))
    
    # Connect the navigation functions to the buttons
    batch_prev_btn.config(command=lambda: change_batch(-1))
    batch_next_btn.config(command=lambda: change_batch(1))
    timestep_prev_btn.config(command=lambda: change_timestep(-1))
    timestep_next_btn.config(command=lambda: change_timestep(1))
    jump_btn.config(command=jump_to_4d_slice)
    
    # Bind Enter keys
    batch_jump_entry.bind('<Return>', lambda e: jump_to_4d_slice())
    timestep_jump_entry.bind('<Return>', lambda e: jump_to_4d_slice())
    
    # Force update the layout and ensure visibility
    tab_nav_frame.update_idletasks()
    tab_nav_frame.lift()  # Bring to front
    print(f"  Navigation frame layout updated and raised to front")
    
    # Verify all controls are present
    children = tab_nav_frame.winfo_children()
    print(f"  Navigation frame has {len(children)} child widgets")
    for i, child in enumerate(children):
        if hasattr(child, 'cget'):
            try:
                text = child.cget('text') if 'text' in child.configure() else 'N/A'
                print(f"    Child {i}: {type(child).__name__} - text: {text}")
            except:
                print(f"    Child {i}: {type(child).__name__} - text: N/A")
        else:
            print(f"    Child {i}: {type(child).__name__}")
    
    # Ensure navigation frame is visible and not covered
    tab_nav_frame.pack_configure(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))
    print(f"  Navigation frame repacked to ensure visibility")

def configure_3d_navigation_for_tab(tab_frame, data, data_type, nav_frame):
    """Configure single-level navigation for a specific 3D array tab"""
    # Create a navigation frame for this tab
    tab_nav_frame = ttk.Frame(tab_frame)
    tab_nav_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
    
    # Add a border to make the navigation frame visible
    tab_nav_frame.configure(relief="solid", borderwidth=1)
    
    # Single-level navigation
    ttk.Label(tab_nav_frame, text="Slice:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Previous button
    prev_btn = ttk.Button(tab_nav_frame, text="◀", width=3)
    prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current slice display
    slice_var = tk.IntVar(value=0)
    slice_label = ttk.Label(tab_nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    slice_label.pack(side=tk.LEFT, padx=2)
    
    # Next button
    next_btn = ttk.Button(tab_nav_frame, text="▶", width=3)
    next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Slice info
    if data_type == "numpy_array":
        total_slices = data.shape[0]
    else:  # JSON 3D array
        total_slices = len(data)
    ttk.Label(tab_nav_frame, text=f"of {total_slices} total slices", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Jump to specific slice
    ttk.Label(tab_nav_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
    jump_var = tk.StringVar(value="0")
    jump_entry = ttk.Entry(tab_nav_frame, textvariable=jump_var, width=6)
    jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    jump_btn = ttk.Button(tab_nav_frame, text="Go")
    jump_btn.pack(side=tk.LEFT)
    
    # Store navigation state
    tab_frame.slice_var = slice_var
    tab_frame.slice_label = slice_label
    tab_frame.jump_var = jump_var
    
    # Navigation functions for this specific tab
    def change_slice(delta):
        current = slice_var.get()
        if data_type == "numpy_array":
            total_slices = data.shape[0]
        else:  # JSON 3D array
            total_slices = len(data)
        
        new_slice = current + delta
        if 0 <= new_slice < total_slices:
            slice_var.set(new_slice)
            slice_label.config(text=str(new_slice))
            jump_var.set(str(new_slice))
            # Update this specific tab
            update_3d_slice_for_tab(tab_frame, new_slice)
    
    def jump_to_slice():
        try:
            new_slice = int(jump_var.get())
            if data_type == "numpy_array":
                total_slices = data.shape[0]
            else:  # JSON 3D array
                total_slices = len(data)
            
            if 0 <= new_slice < total_slices:
                slice_var.set(new_slice)
                slice_label.config(text=str(new_slice))
                # Update this specific tab
                update_3d_slice_for_tab(tab_frame, new_slice)
            else:
                jump_var.set(str(slice_var.get()))  # Reset to current
        except ValueError:
            jump_var.set(str(slice_var.get()))  # Reset to current
        
    # Connect the navigation functions to the buttons
    prev_btn.config(command=lambda: change_slice(-1))
    next_btn.config(command=lambda: change_slice(1))
    jump_btn.config(command=jump_to_slice)
    jump_entry.bind('<Return>', lambda e: jump_to_slice())

def update_4d_slice_for_tab(tab_frame, batch_idx, timestep_idx):
    """Update a specific tab's 4D slice display"""
    if not hasattr(tab_frame, '_tree') or not hasattr(tab_frame, '_data'):
        return
    
    tree = tab_frame._tree
    data = tab_frame._data
    
    if len(data.shape) != 4:
        return
    
    if 0 <= batch_idx < data.shape[0] and 0 <= timestep_idx < data.shape[1]:
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Define action type colors
        action_colors = {
            0: "lightblue",    # move
            1: "lightgreen",   # click
            2: "lightyellow",  # key_press/key_release
            3: "lightcoral"    # scroll
        }
        
        # Show ALL actions for the selected batch and timestep
        # data[batch_idx, timestep_idx, :, :] gives us (101, 8)
        for i in range(data.shape[2]):  # 101 actions
            try:
                values = [f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val) for val in data[batch_idx, timestep_idx, i]]
                
                # Insert the row
                item = tree.insert("", tk.END, text=f"Action {i}", values=values)
                
                # Color code based on action type (index 1 in the 8-feature vector: [count, type, x, y, button, key, scroll_dx, scroll_dy])
                # Only color actions that have actual actions (count > 0)
                if len(values) > 1:
                    try:
                        action_count = int(float(values[0]))  # Action count is at index 0
                        action_type = int(float(values[1]))   # Action type is at index 1
                        
                        # Debug: print first few actions to see the structure
                        if i < 5:
                            print(f"Action {i}: count={action_count}, type={action_type}, values={values[:3]}")
                        
                        # Only color if there are actual actions (count > 0) and it's a valid action type
                        if action_count > 0 and action_type in action_colors:
                            tree.tag_configure(f"action_{action_type}", background=action_colors[action_type])
                            tree.item(item, tags=(f"action_{action_type}",))
                        elif action_count > 0:
                            print(f"Action {i}: Non-zero action with type {action_type} (not in color map)")
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing action {i}: {e}")
                        pass  # Skip coloring if we can't parse the action type
                        
            except Exception as e:
                print(f"Error processing action {i}: {e}")
                values = [str(val) for val in data[batch_idx, timestep_idx, i]]
                tree.insert("", tk.END, text=f"Action {i}", values=values)

def update_3d_slice_for_tab(tab_frame, slice_idx):
    """Update a specific tab's 3D slice display"""
    if not hasattr(tab_frame, '_tree') or not hasattr(tab_frame, '_data'):
        return
    
    tree = tab_frame._tree
    data = tab_frame._data
    data_type = tab_frame._data_type
    
    if data_type == "numpy_array":
        if 0 <= slice_idx < data.shape[0]:
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Define action type colors for action targets
            action_colors = {
                0: "lightblue",    # move
                1: "lightgreen",   # click
                2: "lightyellow",  # key_press/key_release
                3: "lightcoral"    # scroll
            }
            
            # Add ALL data for this slice
            if len(data.shape) == 3:
                # 3D array: data[slice_idx, i, j] - show ALL timesteps for this batch
                for i in range(data.shape[1]):  # 10 timesteps
                    try:
                        values = [f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val) for val in data[slice_idx, i]]
                        
                        # Insert the row
                        item = tree.insert("", tk.END, text=f"Timestep {i}", values=values)
                        
                        # Color code based on action type for action targets
                        # Check if this looks like action data (8 features)
                        if len(values) == 8:
                            try:
                                action_count = int(float(values[0]))  # Action count is at index 0
                                action_type = int(float(values[1]))   # Action type is at index 1
                                
                                # Debug: print first few actions
                                if i < 3:
                                    print(f"Action Target {slice_idx}, Timestep {i}: count={action_count}, type={action_type}, values={values[:3]}")
                                
                                # Only color if there are actual actions (count > 0) and it's a valid action type
                                if action_count > 0 and action_type in action_colors:
                                    tree.tag_configure(f"action_{action_type}", background=action_colors[action_type])
                                    tree.item(item, tags=(f"action_{action_type}",))
                                elif action_count > 0:
                                    print(f"Action Target {slice_idx}, Timestep {i}: Non-zero action with type {action_type} (not in color map)")
                            except (ValueError, IndexError):
                                pass
                                
                    except:
                        values = [str(val) for val in data[slice_idx, i]]
                        tree.insert("", tk.END, text=f"Timestep {i}", values=values)
            elif len(data.shape) == 4:
                # 4D array: data[slice_idx, i, j, k] - show ALL timesteps for this batch
                for i in range(data.shape[1]):  # 10 timesteps
                    try:
                        values = [f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val) for val in data[slice_idx, i, 0]]  # Show first slice of dimension 3
                        
                        # Insert the row
                        item = tree.insert("", tk.END, text=f"Timestep {i}", values=values)
                        
                        # Color code based on action type for action targets
                        if len(values) == 8:
                            try:
                                action_count = int(float(values[0]))  # Action count is at index 0
                                action_type = int(float(values[1]))   # Action type is at index 1
                                
                                if action_count > 0 and action_type in action_colors:
                                    tree.tag_configure(f"action_{action_type}", background=action_colors[action_type])
                                    tree.item(item, tags=(f"action_{action_type}",))
                            except (ValueError, IndexError):
                                pass
                                
                    except:
                        values = [str(val) for val in data[slice_idx, i, 0]]
                        tree.insert("", tk.END, text=f"Timestep {i}", values=values)
    else:  # JSON 3D array
        if 0 <= slice_idx < len(data):
            # Update JSON slice
            update_3d_json_slice(tree, slice_idx)
        
def configure_basic_navigation(nav_frame, data, data_type):
    """Configure basic navigation for 1D/2D arrays"""
    # For 1D/2D arrays, just show info
    if data_type == "numpy_array":
        if len(data.shape) == 1:
            ttk.Label(nav_frame, text=f"1D Array: {data.shape[0]} elements", font=("Arial", 9)).pack(side=tk.LEFT)
        elif len(data.shape) == 2:
            ttk.Label(nav_frame, text=f"2D Array: {data.shape[0]} rows × {data.shape[1]} columns", font=("Arial", 9)).pack(side=tk.LEFT)
    else:
        ttk.Label(nav_frame, text=f"Data: {data_type}", font=("Arial", 9)).pack(side=tk.LEFT)

def configure_4d_navigation(nav_frame, data, data_files):
    """Configure dual-level navigation for 4D arrays (batch + timestep)"""
    # First level: Batch navigation
    ttk.Label(nav_frame, text="Batch:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Batch previous button
    batch_prev_btn = ttk.Button(nav_frame, text="◀", width=3)
    batch_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current batch display
    batch_var = tk.IntVar(value=0)
    batch_label = ttk.Label(nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    batch_label.pack(side=tk.LEFT, padx=2)
    
    # Batch next button
    batch_next_btn = ttk.Button(nav_frame, text="▶", width=3)
    batch_next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Batch info
    total_batches = data.shape[0]
    ttk.Label(nav_frame, text=f"of {total_batches} total batches", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Second level: Timestep navigation
    ttk.Label(nav_frame, text="Timestep:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 5))
    
    # Timestep previous button
    timestep_prev_btn = ttk.Button(nav_frame, text="◀", width=3)
    timestep_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current timestep display
    timestep_var = tk.IntVar(value=0)
    timestep_label = ttk.Label(nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    timestep_label.pack(side=tk.LEFT, padx=2)
    
    # Timestep next button
    timestep_next_btn = ttk.Button(nav_frame, text="▶", width=3)
    timestep_next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Timestep info
    total_timesteps = data.shape[1]
    ttk.Label(nav_frame, text=f"of {total_timesteps} total timesteps", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Jump to specific batch/timestep
    ttk.Label(nav_frame, text="Jump to batch:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
    batch_jump_var = tk.StringVar(value="0")
    batch_jump_entry = ttk.Entry(nav_frame, textvariable=batch_jump_var, width=6)
    batch_jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    ttk.Label(nav_frame, text="timestep:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
    timestep_jump_var = tk.StringVar(value="0")
    timestep_jump_entry = ttk.Entry(nav_frame, textvariable=timestep_jump_var, width=6)
    timestep_jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    jump_btn = ttk.Button(nav_frame, text="Go")
    jump_btn.pack(side=tk.LEFT)
    
    # Store navigation state in nav_frame for access by other functions
    nav_frame.batch_var = batch_var
    nav_frame.batch_label = batch_label
    nav_frame.timestep_var = timestep_var
    nav_frame.timestep_label = timestep_label
    nav_frame.batch_jump_var = batch_jump_var
    nav_frame.timestep_jump_var = timestep_jump_var
    
    # Function to update timestamp display
    def update_timestamp_display(batch_idx, timestep_idx):
        try:
            # Look for original Unix timestamps data first
            original_unix_timestamps_data = None
            for data_file in data_files:
                if "original_unix_timestamps" in data_file['filename'].lower():
                    original_unix_timestamps_data = data_file['data']
                    break
            
            if original_unix_timestamps_data is not None and len(original_unix_timestamps_data.shape) >= 1:
                # Extract timestamp from original Unix timestamps data
                if batch_idx < len(original_unix_timestamps_data):
                    # Get the actual Unix timestamp for this batch
                    unix_timestamp = original_unix_timestamps_data[batch_idx]
                    
                    # Convert Unix timestamp to readable format
                    try:
                        from datetime import datetime
                        dt = datetime.fromtimestamp(unix_timestamp / 1000)  # Convert from milliseconds
                        readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        timestamp_text = f"Batch {batch_idx}, Timestep {timestep_idx} - {readable_time} (Unix: {unix_timestamp})"
                    except:
                        # Fallback if datetime conversion fails
                        timestamp_text = f"Batch {batch_idx}, Timestep {timestep_idx} - Unix: {unix_timestamp}"
                    
                    nav_frame.timestamp_value.config(text=timestamp_text)
                else:
                    nav_frame.timestamp_value.config(text="Invalid batch/timestep")
            else:
                # Fallback: Look for regular timestamps data
                timestamps_data = None
                for data_file in data_files:
                    if "timestamps" in data_file['filename'].lower():
                        timestamps_data = data_file['data']
                        break
                
                if timestamps_data is not None and len(timestamps_data.shape) >= 1:
                    # Extract timestamp from timestamps data
                    if batch_idx < len(timestamps_data):
                        # Get the actual Unix timestamp for this batch
                        unix_timestamp = timestamps_data[batch_idx]
                        
                        # Convert Unix timestamp to readable format
                        try:
                            from datetime import datetime
                            dt = datetime.fromtimestamp(unix_timestamp / 1000)  # Convert from milliseconds
                            readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                            timestamp_text = f"Batch {batch_idx}, Timestep {timestep_idx} - {readable_time} (Unix: {unix_timestamp})"
                        except:
                            # Fallback if datetime conversion fails
                            timestamp_text = f"Batch {batch_idx}, Timestep {timestep_idx} - Unix: {unix_timestamp}"
                        
                        nav_frame.timestamp_value.config(text=timestamp_text)
                    else:
                        nav_frame.timestamp_value.config(text="Invalid batch/timestep")
                else:
                    # Fallback: Look for gamestate data to extract timestamp
                    gamestate_data = None
                    for data_file in data_files:
                        if "gamestate" in data_file['filename'].lower():
                            gamestate_data = data_file['data']
                            break
                    
                    if gamestate_data is not None and len(gamestate_data.shape) >= 2:
                        # Extract timestamp from gamestate data
                        # Assuming the first feature is timestamp or we can derive it
                        if batch_idx < gamestate_data.shape[0] and timestep_idx < gamestate_data.shape[1]:
                            # For now, show batch and timestep info
                            # In a real implementation, you'd extract actual timestamps from the gamestate data
                            timestamp_text = f"Batch {batch_idx}, Timestep {timestep_idx}"
                            nav_frame.timestamp_value.config(text=timestamp_text)
                        else:
                            nav_frame.timestamp_value.config(text="Invalid batch/timestep")
                    else:
                        nav_frame.timestamp_value.config(text=f"Batch {batch_idx}, Timestep {timestep_idx}")
        except Exception as e:
            nav_frame.timestamp_value.config(text=f"Error: {str(e)}")
    
    # Navigation functions
    def change_batch(delta):
        current = batch_var.get()
        new_batch = current + delta
        if 0 <= new_batch < total_batches:
            batch_var.set(new_batch)
            batch_label.config(text=str(new_batch))
            batch_jump_var.set(str(new_batch))
            # Update timestamp display
            update_timestamp_display(new_batch, timestep_var.get())
            # Update all tabs
            update_all_tabs_4d(nav_frame, new_batch, timestep_var.get())
    
    def change_timestep(delta):
        current = timestep_var.get()
        new_timestep = current + delta
        if 0 <= new_timestep < total_timesteps:
            timestep_var.set(new_timestep)
            timestep_label.config(text=str(new_timestep))
            timestep_jump_var.set(str(new_timestep))
            # Update timestamp display
            update_timestamp_display(batch_var.get(), new_timestep)
            # Update all tabs
            update_all_tabs_4d(nav_frame, batch_var.get(), new_timestep)
    
    def jump_to_4d_slice():
        try:
            new_batch = int(batch_jump_var.get())
            new_timestep = int(timestep_jump_var.get())
            if 0 <= new_batch < total_batches and 0 <= new_timestep < total_timesteps:
                batch_var.set(new_batch)
                timestep_var.set(new_timestep)
                batch_label.config(text=str(new_batch))
                timestep_label.config(text=str(new_timestep))
                # Update timestamp display
                update_timestamp_display(new_batch, new_timestep)
                # Update all tabs
                update_all_tabs_4d(nav_frame, new_batch, new_timestep)
            else:
                batch_jump_var.set(str(batch_var.get()))
                timestep_jump_var.set(str(timestep_var.get()))
        except ValueError:
            batch_jump_var.set(str(batch_var.get()))
            timestep_jump_var.set(str(timestep_var.get()))
    
    # Connect the navigation functions to the buttons
    batch_prev_btn.config(command=lambda: change_batch(-1))
    batch_next_btn.config(command=lambda: change_batch(1))
    timestep_prev_btn.config(command=lambda: change_timestep(-1))
    timestep_next_btn.config(command=lambda: change_timestep(1))
    jump_btn.config(command=jump_to_4d_slice)
    
    # Bind Enter keys
    batch_jump_entry.bind('<Return>', lambda e: jump_to_4d_slice())
    timestep_jump_entry.bind('<Return>', lambda e: jump_to_4d_slice())
    
    # Initialize timestamp display
    update_timestamp_display(0, 0)

def update_all_tabs_4d(nav_frame, batch_idx, timestep_idx):
    """Update all tabs with new 4D navigation state"""
    # Get the notebook from nav_frame
    if not hasattr(nav_frame, 'notebook'):
        return
    
    notebook = nav_frame.notebook
    
    # Update each tab
    for tab_id in notebook.tabs():
        tab_frame = notebook.nametowidget(tab_id)
        if hasattr(tab_frame, '_data_type') and hasattr(tab_frame, '_data'):
            if tab_frame._data_type == "numpy_array" and len(tab_frame._data.shape) == 4:
                update_4d_slice_for_tab(tab_frame, batch_idx, timestep_idx)
            elif tab_frame._data_type == "numpy_array" and len(tab_frame._data.shape) == 3:
                # For 3D arrays, use batch_idx as the slice index
                update_3d_slice_for_tab(tab_frame, batch_idx)
            # For 1D/2D arrays, no update needed

def configure_3d_navigation(nav_frame, data, data_type, data_files):
    """Configure single-level navigation for 3D arrays"""
    # Single-level navigation
    ttk.Label(nav_frame, text="Slice:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
    
    # Previous button
    prev_btn = ttk.Button(nav_frame, text="◀", width=3)
    prev_btn.pack(side=tk.LEFT, padx=(0, 2))
    
    # Current slice display
    slice_var = tk.IntVar(value=0)
    slice_label = ttk.Label(nav_frame, text="0", font=("Arial", 10, "bold"), width=6)
    slice_label.pack(side=tk.LEFT, padx=2)
    
    # Next button
    next_btn = ttk.Button(nav_frame, text="▶", width=3)
    next_btn.pack(side=tk.LEFT, padx=(2, 10))
    
    # Slice info
    if data_type == "numpy_array":
        total_slices = data.shape[0]
    else:  # JSON 3D array
        total_slices = len(data)
    ttk.Label(nav_frame, text=f"of {total_slices} total slices", font=("Arial", 9)).pack(side=tk.LEFT)
    
    # Jump to specific slice
    ttk.Label(nav_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
    jump_var = tk.StringVar(value="0")
    jump_entry = ttk.Entry(nav_frame, textvariable=jump_var, width=6)
    jump_entry.pack(side=tk.LEFT, padx=(0, 5))
    
    jump_btn = ttk.Button(nav_frame, text="Go")
    jump_btn.pack(side=tk.LEFT)
    
    # Store navigation state
    nav_frame.slice_var = slice_var
    nav_frame.slice_label = slice_label
    nav_frame.jump_var = jump_var
    
    # Function to update timestamp display
    def update_timestamp_display(slice_idx):
        try:
            # Look for original Unix timestamps data first
            original_unix_timestamps_data = None
            for data_file in data_files:
                if "original_unix_timestamps" in data_file['filename'].lower():
                    original_unix_timestamps_data = data_file['data']
                    break
            
            if original_unix_timestamps_data is not None and len(original_unix_timestamps_data.shape) >= 1:
                # Extract timestamp from original Unix timestamps data
                if slice_idx < len(original_unix_timestamps_data):
                    # Get the actual Unix timestamp for this batch
                    unix_timestamp = original_unix_timestamps_data[slice_idx]
                    
                    # Convert Unix timestamp to readable format
                    try:
                        from datetime import datetime
                        dt = datetime.fromtimestamp(unix_timestamp / 1000)  # Convert from milliseconds
                        readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        timestamp_text = f"Batch {slice_idx} - {readable_time} (Unix: {unix_timestamp})"
                    except:
                        # Fallback if datetime conversion fails
                        timestamp_text = f"Batch {slice_idx} - Unix: {unix_timestamp}"
                    
                    nav_frame.timestamp_value.config(text=timestamp_text)
                else:
                    nav_frame.timestamp_value.config(text="Invalid slice")
            else:
                # Fallback: Look for regular timestamps data
                timestamps_data = None
                for data_file in data_files:
                    if "timestamps" in data_file['filename'].lower():
                        timestamps_data = data_file['data']
                        break
                
                if timestamps_data is not None and len(timestamps_data.shape) >= 1:
                    # Extract timestamp from timestamps data
                    if slice_idx < len(timestamps_data):
                        # Get the actual Unix timestamp for this batch
                        unix_timestamp = timestamps_data[slice_idx]
                        
                        # Convert Unix timestamp to readable format
                        try:
                            from datetime import datetime
                            dt = datetime.fromtimestamp(unix_timestamp / 1000)  # Convert from milliseconds
                            readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                            timestamp_text = f"Batch {slice_idx} - {readable_time} (Unix: {unix_timestamp})"
                        except:
                            # Fallback if datetime conversion fails
                            timestamp_text = f"Batch {slice_idx} - Unix: {unix_timestamp}"
                        
                        nav_frame.timestamp_value.config(text=timestamp_text)
                    else:
                        nav_frame.timestamp_value.config(text="Invalid slice")
                else:
                    # Fallback: Look for gamestate data to extract timestamp
                    gamestate_data = None
                    for data_file in data_files:
                        if "gamestate" in data_file['filename'].lower():
                            gamestate_data = data_file['data']
                            break
                    
                    if gamestate_data is not None and len(gamestate_data.shape) >= 2:
                        # Extract timestamp from gamestate data
                        if slice_idx < gamestate_data.shape[0]:
                            # For now, show slice info
                            # In a real implementation, you'd extract actual timestamps from the gamestate data
                            timestamp_text = f"Batch {slice_idx}"
                            nav_frame.timestamp_value.config(text=timestamp_text)
                        else:
                            nav_frame.timestamp_value.config(text="Invalid slice")
                    else:
                        nav_frame.timestamp_value.config(text=f"Batch {slice_idx}")
        except Exception as e:
            nav_frame.timestamp_value.config(text=f"Error: {str(e)}")
    
    # Navigation functions for this specific tab
    def change_slice(delta):
        current = slice_var.get()
        if data_type == "numpy_array":
            total_slices = data.shape[0]
        else:  # JSON 3D array
            total_slices = len(data)
        
        new_slice = current + delta
        if 0 <= new_slice < total_slices:
            slice_var.set(new_slice)
            slice_label.config(text=str(new_slice))
            jump_var.set(str(new_slice))
            # Update timestamp display
            update_timestamp_display(new_slice)
            # Update all tabs
            update_all_tabs_3d(nav_frame, new_slice)
    
    def jump_to_slice():
        try:
            new_slice = int(jump_var.get())
            if data_type == "numpy_array":
                total_slices = data.shape[0]
            else:  # JSON 3D array
                total_slices = len(data)
            
            if 0 <= new_slice < total_slices:
                slice_var.set(new_slice)
                slice_label.config(text=str(new_slice))
                # Update timestamp display
                update_timestamp_display(new_slice)
                # Update all tabs
                update_all_tabs_3d(nav_frame, new_slice)
            else:
                jump_var.set(str(slice_var.get()))  # Reset to current
        except ValueError:
            jump_var.set(str(slice_var.get()))  # Reset to current
    
    # Connect the navigation functions to the buttons
    prev_btn.config(command=lambda: change_slice(-1))
    next_btn.config(command=lambda: change_slice(1))
    jump_btn.config(command=jump_to_slice)
    jump_entry.bind('<Return>', lambda e: jump_to_slice())
    
    # Initialize timestamp display
    update_timestamp_display(0)

def update_all_tabs_3d(nav_frame, slice_idx):
    """Update all tabs with new 3D navigation state"""
    # Get the notebook from nav_frame
    if not hasattr(nav_frame, 'notebook'):
        return
    
    notebook = nav_frame.notebook
    
    # Update each tab
    for tab_id in notebook.tabs():
        tab_frame = notebook.nametowidget(tab_id)
        if hasattr(tab_frame, '_data_type') and hasattr(tab_frame, '_data'):
            if (tab_frame._data_type == "numpy_array" and len(tab_frame._data.shape) == 3) or \
               (tab_frame._data_type == "json" and isinstance(tab_frame._data, list) and tab_frame._data and 
                isinstance(tab_frame._data[0], list) and tab_frame._data[0] and isinstance(tab_frame._data[0][0], list)):
                update_3d_slice_for_tab(tab_frame, slice_idx)
            elif tab_frame._data_type == "numpy_array" and len(tab_frame._data.shape) == 4:
                # For 4D arrays, use slice_idx as the batch index and keep current timestep
                current_timestep = getattr(nav_frame, 'timestep_var', tk.IntVar(value=0)).get()
                update_4d_slice_for_tab(tab_frame, slice_idx, current_timestep)
            # For 1D/2D arrays, no update needed

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
            try:
                formatted_val = f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val)
            except:
                formatted_val = str(val)
            tree.insert("", tk.END, values=(i, formatted_val))
            
    elif len(data.shape) == 2:
        tree["columns"] = tuple([f"col_{i}" for i in range(data.shape[1])])
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Row")
        
        for i in range(data.shape[1]):
            tree.column(f"col_{i}", anchor=tk.CENTER, width=120)  # Consistent width
            tree.heading(f"col_{i}", text=f"Col {i}")
        
        for i in range(data.shape[0]):
            try:
                values = [f"{val:.6f}" if isinstance(val, (int, float, np.number)) else str(val) for val in data[i]]
            except:
                values = [str(val) for val in data[i]]
            tree.insert("", tk.END, text=f"Row {i}", values=values)
            
    elif len(data.shape) == 3:
        # For 3D arrays, show slice navigation with descriptive column headers
        tree["columns"] = tuple([f"F{i}" for i in range(data.shape[2])])  # Feature columns
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Timestep")
        
        for i in range(data.shape[2]):
            tree.column(f"F{i}", anchor=tk.CENTER, width=120)  # Consistent width
            tree.heading(f"F{i}", text=f"F{i}")  # Feature i
    else:
        # For 4D+ arrays, show slice navigation for first two dimensions
        tree["columns"] = tuple([f"F{i}" for i in range(data.shape[-1])])  # Feature columns
        tree.column("#0", width=100, stretch=tk.NO)
        tree.heading("#0", text="Action")
        
        for i in range(data.shape[-1]):
            tree.column(f"F{i}", anchor=tk.CENTER, width=120)  # Consistent width
            tree.heading(f"F{i}", text=f"F{i}")  # Feature i

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
                if isinstance(val, float):
                    # Format floats consistently with NumPy display (6 decimal places)
                    values.append(f"{val:.6f}")
                elif isinstance(val, int):
                    # Format integers as strings without decimal places
                    values.append(str(val))
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
    if len(sys.argv) < 2:
        print("Usage: python print_numpy_array.py <file_path1> [file_path2] ...")
        print("Supported formats: .npy (NumPy arrays), .json (JSON files), .csv (CSV files)")
        sys.exit(1)
    
    filepaths = sys.argv[1:]
    display_data(filepaths)