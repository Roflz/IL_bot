#!/usr/bin/env python3
"""
Tool to display action sequence data in a table format
"""

import numpy as np
import sys
import tkinter as tk
from tkinter import ttk

def display_action_sequence(filepath):
    """Load and display action sequence in a tkinter window"""
    try:
        array = np.load(filepath)
        print(f"Action sequence shape: {array.shape}")
        print(f"Data type: {array.dtype}")
        
        # Create window
        root = tk.Tk()
        root.title(f"Action Sequence: {array.shape}")
        root.geometry("1400x900")
        
        # Add navigation at the top for timesteps
        nav_frame = ttk.Frame(root)
        nav_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        # Navigation controls
        ttk.Label(nav_frame, text="Timestep:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        # Previous button
        prev_btn = ttk.Button(nav_frame, text="◀", width=3, command=lambda: change_timestep(-1))
        prev_btn.pack(side=tk.LEFT, padx=(0, 2))
        
        # Current timestep display
        timestep_var = tk.IntVar(value=0)
        timestep_label = ttk.Label(nav_frame, text="T0", font=("Arial", 10, "bold"), width=6)
        timestep_label.pack(side=tk.LEFT, padx=2)
        
        # Next button
        next_btn = ttk.Button(nav_frame, text="▶", width=3, command=lambda: change_timestep(1))
        next_btn.pack(side=tk.LEFT, padx=(2, 10))
        
        # Timestep info
        ttk.Label(nav_frame, text=f"of {array.shape[0]} total timesteps", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Jump to specific timestep
        ttk.Label(nav_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        jump_var = tk.StringVar(value="0")
        jump_entry = ttk.Entry(nav_frame, textvariable=jump_var, width=6)
        jump_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        jump_btn = ttk.Button(nav_frame, text="Go", command=lambda: jump_to_timestep())
        jump_btn.pack(side=tk.LEFT)
        
        # Bind Enter key to jump
        jump_entry.bind('<Return>', lambda e: jump_to_timestep())
        
        # Add action info
        if array.shape[1] > 0:
            ttk.Label(nav_frame, text="|", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 20))
            ttk.Label(nav_frame, text=f"Actions per timestep: {array.shape[1]}", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Create main frame for the table
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Container for tree + scrollbars
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tree
        tree = ttk.Treeview(tree_frame)
        
        # Configure columns for action data
        if array.shape[1] > 0:
            # Create columns for action elements
            columns = ["action_count"]
            # Add columns for each action element (assuming 8 elements per action)
            for i in range(8):
                columns.append(f"elem_{i}")
            
            tree["columns"] = tuple(columns)
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Action #")
            
            # Configure column headers
            tree.column("action_count", anchor=tk.CENTER, width=80)
            tree.heading("action_count", text="Count")
            
            for i in range(8):
                col_name = f"elem_{i}"
                tree.column(col_name, anchor=tk.CENTER, width=80)
                tree.heading(col_name, text=f"Element {i}")
        else:
            # Fallback for empty arrays
            tree["columns"] = ("value",)
            tree.column("#0", width=100, stretch=tk.NO)
            tree.column("value", anchor=tk.CENTER, width=150)
            tree.heading("#0", text="Index")
            tree.heading("value", text="Value")
        
        # Scrollbars
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
        
        # Navigation functions
        def change_timestep(delta):
            current = timestep_var.get()
            new_timestep = current + delta
            if 0 <= new_timestep < array.shape[0]:
                timestep_var.set(new_timestep)
                update_timestep()
        
        def jump_to_timestep():
            try:
                new_timestep = int(jump_var.get())
                if 0 <= new_timestep < array.shape[0]:
                    timestep_var.set(new_timestep)
                    update_timestep()
                else:
                    jump_var.set(str(timestep_var.get()))  # Reset to current
            except ValueError:
                jump_var.set(str(timestep_var.get()))  # Reset to current
        
        def update_timestep():
            timestep_idx = timestep_var.get()
            if 0 <= timestep_idx < array.shape[0]:
                # Update label
                timestep_label.config(text=f"T{timestep_idx}")
                jump_var.set(str(timestep_idx))
                
                # Clear existing items
                for item in tree.get_children():
                    tree.delete(item)
                
                # Add new timestep data
                if array.shape[1] > 0:
                    # Parse action data from the flattened tensor
                    action_data = array[timestep_idx]
                    
                    # The first element is the action count
                    action_count = int(action_data[0]) if len(action_data) > 0 else 0
                    
                    if action_count > 0:
                        # Parse individual actions (each action has 8 elements)
                        for action_idx in range(action_count):
                            start_idx = 1 + action_idx * 8
                            if start_idx + 7 < len(action_data):
                                action_elements = action_data[start_idx:start_idx + 8]
                                
                                # Format the values
                                values = [action_count]
                                for elem in action_elements:
                                    values.append(f"{elem:.3f}")
                                
                                tree.insert("", tk.END, text=f"Action {action_idx + 1}", values=values)
                            else:
                                # Incomplete action data
                                values = [action_count] + ["N/A"] * 8
                                tree.insert("", tk.END, text=f"Action {action_idx + 1} (incomplete)", values=values)
                    else:
                        # No actions in this timestep
                        tree.insert("", tk.END, text="No actions", values=["0"] + [""] * 8)
                else:
                    # Empty array
                    tree.insert("", tk.END, text="Empty", values=["No data"])
        
        # Initial timestep
        update_timestep()
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error loading action sequence: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_action_sequence.py <numpy_file_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    display_action_sequence(filepath)
