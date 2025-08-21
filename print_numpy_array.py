#!/usr/bin/env python3
"""
Simple script to display a numpy array in a table format
"""

import numpy as np
import sys
import tkinter as tk
from tkinter import ttk

def display_numpy_array(filepath):
    """Load and display numpy array in a tkinter window"""
    try:
        array = np.load(filepath)
        print(f"Array shape: {array.shape}")
        print(f"Array dtype: {array.dtype}")
        
        # Create window
        root = tk.Tk()
        root.title(f"Numpy Array: {array.shape}")
        root.geometry("1400x900")
        
        # Add 3D navigation at the top if needed
        if len(array.shape) >= 3:
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
            ttk.Label(nav_frame, text=f"of {array.shape[0]} total slices", font=("Arial", 9)).pack(side=tk.LEFT)
            
            # Jump to specific slice
            ttk.Label(nav_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
            jump_var = tk.StringVar(value="0")
            jump_entry = ttk.Entry(nav_frame, textvariable=jump_var, width=6)
            jump_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            jump_btn = ttk.Button(nav_frame, text="Go", command=lambda: jump_to_slice())
            jump_btn.pack(side=tk.LEFT)
            
            # Bind Enter key to jump
            jump_entry.bind('<Return>', lambda e: jump_to_slice())
        
        # Create main frame for the table
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Container for tree + scrollbars
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # CREATE TREE *INSIDE* tree_frame (not main_frame)
        tree = ttk.Treeview(tree_frame)
        
        # Configure columns based on array shape
        if len(array.shape) == 1:
            tree["columns"] = ("index", "value")
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("index", anchor=tk.CENTER, width=100)
            tree.column("value", anchor=tk.CENTER, width=150)
            tree.heading("#0", text="")
            tree.heading("index", text="Index")
            tree.heading("value", text="Value")
            
            for i, val in enumerate(array):
                tree.insert("", tk.END, values=(i, f"{val:.6f}"))
                
        elif len(array.shape) == 2:
            tree["columns"] = tuple([f"col_{i}" for i in range(array.shape[1])])
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Row")
            
            for i in range(array.shape[1]):
                tree.column(f"col_{i}", anchor=tk.CENTER, width=100)
                tree.heading(f"col_{i}", text=f"Col {i}")
            
            for i in range(array.shape[0]):
                values = [f"{val:.6f}" for val in array[i]]
                tree.insert("", tk.END, text=f"Row {i}", values=values)
                
        else:
            # For 3D+, show slice navigation
            tree["columns"] = tuple([f"col_{i}" for i in range(array.shape[2])])
            tree.column("#0", width=100, stretch=tk.NO)
            tree.heading("#0", text="Row")
            
            for i in range(array.shape[2]):
                tree.column(f"col_{i}", anchor=tk.CENTER, width=100)
                tree.heading(f"col_{i}", text=f"Col {i}")
        
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
        
        # 3D navigation functions
        if len(array.shape) >= 3:
            def change_slice(delta):
                current = slice_var.get()
                new_slice = current + delta
                if 0 <= new_slice < array.shape[0]:
                    slice_var.set(new_slice)
                    update_3d_slice()
            
            def jump_to_slice():
                try:
                    new_slice = int(jump_var.get())
                    if 0 <= new_slice < array.shape[0]:
                        slice_var.set(new_slice)
                        update_3d_slice()
                    else:
                        jump_var.set(str(slice_var.get()))  # Reset to current
                except ValueError:
                    jump_var.set(str(slice_var.get()))  # Reset to current
            
            def update_3d_slice():
                slice_idx = slice_var.get()
                if 0 <= slice_idx < array.shape[0]:
                    # Update label
                    slice_label.config(text=str(slice_idx))
                    jump_var.set(str(slice_idx))
                    
                    # Clear existing items
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # Add new slice data
                    for i in range(array.shape[1]):
                        values = [f"{val:.6f}" for val in array[slice_idx, i]]
                        tree.insert("", tk.END, text=f"Row {i}", values=values)
            
            # Initial slice
            update_3d_slice()
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error loading array: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_numpy_array.py <numpy_file_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    display_numpy_array(filepath)
