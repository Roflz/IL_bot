#!/usr/bin/env python3
"""
Simple Recorder GUI
==================

A minimal GUI for running plans with the simple_recorder system.
Allows selecting plans, running multiple plans in sequence, and pausing between plans.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import threading
import time
import os
import logging
import io
import sys
import json
from pathlib import Path
import psutil
from typing import Dict, List, Any, Optional, TypedDict

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_rj_loop import AVAILABLE_PLANS


class PlanEntry(TypedDict):
    """Data structure for storing plan information with rules and parameters."""
    name: str                 # plan id key (e.g., "ge", "woodcutting_2")
    label: str                # display label from AVAILABLE_PLANS
    rules: Dict[str, Any]     # {"max_minutes": int|None, "stop_skill": str|None, "stop_items": [{"name": str, "qty": int}]}
    params: Dict[str, Any]    # {"buy_items": [...], "sell_items": [...], "generic": {...}, "compiled_ge_buy": str, "compiled_ge_sell": str}


class PlanEditor:
    """Modal editor for plan rules and parameters."""
    
    def __init__(self, parent, plan_entry: PlanEntry, available_plans: Dict[str, Any]):
        self.parent = parent
        self.plan_entry = plan_entry.copy()
        self.available_plans = available_plans
        self.result = None
        
        # Create modal window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Edit Parameters: {plan_entry['label']}")
        self.window.geometry("600x500")
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.window.winfo_screenheight() // 2) - (500 // 2)
        self.window.geometry(f"600x500+{x}+{y}")
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the editor widgets."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for Rules and Parameters
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Rules tab
        rules_frame = ttk.Frame(notebook, padding="10")
        notebook.add(rules_frame, text="Rules")
        self.create_rules_tab(rules_frame)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook, padding="10")
        notebook.add(params_frame, text="Parameters")
        self.create_parameters_tab(params_frame)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.RIGHT)
        
    def create_rules_tab(self, parent):
        """Create the rules editing tab."""
        # Max Time
        ttk.Label(parent, text="Max Time (minutes):", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        max_time_value = self.plan_entry['rules'].get('max_minutes', 0)
        self.max_time_var = tk.StringVar(value=str(max_time_value) if max_time_value else "0")
        max_time_spinbox = ttk.Spinbox(parent, from_=0, to=10000, textvariable=self.max_time_var, width=10)
        max_time_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Stop Skill
        ttk.Label(parent, text="Stop at Skill:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Skill name and level frame
        skill_frame = ttk.Frame(parent)
        skill_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        stop_skill_value = self.plan_entry['rules'].get('stop_skill', '')
        stop_skill_level = self.plan_entry['rules'].get('stop_skill_level', 0)
        
        self.stop_skill_var = tk.StringVar(value=stop_skill_value if stop_skill_value else "")
        skill_combo = ttk.Combobox(skill_frame, textvariable=self.stop_skill_var, width=12, state="readonly")
        skill_list = ["", "Attack", "Strength", "Defence", "Ranged", "Magic", "Woodcutting", "Fishing", 
                     "Cooking", "Mining", "Smithing", "Firemaking", "Crafting", "Fletching", "Runecraft", 
                     "Herblore", "Agility", "Thieving", "Slayer", "Farming", "Construction", "Hunter", "Prayer"]
        skill_combo['values'] = skill_list
        skill_combo.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        ttk.Label(skill_frame, text="Level:").grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        self.stop_skill_level_var = tk.StringVar(value=str(stop_skill_level) if stop_skill_level else "0")
        skill_level_spinbox = ttk.Spinbox(skill_frame, from_=1, to=99, textvariable=self.stop_skill_level_var, width=5)
        skill_level_spinbox.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Total Level
        ttk.Label(parent, text="Total Level:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        total_level_value = self.plan_entry['rules'].get('total_level', 0)
        self.total_level_var = tk.StringVar(value=str(total_level_value) if total_level_value else "0")
        total_level_spinbox = ttk.Spinbox(parent, from_=0, to=2277, textvariable=self.total_level_var, width=10)
        total_level_spinbox.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Stop Items
        ttk.Label(parent, text="Stop with Items:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.NW, pady=(10, 5))
        
        # Items frame
        items_frame = ttk.Frame(parent)
        items_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        items_frame.columnconfigure(0, weight=1)
        
        # Items treeview
        self.items_tree = ttk.Treeview(items_frame, columns=('name', 'qty'), show='headings', height=4)
        self.items_tree.heading('name', text='Item Name')
        self.items_tree.heading('qty', text='Quantity')
        self.items_tree.column('name', width=150)
        self.items_tree.column('qty', width=80)
        self.items_tree.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Items buttons
        items_btn_frame = ttk.Frame(items_frame)
        items_btn_frame.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(items_btn_frame, text="Add", command=self.add_stop_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(items_btn_frame, text="Edit", command=self.edit_stop_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(items_btn_frame, text="Remove", command=self.remove_stop_item).pack(side=tk.LEFT)
        
        # Load existing stop items
        for item in self.plan_entry['rules'].get('stop_items', []):
            self.items_tree.insert('', 'end', values=(item['name'], item['qty']))
            
    def create_parameters_tab(self, parent):
        """Create the parameters editing tab."""
        plan_name = self.plan_entry['name']
        
        if 'ge' == plan_name.lower():
            self.create_ge_parameters_tab(parent)
        elif plan_name == 'ge_trade':
            self.create_ge_trade_parameters_tab(parent)
        else:
            self.create_generic_parameters_tab(parent)
            
    def create_ge_parameters_tab(self, parent):
        """Create GE-specific parameters tab."""
        # Buy Items
        ttk.Label(parent, text="Buy Items:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.NW, pady=(0, 5))
        
        buy_frame = ttk.Frame(parent)
        buy_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 10))
        buy_frame.columnconfigure(0, weight=1)
        
        self.buy_tree = ttk.Treeview(buy_frame, columns=('name', 'qty', 'bumps', 'price'), show='headings', height=3)
        self.buy_tree.heading('name', text='Name')
        self.buy_tree.heading('qty', text='Qty')
        self.buy_tree.heading('bumps', text='Bumps')
        self.buy_tree.heading('price', text='Set Price')
        self.buy_tree.column('name', width=120)
        self.buy_tree.column('qty', width=60)
        self.buy_tree.column('bumps', width=60)
        self.buy_tree.column('price', width=80)
        self.buy_tree.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        buy_btn_frame = ttk.Frame(buy_frame)
        buy_btn_frame.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(buy_btn_frame, text="Add", command=self.add_buy_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buy_btn_frame, text="Edit", command=self.edit_buy_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buy_btn_frame, text="Remove", command=self.remove_buy_item).pack(side=tk.LEFT)
        
        # Sell Items
        ttk.Label(parent, text="Sell Items:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.NW, pady=(10, 5))
        
        sell_frame = ttk.Frame(parent)
        sell_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        sell_frame.columnconfigure(0, weight=1)
        
        self.sell_tree = ttk.Treeview(sell_frame, columns=('name', 'qty', 'bumps', 'price'), show='headings', height=3)
        self.sell_tree.heading('name', text='Name')
        self.sell_tree.heading('qty', text='Qty')
        self.sell_tree.heading('bumps', text='Bumps')
        self.sell_tree.heading('price', text='Set Price')
        self.sell_tree.column('name', width=120)
        self.sell_tree.column('qty', width=60)
        self.sell_tree.column('bumps', width=60)
        self.sell_tree.column('price', width=80)
        self.sell_tree.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        sell_btn_frame = ttk.Frame(sell_frame)
        sell_btn_frame.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(sell_btn_frame, text="Add", command=self.add_sell_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(sell_btn_frame, text="Edit", command=self.edit_sell_item).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(sell_btn_frame, text="Remove", command=self.remove_sell_item).pack(side=tk.LEFT)
        
        # Load existing items
        for item in self.plan_entry['params'].get('buy_items', []):
            self.buy_tree.insert('', 'end', values=(item['name'], item['quantity'], item['bumps'], item['set_price']))
            
        for item in self.plan_entry['params'].get('sell_items', []):
            self.sell_tree.insert('', 'end', values=(item['name'], item['quantity'], item['bumps'], item['set_price']))
            
    def create_ge_trade_parameters_tab(self, parent):
        """Create GE trade-specific parameters tab."""
        # Role selection
        ttk.Label(parent, text="Role:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Get existing role or default to worker
        existing_role = self.plan_entry['params'].get('role', 'worker')
        self.role_var = tk.StringVar(value=existing_role)
        
        role_frame = ttk.Frame(parent)
        role_frame.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        ttk.Radiobutton(role_frame, text="Worker", variable=self.role_var, value="worker").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(role_frame, text="Mule", variable=self.role_var, value="mule").pack(side=tk.LEFT)
        
        # Help text
        help_text = "Worker: Initiates trades and offers coins. Mule: Accepts trades and waits for coins."
        ttk.Label(parent, text=help_text, style='Info.TLabel').grid(row=1, column=0, columnspan=2, pady=5)
            
    def create_generic_parameters_tab(self, parent):
        """Create generic parameters tab for non-GE plans."""
        ttk.Label(parent, text="Parameters:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.NW, pady=(0, 5))
        
        params_frame = ttk.Frame(parent)
        params_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 0))
        params_frame.columnconfigure(0, weight=1)
        
        self.params_tree = ttk.Treeview(params_frame, columns=('key', 'value'), show='headings', height=6)
        self.params_tree.heading('key', text='Key')
        self.params_tree.heading('value', text='Value')
        self.params_tree.column('key', width=150)
        self.params_tree.column('value', width=200)
        self.params_tree.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        params_btn_frame = ttk.Frame(params_frame)
        params_btn_frame.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(params_btn_frame, text="Add", command=self.add_generic_param).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(params_btn_frame, text="Edit", command=self.edit_generic_param).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(params_btn_frame, text="Remove", command=self.remove_generic_param).pack(side=tk.LEFT)
        
        # Load existing generic parameters
        for key, value in self.plan_entry['params'].get('generic', {}).items():
            self.params_tree.insert('', 'end', values=(key, value))
    
    def add_stop_item(self):
        """Add a new stop item."""
        self.edit_item_dialog("Add Stop Item", "", 0, self.items_tree)
    
    def edit_stop_item(self):
        """Edit selected stop item."""
        selection = self.items_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to edit.")
            return
        
        item = self.items_tree.item(selection[0])
        values = item['values']
        self.edit_item_dialog("Edit Stop Item", values[0], int(values[1]), self.items_tree, selection[0])
    
    def remove_stop_item(self):
        """Remove selected stop item."""
        selection = self.items_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to remove.")
            return
        
        self.items_tree.delete(selection[0])
    
    def add_buy_item(self):
        """Add a new buy item."""
        self.edit_ge_item_dialog("Add Buy Item", "", 0, 0, 0, self.buy_tree)
    
    def edit_buy_item(self):
        """Edit selected buy item."""
        selection = self.buy_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to edit.")
            return
        
        item = self.buy_tree.item(selection[0])
        values = item['values']
        self.edit_ge_item_dialog("Edit Buy Item", values[0], int(values[1]), int(values[2]), int(values[3]), self.buy_tree, selection[0])
    
    def remove_buy_item(self):
        """Remove selected buy item."""
        selection = self.buy_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to remove.")
            return
        
        self.buy_tree.delete(selection[0])
    
    def add_sell_item(self):
        """Add a new sell item."""
        self.edit_ge_item_dialog("Add Sell Item", "", 0, 0, 0, self.sell_tree)
    
    def edit_sell_item(self):
        """Edit selected sell item."""
        selection = self.sell_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to edit.")
            return
        
        item = self.sell_tree.item(selection[0])
        values = item['values']
        self.edit_ge_item_dialog("Edit Sell Item", values[0], int(values[1]), int(values[2]), int(values[3]), self.sell_tree, selection[0])
    
    def remove_sell_item(self):
        """Remove selected sell item."""
        selection = self.sell_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an item to remove.")
            return
        
        self.sell_tree.delete(selection[0])
    
    def add_generic_param(self):
        """Add a new generic parameter."""
        self.edit_generic_dialog("Add Parameter", "", "", self.params_tree)
    
    def edit_generic_param(self):
        """Edit selected generic parameter."""
        selection = self.params_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a parameter to edit.")
            return
        
        item = self.params_tree.item(selection[0])
        values = item['values']
        self.edit_generic_dialog("Edit Parameter", values[0], values[1], self.params_tree, selection[0])
    
    def remove_generic_param(self):
        """Remove selected generic parameter."""
        selection = self.params_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a parameter to remove.")
            return
        
        self.params_tree.delete(selection[0])
    
    def edit_item_dialog(self, title, name, qty, tree, item_id=None):
        """Edit stop item dialog."""
        dialog = tk.Toplevel(self.window)
        dialog.title(title)
        dialog.geometry("300x150")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
        y = (dialog.winfo_screenheight() // 2) - (150 // 2)
        dialog.geometry(f"300x150+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Item Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(frame, text="Quantity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        qty_var = tk.StringVar(value=str(qty))
        qty_spinbox = ttk.Spinbox(frame, from_=0, to=999999, textvariable=qty_var, width=10)
        qty_spinbox.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        def ok_clicked():
            if not name_var.get().strip():
                messagebox.showerror("Error", "Item name cannot be empty.")
                return
            
            try:
                qty_val = int(qty_var.get())
                if qty_val < 0:
                    messagebox.showerror("Error", "Quantity must be non-negative.")
                    return
            except ValueError:
                messagebox.showerror("Error", "Quantity must be a valid number.")
                return
            
            if item_id:
                tree.item(item_id, values=(name_var.get().strip(), qty_val))
            else:
                tree.insert('', 'end', values=(name_var.get().strip(), qty_val))
            
            dialog.destroy()
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def edit_ge_item_dialog(self, title, name, qty, bumps, price, tree, item_id=None):
        """Edit GE item dialog."""
        dialog = tk.Toplevel(self.window)
        dialog.title(title)
        dialog.geometry("350x200")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"350x200+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(frame, text="Quantity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        qty_var = tk.StringVar(value=str(qty) if qty is not None else "0")
        qty_spinbox = ttk.Spinbox(frame, from_=-1, to=999999, textvariable=qty_var, width=10)
        qty_spinbox.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Add help text for -1
        help_label = ttk.Label(frame, text="(-1 = sell all)", style='Info.TLabel')
        help_label.grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=5)
        
        ttk.Label(frame, text="Bumps:").grid(row=2, column=0, sticky=tk.W, pady=5)
        bumps_var = tk.StringVar(value=str(bumps) if bumps is not None else "0")
        bumps_spinbox = ttk.Spinbox(frame, from_=0, to=999, textvariable=bumps_var, width=10)
        bumps_spinbox.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(frame, text="Set Price:").grid(row=3, column=0, sticky=tk.W, pady=5)
        price_var = tk.StringVar(value=str(price) if price is not None else "0")
        price_spinbox = ttk.Spinbox(frame, from_=0, to=999999999, textvariable=price_var, width=10)
        price_spinbox.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        def ok_clicked():
            if not name_var.get().strip():
                messagebox.showerror("Error", "Item name cannot be empty.")
                return
            
            try:
                qty_str = qty_var.get().strip()
                bumps_str = bumps_var.get().strip()
                price_str = price_var.get().strip()
                
                qty_val = int(qty_str) if qty_str else 0
                bumps_val = int(bumps_str) if bumps_str else 0
                price_val = int(price_str) if price_str else 0
                
                if qty_val < -1 or bumps_val < 0 or price_val < 0:
                    messagebox.showerror("Error", "Quantity must be -1 or non-negative, other values must be non-negative.")
                    return
            except ValueError:
                messagebox.showerror("Error", "All values must be valid numbers.")
                return
            
            if item_id:
                tree.item(item_id, values=(name_var.get().strip(), qty_val, bumps_val, price_val))
            else:
                tree.insert('', 'end', values=(name_var.get().strip(), qty_val, bumps_val, price_val))
            
            dialog.destroy()
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def edit_generic_dialog(self, title, key, value, tree, item_id=None):
        """Edit generic parameter dialog."""
        dialog = tk.Toplevel(self.window)
        dialog.title(title)
        dialog.geometry("300x150")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
        y = (dialog.winfo_screenheight() // 2) - (150 // 2)
        dialog.geometry(f"300x150+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        key_var = tk.StringVar(value=key)
        key_entry = ttk.Entry(frame, textvariable=key_var, width=20)
        key_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(frame, text="Value:").grid(row=1, column=0, sticky=tk.W, pady=5)
        value_var = tk.StringVar(value=value)
        value_entry = ttk.Entry(frame, textvariable=value_var, width=20)
        value_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        def ok_clicked():
            if not key_var.get().strip():
                messagebox.showerror("Error", "Key cannot be empty.")
                return
            
            if item_id:
                tree.item(item_id, values=(key_var.get().strip(), value_var.get().strip()))
            else:
                tree.insert('', 'end', values=(key_var.get().strip(), value_var.get().strip()))
            
            dialog.destroy()
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def ok_clicked(self):
        """Handle OK button click."""
        try:
            # Collect rules with proper validation
            max_time_str = self.max_time_var.get().strip()
            max_minutes = int(max_time_str) if max_time_str and max_time_str != 'None' else 0
            
            stop_skill_str = self.stop_skill_var.get().strip()
            stop_skill = stop_skill_str if stop_skill_str and stop_skill_str != 'None' else None
            
            stop_skill_level_str = self.stop_skill_level_var.get().strip()
            stop_skill_level = int(stop_skill_level_str) if stop_skill_level_str and stop_skill_level_str != 'None' else 0
            
            total_level_str = self.total_level_var.get().strip()
            total_level = int(total_level_str) if total_level_str and total_level_str != 'None' else 0
            
            # Collect stop items
            stop_items = []
            for item in self.items_tree.get_children():
                values = self.items_tree.item(item)['values']
                if values[0] and values[1]:  # Check both name and quantity exist
                    stop_items.append({'name': values[0], 'qty': int(values[1])})
            
            # Collect parameters based on plan type
            params = {}
            
            if 'ge' == self.plan_entry['name'].lower():
                # Collect buy items
                buy_items = []
                for item in self.buy_tree.get_children():
                    values = self.buy_tree.item(item)['values']
                    if values[0]:  # Only process if name exists
                        buy_items.append({
                            'name': values[0],
                            'quantity': int(values[1]) if values[1] else 0,
                            'bumps': int(values[2]) if values[2] else 0,
                            'set_price': int(values[3]) if values[3] else 0
                        })
                
                # Collect sell items
                sell_items = []
                for item in self.sell_tree.get_children():
                    values = self.sell_tree.item(item)['values']
                    if values[0]:  # Only process if name exists
                        sell_items.append({
                            'name': values[0],
                            'quantity': int(values[1]) if values[1] else 0,
                            'bumps': int(values[2]) if values[2] else 0,
                            'set_price': int(values[3]) if values[3] else 0
                        })
                
                # Compile GE strings for backend compatibility
                compiled_buy = ",".join([f"{item['name']}:{item['quantity']}:{item['bumps']}:{item['set_price']}" for item in buy_items])
                compiled_sell = ",".join([f"{item['name']}:{item['quantity']}:{item['bumps']}:{item['set_price']}" for item in sell_items])
                
                params = {
                    'buy_items': buy_items,
                    'sell_items': sell_items,
                    'compiled_ge_buy': compiled_buy,
                    'compiled_ge_sell': compiled_sell
                }
            elif self.plan_entry['name'] == 'ge_trade':
                # Collect role parameter for ge_trade
                role = getattr(self, 'role_var', None)
                if role:
                    params = {'role': role.get()}
                else:
                    params = {'role': 'worker'}  # Default
            else:
                # Collect generic parameters
                generic = {}
                for item in self.params_tree.get_children():
                    values = self.params_tree.item(item)['values']
                    if values[0]:  # Only process if key exists
                        generic[values[0]] = values[1]
                
                params = {'generic': generic}
            
            # Update plan entry
            self.plan_entry['rules'] = {
                'max_minutes': max_minutes if max_minutes > 0 else None,
                'stop_skill': stop_skill,
                'stop_skill_level': stop_skill_level if stop_skill and stop_skill_level > 0 else None,
                'total_level': total_level if total_level > 0 else None,
                'stop_items': stop_items
            }
            self.plan_entry['params'] = params
            
            self.result = self.plan_entry
            self.window.destroy()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number format: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    
    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.result = None
        self.window.destroy()


class SimpleRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Recorder - Plan Runner")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Configure style
        self.setup_styles()
        
        # Variables
        self.selected_plans = []
        self.pause_between_plans = tk.BooleanVar(value=True)
        self.pause_duration = tk.IntVar(value=5)
        self.session_dir = tk.StringVar(value="D:\\bots\\exports\\inst_2")
        self.port = tk.IntVar(value=17002)
        self.is_running = False
        self.current_plan_index = 0
        
        # RuneLite launcher variables
        self.selected_credentials = []
        self.runelite_process = None
        self.build_maven = tk.BooleanVar(value=True)  # Default to building Maven
        
        # Instance management
        self.instance_tabs = {}  # Dictionary to store instance tab references
        self.instance_ports = {}  # Dictionary to map instance names to ports
        self.detected_clients = {}  # Dictionary to track detected RuneLite clients
        self.client_detection_running = False
        
        # Base completion patterns for quest plans
        self.base_completion_patterns = [
            'phase: done',  # Quest plans completion
            'status: done', 'plan done', 'execution done',
            'plan completed', 'execution completed', 'finished successfully',
            'task completed', 'mission accomplished', 'objective complete'
        ]
        
        # Create GUI
        self.create_widgets()
        
        # Center window
    
    def _get_completion_patterns_for_plan(self, plan_name):
        """Get completion patterns based on the plan type."""
        # Utility plans that can be run standalone
        utility_plans = ['ge', 'bank_plan', 'attack_npcs']
        
        if plan_name in utility_plans:
            # For utility plans run directly, detect their completion phases
            if plan_name == 'ge':
                return self.base_completion_patterns + ['phase: complete']
            elif plan_name == 'bank_plan':
                return self.base_completion_patterns + ['phase: setup_complete']
            elif plan_name == 'attack_npcs':
                return self.base_completion_patterns + ['phase: complete']
        else:
            # For main plans (quests, farming), only detect quest completion
            return self.base_completion_patterns

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Simple Recorder Plan Runner", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(1, weight=1)
        
        # RuneLite Launcher tab
        runelite_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(runelite_tab, text="RuneLite Launcher")
        runelite_tab.columnconfigure(0, weight=1)
        runelite_tab.columnconfigure(1, weight=1)
        
        # RuneLite configuration
        ttk.Label(runelite_tab, text="RuneLite Instance Launcher", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Configuration frame
        runelite_config = ttk.Frame(runelite_tab)
        runelite_config.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        runelite_config.columnconfigure(1, weight=1)
        
        # Base port
        ttk.Label(runelite_config, text="Base Port:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.base_port = tk.IntVar(value=17000)
        ttk.Entry(runelite_config, textvariable=self.base_port, width=8).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Delay between launches
        ttk.Label(runelite_config, text="Delay (seconds):", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.launch_delay = tk.IntVar(value=0)
        ttk.Entry(runelite_config, textvariable=self.launch_delay, width=8).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Instance count display (read-only)
        ttk.Label(runelite_config, text="Instances to Launch:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.instance_count_label = ttk.Label(runelite_config, text="0", style='Info.TLabel')
        self.instance_count_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Credential selection
        ttk.Label(runelite_tab, text="Credential Files:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(10, 3))
        
        # Available credentials (left)
        cred_frame = ttk.Frame(runelite_tab)
        cred_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5), padx=(0, 5))
        cred_frame.columnconfigure(0, weight=1)
        
        self.credentials_listbox = tk.Listbox(cred_frame, selectmode=tk.MULTIPLE, height=8)
        self.credentials_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        cred_scrollbar = ttk.Scrollbar(cred_frame, orient=tk.VERTICAL, command=self.credentials_listbox.yview)
        cred_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.credentials_listbox.configure(yscrollcommand=cred_scrollbar.set)
        
        # Selected credentials (right)
        selected_cred_frame = ttk.Frame(runelite_tab)
        selected_cred_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(5, 0))
        selected_cred_frame.columnconfigure(0, weight=1)
        
        self.selected_credentials_listbox = tk.Listbox(selected_cred_frame, height=8)
        self.selected_credentials_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        selected_cred_scrollbar = ttk.Scrollbar(selected_cred_frame, orient=tk.VERTICAL, command=self.selected_credentials_listbox.yview)
        selected_cred_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.selected_credentials_listbox.configure(yscrollcommand=selected_cred_scrollbar.set)
        
        # Credential controls
        cred_controls = ttk.Frame(runelite_tab)
        cred_controls.grid(row=4, column=0, columnspan=2, pady=(5, 10))
        
        ttk.Button(cred_controls, text="Add Selected", command=self.add_credential).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(cred_controls, text="Remove Selected", command=self.remove_credential).grid(row=0, column=1, padx=3)
        ttk.Button(cred_controls, text="Clear All", command=self.clear_credentials).grid(row=0, column=2, padx=3)
        ttk.Button(cred_controls, text="↑", command=self.move_credential_up).grid(row=0, column=3, padx=3)
        ttk.Button(cred_controls, text="↓", command=self.move_credential_down).grid(row=0, column=4, padx=3)
        
        # Launch controls
        launch_controls = ttk.Frame(runelite_tab)
        launch_controls.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        # Build Maven checkbox
        self.build_maven_checkbox = ttk.Checkbutton(launch_controls, text="Build Maven", 
                                                   variable=self.build_maven)
        self.build_maven_checkbox.grid(row=0, column=0, padx=(0, 10))
        
        self.launch_button = ttk.Button(launch_controls, text="Launch RuneLite Instances", 
                                      command=self.launch_runelite, style='Action.TButton')
        self.launch_button.grid(row=0, column=1, padx=(0, 10))
        
        self.stop_runelite_button = ttk.Button(launch_controls, text="Stop All Instances", 
                                            command=self.stop_runelite, style='Danger.TButton')
        self.stop_runelite_button.grid(row=0, column=2, padx=5)
        
        # Client detection controls
        detection_controls = ttk.Frame(runelite_tab)
        detection_controls.grid(row=6, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Label(detection_controls, text="Client Detection:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        detection_buttons = ttk.Frame(detection_controls)
        detection_buttons.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.start_detection_button = ttk.Button(detection_buttons, text="Start Auto-Detection", 
                                               command=self.start_client_detection, style='Action.TButton')
        self.start_detection_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_detection_button = ttk.Button(detection_buttons, text="Stop Auto-Detection", 
                                             command=self.stop_client_detection, style='Danger.TButton')
        self.stop_detection_button.grid(row=0, column=1, padx=5)
        
        self.detect_now_button = ttk.Button(detection_buttons, text="Detect Now", 
                                         command=self.detect_running_clients, style='Info.TButton')
        self.detect_now_button.grid(row=0, column=2, padx=5)
        
        # Test detection button (for debugging)
        self.test_detection_button = ttk.Button(detection_buttons, text="Test Detection", 
                                             command=self.test_client_detection, style='Info.TButton')
        self.test_detection_button.grid(row=0, column=3, padx=5)
        
        # Detection status
        self.detection_status_label = ttk.Label(detection_controls, text="Auto-detection: Stopped", style='Info.TLabel')
        self.detection_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Populate credentials
        self.populate_credentials()
        
        # Output tab
        output_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(output_tab, text="Output")
        output_tab.columnconfigure(0, weight=1)
        output_tab.rowconfigure(0, weight=1)
        
        text_frame = ttk.Frame(output_tab)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(text_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
    
    def setup_styles(self):
        """Configure the GUI styles and colors."""
        style = ttk.Style()
        
        # Configure colors - gentle, comfortable scheme
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 16, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 10, 'bold'),
                       foreground='#34495e')
        
        style.configure('Info.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#7f8c8d')
        
        style.configure('Success.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#27ae60')
        
        style.configure('Error.TLabel',
                       font=('Segoe UI', 9),
                       foreground='#e74c3c')
        
        # Configure buttons
        style.configure('Action.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Danger.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def populate_plans(self):
        """Populate the plans listbox with available plans."""
        for plan_id, plan_class in AVAILABLE_PLANS.items():
            # Get the label from the plan class
            label = getattr(plan_class, 'label', 'No description')
            display_text = f"{plan_id} - {label}"
            self.plans_listbox.insert(tk.END, display_text)
    
    
    
    def browse_directory(self):
        """Open directory browser for session directory."""
        directory = filedialog.askdirectory(initialdir=self.session_dir.get())
        if directory:
            self.session_dir.set(directory)
    
    def log_message(self, message, level='info'):
        """Add a message to the log output."""
        self.log_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding based on level
        if level == 'error':
            color = '#e74c3c'
        elif level == 'success':
            color = '#27ae60'
        elif level == 'warning':
            color = '#f39c12'
        else:
            color = '#34495e'
        
        # Insert message
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Apply color to the last line
        start_line = self.log_text.index("end-2l")
        end_line = self.log_text.index("end-1l")
        self.log_text.tag_add(level, start_line, end_line)
        self.log_text.tag_config(level, foreground=color)
        
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)
    
    def log_message_to_instance(self, instance_name, message, level='info'):
        """Add a message to a specific instance's log output."""
        if instance_name not in self.instance_tabs:
            return
        
        instance_tab = self.instance_tabs[instance_name]
        log_text = instance_tab.log_text
        
        log_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding based on level
        if level == 'error':
            color = '#e74c3c'
        elif level == 'success':
            color = '#27ae60'
        elif level == 'warning':
            color = '#f39c12'
        else:
            color = '#34495e'
        
        # Insert message
        log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Apply color to the last line
        start_line = log_text.index("end-2l")
        end_line = log_text.index("end-1l")
        log_text.tag_add(level, start_line, end_line)
        log_text.tag_config(level, foreground=color)
        
        log_text.config(state=tk.DISABLED)
        log_text.see(tk.END)
    
    def populate_credentials(self):
        """Populate the credentials listbox with available credential files."""
        credentials_dir = Path("D:/repos/bot_runelite_IL/credentials")
        if credentials_dir.exists():
            for cred_file in sorted(credentials_dir.glob("*.properties")):
                self.credentials_listbox.insert(tk.END, cred_file.name)
        else:
            self.log_message("Credentials directory not found: D:/repos/bot_runelite_IL/credentials", 'error')
    
    def add_credential(self):
        """Add selected credentials to the selected list."""
        selection = self.credentials_listbox.curselection()
        for index in selection:
            cred_name = self.credentials_listbox.get(index)
            if cred_name not in self.selected_credentials:
                self.selected_credentials.append(cred_name)
        self.update_selected_credentials_display()
    
    def remove_credential(self):
        """Remove selected credentials from the selected list."""
        selection = self.selected_credentials_listbox.curselection()
        for index in reversed(selection):  # Reverse to maintain indices
            self.selected_credentials.pop(index)
        self.update_selected_credentials_display()
    
    def clear_credentials(self):
        """Clear all selected credentials."""
        self.selected_credentials = []
        self.update_selected_credentials_display()
    
    def move_credential_up(self):
        """Move selected credential up in the order."""
        selection = self.selected_credentials_listbox.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            # Swap with previous item
            item = self.selected_credentials.pop(index)
            self.selected_credentials.insert(index - 1, item)
            self.update_selected_credentials_display()
            self.selected_credentials_listbox.selection_set(index - 1)
    
    def move_credential_down(self):
        """Move selected credential down in the order."""
        selection = self.selected_credentials_listbox.curselection()
        if selection and selection[0] < len(self.selected_credentials) - 1:
            index = selection[0]
            # Swap with next item
            item = self.selected_credentials.pop(index)
            self.selected_credentials.insert(index + 1, item)
            self.update_selected_credentials_display()
            self.selected_credentials_listbox.selection_set(index + 1)
    
    def update_selected_credentials_display(self):
        """Update the selected credentials display."""
        self.selected_credentials_listbox.delete(0, tk.END)
        for i, cred_name in enumerate(self.selected_credentials):
            self.selected_credentials_listbox.insert(tk.END, f"{i+1}. {cred_name}")
        
        # Update instance count display
        self.instance_count_label.config(text=str(len(self.selected_credentials)))

    def clear_all_instance_tabs(self):
        """Remove all instance tabs."""
        usernames_to_remove = list(self.instance_tabs.keys())
        for username in usernames_to_remove:
            self.remove_instance_tab(username)
        self.log_message(f"Cleared all instance tabs", 'info')
    
    def create_instance_tab(self, username, port):
        """Create a new tab for a RuneLite instance."""
        # Check if tab already exists
        if username in self.instance_tabs:
            self.log_message(f"Instance tab already exists: {username}", 'info')
            # Just update the port if it changed
            self.instance_ports[username] = port
            return
        
        self.log_message(f"Creating instance tab: {username}", 'info')
        
        # Create the main instance tab
        instance_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(instance_tab, text=username)
        instance_tab.columnconfigure(0, weight=1)
        instance_tab.rowconfigure(0, weight=1)
        
        # Create sub-notebook for Plan Runner and Output
        sub_notebook = ttk.Notebook(instance_tab)
        sub_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Store reference
        self.instance_tabs[username] = instance_tab
        self.instance_ports[username] = port
        
        self.log_message(f"Instance tab created and stored: {username}", 'info')
        
        # Create Plan Runner sub-tab
        plan_runner_tab = ttk.Frame(sub_notebook, padding="10")
        sub_notebook.add(plan_runner_tab, text="Plan Runner")
        plan_runner_tab.columnconfigure(0, weight=1)
        plan_runner_tab.columnconfigure(1, weight=1)
        
        # Create Output sub-tab
        output_tab = ttk.Frame(sub_notebook, padding="5")
        sub_notebook.add(output_tab, text="Output")
        output_tab.columnconfigure(0, weight=1)
        output_tab.rowconfigure(0, weight=1)
        
        # Create Statistics sub-tab
        stats_tab = ttk.Frame(sub_notebook, padding="10")
        sub_notebook.add(stats_tab, text="Statistics")
        stats_tab.columnconfigure(0, weight=1)
        stats_tab.rowconfigure(0, weight=1)
        
        # Store sub-tab references
        instance_tab.plan_runner_tab = plan_runner_tab
        instance_tab.output_tab = output_tab
        instance_tab.stats_tab = stats_tab
        instance_tab.sub_notebook = sub_notebook
        
        # Plan Runner sub-tab content - Redesigned with structured widgets
        # Top row: Config + Controls + Save/Load
        top_row = ttk.Frame(plan_runner_tab)
        top_row.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        top_row.columnconfigure(0, weight=1)
        top_row.columnconfigure(1, weight=1)
        
        # Left side: Session config
        config_frame = ttk.Frame(top_row)
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Session Directory
        ttk.Label(config_frame, text="Session Dir:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        dir_frame = ttk.Frame(config_frame)
        dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        dir_frame.columnconfigure(0, weight=1)
        
        session_dir = tk.StringVar(value=f"D:\\bots\\exports\\{username.lower()}")
        dir_entry = ttk.Entry(dir_frame, textvariable=session_dir, width=25)
        dir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 3))
        ttk.Button(dir_frame, text="Browse", command=lambda: self.browse_directory_for_instance(username, session_dir)).grid(row=0, column=1)
        
        # Port (read-only for instances)
        ttk.Label(config_frame, text="Port:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        port_label = ttk.Label(config_frame, text=str(port), style='Info.TLabel')
        port_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Right side: Controls + Save/Load
        controls_frame = ttk.Frame(top_row)
        controls_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Control buttons
        ttk.Button(controls_frame, text="Start Plans", command=lambda: self.start_plans_for_instance(username, session_dir.get(), port)).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(controls_frame, text="Stop", command=lambda: self.stop_plans_for_instance(username)).grid(row=0, column=1, padx=3)
        
        # Pause between plans checkbox
        pause_var = tk.BooleanVar(value=False)
        pause_checkbox = ttk.Checkbutton(controls_frame, text="Pause between plans", variable=pause_var)
        pause_checkbox.grid(row=0, column=2, padx=(10, 0))
        
        # Store pause variable in instance tab
        instance_tab.pause_var = pause_var
        
        # Save/Load buttons
        ttk.Button(controls_frame, text="Save Sequence...", command=lambda: self.save_sequence_for_instance(username)).grid(row=1, column=0, padx=(0, 3), pady=(5, 0))
        ttk.Button(controls_frame, text="Load Sequence...", command=lambda: self.load_sequence_for_instance(username)).grid(row=1, column=1, padx=3, pady=(5, 0))
        
        # Main content: Split into left (plans) and right (details)
        main_content = ttk.Frame(plan_runner_tab)
        main_content.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        main_content.columnconfigure(0, weight=1)
        main_content.columnconfigure(1, weight=1)
        plan_runner_tab.rowconfigure(1, weight=1)
        
        # Left side: Plan selection
        left_frame = ttk.LabelFrame(main_content, text="Plan Selection", padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        left_frame.rowconfigure(3, weight=1)
        
        # Available plans
        ttk.Label(left_frame, text="Available Plans:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        available_listbox = tk.Listbox(left_frame, height=6)
        available_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Populate available plans
        for plan_id, plan_class in AVAILABLE_PLANS.items():
            # Get label from plan class if it has one, otherwise use the plan_id
            label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
            available_listbox.insert(tk.END, f"{label} ({plan_id})")
        
        # Selected plans
        ttk.Label(left_frame, text="Selected Plans:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        selected_listbox = tk.Listbox(left_frame, height=6)
        selected_listbox.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Plan controls
        plan_controls = ttk.Frame(left_frame)
        plan_controls.grid(row=4, column=0, pady=(0, 5))
        
        ttk.Button(plan_controls, text="Add →", command=lambda: self.add_plan_to_selection(username, available_listbox, selected_listbox)).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(plan_controls, text="Remove ←", command=lambda: self.remove_plan_from_selection(username, selected_listbox)).grid(row=0, column=1, padx=3)
        ttk.Button(plan_controls, text="Move Up", command=lambda: self.move_plan_up(username, selected_listbox)).grid(row=0, column=2, padx=3)
        ttk.Button(plan_controls, text="Move Down", command=lambda: self.move_plan_down(username, selected_listbox)).grid(row=0, column=3, padx=3)
        
        # Right side: Details panel
        right_frame = ttk.LabelFrame(main_content, text="Plan Details", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Details controls
        details_controls = ttk.Frame(right_frame)
        details_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(details_controls, text="Edit Parameters...", command=lambda: self.edit_plan_parameters(username, selected_listbox)).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(details_controls, text="Clear Parameters", command=lambda: self.clear_plan_parameters(username, selected_listbox)).grid(row=0, column=1, padx=3)
        ttk.Button(details_controls, text="Clear Rules", command=lambda: self.clear_plan_rules(username, selected_listbox)).grid(row=0, column=2, padx=3)
        
        # Details display
        details_frame = ttk.Frame(right_frame)
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        details_frame.rowconfigure(1, weight=1)
        
        # Rules treeview
        ttk.Label(details_frame, text="Rules:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        rules_tree = ttk.Treeview(details_frame, show='tree', height=4)
        rules_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Parameters treeview
        ttk.Label(details_frame, text="Parameters:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 2))
        params_tree = ttk.Treeview(details_frame, show='tree', height=4)
        params_tree.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind selection change to update details
        selected_listbox.bind('<<ListboxSelect>>', lambda e: self.update_plan_details(username, selected_listbox, rules_tree, params_tree))
        
        # Status display
        status_frame = ttk.Frame(plan_runner_tab)
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Status:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        instance_tab.status_label = ttk.Label(status_frame, text="Ready", style='Info.TLabel')
        instance_tab.status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Progress bar
        instance_tab.progress = ttk.Progressbar(status_frame, mode='determinate')
        instance_tab.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Store references for this instance
        instance_tab.available_listbox = available_listbox
        instance_tab.selected_listbox = selected_listbox
        instance_tab.session_dir = session_dir
        instance_tab.rules_tree = rules_tree
        instance_tab.params_tree = params_tree
        instance_tab.plan_entries = []  # List of PlanEntry objects
        instance_tab.is_running = False
        instance_tab.current_plan_index = 0
        instance_tab.current_process = None  # Track the current subprocess
        
        # Output tab content
        text_frame = ttk.Frame(output_tab)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        instance_tab.log_text = tk.Text(text_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        instance_tab.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=instance_tab.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        instance_tab.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Statistics tab content
        stats_content_frame = ttk.Frame(stats_tab)
        stats_content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_content_frame.columnconfigure(0, weight=1)
        stats_content_frame.rowconfigure(1, weight=1)
        
        # Current Status section
        status_frame = ttk.LabelFrame(stats_content_frame, text="Current Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # Current plan
        ttk.Label(status_frame, text="Current Plan:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        instance_tab.current_plan_label = ttk.Label(status_frame, text="None", style='Info.TLabel')
        instance_tab.current_plan_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Current phase
        ttk.Label(status_frame, text="Current Phase:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        instance_tab.current_phase_label = ttk.Label(status_frame, text="None", style='Info.TLabel')
        instance_tab.current_phase_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Runtime
        ttk.Label(status_frame, text="Runtime:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        instance_tab.runtime_label = ttk.Label(status_frame, text="00:00:00", style='Info.TLabel')
        instance_tab.runtime_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Rules section
        rules_frame = ttk.LabelFrame(stats_content_frame, text="Active Rules", padding="10")
        rules_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        rules_frame.columnconfigure(0, weight=1)
        rules_frame.rowconfigure(0, weight=1)
        
        # Rules treeview
        instance_tab.stats_rules_tree = ttk.Treeview(rules_frame, show='tree', height=8)
        instance_tab.stats_rules_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize with default values
        self.update_statistics_display(username)
        
        return instance_tab
    
    def update_statistics_display(self, instance_name):
        """Update the statistics display for an instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab:
            return
        
        # Update current plan
        if hasattr(instance_tab, 'current_plan_label'):
            current_plan = getattr(instance_tab, 'current_plan_name', 'None')
            instance_tab.current_plan_label.config(text=current_plan)
        
        # Update current phase
        if hasattr(instance_tab, 'current_phase_label'):
            current_phase = getattr(instance_tab, 'current_phase', 'None')
            instance_tab.current_phase_label.config(text=current_phase)
        
        # Update runtime
        if hasattr(instance_tab, 'runtime_label'):
            start_time = getattr(instance_tab, 'start_time', None)
            if start_time:
                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                runtime_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                instance_tab.runtime_label.config(text=runtime_text)
            else:
                instance_tab.runtime_label.config(text="00:00:00")
        
        # Update rules display
        if hasattr(instance_tab, 'stats_rules_tree'):
            # Clear existing rules
            for item in instance_tab.stats_rules_tree.get_children():
                instance_tab.stats_rules_tree.delete(item)
            
            # Get current plan rules
            current_plan_index = getattr(instance_tab, 'current_plan_index', 0)
            if current_plan_index < len(instance_tab.plan_entries):
                plan_entry = instance_tab.plan_entries[current_plan_index]
                rules = plan_entry.get('rules', {})
                
                # Add rules to tree
                if rules.get('max_minutes'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Time Limit: {rules['max_minutes']} minutes")
                if rules.get('stop_skill'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Stop at Skill: {rules['stop_skill']} level {rules.get('stop_skill_level', 0)}")
                if rules.get('total_level'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Total Level: {rules['total_level']}")
                if rules.get('stop_items'):
                    items_node = instance_tab.stats_rules_tree.insert('', 'end', text="Stop with Items:")
                    for item in rules['stop_items']:
                        instance_tab.stats_rules_tree.insert(items_node, 'end', text=f"{item['name']} x{item['qty']}")
                
                if not any(rules.values()):
                    instance_tab.stats_rules_tree.insert('', 'end', text="No rules configured")
    
    def start_statistics_timer(self, instance_name):
        """Start a timer to update statistics display every second."""
        def update_timer():
            if instance_name in self.instance_tabs:
                instance_tab = self.instance_tabs[instance_name]
                if getattr(instance_tab, 'is_running', False):
                    self.update_statistics_display(instance_name)
                    # Schedule next update in 1 second
                    self.root.after(1000, update_timer)
        
        # Start the timer
        update_timer()
    
    def stop_statistics_timer(self, instance_name):
        """Stop the statistics timer for an instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab:
            instance_tab.is_running = False
            # Clear current tracking
            instance_tab.current_plan_name = "None"
            instance_tab.current_phase = "Stopped"
            self.update_statistics_display(instance_name)
    
    def browse_directory_for_instance(self, instance_name, session_dir_var):
        """Browse for directory for a specific instance."""
        directory = filedialog.askdirectory(initialdir=session_dir_var.get())
        if directory:
            session_dir_var.set(directory)
    
    def add_plan_to_instance(self, instance_name, available_listbox, selected_listbox):
        """Add selected plans to instance."""
        selected_indices = available_listbox.curselection()
        for index in reversed(selected_indices):  # Reverse to maintain order
            plan_name = available_listbox.get(index)
            selected_listbox.insert(tk.END, plan_name)
    
    def remove_plan_from_instance(self, instance_name, selected_listbox):
        """Remove selected plans from instance."""
        selected_indices = selected_listbox.curselection()
        for index in reversed(selected_indices):  # Reverse to maintain order
            selected_listbox.delete(index)
    
    def move_plan_up_in_instance(self, instance_name, selected_listbox):
        """Move selected plan up in instance."""
        selected_indices = selected_listbox.curselection()
        if selected_indices and selected_indices[0] > 0:
            index = selected_indices[0]
            item = selected_listbox.get(index)
            selected_listbox.delete(index)
            selected_listbox.insert(index - 1, item)
            selected_listbox.selection_set(index - 1)
    
    def move_plan_down_in_instance(self, instance_name, selected_listbox):
        """Move selected plan down in instance."""
        selected_indices = selected_listbox.curselection()
        if selected_indices and selected_indices[0] < selected_listbox.size() - 1:
            index = selected_indices[0]
            item = selected_listbox.get(index)
            selected_listbox.delete(index)
            selected_listbox.insert(index + 1, item)
            selected_listbox.selection_set(index + 1)
    
    def start_plans_for_instance(self, instance_name, session_dir, port):
        """Start plans for a specific instance."""
        instance_tab = self.instance_tabs[instance_name]
        
        if instance_tab.is_running:
            messagebox.showwarning("Already Running", f"Plans are already running for {instance_name}.")
            return
        
        # Get selected plans
        selected_plans = []
        for i in range(instance_tab.selected_listbox.size()):
            selected_plans.append(instance_tab.selected_listbox.get(i))
        
        if not selected_plans:
            messagebox.showwarning("No Plans Selected", f"Please select at least one plan for {instance_name}.")
            return
        
        # Start plans in a separate thread
        def run_plans():
            try:
                instance_tab.is_running = True
                instance_tab.start_time = time.time()  # Track start time for runtime
                instance_tab.current_plan_index = 0  # Track current plan index
                instance_tab.status_label.config(text="Starting...", style='Info.TLabel')
                instance_tab.progress['maximum'] = len(selected_plans)
                instance_tab.progress['value'] = 0
                
                # Start statistics update timer
                self.start_statistics_timer(instance_name)
                
                self.log_message_to_instance(instance_name, f"Starting execution of {len(selected_plans)} plans", 'info')
                
                for i, plan_name in enumerate(selected_plans):
                    if not instance_tab.is_running:  # Check if stopped
                        break
                    
                    # Update current plan tracking
                    instance_tab.current_plan_index = i
                    instance_tab.current_plan_name = plan_name
                    instance_tab.current_phase = "Starting"
                    
                    instance_tab.status_label.config(text=f"Running: {plan_name}", style='Info.TLabel')
                    instance_tab.progress['value'] = i
                    
                    # Update statistics display
                    self.update_statistics_display(instance_name)
                    
                    self.log_message_to_instance(instance_name, f"Starting plan {i+1}/{len(selected_plans)}: {plan_name}", 'info')
                    
                    # Get rules and parameters from plan entries
                    rules_args = []
                    param_args = []
                    
                    # Find the plan entry for this plan
                    plan_entry = None
                    # Extract plan ID from the display name (format: "Display Name (plan_id)")
                    plan_id = plan_name
                    if '(' in plan_name and ')' in plan_name:
                        plan_id = plan_name.split('(')[-1].rstrip(')')
                        for entry in instance_tab.plan_entries:
                            if entry['name'] == plan_id:
                                plan_entry = entry
                                break
                    
                    if plan_entry:
                        # Extract rules
                        rules = plan_entry.get('rules', {})
                        
                        # Time rule
                        if rules.get('max_minutes'):
                            rules_args.extend(["--max-runtime", str(rules['max_minutes'])])
                        
                        # Skill rule
                        if rules.get('stop_skill') and rules.get('stop_skill_level'):
                            rules_args.extend(["--stop-skill", f"{rules['stop_skill']}:{rules['stop_skill_level']}"])
                        
                        # Total level rule
                        if rules.get('total_level'):
                            rules_args.extend(["--total-level", str(rules['total_level'])])
                        
                        # Item rules
                        for item in rules.get('stop_items', []):
                            rules_args.extend(["--stop-item", f"{item['name']}:{item['qty']}"])
                        
                        # Extract parameters
                        params = plan_entry.get('params', {})
                        
                        # GE parameters
                        if 'compiled_ge_buy' in params and params['compiled_ge_buy']:
                            param_args.extend(["--buy-items", params['compiled_ge_buy']])
                        if 'compiled_ge_sell' in params and params['compiled_ge_sell']:
                            param_args.extend(["--sell-items", params['compiled_ge_sell']])
                        
                        # GE Trade role parameter
                        if plan_id == "ge_trade" and 'role' in params:
                            param_args.extend(["--role", params['role']])
                    
                    # Run the plan
                    success = self.execute_plan_for_instance(instance_name, plan_id, session_dir, port, rules_args, param_args)
                    self.log_message_to_instance(instance_name, f"DEBUG: Plan {plan_id} returned success = {success}", 'info')
                    if not success:
                        self.log_message_to_instance(instance_name, f"Plan {plan_id} failed, stopping execution", 'error')
                        break
                    else:
                        self.log_message_to_instance(instance_name, f"Plan {plan_id} completed successfully, moving to next plan", 'info')
                    
                    # Pause between plans if checkbox is checked and not the last one
                    if i < len(selected_plans) - 1 and instance_tab.is_running:
                        if instance_tab.pause_var.get():
                            instance_tab.status_label.config(text=f"Paused between plans - click to continue", style='Info.TLabel')
                            self.log_message_to_instance(instance_name, "Paused between plans (waiting for user to uncheck pause)", 'info')
                            # Wait until pause is unchecked
                            while instance_tab.pause_var.get() and instance_tab.is_running:
                                time.sleep(1)
                            if instance_tab.is_running:
                                self.log_message_to_instance(instance_name, "Resuming plan execution...", 'info')
                        else:
                            # Just a brief pause for smooth transition
                            instance_tab.status_label.config(text=f"Transitioning to next plan...", style='Info.TLabel')
                            self.log_message_to_instance(instance_name, "Moving to next plan...", 'info')
                            time.sleep(2)
                
                if instance_tab.is_running:
                    instance_tab.status_label.config(text="All plans completed", style='Success.TLabel')
                    instance_tab.progress['value'] = len(selected_plans)
                    self.log_message_to_instance(instance_name, "All plans completed successfully", 'success')
                
            except Exception as e:
                instance_tab.status_label.config(text=f"Error: {str(e)}", style='Error.TLabel')
                self.log_message_to_instance(instance_name, f"Execution error: {str(e)}", 'error')
            finally:
                instance_tab.is_running = False
        
        threading.Thread(target=run_plans, daemon=True).start()

    def execute_plan_for_instance(self, instance_name, plan_name, session_dir, port, rules_args=None, param_args=None, timeout_minutes=None):
        """Execute a specific plan for an instance."""
        try:
            self.log_message_to_instance(instance_name, f"Executing plan: {plan_name}", 'info')
            
            # Extract timeout from rules_args if not provided
            if timeout_minutes is None and rules_args:
                for i, arg in enumerate(rules_args):
                    if arg == "--max-runtime" and i + 1 < len(rules_args):
                        try:
                            timeout_minutes = int(rules_args[i + 1])
                            break
                        except ValueError:
                            pass
            
            if timeout_minutes is not None:
                self.log_message_to_instance(instance_name, f"Using timeout: {timeout_minutes} minutes", 'info')
            else:
                self.log_message_to_instance(instance_name, "No timeout specified - will run until completion", 'info')
            
            # Determine completion patterns based on plan type
            completion_patterns = self._get_completion_patterns_for_plan(plan_name)
            self.log_message_to_instance(instance_name, f"DEBUG: Using completion patterns: {completion_patterns}", 'info')
            
            # Create the command
            cmd = [
                sys.executable, "run_rj_loop.py",
                plan_name,  # Plan name as positional argument
                "--session-dir", session_dir,
                "--port", str(port)
            ]
            
            # Add rules arguments if provided
            if rules_args:
                cmd.extend(rules_args)
            
            # Add parameter arguments if provided
            if param_args:
                cmd.extend(param_args)
            
            self.log_message_to_instance(instance_name, f"Command: {' '.join(cmd)}", 'info')
            
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run the plan with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=env,
                cwd=os.getcwd(),
                bufsize=1,
                universal_newlines=True
            )
            
            # Store the process for stop functionality
            if instance_name in self.instance_tabs:
                self.instance_tabs[instance_name].current_process = process
            
            # Stream output in real-time with completion detection
            plan_completed = False
            start_time = time.time()
            
            while True:
                # Check for timeout only if one is specified
                if timeout_minutes is not None:
                    timeout_seconds = timeout_minutes * 60
                    if time.time() - start_time > timeout_seconds:
                        self.log_message_to_instance(instance_name, f"Plan timeout after {timeout_minutes} minutes, stopping...", 'warning')
                        try:
                            process.terminate()
                            time.sleep(1)
                            if process.poll() is None:
                                process.kill()
                        except Exception as e:
                            self.log_message_to_instance(instance_name, f"Error stopping timed-out process: {e}", 'error')
                        # Treat timeout as successful completion so next plans can run
                        self.log_message_to_instance(instance_name, f"Plan completed after {timeout_minutes} minutes (timeout)", 'success')
                        return True
                # Check if the instance is still running (for stop functionality)
                if instance_name in self.instance_tabs:
                    instance_tab = self.instance_tabs[instance_name]
                    if not getattr(instance_tab, 'is_running', True):
                        self.log_message_to_instance(instance_name, "Stopping plan execution...", 'warning')
                        try:
                            process.terminate()
                            # Give it a moment to terminate gracefully
                            time.sleep(1)
                            if process.poll() is None:
                                process.kill()  # Force kill if it doesn't terminate
                        except Exception as e:
                            self.log_message_to_instance(instance_name, f"Error stopping process: {e}", 'error')
                        self.log_message_to_instance(instance_name, "Plan execution stopped by user", 'warning')
                        return False
                
                # Read output line by line
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    # Remove trailing newline and display in real-time
                    line = output.strip()
                    if line:
                        self.log_message_to_instance(instance_name, line, 'info')
                        
                        # Check for plan completion indicators
                        line_lower = line.lower()
                        if any(completion_indicator in line_lower for completion_indicator in completion_patterns):
                            plan_completed = True
                            self.log_message_to_instance(instance_name, f"Plan completion detected: {line}", 'success')
                            self.log_message_to_instance(instance_name, f"DEBUG: plan_completed = {plan_completed}", 'info')
                            # STOP THE CURRENT PROCESS IMMEDIATELY
                            try:
                                process.terminate()
                                time.sleep(1)
                                if process.poll() is None:
                                    process.kill()
                                self.log_message_to_instance(instance_name, "Process terminated due to completion", 'info')
                            except Exception as e:
                                self.log_message_to_instance(instance_name, f"Error terminating completed process: {e}", 'error')
                            break
                        
                        # Update GUI in real-time
                        self.root.update_idletasks()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Clear the process reference
            if instance_name in self.instance_tabs:
                self.instance_tabs[instance_name].current_process = None
            
            # Check if plan completed successfully
            if plan_completed or return_code == 0:
                self.log_message_to_instance(instance_name, f"Plan {plan_name} completed successfully", 'success')
                return True
            else:
                self.log_message_to_instance(instance_name, f"Plan failed with return code {return_code}", 'error')
                raise Exception(f"Plan failed with return code {return_code}")
                
        except Exception as e:
            self.log_message_to_instance(instance_name, f"Failed to execute plan {plan_name}: {str(e)}", 'error')
            raise Exception(f"Failed to execute plan {plan_name}: {str(e)}")
    
    def launch_runelite(self):
        """Launch RuneLite instances using the PowerShell script."""
        if not self.selected_credentials:
            messagebox.showwarning("No Credentials Selected", "Please select at least one credential file.")
            return
        
        instance_count = len(self.selected_credentials)
        if instance_count <= 0:
            messagebox.showerror("Invalid Configuration", "Instance count must be greater than 0.")
            return
        
        try:
            # Build PowerShell command
            script_path = Path("D:/repos/bot_runelite_IL/launch-runelite.ps1")
            if not script_path.exists():
                messagebox.showerror("Script Not Found", f"RuneLite launcher script not found at {script_path}")
                return
            
            # Create credential files array for PowerShell
            cred_files = "', '".join(self.selected_credentials)
            cred_files = f"@('{cred_files}')"
            
            # Build PowerShell command
            build_maven_flag = "-BuildMaven" if self.build_maven.get() else ""
            ps_cmd = [
                "powershell.exe",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"& '{script_path}' -Count {instance_count} -BasePort {self.base_port.get()} -DelaySeconds {self.launch_delay.get()} -CredentialFiles {cred_files} {build_maven_flag}"
            ]
            
            self.log_message(f"Launching RuneLite instances...", 'info')
            self.log_message(f"Command: {' '.join(ps_cmd)}", 'info')
            
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Launch PowerShell script
            self.runelite_process = subprocess.Popen(
                ps_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_runelite_launch, daemon=True)
            self.monitor_thread.start()
            
            # Create instance tabs immediately after successful launch start
            self.root.after(2000, self.create_instance_tabs)  # Wait 2 seconds for instances to start
            
            self.launch_button.config(state=tk.DISABLED)
            self.log_message("RuneLite launcher started", 'success')
            
        except Exception as e:
            self.log_message(f"Error launching RuneLite: {str(e)}", 'error')
            messagebox.showerror("Launch Error", f"Failed to launch RuneLite instances: {str(e)}")
    
    def monitor_runelite_launch(self):
        """Monitor the RuneLite launcher process."""
        try:
            while True:
                if not self.runelite_process:
                    break
                
                output = self.runelite_process.stdout.readline()
                if output == '' and self.runelite_process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    if line:
                        self.log_message(f"[RuneLite] {line}", 'info')
                        self.root.update_idletasks()
            
            # Process completed
            return_code = self.runelite_process.wait()
            self.runelite_process = None
            
            if return_code == 0:
                self.log_message("RuneLite instances launched successfully", 'success')
            else:
                self.log_message(f"RuneLite launcher failed with return code {return_code}", 'error')
            
        except Exception as e:
            self.log_message(f"Error monitoring RuneLite launcher: {str(e)}", 'error')
        finally:
            self.launch_button.config(state=tk.NORMAL)
    
    def create_instance_tabs(self):
        """Create tabs for each launched RuneLite instance."""
        try:
            self.log_message("Starting to create instance tabs...", 'info')
            base_port = self.base_port.get()
            self.log_message(f"Base port: {base_port}, Selected credentials: {self.selected_credentials}", 'info')
            
            for i, cred_name in enumerate(self.selected_credentials):
                # Extract username from credential filename (remove .properties)
                username = cred_name.replace('.properties', '')
                port = base_port + i
                
                self.log_message(f"Creating tab for {username} on port {port}", 'info')
                
                # Create the instance tab
                self.create_instance_tab(username, port)
                
                self.log_message(f"Created tab for {username} on port {port}", 'info')
            
            # Switch to the first instance tab
            if self.instance_tabs:
                first_instance = list(self.instance_tabs.keys())[0]
                self.log_message(f"Switching to first instance tab: {first_instance}", 'info')
                self.notebook.select(self.instance_tabs[first_instance])
            else:
                self.log_message("No instance tabs were created!", 'error')
                
        except Exception as e:
            self.log_message(f"Error creating instance tabs: {str(e)}", 'error')
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", 'error')
    
    def stop_runelite(self):
        """Stop all RuneLite instances and running plans."""
        try:
            # First stop all running plans
            self.log_message("Stopping all running plans...", 'info')
            self.stop_all_instances()
            
            # Then stop RuneLite instances using the PID file
            pid_file = Path("D:/bots/instances/runelite-pids.txt")
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 1:
                            try:
                                pid = int(parts[0])
                                # Try to terminate the process
                                import psutil
                                process = psutil.Process(pid)
                                process.terminate()
                                self.log_message(f"Stopped RuneLite instance PID {pid}", 'info')
                            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                                pass
                
                # Remove PID file
                pid_file.unlink()
                self.log_message("All RuneLite instances stopped", 'success')
            else:
                self.log_message("No RuneLite instances found to stop", 'warning')
                
        except Exception as e:
            self.log_message(f"Error stopping RuneLite instances: {str(e)}", 'error')
            messagebox.showerror("Stop Error", f"Failed to stop RuneLite instances: {str(e)}")
    
    def stop_plans_for_instance(self, instance_name):
        """Stop plans for a specific instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab:
            instance_tab.is_running = False
            if hasattr(instance_tab, 'current_process') and instance_tab.current_process:
                try:
                    instance_tab.current_process.terminate()
                    instance_tab.current_process = None
                except Exception as e:
                    self.log_message_to_instance(instance_name, f"Error stopping process: {e}", 'error')
            
            # Stop statistics timer
            self.stop_statistics_timer(instance_name)
            
            instance_tab.status_label.config(text="Stopped", style='Warning.TLabel')
            self.log_message_to_instance(instance_name, "Plans stopped", 'info')
    
    def stop_all_instances(self):
        """Stop all running instances and their processes."""
        for instance_name, instance_tab in self.instance_tabs.items():
            if getattr(instance_tab, 'is_running', False):
                self.log_message(f"Stopping instance: {instance_name}", 'info')
                instance_tab.is_running = False
                instance_tab.status_label.config(text="Stopped", style='Error.TLabel')
                # Stop statistics timer
                self.stop_statistics_timer(instance_name)
                
                # Terminate the current process if it exists
                if hasattr(instance_tab, 'current_process') and instance_tab.current_process:
                    try:
                        instance_tab.current_process.terminate()
                        time.sleep(0.5)
                        if instance_tab.current_process.poll() is None:
                            instance_tab.current_process.kill()
                        instance_tab.current_process = None
                    except Exception as e:
                        self.log_message(f"Error stopping {instance_name}: {e}", 'error')
    
    def set_rules_for_instance(self, username, time_var, skill_var, level_var, item_var, quantity_var):
        """Set rules for an instance."""
        if username not in self.instance_tabs:
            self.log_message(f"Instance {username} not found", 'error')
            return
        
        instance_tab = self.instance_tabs[username]
        
        # Get selected plan
        selected_plans = [instance_tab.selected_listbox.get(i) for i in instance_tab.selected_listbox.curselection()]
        if not selected_plans:
            self.log_message("No plan selected", 'warning')
            return
        
        plan_name = selected_plans[0]
        
        # Build rules display
        rules_text = []
        
        # Time rule
        time_value = time_var.get().strip()
        if time_value:
            try:
                duration = int(time_value)
                if duration > 0:
                    rules_text.append(f"Time: {duration} min")
            except ValueError:
                self.log_message(f"Invalid time value: {time_value}", 'error')
        
        # Skill rule
        skill_name = skill_var.get().strip()
        level_value = level_var.get().strip()
        if skill_name and level_value:
            try:
                level = int(level_value)
                if level > 0:
                    rules_text.append(f"Skill: {skill_name} level {level}")
            except ValueError:
                self.log_message(f"Invalid level value: {level_value}", 'error')
        
        # Item rule
        item_name = item_var.get().strip()
        quantity_value = quantity_var.get().strip()
        if item_name and quantity_value:
            try:
                quantity = int(quantity_value)
                if quantity > 0:
                    rules_text.append(f"Item: {quantity} {item_name}")
            except ValueError:
                self.log_message(f"Invalid quantity value: {quantity_value}", 'error')
        
        # Update display
        if rules_text:
            instance_tab.rules_display.config(text=f"Rules for {plan_name}: {', '.join(rules_text)}")
            self.log_message(f"Set rules for {plan_name}: {', '.join(rules_text)}", 'success')
        else:
            instance_tab.rules_display.config(text="No rules set")
            self.log_message("No valid rules to set", 'warning')
    
    def clear_rules_for_instance(self, username, time_var, skill_var, level_var, item_var, quantity_var):
        """Clear rules for an instance."""
        if username not in self.instance_tabs:
            self.log_message(f"Instance {username} not found", 'error')
            return
        
        instance_tab = self.instance_tabs[username]
        
        # Clear form
        time_var.set("")
        skill_var.set("")
        level_var.set("")
        item_var.set("")
        quantity_var.set("")
        
        # Update display
        instance_tab.rules_display.config(text="No rules set")
        self.log_message(f"Cleared rules for {username}", 'info')
    
    def configure_plan_parameters(self, username, selected_listbox):
        """Configure parameters for the selected plan."""
        if username not in self.instance_tabs:
            return
        
        instance_tab = self.instance_tabs[username]
        
        # Get selected plan
        selection = selected_listbox.curselection()
        if not selection:
            # Clear parameters display
            instance_tab.params_display.config(text="Select a plan to configure parameters")
            return
        
        plan_name = selected_listbox.get(selection[0])
        
        # Clear existing parameter widgets
        for widget in instance_tab.params_frame.winfo_children():
            if widget != instance_tab.params_display:
                widget.destroy()
        
        # Configure parameters based on plan
        if plan_name == "ge":
            self.configure_ge_parameters(instance_tab, plan_name)
        elif plan_name == "ge_trade":
            self.configure_ge_trade_parameters(instance_tab, plan_name)
        else:
            instance_tab.params_display.config(text=f"No parameters needed for {plan_name}")
    
    def configure_ge_parameters(self, instance_tab, plan_name):
        """Configure GE trade parameters."""
        instance_tab.params_display.config(text=f"Configure {plan_name} parameters:")
        
        # Buy items
        ttk.Label(instance_tab.params_frame, text="Buy Items (name:qty:bumps:price):", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        buy_var = tk.StringVar(value="")
        buy_entry = ttk.Entry(instance_tab.params_frame, textvariable=buy_var, width=60)
        buy_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Sell items
        ttk.Label(instance_tab.params_frame, text="Sell Items (name:qty:bumps:price):", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        sell_var = tk.StringVar(value="")
        sell_entry = ttk.Entry(instance_tab.params_frame, textvariable=sell_var, width=60)
        sell_entry.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Help text
        help_text = "Format: Trout:50:5:0,Bronze scimitar:1:0:1000 (bumps=0 uses set_price)"
        ttk.Label(instance_tab.params_frame, text=help_text, style='Info.TLabel').grid(row=3, column=0, columnspan=2, pady=2)
        
        # Store references
        instance_tab.plan_params[plan_name] = {
            'buy_var': buy_var,
            'sell_var': sell_var
        }
    
    def configure_ge_trade_parameters(self, instance_tab, plan_name):
        """Configure GE trade plan parameters."""
        instance_tab.params_display.config(text=f"Configure {plan_name} parameters:")
        
        # Role selection
        ttk.Label(instance_tab.params_frame, text="Role:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        role_var = tk.StringVar(value="worker")
        role_frame = ttk.Frame(instance_tab.params_frame)
        role_frame.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        ttk.Radiobutton(role_frame, text="Worker", variable=role_var, value="worker").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(role_frame, text="Mule", variable=role_var, value="mule").pack(side=tk.LEFT)
        
        # Help text
        help_text = "Worker: Initiates trades and offers coins. Mule: Accepts trades and waits for coins."
        ttk.Label(instance_tab.params_frame, text=help_text, style='Info.TLabel').grid(row=2, column=0, columnspan=2, pady=2)
        
        # Store references
        instance_tab.plan_params[plan_name] = {
            'role_var': role_var
        }
    
    def on_closing(self):
        """Handle window closing."""
        # Check if any instances are running
        any_running = any(getattr(instance_tab, 'is_running', False) for instance_tab in self.instance_tabs.values())
        
        if any_running:
            if messagebox.askokcancel("Quit", "Some instances are still running. Do you want to stop all instances and quit?"):
                self.stop_all_instances()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def add_plan_to_selection(self, username, available_listbox, selected_listbox):
        """Add selected plan to the selection list."""
        selection = available_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to add.")
            return
        
        plan_text = available_listbox.get(selection[0])
        plan_id = plan_text.split(' (')[-1].rstrip(')')
        
        # Check if already selected
        for i in range(selected_listbox.size()):
            selected_text = selected_listbox.get(i)
            selected_plan_id = selected_text.split(' (')[-1].rstrip(')')
            if plan_id == selected_plan_id:
                messagebox.showwarning("Already Selected", f"Plan {plan_id} is already in the selection.")
                return
        
        # Add to selected list
        selected_listbox.insert(tk.END, plan_text)
        
        # Create PlanEntry for this plan
        plan_class = AVAILABLE_PLANS[plan_id]
        label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
        plan_entry = PlanEntry(
            name=plan_id,
            label=label,
            rules={'max_minutes': None, 'stop_skill': None, 'stop_items': []},
            params={'generic': {}}
        )
        
        # Get the instance tab and add to plan_entries
        instance_tab = self.instance_tabs.get(username)
        if instance_tab:
            instance_tab.plan_entries.append(plan_entry)
    
    def remove_plan_from_selection(self, username, selected_listbox):
        """Remove selected plan from the selection list."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to remove.")
            return
        
        # Remove from listbox
        selected_listbox.delete(selection[0])
        
        # Remove from plan_entries
        instance_tab = self.instance_tabs.get(username)
        if instance_tab and selection[0] < len(instance_tab.plan_entries):
            del instance_tab.plan_entries[selection[0]]
        
        # Update details display
        self.update_plan_details(username, selected_listbox, 
                               getattr(instance_tab, 'rules_tree', None), 
                               getattr(instance_tab, 'params_tree', None))
    
    def move_plan_up(self, username, selected_listbox):
        """Move selected plan up in the list."""
        selection = selected_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        
        index = selection[0]
        # Swap in listbox
        item = selected_listbox.get(index)
        selected_listbox.delete(index)
        selected_listbox.insert(index - 1, item)
        selected_listbox.selection_set(index - 1)
        
        # Swap in plan_entries
        instance_tab = self.instance_tabs.get(username)
        if instance_tab and index < len(instance_tab.plan_entries):
            instance_tab.plan_entries[index], instance_tab.plan_entries[index - 1] = instance_tab.plan_entries[index - 1], instance_tab.plan_entries[index]
    
    def move_plan_down(self, username, selected_listbox):
        """Move selected plan down in the list."""
        selection = selected_listbox.curselection()
        if not selection or selection[0] == selected_listbox.size() - 1:
            return
        
        index = selection[0]
        # Swap in listbox
        item = selected_listbox.get(index)
        selected_listbox.delete(index)
        selected_listbox.insert(index + 1, item)
        selected_listbox.selection_set(index + 1)
        
        # Swap in plan_entries
        instance_tab = self.instance_tabs.get(username)
        if instance_tab and index < len(instance_tab.plan_entries) - 1:
            instance_tab.plan_entries[index], instance_tab.plan_entries[index + 1] = instance_tab.plan_entries[index + 1], instance_tab.plan_entries[index]
    
    def edit_plan_parameters(self, username, selected_listbox):
        """Edit parameters for the selected plan."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to edit.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab:
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            messagebox.showerror("Error", "Plan entry not found.")
            return
        
        plan_entry = instance_tab.plan_entries[index]
        
        # Open editor
        editor = PlanEditor(self.root, plan_entry, AVAILABLE_PLANS)
        self.root.wait_window(editor.window)
        
        if editor.result:
            # Update the plan entry
            instance_tab.plan_entries[index] = editor.result
            self.update_plan_details(username, selected_listbox, 
                                   getattr(instance_tab, 'rules_tree', None), 
                                   getattr(instance_tab, 'params_tree', None))
    
    def clear_plan_parameters(self, username, selected_listbox):
        """Clear parameters for the selected plan."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to clear parameters for.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if instance_tab:
            index = selection[0]
            if index < len(instance_tab.plan_entries):
                instance_tab.plan_entries[index]['params'] = {'generic': {}}
                self.update_plan_details(username, selected_listbox, 
                                       getattr(instance_tab, 'rules_tree', None), 
                                       getattr(instance_tab, 'params_tree', None))
    
    def clear_plan_rules(self, username, selected_listbox):
        """Clear rules for the selected plan."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to clear rules for.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if instance_tab:
            index = selection[0]
            if index < len(instance_tab.plan_entries):
                instance_tab.plan_entries[index]['rules'] = {'max_minutes': None, 'stop_skill': None, 'stop_items': []}
                self.update_plan_details(username, selected_listbox, 
                                       getattr(instance_tab, 'rules_tree', None), 
                                       getattr(instance_tab, 'params_tree', None))
    
    def update_plan_details(self, username, selected_listbox, rules_tree, params_tree):
        """Update the details panel for the selected plan."""
        if not rules_tree or not params_tree:
            return
            
        selection = selected_listbox.curselection()
        if not selection:
            # Clear details
            for item in rules_tree.get_children():
                rules_tree.delete(item)
            for item in params_tree.get_children():
                params_tree.delete(item)
            return
        
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab:
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            return
        
        plan_entry = instance_tab.plan_entries[index]
        
        # Clear existing details
        for item in rules_tree.get_children():
            rules_tree.delete(item)
        for item in params_tree.get_children():
            params_tree.delete(item)
        
        # Update rules
        rules = plan_entry['rules']
        if rules.get('max_minutes'):
            rules_tree.insert('', 'end', text=f"Max Time: {rules['max_minutes']} minutes")
        if rules.get('stop_skill'):
            rules_tree.insert('', 'end', text=f"Stop at Skill: {rules['stop_skill']}")
        if rules.get('total_level'):
            rules_tree.insert('', 'end', text=f"Total Level: {rules['total_level']}")
        if rules.get('stop_items'):
            items_node = rules_tree.insert('', 'end', text="Stop with Items:")
            for item in rules['stop_items']:
                rules_tree.insert(items_node, 'end', text=f"{item['name']} x{item['qty']}")
        
        # Update parameters
        params = plan_entry['params']
        if 'ge' == plan_entry['name'].lower():
            if params.get('buy_items'):
                buy_node = params_tree.insert('', 'end', text="Buy Items:")
                for item in params['buy_items']:
                    params_tree.insert(buy_node, 'end', text=f"{item['name']} x{item['quantity']} (bumps: {item['bumps']}, price: {item['set_price']})")
            
            if params.get('sell_items'):
                sell_node = params_tree.insert('', 'end', text="Sell Items:")
                for item in params['sell_items']:
                    params_tree.insert(sell_node, 'end', text=f"{item['name']} x{item['quantity']} (bumps: {item['bumps']}, price: {item['set_price']})")
        else:
            if params.get('generic'):
                for key, value in params['generic'].items():
                    params_tree.insert('', 'end', text=f"{key}: {value}")
    
    def save_sequence_for_instance(self, username):
        """Save the current plan sequence to a JSON file."""
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab or not instance_tab.plan_entries:
            messagebox.showwarning("No Plans", "No plans selected to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Plan Sequence"
        )
        
        if not filename:
            return
        
        try:
            sequence_data = {
                "version": 1,
                "session_dir": instance_tab.session_dir.get(),
                "port": self.instance_ports.get(username, 17000),
                "plans": instance_tab.plan_entries
            }
            
            with open(filename, 'w') as f:
                json.dump(sequence_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Plan sequence saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save sequence: {e}")
    
    def load_sequence_for_instance(self, username):
        """Load a plan sequence from a JSON file."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Plan Sequence"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                sequence_data = json.load(f)
            
            # Validate version
            if sequence_data.get('version') != 1:
                messagebox.showerror("Error", "Unsupported sequence file version.")
                return
            
            instance_tab = self.instance_tabs.get(username)
            if not instance_tab:
                return
            
            # Update session directory and port
            if 'session_dir' in sequence_data:
                instance_tab.session_dir.set(sequence_data['session_dir'])
            if 'port' in sequence_data:
                self.instance_ports[username] = sequence_data['port']
            
            # Clear current selection
            instance_tab.selected_listbox.delete(0, tk.END)
            instance_tab.plan_entries.clear()
            
            # Load plans
            if 'plans' in sequence_data:
                for plan_data in sequence_data['plans']:
                    # Add to listbox
                    plan_text = f"{plan_data['label']} ({plan_data['name']})"
                    instance_tab.selected_listbox.insert(tk.END, plan_text)
                    
                    # Add to plan_entries
                    instance_tab.plan_entries.append(plan_data)
            
            # Update details
            self.update_plan_details(username, instance_tab.selected_listbox, 
                                   getattr(instance_tab, 'rules_tree', None), 
                                   getattr(instance_tab, 'params_tree', None))
            
            messagebox.showinfo("Success", f"Plan sequence loaded from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sequence: {e}")
    
    def detect_running_clients(self):
        """Detect running RuneLite clients and create/remove instance tabs accordingly."""
        try:
            
            # Find RuneLite processes
            runelite_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('runelite' in arg.lower() for arg in cmdline):
                            runelite_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for IPC ports in the range we use (17000-17099)
            detected_ports = set()
            for proc in runelite_processes:
                try:
                    # Check if process has network connections on our port range
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            detected_ports.add(conn.laddr.port)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Create instance tabs for detected ports
            for port in detected_ports:
                instance_name = f"detected_{port}"
                if instance_name not in self.instance_tabs:
                    self.log_message(f"Detected RuneLite client on port {port}", 'info')
                    self.create_instance_tab(instance_name, port)
                    self.detected_clients[instance_name] = port
            
            # Remove instance tabs for clients that are no longer running
            clients_to_remove = []
            for instance_name, port in self.detected_clients.items():
                if port not in detected_ports:
                    clients_to_remove.append(instance_name)
            
            for instance_name in clients_to_remove:
                self.log_message(f"RuneLite client on port {self.detected_clients[instance_name]} no longer running", 'info')
                self.remove_instance_tab(instance_name)
                del self.detected_clients[instance_name]
                
        except Exception as e:
            self.log_message(f"Error detecting clients: {e}", 'error')
    
    def remove_instance_tab(self, instance_name):
        """Remove an instance tab and clean up references."""
        if instance_name in self.instance_tabs:
            # Stop any running plans for this instance
            if hasattr(self.instance_tabs[instance_name], 'is_running') and self.instance_tabs[instance_name].is_running:
                self.stop_plans_for_instance(instance_name)
            
            # Remove the tab from the notebook
            instance_tab = self.instance_tabs[instance_name]
            self.notebook.forget(instance_tab)
            
            # Clean up references
            del self.instance_tabs[instance_name]
            if instance_name in self.instance_ports:
                del self.instance_ports[instance_name]
    
    def start_client_detection(self):
        """Start automatic client detection."""
        if not self.client_detection_running:
            self.client_detection_running = True
            self.detection_status_label.config(text="Auto-detection: Running", style='Success.TLabel')
            self.log_message("Started automatic client detection", 'info')
            self.detect_running_clients()
            # Schedule next detection in 5 seconds
            self.root.after(5000, self._client_detection_loop)
    
    def stop_client_detection(self):
        """Stop automatic client detection."""
        self.client_detection_running = False
        self.detection_status_label.config(text="Auto-detection: Stopped", style='Info.TLabel')
        self.log_message("Stopped automatic client detection", 'info')
    
    def _client_detection_loop(self):
        """Internal method for client detection loop."""
        if self.client_detection_running:
            self.detect_running_clients()
            # Schedule next detection in 5 seconds
            self.root.after(5000, self._client_detection_loop)
    
    def test_client_detection(self):
        """Test method to debug client detection."""
        try:
            self.log_message("Testing client detection...", 'info')
            
            # Find all Java processes
            java_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline:
                            java_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': ' '.join(cmdline[:3]) + '...' if len(cmdline) > 3 else ' '.join(cmdline)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.log_message(f"Found {len(java_processes)} Java processes:", 'info')
            for proc in java_processes:
                self.log_message(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline']}", 'info')
            
            # Check for RuneLite processes specifically
            runelite_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('runelite' in arg.lower() for arg in cmdline):
                            runelite_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.log_message(f"Found {len(runelite_processes)} RuneLite processes", 'info')
            
            # Check for IPC ports
            detected_ports = set()
            for proc in runelite_processes:
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            detected_ports.add(conn.laddr.port)
                            self.log_message(f"  Found IPC port {conn.laddr.port} on PID {proc.pid}", 'info')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if detected_ports:
                self.log_message(f"Detected IPC ports: {sorted(detected_ports)}", 'info')
            else:
                self.log_message("No IPC ports detected in range 17000-17099", 'info')
                
        except Exception as e:
            self.log_message(f"Error testing client detection: {e}", 'error')


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = SimpleRecorderGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # No need to bind listbox selection change since we removed the old plan runner
    
    root.mainloop()


if __name__ == "__main__":
    main()
