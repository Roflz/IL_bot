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
import sys
import json
from pathlib import Path
import psutil
from typing import Dict, Any, TypedDict

from helpers.ipc import IPCClient
from utils.stats_monitor import StatsMonitor

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
        elif plan_name == 'wait_plan':
            self.create_wait_parameters_tab(parent)
        elif plan_name == 'tutorial_island':
            self.create_tutorial_island_parameters_tab(parent)
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
            
    def create_wait_parameters_tab(self, parent):
        """Create wait-specific parameters tab."""
        # Wait Time
        ttk.Label(parent, text="Wait Time (minutes):", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Get existing wait_minutes or default to 1.0
        existing_wait = self.plan_entry['params'].get('generic', {}).get('wait_minutes', 1.0)
        self.wait_minutes_var = tk.StringVar(value=str(existing_wait))
        
        wait_frame = ttk.Frame(parent)
        wait_frame.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        wait_spinbox = ttk.Spinbox(wait_frame, from_=0.1, to=1440.0, increment=0.1, 
                                  textvariable=self.wait_minutes_var, width=10)
        wait_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(wait_frame, text="minutes").pack(side=tk.LEFT, padx=(5, 0))
        
        # Help text
        help_text = "How many minutes to wait before completing the plan (0.1 to 1440 minutes = 24 hours max)"
        ttk.Label(parent, text=help_text, style='Info.TLabel').grid(row=1, column=0, columnspan=2, pady=5)
    
    def create_tutorial_island_parameters_tab(self, parent):
        """Create tutorial island-specific parameters tab."""
        # Credentials File
        ttk.Label(parent, text="Credentials File:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Get existing credentials_file or default to empty
        existing_cred = self.plan_entry['params'].get('generic', {}).get('credentials_file', '')
        self.credentials_file_var = tk.StringVar(value=str(existing_cred))
        
        cred_entry = ttk.Entry(parent, textvariable=self.credentials_file_var, width=30)
        cred_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Help text
        help_text = "Enter the filename (without .properties) to rename credentials file to match character name"
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
            elif self.plan_entry['name'] == 'wait_plan':
                # Collect wait_minutes parameter for wait plan
                wait_minutes = getattr(self, 'wait_minutes_var', None)
                if wait_minutes:
                    try:
                        wait_value = float(wait_minutes.get())
                        if wait_value < 0.1:
                            wait_value = 0.1
                        elif wait_value > 1440.0:
                            wait_value = 1440.0
                        params = {'generic': {'wait_minutes': wait_value}}
                    except ValueError:
                        params = {'generic': {'wait_minutes': 1.0}}  # Default
                else:
                    params = {'generic': {'wait_minutes': 1.0}}  # Default
            elif self.plan_entry['name'] == 'tutorial_island':
                # Collect credentials_file parameter for tutorial island
                credentials_file = getattr(self, 'credentials_file_var', None)
                if credentials_file:
                    cred_value = credentials_file.get().strip()
                    params = {'generic': {'credentials_file': cred_value}}
                else:
                    params = {'generic': {'credentials_file': ''}}  # Default
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
        
        # Initialize skill icons dict
        self.skill_icons = {}
        
        # Create GUI
        self.create_widgets()
        
        # Load skill icons once at startup
        self._load_skill_icons()
        
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
        # Focus the corresponding RuneLite window when switching instance tabs
        self.notebook.bind("<<NotebookTabChanged>>", self.on_instance_tab_changed)
        
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
    
    def update_instance_phase(self, instance_name, phase):
        """Update the current phase display for an instance."""
        if instance_name not in self.instance_tabs:
            return
        
        instance_tab = self.instance_tabs[instance_name]
        if hasattr(instance_tab, 'current_phase_label'):
            instance_tab.current_phase_label.config(text=phase)
            instance_tab.current_phase = phase
    
    
    
    
    
    
    
    
    def load_character_stats(self, username):
        """Load character stats from CSV file and return the data."""
        try:
            # Use absolute path matching stats_monitor
            csv_path = Path(__file__).parent / "character_data" / "character_stats.csv"
            if not csv_path.exists():
                return None
            
            import csv
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Find the most recent entry for this username
                latest_entry = None
                for row in reader:
                    if row.get('username') == username:
                        latest_entry = row
                
                return latest_entry
                    
        except Exception as e:
            logging.error(f"Error loading character stats: {e}")
            return None
    
    def _load_skill_icons(self):
        """Load skill icon images from files."""
        if not hasattr(self, 'skill_icons'):
            self.skill_icons = {}
        
        try:
            from PIL import Image, ImageTk
            
            # Skill name to icon filename mapping (using actual filenames from skill_icons folder)
            skill_icon_map = {
                'attack': 'attack.png',
                'strength': 'strength.png',
                'defence': 'defence.png',
                'hitpoints': 'hitpoints.png',
                'ranged': 'ranged.png',
                'prayer': 'prayer.png',
                'magic': 'magic.png',
                'cooking': 'cooking.png',
                'woodcutting': 'woodcutting.png',
                'fletching': 'fletching.png',
                'fishing': 'fishing.png',
                'firemaking': 'firemaking.png',
                'crafting': 'crafting.png',
                'smithing': 'smithing.png',
                'mining': 'mining.png',
                'herblore': 'herblore.png',
                'agility': 'agility.png',
                'thieving': 'thieving.png',
                'slayer': 'slayer.png',
                'farming': 'farming.png',
                'runecraft': 'runecrafting.png',  # Note: filename is "runecrafting" not "runecraft"
                'hunter': 'hunter.png',
                'construction': 'construction.png'
            }
            
            # Use absolute path relative to gui.py location
            icons_dir = Path(__file__).parent / "skill_icons"
            logging.info(f"[GUI] Loading skill icons from: {icons_dir}")
            if icons_dir.exists():
                for skill_name, icon_file in skill_icon_map.items():
                    icon_path = icons_dir / icon_file
                    if icon_path.exists():
                        try:
                            # Load and resize icon (assuming icons are ~20-30px, resize to 20x20)
                            img = Image.open(icon_path)
                            img = img.resize((20, 20), Image.Resampling.LANCZOS)
                            # Store as PhotoImage - keep reference in instance to prevent garbage collection
                            photo = ImageTk.PhotoImage(img)
                            self.skill_icons[skill_name] = photo
                            logging.debug(f"[GUI] Loaded icon: {icon_file}")
                        except Exception as e:
                            logging.warning(f"Could not load icon {icon_file}: {e}")
                            self.skill_icons[skill_name] = None
                    else:
                        logging.warning(f"Icon file not found: {icon_path}")
                        self.skill_icons[skill_name] = None
            else:
                logging.warning(f"Skill icons directory does not exist: {icons_dir}")
        except ImportError as e:
            logging.error(f"PIL/Pillow not available, skill icons will not be displayed. Error: {e}")
            logging.error("Please install Pillow: pip install Pillow")
            self.skill_icons = {}
        except Exception as e:
            logging.error(f"Error loading skill icons: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.skill_icons = {}
        
        # Log summary of loaded icons
        loaded_count = sum(1 for icon in self.skill_icons.values() if icon is not None)
        logging.info(f"[GUI] Loaded {loaded_count}/{len(skill_icon_map)} skill icons")
    
    def estimate_item_value(self, item_key, quantity):
        """Estimate the value of an item (simplified pricing)."""
        # Basic pricing estimates (you can expand this)
        prices = {
            'coins': 1,
            'cowhides': 100,
            'logs': 25,
            'leather': 200,
            'bow_string': 50,
            'iron_bar': 150,
            'steel_bar': 300,
            'coal': 30,
            'iron_ore': 50,
            'raw_fish': 20,
            'cooked_fish': 40,
            'bronze_bar': 100,
            'silver_bar': 250,
            'gold_bar': 500,
            'mithril_bar': 600,
            'adamant_bar': 1200,
            'rune_bar': 3000,
            'bronze_ore': 25,
            'silver_ore': 100,
            'gold_ore': 200,
            'mithril_ore': 300,
            'adamant_ore': 600,
            'rune_ore': 1500,
            'willow_logs': 50,
            'oak_logs': 100,
            'maple_logs': 200,
            'yew_logs': 500,
            'magic_logs': 1000,
            'redwood_logs': 2000
        }
        
        price_per_item = prices.get(item_key, 0)
        return price_per_item * quantity
    
    def update_stats_text(self, username):
        """Update the stats display for an instance with icons."""
        try:
            instance_tab = self.instance_tabs.get(username)
            if not instance_tab or not hasattr(instance_tab, 'skills_scrollable_frame'):
                return
            
            # Load current stats
            stats_data = self.load_character_stats(username)
            
            # Update logged-in time label
            if hasattr(instance_tab, 'logged_in_time_label'):
                logged_in_time = stats_data.get('logged_in_time', 0) if stats_data else 0
                try:
                    logged_in_seconds = float(logged_in_time)
                    hours = int(logged_in_seconds // 3600)
                    minutes = int((logged_in_seconds % 3600) // 60)
                    seconds = int(logged_in_seconds % 60)
                    time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                except (ValueError, TypeError):
                    time_str = "0:00:00"
                instance_tab.logged_in_time_label.config(text=time_str)
            
            if not stats_data:
                # Clear all sections and show no data message
                for widget in instance_tab.skills_scrollable_frame.winfo_children():
                    widget.destroy()
                for widget in instance_tab.inventory_scrollable_frame.winfo_children():
                    widget.destroy()
                for widget in instance_tab.equipment_scrollable_frame.winfo_children():
                    widget.destroy()
                
                no_data_label = ttk.Label(instance_tab.skills_scrollable_frame, text="No stats data available")
                no_data_label.grid(row=0, column=0, padx=5, pady=5)
                return
            
            # Clear existing widgets in all three sections
            for widget in instance_tab.skills_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in instance_tab.inventory_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in instance_tab.equipment_scrollable_frame.winfo_children():
                widget.destroy()
            
            # ========== SKILLS SECTION ==========
            skills_header = ttk.Label(instance_tab.skills_scrollable_frame, text="SKILLS", font=("Arial", 10, "bold"))
            skills_header.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5, 2))
            
            # Configure grid columns (3 columns)
            for col in range(3):
                instance_tab.skills_scrollable_frame.columnconfigure(col, weight=1, minsize=80)
            
            # Skills in exact game order (3 columns, top to bottom in each column)
            # Column 1: attack, strength, defence, ranged, prayer, magic, runecraft, construction
            # Column 2: hitpoints, agility, herblore, thieving, crafting, fletching, slayer, hunter
            # Column 3: mining, smithing, fishing, cooking, firemaking, woodcutting, farming
            skills_columns = [
                # Column 1 (8 skills)
                [
                    ("attack", "Attack", stats_data.get('attack_level', 1), stats_data.get('attack_xp', 0)),
                    ("strength", "Strength", stats_data.get('strength_level', 1), stats_data.get('strength_xp', 0)),
                    ("defence", "Defence", stats_data.get('defence_level', 1), stats_data.get('defence_xp', 0)),
                    ("ranged", "Ranged", stats_data.get('ranged_level', 1), stats_data.get('ranged_xp', 0)),
                    ("prayer", "Prayer", stats_data.get('prayer_level', 1), stats_data.get('prayer_xp', 0)),
                    ("magic", "Magic", stats_data.get('magic_level', 1), stats_data.get('magic_xp', 0)),
                    ("runecraft", "Runecraft", stats_data.get('runecraft_level', 1), stats_data.get('runecraft_xp', 0)),
                    ("construction", "Construction", stats_data.get('construction_level', 1), stats_data.get('construction_xp', 0))
                ],
                # Column 2 (8 skills)
                [
                    ("hitpoints", "Hitpoints", stats_data.get('hitpoints_level', 10), stats_data.get('hitpoints_xp', 1154)),
                    ("agility", "Agility", stats_data.get('agility_level', 1), stats_data.get('agility_xp', 0)),
                    ("herblore", "Herblore", stats_data.get('herblore_level', 1), stats_data.get('herblore_xp', 0)),
                    ("thieving", "Thieving", stats_data.get('thieving_level', 1), stats_data.get('thieving_xp', 0)),
                    ("crafting", "Crafting", stats_data.get('crafting_level', 1), stats_data.get('crafting_xp', 0)),
                    ("fletching", "Fletching", stats_data.get('fletching_level', 1), stats_data.get('fletching_xp', 0)),
                    ("slayer", "Slayer", stats_data.get('slayer_level', 1), stats_data.get('slayer_xp', 0)),
                    ("hunter", "Hunter", stats_data.get('hunter_level', 1), stats_data.get('hunter_xp', 0))
                ],
                # Column 3 (7 skills + total level)
                [
                    ("mining", "Mining", stats_data.get('mining_level', 1), stats_data.get('mining_xp', 0)),
                    ("smithing", "Smithing", stats_data.get('smithing_level', 1), stats_data.get('smithing_xp', 0)),
                    ("fishing", "Fishing", stats_data.get('fishing_level', 1), stats_data.get('fishing_xp', 0)),
                    ("cooking", "Cooking", stats_data.get('cooking_level', 1), stats_data.get('cooking_xp', 0)),
                    ("firemaking", "Firemaking", stats_data.get('firemaking_level', 1), stats_data.get('firemaking_xp', 0)),
                    ("woodcutting", "Woodcutting", stats_data.get('woodcutting_level', 1), stats_data.get('woodcutting_xp', 0)),
                    ("farming", "Farming", stats_data.get('farming_level', 1), stats_data.get('farming_xp', 0))
                ]
            ]
            
            # Display skills column by column
            for col_idx, column_skills in enumerate(skills_columns):
                for row_idx, (skill_key, skill_name, level, xp) in enumerate(column_skills):
                    # Convert to int if they're strings
                    try:
                        level_int = int(level) if level else 0
                        xp_int = int(xp) if xp else 0
                    except (ValueError, TypeError):
                        level_int = 0
                        xp_int = 0
                    
                    # Create skill row frame (no border, no background box)
                    skill_frame = tk.Frame(instance_tab.skills_scrollable_frame, bg="#f0f0f0")
                    skill_frame.grid(row=row_idx + 1, column=col_idx, sticky=(tk.W, tk.E), padx=2, pady=1)
                    
                    # Icon on the left
                    skill_icon = self.skill_icons.get(skill_key)
                    if skill_icon and skill_icon is not None:
                        icon_label = tk.Label(skill_frame, image=skill_icon, bg="#f0f0f0")
                        icon_label.image = skill_icon  # Keep reference
                    else:
                        icon_label = tk.Label(skill_frame, text=skill_name[0].upper(), font=("Arial", 10, "bold"), 
                                             width=2, bg="#f0f0f0", fg="#333333")
                    icon_label.pack(side=tk.LEFT, padx=(0, 0))
                    
                    # Stats numbers on the right - right-aligned
                    stats_text = f"{level_int} / {level_int}"
                    stats_label = tk.Label(skill_frame, text=stats_text, font=("Arial", 8), bg="#f0f0f0", 
                                          fg="#000000", anchor=tk.E, width=6)
                    stats_label.pack(side=tk.LEFT)
                
                # Add total level at bottom of column 3
                if col_idx == 2:
                    # Calculate total level
                    total_level = sum(
                        int(stats_data.get(f'{skill[0]}_level', 0) or 0)
                        for column in skills_columns
                        for skill in column
                        if skill[0] != 'total'  # Skip any 'total' entries
                    )
                    
                    total_frame = tk.Frame(instance_tab.skills_scrollable_frame, bg="#f0f0f0")
                    total_frame.grid(row=8, column=2, sticky=(tk.W, tk.E), padx=2, pady=1)
                    
                    total_label = tk.Label(total_frame, text=f"Total level:\n{total_level}", 
                                          font=("Arial", 9, "bold"), bg="#f0f0f0", fg="#000000", anchor=tk.E)
                    total_label.pack()
            
            # Update skills canvas scroll region
            instance_tab.skills_scrollable_frame.update_idletasks()
            instance_tab.skills_canvas.configure(scrollregion=instance_tab.skills_canvas.bbox("all"))
            
            # ========== INVENTORY SECTION ==========
            inventory_header = ttk.Label(instance_tab.inventory_scrollable_frame, text="INVENTORY", font=("Arial", 10, "bold"))
            inventory_header.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 2))
            row = 1
            
            # Inventory items - show ALL items currently in inventory (anything starting with 'inventory_')
            for key in sorted(stats_data.keys()):
                if key.startswith('inventory_'):
                    try:
                        qty = int(stats_data.get(key, 0) or 0)
                    except (ValueError, TypeError):
                        qty = 0
                    if qty > 0:
                        # Extract item name from key (inventory_itemname -> "Itemname")
                        item_name = key.replace('inventory_', '').replace('_', ' ').title()
                        item_label = ttk.Label(instance_tab.inventory_scrollable_frame, text=f"{item_name}: {qty:,}", font=("Arial", 8))
                        item_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
                        row += 1
            
            # Update inventory canvas scroll region
            instance_tab.inventory_scrollable_frame.update_idletasks()
            instance_tab.inventory_canvas.configure(scrollregion=instance_tab.inventory_canvas.bbox("all"))
            
            # ========== EQUIPMENT SECTION ==========
            equipped_header = ttk.Label(instance_tab.equipment_scrollable_frame, text="EQUIPPED", font=("Arial", 10, "bold"))
            equipped_header.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 2))
            row = 1
            
            equipped_items = [
                ("Helmet", stats_data.get('equipped_helmet', '')),
                ("Cape", stats_data.get('equipped_cape', '')),
                ("Amulet", stats_data.get('equipped_amulet', '')),
                ("Weapon", stats_data.get('equipped_weapon', '')),
                ("Body", stats_data.get('equipped_body', '')),
                ("Shield", stats_data.get('equipped_shield', '')),
                ("Legs", stats_data.get('equipped_legs', '')),
                ("Gloves", stats_data.get('equipped_gloves', '')),
                ("Boots", stats_data.get('equipped_boots', '')),
                ("Ring", stats_data.get('equipped_ring', ''))
            ]
            
            for slot_name, item in equipped_items:
                if item:
                    eq_label = ttk.Label(instance_tab.equipment_scrollable_frame, text=f"{slot_name}: {item}", font=("Arial", 8))
                    eq_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
                    row += 1
            
            # Update equipment canvas scroll region
            instance_tab.equipment_scrollable_frame.update_idletasks()
            instance_tab.equipment_canvas.configure(scrollregion=instance_tab.equipment_canvas.bbox("all"))
            
            # ========== KEY ITEMS TOTALS (when bank is open) ==========
            # Check if key_items_frame exists
            if not hasattr(instance_tab, 'key_items_frame'):
                logging.warning(f"key_items_frame not found for {username}")
            else:
                # Clear existing key items display
                for widget in instance_tab.key_items_frame.winfo_children():
                    widget.destroy()
                
                # Check if bank is open by checking IPC
                bank_is_open = False
                try:
                    port = self.instance_ports.get(username)
                    if port:
                        ipc = IPCClient(host="127.0.0.1", port=port, timeout_s=0.5)
                        bank_data = ipc.get_bank()
                        bank_is_open = bank_data and bank_data.get("ok", False)
                        if bank_is_open:
                            logging.info(f"[GUI] Bank is open for {username} - showing key items totals")
                        else:
                            logging.debug(f"[GUI] Bank is not open for {username}")
                except Exception as e:
                    # If we can't check, don't show totals
                    logging.warning(f"[GUI] Could not check bank status for {username}: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    bank_is_open = False
                
                if bank_is_open:
                    # Key items to show totals for (bank + inventory)
                    key_items_list = [
                        ('coins', 'Coins'),
                        ('cowhides', 'Cowhides'),
                        ('logs', 'Logs'),
                        ('leather', 'Leather'),
                        ('bow_string', 'Bow String'),
                        ('iron_bar', 'Iron Bar'),
                        ('steel_bar', 'Steel Bar'),
                        ('coal', 'Coal'),
                        ('iron_ore', 'Iron Ore'),
                        ('raw_fish', 'Raw Fish'),
                        ('cooked_fish', 'Cooked Fish')
                    ]
                    
                    # Header
                    key_items_header = ttk.Label(instance_tab.key_items_frame, text="Key Items (Bank + Inventory):", 
                                               font=("Arial", 9, "bold"))
                    key_items_header.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
                    
                    row = 1
                    items_shown = 0
                    for item_key, item_name in key_items_list:
                        try:
                            bank_count = int(stats_data.get(f'{item_key}_bank', 0) or 0)
                            inv_count = int(stats_data.get(f'{item_key}_inventory', 0) or 0)
                            total = bank_count + inv_count
                        except (ValueError, TypeError):
                            bank_count = 0
                            inv_count = 0
                            total = 0
                        
                        if total > 0:
                            item_label = ttk.Label(instance_tab.key_items_frame, 
                                                 text=f"{item_name}: {total:,}", 
                                                 font=("Arial", 8))
                            item_label.grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=1)
                            row += 1
                            items_shown += 1
                            logging.debug(f"[GUI] Showing {item_name}: bank={bank_count}, inventory={inv_count}, total={total}")
                    
                    if items_shown == 0:
                        # Show a message if no items found
                        no_items_label = ttk.Label(instance_tab.key_items_frame, 
                                                  text="No key items found", 
                                                  font=("Arial", 8), 
                                                  foreground="gray")
                        no_items_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=1)
                        logging.debug(f"[GUI] Bank is open but no key items found for {username}")
                else:
                    # Show a message when bank is not open
                    no_bank_label = ttk.Label(instance_tab.key_items_frame, 
                                              text="Open bank to see totals", 
                                              font=("Arial", 8), 
                                              foreground="gray")
                    no_bank_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=1)
            
        except Exception as e:
            logging.error(f"Error updating stats text for {username}: {e}")
    
    def start_stats_monitor(self, username, port):
        """Start the stats monitor for an instance."""
        logging.info(f"[GUI] Starting stats monitor for {username} on port {port}")
        try:
            # Check if monitor already exists
            if hasattr(self, 'stats_monitors') and username in self.stats_monitors:
                existing_monitor = self.stats_monitors[username]
                # Check if it's still running
                if existing_monitor.running:
                    logging.debug(f"[GUI] Stats monitor already running for {username}, skipping")
                    return
                else:
                    # Monitor exists but isn't running, remove it
                    logging.debug(f"[GUI] Removing non-running monitor for {username}")
                    del self.stats_monitors[username]
            
            # Create callback to update GUI when CSV is updated
            def on_csv_update(updated_username):
                """Callback to update GUI stats display when CSV is updated."""
                # Schedule GUI update on main thread
                self.root.after(0, lambda: self.update_stats_text(updated_username))
            
            # Create callback for username changes
            def on_username_changed(old_username, new_username):
                """Callback when username changes (e.g., unnamed_character -> actual name)."""
                # Schedule on main thread
                self.root.after(0, lambda: self._handle_username_change(old_username, new_username))
            
            monitor = StatsMonitor(port, username, on_csv_update=on_csv_update, 
                                 on_username_changed=on_username_changed)
            logging.info(f"[GUI] Created StatsMonitor for {username}, starting...")
            monitor.start()
            logging.info(f"[GUI] StatsMonitor.start() called for {username}")
            
            # Store monitor reference
            if not hasattr(self, 'stats_monitors'):
                self.stats_monitors = {}
            self.stats_monitors[username] = monitor
            
            logging.info(f"[GUI] Started stats monitor for {username} on port {port}")
            
        except Exception as e:
            logging.error(f"[GUI] Error starting stats monitor for {username}: {e}")
    
    def _write_rule_params_to_file(self, username):
        """Write rule parameters to JSON file for StatsMonitor to read."""
        try:
            instance_tab = self.instance_tabs.get(username)
            if not instance_tab:
                return
            
            # Collect all rules from all plan entries
            all_rules = {}
            for plan_entry in instance_tab.plan_entries:
                rules = plan_entry.get('rules', {})
                for key, value in rules.items():
                    if value:  # Only include non-empty values
                        all_rules[key] = value
            
            # Add start_time if not present
            if 'start_time' not in all_rules:
                from datetime import datetime
                all_rules['start_time'] = datetime.now().isoformat()
            
            # Write to JSON file
            rule_params_file = Path(__file__).parent / "character_data" / f"rule_params_{username}.json"
            rule_params_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(rule_params_file, 'w', encoding='utf-8') as f:
                json.dump(all_rules, f, indent=2)
            
            logging.info(f"[GUI] Wrote rule parameters to {rule_params_file}: {all_rules}")
            
        except Exception as e:
            logging.error(f"[GUI] Error writing rule parameters for {username}: {e}")
    
    def stop_stats_monitor(self, username):
        """Stop the stats monitor for an instance."""
        if hasattr(self, 'stats_monitors') and username in self.stats_monitors:
            self.stats_monitors[username].stop()
            del self.stats_monitors[username]
            logging.info(f"[GUI] Stopped stats monitor for {username}")
    
    def _handle_username_change(self, old_username, new_username):
        """Handle username change (e.g., unnamed_character -> actual name)."""
        try:
            logging.info(f"[GUI] Handling username change: {old_username} -> {new_username}")
            
            # Update instance_tabs dictionary key
            if old_username in self.instance_tabs:
                # Store the tab reference
                instance_tab = self.instance_tabs[old_username]
                
                # Remove old key
                del self.instance_tabs[old_username]
                
                # Add with new key
                self.instance_tabs[new_username] = instance_tab
                
                # Update instance_ports
                if old_username in self.instance_ports:
                    port = self.instance_ports[old_username]
                    del self.instance_ports[old_username]
                    self.instance_ports[new_username] = port
                
                # Update stats_monitors
                if hasattr(self, 'stats_monitors') and old_username in self.stats_monitors:
                    monitor = self.stats_monitors[old_username]
                    del self.stats_monitors[old_username]
                    self.stats_monitors[new_username] = monitor
                
                # Update notebook tab text
                self.notebook.forget(instance_tab)
                self.notebook.add(instance_tab, text=new_username)
                
                logging.info(f"[GUI] Updated instance tab from {old_username} to {new_username}")
            
            # Update selected_credentials list if it contains the old credential
            old_cred_name = f"{old_username}.properties"
            new_cred_name = f"{new_username}.properties"
            if old_cred_name in self.selected_credentials:
                index = self.selected_credentials.index(old_cred_name)
                self.selected_credentials[index] = new_cred_name
                self.update_selected_credentials_display()
                logging.info(f"[GUI] Updated selected credentials: {old_cred_name} -> {new_cred_name}")
                
        except Exception as e:
            logging.error(f"[GUI] Error handling username change: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def refresh_stats_for_instance(self, instance_name):
        """Refresh stats display for a specific instance."""
        if instance_name in self.instance_tabs:
            self.update_stats_text(instance_name)

    def on_instance_tab_changed(self, event):
        """When switching tabs, focus that instance's RuneLite window."""
        try:
            selected_tab_id = event.widget.select()
            selected_frame = event.widget.nametowidget(selected_tab_id)
            # Find which username maps to this frame
            for username, tab_frame in self.instance_tabs.items():
                if tab_frame is selected_frame:
                    self.focus_runelite_window(username)
                    break
        except Exception:
            pass

    def focus_runelite_window(self, username: str) -> None:
        """Attempt to focus the RuneLite window for this username using port matching."""
        try:
            import win32gui
            import win32con
            import psutil
        except Exception:
            return

        # Get the port for this instance
        port = self.instance_ports.get(username)
        if not port:
            return

        # Collect all top-level window info
        window_info = []
        
        def enum_handler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and "runelite" in title.lower():
                    try:
                        # Get process ID from window
                        _, pid = win32gui.GetWindowThreadProcessId(hwnd)
                        window_info.append((hwnd, title, pid))
                    except Exception:
                        pass
        
        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception:
            return

        if not window_info:
            return

        # Try to find the window by matching the process port
        target_hwnd = None
        for hwnd, title, pid in window_info:
            try:
                # Check if this process is listening on our target port
                process = psutil.Process(pid)
                connections = process.connections()
                for conn in connections:
                    # Check if this process is listening on the target port
                    if conn.laddr.port == port:
                        target_hwnd = hwnd
                        break
                if target_hwnd:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if target_hwnd is None:
            return

        try:
            if win32gui.IsIconic(target_hwnd):
                win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(target_hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(target_hwnd)
        except Exception:
            # Best-effort; ignore focus errors
            pass
    
    
    def populate_credentials(self):
        """Populate the credentials listbox with available credential files."""
        credentials_dir = Path("/credentials")
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
        plan_runner_tab = ttk.Frame(sub_notebook, padding="0")
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
        # Top row: Config + Skills + Inventory + Equipment
        top_row = ttk.Frame(plan_runner_tab, padding="10")
        top_row.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        top_row.columnconfigure(0, weight=0)  # Config - no expansion
        top_row.columnconfigure(1, weight=1)  # Skills
        top_row.columnconfigure(2, weight=1)   # Inventory
        top_row.columnconfigure(3, weight=1)  # Equipment
        
        # Left side: Session config
        config_frame = ttk.Frame(top_row)
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
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
        
        # Credential file (read-only for instances)
        ttk.Label(config_frame, text="Credential:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        # Find the credential file for this instance
        cred_file_name = None
        for selected_cred in self.selected_credentials:
            # Extract username from credential filename (remove .properties)
            cred_username = selected_cred.replace('.properties', '')
            if cred_username == username:
                cred_file_name = selected_cred
                break
        
        if cred_file_name:
            cred_label = ttk.Label(config_frame, text=cred_file_name, style='Info.TLabel')
            cred_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        else:
            cred_label = ttk.Label(config_frame, text="Not found", style='Warning.TLabel')
            cred_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Current Plan (read-only, updates during execution)
        ttk.Label(config_frame, text="Current Plan:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=2)
        current_plan_label = ttk.Label(config_frame, text="None", style='Info.TLabel')
        current_plan_label.grid(row=3, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Current Phase (read-only, updates during execution)
        ttk.Label(config_frame, text="Current Phase:", style='Header.TLabel').grid(row=4, column=0, sticky=tk.W, pady=2)
        current_phase_label = ttk.Label(config_frame, text="Idle", style='Info.TLabel')
        current_phase_label.grid(row=4, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Logged In Time (read-only, updates from CSV)
        ttk.Label(config_frame, text="Logged In Time:", style='Header.TLabel').grid(row=5, column=0, sticky=tk.W, pady=2)
        logged_in_time_label = ttk.Label(config_frame, text="0:00:00", style='Info.TLabel')
        logged_in_time_label.grid(row=5, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Control buttons below Logged In Time
        control_buttons_frame = ttk.Frame(config_frame)
        control_buttons_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Start button with green triangle symbol
        start_button = tk.Button(control_buttons_frame, text="▶", bg="#4CAF50", fg="white", 
                                font=("Arial", 10, "bold"), width=2, height=1,
                                command=lambda: self.start_plans_for_instance(username, session_dir.get(), port))
        start_button.grid(row=0, column=0, padx=(0, 5))
        
        # Stop button with red square symbol
        stop_button = tk.Button(control_buttons_frame, text="■", bg="#F44336", fg="white",
                               font=("Arial", 10, "bold"), width=2, height=1,
                               command=lambda: self.stop_plans_for_instance(username))
        stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # Pause between plans checkbox
        pause_var = tk.BooleanVar(value=False)
        pause_checkbox = ttk.Checkbutton(control_buttons_frame, text="Pause between plans", variable=pause_var)
        pause_checkbox.grid(row=0, column=2, padx=(10, 0))
        
        # Store pause variable in instance tab
        instance_tab.pause_var = pause_var
        
        # Key Items Totals section (shown when bank is open)
        key_items_frame = ttk.Frame(config_frame)
        key_items_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Store reference for updating
        instance_tab.key_items_frame = key_items_frame
        
        # Store references for updating during execution
        instance_tab.current_plan_label = current_plan_label
        instance_tab.current_phase_label = current_phase_label
        instance_tab.logged_in_time_label = logged_in_time_label
        
        # Helper function to create a scrollable stats section
        def create_stats_section(parent, col):
            container = ttk.Frame(parent)
            container.grid(row=0, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 5))
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)
            
            canvas = tk.Canvas(container, bg="#f0f0f0", highlightthickness=0)
            scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            return container, scrollable_frame, canvas
        
        # Skills section (column 1)
        skills_container, skills_scrollable_frame, skills_canvas = create_stats_section(top_row, 1)
        
        # Inventory section (column 2)
        inventory_container, inventory_scrollable_frame, inventory_canvas = create_stats_section(top_row, 2)
        
        # Equipment section (column 3)
        equipment_container, equipment_scrollable_frame, equipment_canvas = create_stats_section(top_row, 3)
        
        # Store references for updates
        instance_tab.skills_container = skills_container
        instance_tab.skills_scrollable_frame = skills_scrollable_frame
        instance_tab.skills_canvas = skills_canvas
        instance_tab.inventory_container = inventory_container
        instance_tab.inventory_scrollable_frame = inventory_scrollable_frame
        instance_tab.inventory_canvas = inventory_canvas
        instance_tab.equipment_container = equipment_container
        instance_tab.equipment_scrollable_frame = equipment_scrollable_frame
        instance_tab.equipment_canvas = equipment_canvas
        
        # Start stats monitor for this instance
        self.start_stats_monitor(username, port)
        
        # Initial stats update
        self.update_stats_text(username)
        
        # Main content: Left column (plans + details), Right column (stats)
        main_content = ttk.Frame(plan_runner_tab)
        main_content.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        main_content.columnconfigure(0, weight=1)  # Left column - expandable
        main_content.columnconfigure(1, weight=1)  # Right column - expandable
        plan_runner_tab.rowconfigure(1, weight=1)
        
        # Left column: Plan selection (top) and saved sequences (bottom)
        left_column = ttk.Frame(main_content)
        left_column.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_column.columnconfigure(0, weight=1)
        left_column.rowconfigure(0, weight=1)  # Plan selection - expandable
        left_column.rowconfigure(1, weight=0)  # Saved sequences - fixed
        
        # Left side: Plan selection
        left_frame = ttk.LabelFrame(left_column, text="Plan Selection", padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        left_frame.columnconfigure(0, weight=1)  # Available plans column
        left_frame.columnconfigure(1, weight=0)  # Control buttons column
        left_frame.columnconfigure(2, weight=1)  # Selected plans column
        left_frame.rowconfigure(1, weight=1)     # Plans row
        
        # Available plans
        ttk.Label(left_frame, text="Available Plans:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        available_listbox = tk.Listbox(left_frame, height=6)
        available_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(0, 5))
        
        # Populate available plans
        for plan_id, plan_class in AVAILABLE_PLANS.items():
            # Get label from plan class if it has one, otherwise use the plan_id
            label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
            available_listbox.insert(tk.END, f"{label} ({plan_id})")
        
        # Control buttons (middle column)
        plan_controls = ttk.Frame(left_frame)
        plan_controls.grid(row=1, column=1, sticky=(tk.N, tk.S), padx=5)
        
        ttk.Button(plan_controls, text="→", command=lambda: self.add_plan_to_selection(username, available_listbox, selected_listbox)).grid(row=0, column=0, pady=(0, 3))
        ttk.Button(plan_controls, text="←", command=lambda: self.remove_plan_from_selection(username, selected_listbox)).grid(row=1, column=0, pady=3)
        ttk.Button(plan_controls, text="↑", command=lambda: self.move_plan_up(username, selected_listbox)).grid(row=2, column=0, pady=3)
        ttk.Button(plan_controls, text="↓", command=lambda: self.move_plan_down(username, selected_listbox)).grid(row=3, column=0, pady=3)
        ttk.Button(plan_controls, text="✕", command=lambda: self.clear_selected_plans(username, selected_listbox)).grid(row=4, column=0, pady=(10, 0))
        
        # Selected plans
        ttk.Label(left_frame, text="Selected Plans:", style='Header.TLabel').grid(row=0, column=2, sticky=tk.W, pady=(0, 5))
        selected_listbox = tk.Listbox(left_frame, height=6)
        selected_listbox.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(5, 0))
        
        # Saved Sequences (moved under Plan Selection)
        sequences_frame = ttk.LabelFrame(left_column, text="Saved Sequences", padding="5")
        sequences_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        sequences_frame.columnconfigure(0, weight=1)
        sequences_frame.rowconfigure(1, weight=1)

        sequences_listbox = tk.Listbox(sequences_frame, height=4)
        sequences_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        seq_controls = ttk.Frame(sequences_frame)
        seq_controls.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Button(seq_controls, text="Save", command=lambda: self.save_sequence_for_instance(username)).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(seq_controls, text="Load", command=lambda: self.load_sequence_from_list(username, sequences_listbox)).grid(row=0, column=1, padx=3)
        ttk.Button(seq_controls, text="Delete", command=lambda: self.delete_sequence_from_list(username, sequences_listbox)).grid(row=0, column=2, padx=3)

        self.populate_sequences_list(username, sequences_listbox)
        instance_tab.sequences_listbox = sequences_listbox

        # Plan Details moved to right column (where stats used to be)
        right_frame = ttk.LabelFrame(main_content, text="Plan Details", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Details controls
        details_controls = ttk.Frame(right_frame)
        details_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(details_controls, text="Clear Parameters", command=lambda: self.clear_plan_parameters(username, selected_listbox)).grid(row=0, column=0, padx=3)
        ttk.Button(details_controls, text="Clear Rules", command=lambda: self.clear_plan_rules(username, selected_listbox)).grid(row=0, column=1, padx=3)
        
        # Editing section - inline editing for rules and parameters (moved to top)
        editing_frame = ttk.LabelFrame(right_frame, text="Edit", padding="5")
        editing_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        editing_frame.columnconfigure(1, weight=1)
        
        # Rules editing - with user-friendly widgets
        ttk.Label(editing_frame, text="Add Rule:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        rules_edit_frame = ttk.Frame(editing_frame)
        rules_edit_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Rule type dropdown
        rule_type_var = tk.StringVar(value="Time")
        rule_type_combo = ttk.Combobox(rules_edit_frame, textvariable=rule_type_var, width=12, 
                                       values=["Time", "Skill", "Item", "Total Level"], state="readonly")
        rule_type_combo.grid(row=0, column=0, padx=(0, 5))
        
        # Dynamic rule input frame (will be populated based on rule type)
        rule_input_frame = ttk.Frame(rules_edit_frame)
        rule_input_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Time rule widget
        time_spinbox = ttk.Spinbox(rule_input_frame, from_=0, to=10000, width=10)
        time_spinbox.set("0")
        time_label = ttk.Label(rule_input_frame, text="minutes")
        
        # Skill rule widgets
        skill_list = ["Attack", "Strength", "Defence", "Ranged", "Magic", "Woodcutting", "Fishing", 
                     "Cooking", "Mining", "Smithing", "Firemaking", "Crafting", "Fletching", "Runecraft", 
                     "Herblore", "Agility", "Thieving", "Slayer", "Farming", "Construction", "Hunter", "Prayer"]
        skill_var = tk.StringVar(value="")
        skill_combo = ttk.Combobox(rule_input_frame, textvariable=skill_var, values=skill_list, width=12, state="readonly")
        skill_level_spinbox = ttk.Spinbox(rule_input_frame, from_=1, to=99, width=5)
        skill_level_spinbox.set("1")
        skill_label = ttk.Label(rule_input_frame, text="level")
        
        # Item rule widgets
        item_name_entry = ttk.Entry(rule_input_frame, width=15)
        item_name_entry.insert(0, "item name")
        item_qty_spinbox = ttk.Spinbox(rule_input_frame, from_=1, to=99999, width=8)
        item_qty_spinbox.set("1")
        item_x_label = ttk.Label(rule_input_frame, text="x")
        
        # Total Level rule widget
        total_level_spinbox = ttk.Spinbox(rule_input_frame, from_=0, to=2277, width=10)
        total_level_spinbox.set("0")
        total_level_label = ttk.Label(rule_input_frame, text="level")
        
        def show_rule_input(*args):
            """Show appropriate input widgets based on selected rule type."""
            # Clear all widgets
            for widget in rule_input_frame.winfo_children():
                widget.grid_remove()
            
            rule_type = rule_type_var.get()
            if rule_type == "Time":
                time_spinbox.grid(row=0, column=0, padx=(0, 5))
                time_label.grid(row=0, column=1)
            elif rule_type == "Skill":
                skill_combo.grid(row=0, column=0, padx=(0, 5))
                skill_level_spinbox.grid(row=0, column=1, padx=(0, 5))
                skill_label.grid(row=0, column=2)
            elif rule_type == "Item":
                item_name_entry.grid(row=0, column=0, padx=(0, 5))
                item_x_label.grid(row=0, column=1, padx=(0, 5))
                item_qty_spinbox.grid(row=0, column=2)
            elif rule_type == "Total Level":
                total_level_spinbox.grid(row=0, column=0, padx=(0, 5))
                total_level_label.grid(row=0, column=1)
        
        rule_type_var.trace('w', show_rule_input)
        show_rule_input()  # Initial display
        
        # Add rule button
        def add_rule_cmd():
            rule_data = {
                'Time': (time_spinbox, None),
                'Skill': (skill_combo, skill_level_spinbox),
                'Item': (item_name_entry, item_qty_spinbox),
                'Total Level': (total_level_spinbox, None)
            }
            self.add_rule_inline_advanced(username, selected_listbox, rule_type_var, rule_data,
                                          rules_scrollable_frame, rules_canvas)
        
        ttk.Button(rules_edit_frame, text="Add", width=8, command=add_rule_cmd).grid(row=0, column=2)
        
        # Parameters editing - dynamic based on plan type
        ttk.Label(editing_frame, text="Add Parameter:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        params_edit_container = ttk.Frame(editing_frame)
        params_edit_container.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        params_edit_container.columnconfigure(0, weight=1)
        
        # This will be populated when plan selection changes
        params_edit_frame = ttk.Frame(params_edit_container)
        params_edit_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Store references for dynamic updates
        instance_tab.params_edit_container = params_edit_container
        instance_tab.params_edit_frame = params_edit_frame
        
        # Details display - create scrollable sections like Inventory/Equipped (below Edit section)
        details_frame = ttk.Frame(right_frame)
        details_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        
        # Rules section with scrollable frame (auto-sizing, starts at 0 height)
        ttk.Label(details_frame, text="Rules", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 2))
        
        # Create scrollable frame for rules (auto-sized based on content)
        rules_container = ttk.Frame(details_frame)
        rules_container.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        rules_container.columnconfigure(0, weight=1)
        
        rules_canvas = tk.Canvas(rules_container, height=0)
        rules_scrollbar = ttk.Scrollbar(rules_container, orient=tk.VERTICAL, command=rules_canvas.yview)
        rules_scrollable_frame = ttk.Frame(rules_canvas)
        
        rules_canvas.configure(yscrollcommand=rules_scrollbar.set)
        rules_canvas.create_window((0, 0), window=rules_scrollable_frame, anchor=tk.NW)
        
        rules_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        rules_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Parameters section with scrollable frame (auto-sizing, starts at 0 height)
        ttk.Label(details_frame, text="Parameters", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5, pady=(5, 2))
        
        params_container = ttk.Frame(details_frame)
        params_container.grid(row=3, column=0, sticky=(tk.W, tk.E))
        params_container.columnconfigure(0, weight=1)
        
        params_canvas = tk.Canvas(params_container, height=0)
        params_scrollbar = ttk.Scrollbar(params_container, orient=tk.VERTICAL, command=params_canvas.yview)
        params_scrollable_frame = ttk.Frame(params_canvas)
        
        params_canvas.configure(yscrollcommand=params_scrollbar.set)
        params_canvas.create_window((0, 0), window=params_scrollable_frame, anchor=tk.NW)
        
        params_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        params_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Store references for updates
        instance_tab.rules_scrollable_frame = rules_scrollable_frame
        instance_tab.rules_canvas = rules_canvas
        instance_tab.params_scrollable_frame = params_scrollable_frame
        instance_tab.params_canvas = params_canvas
        instance_tab.selected_listbox = selected_listbox
        
        # Initialize parameter widgets
        self.update_parameter_widgets(username, selected_listbox)
        
        # Bind selection change to update details and parameter widgets
        def on_plan_selection(e):
            self.update_plan_details_inline(username, selected_listbox)
            self.update_parameter_widgets(username, selected_listbox)
        
        selected_listbox.bind('<<ListboxSelect>>', on_plan_selection)
        
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
        
        # Configure grid weights properly
        plan_runner_tab.rowconfigure(0, weight=0)  # Top row (config/controls) - fixed height
        plan_runner_tab.rowconfigure(1, weight=1)  # Main content - expandable
        plan_runner_tab.rowconfigure(5, weight=0)  # Status frame - fixed height
        
        # Store references for this instance
        instance_tab.available_listbox = available_listbox
        instance_tab.selected_listbox = selected_listbox
        instance_tab.session_dir = session_dir
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
        
        # Write rule parameters to file before starting plans
        self._write_rule_params_to_file(instance_name)
        
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
                    
                    # Update the display labels
                    instance_tab.current_plan_label.config(text=plan_name)
                    instance_tab.current_phase_label.config(text="Starting")
                    
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
                        
                        # Wait plan parameters
                        if plan_id == "wait_plan" and 'generic' in params and 'wait_minutes' in params['generic']:
                            param_args.extend(["--wait-minutes", str(params['generic']['wait_minutes'])])
                        
                        # Tutorial island parameters
                        if plan_id == "tutorial_island":
                            # Check if we have a credentials_file parameter
                            if 'generic' in params and 'credentials_file' in params['generic']:
                                param_args.extend(["--credentials-file", str(params['generic']['credentials_file'])])
                            # Auto-detect unnamed credentials for this instance
                            else:
                                # Find the credential name for this instance
                                cred_name = None
                                for i, selected_cred in enumerate(self.selected_credentials):
                                    # Extract username from credential filename (remove .properties)
                                    username = selected_cred.replace('.properties', '')
                                    if username == instance_name:
                                        cred_name = selected_cred
                                        break
                                
                                if cred_name and cred_name.startswith("unnamed_character_"):
                                    # Extract the credential name without .properties extension
                                    cred_name_without_ext = cred_name.replace('.properties', '')
                                    param_args.extend(["--credentials-file", cred_name_without_ext])
                                    self.log_message_to_instance(instance_name, f"Auto-detected unnamed credential: {cred_name_without_ext}", 'info')
                    
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
                    # Reset phase display
                    instance_tab.current_plan_label.config(text="None")
                    instance_tab.current_phase_label.config(text="Idle")
                    self.log_message_to_instance(instance_name, "All plans completed successfully", 'success')
                
            except Exception as e:
                instance_tab.status_label.config(text=f"Error: {str(e)}", style='Error.TLabel')
                # Reset phase display on error
                instance_tab.current_plan_label.config(text="Error")
                instance_tab.current_phase_label.config(text="Failed")
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
                cwd=Path(__file__).parent,  # Run from simple_recorder directory
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
                        
                        # Check for phase updates
                        if line.startswith("phase: "):
                            phase = line[7:].strip()  # Extract phase after "phase: "
                            self.update_instance_phase(instance_name, phase)
                        
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
            script_path = Path("/launch-runelite.ps1")
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
        elif plan_name == "wait_plan":
            self.configure_wait_parameters(instance_tab, plan_name)
        elif plan_name == "tutorial_island":
            self.configure_tutorial_island_parameters(instance_tab, plan_name)
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
    
    def configure_wait_parameters(self, instance_tab, plan_name):
        """Configure wait plan parameters."""
        instance_tab.params_display.config(text=f"Configure {plan_name} parameters:")
        
        # Wait time
        ttk.Label(instance_tab.params_frame, text="Wait Time (minutes):", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        wait_var = tk.StringVar(value="1.0")
        wait_spinbox = ttk.Spinbox(instance_tab.params_frame, from_=0.1, to=1440.0, increment=0.1, 
                                  textvariable=wait_var, width=10)
        wait_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        ttk.Label(instance_tab.params_frame, text="minutes").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Help text
        help_text = "How many minutes to wait before completing the plan (0.1 to 1440 minutes = 24 hours max)"
        ttk.Label(instance_tab.params_frame, text=help_text, style='Info.TLabel').grid(row=2, column=0, columnspan=3, pady=2)
        
        # Store references
        instance_tab.plan_params[plan_name] = {
            'wait_var': wait_var
        }
    
    def configure_tutorial_island_parameters(self, instance_tab, plan_name):
        """Configure tutorial island parameters."""
        instance_tab.params_display.config(text=f"Configure {plan_name} parameters:")
        
        # Credentials file parameter
        ttk.Label(instance_tab.params_frame, text="Credentials File:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        cred_var = tk.StringVar(value="")
        cred_entry = ttk.Entry(instance_tab.params_frame, textvariable=cred_var, width=30)
        cred_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Help text
        help_text = "Enter the filename (without .properties) to rename credentials file to match character name"
        ttk.Label(instance_tab.params_frame, text=help_text, style='Info.TLabel').grid(row=2, column=0, columnspan=2, pady=2)
        
        # Store references
        instance_tab.plan_params[plan_name] = {
            'cred_var': cred_var
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
        
        # Add to selected list (allow duplicates)
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
        self.update_plan_details_inline(username, selected_listbox)
    
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
    
    def clear_selected_plans(self, username, selected_listbox):
        """Clear all selected plans."""
        selected_listbox.delete(0, tk.END)
        
        # Clear plan_entries
        instance_tab = self.instance_tabs.get(username)
        if instance_tab:
            instance_tab.plan_entries.clear()
    
    def populate_sequences_list(self, username, sequences_listbox):
        """Populate the sequences listbox with saved sequences."""
        try:
            # Use absolute path based on script location
            sequences_dir = Path(__file__).parent / "plan_sequences"
            if sequences_dir.exists():
                sequences_listbox.delete(0, tk.END)
                for seq_file in sorted(sequences_dir.glob("*.json")):
                    # Extract sequence name from filename
                    seq_name = seq_file.stem
                    sequences_listbox.insert(tk.END, seq_name)
        except Exception as e:
            logging.error(f"Error populating sequences list: {e}")
    
    def load_sequence_from_list(self, username, sequences_listbox):
        """Load a sequence from the sequences listbox."""
        selection = sequences_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sequence to load.")
            return
        
        seq_name = sequences_listbox.get(selection[0])
        try:
            # Load the sequence file (use absolute path)
            seq_file = Path(__file__).parent / "plan_sequences" / f"{seq_name}.json"
            if seq_file.exists():
                import json
                with open(seq_file, 'r') as f:
                    sequence_data = json.load(f)
                
                # Clear current selection
                instance_tab = self.instance_tabs.get(username)
                if instance_tab and hasattr(instance_tab, 'selected_listbox'):
                    instance_tab.selected_listbox.delete(0, tk.END)
                    instance_tab.plan_entries.clear()
                
                # Load the sequence data
                # Validate version
                if sequence_data.get('version') != 1:
                    messagebox.showerror("Error", "Unsupported sequence file version.")
                    return
                
                # Update session directory and port
                if 'session_dir' in sequence_data:
                    instance_tab.session_dir.set(sequence_data['session_dir'])
                if 'port' in sequence_data:
                    self.instance_ports[username] = sequence_data['port']
                
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
                
                # Refresh sequences list
                self.populate_sequences_list(username, sequences_listbox)
                
                messagebox.showinfo("Success", f"Loaded sequence: {seq_name}")
            else:
                messagebox.showerror("Error", f"Sequence file not found: {seq_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sequence: {e}")
    
    def delete_sequence_from_list(self, username, sequences_listbox):
        """Delete a sequence from the sequences listbox."""
        selection = sequences_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sequence to delete.")
            return
        
        seq_name = sequences_listbox.get(selection[0])
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the sequence '{seq_name}'?"):
            try:
                seq_file = Path(__file__).parent / "plan_sequences" / f"{seq_name}.json"
                if seq_file.exists():
                    seq_file.unlink()  # Delete the file
                    
                    # Refresh sequences list
                    self.populate_sequences_list(username, sequences_listbox)
                    
                    messagebox.showinfo("Success", f"Deleted sequence: {seq_name}")
                else:
                    messagebox.showerror("Error", f"Sequence file not found: {seq_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete sequence: {e}")
    
    def add_rule_inline_advanced(self, username, selected_listbox, rule_type_var, rule_data, rules_scrollable_frame, rules_canvas):
        """Add a rule inline using advanced widgets (spinbox, combobox, etc.)."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to add a rule to.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab:
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            messagebox.showerror("Error", "Plan entry not found.")
            return
        
        plan_entry = instance_tab.plan_entries[index]
        rules = plan_entry.get('rules', {})
        rule_type = rule_type_var.get()
        
        try:
            if rule_type == "Time":
                widget1, _ = rule_data['Time']
                minutes = int(widget1.get())
                if minutes > 0:
                    rules['max_minutes'] = minutes
                    widget1.set("0")
            elif rule_type == "Skill":
                widget1, widget2 = rule_data['Skill']
                skill_name = widget1.get().strip()
                if skill_name:
                    level = int(widget2.get())
                    rules['stop_skill'] = skill_name
                    rules['stop_skill_level'] = level
                    widget1.set("")
                    widget2.set("1")
            elif rule_type == "Item":
                widget1, widget2 = rule_data['Item']
                item_name = widget1.get().strip()
                if item_name and item_name != "item name":
                    quantity = int(widget2.get())
                    if 'stop_items' not in rules:
                        rules['stop_items'] = []
                    rules['stop_items'].append({'name': item_name, 'qty': quantity})
                    widget1.delete(0, tk.END)
                    widget1.insert(0, "item name")
                    widget2.set("1")
            elif rule_type == "Total Level":
                widget1, _ = rule_data['Total Level']
                level = int(widget1.get())
                if level > 0:
                    rules['total_level'] = level
                    widget1.set("0")
            
            plan_entry['rules'] = rules
            self.update_plan_details_inline(username, selected_listbox)
            
            # Write rule parameters to file for StatsMonitor to read
            self._write_rule_params_to_file(username)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")
    
    def add_rule_inline(self, username, selected_listbox, rule_type_var, rule_value_entry, rules_scrollable_frame, rules_canvas):
        """Add a rule inline (without popup)."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to add a rule to.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab:
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            messagebox.showerror("Error", "Plan entry not found.")
            return
        
        plan_entry = instance_tab.plan_entries[index]
        rules = plan_entry.get('rules', {})
        rule_type = rule_type_var.get()
        rule_value = rule_value_entry.get().strip()
        
        try:
            if rule_type == "Time":
                minutes = int(rule_value)
                rules['max_minutes'] = minutes
                rule_value_entry.delete(0, tk.END)
                rule_value_entry.insert(0, "minutes")
            elif rule_type == "Skill":
                # Format: "skill_name level"
                parts = rule_value.split()
                if len(parts) >= 2:
                    skill_name = parts[0]
                    level = int(parts[-1])
                    rules['stop_skill'] = skill_name
                    rules['stop_skill_level'] = level
                    rule_value_entry.delete(0, tk.END)
                    rule_value_entry.insert(0, "skill_name level")
            elif rule_type == "Item":
                # Format: "item_name quantity"
                parts = rule_value.split()
                if len(parts) >= 2:
                    item_name = ' '.join(parts[:-1])
                    quantity = int(parts[-1])
                    if 'stop_items' not in rules:
                        rules['stop_items'] = []
                    rules['stop_items'].append({'name': item_name, 'qty': quantity})
                    rule_value_entry.delete(0, tk.END)
                    rule_value_entry.insert(0, "item_name quantity")
            elif rule_type == "Total Level":
                level = int(rule_value)
                rules['total_level'] = level
                rule_value_entry.delete(0, tk.END)
                rule_value_entry.insert(0, "level")
            
            plan_entry['rules'] = rules
            self.update_plan_details_inline(username, selected_listbox)
        except (ValueError, IndexError) as e:
            messagebox.showerror("Error", f"Invalid {rule_type.lower()} value: {rule_value}")
    
    def add_parameter_inline(self, username, selected_listbox, param_key_entry, param_value_entry, params_scrollable_frame, params_canvas):
        """Add a parameter inline (without popup)."""
        selection = selected_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plan to add a parameter to.")
            return
        
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab:
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            messagebox.showerror("Error", "Plan entry not found.")
            return
        
        plan_entry = instance_tab.plan_entries[index]
        params = plan_entry.get('params', {})
        param_key = param_key_entry.get().strip()
        param_value = param_value_entry.get().strip()
        
        if not param_key or not param_value:
            messagebox.showwarning("Invalid Input", "Both key and value are required.")
            return
        
        # Handle GE plans differently
        if 'ge' == plan_entry['name'].lower():
            messagebox.showinfo("Info", "GE plans require specialized parameter editing. Use the full editor for GE plans.")
            return
        
        # For generic plans, add to generic params
        if 'generic' not in params:
            params['generic'] = {}
        
        # Try to convert value to appropriate type
        try:
            # Try integer first
            param_value = int(param_value)
        except ValueError:
            try:
                # Try float
                param_value = float(param_value)
            except ValueError:
                # Keep as string
                pass
        
        params['generic'][param_key] = param_value
        plan_entry['params'] = params
        
        # Clear entry fields
        param_key_entry.delete(0, tk.END)
        param_key_entry.insert(0, "key")
        param_value_entry.delete(0, tk.END)
        param_value_entry.insert(0, "value")
        
        self.update_plan_details_inline(username, selected_listbox)
    
    def edit_plan_parameters(self, username, selected_listbox):
        """Edit parameters for the selected plan (legacy - kept for compatibility)."""
        # For now, show info that inline editing is preferred
        messagebox.showinfo("Info", "Please use the inline editing controls in the Plan Details section.")
    
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
                self.update_plan_details_inline(username, selected_listbox)
    
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
                self.update_plan_details_inline(username, selected_listbox)
                
                # Write rule parameters to file for StatsMonitor to read
                self._write_rule_params_to_file(username)
    
    def update_plan_details_inline(self, username, selected_listbox):
        """Update the details panel for the selected plan (inline display with labels)."""
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab or not hasattr(instance_tab, 'rules_scrollable_frame'):
            return
            
        selection = selected_listbox.curselection()
        
        # Clear existing details
        for widget in instance_tab.rules_scrollable_frame.winfo_children():
            widget.destroy()
        for widget in instance_tab.params_scrollable_frame.winfo_children():
            widget.destroy()
        
        if not selection:
            # Keep sections at 0 height when no plan selected
            instance_tab.rules_scrollable_frame.update_idletasks()
            instance_tab.rules_canvas.configure(scrollregion="0 0 0 0", height=0)
            instance_tab.params_scrollable_frame.update_idletasks()
            instance_tab.params_canvas.configure(scrollregion="0 0 0 0", height=0)
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            return
        
        plan_entry = instance_tab.plan_entries[index]
        
        # Display rules as simple labels
        row = 0
        rules = plan_entry.get('rules', {})
        if rules.get('max_minutes'):
            rule_label = ttk.Label(instance_tab.rules_scrollable_frame, 
                                   text=f"Max Time: {rules['max_minutes']} minutes", font=("Arial", 8))
            rule_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
            row += 1
        
        if rules.get('stop_skill'):
            skill_name = rules.get('stop_skill', '')
            skill_level = rules.get('stop_skill_level', 0)
            rule_label = ttk.Label(instance_tab.rules_scrollable_frame, 
                                   text=f"Stop at Skill: {skill_name} level {skill_level}", font=("Arial", 8))
            rule_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
            row += 1
        
        if rules.get('total_level'):
            rule_label = ttk.Label(instance_tab.rules_scrollable_frame, 
                                   text=f"Total Level: {rules['total_level']}", font=("Arial", 8))
            rule_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
            row += 1
        
        if rules.get('stop_items'):
            for item in rules['stop_items']:
                rule_label = ttk.Label(instance_tab.rules_scrollable_frame, 
                                      text=f"Stop with Item: {item['name']} x{item['qty']}", font=("Arial", 8))
                rule_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
                row += 1
        
        if row == 0:
            # Don't show "No rules set" - keep section at 0 height when empty
            pass
        
        # Update rules canvas scroll region and resize based on content
        instance_tab.rules_scrollable_frame.update_idletasks()
        bbox = instance_tab.rules_canvas.bbox("all")
        if bbox and bbox[3] > bbox[1]:
            # Has content - calculate height based on content
            content_height = bbox[3] - bbox[1] + 4
            instance_tab.rules_canvas.configure(scrollregion=bbox, height=min(content_height, 150))
        else:
            # No content - set to 0 height
            instance_tab.rules_canvas.configure(scrollregion="0 0 0 0", height=0)
        
        # Display parameters as simple labels
        row = 0
        params = plan_entry.get('params', {})
        plan_index = index  # Capture for closure
        
        if 'ge' == plan_entry['name'].lower():
            if params.get('buy_items'):
                for idx, item in enumerate(params['buy_items']):
                    # Create frame for each item with delete button
                    item_frame = ttk.Frame(instance_tab.params_scrollable_frame)
                    item_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=5, pady=1)
                    
                    param_label = ttk.Label(item_frame, 
                                           text=f"Buy: {item['name']} x{item['quantity']} (bumps: {item['bumps']}, price: {item['set_price']})", 
                                           font=("Arial", 8))
                    param_label.grid(row=0, column=0, sticky=tk.W)
                    
                    # Delete button for this item
                    def delete_buy_item(item_idx=idx, plan_idx=plan_index):
                        if plan_idx < len(instance_tab.plan_entries):
                            params = instance_tab.plan_entries[plan_idx].get('params', {})
                            if 'buy_items' in params and item_idx < len(params['buy_items']):
                                params['buy_items'].pop(item_idx)
                                instance_tab.plan_entries[plan_idx]['params'] = params
                                self.update_plan_details_inline(username, selected_listbox)
                    
                    ttk.Button(item_frame, text="✕", width=2, command=delete_buy_item).grid(row=0, column=1, padx=(5, 0))
                    row += 1
            
            if params.get('sell_items'):
                for idx, item in enumerate(params['sell_items']):
                    # Create frame for each item with delete button
                    item_frame = ttk.Frame(instance_tab.params_scrollable_frame)
                    item_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=5, pady=1)
                    
                    param_label = ttk.Label(item_frame, 
                                           text=f"Sell: {item['name']} x{item['quantity']} (bumps: {item['bumps']}, price: {item['set_price']})", 
                                           font=("Arial", 8))
                    param_label.grid(row=0, column=0, sticky=tk.W)
                    
                    # Delete button for this item
                    def delete_sell_item(item_idx=idx, plan_idx=plan_index):
                        if plan_idx < len(instance_tab.plan_entries):
                            params = instance_tab.plan_entries[plan_idx].get('params', {})
                            if 'sell_items' in params and item_idx < len(params['sell_items']):
                                params['sell_items'].pop(item_idx)
                                instance_tab.plan_entries[plan_idx]['params'] = params
                                self.update_plan_details_inline(username, selected_listbox)
                    
                    ttk.Button(item_frame, text="✕", width=2, command=delete_sell_item).grid(row=0, column=1, padx=(5, 0))
                    row += 1
        else:
            if params.get('generic'):
                for key, value in params['generic'].items():
                    param_label = ttk.Label(instance_tab.params_scrollable_frame, 
                                           text=f"{key}: {value}", font=("Arial", 8))
                    param_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=1)
                    row += 1
        
        if row == 0:
            # Don't show "No parameters set" - keep section at 0 height when empty
            pass
        
        # Update params canvas scroll region and resize based on content
        instance_tab.params_scrollable_frame.update_idletasks()
        bbox = instance_tab.params_canvas.bbox("all")
        if bbox and bbox[3] > bbox[1]:
            # Has content - calculate height based on content
            content_height = bbox[3] - bbox[1] + 4
            instance_tab.params_canvas.configure(scrollregion=bbox, height=min(content_height, 150))
        else:
            # No content - set to 0 height
            instance_tab.params_canvas.configure(scrollregion="0 0 0 0", height=0)
    
    def update_parameter_widgets(self, username, selected_listbox):
        """Update parameter editing widgets based on selected plan type."""
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab or not hasattr(instance_tab, 'params_edit_frame'):
            return
        
        selection = selected_listbox.curselection()
        if not selection:
            # Show generic widgets
            self._setup_generic_param_widgets(username, selected_listbox)
            return
        
        index = selection[0]
        if index >= len(instance_tab.plan_entries):
            return
        
        plan_entry = instance_tab.plan_entries[index]
        plan_name = plan_entry['name'].lower()
        
        # Clear existing widgets
        for widget in instance_tab.params_edit_frame.winfo_children():
            widget.destroy()
        
        if 'ge' == plan_name:
            self._setup_ge_param_widgets(username, selected_listbox)
        elif plan_name == 'ge_trade':
            self._setup_ge_trade_param_widgets(username, selected_listbox)
        elif plan_name == 'wait_plan':
            self._setup_wait_param_widgets(username, selected_listbox)
        elif plan_name == 'tutorial_island':
            self._setup_tutorial_island_param_widgets(username, selected_listbox)
        else:
            self._setup_generic_param_widgets(username, selected_listbox)
    
    def _setup_generic_param_widgets(self, username, selected_listbox):
        """Setup generic parameter editing widgets (key/value entries)."""
        instance_tab = self.instance_tabs.get(username)
        
        param_key_entry = ttk.Entry(instance_tab.params_edit_frame, width=15)
        param_key_entry.grid(row=0, column=0, padx=(0, 5))
        param_key_entry.insert(0, "key")
        
        param_value_entry = ttk.Entry(instance_tab.params_edit_frame, width=15)
        param_value_entry.grid(row=0, column=1, padx=(0, 5))
        param_value_entry.insert(0, "value")
        
        ttk.Button(instance_tab.params_edit_frame, text="Add", width=8,
                  command=lambda: self.add_parameter_inline(username, selected_listbox, param_key_entry, param_value_entry,
                                                            instance_tab.params_scrollable_frame, instance_tab.params_canvas)).grid(row=0, column=2)
    
    def _setup_ge_param_widgets(self, username, selected_listbox):
        """Setup GE-specific parameter widgets (buy/sell items)."""
        instance_tab = self.instance_tabs.get(username)
        
        # Type selector (Buy or Sell)
        ttk.Label(instance_tab.params_edit_frame, text="Type:", font=("Arial", 8)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ge_type_var = tk.StringVar(value="Buy")
        ge_type_combo = ttk.Combobox(instance_tab.params_edit_frame, textvariable=ge_type_var, 
                                     values=["Buy", "Sell"], width=8, state="readonly")
        ge_type_combo.grid(row=0, column=1, padx=(0, 5))
        
        # Item name
        ttk.Label(instance_tab.params_edit_frame, text="Item:", font=("Arial", 8)).grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        ge_item_name_entry = ttk.Entry(instance_tab.params_edit_frame, width=15)
        ge_item_name_entry.grid(row=0, column=3, padx=(5, 0))
        ge_item_name_entry.insert(0, "item name")
        
        # Quantity (allow -1 for sell all)
        ttk.Label(instance_tab.params_edit_frame, text="Qty:", font=("Arial", 8)).grid(row=0, column=4, sticky=tk.W, padx=(5, 0))
        ge_qty_spinbox = ttk.Spinbox(instance_tab.params_edit_frame, from_=-1, to=99999, width=8)
        ge_qty_spinbox.set("1")
        ge_qty_spinbox.grid(row=0, column=5, padx=(5, 0))
        ttk.Label(instance_tab.params_edit_frame, text="(-1=all)", font=("Arial", 7), foreground="gray").grid(row=0, column=6, padx=(2, 0))
        
        # Bumps
        ttk.Label(instance_tab.params_edit_frame, text="Bumps:", font=("Arial", 8)).grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ge_bumps_spinbox = ttk.Spinbox(instance_tab.params_edit_frame, from_=0, to=100, width=8)
        ge_bumps_spinbox.set("0")
        ge_bumps_spinbox.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        
        # Price
        ttk.Label(instance_tab.params_edit_frame, text="Price:", font=("Arial", 8)).grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        ge_price_spinbox = ttk.Spinbox(instance_tab.params_edit_frame, from_=1, to=2147483647, width=12)
        ge_price_spinbox.set("1")
        ge_price_spinbox.grid(row=1, column=3, padx=(5, 0), pady=(5, 0))
        
        # Add button
        def add_ge_item():
            selection = selected_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a plan to add an item to.")
                return
            
            index = selection[0]
            if index >= len(instance_tab.plan_entries):
                return
            
            plan_entry = instance_tab.plan_entries[index]
            params = plan_entry.get('params', {})
            
            item_type = ge_type_var.get().lower()  # "buy" or "sell"
            item_name = ge_item_name_entry.get().strip()
            
            if not item_name or item_name == "item name":
                messagebox.showwarning("Invalid Input", "Please enter an item name.")
                return
            
            try:
                qty_str = ge_qty_spinbox.get().strip()
                quantity = int(qty_str) if qty_str else 1
                bumps = int(ge_bumps_spinbox.get())
                price_str = ge_price_spinbox.get().strip()
                price = int(price_str) if price_str else 1
                if quantity < -1:
                    quantity = -1
            except ValueError:
                messagebox.showerror("Error", "Invalid numeric value.")
                return
            
            # Initialize lists if needed
            if f'{item_type}_items' not in params:
                params[f'{item_type}_items'] = []
            
            # Add the item
            item_data = {
                'name': item_name,
                'quantity': quantity,
                'bumps': bumps,
                'set_price': price
            }
            params[f'{item_type}_items'].append(item_data)
            plan_entry['params'] = params
            
            # Clear entry fields
            ge_item_name_entry.delete(0, tk.END)
            ge_item_name_entry.insert(0, "item name")
            ge_qty_spinbox.set("1")
            ge_bumps_spinbox.set("0")
            ge_price_spinbox.set("1")
            
            # Update display
            self.update_plan_details_inline(username, selected_listbox)
        
        ttk.Button(instance_tab.params_edit_frame, text="Add", width=8, command=add_ge_item).grid(row=1, column=4, padx=(5, 0), pady=(5, 0))
    
    def _setup_ge_trade_param_widgets(self, username, selected_listbox):
        """Setup GE trade parameter widgets."""
        instance_tab = self.instance_tabs.get(username)
        
        ttk.Label(instance_tab.params_edit_frame, text="Role:", font=("Arial", 8)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        role_var = tk.StringVar(value="worker")
        ttk.Radiobutton(instance_tab.params_edit_frame, text="Worker", variable=role_var, value="worker").grid(row=0, column=1, padx=(0, 5))
        ttk.Radiobutton(instance_tab.params_edit_frame, text="Mule", variable=role_var, value="mule").grid(row=0, column=2, padx=(0, 5))
        
        def add_role_param():
            selection = selected_listbox.curselection()
            if not selection:
                return
            index = selection[0]
            if index < len(instance_tab.plan_entries):
                plan_entry = instance_tab.plan_entries[index]
                params = plan_entry.get('params', {})
                params['role'] = role_var.get()
                plan_entry['params'] = params
                self.update_plan_details_inline(username, selected_listbox)
        
        ttk.Button(instance_tab.params_edit_frame, text="Set", width=8, command=add_role_param).grid(row=0, column=3)
    
    def _setup_wait_param_widgets(self, username, selected_listbox):
        """Setup wait plan parameter widgets."""
        instance_tab = self.instance_tabs.get(username)
        
        ttk.Label(instance_tab.params_edit_frame, text="Wait Time:", font=("Arial", 8)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        wait_spinbox = ttk.Spinbox(instance_tab.params_edit_frame, from_=0.1, to=1440.0, increment=0.1, width=10)
        wait_spinbox.set("1.0")
        wait_spinbox.grid(row=0, column=1, padx=(0, 5))
        ttk.Label(instance_tab.params_edit_frame, text="minutes", font=("Arial", 8)).grid(row=0, column=2, padx=(0, 5))
        
        def add_wait_param():
            selection = selected_listbox.curselection()
            if not selection:
                return
            index = selection[0]
            if index < len(instance_tab.plan_entries):
                plan_entry = instance_tab.plan_entries[index]
                params = plan_entry.get('params', {})
                if 'generic' not in params:
                    params['generic'] = {}
                params['generic']['wait_minutes'] = float(wait_spinbox.get())
                plan_entry['params'] = params
                self.update_plan_details_inline(username, selected_listbox)
        
        ttk.Button(instance_tab.params_edit_frame, text="Set", width=8, command=add_wait_param).grid(row=0, column=3)
    
    def _setup_tutorial_island_param_widgets(self, username, selected_listbox):
        """Setup tutorial island parameter widgets."""
        instance_tab = self.instance_tabs.get(username)
        
        ttk.Label(instance_tab.params_edit_frame, text="Credentials File:", font=("Arial", 8)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        cred_entry = ttk.Entry(instance_tab.params_edit_frame, width=20)
        cred_entry.grid(row=0, column=1, padx=(0, 5))
        cred_entry.insert(0, "filename (no .properties)")
        
        def add_cred_param():
            selection = selected_listbox.curselection()
            if not selection:
                return
            index = selection[0]
            if index < len(instance_tab.plan_entries):
                plan_entry = instance_tab.plan_entries[index]
                params = plan_entry.get('params', {})
                if 'generic' not in params:
                    params['generic'] = {}
                params['generic']['credentials_file'] = cred_entry.get().strip()
                plan_entry['params'] = params
                self.update_plan_details_inline(username, selected_listbox)
        
        ttk.Button(instance_tab.params_edit_frame, text="Set", width=8, command=add_cred_param).grid(row=0, column=2)
    
    def update_plan_details(self, username, selected_listbox, rules_tree, params_tree):
        """Update the details panel for the selected plan (legacy method for compatibility)."""
        # Redirect to inline version
        self.update_plan_details_inline(username, selected_listbox)
        self.update_parameter_widgets(username, selected_listbox)
    
    def save_sequence_for_instance(self, username):
        """Save the current plan sequence to a JSON file."""
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab or not instance_tab.plan_entries:
            messagebox.showwarning("No Plans", "No plans selected to save.")
            return
        
        # Default to plan_sequences directory
        default_dir = Path(__file__).parent / "plan_sequences"
        default_dir.mkdir(parents=True, exist_ok=True)
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Plan Sequence",
            initialdir=str(default_dir)
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
            
            # Refresh sequences list if it exists
            if hasattr(instance_tab, 'sequences_listbox'):
                self.populate_sequences_list(username, instance_tab.sequences_listbox)
            
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
            
            # Stop stats monitor for this instance
            self.stop_stats_monitor(instance_name)
    
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
