"""
Plan Editor Module
==================

Modal editor for plan rules and parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, TypedDict


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
