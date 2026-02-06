"""
Instance Manager Module
=======================

Manages instance tabs, plan execution, and instance lifecycle.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Optional, Callable
import time
import logging
import subprocess
import threading
import sys
import os
import json
from pathlib import Path
from datetime import datetime

from run_rj_loop import AVAILABLE_PLANS
from gui.plan_editor import PlanEntry
from tkinter import filedialog, messagebox


class InstanceManager:
    """Manages RuneLite instance tabs and plan execution."""
    
    def __init__(self, root, notebook: ttk.Notebook, instance_tabs: Dict, 
                 instance_ports: Dict, log_message_callback: Callable,
                 log_message_to_instance_callback: Callable = None,
                 update_stats_text_callback: Callable = None,
                 start_stats_monitor_callback: Callable = None,
                 update_statistics_display_callback: Callable = None,
                 start_statistics_timer_callback: Callable = None,
                 stop_statistics_timer_callback: Callable = None,
                 update_plan_details_callback: Callable = None,
                 update_parameter_widgets_callback: Callable = None,
                 selected_credentials: List[str] = None,
                 base_completion_patterns: List[str] = None):
        """
        Initialize instance manager.
        
        Args:
            root: Root tkinter window
            notebook: Main notebook widget for tabs
            instance_tabs: Dictionary to store instance tab references
            instance_ports: Dictionary mapping instance names to ports
            log_message_callback: Callback function for logging messages
            log_message_to_instance_callback: Callback for instance-specific logging
            update_stats_text_callback: Callback to update stats display
            start_stats_monitor_callback: Callback to start stats monitor
            update_statistics_display_callback: Callback to update statistics display
            start_statistics_timer_callback: Callback to start statistics timer
            stop_statistics_timer_callback: Callback to stop statistics timer
            update_plan_details_callback: Callback to update plan details
            update_parameter_widgets_callback: Callback to update parameter widgets
            selected_credentials: List of selected credential filenames
            base_completion_patterns: Base completion patterns for plans
        """
        self.root = root
        self.notebook = notebook
        self.instance_tabs = instance_tabs
        self.instance_ports = instance_ports
        self.log_message = log_message_callback
        self.log_message_to_instance = log_message_to_instance_callback or (lambda name, msg, level='info': None)
        self.update_stats_text = update_stats_text_callback or (lambda name: None)
        self.start_stats_monitor = start_stats_monitor_callback or (lambda name, port: None)
        self.update_statistics_display = update_statistics_display_callback or (lambda name: None)
        self.start_statistics_timer = start_statistics_timer_callback or (lambda name: None)
        self.stop_statistics_timer = stop_statistics_timer_callback or (lambda name: None)
        self.update_plan_details = update_plan_details_callback or (lambda name, listbox: None)
        self.update_parameter_widgets = update_parameter_widgets_callback or (lambda name, listbox: None)
        self.selected_credentials = selected_credentials or []
        self.base_completion_patterns = base_completion_patterns or ['quest complete', 'quest completed']
    
    def create_instance_tab(self, instance_name: str, port: int,
                           browse_directory_callback: Callable = None,
                           populate_sequences_callback: Callable = None,
                           save_sequence_callback: Callable = None,
                           load_sequence_callback: Callable = None,
                           delete_sequence_callback: Callable = None,
                           add_plan_callback: Callable = None,
                           remove_plan_callback: Callable = None,
                           move_plan_up_callback: Callable = None,
                           move_plan_down_callback: Callable = None,
                           clear_plans_callback: Callable = None,
                           update_plan_details_callback: Callable = None,
                           update_parameter_widgets_callback: Callable = None,
                           add_rule_callback: Callable = None,
                           clear_params_callback: Callable = None,
                           clear_rules_callback: Callable = None):
        """
        Create a new instance tab in the notebook.
        
        Creates all UI widgets for an instance tab including:
        - Main instance tab with sub-notebook (Plan Runner, Output, Statistics)
        - Plan selection UI (available/selected plans)
        - Plan details UI (rules/parameters editing)
        - Statistics display (skills/inventory/equipment)
        - Control buttons (start/stop/pause)
        - Log output area
        """
        # Check if tab already exists
        if instance_name in self.instance_tabs:
            self.log_message(f"Instance tab already exists: {instance_name}", 'info')
            self.instance_ports[instance_name] = port
            return self.instance_tabs[instance_name]
        
        self.log_message(f"Creating instance tab: {instance_name}", 'info')
        
        # Create the main instance tab
        instance_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(instance_tab, text=instance_name)
        instance_tab.columnconfigure(0, weight=1)
        instance_tab.rowconfigure(0, weight=1)
        
        # Store reference
        self.instance_tabs[instance_name] = instance_tab
        self.instance_ports[instance_name] = port
        
        self.log_message(f"Instance tab created and stored: {instance_name}", 'info')
        
        # Create sub-notebook for Plan Runner, Output, and Statistics
        sub_notebook = ttk.Notebook(instance_tab)
        sub_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        
        # Plan Runner sub-tab content - Top row: Config + Skills + Inventory + Equipment
        top_row = ttk.Frame(plan_runner_tab, padding="10")
        top_row.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        top_row.columnconfigure(0, weight=0)  # Config - no expansion
        top_row.columnconfigure(1, weight=1)  # Skills
        top_row.columnconfigure(2, weight=1)  # Inventory
        top_row.columnconfigure(3, weight=1)  # Equipment
        
        # Left side: Session config
        config_frame = ttk.Frame(top_row)
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Session Directory
        ttk.Label(config_frame, text="Session Dir:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        dir_frame = ttk.Frame(config_frame)
        dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        dir_frame.columnconfigure(0, weight=1)
        
        session_dir = tk.StringVar(value=f"D:\\bots\\exports\\{instance_name.lower()}")
        dir_entry = ttk.Entry(dir_frame, textvariable=session_dir, width=25)
        dir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 3))
        
        # Browse button - use callback if provided, otherwise use default
        browse_cmd = browse_directory_callback or (lambda: self._browse_directory_for_instance(instance_name, session_dir))
        ttk.Button(dir_frame, text="Browse", command=lambda: browse_cmd()).grid(row=0, column=1)
        
        # Port (read-only for instances)
        ttk.Label(config_frame, text="Port:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        port_label = ttk.Label(config_frame, text=str(port), style='Info.TLabel')
        port_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Credential file (read-only for instances)
        ttk.Label(config_frame, text="Credential:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        cred_file_name = None
        for selected_cred in self.selected_credentials:
            cred_username = selected_cred.replace('.properties', '')
            if cred_username == instance_name:
                cred_file_name = selected_cred
                break
        
        if cred_file_name:
            cred_label = ttk.Label(config_frame, text=cred_file_name, style='Info.TLabel')
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
        
        # Control buttons
        control_buttons_frame = ttk.Frame(config_frame)
        control_buttons_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Start button
        start_button = tk.Button(control_buttons_frame, text="▶", bg="#4CAF50", fg="white", 
                                font=("Arial", 10, "bold"), width=2, height=1,
                                command=lambda: self.start_plans_for_instance(instance_name, session_dir.get(), port))
        start_button.grid(row=0, column=0, padx=(0, 5))
        
        # Stop button
        stop_button = tk.Button(control_buttons_frame, text="■", bg="#F44336", fg="white",
                               font=("Arial", 10, "bold"), width=2, height=1,
                               command=lambda: self.stop_plans_for_instance(instance_name))
        stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # Pause between plans checkbox
        pause_var = tk.BooleanVar(value=False)
        pause_checkbox = ttk.Checkbutton(control_buttons_frame, text="Pause between plans", variable=pause_var)
        pause_checkbox.grid(row=0, column=2, padx=(10, 0))
        instance_tab.pause_var = pause_var
        
        # Key Items Totals section (shown when bank is open)
        key_items_frame = ttk.Frame(config_frame)
        key_items_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
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
        
        # Skills, Inventory, Equipment sections
        skills_container, skills_scrollable_frame, skills_canvas = create_stats_section(top_row, 1)
        inventory_container, inventory_scrollable_frame, inventory_canvas = create_stats_section(top_row, 2)
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
        self.start_stats_monitor(instance_name, port)
        
        # Initial stats update
        self.update_stats_text(instance_name)
        
        # Main content: Left column (plans + details), Right column (stats)
        main_content = ttk.Frame(plan_runner_tab)
        main_content.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        main_content.columnconfigure(0, weight=1)
        main_content.columnconfigure(1, weight=1)
        plan_runner_tab.rowconfigure(1, weight=1)
        
        # Left column: Plan selection (top) and saved sequences (bottom)
        left_column = ttk.Frame(main_content)
        left_column.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_column.columnconfigure(0, weight=1)
        left_column.rowconfigure(0, weight=1)
        left_column.rowconfigure(1, weight=0)
        
        # Left side: Plan selection
        left_frame = ttk.LabelFrame(left_column, text="Plan Selection", padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=0)
        left_frame.columnconfigure(2, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # Available plans
        ttk.Label(left_frame, text="Available Plans:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        available_listbox = tk.Listbox(left_frame, height=6)
        available_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(0, 5))
        
        # Populate available plans
        for plan_id, plan_class in AVAILABLE_PLANS.items():
            label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
            available_listbox.insert(tk.END, f"{label} ({plan_id})")
        
        # Control buttons (middle column)
        plan_controls = ttk.Frame(left_frame)
        plan_controls.grid(row=1, column=1, sticky=(tk.N, tk.S), padx=5)
        
        # Use callbacks if provided, otherwise use instance manager methods
        add_plan_fn = add_plan_callback or (lambda: self._add_plan_to_selection(instance_name, available_listbox, selected_listbox))
        remove_plan_fn = remove_plan_callback or (lambda: self._remove_plan_from_selection(instance_name, selected_listbox))
        move_up_fn = move_plan_up_callback or (lambda: self._move_plan_up(instance_name, selected_listbox))
        move_down_fn = move_plan_down_callback or (lambda: self._move_plan_down(instance_name, selected_listbox))
        clear_fn = clear_plans_callback or (lambda: self._clear_selected_plans(instance_name, selected_listbox))
        
        ttk.Button(plan_controls, text="→", command=add_plan_fn).grid(row=0, column=0, pady=(0, 3))
        ttk.Button(plan_controls, text="←", command=remove_plan_fn).grid(row=1, column=0, pady=3)
        ttk.Button(plan_controls, text="↑", command=move_up_fn).grid(row=2, column=0, pady=3)
        ttk.Button(plan_controls, text="↓", command=move_down_fn).grid(row=3, column=0, pady=3)
        ttk.Button(plan_controls, text="✕", command=clear_fn).grid(row=4, column=0, pady=(10, 0))
        
        # Selected plans
        ttk.Label(left_frame, text="Selected Plans:", style='Header.TLabel').grid(row=0, column=2, sticky=tk.W, pady=(0, 5))
        selected_listbox = tk.Listbox(left_frame, height=6)
        selected_listbox.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(5, 0))
        
        # Saved Sequences
        sequences_frame = ttk.LabelFrame(left_column, text="Saved Sequences", padding="5")
        sequences_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        sequences_frame.columnconfigure(0, weight=1)
        sequences_frame.rowconfigure(1, weight=1)
        
        sequences_listbox = tk.Listbox(sequences_frame, height=4)
        sequences_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        seq_controls = ttk.Frame(sequences_frame)
        seq_controls.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Use callbacks if provided
        save_seq_fn = save_sequence_callback or (lambda: self._save_sequence_for_instance(instance_name))
        load_seq_fn = load_sequence_callback or (lambda: self._load_sequence_from_list(instance_name, sequences_listbox))
        delete_seq_fn = delete_sequence_callback or (lambda: self._delete_sequence_from_list(instance_name, sequences_listbox))
        populate_seq_fn = populate_sequences_callback or (lambda: self._populate_sequences_list(instance_name, sequences_listbox))
        
        ttk.Button(seq_controls, text="Save", command=save_seq_fn).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(seq_controls, text="Load", command=load_seq_fn).grid(row=0, column=1, padx=3)
        ttk.Button(seq_controls, text="Delete", command=delete_seq_fn).grid(row=0, column=2, padx=3)
        
        populate_seq_fn()
        instance_tab.sequences_listbox = sequences_listbox
        
        # Plan Details moved to right column
        right_frame = ttk.LabelFrame(main_content, text="Plan Details", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Details controls
        details_controls = ttk.Frame(right_frame)
        details_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        clear_params_fn = clear_params_callback or (lambda: self._clear_plan_parameters(instance_name, selected_listbox))
        clear_rules_fn = clear_rules_callback or (lambda: self._clear_plan_rules(instance_name, selected_listbox))
        
        ttk.Button(details_controls, text="Clear Parameters", command=clear_params_fn).grid(row=0, column=0, padx=3)
        ttk.Button(details_controls, text="Clear Rules", command=clear_rules_fn).grid(row=0, column=1, padx=3)
        
        # Editing section - inline editing for rules and parameters
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
        
        # Dynamic rule input frame
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
        show_rule_input()
        
        # Add rule button - use callback if provided
        def add_rule_cmd():
            rule_data = {
                'Time': (time_spinbox, None),
                'Skill': (skill_combo, skill_level_spinbox),
                'Item': (item_name_entry, item_qty_spinbox),
                'Total Level': (total_level_spinbox, None)
            }
            add_rule_fn = add_rule_callback or (lambda: self._add_rule_inline_advanced(instance_name, selected_listbox, rule_type_var, rule_data, rules_scrollable_frame, rules_canvas))
            add_rule_fn()
        
        ttk.Button(rules_edit_frame, text="Add", width=8, command=add_rule_cmd).grid(row=0, column=2)
        
        # Parameters editing - dynamic based on plan type
        ttk.Label(editing_frame, text="Add Parameter:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        params_edit_container = ttk.Frame(editing_frame)
        params_edit_container.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        params_edit_container.columnconfigure(0, weight=1)
        
        params_edit_frame = ttk.Frame(params_edit_container)
        params_edit_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        instance_tab.params_edit_container = params_edit_container
        instance_tab.params_edit_frame = params_edit_frame
        
        # Details display - scrollable sections
        details_frame = ttk.Frame(right_frame)
        details_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        
        # Rules section
        ttk.Label(details_frame, text="Rules", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 2))
        
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
        
        # Parameters section
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
        
        # Store references
        instance_tab.rules_scrollable_frame = rules_scrollable_frame
        instance_tab.rules_canvas = rules_canvas
        instance_tab.params_scrollable_frame = params_scrollable_frame
        instance_tab.params_canvas = params_canvas
        instance_tab.selected_listbox = selected_listbox
        
        # Initialize parameter widgets
        update_param_fn = update_parameter_widgets_callback or (lambda: self.update_parameter_widgets(instance_name, selected_listbox))
        update_param_fn()
        
        # Bind selection change to update details and parameter widgets
        def on_plan_selection(e):
            update_details_fn = update_plan_details_callback or (lambda: self._update_plan_details_inline(instance_name, selected_listbox))
            update_details_fn()
            update_param_fn()
        
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
        
        # Configure grid weights
        plan_runner_tab.rowconfigure(0, weight=0)
        plan_runner_tab.rowconfigure(1, weight=1)
        plan_runner_tab.rowconfigure(5, weight=0)
        
        # Store references
        instance_tab.available_listbox = available_listbox
        instance_tab.selected_listbox = selected_listbox
        instance_tab.session_dir = session_dir
        instance_tab.plan_entries = []
        instance_tab.is_running = False
        instance_tab.current_plan_index = 0
        instance_tab.current_process = None
        instance_tab.current_plan_name = "None"
        instance_tab.current_phase = "Idle"
        instance_tab.start_time = None
        
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
        self.update_statistics_display(instance_name)
        
        return instance_tab
    
    # Helper methods that need to be implemented (stubs for now)
    def _browse_directory_for_instance(self, instance_name: str, session_dir_var: tk.StringVar):
        """Browse for directory for a specific instance."""
        directory = filedialog.askdirectory(initialdir=session_dir_var.get())
        if directory:
            session_dir_var.set(directory)
    
    def _add_plan_to_selection(self, instance_name: str, available_listbox: tk.Listbox, selected_listbox: tk.Listbox):
        """Add selected plan from available to selected."""
        selected_indices = available_listbox.curselection()
        for index in reversed(selected_indices):
            plan_name = available_listbox.get(index)
            selected_listbox.insert(tk.END, plan_name)
            
            # Create PlanEntry for this plan
            plan_id = plan_name
            if '(' in plan_name and ')' in plan_name:
                plan_id = plan_name.split('(')[-1].rstrip(')')
            
            plan_class = AVAILABLE_PLANS.get(plan_id)
            if plan_class:
                label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
                plan_entry = PlanEntry(
                    name=plan_id,
                    label=label,
                    rules={'max_minutes': None, 'stop_skill': None, 'stop_items': [], 'total_level': None},
                    params={'generic': {}}
                )
                instance_tab = self.instance_tabs.get(instance_name)
                if instance_tab:
                    instance_tab.plan_entries.append(plan_entry)
    
    def _remove_plan_from_selection(self, instance_name: str, selected_listbox: tk.Listbox):
        """Remove selected plan from selection."""
        selection = selected_listbox.curselection()
        if not selection:
            return
        
        selected_listbox.delete(selection[0])
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and selection[0] < len(instance_tab.plan_entries):
            del instance_tab.plan_entries[selection[0]]
    
    def _move_plan_up(self, instance_name: str, selected_listbox: tk.Listbox):
        """Move selected plan up."""
        selection = selected_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        
        index = selection[0]
        item = selected_listbox.get(index)
        selected_listbox.delete(index)
        selected_listbox.insert(index - 1, item)
        selected_listbox.selection_set(index - 1)
        
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and index < len(instance_tab.plan_entries):
            instance_tab.plan_entries[index], instance_tab.plan_entries[index - 1] = instance_tab.plan_entries[index - 1], instance_tab.plan_entries[index]
    
    def _move_plan_down(self, instance_name: str, selected_listbox: tk.Listbox):
        """Move selected plan down."""
        selection = selected_listbox.curselection()
        if not selection or selection[0] == selected_listbox.size() - 1:
            return
        
        index = selection[0]
        item = selected_listbox.get(index)
        selected_listbox.delete(index)
        selected_listbox.insert(index + 1, item)
        selected_listbox.selection_set(index + 1)
        
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and index < len(instance_tab.plan_entries) - 1:
            instance_tab.plan_entries[index], instance_tab.plan_entries[index + 1] = instance_tab.plan_entries[index + 1], instance_tab.plan_entries[index]
    
    def _clear_selected_plans(self, instance_name: str, selected_listbox: tk.Listbox):
        """Clear all selected plans."""
        selected_listbox.delete(0, tk.END)
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab:
            instance_tab.plan_entries.clear()
    
    def _populate_sequences_list(self, instance_name: str, sequences_listbox: tk.Listbox):
        """Populate sequences listbox."""
        try:
            sequences_dir = Path(__file__).resolve().parent.parent / "plan_sequences"
            if sequences_dir.exists():
                sequences_listbox.delete(0, tk.END)
                for seq_file in sorted(sequences_dir.glob("*.json")):
                    sequences_listbox.insert(tk.END, seq_file.stem)
        except Exception as e:
            logging.error(f"Error populating sequences list: {e}")
    
    def _save_sequence_for_instance(self, instance_name: str):
        """Save sequence for instance - stub."""
        # TODO: Implement full sequence saving
        messagebox.showinfo("Not Implemented", "Sequence saving not yet implemented")
    
    def _load_sequence_from_list(self, instance_name: str, sequences_listbox: tk.Listbox):
        """Load sequence from list - stub."""
        # TODO: Implement full sequence loading
        messagebox.showinfo("Not Implemented", "Sequence loading not yet implemented")
    
    def _delete_sequence_from_list(self, instance_name: str, sequences_listbox: tk.Listbox):
        """Delete sequence from list - stub."""
        # TODO: Implement full sequence deletion
        messagebox.showinfo("Not Implemented", "Sequence deletion not yet implemented")
    
    def _update_plan_details_inline(self, instance_name: str, selected_listbox: tk.Listbox):
        """Update plan details inline - stub."""
        # TODO: Implement full plan details update
        pass
    
    def _add_rule_inline_advanced(self, instance_name: str, selected_listbox: tk.Listbox, 
                                  rule_type_var: tk.StringVar, rule_data: dict,
                                  rules_scrollable_frame: ttk.Frame, rules_canvas: tk.Canvas):
        """Add rule inline advanced - stub."""
        # TODO: Implement full rule addition
        messagebox.showinfo("Not Implemented", "Rule addition not yet implemented")
    
    def _clear_plan_parameters(self, instance_name: str, selected_listbox: tk.Listbox):
        """Clear plan parameters - stub."""
        # TODO: Implement parameter clearing
        messagebox.showinfo("Not Implemented", "Parameter clearing not yet implemented")
    
    def _clear_plan_rules(self, instance_name: str, selected_listbox: tk.Listbox):
        """Clear plan rules - stub."""
        # TODO: Implement rule clearing
        messagebox.showinfo("Not Implemented", "Rule clearing not yet implemented")
    
    def update_parameter_widgets(self, instance_name: str, selected_listbox: tk.Listbox):
        """Update parameter widgets based on selected plan - stub."""
        # TODO: Implement parameter widget updates
        # This should populate params_edit_frame with plan-specific parameter widgets
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab or not hasattr(instance_tab, 'params_edit_frame'):
            return
        
        # Clear existing widgets
        for widget in instance_tab.params_edit_frame.winfo_children():
            widget.destroy()
        
        # TODO: Add plan-specific parameter widgets based on selected plan
        selection = selected_listbox.curselection()
        if not selection:
            return
        
        # Placeholder - should create widgets based on plan type
        pass
    
    def add_plan_to_instance(self, instance_name: str, available_listbox: tk.Listbox, 
                            selected_listbox: tk.Listbox):
        """Add selected plan to instance."""
        selected_indices = available_listbox.curselection()
        for index in reversed(selected_indices):  # Reverse to maintain order
            plan_name = available_listbox.get(index)
            selected_listbox.insert(tk.END, plan_name)
    
    def remove_plan_from_instance(self, instance_name: str, selected_listbox: tk.Listbox):
        """Remove selected plan from instance."""
        selected_indices = selected_listbox.curselection()
        for index in reversed(selected_indices):  # Reverse to maintain order
            selected_listbox.delete(index)
    
    def move_plan_up_in_instance(self, instance_name: str, selected_listbox: tk.Listbox):
        """Move selected plan up in instance."""
        selected_indices = selected_listbox.curselection()
        if selected_indices and selected_indices[0] > 0:
            index = selected_indices[0]
            item = selected_listbox.get(index)
            selected_listbox.delete(index)
            selected_listbox.insert(index - 1, item)
            selected_listbox.selection_set(index - 1)
    
    def move_plan_down_in_instance(self, instance_name: str, selected_listbox: tk.Listbox):
        """Move selected plan down in instance."""
        selected_indices = selected_listbox.curselection()
        if selected_indices and selected_indices[0] < selected_listbox.size() - 1:
            index = selected_indices[0]
            item = selected_listbox.get(index)
            selected_listbox.delete(index)
            selected_listbox.insert(index + 1, item)
            selected_listbox.selection_set(index + 1)
    
    def start_plans_for_instance(self, instance_name: str, session_dir: str, port: int,
                                update_instance_phase_callback: Callable = None):
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
        
        # Store update_instance_phase callback
        if update_instance_phase_callback:
            self.update_instance_phase = update_instance_phase_callback
        
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
                                for j, selected_cred in enumerate(self.selected_credentials):
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
    
    def stop_plans_for_instance(self, instance_name: str):
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
    
    def execute_plan_for_instance(self, instance_name: str, plan_id: str, 
                                  session_dir: str, port: int, rules_args: List[str] = None, 
                                  param_args: List[str] = None, timeout_minutes: int = None) -> bool:
        """Execute a single plan for an instance."""
        if rules_args is None:
            rules_args = []
        if param_args is None:
            param_args = []
        
        try:
            self.log_message_to_instance(instance_name, f"Executing plan: {plan_id}", 'info')
            
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
            completion_patterns = self._get_completion_patterns_for_plan(plan_id)
            self.log_message_to_instance(instance_name, f"DEBUG: Using completion patterns: {completion_patterns}", 'info')
            
            # Create the command
            cmd = [
                sys.executable, "run_rj_loop.py",
                plan_id,  # Plan name as positional argument
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
                cwd=Path(__file__).resolve().parent.parent,  # Run from project root
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
                            time.sleep(1)
                            if process.poll() is None:
                                process.kill()
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
                        
                        # Check for phase updates (delegate to callback if available)
                        if line.startswith("phase: "):
                            phase = line[7:].strip()
                            if hasattr(self, 'update_instance_phase'):
                                self.update_instance_phase(instance_name, phase)
                        
                        # Check for plan completion indicators
                        line_lower = line.lower()
                        if any(completion_indicator in line_lower for completion_indicator in completion_patterns):
                            plan_completed = True
                            self.log_message_to_instance(instance_name, f"Plan completion detected: {line}", 'success')
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
                self.log_message_to_instance(instance_name, f"Plan {plan_id} completed successfully", 'success')
                return True
            else:
                self.log_message_to_instance(instance_name, f"Plan failed with return code {return_code}", 'error')
                raise Exception(f"Plan failed with return code {return_code}")
                
        except Exception as e:
            self.log_message_to_instance(instance_name, f"Failed to execute plan {plan_id}: {str(e)}", 'error')
            raise Exception(f"Failed to execute plan {plan_id}: {str(e)}")
    
    def on_instance_tab_changed(self, event, focus_window_callback: Callable = None):
        """Handle instance tab change event."""
        try:
            selected_tab_id = event.widget.select()
            selected_frame = event.widget.nametowidget(selected_tab_id)
            # Find which username maps to this frame
            for username, tab_frame in self.instance_tabs.items():
                if tab_frame == selected_frame:
                    # Focus the corresponding RuneLite window
                    if focus_window_callback:
                        focus_window_callback(username)
                    break
        except Exception as e:
            logging.error(f"Error handling tab change: {e}")
    
    def _get_completion_patterns_for_plan(self, plan_name: str) -> List[str]:
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
    
    def _write_rule_params_to_file(self, instance_name: str):
        """Write rule parameters to JSON file for StatsMonitor."""
        try:
            instance_tab = self.instance_tabs.get(instance_name)
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
                all_rules['start_time'] = datetime.now().isoformat()
            
            # Write to JSON file
            rule_params_file = Path(__file__).resolve().parent.parent / "character_data" / f"rule_params_{instance_name}.json"
            rule_params_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(rule_params_file, 'w', encoding='utf-8') as f:
                json.dump(all_rules, f, indent=2)
            
            logging.info(f"[GUI] Wrote rule parameters to {rule_params_file}: {all_rules}")
            
        except Exception as e:
            logging.error(f"[GUI] Error writing rule parameters for {instance_name}: {e}")
