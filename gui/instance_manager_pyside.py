"""
Instance Manager Module (PySide6)
=================================

Manages individual RuneLite instance tabs and their plan execution.
"""

from PySide6.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QPushButton, QTextEdit, QGroupBox, QLineEdit, QListWidget,
                               QScrollArea, QFrame, QCheckBox, QComboBox, QSpinBox, QProgressBar,
                               QFileDialog, QListWidgetItem, QMessageBox, QTreeWidget, QTreeWidgetItem)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
import time
import sys
import os
import subprocess
import json
import logging
from datetime import datetime
from run_rj_loop import AVAILABLE_PLANS
from gui.unified_stats_panel import UnifiedStatsPanel


class InstanceManager:
    """Manages individual RuneLite instance tabs and their plan execution."""
    
    def __init__(self, root: QWidget, notebook: QTabWidget, instance_tabs: Dict,
                 instance_ports: Dict, detected_clients: Dict, skill_icons: Dict,
                 stats_monitors: Dict, base_completion_patterns: List[str],
                 selected_credentials: List[str], log_callback: Optional[Callable] = None,
                 stop_statistics_timer_callback: Optional[Callable] = None,
                 update_plan_details_callback: Optional[Callable] = None,
                 update_parameter_widgets_callback: Optional[Callable] = None,
                 update_stats_text_callback: Optional[Callable] = None,
                 start_stats_monitor_callback: Optional[Callable] = None,
                 get_credential_name_callback: Optional[Callable] = None,
                 log_message_to_instance_callback: Optional[Callable] = None):
        """Initialize instance manager."""
        self.root = root
        self.notebook = notebook
        self.instance_tabs = instance_tabs
        self.instance_ports = instance_ports
        self.detected_clients = detected_clients
        self.skill_icons = skill_icons
        self.stats_monitors = stats_monitors
        self.base_completion_patterns = base_completion_patterns
        self.selected_credentials = selected_credentials
        self.log_callback = log_callback or (lambda msg, level='info': None)
        self.stop_statistics_timer_callback = stop_statistics_timer_callback
        self.update_plan_details_callback = update_plan_details_callback
        self.update_parameter_widgets_callback = update_parameter_widgets_callback
        self.update_stats_text_callback = update_stats_text_callback or (lambda name: None)
        self.start_stats_monitor_callback = start_stats_monitor_callback or (lambda name, port: None)
        self.get_credential_name_callback = get_credential_name_callback or (lambda name: name)
        self.log_message_to_instance_callback = log_message_to_instance_callback or (lambda name, msg, level='info': None)
    
    def _build_plan_directory_structure(self) -> Dict[str, Dict]:
        """
        Build a directory structure mapping for plans.
        Returns a nested dict structure like:
        {
            'f2p': {'tutorial_island': plan_class, ...},
            'p2p': {'cooking': plan_class, ...},
            'utilities': {'bank_plan': plan_class, ...},
            'crafting': {'crafting': plan_class, ...},
            'root': {'blast_furnace': plan_class, ...}
        }
        """
        structure = {}
        plans_dir = Path(__file__).resolve().parent.parent / "plans"
        
        # Scan root plans directory
        root_plans = {}
        for plan_file in plans_dir.glob("*.py"):
            if plan_file.name.startswith("__"):
                continue
            plan_name = plan_file.stem
            plan_key = plan_name
            if plan_key in AVAILABLE_PLANS:
                root_plans[plan_name] = AVAILABLE_PLANS[plan_key]
        if root_plans:
            structure['root'] = root_plans
        
        # Scan subdirectories
        for subdir in plans_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("__"):
                subdir_plans = {}
                for plan_file in subdir.glob("*.py"):
                    if plan_file.name.startswith("__"):
                        continue
                    plan_name = plan_file.stem
                    # Plans in subdirectories use composite keys like "p2p_cooking"
                    plan_key = f"{subdir.name}_{plan_name}"
                    if plan_key in AVAILABLE_PLANS:
                        subdir_plans[plan_name] = AVAILABLE_PLANS[plan_key]
                if subdir_plans:
                    structure[subdir.name] = subdir_plans
        
        return structure
    
    def _populate_plan_tree(self, tree_widget: QTreeWidget):
        """Populate the plan tree widget with plans organized by directory structure."""
        tree_widget.clear()
        tree_widget.setHeaderLabel("Available Plans")
        tree_widget.setRootIsDecorated(True)
        
        structure = self._build_plan_directory_structure()
        
        # First, add root-level plans directly as top-level items (no directory node)
        if 'root' in structure:
            root_plans = structure['root']
            for plan_name, plan_class in sorted(root_plans.items()):
                label = getattr(plan_class, 'label', plan_name.replace('_', ' ').title())
                plan_id = plan_name
                plan_item = QTreeWidgetItem(tree_widget, [label])
                # Store plan_id and plan_class in item data
                plan_item.setData(0, Qt.ItemDataRole.UserRole, plan_id)
                plan_item.setData(0, Qt.ItemDataRole.UserRole + 1, plan_class)
        
        # Then add subdirectories with their plans
        dirs = [d for d in structure.keys() if d != 'root']
        dirs.sort()
        
        for dir_name in dirs:
            plans_dict = structure[dir_name]
            
            # Create directory item
            dir_item = QTreeWidgetItem(tree_widget, [dir_name])
            dir_item.setExpanded(True)
            
            # Add plans under this directory
            for plan_name, plan_class in sorted(plans_dict.items()):
                label = getattr(plan_class, 'label', plan_name.replace('_', ' ').title())
                plan_id = f"{dir_name}_{plan_name}"
                
                plan_item = QTreeWidgetItem(dir_item, [label])
                # Store plan_id and plan_class in item data
                plan_item.setData(0, Qt.ItemDataRole.UserRole, plan_id)
                plan_item.setData(0, Qt.ItemDataRole.UserRole + 1, plan_class)
    
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
                           clear_rules_callback: Callable = None,
                           start_plans_callback: Callable = None,
                           stop_plans_callback: Callable = None):
        """Create a new instance tab with full Plan Runner functionality."""
        # Check if tab already exists
        if instance_name in self.instance_tabs:
            self.log_callback(f"Instance tab already exists: {instance_name}", 'info')
            self.instance_ports[instance_name] = port
            return self.instance_tabs[instance_name]
        
        self.log_callback(f"Creating instance tab: {instance_name} on port {port}", 'info')
        
        # Create the main instance tab widget
        instance_tab = QWidget()
        main_layout = QVBoxLayout(instance_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Store reference EARLY so callbacks can find the tab
        # (This must be done before _create_plan_runner_tab calls update_stats_text_callback)
        self.instance_tabs[instance_name] = instance_tab
        self.instance_ports[instance_name] = port
        self.log_callback(f"[TIMING] Stored instance_tab in instance_tabs BEFORE creating plan_runner_tab: {instance_name}", 'info')
        
        # Create sub-notebook for Plan Runner, Output, and Statistics
        sub_notebook = QTabWidget()
        main_layout.addWidget(sub_notebook)
        
        # Create Plan Runner sub-tab
        plan_runner_tab = self._create_plan_runner_tab(
            instance_name, port,
            browse_directory_callback,
            populate_sequences_callback,
            save_sequence_callback,
            load_sequence_callback,
            delete_sequence_callback,
            add_plan_callback,
            remove_plan_callback,
            move_plan_up_callback,
            move_plan_down_callback,
            clear_plans_callback,
            update_plan_details_callback,
            update_parameter_widgets_callback,
            add_rule_callback,
            clear_params_callback,
            clear_rules_callback,
            start_plans_callback,
            stop_plans_callback
        )
        
        sub_notebook.addTab(plan_runner_tab, "Plan Runner")
        
        # Create Output sub-tab
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(5, 5, 5, 5)
        
        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setPlaceholderText(f"Output for {instance_name} will appear here...")
        output_text.setFont(QFont("Consolas", 9))
        
        # Ensure the text widget uses theme-aware colors
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPalette
        palette = output_text.palette()
        app_palette = QApplication.instance().palette()
        palette.setColor(QPalette.ColorRole.Base, app_palette.color(QPalette.ColorRole.Base))
        palette.setColor(QPalette.ColorRole.Text, app_palette.color(QPalette.ColorRole.Text))
        output_text.setPalette(palette)
        
        output_layout.addWidget(output_text)
        
        sub_notebook.addTab(output_tab, "Output")
        
        # Create Statistics sub-tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        
        stats_label = QLabel(f"Statistics for {instance_name}")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_label)
        
        sub_notebook.addTab(stats_tab, "Statistics")
        
        # Store references
        instance_tab.plan_runner_tab = plan_runner_tab
        instance_tab.output_tab = output_tab
        instance_tab.stats_tab = stats_tab
        instance_tab.sub_notebook = sub_notebook
        instance_tab.output_text = output_text  # Store output text widget for logging
        instance_tab.instance_name = instance_name  # Store instance_name in widget for easy lookup
        
        # Get credential name for tab display
        tab_display_name = self.get_credential_name_callback(instance_name)
        if tab_display_name == "Detected" or tab_display_name == "Unknown":
            # Fallback to instance_name if credential not found
            tab_display_name = instance_name
        
        # Add tab to main notebook with credential name as display name
        self.notebook.addTab(instance_tab, tab_display_name)
        
        # Reference already stored above (before _create_plan_runner_tab)
        # Just verify it's still there
        if instance_name not in self.instance_tabs:
            self.instance_tabs[instance_name] = instance_tab
            self.instance_ports[instance_name] = port
            self.log_callback(f"[TIMING] Re-stored instance_tab (shouldn't happen): {instance_name}", 'warning')
        else:
            self.log_callback(f"[TIMING] Instance_tab already stored (as expected): {instance_name}", 'info')
        
        # NOW that plan_runner_tab is assigned to instance_tab, start stats monitor and update stats
        # Get credential name for stats monitoring (use credential name, not instance_name)
        credential_name = self.get_credential_name_callback(instance_name)
        self.log_callback(f"[STATS] Got credential name for {instance_name}: {credential_name}", 'info')
        # If credential name is "Detected" or "Unknown", fall back to instance_name
        if credential_name == "Detected" or credential_name == "Unknown":
            credential_name = instance_name
            self.log_callback(f"[STATS] Credential name was Detected/Unknown, using instance_name: {credential_name}", 'info')
        
        # Start stats monitor for this instance (use credential name for CSV lookup)
        # Pass instance_name so statistics can map credential name to instance_name
        self.log_callback(f"[STATS] Starting stats monitor: credential={credential_name}, port={port}, instance={instance_name}", 'info')
        try:
            import inspect
            sig = inspect.signature(self.start_stats_monitor_callback)
            if 'instance_name' in sig.parameters:
                self.start_stats_monitor_callback(credential_name, port, instance_name=instance_name)
            else:
                self.start_stats_monitor_callback(credential_name, port)
        except (AttributeError, ValueError):
            # Fallback if signature inspection fails
            self.start_stats_monitor_callback(credential_name, port)
        
        # Initial stats update (pass instance_name - update_stats_text will handle credential lookup)
        # This is called AFTER plan_runner_tab is assigned to instance_tab, so it can find it
        self.log_callback(f"[STATS] Calling update_stats_text_callback for instance: {instance_name}", 'info')
        self.update_stats_text_callback(instance_name)
        
        # Log initial message to instance output tab
        self.log_message_to_instance_callback(instance_name, f"Instance tab created for {instance_name} on port {port}", 'info')
        self.log_message_to_instance_callback(instance_name, f"Stats monitor started for credential: {credential_name}", 'info')
        
        self.log_callback(f"Instance tab created successfully: {instance_name}", 'success')
        
        return instance_tab
    
    def _create_plan_runner_tab(self, instance_name: str, port: int,
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
                                clear_rules_callback: Callable = None,
                                start_plans_callback: Callable = None,
                                stop_plans_callback: Callable = None):
        """Create the Plan Runner sub-tab with all widgets."""
        plan_runner_tab = QWidget()
        main_layout = QVBoxLayout(plan_runner_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left: Session Config Panel
        config_group = QGroupBox("Session Config")
        config_layout = QGridLayout()
        
        # Session Directory
        config_layout.addWidget(QLabel("Session Dir:"), 0, 0)
        dir_frame = QHBoxLayout()
        session_dir_edit = QLineEdit()
        session_dir_edit.setText(f"D:\\bots\\exports\\{instance_name.lower()}")
        dir_frame.addWidget(session_dir_edit)
        browse_btn = QPushButton("Browse")
        if browse_directory_callback:
            browse_btn.clicked.connect(lambda: browse_directory_callback(instance_name, session_dir_edit))
        else:
            browse_btn.clicked.connect(lambda: self._browse_directory(instance_name, session_dir_edit))
        dir_frame.addWidget(browse_btn)
        config_layout.addLayout(dir_frame, 0, 1)
        
        # Port (read-only)
        config_layout.addWidget(QLabel("Port:"), 1, 0)
        port_label = QLabel(str(port))
        config_layout.addWidget(port_label, 1, 1)
        
        # Credential (read-only)
        config_layout.addWidget(QLabel("Credential:"), 2, 0)
        cred_file_name = None
        for selected_cred in self.selected_credentials:
            cred_username = selected_cred.replace('.properties', '')
            if cred_username == instance_name:
                cred_file_name = selected_cred
                break
        cred_label = QLabel(cred_file_name if cred_file_name else "Not found")
        config_layout.addWidget(cred_label, 2, 1)
        
        # Current Plan (read-only)
        config_layout.addWidget(QLabel("Current Plan:"), 3, 0)
        current_plan_label = QLabel("None")
        config_layout.addWidget(current_plan_label, 3, 1)
        
        # Current Phase (read-only)
        config_layout.addWidget(QLabel("Current Phase:"), 4, 0)
        current_phase_label = QLabel("Idle")
        config_layout.addWidget(current_phase_label, 4, 1)
        
        # Logged In Time (read-only)
        config_layout.addWidget(QLabel("Logged In Time:"), 5, 0)
        logged_in_time_label = QLabel("0:00:00")
        config_layout.addWidget(logged_in_time_label, 5, 1)
        
        # Control buttons
        control_buttons_layout = QHBoxLayout()
        start_button = QPushButton("▶")
        start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; min-width: 30px;")
        if start_plans_callback:
            start_button.clicked.connect(lambda: start_plans_callback(instance_name, session_dir_edit.text(), port))
        control_buttons_layout.addWidget(start_button)
        
        stop_button = QPushButton("■")
        stop_button.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; min-width: 30px;")
        if stop_plans_callback:
            stop_button.clicked.connect(lambda: stop_plans_callback(instance_name))
        control_buttons_layout.addWidget(stop_button)
        
        pause_checkbox = QCheckBox("Pause between plans")
        control_buttons_layout.addWidget(pause_checkbox)
        control_buttons_layout.addStretch()
        
        config_layout.addLayout(control_buttons_layout, 6, 0, 1, 2)
        
        # Key Items Totals section (placeholder, shown when bank is open)
        key_items_group = QGroupBox("Key Items Totals")
        key_items_layout = QVBoxLayout()
        key_items_group.setLayout(key_items_layout)
        key_items_group.hide()  # Hidden by default
        config_layout.addWidget(key_items_group, 7, 0, 1, 2)
        
        config_group.setLayout(config_layout)
        
        # Create main horizontal container: Left column (config + plans) | Right column (stats)
        main_horizontal_container = QWidget()
        main_horizontal_layout = QHBoxLayout(main_horizontal_container)
        main_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left column: Config at top, then Plan Selection, Plan Details, Saved Sequences below
        left_column_container = QWidget()
        left_column_layout = QVBoxLayout(left_column_container)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.addWidget(config_group)
        
        # Main content: Vertical layout with Plan Details above Plan Selection
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create selected_listbox early so it can be referenced in Plan Details callbacks
        selected_listbox = QListWidget()
        selected_listbox.setMaximumHeight(150)
        
        # Plan Details (moved above Plan Selection)
        plan_details_group = QGroupBox("Plan Details")
        plan_details_layout = QHBoxLayout()  # Changed to horizontal layout
        
        # Left side: Controls and Description (75% width)
        left_side_container = QWidget()
        left_side_layout = QVBoxLayout(left_side_container)
        left_side_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top row: Clear buttons and Add controls
        controls_row = QHBoxLayout()
        
        # Clear buttons
        clear_buttons_layout = QVBoxLayout()
        clear_params_btn = QPushButton("Clear Parameters")
        clear_rules_btn = QPushButton("Clear Rules")
        if clear_params_callback:
            clear_params_btn.clicked.connect(lambda: clear_params_callback(instance_name, selected_listbox))
        if clear_rules_callback:
            clear_rules_btn.clicked.connect(lambda: clear_rules_callback(instance_name, selected_listbox))
        clear_buttons_layout.addWidget(clear_params_btn)
        clear_buttons_layout.addWidget(clear_rules_btn)
        controls_row.addLayout(clear_buttons_layout)
        
        # Add Rule section (moved to left)
        rule_edit_layout = QHBoxLayout()
        rule_edit_layout.addWidget(QLabel("Add Rule:"))
        rule_type_combo = QComboBox()
        rule_type_combo.addItems(["Time", "Skill", "Item", "Total Level"])
        rule_type_combo.setCurrentText("Time")
        rule_edit_layout.addWidget(rule_type_combo)
        
        # Dynamic rule input frame
        rule_input_widget = QWidget()
        rule_input_layout = QHBoxLayout(rule_input_widget)
        rule_input_layout.setContentsMargins(0, 0, 0, 0)
        
        # Time rule widgets
        time_spinbox = QSpinBox()
        time_spinbox.setRange(0, 10000)
        time_spinbox.setValue(0)
        time_label = QLabel("minutes")
        
        # Skill rule widgets
        skill_list = ["Attack", "Strength", "Defence", "Ranged", "Magic", "Woodcutting", "Fishing", 
                     "Cooking", "Mining", "Smithing", "Firemaking", "Crafting", "Fletching", "Runecraft", 
                     "Herblore", "Agility", "Thieving", "Slayer", "Farming", "Construction", "Hunter", "Prayer"]
        skill_combo = QComboBox()
        skill_combo.addItems(skill_list)
        skill_level_spinbox = QSpinBox()
        skill_level_spinbox.setRange(1, 99)
        skill_level_spinbox.setValue(1)
        skill_label = QLabel("level")
        
        # Item rule widgets
        item_name_edit = QLineEdit()
        item_name_edit.setPlaceholderText("item name")
        item_qty_spinbox = QSpinBox()
        item_qty_spinbox.setRange(1, 99999)
        item_qty_spinbox.setValue(1)
        item_x_label = QLabel("x")
        
        # Total Level rule widget
        total_level_spinbox = QSpinBox()
        total_level_spinbox.setRange(0, 2277)
        total_level_spinbox.setValue(0)
        total_level_label = QLabel("level")
        
        # Initially hide all widgets
        time_spinbox.hide()
        time_label.hide()
        skill_combo.hide()
        skill_level_spinbox.hide()
        skill_label.hide()
        item_name_edit.hide()
        item_x_label.hide()
        item_qty_spinbox.hide()
        total_level_spinbox.hide()
        total_level_label.hide()
        
        # Function to show appropriate input widgets
        def show_rule_input():
            # Hide all widgets first
            time_spinbox.hide()
            time_label.hide()
            skill_combo.hide()
            skill_level_spinbox.hide()
            skill_label.hide()
            item_name_edit.hide()
            item_x_label.hide()
            item_qty_spinbox.hide()
            total_level_spinbox.hide()
            total_level_label.hide()
            
            # Clear layout
            while rule_input_layout.count():
                child = rule_input_layout.takeAt(0)
            
            rule_type = rule_type_combo.currentText()
            if rule_type == "Time":
                rule_input_layout.addWidget(time_spinbox)
                rule_input_layout.addWidget(time_label)
                time_spinbox.show()
                time_label.show()
            elif rule_type == "Skill":
                rule_input_layout.addWidget(skill_combo)
                rule_input_layout.addWidget(skill_level_spinbox)
                rule_input_layout.addWidget(skill_label)
                skill_combo.show()
                skill_level_spinbox.show()
                skill_label.show()
            elif rule_type == "Item":
                rule_input_layout.addWidget(item_name_edit)
                rule_input_layout.addWidget(item_x_label)
                rule_input_layout.addWidget(item_qty_spinbox)
                item_name_edit.show()
                item_x_label.show()
                item_qty_spinbox.show()
            elif rule_type == "Total Level":
                rule_input_layout.addWidget(total_level_spinbox)
                rule_input_layout.addWidget(total_level_label)
                total_level_spinbox.show()
                total_level_label.show()
            rule_input_layout.addStretch()
        
        rule_type_combo.currentTextChanged.connect(show_rule_input)
        show_rule_input()  # Initial display
        
        rule_edit_layout.addWidget(rule_input_widget, 1)
        
        add_rule_btn = QPushButton("Add")
        if add_rule_callback:
            def add_rule():
                rule_data = {
                    'Time': (time_spinbox, None),
                    'Skill': (skill_combo, skill_level_spinbox),
                    'Item': (item_name_edit, item_qty_spinbox),
                    'Total Level': (total_level_spinbox, None)
                }
                add_rule_callback(instance_name, selected_listbox, rule_type_combo, rule_data)
            add_rule_btn.clicked.connect(add_rule)
        rule_edit_layout.addWidget(add_rule_btn)
        controls_row.addLayout(rule_edit_layout, 1)
        
        # Add Parameter section
        param_edit_layout = QHBoxLayout()
        param_edit_layout.addWidget(QLabel("Add Parameter:"))
        params_edit_container = QWidget()
        params_edit_layout = QVBoxLayout(params_edit_container)
        params_edit_layout.setContentsMargins(0, 0, 0, 0)
        param_edit_layout.addWidget(params_edit_container, 1)
        controls_row.addLayout(param_edit_layout, 1)
        
        left_side_layout.addLayout(controls_row)
        
        # Description section
        description_group = QGroupBox("Description")
        description_layout = QVBoxLayout()
        description_text = QTextEdit()
        description_text.setReadOnly(True)
        description_layout.addWidget(description_text)
        description_group.setLayout(description_layout)
        left_side_layout.addWidget(description_group, 1)
        
        plan_details_layout.addWidget(left_side_container, 3)  # 75% width
        
        # Right side: Rules and Parameters container (25% width, extending from top)
        rules_params_container = QWidget()
        rules_params_layout = QVBoxLayout(rules_params_container)
        rules_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Rules section
        rules_label = QLabel("Rules")
        rules_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        rules_params_layout.addWidget(rules_label)
        
        rules_scroll = QScrollArea()
        rules_scroll.setWidgetResizable(True)
        rules_content = QWidget()
        rules_content_layout = QVBoxLayout(rules_content)
        rules_scroll.setWidget(rules_content)
        rules_params_layout.addWidget(rules_scroll, 1)  # Give it stretch
        
        # Parameters section
        params_label = QLabel("Parameters")
        params_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        rules_params_layout.addWidget(params_label)
        
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_content = QWidget()
        params_content_layout = QVBoxLayout(params_content)
        params_scroll.setWidget(params_content)
        rules_params_layout.addWidget(params_scroll, 1)  # Give it stretch
        
        plan_details_layout.addWidget(rules_params_container, 1)  # 25% width
        plan_details_group.setLayout(plan_details_layout)
        main_content_layout.addWidget(plan_details_group)
        
        # Plan Selection
        plan_selection_group = QGroupBox("Plan Selection")
        plan_selection_layout = QVBoxLayout()
        
        # Labels row
        labels_layout = QHBoxLayout()
        available_label = QLabel("Available Plans:")
        labels_layout.addWidget(available_label)
        labels_layout.addStretch()
        selected_label = QLabel("Selected Plans:")
        labels_layout.addWidget(selected_label)
        plan_selection_layout.addLayout(labels_layout)
        
        # Lists and controls row
        lists_layout = QHBoxLayout()
        
        # Use QTreeWidget instead of QListWidget for hierarchical plan display
        available_tree = QTreeWidget()
        available_tree.setMaximumHeight(150)
        available_tree.setHeaderLabel("Available Plans")
        available_tree.setRootIsDecorated(True)
        # Populate available plans in tree structure
        self._populate_plan_tree(available_tree)
        
        # Control buttons (middle)
        plan_controls_layout = QVBoxLayout()
        add_plan_btn = QPushButton("→")
        remove_plan_btn = QPushButton("←")
        move_up_btn = QPushButton("↑")
        move_down_btn = QPushButton("↓")
        clear_plans_btn = QPushButton("✕")
        
        # Connect buttons to callbacks
        if add_plan_callback:
            add_plan_btn.clicked.connect(lambda: add_plan_callback(instance_name, available_tree, selected_listbox))
        if remove_plan_callback:
            remove_plan_btn.clicked.connect(lambda: remove_plan_callback(instance_name, selected_listbox))
        if move_plan_up_callback:
            move_up_btn.clicked.connect(lambda: move_plan_up_callback(instance_name, selected_listbox))
        if move_plan_down_callback:
            move_down_btn.clicked.connect(lambda: move_plan_down_callback(instance_name, selected_listbox))
        if clear_plans_callback:
            clear_plans_btn.clicked.connect(lambda: clear_plans_callback(instance_name, selected_listbox))
        
        plan_controls_layout.addWidget(add_plan_btn)
        plan_controls_layout.addWidget(remove_plan_btn)
        plan_controls_layout.addWidget(move_up_btn)
        plan_controls_layout.addWidget(move_down_btn)
        plan_controls_layout.addWidget(clear_plans_btn)
        plan_controls_layout.addStretch()
        
        # Connect selection change to update details for both available tree and selected listbox
        if update_plan_details_callback:
            # Connect available tree selection
            available_tree.itemSelectionChanged.connect(
                lambda: update_plan_details_callback(instance_name, available_tree)
            )
            # Connect selected listbox selection
            selected_listbox.itemSelectionChanged.connect(
                lambda: update_plan_details_callback(instance_name, selected_listbox)
            )
        if update_parameter_widgets_callback:
            selected_listbox.itemSelectionChanged.connect(
                lambda: update_parameter_widgets_callback(instance_name, selected_listbox)
            )
        
        lists_layout.addWidget(available_tree)
        lists_layout.addLayout(plan_controls_layout)
        lists_layout.addWidget(selected_listbox)
        plan_selection_layout.addLayout(lists_layout)
        plan_selection_group.setLayout(plan_selection_layout)
        main_content_layout.addWidget(plan_selection_group)
        
        main_content_layout.addStretch()
        left_column_layout.addWidget(main_content, 1)
        
        # Right side: Unified Stats Panel (Skills, Inventory, Equipment with tab buttons)
        unified_stats_panel = UnifiedStatsPanel()
        
        # Store references for statistics updates
        skills_content_layout = unified_stats_panel.get_skills_layout()
        inventory_content = unified_stats_panel.get_inventory_panel()
        inventory_content_layout = None  # Inventory panel manages its own layout internally
        equipment_content = unified_stats_panel.get_equipment_panel()
        equipment_content_layout = None  # Equipment panel manages its own layout internally
        
        # Add left column and unified stats panel to main horizontal layout
        # Align unified stats panel at the top (not stretching)
        main_horizontal_layout.addWidget(left_column_container, 1)
        main_horizontal_layout.addWidget(unified_stats_panel, 0, Qt.AlignmentFlag.AlignTop)
        
        main_layout.addWidget(main_horizontal_container, 1)
        
        # Bottom: Status and Progress
        status_frame = QWidget()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        status_label_layout = QHBoxLayout()
        status_label_layout.addWidget(QLabel("Status:"))
        status_label = QLabel("Ready")
        status_label_layout.addWidget(status_label)
        status_label_layout.addStretch()
        status_layout.addLayout(status_label_layout)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        status_layout.addWidget(progress_bar)
        
        main_layout.addWidget(status_frame)
        
        # Store all references for later updates
        plan_runner_tab.session_dir_edit = session_dir_edit
        plan_runner_tab.port_label = port_label
        plan_runner_tab.cred_label = cred_label
        plan_runner_tab.current_plan_label = current_plan_label
        plan_runner_tab.current_phase_label = current_phase_label
        plan_runner_tab.logged_in_time_label = logged_in_time_label
        plan_runner_tab.start_button = start_button
        plan_runner_tab.stop_button = stop_button
        plan_runner_tab.pause_checkbox = pause_checkbox
        plan_runner_tab.key_items_group = key_items_group
        plan_runner_tab.unified_stats_panel = unified_stats_panel
        plan_runner_tab.skills_content = unified_stats_panel.skills_content
        plan_runner_tab.skills_content_layout = skills_content_layout
        plan_runner_tab.inventory_content = inventory_content
        plan_runner_tab.inventory_content_layout = inventory_content_layout
        plan_runner_tab.equipment_content = equipment_content
        plan_runner_tab.equipment_content_layout = equipment_content_layout
        plan_runner_tab.available_tree = available_tree
        plan_runner_tab.selected_listbox = selected_listbox
        plan_runner_tab.params_edit_container = params_edit_container
        plan_runner_tab.params_edit_layout = params_edit_layout
        plan_runner_tab.rules_content = rules_content
        plan_runner_tab.rules_content_layout = rules_content_layout
        plan_runner_tab.params_content = params_content
        plan_runner_tab.params_content_layout = params_content_layout
        plan_runner_tab.description_text = description_text
        plan_runner_tab.status_label = status_label
        plan_runner_tab.progress_bar = progress_bar
        plan_runner_tab.plan_entries = []
        plan_runner_tab.is_running = False
        plan_runner_tab.current_plan_index = 0
        plan_runner_tab.current_plan_name = "None"
        plan_runner_tab.current_phase = "Idle"
        plan_runner_tab.current_process = None
        
        # Return plan_runner_tab - stats monitor will be started AFTER it's assigned to instance_tab
        return plan_runner_tab
    
    def _browse_directory(self, instance_name: str, dir_edit: QLineEdit):
        """Browse for session directory."""
        current_dir = dir_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self.root,
            f"Select Session Directory for {instance_name}",
            current_dir if current_dir else str(Path.home())
        )
        if directory:
            dir_edit.setText(directory)
    
    def _add_plan_to_selection(self, instance_name: str, available_tree: QTreeWidget, selected_listbox: QListWidget):
        """Add selected plan from available tree to selected list."""
        current_item = available_tree.currentItem()
        if not current_item:
            return
        
        # Skip if clicking on a directory item (only allow leaf items)
        if current_item.childCount() > 0:
            return
        
        # Get plan_id from item data
        plan_id = current_item.data(0, Qt.ItemDataRole.UserRole)
        if not plan_id:
            return
        
        # Get plan class from item data
        plan_class = current_item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not plan_class:
            plan_class = AVAILABLE_PLANS.get(plan_id)
        
        if not plan_class:
            return
        
        # Get label for display
        label = getattr(plan_class, 'label', plan_id.replace('_', ' ').title())
        display_text = f"{label} ({plan_id})"
        
        # Add to selected list
        selected_listbox.addItem(display_text)
        
        # Create PlanEntry for this plan
        from gui.plan_editor import PlanEntry
        plan_entry = PlanEntry(
            name=plan_id,
            label=label,
            rules={'max_minutes': None, 'stop_skill': None, 'stop_items': [], 'total_level': None},
            params={'generic': {}}
        )
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if not hasattr(plan_runner_tab, 'plan_entries'):
                plan_runner_tab.plan_entries = []
            plan_runner_tab.plan_entries.append(plan_entry)
    
    def _remove_plan_from_selection(self, instance_name: str, selected_listbox: QListWidget):
        """Remove selected plan from selection."""
        current_row = selected_listbox.currentRow()
        if current_row < 0:
            return
        
        selected_listbox.takeItem(current_row)
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if hasattr(plan_runner_tab, 'plan_entries') and current_row < len(plan_runner_tab.plan_entries):
                del plan_runner_tab.plan_entries[current_row]
    
    def _move_plan_up(self, instance_name: str, selected_listbox: QListWidget):
        """Move selected plan up."""
        current_row = selected_listbox.currentRow()
        if current_row <= 0:
            return
        
        item = selected_listbox.takeItem(current_row)
        selected_listbox.insertItem(current_row - 1, item)
        selected_listbox.setCurrentRow(current_row - 1)
        
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if hasattr(plan_runner_tab, 'plan_entries') and current_row < len(plan_runner_tab.plan_entries):
                plan_runner_tab.plan_entries[current_row], plan_runner_tab.plan_entries[current_row - 1] = \
                    plan_runner_tab.plan_entries[current_row - 1], plan_runner_tab.plan_entries[current_row]
    
    def _move_plan_down(self, instance_name: str, selected_listbox: QListWidget):
        """Move selected plan down."""
        current_row = selected_listbox.currentRow()
        if current_row < 0 or current_row >= selected_listbox.count() - 1:
            return
        
        item = selected_listbox.takeItem(current_row)
        selected_listbox.insertItem(current_row + 1, item)
        selected_listbox.setCurrentRow(current_row + 1)
        
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if hasattr(plan_runner_tab, 'plan_entries') and current_row < len(plan_runner_tab.plan_entries) - 1:
                plan_runner_tab.plan_entries[current_row], plan_runner_tab.plan_entries[current_row + 1] = \
                    plan_runner_tab.plan_entries[current_row + 1], plan_runner_tab.plan_entries[current_row]
    
    def _clear_selected_plans(self, instance_name: str, selected_listbox: QListWidget):
        """Clear all selected plans."""
        selected_listbox.clear()
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if hasattr(plan_runner_tab, 'plan_entries'):
                plan_runner_tab.plan_entries.clear()
    
    def _add_rule_inline_advanced(self, instance_name: str, selected_listbox: QListWidget, 
                                   rule_type_combo: QComboBox, rule_data: dict):
        """Add rule with advanced widgets."""
        current_row = selected_listbox.currentRow()
        if current_row < 0:
            self.log_callback("No plan selected", 'warning')
            return
        
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab or not hasattr(instance_tab, 'plan_runner_tab'):
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        if not hasattr(plan_runner_tab, 'plan_entries') or current_row >= len(plan_runner_tab.plan_entries):
            return
        
        plan_entry = plan_runner_tab.plan_entries[current_row]
        rule_type = rule_type_combo.currentText()
        
        # Get rule data from widgets
        if rule_type == "Time":
            widget, _ = rule_data['Time']
            minutes = widget.value()
            if minutes > 0:
                plan_entry['rules']['max_minutes'] = minutes
        elif rule_type == "Skill":
            skill_combo, level_spinbox = rule_data['Skill']
            skill = skill_combo.currentText()
            level = level_spinbox.value()
            if skill:
                plan_entry['rules']['stop_skill'] = skill
                plan_entry['rules']['stop_skill_level'] = level
        elif rule_type == "Item":
            item_edit, qty_spinbox = rule_data['Item']
            item_name = item_edit.text().strip()
            qty = qty_spinbox.value()
            if item_name:
                if 'stop_items' not in plan_entry['rules']:
                    plan_entry['rules']['stop_items'] = []
                plan_entry['rules']['stop_items'].append({'name': item_name, 'qty': qty})
        elif rule_type == "Total Level":
            widget, _ = rule_data['Total Level']
            total_level = widget.value()
            if total_level > 0:
                plan_entry['rules']['total_level'] = total_level
        
        # Update rules display
        if hasattr(plan_runner_tab, 'rules_content_layout'):
            self._update_rules_display(plan_runner_tab, plan_entry)
    
    def _update_rules_display(self, plan_runner_tab, plan_entry):
        """Update the rules display section."""
        # Clear existing rules
        while plan_runner_tab.rules_content_layout.count():
            child = plan_runner_tab.rules_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        rules = plan_entry.get('rules', {})
        if rules.get('max_minutes'):
            label = QLabel(f"Time Limit: {rules['max_minutes']} minutes")
            plan_runner_tab.rules_content_layout.addWidget(label)
        if rules.get('stop_skill'):
            label = QLabel(f"Stop at Skill: {rules['stop_skill']} level {rules.get('stop_skill_level', 0)}")
            plan_runner_tab.rules_content_layout.addWidget(label)
        if rules.get('total_level'):
            label = QLabel(f"Total Level: {rules['total_level']}")
            plan_runner_tab.rules_content_layout.addWidget(label)
        if rules.get('stop_items'):
            for item in rules['stop_items']:
                label = QLabel(f"Stop with Item: {item['name']} x{item['qty']}")
                plan_runner_tab.rules_content_layout.addWidget(label)
        
        plan_runner_tab.rules_content_layout.addStretch()
    
    def _clear_plan_parameters(self, instance_name: str, selected_listbox: QListWidget):
        """Clear plan parameters."""
        current_row = selected_listbox.currentRow()
        if current_row < 0:
            return
        
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab or not hasattr(instance_tab, 'plan_runner_tab'):
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        if hasattr(plan_runner_tab, 'plan_entries') and current_row < len(plan_runner_tab.plan_entries):
            plan_runner_tab.plan_entries[current_row]['params'] = {'generic': {}}
            # Update parameter display
            if hasattr(plan_runner_tab, 'params_content_layout'):
                while plan_runner_tab.params_content_layout.count():
                    child = plan_runner_tab.params_content_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                plan_runner_tab.params_content_layout.addStretch()
    
    def _clear_plan_rules(self, instance_name: str, selected_listbox: QListWidget):
        """Clear plan rules."""
        current_row = selected_listbox.currentRow()
        if current_row < 0:
            return
        
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab or not hasattr(instance_tab, 'plan_runner_tab'):
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        if hasattr(plan_runner_tab, 'plan_entries') and current_row < len(plan_runner_tab.plan_entries):
            plan_runner_tab.plan_entries[current_row]['rules'] = {
                'max_minutes': None, 'stop_skill': None, 'stop_items': [], 'total_level': None
            }
            # Update rules display
            if hasattr(plan_runner_tab, 'rules_content_layout'):
                while plan_runner_tab.rules_content_layout.count():
                    child = plan_runner_tab.rules_content_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                plan_runner_tab.rules_content_layout.addStretch()
    
    def remove_instance_tab(self, instance_name: str):
        """Remove an instance tab from the notebook."""
        self.log_callback(f"remove_instance_tab called for: {instance_name}", 'info')
        
        if instance_name not in self.instance_tabs:
            self.log_callback(f"Instance tab not found in instance_tabs: {instance_name}", 'warning')
            return
        
        instance_tab = self.instance_tabs[instance_name]
        self.log_callback(f"Found instance tab widget for: {instance_name}", 'info')
        
        # Find and remove the tab from the notebook
        # Check by widget reference or by instance_name stored in widget
        tab_removed = False
        for i in range(self.notebook.count()):
            widget = self.notebook.widget(i)
            # Check if this is the widget we're looking for
            if widget == instance_tab:
                self.notebook.removeTab(i)
                tab_removed = True
                tab_text = self.notebook.tabText(i) if i < self.notebook.count() else "unknown"
                self.log_callback(f"Removed tab at index {i} (text: {tab_text})", 'info')
                break
            # Also check by instance_name stored in widget (in case widget reference doesn't match)
            elif hasattr(widget, 'instance_name') and widget.instance_name == instance_name:
                self.notebook.removeTab(i)
                tab_removed = True
                tab_text = self.notebook.tabText(i) if i < self.notebook.count() else "unknown"
                self.log_callback(f"Removed tab at index {i} by instance_name (text: {tab_text})", 'info')
                break
        
        if not tab_removed:
            self.log_callback(f"Warning: Tab for {instance_name} not found in notebook", 'warning')
        
        # Remove from dictionaries
        if instance_name in self.instance_tabs:
            del self.instance_tabs[instance_name]
        if instance_name in self.instance_ports:
            del self.instance_ports[instance_name]
        
        self.log_callback(f"Removed instance tab: {instance_name}", 'success')
    
    def start_plans_for_instance(self, instance_name: str, session_dir: str, port: int,
                                update_instance_phase_callback: Callable = None):
        """Start plans for a specific instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab:
            QMessageBox.warning(self.root, "Error", f"Instance tab not found for {instance_name}.")
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        
        if plan_runner_tab.is_running:
            QMessageBox.warning(self.root, "Already Running", f"Plans are already running for {instance_name}.")
            return
        
        # Get selected plans
        selected_plans = []
        selected_listbox = plan_runner_tab.selected_listbox
        for i in range(selected_listbox.count()):
            selected_plans.append(selected_listbox.item(i).text())
        
        if not selected_plans:
            QMessageBox.warning(self.root, "No Plans Selected", f"Please select at least one plan for {instance_name}.")
            return
        
        # Write rule parameters to file before starting plans
        self._write_rule_params_to_file(instance_name)
        
        # Store update_instance_phase callback
        if update_instance_phase_callback:
            self.update_instance_phase = update_instance_phase_callback
        
        # Start plans in a separate thread
        def run_plans():
            try:
                plan_runner_tab.is_running = True
                plan_runner_tab.start_time = time.time()  # Track start time for runtime
                plan_runner_tab.current_plan_index = 0  # Track current plan index
                plan_runner_tab.status_label.setText("Starting...")
                plan_runner_tab.progress_bar.setMaximum(len(selected_plans))
                plan_runner_tab.progress_bar.setValue(0)
                
                # Start statistics update timer
                if self.start_stats_monitor_callback:
                    self.start_stats_monitor_callback(instance_name, port)
                
                self.log_message_to_instance_callback(instance_name, f"Starting execution of {len(selected_plans)} plans", 'info')
                
                for i, plan_name in enumerate(selected_plans):
                    if not plan_runner_tab.is_running:  # Check if stopped
                        break
                    
                    # Update current plan tracking
                    plan_runner_tab.current_plan_index = i
                    plan_runner_tab.current_plan_name = plan_name
                    plan_runner_tab.current_phase = "Starting"
                    
                    # Update the display labels
                    plan_runner_tab.current_plan_label.setText(plan_name)
                    plan_runner_tab.current_phase_label.setText("Starting")
                    
                    plan_runner_tab.status_label.setText(f"Running: {plan_name}")
                    plan_runner_tab.progress_bar.setValue(i)
                    
                    # Update statistics display
                    if self.update_stats_text_callback:
                        self.update_stats_text_callback(instance_name)
                    
                    self.log_message_to_instance_callback(instance_name, f"Starting plan {i+1}/{len(selected_plans)}: {plan_name}", 'info')
                    
                    # Get rules and parameters from plan entries
                    rules_args = []
                    param_args = []
                    
                    # Find the plan entry for this plan
                    plan_entry = None
                    # Extract plan ID from the display name (format: "Display Name (plan_id)")
                    plan_id = plan_name
                    if '(' in plan_name and ')' in plan_name:
                        plan_id = plan_name.split('(')[-1].rstrip(')')
                        for entry in plan_runner_tab.plan_entries:
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
                                    self.log_message_to_instance_callback(instance_name, f"Auto-detected unnamed credential: {cred_name_without_ext}", 'info')
                    
                    # Run the plan
                    success = self.execute_plan_for_instance(instance_name, plan_id, session_dir, port, rules_args, param_args)
                    self.log_message_to_instance_callback(instance_name, f"DEBUG: Plan {plan_id} returned success = {success}", 'info')
                    if not success:
                        self.log_message_to_instance_callback(instance_name, f"Plan {plan_id} failed, stopping execution", 'error')
                        break
                    else:
                        self.log_message_to_instance_callback(instance_name, f"Plan {plan_id} completed successfully, moving to next plan", 'info')
                    
                    # Pause between plans if checkbox is checked and not the last one
                    if i < len(selected_plans) - 1 and plan_runner_tab.is_running:
                        if plan_runner_tab.pause_checkbox.isChecked():
                            plan_runner_tab.status_label.setText(f"Paused between plans - click to continue")
                            self.log_message_to_instance_callback(instance_name, "Paused between plans (waiting for user to uncheck pause)", 'info')
                            # Wait until pause is unchecked
                            while plan_runner_tab.pause_checkbox.isChecked() and plan_runner_tab.is_running:
                                time.sleep(1)
                            if plan_runner_tab.is_running:
                                self.log_message_to_instance_callback(instance_name, "Resuming plan execution...", 'info')
                        else:
                            # Just a brief pause for smooth transition
                            plan_runner_tab.status_label.setText(f"Transitioning to next plan...")
                            self.log_message_to_instance_callback(instance_name, "Moving to next plan...", 'info')
                            time.sleep(2)
                
                if plan_runner_tab.is_running:
                    plan_runner_tab.status_label.setText("All plans completed")
                    plan_runner_tab.progress_bar.setValue(len(selected_plans))
                    # Reset phase display
                    plan_runner_tab.current_plan_label.setText("None")
                    plan_runner_tab.current_phase_label.setText("Idle")
                    self.log_message_to_instance_callback(instance_name, "All plans completed successfully", 'success')
                
            except Exception as e:
                plan_runner_tab.status_label.setText(f"Error: {str(e)}")
                # Reset phase display on error
                plan_runner_tab.current_plan_label.setText("Error")
                plan_runner_tab.current_phase_label.setText("Failed")
                self.log_message_to_instance_callback(instance_name, f"Execution error: {str(e)}", 'error')
            finally:
                plan_runner_tab.is_running = False
        
        threading.Thread(target=run_plans, daemon=True).start()
    
    def execute_plan_for_instance(self, instance_name: str, plan_id: str, 
                                  session_dir: str, port: int, rules_args: List[str] = None, 
                                  param_args: List[str] = None, timeout_minutes: int = None) -> bool:
        """Execute a single plan for an instance."""
        if rules_args is None:
            rules_args = []
        if param_args is None:
            param_args = []
        
        try:
            self.log_message_to_instance_callback(instance_name, f"Executing plan: {plan_id}", 'info')
            
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
                self.log_message_to_instance_callback(instance_name, f"Using timeout: {timeout_minutes} minutes", 'info')
            else:
                self.log_message_to_instance_callback(instance_name, "No timeout specified - will run until completion", 'info')
            
            # Determine completion patterns based on plan type
            completion_patterns = self._get_completion_patterns_for_plan(plan_id)
            self.log_message_to_instance_callback(instance_name, f"DEBUG: Using completion patterns: {completion_patterns}", 'info')
            
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
            
            self.log_message_to_instance_callback(instance_name, f"Command: {' '.join(cmd)}", 'info')
            
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
                instance_tab = self.instance_tabs[instance_name]
                if hasattr(instance_tab, 'plan_runner_tab'):
                    instance_tab.plan_runner_tab.current_process = process
            
            # Stream output in real-time with completion detection
            plan_completed = False
            start_time = time.time()
            
            while True:
                # Check for timeout only if one is specified
                if timeout_minutes is not None:
                    timeout_seconds = timeout_minutes * 60
                    if time.time() - start_time > timeout_seconds:
                        self.log_message_to_instance_callback(instance_name, f"Plan timeout after {timeout_minutes} minutes, stopping...", 'warning')
                        try:
                            process.terminate()
                            time.sleep(1)
                            if process.poll() is None:
                                process.kill()
                        except Exception as e:
                            self.log_message_to_instance_callback(instance_name, f"Error stopping timed-out process: {e}", 'error')
                        # Treat timeout as successful completion so next plans can run
                        self.log_message_to_instance_callback(instance_name, f"Plan completed after {timeout_minutes} minutes (timeout)", 'success')
                        return True
                
                # Check if the instance is still running (for stop functionality)
                if instance_name in self.instance_tabs:
                    instance_tab = self.instance_tabs[instance_name]
                    if hasattr(instance_tab, 'plan_runner_tab'):
                        plan_runner_tab = instance_tab.plan_runner_tab
                        if not getattr(plan_runner_tab, 'is_running', True):
                            self.log_message_to_instance_callback(instance_name, "Stopping plan execution...", 'warning')
                            try:
                                process.terminate()
                                time.sleep(1)
                                if process.poll() is None:
                                    process.kill()
                            except Exception as e:
                                self.log_message_to_instance_callback(instance_name, f"Error stopping process: {e}", 'error')
                            self.log_message_to_instance_callback(instance_name, "Plan execution stopped by user", 'warning')
                            return False
                
                # Read output line by line
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    # Remove trailing newline and display in real-time
                    line = output.strip()
                    if line:
                        self.log_message_to_instance_callback(instance_name, line, 'info')
                        
                        # Check for phase updates (delegate to callback if available)
                        if line.startswith("phase: "):
                            phase = line[7:].strip()
                            if hasattr(self, 'update_instance_phase'):
                                self.update_instance_phase(instance_name, phase)
                        
                        # Check for plan completion indicators
                        line_lower = line.lower()
                        if any(completion_indicator in line_lower for completion_indicator in completion_patterns):
                            plan_completed = True
                            self.log_message_to_instance_callback(instance_name, f"Plan completion detected: {line}", 'success')
                            # STOP THE CURRENT PROCESS IMMEDIATELY
                            try:
                                process.terminate()
                                time.sleep(1)
                                if process.poll() is None:
                                    process.kill()
                                self.log_message_to_instance_callback(instance_name, "Process terminated due to completion", 'info')
                            except Exception as e:
                                self.log_message_to_instance_callback(instance_name, f"Error terminating completed process: {e}", 'error')
                            break
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Clear the process reference
            if instance_name in self.instance_tabs:
                instance_tab = self.instance_tabs[instance_name]
                if hasattr(instance_tab, 'plan_runner_tab'):
                    instance_tab.plan_runner_tab.current_process = None
            
            # Check if plan completed successfully
            if plan_completed or return_code == 0:
                self.log_message_to_instance_callback(instance_name, f"Plan {plan_id} completed successfully", 'success')
                return True
            else:
                self.log_message_to_instance_callback(instance_name, f"Plan failed with return code {return_code}", 'error')
                raise Exception(f"Plan failed with return code {return_code}")
                
        except Exception as e:
            self.log_message_to_instance_callback(instance_name, f"Failed to execute plan {plan_id}: {str(e)}", 'error')
            raise Exception(f"Failed to execute plan {plan_id}: {str(e)}")
    
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
            if not instance_tab or not hasattr(instance_tab, 'plan_runner_tab'):
                return
            
            plan_runner_tab = instance_tab.plan_runner_tab
            
            # Collect all rules from all plan entries
            all_rules = {}
            if hasattr(plan_runner_tab, 'plan_entries'):
                for plan_entry in plan_runner_tab.plan_entries:
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
    
    def stop_plans_for_instance(self, instance_name: str):
        """Stop plans for a specific instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab or not hasattr(instance_tab, 'plan_runner_tab'):
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        plan_runner_tab.is_running = False
        
        if hasattr(plan_runner_tab, 'current_process') and plan_runner_tab.current_process:
            try:
                plan_runner_tab.current_process.terminate()
                plan_runner_tab.current_process = None
            except Exception as e:
                self.log_message_to_instance_callback(instance_name, f"Error stopping process: {e}", 'error')
        
        # Stop statistics timer
        if self.stop_statistics_timer_callback:
            self.stop_statistics_timer_callback(instance_name)
        
        plan_runner_tab.status_label.setText("Stopped")
        self.log_message_to_instance_callback(instance_name, "Plans stopped", 'info')
