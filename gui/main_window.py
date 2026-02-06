"""
Main Window Module
=================

Main GUI window that orchestrates all components.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Callable

from run_rj_loop import AVAILABLE_PLANS
from gui.plan_editor import PlanEditor, PlanEntry
from gui.config_manager import ConfigManager
from gui.launcher import RuneLiteLauncher
from gui.client_detector import ClientDetector
from gui.instance_manager import InstanceManager
from gui.statistics import StatisticsDisplay
from gui.widgets import WidgetFactory
from gui.logging_utils import LoggingUtils
from helpers.ipc import IPCClient
from utils.stats_monitor import StatsMonitor


class SimpleRecorderGUI:
    """Main GUI application window."""
    
    def __init__(self, root):
        """
        Initialize the main GUI window.
        
        Args:
            root: Root tkinter window
        """
        self.root = root
        self.root.title("Simple Recorder - Plan Runner")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize variables
        self._init_variables()
        
        # Initialize component managers
        self._init_components()
        
        # Configure style
        WidgetFactory.setup_styles()
        
        # Create GUI
        self.create_widgets()
        
        # Load configuration after widgets are created
        self.config_manager.load_config()
        
        # Center window
        WidgetFactory.center_window(self.root)
    
    def _init_variables(self):
        """Initialize all variables."""
        # RuneLite launcher variables
        self.selected_credentials = []
        self.runelite_process = None
        self.build_maven = tk.BooleanVar(value=True)
        self.base_port = tk.IntVar(value=17000)
        self.launch_delay = tk.IntVar(value=0)
        
        # Configuration variables (needed for ConfigManager)
        self.config_vars = {
            "projectDir": tk.StringVar(),
            "classPathFile": tk.StringVar(),
            "javaExe": tk.StringVar(),
            "baseDir": tk.StringVar(),
            "exportsBase": tk.StringVar(),
            "credentialsDir": tk.StringVar(),
            "autoDetect": tk.BooleanVar(value=True)
        }
        
        # Path entries (will be populated when widgets are created)
        self.path_entries = {}
        
        # Instance management
        self.instance_tabs = {}
        self.instance_ports = {}
        self.detected_clients = {}
        self.client_detection_running = False
        
        # Base completion patterns
        self.base_completion_patterns = [
            'phase: done',
            'status: done', 'plan done', 'execution done',
            'plan completed', 'execution completed', 'finished successfully',
            'task completed', 'mission accomplished', 'objective complete'
        ]
        
        # Initialize skill icons dict
        self.skill_icons = {}
        
        # Stats monitors
        self.stats_monitors = {}
    
    def _init_components(self):
        """Initialize all component managers."""
        # Configuration file
        self.config_file = Path(__file__).resolve().parent.parent / "launch-config.json"
        
        # Initialize logging - create wrapper methods
        # Log text widget will be created in create_widgets
        self.log_text = None
        
        # Initialize config manager (needs config_vars and path_entries)
        # path_entries will be empty initially, populated when widgets are created
        self.config_manager = ConfigManager(
            self.config_file,
            self.config_vars,
            self.path_entries
        )
        
        # Initialize statistics (needs skill icons)
        self.statistics = StatisticsDisplay(
            self.root,
            self.instance_tabs,
            self.instance_ports,
            self.skill_icons,
            self.stats_monitors,
            self._log_message  # Use wrapper method
        )
        # Load skill icons
        self.skill_icons = self.statistics.load_skill_icons()
        
        # Initialize launcher (will be fully initialized in create_widgets)
        self.launcher = None
        
        # Initialize client detector (will be fully initialized in create_widgets)
        self.client_detector = None
        
        # Initialize instance manager (will be fully initialized in create_widgets)
        self.instance_manager = None
    
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
        
        # Create a placeholder detection status label (will be created in launcher tab)
        detection_status_label = ttk.Label(ttk.Frame(), text="Auto-detection: Stopped", style='Info.TLabel')
        
        # Initialize client detector with notebook
        self.client_detector = ClientDetector(
            self.root,
            self.notebook,
            self.instance_tabs,
            self.instance_ports,
            self.detected_clients,
            detection_status_label,
            self._log_message
        )
        
        # Initialize instance manager
        self.instance_manager = InstanceManager(
            self.root,
            self.notebook,
            self.instance_tabs,
            self.instance_ports,
            self._log_message,
            self._log_message_to_instance,
            self.statistics.update_stats_text,
            self.statistics.start_stats_monitor,
            self.statistics.update_statistics_display,
            self.statistics.start_statistics_timer,
            self.statistics.stop_statistics_timer,
            self._update_plan_details_wrapper,
            self._update_parameter_widgets_wrapper,
            self.selected_credentials,
            self.base_completion_patterns
        )
        
        # Create Client tab first (before initializing launcher, as it needs widgets)
        # Note: The tab is already added in _create_launcher_tab() with text "Client"
        launcher_tab, launcher_widgets = self._create_launcher_tab()
        
        # Initialize launcher with the widgets
        self.launcher = RuneLiteLauncher(
            self.root,
            self.config_manager.config_vars,  # Pass the config_vars dict
            self.base_port,
            self.launch_delay,
            self.build_maven,
            launcher_widgets['credentials_listbox'],
            launcher_widgets['selected_credentials_listbox'],
            self.selected_credentials,
            self._log_message,
            launcher_widgets.get('instance_count_label'),
            launcher_widgets.get('launch_button'),
            self.create_instance_tab_wrapper
        )
        
        # Set up client detector callbacks
        self.client_detector.create_instance_tab_callback = self.create_instance_tab_wrapper
        def remove_instance_callback(name):
            """Callback to remove instance tab and update display."""
            if hasattr(self.instance_manager, 'remove_instance_tab'):
                self.instance_manager.remove_instance_tab(name)
            self._remove_instance_from_display(name)
        
        self.client_detector.remove_instance_tab_callback = remove_instance_callback
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.instance_manager.on_instance_tab_changed(e, self._focus_runelite_window))
        
        # Create Output tab (creates self.log_text)
        self._create_output_tab()
        
        # Load configuration (now that log_text exists for callbacks)
        self.config_manager.load_config(
            base_port_var=self.base_port,
            launch_delay_var=self.launch_delay,
            build_maven_var=self.build_maven,
            log_callback=self._log_message
        )
        
        # Populate credentials
        self.launcher.populate_credentials()
        
        # Connect launcher stop button
        if 'stop_button' in launcher_widgets:
            launcher_widgets['stop_button'].config(command=lambda: self.launcher.stop_runelite(self._stop_all_instances))
        
        # Connect setup dependencies button
        if 'setup_deps_button' in launcher_widgets:
            launcher_widgets['setup_deps_button'].config(command=self.launcher.setup_dependencies)
        
        # Connect credential management buttons
        if 'add_cred_button' in launcher_widgets:
            launcher_widgets['add_cred_button'].config(command=self.launcher.add_credential)
        if 'remove_cred_button' in launcher_widgets:
            launcher_widgets['remove_cred_button'].config(command=self.launcher.remove_credential)
        if 'move_up_button' in launcher_widgets:
            launcher_widgets['move_up_button'].config(command=self.launcher.move_credential_up)
        if 'move_down_button' in launcher_widgets:
            launcher_widgets['move_down_button'].config(command=self.launcher.move_credential_down)
        if 'clear_creds_button' in launcher_widgets:
            launcher_widgets['clear_creds_button'].config(command=self.launcher.clear_credentials)
        
        # Connect launch button
        if 'launch_button' in launcher_widgets:
            launcher_widgets['launch_button'].config(command=self.launcher.launch_runelite)
        
        # Connect detection buttons
        if 'start_detection_button' in launcher_widgets:
            launcher_widgets['start_detection_button'].config(command=self.client_detector.start_client_detection)
        if 'stop_detection_button' in launcher_widgets:
            launcher_widgets['stop_detection_button'].config(command=self.client_detector.stop_client_detection)
        if 'detect_now_button' in launcher_widgets:
            launcher_widgets['detect_now_button'].config(command=lambda: self.client_detector.detect_running_clients(
                self.create_instance_tab_wrapper,
                lambda name: None,  # remove callback
                self._log_message
            ))
        if 'test_detection_button' in launcher_widgets:
            launcher_widgets['test_detection_button'].config(command=lambda: self.client_detector.test_client_detection(self._log_message))
        
        # Update detection status label reference
        if 'detection_status_label' in launcher_widgets:
            self.client_detector.detection_status_label = launcher_widgets['detection_status_label']
    
    def _create_launcher_tab(self):
        """
        Create the Client tab with sub-tabs for Setup & Configuration and Launcher.
        Returns (tab_frame, widgets_dict) for use by launcher.
        """
        client_tab = ttk.Frame(self.notebook, padding="0")
        self.notebook.add(client_tab, text="Client")
        
        # Create sub-notebook for Launcher, Setup & Configuration, and Output tabs
        sub_notebook = ttk.Notebook(client_tab)
        sub_notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Store sub_notebook for use in creating Output tab
        self.client_sub_notebook = sub_notebook
        
        # ===== Launcher Sub-tab (leftmost) =====
        launcher_tab = ttk.Frame(sub_notebook, padding="10")
        sub_notebook.add(launcher_tab, text="Launcher")
        
        # Create main container with two columns (left for content, right for instance manager)
        launcher_main_frame = ttk.Frame(launcher_tab)
        launcher_main_frame.pack(fill=tk.BOTH, expand=True)
        launcher_main_frame.columnconfigure(0, weight=2)  # Left side (content)
        launcher_main_frame.columnconfigure(1, weight=1)  # Right side (instance manager)
        
        # Left side: scrollable canvas for launcher content
        launcher_left_frame = ttk.Frame(launcher_main_frame)
        launcher_left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        launcher_canvas = tk.Canvas(launcher_left_frame, highlightthickness=0)
        launcher_scrollbar = ttk.Scrollbar(launcher_left_frame, orient="vertical", command=launcher_canvas.yview)
        launcher_scrollable_frame = ttk.Frame(launcher_canvas)
        
        launcher_scrollable_frame.bind(
            "<Configure>",
            lambda e: launcher_canvas.configure(scrollregion=launcher_canvas.bbox("all"))
        )
        
        launcher_canvas.create_window((0, 0), window=launcher_scrollable_frame, anchor="nw")
        launcher_canvas.configure(yscrollcommand=launcher_scrollbar.set)
        
        # Grid layout for scrollable content
        launcher_scrollable_frame.columnconfigure(0, weight=1)
        
        # Right side: Instance Manager
        instance_manager_frame = ttk.LabelFrame(launcher_main_frame, text="Instance Manager", padding="10")
        instance_manager_frame.grid(row=0, column=1, sticky="nsew")
        instance_manager_frame.columnconfigure(0, weight=1)
        instance_manager_frame.rowconfigure(1, weight=1)
        
        # Instance list header
        ttk.Label(instance_manager_frame, text="Active Instances:", font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Create Treeview for instance list
        instance_tree_frame = ttk.Frame(instance_manager_frame)
        instance_tree_frame.grid(row=1, column=0, sticky="nsew")
        instance_tree_frame.columnconfigure(0, weight=1)
        instance_tree_frame.rowconfigure(0, weight=1)
        
        # Treeview with columns
        instance_tree = ttk.Treeview(instance_tree_frame, columns=("credential", "port"), show="tree headings", height=10)
        instance_tree.heading("#0", text="Instance")
        instance_tree.heading("credential", text="Credential")
        instance_tree.heading("port", text="Port")
        instance_tree.column("#0", width=120, minwidth=100)
        instance_tree.column("credential", width=150, minwidth=120)
        instance_tree.column("port", width=80, minwidth=60)
        
        # Scrollbar for treeview
        instance_tree_scrollbar = ttk.Scrollbar(instance_tree_frame, orient="vertical", command=instance_tree.yview)
        instance_tree.configure(yscrollcommand=instance_tree_scrollbar.set)
        
        instance_tree.grid(row=0, column=0, sticky="nsew")
        instance_tree_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Store instance tree for updates
        self.instance_tree = instance_tree
        
        row = 0
        
        # ===== Launch Settings Section =====
        launch_settings_frame = ttk.LabelFrame(launcher_scrollable_frame, text="Launch Settings", padding="10")
        launch_settings_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        launch_settings_frame.columnconfigure(1, weight=1)
        
        launch_row = 0
        ttk.Label(launch_settings_frame, text="Base Port:").grid(row=launch_row, column=0, sticky="w", padx=(0, 5), pady=2)
        ttk.Spinbox(launch_settings_frame, from_=1000, to=65535, textvariable=self.base_port, width=15).grid(row=launch_row, column=1, sticky="w", pady=2)
        launch_row += 1
        
        ttk.Label(launch_settings_frame, text="Launch Delay (seconds):").grid(row=launch_row, column=0, sticky="w", padx=(0, 5), pady=2)
        ttk.Spinbox(launch_settings_frame, from_=0, to=60, textvariable=self.launch_delay, width=15).grid(row=launch_row, column=1, sticky="w", pady=2)
        launch_row += 1
        
        ttk.Checkbutton(launch_settings_frame, text="Build Maven before launch", variable=self.build_maven).grid(row=launch_row, column=0, columnspan=2, sticky="w", pady=2)
        
        row += 1
        
        # ===== Credential Selection Section =====
        creds_frame = ttk.LabelFrame(launcher_scrollable_frame, text="Credential Selection", padding="10")
        creds_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        creds_frame.columnconfigure(0, weight=1)
        creds_frame.columnconfigure(2, weight=1)
        
        # Available credentials
        ttk.Label(creds_frame, text="Available Credentials:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        credentials_listbox = tk.Listbox(creds_frame, selectmode=tk.MULTIPLE, height=8)
        credentials_listbox.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        # Buttons between listboxes
        cred_buttons_frame = ttk.Frame(creds_frame)
        cred_buttons_frame.grid(row=1, column=1, padx=5)
        add_cred_button = ttk.Button(cred_buttons_frame, text=">", width=3)
        add_cred_button.pack(pady=2)
        remove_cred_button = ttk.Button(cred_buttons_frame, text="<", width=3)
        remove_cred_button.pack(pady=2)
        move_up_button = ttk.Button(cred_buttons_frame, text="↑", width=3)
        move_up_button.pack(pady=2)
        move_down_button = ttk.Button(cred_buttons_frame, text="↓", width=3)
        move_down_button.pack(pady=2)
        clear_creds_button = ttk.Button(cred_buttons_frame, text="Clear", width=5)
        clear_creds_button.pack(pady=2)
        
        # Selected credentials
        ttk.Label(creds_frame, text="Selected Credentials (Launch Order):").grid(row=0, column=2, sticky="w", pady=(0, 5))
        selected_credentials_listbox = tk.Listbox(creds_frame, height=8)
        selected_credentials_listbox.grid(row=1, column=2, sticky="nsew")
        
        creds_frame.rowconfigure(1, weight=1)
        
        row += 1
        
        # ===== Launch Controls Section =====
        launch_controls_frame = ttk.LabelFrame(launcher_scrollable_frame, text="Launch Controls", padding="10")
        launch_controls_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        launch_controls_row = 0
        instance_count_label = ttk.Label(launch_controls_frame, text="Instances: 0", style='Info.TLabel')
        instance_count_label.grid(row=launch_controls_row, column=0, sticky="w", padx=(0, 10))
        
        launch_button = ttk.Button(launch_controls_frame, text="Launch RuneLite Instances", style='Action.TButton')
        launch_button.grid(row=launch_controls_row, column=1, padx=5)
        
        stop_button = ttk.Button(launch_controls_frame, text="Stop All Instances", style='Danger.TButton')
        stop_button.grid(row=launch_controls_row, column=2, padx=5)
        
        row += 1
        
        # ===== Auto-Detection Section =====
        detection_frame = ttk.LabelFrame(launcher_scrollable_frame, text="Client Auto-Detection", padding="10")
        detection_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        detection_frame.columnconfigure(0, weight=1)
        
        detection_row = 0
        detection_status_label = ttk.Label(detection_frame, text="Auto-detection: Stopped", style='Info.TLabel')
        detection_status_label.grid(row=detection_row, column=0, columnspan=4, sticky="w", pady=(0, 5))
        detection_row += 1
        
        start_detection_button = ttk.Button(detection_frame, text="Start Auto-Detection", style='Action.TButton')
        start_detection_button.grid(row=detection_row, column=0, padx=(0, 5))
        
        stop_detection_button = ttk.Button(detection_frame, text="Stop Auto-Detection", style='Danger.TButton')
        stop_detection_button.grid(row=detection_row, column=1, padx=(0, 5))
        
        detect_now_button = ttk.Button(detection_frame, text="Detect Now", style='Info.TButton')
        detect_now_button.grid(row=detection_row, column=2, padx=(0, 5))
        
        test_detection_button = ttk.Button(detection_frame, text="Test Detection", style='Info.TButton')
        test_detection_button.grid(row=detection_row, column=3)
        
        row += 1
        
        # Pack launcher canvas and scrollbar
        launcher_canvas.grid(row=0, column=0, sticky="nsew")
        launcher_scrollbar.grid(row=0, column=1, sticky="ns")
        launcher_tab.rowconfigure(0, weight=1)
        launcher_tab.columnconfigure(0, weight=1)
        
        # Mouse wheel scrolling for launcher tab
        def _on_launcher_mousewheel(event):
            launcher_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        launcher_canvas.bind("<MouseWheel>", _on_launcher_mousewheel)
        launcher_scrollable_frame.bind("<MouseWheel>", _on_launcher_mousewheel)
        
        # ===== Setup & Configuration Sub-tab =====
        setup_tab = ttk.Frame(sub_notebook, padding="10")
        sub_notebook.add(setup_tab, text="Setup & Configuration")
        
        # Create scrollable canvas for setup tab
        setup_canvas = tk.Canvas(setup_tab, highlightthickness=0)
        setup_scrollbar = ttk.Scrollbar(setup_tab, orient="vertical", command=setup_canvas.yview)
        setup_scrollable_frame = ttk.Frame(setup_canvas)
        
        setup_scrollable_frame.bind(
            "<Configure>",
            lambda e: setup_canvas.configure(scrollregion=setup_canvas.bbox("all"))
        )
        
        setup_canvas.create_window((0, 0), window=setup_scrollable_frame, anchor="nw")
        setup_canvas.configure(yscrollcommand=setup_scrollbar.set)
        
        # Grid layout
        setup_scrollable_frame.columnconfigure(0, weight=1)
        setup_scrollable_frame.columnconfigure(1, weight=1)
        
        # ===== Setup & Configuration Section =====
        config_frame = ttk.LabelFrame(setup_scrollable_frame, text="Setup & Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        config_row = 0
        
        # Project Directory
        ttk.Label(config_frame, text="Project Directory:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        project_dir_entry = ttk.Entry(config_frame, textvariable=self.config_vars["projectDir"], width=50)
        project_dir_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("projectDir", False)).grid(row=config_row, column=2, pady=2)
        self.path_entries["projectDir"] = project_dir_entry
        config_row += 1
        
        # Class Path File
        ttk.Label(config_frame, text="Class Path File:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        classpath_entry = ttk.Entry(config_frame, textvariable=self.config_vars["classPathFile"], width=50)
        classpath_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("classPathFile", True)).grid(row=config_row, column=2, pady=2)
        self.path_entries["classPathFile"] = classpath_entry
        config_row += 1
        
        # Java Executable
        ttk.Label(config_frame, text="Java Executable:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        java_entry = ttk.Entry(config_frame, textvariable=self.config_vars["javaExe"], width=50)
        java_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("javaExe", True)).grid(row=config_row, column=2, pady=2)
        self.path_entries["javaExe"] = java_entry
        config_row += 1
        
        # Base Directory
        ttk.Label(config_frame, text="Base Directory:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        base_dir_entry = ttk.Entry(config_frame, textvariable=self.config_vars["baseDir"], width=50)
        base_dir_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("baseDir", False)).grid(row=config_row, column=2, pady=2)
        self.path_entries["baseDir"] = base_dir_entry
        config_row += 1
        
        # Exports Base
        ttk.Label(config_frame, text="Exports Base:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        exports_entry = ttk.Entry(config_frame, textvariable=self.config_vars["exportsBase"], width=50)
        exports_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("exportsBase", False)).grid(row=config_row, column=2, pady=2)
        self.path_entries["exportsBase"] = exports_entry
        config_row += 1
        
        # Credentials Directory
        ttk.Label(config_frame, text="Credentials Directory:").grid(row=config_row, column=0, sticky="w", padx=(0, 5), pady=2)
        creds_entry = ttk.Entry(config_frame, textvariable=self.config_vars["credentialsDir"], width=50)
        creds_entry.grid(row=config_row, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.config_manager.browse_path("credentialsDir", False)).grid(row=config_row, column=2, pady=2)
        self.path_entries["credentialsDir"] = creds_entry
        config_row += 1
        
        # Auto-detect checkbox
        auto_detect_check = ttk.Checkbutton(
            config_frame, 
            text="Auto-detect paths on startup",
            variable=self.config_vars["autoDetect"],
            command=self.config_manager.toggle_auto_detect
        )
        auto_detect_check.grid(row=config_row, column=0, columnspan=2, sticky="w", pady=(5, 0))
        config_row += 1
        
        # Config buttons
        config_buttons_frame = ttk.Frame(config_frame)
        config_buttons_frame.grid(row=config_row, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(config_buttons_frame, text="Auto-detect Paths", command=lambda: self.config_manager.auto_detect_paths(self._log_message)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_buttons_frame, text="Save Config", command=lambda: self.config_manager.save_config(
            self.base_port, self.launch_delay, self.build_maven, self._log_message
        )).pack(side=tk.LEFT, padx=(0, 5))
        setup_deps_button = ttk.Button(config_buttons_frame, text="Setup Dependencies")
        setup_deps_button.pack(side=tk.LEFT)
        
        # Pack setup canvas and scrollbar
        setup_canvas.grid(row=0, column=0, sticky="nsew")
        setup_scrollbar.grid(row=0, column=1, sticky="ns")
        setup_tab.rowconfigure(0, weight=1)
        setup_tab.columnconfigure(0, weight=1)
        
        # Mouse wheel scrolling for setup tab
        def _on_setup_mousewheel(event):
            setup_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        setup_canvas.bind("<MouseWheel>", _on_setup_mousewheel)
        setup_scrollable_frame.bind("<MouseWheel>", _on_setup_mousewheel)
        
        widgets = {
            'credentials_listbox': credentials_listbox,
            'selected_credentials_listbox': selected_credentials_listbox,
            'instance_count_label': instance_count_label,
            'launch_button': launch_button,
            'stop_button': stop_button,
            'detection_status_label': detection_status_label,
            'start_detection_button': start_detection_button,
            'stop_detection_button': stop_detection_button,
            'detect_now_button': detect_now_button,
            'test_detection_button': test_detection_button,
            'setup_deps_button': setup_deps_button,
            'add_cred_button': add_cred_button,
            'remove_cred_button': remove_cred_button,
            'move_up_button': move_up_button,
            'move_down_button': move_down_button,
            'clear_creds_button': clear_creds_button
        }
        
        return client_tab, widgets
    
    def _create_output_tab(self):
        """Create the Output sub-tab under the Client tab."""
        if not hasattr(self, 'client_sub_notebook') or self.client_sub_notebook is None:
            # If sub_notebook doesn't exist yet, create it (shouldn't happen, but safety check)
            return
        
        output_tab = ttk.Frame(self.client_sub_notebook, padding="5")
        self.client_sub_notebook.add(output_tab, text="Output")
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
    
    def _log_message(self, message: str, level: str = 'info'):
        """Wrapper for logging messages to the main log text widget."""
        if self.log_text:
            LoggingUtils.log_message(self.log_text, message, level)
    
    def _log_message_to_instance(self, instance_name: str, message: str, level: str = 'info'):
        """Wrapper for logging messages to an instance's log text widget."""
        if instance_name in self.instance_tabs:
            instance_tab = self.instance_tabs[instance_name]
            if hasattr(instance_tab, 'log_text'):
                LoggingUtils.log_message(instance_tab.log_text, message, level)
    
    def _stop_all_instances(self):
        """Stop all running instances."""
        if self.instance_manager:
            for instance_name in list(self.instance_tabs.keys()):
                if hasattr(self.instance_tabs[instance_name], 'is_running') and self.instance_tabs[instance_name].is_running:
                    self.instance_manager.stop_plans_for_instance(instance_name)
    
    def create_instance_tab_wrapper(self, instance_name: str, port: int):
        """Wrapper to create instance tab using instance manager."""
        # This will be called by the launcher when instances are created
        # For now, we'll use the instance_manager's create_instance_tab
        # Note: The full create_instance_tab implementation needs to be extracted
        # from gui.py lines 1893-2410
        if self.instance_manager:
            result = self.instance_manager.create_instance_tab(instance_name, port)
            # Update instance manager display
            self._update_instance_manager_display(instance_name, port)
            return result
        return None
    
    def _update_instance_manager_display(self, instance_name: str, port: int):
        """Update the instance manager treeview with a new instance."""
        if not hasattr(self, 'instance_tree') or self.instance_tree is None:
            return
        
        # Find the credential file name for this instance
        credential_file = None
        for cred in self.selected_credentials:
            cred_username = cred.replace('.properties', '')
            if cred_username == instance_name:
                credential_file = cred
                break
        
        # Extract just the credential name (part before .properties)
        credential_name = credential_file.replace('.properties', '') if credential_file else "Unknown"
        
        # Add to treeview
        self.instance_tree.insert("", tk.END, text=instance_name, values=(credential_name, port))
    
    def _remove_instance_from_display(self, instance_name: str):
        """Remove an instance from the instance manager treeview."""
        if not hasattr(self, 'instance_tree') or self.instance_tree is None:
            return
        
        # Find and remove the item
        for item in self.instance_tree.get_children():
            if self.instance_tree.item(item, "text") == instance_name:
                self.instance_tree.delete(item)
                break
    
    def _update_plan_details_wrapper(self, instance_name: str, listbox: tk.Listbox):
        """Wrapper for updating plan details."""
        # This should call the plan editor's update method
        # For now, placeholder
        pass
    
    def _update_parameter_widgets_wrapper(self, instance_name: str, listbox: tk.Listbox):
        """Wrapper for updating parameter widgets."""
        # This should call the plan editor's parameter update method
        # For now, placeholder
        pass
    
    def _focus_runelite_window(self, username: str):
        """Focus the RuneLite window for a given username."""
        # This would focus the RuneLite window - implementation depends on window management
        # For now, placeholder
        pass
    
    def update_instance_phase(self, instance_name: str, phase: str):
        """Update the current phase display for an instance."""
        if instance_name not in self.instance_tabs:
            return
        
        instance_tab = self.instance_tabs[instance_name]
        if hasattr(instance_tab, 'current_phase_label'):
            instance_tab.current_phase_label.config(text=phase)
            instance_tab.current_phase = phase
    
    def on_closing(self):
        """Handle window closing event."""
        # Stop all running instances
        if self.instance_manager:
            for instance_name in list(self.instance_tabs.keys()):
                if hasattr(self.instance_tabs[instance_name], 'is_running') and self.instance_tabs[instance_name].is_running:
                    self.instance_manager.stop_plans_for_instance(instance_name)
        
        # Stop all stats monitors
        for username in list(self.stats_monitors.keys()):
            self.statistics.stop_stats_monitor(username)
        
        # Stop client detection
        if self.client_detector:
            self.client_detector.stop_client_detection()
        
        # Save configuration
        if self.config_manager:
            self.config_manager.save_config()
        
        # Close window
        self.root.destroy()
