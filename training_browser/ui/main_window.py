"""
Main Window

Main window setup with notebook and tabs.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any

from .widgets.feature_table import FeatureTableView
from .widgets.target_view import TargetView
from .widgets.action_tensors_view import ActionTensorsView
from .widgets.sequence_alignment_view import SequenceAlignmentView
from .widgets.feature_analysis_view import FeatureAnalysisView
from .widgets.normalization_view import NormalizationView


class MainWindow:
    """Main window containing all tabs and views."""
    
    def __init__(self, root: tk.Tk, controller):
        """
        Initialize the main window.
        
        Args:
            root: Tkinter root window
            controller: Application controller
        """
        self.root = root
        self.controller = controller
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create navigation bar
        self._create_navigation_bar()
        
        # Create notebook for tabs
        self._create_notebook()
        
        # Create status bar
        self._create_status_bar()
        
        # Initialize views
        self.views = {}
        self._initialize_views()
        
        # Register with controller
        controller.register_view('main_window', self)
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Change Data Root...", command=self._on_change_data_root)
        file_menu.add_command(label="Reload Data", command=self._on_reload_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Translations", 
                                variable=self.controller.tk_show_translations,
                                command=self.controller.toggle_translations)
        view_menu.add_checkbutton(label="Show Normalized Data",
                                variable=self.controller.get_state().show_normalized,
                                command=self.controller.toggle_normalized_data)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._on_about)
    
    def _create_navigation_bar(self):
        """Create the navigation bar."""
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.pack(fill="x", pady=(0, 10))
        
        # Data root display
        self.data_root_label = ttk.Label(nav_frame, text="Data Root: data")
        self.data_root_label.pack(side="left")
        
        # Change data root button
        self.change_data_root_button = ttk.Button(
            nav_frame,
            text="Change Data Root...",
            command=self._on_change_data_root
        )
        self.change_data_root_button.pack(side="right")
        
        # Reload button
        self.reload_button = ttk.Button(
            nav_frame,
            text="Reload",
            command=self._on_reload_data
        )
        self.reload_button.pack(side="right", padx=(0, 10))
        
        # Translation toggle button
        self.translation_button = ttk.Checkbutton(
            nav_frame,
            text="Show Translations",
            variable=self.controller.tk_show_translations,
            command=self.controller.toggle_translations
        )
        self.translation_button.pack(side="right", padx=(0, 10))
        
        # Sequence navigation
        seq_frame = ttk.Frame(nav_frame)
        seq_frame.pack(side="right", padx=(0, 20))
        
        ttk.Label(seq_frame, text="Sequence:").pack(side="left")
        
        self.prev_button = ttk.Button(
            seq_frame,
            text="◀",
            width=3,
            command=self.controller.previous_sequence
        )
        self.prev_button.pack(side="left", padx=(5, 0))
        
        self.sequence_var = tk.StringVar(value="0")
        self.sequence_entry = ttk.Entry(
            seq_frame,
            textvariable=self.sequence_var,
            width=8
        )
        self.sequence_entry.pack(side="left", padx=(5, 0))
        
        self.next_button = ttk.Button(
            seq_frame,
            text="▶",
            width=3,
            command=self.controller.next_sequence
        )
        self.next_button.pack(side="left", padx=(5, 0))
        
        # Bind sequence entry
        self.sequence_entry.bind("<Return>", self._on_sequence_jump)
        self.sequence_entry.bind("<FocusOut>", self._on_sequence_jump)
    
    def _create_notebook(self):
        """Create the notebook with tabs."""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self._create_tabs()
    
    def _create_tabs(self):
        """Create all the tabs."""
        # Gamestate Features tab
        self.features_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.features_tab, text="Gamestate Features")
        
        # Target Actions tab
        self.targets_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.targets_tab, text="Target Actions")
        
        # Action Tensors tab
        self.action_tensors_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.action_tensors_tab, text="Action Tensors")
        
        # Sequence Alignment tab
        self.sequence_alignment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sequence_alignment_tab, text="Sequence Alignment")
        
        # Feature Analysis tab
        self.feature_analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_analysis_tab, text="Feature Analysis")
        
        # Normalization tab
        self.normalization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.normalization_tab, text="Normalization")
    
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill="x", pady=(10, 0))
        
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            font=("Tahoma", 8)
        )
        self.status_label.pack(side="left")
        
        # Data summary
        self.data_summary_label = ttk.Label(
            self.status_frame,
            text="No data loaded",
            font=("Tahoma", 8)
        )
        self.data_summary_label.pack(side="right")
    
    def _initialize_views(self):
        """Initialize all view components."""
        # Feature table view
        self.views['feature_table'] = FeatureTableView(
            self.features_tab,
            self.controller
        )
        self.views['feature_table'].pack(fill="both", expand=True)
        
        # Target view
        self.views['target_view'] = TargetView(
            self.targets_tab,
            self.controller
        )
        self.views['target_view'].pack(fill="both", expand=True)
        
        # Action tensors view
        self.views['action_tensors_view'] = ActionTensorsView(
            self.action_tensors_tab,
            self.controller
        )
        self.views['action_tensors_view'].pack(fill="both", expand=True)
        
        # Sequence alignment view
        self.views['sequence_alignment_view'] = SequenceAlignmentView(
            self.sequence_alignment_tab,
            self.controller
        )
        self.views['sequence_alignment_view'].pack(fill="both", expand=True)
        
        # Feature analysis view
        self.views['feature_analysis_view'] = FeatureAnalysisView(
            self.feature_analysis_tab,
            self.controller
        )
        self.views['feature_analysis_view'].pack(fill="both", expand=True)
        
        # Normalization view
        self.views['normalization_view'] = NormalizationView(
            self.normalization_tab,
            self.controller
        )
        self.views['normalization_view'].pack(fill="both", expand=True)
        
        # Register all views with controller
        for name, view in self.views.items():
            self.controller.register_view(name, view)
        
        # Initial refresh to populate views with data
        self._update_ui_after_data_change()
        # Ensure views render initial data
        self.refresh()
    
    def _on_change_data_root(self):
        """Handle change data root menu action."""
        if self.controller.change_data_root():
            self._update_ui_after_data_change()
    
    def _on_reload_data(self):
        """Handle reload data menu action."""
        if self.controller.reload_data():
            self._update_ui_after_data_change()
    
    def _on_sequence_jump(self, event=None):
        """Handle sequence jump from entry field."""
        try:
            sequence_idx = int(self.sequence_var.get())
            self.controller.jump_to_sequence(sequence_idx)
            self._update_sequence_display()
        except ValueError:
            # Reset to current sequence if invalid input
            self._update_sequence_display()
    
    def _on_about(self):
        """Show about dialog."""
        about_text = """Training Browser v1.0.0

A modular GUI for browsing training data with behavior-preserving refactoring.

Features:
• Browse gamestate features and action sequences
• View normalized and raw data
• Export data to various formats
• Change data root directories
• Feature group filtering and analysis"""
        
        messagebox.showinfo("About Training Browser", about_text)
    
    def _update_ui_after_data_change(self):
        """Update UI elements after data has changed."""
        # Update data root label
        data_root = self.controller.get_state().data_root
        self.data_root_label.config(text=f"Data Root: {data_root}")
        
        # Update data summary
        summary = self.controller.get_data_summary()
        if summary:
            summary_text = f"{summary['sequence_count']} sequences, {summary['feature_count']} features"
            self.data_summary_label.config(text=summary_text)
        else:
            self.data_summary_label.config(text="No data loaded")
        
        # Update sequence display
        self._update_sequence_display()
        
        # Update status
        self.status_label.config(text="Data loaded successfully")
        # Refresh views to reflect newly loaded data
        self.refresh()
    
    def _update_sequence_display(self):
        """Update sequence navigation display."""
        if not self.controller.is_data_loaded():
            self.sequence_var.set("0")
            return
        
        current_seq = self.controller.get_state().current_sequence
        total_seqs = self.controller.get_sequence_count()
        
        self.sequence_var.set(str(current_seq))
        
        # Update button states
        self.prev_button.config(state="normal" if current_seq > 0 else "disabled")
        self.next_button.config(state="normal" if current_seq < total_seqs - 1 else "disabled")
    
    def refresh(self):
        """Refresh the main window."""
        self._update_sequence_display()
        
        # Refresh all views
        for view in self.views.values():
            if hasattr(view, 'refresh'):
                view.refresh()
    
    def update(self):
        """Update the main window."""
        self._update_sequence_display()
        
        # Update all views
        for view in self.views.values():
            if hasattr(view, 'update'):
                view.update()
