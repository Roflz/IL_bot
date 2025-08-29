"""
Controller

Orchestrates views and services, handles UI state and events.
"""

import tkinter as tk
from typing import Optional, Dict, Any
from pathlib import Path

from .state import UIState
from .services.data_loader import load_all, LoadedData
from .services.mapping_service import MappingService
from .services.normalization_service import NormalizationService
from .services.action_decoder import ActionDecoder
from .services.feature_catalog import FeatureCatalog
from .ui.dialogs import ChangeDataRootDialog, show_error_dialog, show_info_dialog


class Controller:
    """Main controller for the training browser application."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the controller.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.state = UIState()
        
        # Tkinter variables
        self.tk_show_translations = tk.BooleanVar(value=self.state.show_translations)
        
        # Services
        self.data_loader = None
        self.mapping_service = None
        self.normalization_service = None
        self.action_decoder = None
        self.feature_catalog = None
        
        # Data
        self.loaded_data: Optional[LoadedData] = None
        
        # Views (will be set by main window)
        self.views = {}
        
        # Initialize with default data root
        self._try_load_data(self.state.data_root)
    
    def _try_load_data(self, data_root: str) -> bool:
        """
        Try to load data from the specified root.
        
        Args:
            data_root: Path to data directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading data from: {data_root}")
            self.loaded_data = load_all(data_root)
            
            # Initialize services
            self._initialize_services()
            
            # Update state
            self.state.data_root = data_root
            
            # Update window title
            self.root.title(f"Training Browser - {data_root}")
            
            # Refresh all views
            self._refresh_all_views()
            
            print(f"✓ Data loaded successfully from {data_root}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load data from {data_root}: {e}"
            print(f"❌ {error_msg}")
            show_error_dialog(self.root, "Data Load Error", error_msg)
            return False
    
    def _initialize_services(self):
        """Initialize all services with loaded data."""
        if not self.loaded_data:
            return
        
        # Initialize mapping service
        self.mapping_service = MappingService(self.loaded_data.id_mappings)
        
        # Initialize normalization service
        feature_mappings_file = str(Path(self.state.data_root) / "05_mappings" / "feature_mappings.json")
        self.normalization_service = NormalizationService(feature_mappings_file)
        
        # Initialize action decoder
        self.action_decoder = ActionDecoder()
        
        # Initialize feature catalog
        self.feature_catalog = FeatureCatalog(self.loaded_data.feature_mappings)
    
    def _refresh_all_views(self):
        """Refresh all registered views."""
        for view_name, view in self.views.items():
            try:
                if hasattr(view, 'refresh'):
                    view.refresh()
                elif hasattr(view, 'update'):
                    view.update()
            except Exception as e:
                print(f"Warning: Failed to refresh view {view_name}: {e}")
    
    def register_view(self, name: str, view):
        """
        Register a view with the controller.
        
        Args:
            name: View name
            view: View instance
        """
        self.views[name] = view
        print(f"Registered view: {name}")
    
    def get_data_root(self) -> str:
        """Get the current data root directory."""
        return self.state.data_root
    
    def get_loaded_data(self) -> Optional[LoadedData]:
        """
        Get the currently loaded data.
        
        Returns:
            LoadedData instance or None if no data loaded
        """
        return self.loaded_data
    
    def get_mapping_service(self) -> Optional[MappingService]:
        """
        Get the mapping service.
        
        Returns:
            MappingService instance or None if not initialized
        """
        return self.mapping_service
    
    def get_normalization_service(self) -> Optional[NormalizationService]:
        """
        Get the normalization service.
        
        Returns:
            NormalizationService instance or None if not initialized
        """
        return self.normalization_service
    
    def get_action_decoder(self) -> Optional[ActionDecoder]:
        """
        Get the action decoder service.
        
        Returns:
            ActionDecoder instance or None if not initialized
        """
        return self.action_decoder
    
    def get_feature_catalog(self) -> Optional[FeatureCatalog]:
        """
        Get the feature catalog service.
        
        Returns:
            FeatureCatalog instance or None if not initialized
        """
        return self.feature_catalog
    
    def get_state(self) -> UIState:
        """
        Get the current UI state.
        
        Returns:
            UIState instance
        """
        return self.state
    
    def change_data_root(self) -> bool:
        """
        Show dialog to change data root and reload data.
        
        Returns:
            True if data root changed successfully, False otherwise
        """
        if not self.loaded_data:
            return False
        
        dialog = ChangeDataRootDialog(self.root, self.state.data_root)
        new_root = dialog.show()
        
        if new_root and new_root != self.state.data_root:
            return self._try_load_data(new_root)
        
        return False
    
    def reload_data(self) -> bool:
        """
        Reload data from the current data root.
        
        Returns:
            True if successful, False otherwise
        """
        return self._try_load_data(self.state.data_root)
    
    def next_sequence(self):
        """Move to the next sequence."""
        if not self.loaded_data:
            return
        
        max_sequence = len(self.loaded_data.input_sequences) - 1
        if self.state.current_sequence < max_sequence:
            self.state.current_sequence += 1
            self._refresh_sequence_views()
    
    def previous_sequence(self):
        """Move to the previous sequence."""
        if self.state.current_sequence > 0:
            self.state.current_sequence -= 1
            self._refresh_sequence_views()
    
    def jump_to_sequence(self, sequence_idx: int):
        """
        Jump to a specific sequence.
        
        Args:
            sequence_idx: Target sequence index
        """
        if not self.loaded_data:
            return
        
        max_sequence = len(self.loaded_data.input_sequences) - 1
        if 0 <= sequence_idx <= max_sequence:
            self.state.current_sequence = sequence_idx
            self._refresh_sequence_views()
    
    def toggle_normalized_data(self):
        """Toggle normalized data display."""
        self.state.show_normalized = not self.state.show_normalized
        self._refresh_feature_views()
    
    def toggle_translations(self):
        """Toggle feature value translations."""
        self.state.show_translations = self.tk_show_translations.get()
        self._refresh_feature_views()
    
    def toggle_action_normalized(self):
        """Toggle normalized action data display."""
        self.state.show_action_normalized = not self.state.show_action_normalized
        self._refresh_action_views()
    
    def set_feature_group_filter(self, filter_group: str):
        """
        Set the feature group filter.
        
        Args:
            filter_group: Feature group to filter by
        """
        if filter_group != self.state.feature_group_filter:
            self.state.feature_group_filter = filter_group
            self._refresh_feature_views()
    
    def _refresh_sequence_views(self):
        """Refresh views that depend on sequence selection."""
        sequence_views = ['feature_table', 'target_view', 'action_tensors_view']
        for view_name in sequence_views:
            if view_name in self.views:
                try:
                    view = self.views[view_name]
                    if hasattr(view, 'refresh'):
                        view.refresh()
                    elif hasattr(view, 'update'):
                        view.update()
                except Exception as e:
                    print(f"Warning: Failed to refresh sequence view {view_name}: {e}")
    
    def _refresh_feature_views(self):
        """Refresh views that depend on feature display settings."""
        feature_views = ['feature_table', 'feature_analysis_view']
        for view_name in feature_views:
            if view_name in self.views:
                try:
                    view = self.views[view_name]
                    if hasattr(view, 'refresh'):
                        view.refresh()
                    elif hasattr(view, 'update'):
                        view.update()
                except Exception as e:
                    print(f"Warning: Failed to refresh feature view {view_name}: {e}")
    
    def _refresh_action_views(self):
        """Refresh views that depend on action display settings."""
        action_views = ['action_tensors_view', 'target_view']
        for view_name in action_views:
            if view_name in action_views:
                try:
                    view = self.views[view_name]
                    if hasattr(view, 'refresh'):
                        view.refresh()
                    elif hasattr(view, 'update'):
                        view.update()
                except Exception as e:
                    print(f"Warning: Failed to refresh action view {view_name}: {e}")
    
    def get_current_sequence_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data for the current sequence.
        
        Returns:
            Dictionary with current sequence data or None if no data
        """
        if not self.loaded_data:
            return None
        
        current_idx = self.state.current_sequence
        if current_idx >= len(self.loaded_data.input_sequences):
            return None
        
        return {
            'sequence_idx': current_idx,
            'input_sequence': self.loaded_data.input_sequences[current_idx],
            'target_sequence': self.loaded_data.target_sequences[current_idx] if current_idx < len(self.loaded_data.target_sequences) else None,
            'action_input_sequence': self.loaded_data.action_input_sequences[current_idx] if self.loaded_data.action_input_sequences is not None and current_idx < len(self.loaded_data.action_input_sequences) else None,
            'normalized_sequence': self.loaded_data.normalized_input_sequences[current_idx] if self.loaded_data.normalized_input_sequences is not None else None
        }
    
    def get_sequence_count(self) -> int:
        """
        Get the total number of sequences.
        
        Returns:
            Number of sequences
        """
        if not self.loaded_data:
            return 0
        return len(self.loaded_data.input_sequences)
    
    def is_data_loaded(self) -> bool:
        """
        Check if data is currently loaded.
        
        Returns:
            True if data is loaded, False otherwise
        """
        return self.loaded_data is not None
    
    def get_data_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Data summary dictionary or None if no data
        """
        if not self.loaded_data:
            return None
        
        return {
            'data_root': self.state.data_root,
            'sequence_count': len(self.loaded_data.input_sequences),
            'feature_count': len(self.loaded_data.feature_mappings),
            'has_normalized_sequences': self.loaded_data.normalized_input_sequences is not None,
            'has_action_sequences': self.loaded_data.action_input_sequences is not None,
            'has_raw_data': self.loaded_data.raw_action_data is not None,
            'has_normalized_data': self.loaded_data.normalized_action_data is not None
        }
