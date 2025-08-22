#!/usr/bin/env python3
"""Live Features View - displays rolling 10x128 feature window using tksheet"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from ...util.formatting import format_value_for_display
from ..styles import create_dark_stringvar, create_dark_booleanvar

LOG = logging.getLogger(__name__)

try:
    from tksheet import Sheet
    TKSHEET_AVAILABLE = True
except ImportError as e:
    TKSHEET_AVAILABLE = False
    LOG.error("tksheet import failed: %s", e)
    raise ImportError("tksheet is required but not available. Install with: pip install tksheet") from e


class LiveFeaturesView(ttk.Frame):
    """View for displaying live feature data in a rolling 10x128 window using tksheet"""
    
    # --- Threading guard ----------------------------------------------------
    def _assert_main_thread(self, where: str):
        cur = threading.current_thread()
        if cur is not threading.main_thread():
            raise RuntimeError(f"Tkinter call from non-main thread in {where}: {cur.name}")
    
    def __init__(self, parent, controller):
        # Precondition checks
        if parent is None:
            raise ValueError("parent cannot be None")
        if controller is None:
            raise ValueError("controller cannot be None")
        if not hasattr(controller, 'ui_state'):
            raise RuntimeError("controller.ui_state not initialized")
        
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.feature_names = None
        self.feature_groups = None
        self._last_window = None  # (10,128)
        
        # Color bits for tracking cell colors (cyan ⇄ white)
        self._color_bits: Optional[np.ndarray] = None
        
        # Schema set flag
        self._schema_set = False
        
        # UI state - sync with controller
        self.show_translations = self.controller.ui_state.show_translations
        self.feature_group_filter = "All"
        self.search_text = ""
        
        # Collapsible groups and favorites
        self.expanded_groups = set()  # Set of expanded group names
        self.favorite_features = set()  # Set of favorite feature indices
        self.group_rows = {}  # Map group names to their row items in the sheet
        
        self._setup_ui()
        self._bind_events()
        
        # Sync translations variable with controller state
        self.translations_var.set(self.show_translations)
        
        # Load favorites from file
        self._load_favorites()
        
        # Initialize groups as collapsed by default (empty set means all collapsed)
        self.expanded_groups = set()  # All feature groups collapsed initially
        
        LOG.info("LiveFeaturesView: initialized with tksheet")
    
    def _setup_ui(self):
        """Setup the user interface with tksheet"""
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # Table gets most space
        
        # Header
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Live Feature Tracking", 
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        
        # Controls frame
        controls_frame = ttk.Frame(self, style="Toolbar.TFrame")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Left controls
        ttk.Button(controls_frame, text="▶ Start", 
                  command=self._start_live_mode).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls_frame, text="⏹ Stop", 
                  command=self._stop_live_mode).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls_frame, text="🗑️ Clear Buffers", 
                  command=self._clear_buffers).grid(row=0, column=2, padx=(0, 12))
        
        # Center controls
        ttk.Button(controls_frame, text="📋 Copy Table", 
                  command=self._copy_to_clipboard).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(controls_frame, text="💾 Export CSV", 
                  command=self._export_to_csv).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(controls_frame, text="📁 Expand All", 
                  command=self._expand_all_groups).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(controls_frame, text="📂 Collapse All", 
                  command=self._collapse_all_groups).grid(row=0, column=5, padx=(0, 12))
        
        # Filter controls
        ttk.Label(controls_frame, text="Group:").grid(row=0, column=6, padx=(0, 4))
        self.group_combo = ttk.Combobox(controls_frame, values=["All"], width=15, state="readonly")
        self.group_combo.grid(row=0, column=7, padx=(0, 12))
        self.group_combo.set("All")
        
        ttk.Label(controls_frame, text="Search:").grid(row=0, column=8, padx=(0, 4))
        self.search_var = create_dark_stringvar(self)
        self.search_entry = ttk.Entry(controls_frame, textvariable=self.search_var, width=20)
        self.search_entry.grid(row=0, column=9, padx=(0, 12))
        
        # Right controls
        self.translations_var = create_dark_booleanvar(self, value=True)
        ttk.Checkbutton(controls_frame, text="Show Translations", 
                       variable=self.translations_var).grid(row=0, column=10)
        
        # Summary line
        self.summary_label = ttk.Label(self, text="Features: 0/128 | Buffer: 0/10 | Status: Ready", 
                                     font=("Arial", 9))
        self.summary_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Build the columns: ["feature", "group", "t0","t1",...,"t9"]
        self._setup_columns()
        
        LOG.info("tksheet initialized; headers set; read-only; dark theme")
    
    def _setup_columns(self):
        """Setup the tksheet with proper columns"""
        # Create tksheet with proper headers - T0 (current) on the left, T9 (oldest) on the right
        headers = ["Feature", "Index", "Group", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        
        # Create parent frame for the sheet
        sheet_frame = ttk.Frame(self)
        sheet_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        sheet_frame.grid_columnconfigure(0, weight=1)
        sheet_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize tksheet
        self.sheet = Sheet(sheet_frame, headers=headers)
        self.sheet.grid(row=0, column=0, sticky="nsew")
        
        # Configure sheet properties
        self.sheet.enable_bindings()
        self.sheet.readonly = True  # Make cells read-only
        
        # Apply dark theme styling
        self.sheet.change_theme("dark")
        
        # Configure column widths
        column_widths = [200, 60, 100, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
        for i, width in enumerate(column_widths):
            self.sheet.column_width(column=i, width=width)

    def _bind_events(self):
        """Bind UI events"""
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_change)
        self.search_var.trace("w", self._on_search_change)
        self.translations_var.trace("w", self._on_translations_change)
        
        # NEW: make group headers clickable in the sheet
        try:
            # Works on recent tksheet
            self.sheet.extra_bindings([
                ("cell_select", self._on_sheet_cell_click),
                ("double_click_cell", self._on_sheet_cell_click),
            ])
            LOG.info("tksheet extra_bindings for clicks installed")
        except Exception as e:
            LOG.warning("tksheet extra_bindings not available: %s (group toggling by click disabled)", e)
    
    def _on_group_change(self, event):
        """Handle group filter change"""
        self.feature_group_filter = self.group_combo.get()
        self._refresh_table()
    
    def _on_search_change(self, *args):
        """Handle search text change"""
        self.search_text = self.search_var.get().lower()
        self._refresh_table()
    
    def _on_translations_change(self, *args):
        """Handle translations toggle change"""
        self.show_translations = self.translations_var.get()
        # Update controller state
        self.controller.ui_state.show_translations = self.show_translations
        self._refresh_table()
    
    def _on_sheet_cell_click(self, event):
        """
        Toggle a group when its header row is clicked. Works across tksheet versions by
        robustly parsing the event argument (dict or tuple).
        """
        try:
            # tksheet >= 6.x typically passes a dict, older may pass a tuple/list
            row = col = None
            if isinstance(event, dict):
                # Try to get row/col from the selected field first
                if 'selected' in event and hasattr(event['selected'], 'row') and hasattr(event['selected'], 'column'):
                    row = event['selected'].row
                    col = event['selected'].column
                else:
                    # Fallback to direct keys
                    row = event.get("row")
                    col = event.get("column")
            elif isinstance(event, (tuple, list)) and len(event) >= 2:
                row, col = event[0], event[1]

            if row is None:
                return

            # Check if click is on a group header row
            group_name = None
            for gname, ginfo in self.group_rows.items():
                if ginfo['header_row'] == row:
                    group_name = gname
                    break
            
            if group_name and (col in (None, 0)):
                self.toggle_group(group_name)  # this calls _refresh_table and logs new state
                return "break"  # prevent tksheet changing selection focus further on this click
        except Exception:
            LOG.exception("_on_sheet_cell_click failed")
    
    def set_schema(self, feature_names: List[str], feature_groups: List[str]):
        """
        Builds the 128 rows; sets "feature", "index", "group", blanks time cells.
        
        Args:
            feature_names: List of 128 feature names
            feature_groups: List of 128 feature groups
            
        Raises:
            ValueError: If lists don't have length 128
        """
        self._assert_main_thread("set_schema")
        
        # Validate inputs
        if len(feature_names) != 128:
            raise ValueError(f"feature_names must have length 128, got {len(feature_names)}")
        if len(feature_groups) != 128:
            raise ValueError(f"feature_groups must have length 128, got {len(feature_groups)}")
        
        self.feature_names = list(feature_names)
        self.feature_groups = list(feature_groups)
        
        # Initialize color bits
        self._color_bits = np.zeros((10, 128), dtype=bool)
        
        # Mark schema as set
        self._schema_set = True
        
        # Build the collapsible table
        self._refresh_table()
        
        LOG.info("Schema set: %d feature names, %d feature groups", len(feature_names), len(feature_groups))
    
    def _realize_all_rows(self):
        """Create rows for all 128 features"""
        # Clear existing data
        self.sheet.clear()
        
        # Create rows for all features
        for i in range(128):
            name = self.feature_names[i]
            group = self.feature_groups[i]
            
            # Create row data: Feature, Index, Group, blank T0..T9
            row_data = [name, str(i), group] + [""] * 10  # 10 time columns
            
            # Insert row
            self.sheet.insert_row(row_data, idx=i)
        
        # Force the sheet to redraw
        self.sheet.refresh()
        
        LOG.info("Realized all 128 feature rows")
    
    def update_translations_state(self, show: bool):
        """Enable/disable translations and re-render the current window."""
        self._assert_main_thread("update_translations_state")
        self.show_translations = bool(show)
        try:
            # keep the local checkbox (if present) in sync
            self.translations_var.set(self.show_translations)
        except Exception:
            pass

        # Force a full repaint using the last window so labels update immediately
        if getattr(self, "_last_window", None) is not None:
            mask = np.ones_like(self._last_window, dtype=bool)
            self.update_from_window(self._last_window, changed_mask=mask)
        else:
            if hasattr(self, "sheet"):
                self.sheet.refresh()
    
    def update_from_window(self, window: np.ndarray, changed_mask=None):
        """
        Update table with window data using changed_mask for change detection.
        
        Args:
            window: np.ndarray shape (10,128), time rows t0..t9, features columns f0..f127
            changed_mask: Boolean mask indicating changed cells, or None to compute automatically
            
        Raises:
            RuntimeError: If schema not set or shapes don't match
        """
        self._assert_main_thread("update_from_window")
        
        # Check schema is set
        if self.feature_names is None:
            raise RuntimeError("update_from_window called before set_schema()")

        if window.shape != (10, 128):
            raise ValueError(f"window shape {window.shape} != (10,128)")

        # If no mask provided, compute vs last window
        if changed_mask is None:
            if self._last_window is None or self._last_window.shape != window.shape:
                changed_mask = np.ones_like(window, dtype=bool)
            else:
                changed_mask = (window != self._last_window)

        # Iterate features, update all 10 time columns for changed positions
        updated = 0
        for f_idx in range(128):
            # Find the actual row in the sheet for this feature
            sheet_row = self._get_feature_sheet_row(f_idx)
            if sheet_row is None:
                continue  # Feature is hidden (collapsed group)
            
            for t_idx in range(10):  # t0..t9 LEFT→RIGHT
                if not changed_mask[t_idx, f_idx]:
                    continue

                value = window[t_idx, f_idx]
                
                # write value (row=sheet_row, col=3+t_idx)
                # Columns: 0=Feature, 1=Index, 2=Group, time starts at col=3
                col = 3 + t_idx

                # Try to translate the raw value if translations are enabled
                mapped = None
                if getattr(self, "show_translations", False):
                    try:
                        # Get the feature's group as a hint from the table's "Group" column (col index 2).
                        group_hint = None
                        if hasattr(self, "sheet"):
                            try:
                                group_hint = self.sheet.get_cell_data(sheet_row, 2)
                            except Exception:
                                group_hint = None

                        mapped = self.controller.mapping_service.translate(f_idx, value, group_hint=group_hint)
                    except Exception:
                        mapped = None  # fail safe: fall back to raw value

                # Prefer mapped label when available; otherwise show the raw number
                text = mapped if (mapped is not None and mapped != "") else f"{value:.0f}"
                self.sheet.set_cell_data(sheet_row, col, text)

                # flip color on each change
                self._color_bits[t_idx, f_idx] = ~self._color_bits[t_idx, f_idx]
                new_color = "#00b3b3" if self._color_bits[t_idx, f_idx] else "#ffffff"
                self.sheet.highlight_cells(row=sheet_row, column=col, fg=new_color, redraw=False)

                updated += 1

        # Refresh the sheet after all updates
        self.sheet.refresh()
        
        # Store last window for next comparison
        self._last_window = window.copy()
        
        # Update Actions group if it exists
        self._update_actions_group()
    
    def _update_actions_group(self):
        """Update the Actions group with current values"""
        if "Actions" not in self.group_rows:
            return
        
        try:
            # Get current action tensors from controller
            action_tensors = self.controller.get_action_features()
            if not action_tensors or len(action_tensors) < 10:
                LOG.debug("No action tensors available for update")
                return
            
            # Update each action feature row
            for row_idx in self.group_rows["Actions"]['feature_rows']:
                # Get the feature name to determine which action type to count
                feature_name = self.sheet.get_cell_data(row_idx, 0)
                
                # Update T0-T9 columns with aggregated values from each timestep's tensor
                for t in range(10):
                    col_idx = 3 + t  # T0 starts at column 3
                    if t < len(action_tensors):
                        timestep_tensor = action_tensors[t]
                        if len(timestep_tensor) > 0:
                            if feature_name == "Action Count":
                                # Action count is always first element
                                value = timestep_tensor[0] if len(timestep_tensor) > 0 else 0
                                self.sheet.set_cell_data(row_idx, col_idx, f"{int(value)}")
                            else:
                                # For other features, count occurrences in the tensor
                                count = self._count_action_type_in_tensor(timestep_tensor, feature_name.lower().replace(" ", "_"))
                                self.sheet.set_cell_data(row_idx, col_idx, f"{count}")
                        else:
                            self.sheet.set_cell_data(row_idx, col_idx, "0")
                    else:
                        self.sheet.set_cell_data(row_idx, col_idx, "0")
            
            
        except Exception as e:
            LOG.error(f"Error updating Actions group: {e}")
    
    def _get_feature_sheet_row(self, feature_idx: int) -> Optional[int]:
        """
        Get the actual sheet row for a feature index, considering collapsed groups.
        
        Args:
            feature_idx: The feature index (0-127)
            
        Returns:
            Sheet row number if feature is visible, None if hidden (collapsed group)
        """
        if not self._schema_set or feature_idx >= len(self.feature_groups):
            return None
        
        group_name = self.feature_groups[feature_idx]
        
        # Check if group is expanded
        if group_name not in self.expanded_groups:
            return None
        
        # Find the group in group_rows
        if group_name not in self.group_rows:
            return None
        
        group_info = self.group_rows[group_name]
        if not group_info['expanded']:
            return None
        
        # Find the feature row within the group
        feature_name = self.feature_names[feature_idx]
        for row_idx in group_info['feature_rows']:
            try:
                if self.sheet.get_cell_data(row_idx, 0) == feature_name:
                    return row_idx
            except Exception as e:
                continue
        
        return None
    
    def _refresh_table(self):
        """Refresh the feature table with collapsible groups and favorites"""
        self._assert_main_thread("_refresh_table")
        try:
            if not self._schema_set:
                LOG.error("LiveFeaturesView: CRITICAL ERROR - schema not set in _refresh_table")
                return
            
            # Update group combo with unique groups
            if self.feature_groups:
                unique_groups = sorted(list(set(self.feature_groups)))
                current_groups = ["All"] + unique_groups
                self.group_combo['values'] = current_groups
                if self.feature_group_filter not in current_groups:
                    self.feature_group_filter = "All"
                    self.group_combo.set("All")
            
            # Clear the sheet and rebuild with collapsible groups
            if hasattr(self, 'sheet'):
                self.sheet.clear()
                self._build_collapsible_table()
            
            # If we have current data, update the table with it
            if hasattr(self, '_last_window') and self._last_window is not None:
                try:
                    # Force an update with the current window data
                    mask = np.ones_like(self._last_window, dtype=bool)
                    self.update_from_window(self._last_window, changed_mask=mask)
                except Exception as e:
                    pass
            
        except Exception as e:
            LOG.exception("LiveFeaturesView: _refresh_table failed with error")
            # Don't re-raise - just log and return gracefully
    
    def _build_collapsible_table(self):
        """Build the table with collapsible feature groups"""
        if not self.feature_names or not self.feature_groups:
            LOG.debug("_build_collapsible_table: no feature names or groups")
            return
        
        LOG.info(f"_build_collapsible_table: building table with {len(self.feature_names)} features in {len(set(self.feature_groups))} groups")
        
        # Group features by their feature group
        grouped_features = {}
        for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
            if group not in grouped_features:
                grouped_features[group] = []
            grouped_features[group].append((i, name, group))
        
        # Sort groups by their first feature index (Player first, then Interaction, Camera, etc.)
        def get_group_order(group_name):
            if group_name == "Player":
                return 0
            elif group_name == "Interaction":
                return 1
            elif group_name == "Camera":
                return 2
            elif group_name == "Inventory":
                return 3
            elif group_name == "Bank":
                return 4
            elif group_name == "Phase Context":
                return 5
            elif group_name == "Game Objects":
                return 6
            elif group_name == "NPCs":
                return 7
            elif group_name == "Tabs":
                return 8
            elif group_name == "Skills":
                return 9
            elif group_name == "Timestamp":
                return 10
            elif group_name == "Actions":
                return 11
            else:
                return 999  # Unknown groups go last
        
        # Ensure Actions group is always included
        all_groups = set(grouped_features.keys())
        all_groups.add("Actions")  # Actions group is always available
        
        sorted_groups = sorted(all_groups, key=get_group_order)
        
        row_idx = 0
        self.group_rows = {}  # Track which rows belong to which groups
        
        for group_name in sorted_groups:
            # Skip Actions group here - it will be handled separately
            if group_name == "Actions":
                continue
                
            features = grouped_features[group_name]
            
            # Check if group should be shown based on filter
            if self.feature_group_filter != "All" and group_name != self.feature_group_filter:
                continue
            
            # Check if group should be shown based on search
            if self.search_text:
                group_has_match = any(self.search_text in name.lower() for _, name, _ in features)
                if not group_has_match:
                    continue
            
            # Add group header row
            is_expanded = group_name in self.expanded_groups
            expand_icon = "▼" if is_expanded else "▶"
            group_header = [f"{expand_icon} {group_name}", "", "", "", "", "", "", "", "", "", "", "", ""]
            
            self.sheet.insert_row(group_header, idx=row_idx)
            
            # Style the group header row
            self.sheet.highlight_cells(row=row_idx, column=0, bg="#4a5568", fg="#ffffff")
            self.sheet.highlight_cells(row=row_idx, column=1, bg="#4a5568", fg="#ffffff")
            self.sheet.highlight_cells(row=row_idx, column=2, bg="#4a5568", fg="#ffffff")
            
            # Store group row info
            self.group_rows[group_name] = {
                'header_row': row_idx,
                'feature_rows': [],
                'expanded': is_expanded
            }
            
            row_idx += 1
            
            # Add feature rows if group is expanded
            if is_expanded:
                for feature_idx, name, group in features:
                    # Check if feature matches search
                    if self.search_text and self.search_text not in name.lower():
                        continue
                    
                    # Create feature row data with current values
                    feature_row = [name, str(feature_idx), group]
                    
                    # Fill in current feature values if available
                    if hasattr(self, '_last_window') and self._last_window is not None:
                        try:
                            # Get the current value for this feature from the last window
                            current_value = self._last_window[-1, feature_idx]  # Use the most recent timestep
                            # Try to translate the value if translations are enabled
                            mapped = None
                            if getattr(self, "show_translations", False):
                                try:
                                    mapped = self.controller.mapping_service.translate(feature_idx, current_value, group_hint=group_name)
                                except Exception:
                                    mapped = None
                            # Use translated value if available, otherwise raw value
                            display_value = mapped if (mapped is not None and mapped != "") else f"{current_value:.0f}"
                            feature_row.extend([display_value] * 10)  # T0-T9 columns all show current value
                        except Exception as e:
                            feature_row.extend([""] * 10)  # T0-T9 columns
                    else:
                        feature_row.extend([""] * 10)  # T0-T9 columns
                    
                    # Insert feature row
                    self.sheet.insert_row(feature_row, idx=row_idx)
                    
                    # Style favorite features
                    if feature_idx in self.favorite_features:
                        self.sheet.highlight_cells(row=row_idx, column=0, bg="#2c5282", fg="#ffffff")
                    
                    # Store feature row info
                    self.group_rows[group_name]['feature_rows'].append(row_idx)
                    
                    row_idx += 1
        
        # Add Actions group if it's in the sorted groups
        if "Actions" in sorted_groups:
            self._add_actions_group(row_idx)
    
    def _add_actions_group(self, start_row_idx: int):
        """Add the Actions group to the table"""
        
        row_idx = start_row_idx
        
        # Add Actions group header
        expand_icon = "▼" if "Actions" in self.expanded_groups else "▶"
        group_header = [f"{expand_icon} Actions", "", "", "", "", "", "", "", "", "", "", "", ""]
        
        self.sheet.insert_row(group_header, idx=row_idx)
        
        # Style the group header row
        self.sheet.highlight_cells(row=row_idx, column=0, bg="#4a5568", fg="#ffffff")
        self.sheet.highlight_cells(row=row_idx, column=1, bg="#4a5568", fg="#ffffff")
        self.sheet.highlight_cells(row=row_idx, column=2, bg="#4a5568", fg="#ffffff")
        
        # Store group row info
        self.group_rows["Actions"] = {
            'header_row': row_idx,
            'feature_rows': [],
            'expanded': "Actions" in self.expanded_groups
        }
        
        row_idx += 1
        
        # Add action feature rows only if group is expanded
        if "Actions" in self.expanded_groups:
            # Define meaningful action features based on aggregated counts
            action_features = [
                ("Action Count", "count"),           # Total actions in window
                ("Mouse Movements", "mouse_movements"), # Sum of mouse movements in window
                ("Clicks", "clicks"),               # Sum of clicks in window
                ("Key Presses", "key_presses"),     # Sum of key presses in window
                ("Key Releases", "key_releases"),   # Sum of key releases in window
                ("Scrolls", "scrolls")              # Sum of scrolls in window
            ]
            
            for feature_name, feature_key in action_features:
                # Create action feature row
                feature_row = [feature_name, f"action_{feature_key}", "Actions"]
                
                # Get current action tensors from controller
                try:
                    action_tensors = self.controller.get_action_features()
                    if len(action_tensors) >= 10:  # Should have 10 timesteps (T0-T9)
                        # Fill T0-T9 columns with aggregated values from each timestep's tensor
                        for t in range(10):
                            if t < len(action_tensors):
                                timestep_tensor = action_tensors[t]
                                if len(timestep_tensor) > 0:
                                    if feature_key == "count":
                                        # Action count is always first element
                                        value = timestep_tensor[0] if len(timestep_tensor) > 0 else 0
                                        feature_row.append(f"{int(value)}")
                                    else:
                                        # For other features, count occurrences in the tensor
                                        count = self._count_action_type_in_tensor(timestep_tensor, feature_key)
                                        feature_row.append(f"{count}")
                                else:
                                    feature_row.append("0")
                            else:
                                feature_row.append("0")
                    else:
                        feature_row.extend(["0"] * 10)
                except Exception as e:
                    feature_row.extend(["0"] * 10)
                
                # Insert action feature row
                self.sheet.insert_row(feature_row, idx=row_idx)
                
                # Store feature row info
                self.group_rows["Actions"]['feature_rows'].append(row_idx)
                
                row_idx += 1
    
    def _count_action_type_in_tensor(self, tensor: List[float], action_type: str) -> int:
        """Count occurrences of a specific action type in an action tensor"""
        if not tensor or len(tensor) < 1:
            return 0
        
        count = 0
        # Action tensor structure: [count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, timestamp2, type2, x2, y2, button2, key2, scroll_dx2, scroll_dy2, ...]
        # Action types: 0=move, 1=click, 2=key_press, 3=key_release, 4=scroll
        
        # Start from index 2 (first action type) and step by 8 (each action has 8 elements)
        for i in range(2, len(tensor), 8):
            if i < len(tensor):
                action_type_code = int(tensor[i])
                
                if action_type == "mouse_movements" and action_type_code == 0:
                    count += 1
                elif action_type == "clicks" and action_type_code == 1:
                    count += 1
                elif action_type == "key_presses" and action_type_code == 2:
                    count += 1
                elif action_type == "key_releases" and action_type_code == 3:
                    count += 1
                elif action_type == "scrolls" and action_type_code == 4:
                    count += 1
        
        return count
    
    def _insert_feature_row(self, feature_idx: int, name: str, group: str, translate_func, row_idx: int):
        """Insert a single feature row into the sheet"""
        # Create row data with proper column mapping
        values = [name, str(feature_idx), group]
        
        # Add blank values for T0..T9 (will be filled by update_table)
        values.extend([""] * 10)
        
        # Insert row
        self.sheet.insert_row(values, idx=row_idx)
        
        # Apply styling based on favorite status
        if feature_idx in self.favorite_features:
            # Highlight favorite rows
            self.sheet.highlight_cells(row=row_idx, column=0, bg="#2c5282", fg="#ffffff")
    
    def _load_favorites(self):
        """Load favorites from file"""
        try:
            import json
            from pathlib import Path
            
            favorites_file = Path("data/favorites.json")
            if favorites_file.exists():
                with open(favorites_file, 'r') as f:
                    self.favorite_features = set(json.load(f))
            else:
                self.favorite_features = set()
        except Exception as e:
            LOG.error(f"Failed to load favorites: {e}")
            self.favorite_features = set()
    
    def _save_favorites(self):
        """Save favorites to file"""
        try:
            import json
            from pathlib import Path
            
            favorites_file = Path("data/favorites.json")
            favorites_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(favorites_file, 'w') as f:
                json.dump(list(self.favorite_features), f)
        except Exception as e:
            LOG.error(f"Failed to save favorites: {e}")
    
    def toggle_favorite(self, feature_idx: int):
        """Toggle favorite status for a feature"""
        if feature_idx in self.favorite_features:
            self.favorite_features.remove(feature_idx)
        else:
            self.favorite_features.add(feature_idx)
        
        self._save_favorites()
        self._refresh_table()
        self._update_summary()
    
    def toggle_group(self, group_name: str):
        """Toggle expansion state for a group"""
        if group_name in self.expanded_groups:
            self.expanded_groups.remove(group_name)
        else:
            self.expanded_groups.add(group_name)
        
        self._refresh_table()
        self._update_summary()
    
    def _expand_all_groups(self):
        """Expand all feature groups"""
        if self.feature_groups:
            unique_groups = set(self.feature_groups)
            for group_name in unique_groups:
                self.expanded_groups.add(group_name)
            self._refresh_table()
            self._update_summary()
    
    def _collapse_all_groups(self):
        """Collapse all feature groups"""
        self.expanded_groups.clear()
        self._refresh_table()
        self._update_summary()
    
    def _update_summary(self):
        """Update the summary label"""
        if not self._schema_set:
            summary = "Features: 0/128 | Buffer: 0/10 | Status: Ready"
        else:
            features_count = 128
            buffer_count = 10
            status = "Active"
            favorites_count = len(self.favorite_features)
            expanded_groups_count = len(self.expanded_groups)
            summary = f"Features: {features_count}/128 | Buffer: {buffer_count}/10 | Status: {status} | Favorites: {favorites_count} | Groups: {expanded_groups_count} expanded"
        
        self.summary_label.config(text=summary)
    
    def _copy_to_clipboard(self):
        """Copy table data to clipboard"""
        if not self._schema_set:
            return
        
        try:
            # Get visible features
            visible_features = []
            for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
                if self.feature_group_filter != "All" and group != self.feature_group_filter:
                    continue
                if self.search_text and self.search_text not in name.lower():
                    continue
                visible_features.append((i, name, group))
            
            # Build CSV-like string
            lines = []
            header = ["Feature", "Index", "Group"] + [f"T{t}" for t in range(10)]  # T0 to T9
            lines.append("\t".join(header))
            
            for feature_idx, name, group in visible_features:
                row = [name, str(feature_idx), group]
                # Add blank values for time columns (data not available in this context)
                row.extend([""] * 10)
                lines.append("\t".join(row))
            
            clipboard_text = "\n".join(lines)
            self.clipboard_clear()
            self.clipboard_append(clipboard_text)
            
        except Exception as e:
            print(f"Failed to copy to clipboard: {e}")
    
    def _export_to_csv(self):
        """Export table data to CSV file"""
        if not self._schema_set:
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Get visible features
            visible_features = []
            for i, (name, group) in enumerate(zip(self.feature_names, self.feature_groups)):
                if self.feature_group_filter != "All" and group != self.feature_group_filter:
                    continue
                if self.search_text and self.search_text not in name.lower():
                    continue
                visible_features.append((i, name, group))
            
            # Write CSV
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                header = ["Feature", "Index", "Group"] + [f"T{t}" for t in range(10)]  # T0 to T9
                writer.writerow(header)
                
                # Data rows
                for feature_idx, name, group in visible_features:
                    row = [name, feature_idx, group]
                    # Add blank values for time columns (data not available in this context)
                    row.extend([""] * 10)
                    writer.writerow(row)
            
        except Exception as e:
            print(f"Failed to export CSV: {e}")
    
    def clear(self):
        """Clear all data from the view"""
        self.feature_names = None
        self.feature_groups = None
        self._color_bits = None
        self._schema_set = False
        self.sheet.clear()
        self._update_summary()

    def _start_live_mode(self):
        """Start live mode"""
        try:
            LOG.info("LiveFeaturesView: Starting live mode...")
            self.controller.start_live_mode()
            LOG.info("LiveFeaturesView: Live mode started successfully")
            self.summary_label.config(text="Features: 0/128 | Buffer: 0/10 | Status: Live Mode Active")
        except Exception as e:
            LOG.error(f"Failed to start live mode: {e}")
            self.summary_label.config(text="Features: 0/128 | Buffer: 0/10 | Status: Failed to Start")

    def _stop_live_mode(self):
        """Stop live mode"""
        try:
            self.controller.stop_live_mode()
        except Exception as e:
            LOG.error(f"Failed to stop live mode: {e}")

    def _clear_buffers(self):
        """Clear all buffers"""
        try:
            self.controller.clear_buffers()
            # Clear the view as well
            self.clear()
        except Exception as e:
            LOG.error(f"Failed to clear buffers: {e}")
