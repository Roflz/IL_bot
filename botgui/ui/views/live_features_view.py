#!/usr/bin/env python3
"""Live Features View - displays rolling 10x128 feature window using tksheet"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import logging
import threading
from typing import Optional, List
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
        
        # Initialize some groups as expanded by default
        self.expanded_groups = {"Player", "Inventory", "Skills"}  # Common groups to show initially
        
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
        controls_frame = ttk.Frame(self)
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
            LOG.debug("sheet click event: type=%s, value=%r", type(event).__name__, event)
            
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

            LOG.debug("sheet click: row=%r col=%r; group_rows keys=%s", row, col, list(self.group_rows.keys()))
            if row is None:
                LOG.debug("sheet click: no row info, event format not supported")
                return

            # Only toggle when the click is on a header row; we optionally restrict to column 0
            if row in self.group_rows and (col in (None, 0)):
                group_name = self.group_rows[row]
                LOG.info("group header clicked: row=%d group=%s (expanded=%s)", row, group_name,
                         group_name in self.expanded_groups)
                self.toggle_group(group_name)  # this calls _refresh_table and logs new state
                return "break"  # prevent tksheet changing selection focus further on this click
            else:
                LOG.debug("sheet click: not on header row (row=%r, col=%r, in_group_rows=%s)", 
                         row, col, row in self.group_rows)
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
        
        # Seed 128 rows (feature name + group) and blank t0..t9 cells
        self._realize_all_rows()
        
        # Initialize color bits
        self._color_bits = np.zeros((10, 128), dtype=bool)
        
        # Mark schema as set
        self._schema_set = True
        
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
            for t_idx in range(10):  # t0..t9 LEFT→RIGHT
                if not changed_mask[t_idx, f_idx]:
                    continue

                value = window[t_idx, f_idx]
                
                # write value (row=f_idx, col=3+t_idx)
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
                                group_hint = self.sheet.get_cell_data(f_idx, 2)
                            except Exception:
                                group_hint = None

                        mapped = self.controller.mapping_service.translate(f_idx, value, group_hint=group_hint)
                    except Exception:
                        mapped = None  # fail safe: fall back to raw value

                # Prefer mapped label when available; otherwise show the raw number
                text = mapped if (mapped is not None and mapped != "") else f"{value:.0f}"
                self.sheet.set_cell_data(f_idx, col, text)

                # flip color on each change
                self._color_bits[t_idx, f_idx] = ~self._color_bits[t_idx, f_idx]
                new_color = "#00b3b3" if self._color_bits[t_idx, f_idx] else "#ffffff"
                self.sheet.highlight_cells(row=f_idx, column=col, fg=new_color, redraw=False)

                updated += 1

        # Refresh the sheet after all updates
        self.sheet.refresh()
        
        # Store last window for next comparison
        self._last_window = window.copy()
        
        LOG.info("update_from_window: updated_cells=%d", updated)
    
    def _refresh_table(self):
        """Refresh the feature table with collapsible groups and favorites"""
        self._assert_main_thread("_refresh_table")
        try:
            LOG.info("LiveFeaturesView: _refresh_table called")
            
            if not self._schema_set:
                LOG.error("LiveFeaturesView: CRITICAL ERROR - schema not set in _refresh_table")
                return
            
            LOG.info(f"LiveFeaturesView: _refresh_table - feature_names count: {len(self.feature_names)}")
            LOG.info(f"LiveFeaturesView: _refresh_table - feature_groups count: {len(self.feature_groups)}")
            
            # Update group combo with unique groups
            if self.feature_groups:
                unique_groups = sorted(list(set(self.feature_groups)))
                current_groups = ["All"] + unique_groups
                self.group_combo['values'] = current_groups
                if self.feature_group_filter not in current_groups:
                    self.feature_group_filter = "All"
                    self.group_combo.set("All")
            
            # For now, just refresh the sheet to show current data
            if hasattr(self, 'sheet'):
                self.sheet.refresh()
            
            LOG.info("LiveFeaturesView: _refresh_table completed")
        
        except Exception as e:
            LOG.exception("LiveFeaturesView: _refresh_table failed with error")
            # Don't re-raise - just log and return gracefully
    
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
        LOG.info("EXPAND ALL GROUPS CLICKED - Before: expanded_groups=%s", self.expanded_groups)
        if self.feature_groups:
            unique_groups = set(self.feature_groups)
            LOG.info("EXPAND ALL GROUPS - Unique groups found: %s", unique_groups)
            for group_name in unique_groups:
                self.expanded_groups.add(group_name)
            LOG.info("EXPAND ALL GROUPS - After: expanded_groups=%s", self.expanded_groups)
            self._refresh_table()
            self._update_summary()
        else:
            LOG.error("EXPAND ALL GROUPS - No feature_groups available!")
    
    def _collapse_all_groups(self):
        """Collapse all feature groups"""
        LOG.info("COLLAPSE ALL GROUPS CLICKED - Before: expanded_groups=%s", self.expanded_groups)
        self.expanded_groups.clear()
        LOG.info("COLLAPSE ALL GROUPS - After: expanded_groups=%s", self.expanded_groups)
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
