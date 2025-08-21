#!/usr/bin/env python3
"""Predictions View - displays predicted action frames"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Optional, List
from ..widgets.tree_with_scrollbars import TreeWithScrollbars
from ...util.formatting import format_prediction_summary
from ..styles import create_dark_booleanvar


class PredictionsView(ttk.Frame):
    """View for displaying model predictions"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.predictions: List[dict] = []
        self.action_encoder = None
        
        # UI state
        self.predictions_enabled = True
        self.track_user_input = False
        
        self._setup_ui()
        self._bind_events()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Table gets most space
        
        # Header
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Model Predictions", 
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        
        # Controls frame
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        controls_frame.grid_columnconfigure(3, weight=1)
        
        # Left controls
        self.predictions_var = create_dark_booleanvar(self, value=True)
        ttk.Checkbutton(controls_frame, text="Run Predictions", 
                       variable=self.predictions_var).grid(row=0, column=0, padx=(0, 12))
        
        self.track_input_var = create_dark_booleanvar(self, value=False)
        ttk.Checkbutton(controls_frame, text="Track My Input", 
                       variable=self.track_input_var).grid(row=0, column=1, padx=(0, 12))
        
        # Center controls
        ttk.Button(controls_frame, text="📁 Load Model", 
                  command=self._load_model).grid(row=0, column=2, padx=(0, 12))
        
        # Right controls
        ttk.Button(controls_frame, text="Sample Gamestate Input Sequence", 
                  command=self._save_gamestate_sample).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(controls_frame, text="Sample Action Input Sequence", 
                  command=self._save_actions_sample).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(controls_frame, text="Clear", 
                  command=self._clear_predictions).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(controls_frame, text="Export CSV", 
                  command=self._export_to_csv).grid(row=0, column=6, padx=(0, 6))
        
        # Status line
        self.status_label = ttk.Label(self, text="Status: Ready | Predictions: 0", 
                                    font=("Arial", 9))
        self.status_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Table
        columns = [
            ("index", "#", 50),
            ("timestamp", "Time", 100),
            ("count", "Count", 60),
            ("dt_ms", "Δt (ms)", 80),
            ("type", "Type", 80),
            ("x", "X", 60),
            ("y", "Y", 60),
            ("button", "Button", 80),
            ("key", "Key", 80),
            ("scroll_dx", "Scroll ΔX", 80),
            ("scroll_dy", "Scroll ΔY", 80)
        ]
        
        self.prediction_tree = TreeWithScrollbars(self, columns, height=15)
        self.prediction_tree.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        
        # Set alternating colors
        self.prediction_tree.set_alternating_colors()
    
    def _bind_events(self):
        """Bind UI events"""
        self.predictions_var.trace("w", self._on_predictions_change)
        self.track_input_var.trace("w", self._on_track_input_change)
    
    def _on_predictions_change(self, *args):
        """Handle predictions toggle change"""
        self.predictions_enabled = self.predictions_var.get()
        if hasattr(self.controller, 'predictor_service'):
            self.controller.predictor_service.enable_predictions(self.predictions_enabled)
    
    def _on_track_input_change(self, *args):
        """Handle track input toggle change"""
        self.track_user_input = self.track_input_var.get()
        # TODO: Implement user input tracking
    
    def update_prediction(self, prediction: np.ndarray, timestamp: float):
        """Update the view with a new prediction"""
        if prediction is None or len(prediction) == 0:
            return
        
        try:
            # Parse prediction data
            count = int(prediction[0])
            if count == 0:
                return
            
            # Create prediction entry
            pred_entry = {
                'timestamp': timestamp,
                'count': count,
                'actions': []
            }
            
            # Parse action data
            for i in range(count):
                base_idx = 1 + i * 8
                if base_idx + 7 < len(prediction):
                    action = {
                        'dt_ms': prediction[base_idx],
                        'type': int(prediction[base_idx + 1]),
                        'x': int(prediction[base_idx + 2]),
                        'y': int(prediction[base_idx + 3]),
                        'button': int(prediction[base_idx + 4]),
                        'key': int(prediction[base_idx + 5]),
                        'scroll_dx': prediction[base_idx + 6],
                        'scroll_dy': prediction[base_idx + 7]
                    }
                    pred_entry['actions'].append(action)
            
            # Add to predictions list
            self.predictions.append(pred_entry)
            
            # Keep only last 100 predictions
            if len(self.predictions) > 100:
                self.predictions = self.predictions[-100:]
            
            # Update display
            self._refresh_table()
            self._update_status()
            
        except Exception as e:
            print(f"Failed to parse prediction: {e}")
    
    def _refresh_table(self):
        """Refresh the prediction table"""
        # Clear existing data
        self.prediction_tree.clear()
        
        # Populate table with predictions
        for pred_idx, prediction in enumerate(self.predictions):
            for action_idx, action in enumerate(prediction['actions']):
                # Create row values
                values = [
                    f"{pred_idx+1}.{action_idx+1}",  # Index
                    self._format_timestamp(prediction['timestamp']),  # Time
                    prediction['count'],  # Count
                    f"{action['dt_ms']:.1f}",  # Δt
                    self._format_action_type(action['type']),  # Type
                    action['x'],  # X
                    action['y'],  # Y
                    self._format_button_type(action['button']),  # Button
                    self._format_key_value(action['key']),  # Key
                    f"{action['scroll_dx']:.1f}",  # Scroll ΔX
                    f"{action['scroll_dy']:.1f}"   # Scroll ΔY
                ]
                
                # Insert row
                item = self.prediction_tree.insert("", "end", values=values)
                
                # Apply alternating colors
                tag = "evenrow" if (pred_idx + action_idx) % 2 == 0 else "oddrow"
                self.prediction_tree.item(item, tags=(tag,))
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        except Exception:
            return f"{timestamp:.1f}s"
    
    def _format_action_type(self, action_type: int) -> str:
        """Format action type for display"""
        if self.action_encoder:
            try:
                return self.action_encoder.get_action_type_name(action_type)
            except Exception:
                pass
        
        # Fallback formatting
        type_names = {1: "Click", 2: "Scroll", 3: "Key"}
        return type_names.get(action_type, f"Type {action_type}")
    
    def _format_button_type(self, button_type: int) -> str:
        """Format button type for display"""
        if self.action_encoder:
            try:
                return self.action_encoder.get_button_name(button_type)
            except Exception:
                pass
        
        # Fallback formatting
        button_names = {1: "Left", 2: "Right", 3: "Middle"}
        return button_names.get(button_type, f"Btn {button_type}")
    
    def _format_key_value(self, key_value: int) -> str:
        """Format key value for display"""
        if key_value == 0:
            return "None"
        
        if self.action_encoder:
            try:
                return self.action_encoder.get_key_name(key_value)
            except Exception:
                pass
        
        # Fallback formatting
        return f"Key {key_value}"
    
    def _update_status(self):
        """Update the status label"""
        total_predictions = len(self.predictions)
        total_actions = sum(pred['count'] for pred in self.predictions)
        
        status = f"Status: {'Active' if self.predictions_enabled else 'Paused'} | "
        status += f"Predictions: {total_predictions} | Actions: {total_actions}"
        
        self.status_label.config(text=status)
    
    def _clear_predictions(self):
        """Clear all predictions"""
        self.predictions.clear()
        self.prediction_tree.clear()
        self._update_status()
    
    def _save_gamestate_sample(self):
        """Save a sample of the current gamestate feature data as numpy array"""
        try:
            import logging
            LOG = logging.getLogger(__name__)
            
            print("DEBUG: _save_gamestate_sample: Starting...")
            LOG.info("_save_gamestate_sample: Starting gamestate sample save...")
            
            # Get current feature window from the controller
            print("DEBUG: Checking controller attributes...")
            if not hasattr(self.controller, 'feature_pipeline'):
                error_msg = "Controller has no feature_pipeline attribute"
                print(f"ERROR: {error_msg}")
                LOG.error(f"_save_gamestate_sample: {error_msg}")
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No feature pipeline available.", parent=self)
                return
                
            print("DEBUG: Checking feature pipeline window...")
            if self.controller.feature_pipeline.window is None:
                error_msg = "Feature pipeline window is None"
                print(f"ERROR: {error_msg}")
                LOG.error(f"_save_gamestate_sample: {error_msg}")
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No feature data available. Collect some data first by running live mode.", parent=self)
                return
            
            # Get current feature window (10, 128)
            print("DEBUG: Getting feature window...")
            feature_window = self.controller.feature_pipeline.window
            LOG.info(f"_save_gamestate_sample: Got feature window with shape: {feature_window.shape}")
            
            # Fix sequence order: Index 0 should be oldest (T-9), Index 9 should be newest (T0)
            print("DEBUG: Fixing sequence order...")
            import numpy as np
            feature_window = np.flipud(feature_window)  # Reverse the order
            print(f"DEBUG: Sequence order fixed: Index 0 = oldest, Index 9 = newest")
            
            # Use shared pipeline methods to properly process gamestate features
            print("DEBUG: Using shared pipeline methods to process gamestate features...")
            try:
                from shared_pipeline.normalize import normalize_features
                from shared_pipeline.feature_map import load_feature_mappings
                from shared_pipeline.features import FeatureExtractor
                
                # Load feature mappings for normalization
                feature_mappings = load_feature_mappings("data/features/feature_mappings.json")
                print(f"DEBUG: Loaded feature mappings for {len(feature_mappings)} features")
                
                # Normalize using the exact same method as the pipeline
                normalized_features = normalize_features(feature_window, "data/features/feature_mappings.json")
                print(f"DEBUG: Features normalized successfully")
                
                # Use normalized features for saving
                feature_window = normalized_features
                
            except Exception as norm_error:
                print(f"ERROR: Failed to process gamestate features using shared pipeline: {norm_error}")
                LOG.error(f"_save_gamestate_sample: Failed to process features: {norm_error}")
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to process gamestate features: {norm_error}", parent=self)
                return
            
            # Auto-save to sample_data directory
            import os
            sample_data_dir = "sample_data"
            os.makedirs(sample_data_dir, exist_ok=True)
            
            filename = os.path.join(sample_data_dir, "sample_gamestate_input_sequence.npy")
            print(f"DEBUG: Auto-saving to: {filename}")
            LOG.info(f"_save_gamestate_sample: Auto-saving to {filename}")
            
            import numpy as np
            print("DEBUG: About to call np.save...")
            np.save(filename, feature_window)
            print("DEBUG: np.save completed successfully")
            
            LOG.info(f"_save_gamestate_sample: Successfully saved normalized gamestate features to {filename}")
            
            # Auto-open visualization
            print("DEBUG: About to open visualization...")
            import subprocess
            try:
                subprocess.Popen(["py", ".\\print_numpy_array.py", filename])
                print("DEBUG: Visualization opened successfully")
                LOG.info(f"_save_gamestate_sample: Opened visualization for {filename}")
            except Exception as viz_error:
                print(f"DEBUG: Failed to open visualization: {viz_error}")
                LOG.warning(f"_save_gamestate_sample: Failed to open visualization: {viz_error}")
            
            # Show success message
            print("DEBUG: Showing success message...")
            from tkinter import messagebox
            messagebox.showinfo("Success", f"Gamestate features processed and saved to:\n{filename}\n\n"
                              f"Shape: {feature_window.shape}\n"
                              f"Data type: {feature_window.dtype}\n"
                              f"Order: Index 0 = oldest (T-9), Index 9 = newest (T0)\n"
                              f"Processing: Using shared_pipeline normalization and feature mapping\n\n"
                              f"Visualization opened automatically!", 
                              parent=self)
            print("DEBUG: _save_gamestate_sample: Completed successfully")
            
        except Exception as e:
            import logging
            import traceback
            print(f"EXCEPTION in _save_gamestate_sample: {e}")
            print(f"TRACEBACK: {traceback.format_exc()}")
            
            LOG = logging.getLogger(__name__)
            LOG.error(f"_save_gamestate_sample: Exception occurred: {e}")
            LOG.error(f"_save_gamestate_sample: Full traceback: {traceback.format_exc()}")
            
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to save gamestate sample: {e}", parent=self)
    
    def _save_actions_sample(self):
        """Save a sample of the current action sequence data as numpy array"""
        try:
            import logging
            import numpy as np
            import time
            LOG = logging.getLogger(__name__)
            
            LOG.info("_save_actions_sample: Starting actions sample save...")
            
            # Get synchronized action tensors from pipeline (last 10)
            print("DEBUG: _save_actions_sample: Reading synchronized action windows from pipeline")
            if not hasattr(self.controller, 'feature_pipeline'):
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No feature pipeline available.", parent=self)
                return
            action_tensors = self.controller.feature_pipeline.get_last_action_windows(10)
            print(f"DEBUG: _save_actions_sample: Raw action_tensors returned: {action_tensors}")
            print(f"DEBUG: _save_actions_sample: action_tensors type: {type(action_tensors)}")
            print(f"DEBUG: _save_actions_sample: action_tensors length: {len(action_tensors) if action_tensors else 0}")
            
            if action_tensors:
                for i, tensor in enumerate(action_tensors):
                    print(f"DEBUG: _save_actions_sample: Tensor {i}: {tensor}")
                    print(f"DEBUG: _save_actions_sample: Tensor {i} type: {type(tensor)}, length: {len(tensor) if tensor else 0}")
            
            LOG.info(f"_save_actions_sample: Got action tensors, count: {len(action_tensors) if action_tensors else 0}")
            
            if not action_tensors or len(action_tensors) < 10:
                error_msg = f"Insufficient action data: got {len(action_tensors) if action_tensors else 0} tensors, need 10"
                LOG.error(f"_save_actions_sample: {error_msg}")
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No action data available. Collect some data first by running live mode.", parent=self)
                return
            
            # Use shared pipeline methods to properly process actions
            print("DEBUG: Using shared pipeline methods to process actions...")
            try:
                from shared_pipeline.actions import convert_raw_actions_to_tensors
                from shared_pipeline.encodings import ActionEncoder
                
                # Create action encoder and convert to proper training format
                encoder = ActionEncoder()
                
                # Process each timestep to ensure proper (100, 8) format
                print(f"DEBUG: Processing {len(action_tensors)} action tensors...")
                
                # Process each timestep to ensure proper (100, 8) format
                processed_actions = []
                max_actions_per_timestep = 100
                
                for timestep_idx, action_tensor in enumerate(action_tensors):
                    print(f"DEBUG: Processing timestep {timestep_idx}")
                    
                    if not action_tensor or len(action_tensor) < 1:
                        # No actions in this timestep
                        timestep_actions = np.zeros((max_actions_per_timestep, 8))
                        processed_actions.append(timestep_actions)
                        continue
                    
                    # Create timestep array (100, 8)
                    timestep_actions = np.zeros((max_actions_per_timestep, 8))
                    
                    # Parse the flattened action tensor: [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
                    if len(action_tensor) >= 1:
                        action_count = int(action_tensor[0])
                        print(f"DEBUG: Timestep {timestep_idx} has {action_count} actions")
                        
                        if action_count > 0:
                            # Each action has 8 features
                            for action_idx in range(min(action_count, max_actions_per_timestep)):
                                start_idx = 1 + action_idx * 8
                                if start_idx + 7 < len(action_tensor):
                                    # Extract the 8 action features
                                    action_features = action_tensor[start_idx:start_idx + 8]
                                    timestep_actions[action_idx] = action_features
                                    print(f"DEBUG: Timestep {timestep_idx}, Action {action_idx}: {action_features}")
                    
                    processed_actions.append(timestep_actions)
                
                # Convert to numpy array (10, 100, 8)
                action_array = np.array(processed_actions)
                print(f"DEBUG: Created action array with shape: {action_array.shape}")
                
            except Exception as e:
                print(f"ERROR: Failed to process actions using shared pipeline: {e}")
                LOG.error(f"_save_actions_sample: Failed to process actions: {e}")
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to process actions: {e}", parent=self)
                return
            
            # Save as JSON (as you've been doing in your workflow)
            import os
            import json
            
            sample_data_dir = "sample_data"
            os.makedirs(sample_data_dir, exist_ok=True)
            
            filename = os.path.join(sample_data_dir, "sample_action_input_sequence.json")
            print(f"DEBUG: Auto-saving actions to: {filename}")
            LOG.info(f"_save_actions_sample: Auto-saving actions to {filename}")
            
            # Save as JSON with metadata
            action_data = {
                "action_sequence": action_tensors,
                "processed_actions": action_array.tolist(),
                "metadata": {
                    "timesteps": len(action_tensors),
                    "tensor_lengths": [len(tensor) for tensor in action_tensors],
                    "max_tensor_length": max(len(tensor) for tensor in action_tensors),
                    "processed_shape": action_array.shape,
                    "timestamp": time.time()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(action_data, f, indent=2)
            
            LOG.info(f"_save_actions_sample: Successfully saved action sequence to {filename}")
            
            # Save processed actions as numpy array for visualization
            viz_filename = os.path.join(sample_data_dir, "sample_action_input_sequence.npy")
            np.save(viz_filename, action_array)
            print(f"DEBUG: Saved processed actions to: {viz_filename}")
            
            # Auto-open visualization
            print("DEBUG: About to open visualization...")
            import subprocess
            try:
                subprocess.Popen(["py", ".\\print_numpy_array.py", viz_filename])
                print("DEBUG: Visualization opened successfully")
                LOG.info(f"_save_actions_sample: Opened visualization for {viz_filename}")
            except Exception as viz_error:
                print(f"DEBUG: Failed to open visualization: {viz_error}")
                LOG.warning(f"_save_actions_sample: Failed to open visualization: {viz_error}")
            
                        # Show success message
            from tkinter import messagebox
            messagebox.showinfo("Success", f"Action sequence processed and saved to:\n{filename}\n\n"
                               f"JSON format with {len(action_tensors)} timesteps\n"
                               f"Processed shape: {action_array.shape}\n"
                               f"Max actions per timestep: {max_actions_per_timestep}\n"
                               f"Action features: [count, timestamp, type, x, y, button, key, scroll_dx, scroll_dy]\n"
                               f"Processing: Using shared_pipeline ActionEncoder and action processing\n\n"
                               f"Visualization opened automatically!", 
                               parent=self)
            
        except Exception as e:
            import logging
            import traceback
            LOG = logging.getLogger(__name__)
            LOG.error(f"_save_actions_sample: Exception occurred: {e}")
            LOG.error(f"_save_actions_sample: Full traceback: {traceback.format_exc()}")
            
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to save actions sample: {e}", parent=self)
    
    def _export_to_csv(self):
        """Export predictions to CSV file"""
        if not self.predictions:
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
            
            # Write CSV
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                header = ["Prediction", "Action", "Timestamp", "Count", "Δt (ms)", 
                         "Type", "X", "Y", "Button", "Key", "Scroll ΔX", "Scroll ΔY"]
                writer.writerow(header)
                
                # Data rows
                for pred_idx, prediction in enumerate(self.predictions):
                    for action_idx, action in enumerate(prediction['actions']):
                        row = [
                            pred_idx + 1,
                            action_idx + 1,
                            prediction['timestamp'],
                            prediction['count'],
                            action['dt_ms'],
                            action['type'],
                            action['x'],
                            action['y'],
                            action['button'],
                            action['key'],
                            action['scroll_dx'],
                            action['scroll_dy']
                        ]
                        writer.writerow(row)
            
        except Exception as e:
            print(f"Failed to export CSV: {e}")
    
    def set_action_encoder(self, action_encoder):
        """Set the action encoder for formatting"""
        self.action_encoder = action_encoder
    
    def clear(self):
        """Clear all data from the view"""
        self.predictions.clear()
        self.prediction_tree.clear()
        self._update_status()

    def _load_model(self):
        """Load a trained model"""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                parent=self,
                title="Load Trained Model",
                filetypes=[
                    ("PyTorch models", "*.pth"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                from pathlib import Path
                model_path = Path(filename)
                success = self.controller.load_model(model_path)
                
                if success:
                    # Update status to show model loaded
                    self.status_label.config(text=f"Model Loaded: {model_path.name}")
                else:
                    from tkinter import messagebox
                    messagebox.showerror("Error", f"Failed to load model: {model_path}", parent=self)
                    
        except Exception as e:
            print(f"Error loading model: {e}")
