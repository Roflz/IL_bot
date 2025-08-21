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
        ttk.Button(controls_frame, text="Clear", 
                  command=self._clear_predictions).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(controls_frame, text="Export CSV", 
                  command=self._export_to_csv).grid(row=0, column=4, padx=(0, 6))
        
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
