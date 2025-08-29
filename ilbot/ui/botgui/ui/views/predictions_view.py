#!/usr/bin/env python3
"""Predictions View - displays predicted action frames"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
from typing import Optional, List
from pathlib import Path
from ..widgets.tree_with_scrollbars import TreeWithScrollbars
from ...util.formatting import format_prediction_summary


class PredictionsView(ttk.Frame):
    """View for displaying model predictions"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.predictions: List[dict] = []
        self.action_encoder = None
        
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
        
        # Controls frame - Simplified and better organized
        controls_frame = ttk.Frame(self, style="Toolbar.TFrame")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        controls_frame.grid_columnconfigure(4, weight=1)
        
        # Row 1: Model selection and main actions
        # Model selection
        ttk.Label(controls_frame, text="Model:").grid(row=0, column=0, padx=(0, 6))
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(controls_frame, textvariable=self.model_var, 
                                          state="readonly", width=20)
        self.model_dropdown.grid(row=0, column=1, padx=(0, 6))
        
        self.refresh_models_btn = ttk.Button(controls_frame, text="🔄", width=3,
                                           command=self._refresh_models_list)
        self.refresh_models_btn.grid(row=0, column=2, padx=(0, 12))
        
        # Main action buttons
        ttk.Button(controls_frame, text="🤖 Run Prediction", 
                  command=self._run_prediction_on_sample).grid(row=0, column=3, padx=(0, 6))
        
        # Raw Output button
        self.raw_output_btn = ttk.Button(
            controls_frame, 
            text="🔍 Raw Output", 
            command=self._show_raw_prediction,
            style="Accent.TButton"
        )
        self.raw_output_btn.grid(row=0, column=4, padx=(0, 12))
        
        ttk.Button(controls_frame, text="Clear", 
                  command=self._clear_predictions).grid(row=0, column=5, padx=(0, 6))
        
        # Row 2: Data visualization buttons
        data_frame = ttk.Frame(controls_frame)
        data_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        
        ttk.Label(data_frame, text="View Data:").grid(row=0, column=0, padx=(0, 6))
        ttk.Button(data_frame, text="📊 Gamestate Sequence", 
                  command=self._load_gamestate_sample).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(data_frame, text="📊 Action Sequence", 
                  command=self._load_actions_sample).grid(row=0, column=2, padx=(0, 6))
        
        # Copy format selection
        ttk.Label(data_frame, text="Copy Format:").grid(row=0, column=3, padx=(12, 6))
        self.copy_format_var = tk.StringVar(value="tab")
        copy_format_combo = ttk.Combobox(data_frame, textvariable=self.copy_format_var, 
                                        values=["tab", "csv", "table"], state="readonly", width=8)
        copy_format_combo.grid(row=0, column=4, padx=(0, 6))
        
        ttk.Button(data_frame, text="📋 Copy Table", 
                  command=self._copy_table_to_clipboard).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(data_frame, text="Export CSV", 
                  command=self._export_to_csv).grid(row=0, column=6, padx=(0, 6))
        
        # Status line
        self.status_label = ttk.Label(self, text="Status: Ready | Predictions: 0", 
                                    font=("Arial", 9))
        self.status_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        
        # Summary statistics frame
        summary_frame = ttk.Frame(self, style="Toolbar.TFrame")
        summary_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 4))
        summary_frame.grid_columnconfigure(3, weight=1)
        
        # Summary labels
        self.total_predictions_label = ttk.Label(summary_frame, text="Predictions: 0", font=("Arial", 9))
        self.total_predictions_label.grid(row=0, column=0, padx=(0, 12))
        
        self.total_actions_label = ttk.Label(summary_frame, text="Total Actions: 0", font=("Arial", 9))
        self.total_actions_label.grid(row=0, column=1, padx=(0, 12))
        
        self.avg_actions_label = ttk.Label(summary_frame, text="Avg Actions: 0.0", font=("Arial", 9))
        self.avg_actions_label.grid(row=0, column=2, padx=(0, 12))
        
        self.action_types_label = ttk.Label(summary_frame, text="Types: Click:0, Scroll:0, Key:0", font=("Arial", 9))
        self.action_types_label.grid(row=0, column=3, padx=(0, 6))
        
        # Table
        columns = [
            ("index", "#", 50),
            ("type", "Type", 80),
            ("dt_ms", "Δt (ms)", 80),
            ("action_type", "Action Type", 80),
            ("x", "X", 60),
            ("y", "Y", 60),
            ("button", "Button", 80),
            ("key", "Key", 80),
            ("scroll_dx", "Scroll ΔX", 80),
            ("scroll_dy", "Scroll ΔY", 80)
        ]
        
        self.prediction_tree = TreeWithScrollbars(self, columns, height=15)
        self.prediction_tree.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
        
        # Set alternating colors
        self.prediction_tree.set_alternating_colors()
        
        # Initialize models list
        self._refresh_models_list()
        
        # Bind model selection change
        self.model_var.trace("w", self._on_model_selection_change)
        
        # Start periodic status updates
        self._schedule_status_update()
    
    def _schedule_status_update(self):
        """Schedule periodic status updates"""
        try:
            # Update status every 2 seconds
            self.check_model_status()
            self.after(2000, self._schedule_status_update)
        except Exception as e:
            print(f"Error in status update: {e}")
    
    def refresh_view(self):
        """Refresh the view and check for new models"""
        try:
            self._refresh_models_list()
            self.check_model_status()
        except Exception as e:
            print(f"Error refreshing view: {e}")
    
    def _bind_events(self):
        """Bind UI events"""
        pass
    
    def _refresh_models_list(self):
        """Refresh the list of available models from the models folder"""
        try:
            import os
            from pathlib import Path
            
            # Look for models in multiple possible locations
            possible_model_dirs = [
                "training_results",  # Current location
                "models",            # Common models folder
                "model",             # Alternative models folder
                "checkpoints",       # Training checkpoints
                "saved_models"       # Saved models folder
            ]
            
            models_found = []
            
            for model_dir in possible_model_dirs:
                if os.path.exists(model_dir):
                    # Look for .pth files
                    for file_path in Path(model_dir).glob("*.pth"):
                        if file_path.is_file():
                            # Create a display name (folder/filename)
                            display_name = f"{model_dir}/{file_path.name}"
                            models_found.append((display_name, str(file_path)))
            
            # If no models found in subdirectories, check current directory
            if not models_found:
                for file_path in Path(".").glob("*.pth"):
                    if file_path.is_file():
                        models_found.append((file_path.name, str(file_path)))
            
            # Update dropdown
            if models_found:
                # Sort by filename for consistent ordering
                models_found.sort(key=lambda x: x[0])
                
                # Update dropdown values
                self.model_dropdown['values'] = [model[0] for model in models_found]
                
                # Store the full paths for later use
                self.model_paths = {model[0]: model[1] for model in models_found}
                
                # Don't auto-select models to avoid startup errors
                # User can manually select a model when they want to use it
                print(f"Models available: {[m[0] for m in models_found]}")
                
                print(f"Found {len(models_found)} models: {[m[0] for m in models_found]}")
            else:
                self.model_dropdown['values'] = ["No models found"]
                self.model_var.set("No models found")
                self.model_paths = {}
                print("No model files found")
                
        except Exception as e:
            print(f"Error refreshing models list: {e}")
            self.model_dropdown['values'] = ["Error loading models"]
            self.model_var.set("Error loading models")
    
    def _on_model_selection_change(self, *args):
        """Handle model selection change in dropdown"""
        try:
            from pathlib import Path
            selected_model = self.model_var.get()
            
            if not selected_model or selected_model in ["No models found", "Error loading models"]:
                return
            
            if hasattr(self, 'model_paths') and selected_model in self.model_paths:
                model_path = Path(self.model_paths[selected_model])
                
                print(f"Loading selected model: {model_path}")
                
                # Load the model through the controller
                try:
                    success = self.controller.load_model(model_path)
                    
                    if success:
                        # Update status
                        self.status_label.config(text=f"Model Loaded: {Path(selected_model).name}")
                        print(f"Successfully loaded model: {selected_model}")
                    else:
                        # Reset selection on failure
                        self.model_var.set("")
                        self.status_label.config(text="Status: Failed to load model")
                        print(f"Failed to load model: {selected_model}")
                        
                        # Show informative error message
                        from tkinter import messagebox
                        messagebox.showwarning("Model Loading Failed", 
                            f"Could not load model: {Path(selected_model).name}\n\n"
                            f"This is normal if you haven't trained a model yet.\n"
                            f"Train a model first, then select it here for predictions.", 
                            parent=self)
                except Exception as e:
                    # Reset selection on error
                    self.model_var.set("")
                    self.status_label.config(text="Status: Model loading error")
                    print(f"Exception loading model {selected_model}: {e}")
                    
                    # Show informative error message
                    from tkinter import messagebox
                    messagebox.showinfo("No Model Available", 
                        f"Model file not found: {Path(selected_model).name}\n\n"
                        f"This is normal if you haven't trained a model yet.\n"
                        f"The GUI will work without a model - you can:\n"
                        f"• Record training data using the Recorder tab\n"
                        f"• View live features in the Live Features tab\n"
                        f"• Train a model, then come back here for predictions", 
                        parent=self)
            else:
                print(f"Model path not found for: {selected_model}")
                
        except Exception as e:
            print(f"Error in model selection change: {e}")
            import traceback
            traceback.print_exc()
    
    def check_model_status(self):
        """Check if a model is currently loaded and update UI accordingly"""
        try:
            from pathlib import Path
            if hasattr(self.controller, 'predictor_service') and self.controller.predictor_service.is_ready():
                # Model is loaded and ready
                if self.model_var.get() and self.model_var.get() not in ["No models found", "Error loading models"]:
                    # Update status to show model is ready
                    model_name = Path(self.model_var.get()).name
                    self.status_label.config(text=f"✅ Model Ready: {model_name} | Predictions: {len(self.predictions)}")
                else:
                    # Model is ready but no selection in dropdown
                    self.status_label.config(text=f"✅ Model Ready | Predictions: {len(self.predictions)}")
            else:
                # No model loaded
                if self.model_var.get() and self.model_var.get() not in ["No models found", "Error loading models"]:
                    # Selection exists but model not ready
                    self.status_label.config(text="⏳ Loading model... | Predictions: 0")
                else:
                    # No model selected
                    self.status_label.config(text="❌ No model loaded | Predictions: 0")
        except Exception as e:
            print(f"Error checking model status: {e}")
            self.status_label.config(text="❌ Error checking model | Predictions: 0")
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return (hasattr(self.controller, 'predictor_service') and 
                self.controller.predictor_service.is_ready())
    

    
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
                    f"{action['dt_ms']:.1f}",  # Δt
                    self._format_action_type(action['type']),  # Type
                    action['x'],  # X
                    action['y'],  # Y
                    self._format_button_type(action['button']),  # Button
                    self._format_key_value(action['key']),  # Key
                    f"{action['scroll_dx']:.1f}",  # Scroll ΔX
                    f"{action['scroll_dy']:.1f}"  # Scroll ΔY
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
        # Check model status first
        self.check_model_status()
        
        # Update predictions count in status if model is ready
        if self.is_model_ready():
            total_predictions = len(self.predictions)
            total_actions = sum(pred['count'] for pred in self.predictions)
            
            current_status = self.status_label.cget("text")
            if "Model Ready" in current_status:
                new_status = current_status.replace(f" | Predictions: {total_predictions}", f" | Predictions: {total_predictions} | Actions: {total_actions}")
                self.status_label.config(text=new_status)
        
        # Update summary statistics
        self._update_summary_stats()
    
    def _update_summary_stats(self, prediction: np.ndarray = None):
        """Update the summary statistics display"""
        try:
            if prediction is not None:
                # Update stats based on the current prediction
                if prediction.shape[0] > 0 and prediction.shape[1] == 8:
                    action_count = int(prediction[0, 0]) if prediction.shape[0] > 0 else 0
                    
                    # Count action types from the prediction
                    action_types = {"Click": 0, "Scroll": 0, "Key": 0}
                    for i in range(1, min(action_count + 1, prediction.shape[0])):
                        action_data = prediction[i]
                        action_type = int(action_data[1]) if len(action_data) > 1 else 0
                        if action_type == 1:
                            action_types["Click"] += 1
                        elif action_type == 2:
                            action_types["Scroll"] += 1
                        elif action_type == 3:
                            action_types["Key"] += 1
                    
                    # Update labels
                    self.total_predictions_label.config(text=f"Predictions: 1")
                    self.total_actions_label.config(text=f"Total Actions: {action_count}")
                    self.avg_actions_label.config(text=f"Avg Actions: {action_count:.1f}")
                    self.action_types_label.config(text=f"Types: Click:{action_types['Click']}, Scroll:{action_types['Scroll']}, Key:{action_types['Key']}")
                else:
                    # Invalid prediction shape
                    self.total_predictions_label.config(text="Predictions: 0")
                    self.total_actions_label.config(text="Total Actions: 0")
                    self.avg_actions_label.config(text="Avg Actions: 0.0")
                    self.action_types_label.config(text="Types: Click:0, Scroll:0, Key:0")
            else:
                # No prediction provided, show zeros
                self.total_predictions_label.config(text="Predictions: 0")
                self.total_actions_label.config(text="Total Actions: 0")
                self.avg_actions_label.config(text="Avg Actions: 0.0")
                self.action_types_label.config(text="Types: Click:0, Scroll:0, Key:0")
            
        except Exception as e:
            print(f"Error updating summary stats: {e}")
            # Set default values on error
            self.total_predictions_label.config(text="Predictions: 0")
            self.total_actions_label.config(text="Total Actions: 0")
            self.avg_actions_label.config(text="Avg Actions: 0.0")
            self.action_types_label.config(text="Types: Click:0, Scroll:0, Key:0")
    
    def _clear_predictions(self):
        """Clear all predictions"""
        self.predictions.clear()
        self.prediction_tree.clear()
        self._update_status()
    
    def _load_gamestate_sample(self):
        """Load and display the saved gamestate sequence from file"""
        try:
            import subprocess
            import os
            
            # Run the print_numpy_array.py script to show the visualization
            script_path = os.path.join(os.getcwd(), "print_numpy_array.py")
            file_path = os.path.join(os.getcwd(), "data", "sample_data", "normalized_gamestate_sequence.npy")
            
            subprocess.Popen(["py", script_path, file_path])
            print(f"DEBUG: Opened visualization for {file_path}")
            
        except Exception as e:
            print(f"EXCEPTION in _load_gamestate_sample: {e}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open gamestate visualization: {e}", parent=self)
    
    def _load_actions_sample(self):
        """Load and display the saved action sequence from file"""
        try:
            import subprocess
            import os
            
            # Run the print_numpy_array.py script to show the visualization
            script_path = os.path.join(os.getcwd(), "print_numpy_array.py")
            file_path = os.path.join(os.getcwd(), "data", "sample_data", "normalized_action_sequence.npy")
            
            subprocess.Popen(["py", script_path, file_path])
            print(f"DEBUG: Opened visualization for {file_path}")
            
        except Exception as e:
            print(f"EXCEPTION in _load_actions_sample: {e}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open action visualization: {e}", parent=self)

    def _run_prediction_on_sample(self):
        """Run prediction on the saved sample data"""
        try:
            # Check if model is loaded
            if not self.is_model_ready():
                from tkinter import messagebox
                messagebox.showwarning("No Model", "Please select and load a model from the dropdown first.", parent=self)
                return
            
            # Check if sample data exists
            import os
            from pathlib import Path
            
            sample_data_dir = Path("data/sample_data")
            gamestate_file = sample_data_dir / "normalized_gamestate_sequence.npy"
            action_file = sample_data_dir / "normalized_action_sequence.npy"
            
            if not gamestate_file.exists() or not action_file.exists():
                from tkinter import messagebox
                messagebox.showwarning("No Sample Data", 
                    "No sample data found. Run live mode first and then stop it to generate sample data.", parent=self)
                return
            
            # Run prediction
            print(f"DEBUG: Running prediction on sample data...")
            print(f"DEBUG: Gamestate file: {gamestate_file}")
            print(f"DEBUG: Action file: {action_file}")
            
            prediction = self.controller.predictor_service.predict_from_sample_data(
                str(gamestate_file), str(action_file)
            )
            
            if prediction is not None:
                print(f"DEBUG: Prediction successful! Shape: {prediction.shape}")
                
                # Parse and display the prediction
                self._display_prediction_result(prediction)
                
                # Show success message
                from tkinter import messagebox
                messagebox.showinfo("Prediction Success", 
                    f"Prediction completed successfully!\n\n"
                    f"Output shape: {prediction.shape}\n"
                    f"Action count: {int(prediction[0, 0])}\n"
                    f"Features per action: 8\n\n"
                    f"Results displayed in the predictions table below.", parent=self)
            else:
                from tkinter import messagebox
                messagebox.showerror("Prediction Failed", 
                    "Failed to generate prediction. Check the console for error details.", parent=self)
                
        except Exception as e:
            print(f"DEBUG: Exception in _run_prediction_on_sample: {e}")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to run prediction: {e}", parent=self)

    def _show_raw_prediction(self):
        """Show the raw prediction output using print_numpy_array.py tool"""
        import subprocess
        subprocess.Popen(["python", "print_numpy_array.py", "data/sample_data/raw_prediction_output.npy"])

    def _display_prediction_result(self, prediction: np.ndarray):
        """Display the prediction result in the table"""
        if prediction is None:
            messagebox.showerror("Error", "No prediction result to display")
            return
        
        # Store the last prediction for raw output access
        self.last_prediction = prediction.copy()
        
        # Save the raw prediction to sample data directory
        try:
            sample_data_dir = Path("data/sample_data")
            sample_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw prediction as numpy array
            prediction_file = sample_data_dir / "raw_prediction_output.npy"
            np.save(prediction_file, prediction)
            
            # Also save as text file for easy viewing
            text_file = sample_data_dir / "raw_prediction_output.txt"
            with open(text_file, 'w') as f:
                f.write(f"Raw Model Prediction Output\n")
                f.write(f"Shape: {prediction.shape}\n")
                f.write(f"Dtype: {prediction.dtype}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for i in range(prediction.shape[0]):
                    f.write(f"Row {i}: ")
                    for j in range(prediction.shape[1]):
                        f.write(f"{prediction[i, j]} ")
                    f.write("\n")
            
            print(f"✅ Raw prediction saved to:")
            print(f"   NumPy: {prediction_file}")
            print(f"   Text: {text_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save prediction: {e}")
        
        # Clear existing table
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)
        
        # Parse the prediction array
        # The model outputs (100, 8) or (101, 8) depending on training data
        # Row 0 should contain the action count, rows 1+ contain the actual actions
        if prediction.shape[0] > 0 and prediction.shape[1] == 8:
            action_count = int(prediction[0, 0]) if prediction.shape[0] > 0 else 0
            
            # Insert action count row
            self.prediction_tree.insert("", "end", values=(
                "0", "COUNT", "", "", "", "", "", "", "", ""
            ))
            
            # Insert actual actions - show ALL actions from the prediction
            for i in range(1, prediction.shape[0]):  # Show all rows, not just first 10
                action_data = prediction[i]
                self.prediction_tree.insert("", "end", values=(
                    str(i),
                    "ACTION",
                    f"{action_data[0]:.6f}",  # timing
                    f"{action_data[1]:.6f}",  # type
                    f"{action_data[2]:.6f}",  # x
                    f"{action_data[3]:.6f}",  # y
                    f"{action_data[4]:.6f}",  # button
                    f"{action_data[5]:.6f}",  # key
                    f"{action_data[6]:.6f}",  # scroll dx
                    f"{action_data[7]:.6f}"   # scroll dy
                ))
            
            # Update summary stats
            self._update_summary_stats(prediction)
        else:
            messagebox.showerror("Error", f"Unexpected prediction shape: {prediction.shape}")
    
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
    
    def _copy_table_to_clipboard(self):
        """Copy the current predictions table to clipboard"""
        if not self.predictions:
            from tkinter import messagebox
            messagebox.showinfo("Info", "No predictions to copy.", parent=self)
            return
        
        try:
            copy_format = self.copy_format_var.get()
            lines = []
            
            # Header
            header = ["Prediction", "Action", "Timestamp", "Count", "Δt (ms)", 
                     "Type", "X", "Y", "Button", "Key", "Scroll ΔX", "Scroll ΔY"]
            
            if copy_format == "tab":
                # Tab-separated format (good for Excel)
                lines.append("\t".join(header))
                separator = "\t"
            elif copy_format == "csv":
                # Comma-separated format
                lines.append(",".join(header))
                separator = ","
            else:  # table format
                # Formatted table with aligned columns
                lines.append(" | ".join(header))
                separator = " | "
            
            # Data rows
            for pred_idx, prediction in enumerate(self.predictions):
                for action_idx, action in enumerate(prediction['actions']):
                    row = [
                        str(pred_idx + 1),
                        str(action_idx + 1),
                        str(prediction['timestamp']),
                        str(prediction['count']),
                        f"{action['dt_ms']:.1f}",
                        str(action['type']),
                        str(action['x']),
                        str(action['y']),
                        str(action['button']),
                        str(action['key']),
                        f"{action['scroll_dx']:.1f}",
                        f"{action['scroll_dy']:.1f}"
                    ]
                    lines.append(separator.join(row))
            
            # Join all lines
            table_text = "\n".join(lines)
            
            # Copy to clipboard
            self.clipboard_clear()
            self.clipboard_append(table_text)
            
            # Show success message
            format_name = {"tab": "Tab-separated", "csv": "CSV", "table": "Formatted table"}[copy_format]
            from tkinter import messagebox
            messagebox.showinfo("Success", 
                f"Table copied to clipboard as {format_name}!\n\n"
                f"Copied {len(self.predictions)} predictions with "
                f"{sum(pred['count'] for pred in self.predictions)} total actions.\n\n"
                f"Format: {format_name}\n"
                f"You can now paste this data into Excel, Google Sheets, or any text editor.",
                parent=self)
            
            print(f"Table copied to clipboard as {format_name}: {len(lines)} rows")
            
        except Exception as e:
            print(f"Failed to copy table to clipboard: {e}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to copy table: {e}", parent=self)
    
    def set_action_encoder(self, action_encoder):
        """Set the action encoder for formatting"""
        self.action_encoder = action_encoder
    
    def clear(self):
        """Clear all data from the view"""
        self.predictions.clear()
        self.prediction_tree.clear()
        self._update_status()



    def _save_sample_input(self):
        """Save a sample of the current model input data"""
        try:
            import time
            print("DEBUG: ===== SAMPLE INPUT BUTTON CLICKED =====")
            
            # Get current feature window from the controller
            if not hasattr(self.controller, 'feature_pipeline') or self.controller.feature_pipeline.window is None:
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No feature data available. Start live mode first to collect data.", parent=self)
                return
            
            # Get current feature window (10, 128)
            feature_window = self.controller.feature_pipeline.window
            print(f"DEBUG: Feature window shape: {feature_window.shape}")
            print(f"DEBUG: Feature window dtype: {feature_window.dtype}")
            
            # Check if all timesteps are identical (stale data issue)
            if feature_window.shape[0] >= 2:
                print("DEBUG: ===== FEATURE WINDOW ANALYSIS =====")
                print(f"DEBUG: Feature window shape: {feature_window.shape}")
                print(f"DEBUG: Feature window dtype: {feature_window.dtype}")
                
                # Check each timestep's timestamp
                for i in range(feature_window.shape[0]):
                    timestamp = feature_window[i, 127]  # timestamp is at index 127
                    print(f"DEBUG: T{i} timestamp: {timestamp}")
                
                first_timestep = feature_window[0]
                last_timestep = feature_window[-1]
                
                if np.array_equal(first_timestep, last_timestep):
                    print("WARNING: First and last timesteps are identical - stale data detected!")
                else:
                    print("DEBUG: First and last timesteps are different")
                
                # Show sample values from first and last timesteps
                print(f"DEBUG: First timestep (T9) sample values: {first_timestep[:5]}...")
                print(f"DEBUG: Last timestep (T0) sample values: {last_timestep[:5]}...")
                
                # Check if ALL timesteps are identical
                all_identical = True
                for i in range(1, feature_window.shape[0]):
                    if not np.array_equal(feature_window[0], feature_window[i]):
                        all_identical = False
                        break
                
                if all_identical:
                    print("CRITICAL ERROR: ALL timesteps are identical - window is not shifting!")
                    print(f"DEBUG: All timesteps have timestamp: {feature_window[0, 127]}")
                else:
                    print("DEBUG: Timesteps are different - window shifting is working")
            
            # Get current action tensors (10 timesteps)
            print("DEBUG: Getting action features from controller...")
            action_tensors = self.controller.get_action_features()
            print(f"DEBUG: Action tensors count: {len(action_tensors) if action_tensors else 0}")
            
            if not action_tensors or len(action_tensors) < 10:
                from tkinter import messagebox
                messagebox.showwarning("No Data", "No action data available. Start live mode first to collect data.", parent=self)
                return
            
            # Debug action tensor structure
            print("DEBUG: ===== ACTION TENSOR ANALYSIS =====")
            print(f"DEBUG: Action tensors count: {len(action_tensors)}")
            
            for i, tensor in enumerate(action_tensors):
                print(f"DEBUG: T{i}: length={len(tensor)}")
                if len(tensor) > 0:
                    print(f"DEBUG: T{i}: first_few_values={tensor[:5] if len(tensor) >= 5 else tensor}")
                    if len(tensor) >= 1:
                        action_count = int(tensor[0]) if isinstance(tensor[0], (int, float)) else 0
                        print(f"DEBUG: T{i}: action_count={action_count}")
                        
                        # If this is an action tensor, show the structure
                        if len(tensor) >= 8:
                            print(f"DEBUG: T{i}: action_data=[count={tensor[0]}, Δt={tensor[1]}, type={tensor[2]}, x={tensor[3]}, y={tensor[4]}, btn={tensor[5]}, key={tensor[6]}, scroll_dx={tensor[7]}]")
                else:
                    print(f"DEBUG: T{i}: empty tensor")
            
            # Check if action tensors have meaningful data
            non_empty_tensors = [t for t in action_tensors if len(t) > 1]
            print(f"DEBUG: Non-empty action tensors: {len(non_empty_tensors)}/10")
            
            if non_empty_tensors:
                print(f"DEBUG: Sample non-empty tensor: {non_empty_tensors[0]}")
            else:
                print("WARNING: All action tensors are empty or have only count=0!")
            
            # Get gamestate timestamps for the action tensors
            current_timestamp = int(time.time() * 1000)
            gamestate_timestamps = [current_timestamp - (9-i)*100 for i in range(10)]
            print(f"DEBUG: Generated gamestate timestamps: {gamestate_timestamps[:3]}...")
            
            # Try to use shared pipeline methods
            try:
                print("DEBUG: Attempting to use shared pipeline conversion methods...")
                from ilbot.pipeline.shared_pipeline import (
                    convert_live_features_to_sequence_format,
                    convert_live_actions_to_raw_format,
                    create_live_training_sequences
                )
                
                # Convert live data to pipeline format
                print("DEBUG: Converting features to sequence format...")
                features = convert_live_features_to_sequence_format(feature_window)
                print(f"DEBUG: Converted features shape: {features.shape}")
                
                print("DEBUG: Converting actions to raw format...")
                raw_actions = convert_live_actions_to_raw_format(action_tensors, gamestate_timestamps)
                print(f"DEBUG: Raw actions count: {len(raw_actions)}")
                if raw_actions:
                    print(f"DEBUG: First raw action: {raw_actions[0]}")
                
                print("DEBUG: Creating training sequences...")
                input_sequences, target_sequences, action_input_sequences = create_live_training_sequences(
                    feature_window, action_tensors, gamestate_timestamps
                )
                print(f"DEBUG: Training sequences created successfully")
                print(f"DEBUG: Input sequences shape: {input_sequences.shape if input_sequences is not None else 'None'}")
                print(f"DEBUG: Target sequences count: {len(target_sequences) if target_sequences else 0}")
                print(f"DEBUG: Action input sequences count: {len(action_input_sequences) if action_input_sequences else 0}")
                
                # Prepare the sample input data
                sample_input = {
                    'temporal_sequence': features.tolist(),  # (10, 128) - ready for model input
                    'action_sequence': action_input_sequences,  # List of 10 action sequences
                    'raw_actions': raw_actions,  # Raw action data for debugging
                    'metadata': {
                        'timestamp': time.time(),
                        'feature_window_shape': feature_window.shape,
                        'action_tensors_count': len(action_tensors),
                        'action_tensor_lengths': [len(tensor) for tensor in action_tensors],
                        'input_sequences_shape': input_sequences.shape if input_sequences is not None else None,
                        'target_sequences_count': len(target_sequences) if target_sequences else 0,
                        'action_input_sequences_count': len(action_input_sequences) if action_input_sequences else 0
                    }
                }
                
            except Exception as conversion_error:
                # Fallback to basic data if conversion fails
                print(f"DEBUG: Conversion failed, using basic data: {conversion_error}")
                import traceback
                traceback.print_exc()
                
                sample_input = {
                    'temporal_sequence': feature_window.tolist(),  # (10, 128)
                    'action_sequence': action_tensors,  # List of 10 action tensors
                    'metadata': {
                        'timestamp': time.time(),
                        'feature_window_shape': feature_window.shape,
                        'action_tensors_count': len(action_tensors),
                        'action_tensor_lengths': [len(tensor) for tensor in action_tensors],
                        'conversion_error': str(conversion_error)
                    }
                }
            
            # Save to file
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfilename="sample_model_input.json"
            )
            
            if not filename:
                return
            
            import json
            with open(filename, 'w') as f:
                json.dump(sample_input, f, indent=2)
            
            print(f"DEBUG: Sample input saved to: {filename}")
            
            # Show success message
            from tkinter import messagebox
            messagebox.showinfo("Success", f"Sample model input saved to:\n{filename}\n\n"
                              f"Feature window: {feature_window.shape}\n"
                              f"Action tensors: {len(action_tensors)} timesteps\n"
                              f"Sample action tensor length: {len(action_tensors[0]) if action_tensors else 0}", 
                              parent=self)
            
        except Exception as e:
            print(f"DEBUG: Exception in _save_sample_input: {e}")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to save sample input: {e}", parent=self)
