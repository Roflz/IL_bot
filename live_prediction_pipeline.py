#!/usr/bin/env python3
"""
Live Prediction Pipeline for Real-time Bot Operation

This script implements the live prediction system that:
1. Extracts features using EXACT SAME logic as training scripts
2. Normalizes features using the same normalization as training
3. Creates temporal sequences (10 gamestates → next actions)
4. Feeds data to the trained model for real-time predictions
5. Handles action data in the same format as training

This ensures complete consistency between training and live inference.
"""

import threading
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import torch

from live_feature_extractor import LiveFeatureExtractor

class LivePredictionPipeline:
    def __init__(self, bot_mode: str = "bot1", model_path: Optional[Path] = None):
        self.bot_mode = bot_mode
        
        # Model settings
        self.model_path = model_path or Path("training_results/model_weights.pth")
        self.model = None
        self.model_loaded = False
        
        # Feature extraction
        self.feature_extractor = LiveFeatureExtractor(bot_mode)
        
        # Data buffers for temporal sequences
        self.sequence_length = 10  # Must match training script
        self.feature_buffer = deque(maxlen=self.sequence_length)  # Stores 128-feature vectors
        self.action_buffer = deque(maxlen=self.sequence_length)   # Stores action frames (101, 8)
        
        # Prediction settings
        self.prediction_interval = 0.6  # 600ms - matches training script
        self.last_prediction_time = 0
        
        # Threading
        self.prediction_thread = None
        self.stop_event = threading.Event()
        self.prediction_active = False
        
        # Latest prediction results
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        
        # Load model if available
        self._load_model()
        
        print(f"[LIVE] Prediction pipeline initialized for {bot_mode}")
        print(f"[LIVE] Model loaded: {self.model_loaded}")
        print(f"[LIVE] Sequence length: {self.sequence_length}")
        print(f"[LIVE] Feature buffer size: {len(self.feature_buffer)}/{self.sequence_length}")
        print(f"[LIVE] Action buffer size: {len(self.action_buffer)}/{self.sequence_length}")

    def _load_model(self):
        """Load the trained model for inference."""
        try:
            if not self.model_path.exists():
                print(f"[LIVE] Warning: Model file not found: {self.model_path}")
                return
            
            # Load model weights
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract model architecture info
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create model instance (you'll need to import your model class)
            # For now, we'll use a placeholder
            print(f"[LIVE] Model checkpoint loaded: {len(state_dict)} parameters")
            self.model_loaded = True
            
        except Exception as e:
            print(f"[LIVE] Error loading model: {e}")
            self.model_loaded = False

    def start_prediction(self):
        """Start the live prediction loop."""
        if self.prediction_active:
            print("[LIVE] Prediction already active")
            return
        
        if not self.model_loaded:
            print("[LIVE] Cannot start prediction: model not loaded")
            return
        
        self.prediction_active = True
        self.stop_event.clear()
        
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()
        
        print("[LIVE] Prediction loop started")

    def stop_prediction(self):
        """Stop the live prediction loop."""
        self.prediction_active = False
        self.stop_event.set()
        
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=1.0)
            self.prediction_thread = None
        
        print("[LIVE] Prediction loop stopped")

    def _prediction_loop(self):
        """Main prediction loop that runs every 600ms."""
        print("[LIVE] Prediction loop started")
        
        while self.prediction_active and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time for a new prediction
                if current_time - self.last_prediction_time >= self.prediction_interval:
                    self._make_prediction()
                    self.last_prediction_time = current_time
                
                # Sleep for a short interval
                time.sleep(0.1)  # 100ms polling
                
            except Exception as e:
                print(f"[LIVE] Prediction loop error: {e}")
                time.sleep(1.0)
        
        print("[LIVE] Prediction loop ended")

    def _make_prediction(self):
        """Make a prediction using the current feature and action buffers."""
        try:
            # Check if we have enough data for prediction
            if len(self.feature_buffer) < self.sequence_length or len(self.action_buffer) < self.sequence_length:
                return
            
            # Extract the last 10 timesteps
            feature_sequence = np.array(list(self.feature_buffer)[-self.sequence_length:], dtype=np.float64)
            action_sequence = np.array(list(self.action_buffer)[-self.sequence_length:], dtype=np.float32)
            
            # Reshape for model input
            # Features: (10, 128) -> (1, 10, 128)
            feature_input = feature_sequence.reshape(1, self.sequence_length, -1)
            
            # Actions: (10, 101, 8) -> (1, 10, 101, 8)
            action_input = action_sequence.reshape(1, self.sequence_length, 101, 8)
            
            # Make prediction (placeholder for now)
            prediction = self._run_model_inference(feature_input, action_input)
            
            # Store prediction result
            with self.prediction_lock:
                self.latest_prediction = {
                    'timestamp': time.time(),
                    'prediction': prediction,
                    'input_features_shape': feature_input.shape,
                    'input_actions_shape': action_input.shape
                }
            
            print(f"[LIVE] Prediction made: {feature_input.shape} -> {prediction.shape if prediction is not None else 'None'}")
            
        except Exception as e:
            print(f"[LIVE] Prediction error: {e}")

    def _run_model_inference(self, feature_input: np.ndarray, action_input: np.ndarray) -> Optional[np.ndarray]:
        """Run model inference (placeholder implementation)."""
        # This is where you would call your actual model
        # For now, return a placeholder prediction
        
        if not self.model_loaded:
            return None
        
        try:
            # Convert to tensors
            feature_tensor = torch.FloatTensor(feature_input)
            action_tensor = torch.FloatTensor(action_input)
            
            # Run inference (placeholder)
            # prediction = self.model(feature_tensor, action_tensor)
            
            # For now, return a dummy prediction
            dummy_prediction = np.zeros((1, 101, 8), dtype=np.float32)
            dummy_prediction[0, 0, 0] = 1.0  # 1 action
            
            return dummy_prediction
            
        except Exception as e:
            print(f"[LIVE] Model inference error: {e}")
            return None

    def feed_gamestate(self, gamestate: Dict):
        """Feed a new gamestate to the prediction pipeline."""
        try:
            # Extract features using the exact same logic as training
            features = self.feature_extractor.extract_live_features(gamestate)
            
            if features is not None and len(features) == 128:
                # Add to feature buffer
                self.feature_buffer.append(features)
                
                # Build action step (placeholder for now)
                action_step = self.feature_extractor.build_action_step(gamestate)
                
                # Add to action buffer
                self.action_buffer.append(action_step)
                
                # Log buffer status
                if len(self.feature_buffer) % 10 == 0:
                    print(f"[LIVE] Buffers: features={len(self.feature_buffer)}/{self.sequence_length}, actions={len(self.action_buffer)}/{self.sequence_length}")
            
        except Exception as e:
            print(f"[LIVE] Error feeding gamestate: {e}")

    def get_latest_prediction(self) -> Optional[Dict]:
        """Get the latest prediction result."""
        with self.prediction_lock:
            return self.latest_prediction.copy() if self.latest_prediction else None

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            'feature_buffer_size': len(self.feature_buffer),
            'action_buffer_size': len(self.action_buffer),
            'sequence_length': self.sequence_length,
            'ready_for_prediction': len(self.feature_buffer) >= self.sequence_length and len(self.action_buffer) >= self.sequence_length,
            'prediction_active': self.prediction_active,
            'last_prediction_time': self.last_prediction_time,
            'model_loaded': self.model_loaded
        }

    def clear_buffers(self):
        """Clear all buffers."""
        self.feature_buffer.clear()
        self.action_buffer.clear()
        print("[LIVE] Buffers cleared")

    def get_feature_extractor(self) -> LiveFeatureExtractor:
        """Get the feature extractor instance."""
        return self.feature_extractor

    def is_ready_for_prediction(self) -> bool:
        """Check if the pipeline is ready to make predictions."""
        return (len(self.feature_buffer) >= self.sequence_length and 
                len(self.action_buffer) >= self.sequence_length and 
                self.model_loaded)

    def get_prediction_summary(self) -> str:
        """Get a human-readable summary of the current prediction state."""
        buffer_status = self.get_buffer_status()
        
        summary = f"Live Prediction Pipeline Status:\n"
        summary += f"  Model: {'✅ Loaded' if buffer_status['model_loaded'] else '❌ Not Loaded'}\n"
        summary += f"  Feature Buffer: {buffer_status['feature_buffer_size']}/{self.sequence_length}\n"
        summary += f"  Action Buffer: {buffer_status['action_buffer_size']}/{self.sequence_length}\n"
        summary += f"  Ready for Prediction: {'✅ Yes' if buffer_status['ready_for_prediction'] else '❌ No'}\n"
        summary += f"  Prediction Active: {'✅ Yes' if buffer_status['prediction_active'] else '❌ No'}\n"
        
        if self.latest_prediction:
            summary += f"  Last Prediction: {time.strftime('%H:%M:%S', time.localtime(self.latest_prediction['timestamp']))}\n"
        else:
            summary += f"  Last Prediction: Never\n"
        
        return summary

def main():
    """Test the live prediction pipeline."""
    print("Testing Live Prediction Pipeline...")
    
    # Create pipeline
    pipeline = LivePredictionPipeline("bot1")
    
    # Print status
    print(pipeline.get_prediction_summary())
    
    # Test with dummy gamestate
    dummy_gamestate = {
        'timestamp': int(time.time() * 1000),
        'player': {
            'world_x': 100,
            'world_y': 200,
            'animation_id': 899,
            'is_moving': False,
            'movement_direction': 'stationary'
        },
        'camera_x': 0,
        'camera_y': 0,
        'camera_z': 0,
        'camera_pitch': 0,
        'camera_yaw': 0,
        'inventory': [],
        'npcs': [],
        'game_objects': [],
        'furnaces': [],
        'tabs': {'currentTab': 0},
        'skills': {'crafting': {'level': 1, 'xp': 0}},
        'phase_context': {
            'cycle_phase': 'unknown',
            'phase_start_time': 0,
            'phase_duration_ms': 0,
            'gamestates_in_phase': 0
        },
        'bank_open': False,
        'bank_item_positions': {},
        'last_interaction': {
            'action': '',
            'item_name': '',
            'target': '',
            'timestamp': 0
        }
    }
    
    # Feed gamestate multiple times to fill buffer
    for i in range(15):
        pipeline.feed_gamestate(dummy_gamestate)
        time.sleep(0.1)
    
    # Print updated status
    print("\nAfter feeding gamestates:")
    print(pipeline.get_prediction_summary())
    
    # Test prediction
    if pipeline.is_ready_for_prediction():
        print("\nStarting prediction...")
        pipeline.start_prediction()
        time.sleep(2.0)
        pipeline.stop_prediction()
        
        # Get latest prediction
        prediction = pipeline.get_latest_prediction()
        if prediction:
            print(f"Latest prediction: {prediction}")
    else:
        print("\nNot ready for prediction")

if __name__ == "__main__":
    main()
