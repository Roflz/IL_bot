#!/usr/bin/env python3
"""Prediction service for model inference"""

import time
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRunner:
    """Wrapper for the trained model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            # Import the model class
            from model.imitation_hybrid_model import ImitationHybridModel
            
            # Create model instance
            self.model = ImitationHybridModel(
                gamestate_dim=128,
                action_dim=8,
                sequence_length=10,
                hidden_dim=256,
                num_attention_heads=8
            )
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.loaded = False
    
    def predict(self, features: np.ndarray, actions: list) -> Optional[np.ndarray]:
        """
        Run prediction on the model.
        
        Args:
            features: Feature array of shape (10, 128)
            actions: List of action frames
            
        Returns:
            Predicted action frame or None if failed
        """
        if not self.loaded or self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Convert to tensors
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, 10, 128)
            
            # Process actions to match expected format
            # Convert list of action frames to tensor format expected by model
            action_tensors = []
            for action_frame in actions:
                if len(action_frame) > 1:  # Has actions
                    # Reshape to (1, 101, 8) format
                    count = int(action_frame[0])
                    if count > 0:
                        # Extract action data and pad to 100 actions
                        action_data = action_frame[1:1+count*8]
                        padded_actions = np.zeros(100 * 8)
                        padded_actions[:len(action_data)] = action_data
                        action_tensor = padded_actions.reshape(100, 8)
                    else:
                        action_tensor = np.zeros((100, 8))
                else:
                    action_tensor = np.zeros((100, 8))
                
                action_tensors.append(action_tensor)
            
            # Stack actions and add batch dimension
            actions_tensor = torch.FloatTensor(action_tensors).unsqueeze(0).to(self.device)  # (1, 10, 100, 8)
            
            # Run inference
            with torch.no_grad():
                prediction = self.model(features_tensor, actions_tensor)
            
            # Convert back to numpy
            prediction_np = prediction.cpu().numpy()[0]  # Remove batch dimension
            
            return prediction_np
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if the model is ready for predictions"""
        return self.loaded and self.model is not None


class PredictorService:
    """Service for managing model predictions"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_runner = None
        self.model_path = model_path
        self.predictions_enabled = False
        self.last_prediction_time = 0
        self.prediction_interval = 0.6  # 600ms
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> bool:
        """Load a model for predictions"""
        try:
            self.model_runner = ModelRunner(model_path)
            self.model_path = model_path
            return self.model_runner.is_ready()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray, actions: list) -> Optional[np.ndarray]:
        """
        Run prediction if enabled and enough time has passed.
        
        Args:
            features: Feature array of shape (10, 128)
            actions: List of action frames
            
        Returns:
            Predicted action frame or None
        """
        if not self.predictions_enabled:
            return None
        
        if not self.model_runner or not self.model_runner.is_ready():
            logger.warning("Model not ready for predictions")
            return None
        
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return None
        
        # Run prediction
        prediction = self.model_runner.predict(features, actions)
        
        if prediction is not None:
            self.last_prediction_time = current_time
        
        return prediction
    
    def enable_predictions(self, enabled: bool):
        """Enable or disable predictions"""
        self.predictions_enabled = enabled
        if enabled:
            logger.info("Predictions enabled")
        else:
            logger.info("Predictions disabled")
    
    def is_ready(self) -> bool:
        """Check if predictor is ready"""
        return self.model_runner is not None and self.model_runner.is_ready()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_runner:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded" if self.model_runner.is_ready() else "Model not ready",
            "path": str(self.model_path) if self.model_path else "None",
            "device": str(self.model_runner.device),
            "predictions_enabled": self.predictions_enabled
        }
