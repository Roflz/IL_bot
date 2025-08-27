#!/usr/bin/env python3
"""
Real-Time OSRS Bot with Safety & Validation
Production-Ready Imitation Learning Bot
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image

from .imitation_hybrid_model import ImitationHybridModel
from .imitation_loss import CombinedImitationLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions the bot can perform"""
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    SCROLL = "scroll"
    WAIT = "wait"

@dataclass
class BotAction:
    """Structured bot action"""
    action_type: ActionType
    x: Optional[float] = None
    y: Optional[float] = None
    key: Optional[str] = None
    duration: Optional[float] = None
    confidence: float = 0.0
    timestamp: float = 0.0

class ActionValidator:
    """Validates bot actions for safety"""
    
    def __init__(self, 
                 max_mouse_speed: float = 1000.0,  # pixels per second
                 max_click_frequency: float = 10.0,  # clicks per second
                 max_key_frequency: float = 20.0,   # keys per second
                 game_bounds: Tuple[int, int] = (800, 600)):
        
        self.max_mouse_speed = max_mouse_speed
        self.max_click_frequency = max_click_frequency
        self.max_key_frequency = max_key_frequency
        self.game_bounds = game_bounds
        
        # Action history for frequency checking
        self.action_history = []
        self.max_history_size = 100
        
        # Safety thresholds
        self.dangerous_areas = [
            (0, 0, 50, 50),      # Top-left corner (logout area)
            (750, 0, 800, 50),   # Top-right corner
            (0, 550, 50, 600),   # Bottom-left corner
            (750, 550, 800, 600) # Bottom-right corner
        ]
    
    def validate_action(self, action: BotAction, 
                       current_gamestate: np.ndarray,
                       previous_action: Optional[BotAction] = None) -> Tuple[bool, str]:
        """
        Validate a bot action for safety
        
        Args:
            action: Action to validate
            current_gamestate: Current game state
            previous_action: Previous action for context
            
        Returns:
            (is_valid, validation_message)
        """
        # 1. Basic action validation
        if not self._validate_basic_action(action):
            return False, "Invalid action structure"
        
        # 2. Boundary checks
        if not self._validate_boundaries(action):
            return False, "Action outside game boundaries"
        
        # 3. Frequency checks
        if not self._validate_frequency(action):
            return False, "Action frequency too high"
        
        # 4. Speed checks
        if not self._validate_speed(action, previous_action):
            return False, "Action speed too high"
        
        # 5. Dangerous area checks
        if not self._validate_dangerous_areas(action):
            return False, "Action in dangerous area"
        
        # 6. Game context validation
        if not self._validate_game_context(action, current_gamestate):
            return False, "Action inappropriate for game context"
        
        # 7. Temporal consistency
        if not self._validate_temporal_consistency(action, previous_action):
            return False, "Action lacks temporal consistency"
        
        # Action is valid
        self._update_action_history(action)
        return True, "Action validated successfully"
    
    def _validate_basic_action(self, action: BotAction) -> bool:
        """Validate basic action structure"""
        if action.action_type == ActionType.MOUSE_MOVE:
            return action.x is not None and action.y is not None
        elif action.action_type in [ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK]:
            return action.x is not None and action.y is not None
        elif action.action_type in [ActionType.KEY_PRESS, ActionType.KEY_RELEASE]:
            return action.key is not None
        elif action.action_type == ActionType.SCROLL:
            return action.duration is not None
        elif action.action_type == ActionType.WAIT:
            return action.duration is not None and action.duration > 0
        
        return False
    
    def _validate_boundaries(self, action: BotAction) -> bool:
        """Validate action is within game boundaries"""
        if action.x is not None:
            if not (0 <= action.x <= self.game_bounds[0]):
                return False
        
        if action.y is not None:
            if not (0 <= action.y <= self.game_bounds[1]):
                return False
        
        return True
    
    def _validate_frequency(self, action: BotAction) -> bool:
        """Validate action frequency is within limits"""
        current_time = time.time()
        
        # Filter recent actions
        recent_actions = [
            a for a in self.action_history 
            if current_time - a.timestamp < 1.0  # Last second
        ]
        
        # Check click frequency
        if action.action_type in [ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK]:
            recent_clicks = [a for a in recent_actions 
                           if a.action_type in [ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK]]
            if len(recent_clicks) >= self.max_click_frequency:
                return False
        
        # Check key frequency
        if action.action_type in [ActionType.KEY_PRESS, ActionType.KEY_RELEASE]:
            recent_keys = [a for a in recent_actions 
                          if a.action_type in [ActionType.KEY_PRESS, ActionType.KEY_RELEASE]]
            if len(recent_keys) >= self.max_key_frequency:
                return False
        
        return True
    
    def _validate_speed(self, action: BotAction, previous_action: Optional[BotAction]) -> bool:
        """Validate action speed is within limits"""
        if not previous_action or action.action_type != ActionType.MOUSE_MOVE:
            return True
        
        if previous_action.action_type != ActionType.MOUSE_MOVE:
            return True
        
        # Calculate mouse movement speed
        dx = action.x - previous_action.x
        dy = action.y - previous_action.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        time_diff = action.timestamp - previous_action.timestamp
        if time_diff <= 0:
            return False
        
        speed = distance / time_diff
        
        return speed <= self.max_mouse_speed
    
    def _validate_dangerous_areas(self, action: BotAction) -> bool:
        """Validate action is not in dangerous areas"""
        if action.x is None or action.y is None:
            return True
        
        for area in self.dangerous_areas:
            x1, y1, x2, y2 = area
            if x1 <= action.x <= x2 and y1 <= action.y <= y2:
                return False
        
        return True
    
    def _validate_game_context(self, action: BotAction, current_gamestate: np.ndarray) -> bool:
        """Validate action makes sense for current game context"""
        # This is a simplified version - in practice, you'd have more sophisticated
        # logic to determine what actions are valid in what contexts
        
        # Example: Check if bank is open and action is appropriate
        if len(current_gamestate) >= 40:  # Bank features start at index 39
            bank_open = current_gamestate[39] > 0  # Assuming bank_open is at index 39
            
            if bank_open and action.action_type == ActionType.LEFT_CLICK:
                # When bank is open, clicking should be on bank interface
                # This is a simplified check
                return True
        
        return True
    
    def _validate_temporal_consistency(self, action: BotAction, 
                                     previous_action: Optional[BotAction]) -> bool:
        """Validate temporal consistency of actions"""
        if not previous_action:
            return True
        
        # Check if action timestamp is reasonable
        time_diff = action.timestamp - previous_action.timestamp
        if time_diff < 0:  # Time going backwards
            return False
        
        # Check if action is too close in time (unrealistic)
        if time_diff < 0.01:  # Less than 10ms between actions
            return False
        
        return True
    
    def _update_action_history(self, action: BotAction):
        """Update action history"""
        self.action_history.append(action)
        
        # Keep only recent actions
        if len(self.action_history) > self.max_history_size:
            self.action_history.pop(0)

class FallbackActionGenerator:
    """Generates fallback actions when validation fails"""
    
    def __init__(self, game_bounds: Tuple[int, int] = (800, 600)):
        self.game_bounds = game_bounds
    
    def generate_fallback_action(self, gamestate: np.ndarray, 
                               failed_action: BotAction) -> BotAction:
        """Generate a safe fallback action"""
        
        # Rule-based fallback generation
        if self._should_bank(gamestate):
            return self._generate_banking_action(gamestate)
        elif self._should_craft(gamestate):
            return self._generate_crafting_action(gamestate)
        else:
            return self._generate_safe_wait_action()
    
    def generate_emergency_action(self, gamestate: np.ndarray) -> BotAction:
        """Generate emergency safe action"""
        return BotAction(
            action_type=ActionType.WAIT,
            duration=1.0,
            confidence=0.0,
            timestamp=time.time()
        )
    
    def _should_bank(self, gamestate: np.ndarray) -> bool:
        """Determine if bot should bank"""
        # Check if inventory is full
        if len(gamestate) >= 39:
            inventory_slots = gamestate[11:39]  # Inventory features
            empty_slots = np.sum(inventory_slots == 0)
            return empty_slots <= 2  # Bank when 2 or fewer empty slots
        
        return False
    
    def _should_craft(self, gamestate: np.ndarray) -> bool:
        """Determine if bot should craft"""
        # Check if we have materials and space
        if len(gamestate) >= 39:
            inventory_slots = gamestate[11:39]
            empty_slots = np.sum(inventory_slots == 0)
            return empty_slots > 2  # Craft when we have space
        
        return False
    
    def _generate_banking_action(self, gamestate: np.ndarray) -> BotAction:
        """Generate banking action"""
        # Find bank booth or bank interface
        # This is simplified - in practice you'd use more sophisticated logic
        
        # Safe banking coordinates (center of screen, away from dangerous areas)
        safe_x = self.game_bounds[0] // 2
        safe_y = self.game_bounds[1] // 2
        
        return BotAction(
            action_type=ActionType.LEFT_CLICK,
            x=safe_x,
            y=safe_y,
            confidence=0.8,
            timestamp=time.time()
        )
    
    def _generate_crafting_action(self, gamestate: np.ndarray) -> BotAction:
        """Generate crafting action"""
        # Safe crafting coordinates
        safe_x = self.game_bounds[0] // 2
        safe_y = self.game_bounds[1] // 2
        
        return BotAction(
            action_type=ActionType.LEFT_CLICK,
            x=safe_x,
            y=safe_y,
            confidence=0.8,
            timestamp=time.time()
        )
    
    def _generate_safe_wait_action(self) -> BotAction:
        """Generate safe wait action"""
        return BotAction(
            action_type=ActionType.WAIT,
            duration=0.5,
            confidence=1.0,
            timestamp=time.time()
        )

class RealTimeBot:
    """Production-ready real-time OSRS bot"""
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device = None,
                 safety_config: Dict = None):
        
        # Device setup
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize components
        self.action_validator = ActionValidator()
        self.fallback_generator = FallbackActionGenerator()
        
        # Bot state
        self.action_history = []
        self.gamestate_buffer = []
        self.last_action_time = time.time()
        self.is_running = False
        
        # Safety configuration
        self.safety_config = safety_config or {
            'max_retries': 3,
            'emergency_stop': True,
            'log_actions': True
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'total_actions': 0,
            'validated_actions': 0,
            'fallback_actions': 0,
            'emergency_actions': 0,
            'average_inference_time': 0.0
        }
        
        logger.info(f"Real-time bot initialized on device: {self.device}")
    
    def _load_model(self, model_path: str) -> ImitationHybridModel:
        """Load the trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = ImitationHybridModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def start(self):
        """Start the bot"""
        self.is_running = True
        logger.info("Bot started")
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
        logger.info("Bot stopped")
    
    def predict_next_action(self, 
                          current_gamestate: np.ndarray,
                          screenshot: np.ndarray,
                          temporal_context: Optional[np.ndarray] = None) -> BotAction:
        """
        Predict the next action using the model
        
        Args:
            current_gamestate: Current gamestate features
            screenshot: Current screenshot
            temporal_context: Optional temporal sequence
            
        Returns:
            Predicted bot action
        """
        if not self.is_running:
            return self.fallback_generator.generate_emergency_action(current_gamestate)
        
        try:
            # Update gamestate buffer
            self._update_gamestate_buffer(current_gamestate)
            
            # Prepare input tensors
            gamestate_features = self._prepare_gamestate_features(current_gamestate)
            screenshot_features = self._prepare_screenshot_features(screenshot)
            temporal_features = self._prepare_temporal_features(temporal_context)
            
            # Predict action
            start_time = time.time()
            with torch.no_grad():
                predictions = self.model(gamestate_features, screenshot_features, temporal_features)
            inference_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(inference_time)
            
            # Convert predictions to bot action
            action = self._convert_predictions_to_action(predictions)
            
            # Validate action
            is_valid, validation_msg = self.action_validator.validate_action(
                action, current_gamestate, 
                self.action_history[-1] if self.action_history else None
            )
            
            if is_valid:
                logger.debug(f"Action validated: {action.action_type}")
                self.performance_metrics['validated_actions'] += 1
            else:
                logger.warning(f"Action validation failed: {validation_msg}")
                
                # Generate fallback action
                action = self.fallback_generator.generate_fallback_action(
                    current_gamestate, action
                )
                self.performance_metrics['fallback_actions'] += 1
                
                # Validate fallback action
                is_valid, fallback_msg = self.action_validator.validate_action(
                    action, current_gamestate
                )
                
                if not is_valid:
                    logger.error(f"Fallback action also failed: {fallback_msg}")
                    action = self.fallback_generator.generate_emergency_action(current_gamestate)
                    self.performance_metrics['emergency_actions'] += 1
            
            # Store action
            self.action_history.append(action)
            self.performance_metrics['total_actions'] += 1
            
            return action
            
        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            return self.fallback_generator.generate_emergency_action(current_gamestate)
    
    def _update_gamestate_buffer(self, gamestate: np.ndarray):
        """Update gamestate buffer for temporal context"""
        self.gamestate_buffer.append(gamestate.copy())
        
        # Keep only last 10 gamestates
        if len(self.gamestate_buffer) > 10:
            self.gamestate_buffer.pop(0)
    
    def _prepare_gamestate_features(self, gamestate: np.ndarray) -> torch.Tensor:
        """Prepare gamestate features for model input"""
        features = torch.FloatTensor(gamestate).unsqueeze(0).to(self.device)
        return features
    
    def _prepare_screenshot_features(self, screenshot: np.ndarray) -> torch.Tensor:
        """Prepare screenshot for model input"""
        # Ensure correct format (3, 224, 224)
        if screenshot.shape[0] != 3:
            screenshot = np.transpose(screenshot, (2, 0, 1))
        
        # Resize if needed
        if screenshot.shape[1:] != (224, 224):
            screenshot = self._resize_screenshot(screenshot)
        
        # Normalize to [0, 1]
        if screenshot.max() > 1.0:
            screenshot = screenshot / 255.0
        
        features = torch.FloatTensor(screenshot).unsqueeze(0).to(self.device)
        return features
    
    def _prepare_temporal_features(self, temporal_context: Optional[np.ndarray]) -> torch.Tensor:
        """Prepare temporal context for model input"""
        if temporal_context is not None:
            features = torch.FloatTensor(temporal_context).unsqueeze(0).to(self.device)
        else:
            # Use gamestate buffer
            if len(self.gamestate_buffer) >= 10:
                buffer_array = np.array(self.gamestate_buffer[-10:])
                features = torch.FloatTensor(buffer_array).unsqueeze(0).to(self.device)
            else:
                # Pad with zeros if not enough history
                padding = np.zeros((10 - len(self.gamestate_buffer), 73))
                buffer_array = np.vstack([padding, self.gamestate_buffer])
                features = torch.FloatTensor(buffer_array).unsqueeze(0).to(self.device)
        
        return features
    
    def _resize_screenshot(self, screenshot: np.ndarray) -> np.ndarray:
        """Resize screenshot to 224x224"""
        # Convert to PIL Image for resizing
        if screenshot.shape[0] == 3:  # CHW format
            screenshot = np.transpose(screenshot, (1, 2, 0))
        
        pil_image = Image.fromarray((screenshot * 255).astype(np.uint8))
        resized = pil_image.resize((224, 224), Image.LANCZOS)
        
        # Convert back to numpy
        resized_array = np.array(resized).astype(np.float32) / 255.0
        
        # Convert back to CHW format
        if resized_array.shape[2] == 3:  # HWC format
            resized_array = np.transpose(resized_array, (2, 0, 1))
        
        return resized_array
    
    def _convert_predictions_to_action(self, predictions: Dict[str, torch.Tensor]) -> BotAction:
        """Convert model predictions to bot action"""
        # Get mouse position
        mouse_pos = predictions['mouse_position'][0].cpu().numpy()
        x, y = mouse_pos[0], mouse_pos[1]
        
        # Get click type
        click_probs = torch.sigmoid(predictions['mouse_click'][0]).cpu().numpy()
        left_click = click_probs[0] > 0.5
        right_click = click_probs[1] > 0.5
        
        # Get key press
        key_probs = torch.softmax(predictions['key_press'][0], dim=0).cpu().numpy()
        key_idx = np.argmax(key_probs)
        
        # Get confidence
        confidence = torch.sigmoid(predictions['confidence'][0]).cpu().numpy()[0]
        
        # Determine action type
        if left_click:
            action_type = ActionType.LEFT_CLICK
        elif right_click:
            action_type = ActionType.RIGHT_CLICK
        elif key_idx > 0:  # Assuming 0 is "no key"
            action_type = ActionType.KEY_PRESS
        else:
            action_type = ActionType.MOUSE_MOVE
        
        # Create action
        action = BotAction(
            action_type=action_type,
            x=x if action_type in [ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK, ActionType.MOUSE_MOVE] else None,
            y=y if action_type in [ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK, ActionType.MOUSE_MOVE] else None,
            key=str(key_idx) if action_type == ActionType.KEY_PRESS else None,
            confidence=confidence,
            timestamp=time.time()
        )
        
        return action
    
    def _update_performance_metrics(self, inference_time: float):
        """Update performance metrics"""
        # Update average inference time
        total_actions = self.performance_metrics['total_actions']
        current_avg = self.performance_metrics['average_inference_time']
        
        if total_actions == 0:
            self.performance_metrics['average_inference_time'] = inference_time
        else:
            self.performance_metrics['average_inference_time'] = (
                (current_avg * total_actions + inference_time) / (total_actions + 1)
            )
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_action_history(self, limit: int = 100) -> List[BotAction]:
        """Get recent action history"""
        return self.action_history[-limit:] if self.action_history else []
    
    def save_bot_state(self, filepath: str):
        """Save bot state to file"""
        state = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': self.performance_metrics,
            'action_history_count': len(self.action_history),
            'gamestate_buffer_size': len(self.gamestate_buffer),
            'is_running': self.is_running
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Bot state saved to {filepath}")
    
    def emergency_stop(self):
        """Emergency stop the bot"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.stop()
        
        # Generate safe emergency action
        emergency_action = self.fallback_generator.generate_emergency_action(np.zeros(73))
        self.action_history.append(emergency_action)
        
        logger.info("Bot stopped safely")

if __name__ == "__main__":
    print("Testing Real-Time Bot...")
    
    # Create dummy model for testing
    model = ImitationHybridModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create bot
    bot = RealTimeBot("dummy_model_path", device)
    
    print(f"✅ Real-time bot created successfully!")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test bot components
    print("\n🧪 Testing bot components...")
    
    # Test action validation
    test_action = BotAction(
        action_type=ActionType.LEFT_CLICK,
        x=400,
        y=300,
        confidence=0.9,
        timestamp=time.time()
    )
    
    is_valid, msg = bot.action_validator.validate_action(test_action, np.zeros(73))
    print(f"Action validation: {is_valid} - {msg}")
    
    # Test fallback generation
    fallback_action = bot.fallback_generator.generate_fallback_action(np.zeros(73), test_action)
    print(f"Fallback action: {fallback_action.action_type}")
    
    print(f"\n🤖 Real-time bot ready for deployment!")














