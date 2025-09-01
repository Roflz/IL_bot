#!/usr/bin/env python3
"""
Comprehensive Automated Bot System for OSRS
Connects trained model outputs to actual mouse and keyboard control
"""

import torch
import numpy as np
import time
import threading
import queue
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

# Bot control imports
try:
    import pyautogui
    import pynput
    from pynput import mouse, keyboard
    from pynput.mouse import Button, Listener as MouseListener
    from pynput.keyboard import Key, Listener as KeyboardListener
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("⚠️  Control libraries not available. Install with: pip install pyautogui pynput")

# Local imports
from ..model.imitation_hybrid_model import ImitationHybridModel
from ..analysis.human_behavior_analyzer import HumanBehaviorAnalyzer
from ..training.enhanced_behavioral_metrics import EnhancedBehavioralMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotAction:
    """Represents a single bot action"""
    action_type: str  # 'click', 'key', 'scroll', 'move'
    x: Optional[float] = None
    y: Optional[float] = None
    button: Optional[str] = None  # 'left', 'right', 'middle'
    key: Optional[str] = None
    key_action: Optional[str] = None  # 'press', 'release'
    scroll_amount: Optional[int] = None
    confidence: float = 0.0
    delay_before: float = 0.0
    delay_after: float = 0.0

@dataclass
class GameState:
    """Represents current game state for context"""
    player_x: float
    player_y: float
    inventory_state: str
    bank_open: bool
    nearby_objects: List[str]
    timestamp: float

class SafetyMonitor:
    """Monitors bot actions for safety and human intervention"""
    
    def __init__(self):
        self.emergency_stop = False
        self.last_human_input = time.time()
        self.suspicious_patterns = []
        self.action_history = []
        
        # Safety thresholds
        self.max_actions_per_minute = 120
        self.max_click_frequency = 10  # clicks per second
        self.suspicious_coordinate_jumps = 500  # pixels
        
    def check_safety(self, action: BotAction, game_state: GameState) -> bool:
        """Check if action is safe to execute"""
        if self.emergency_stop:
            return False
            
        # Check for human input (mouse/keyboard movement)
        if time.time() - self.last_human_input < 0.1:
            logger.warning("Human input detected - pausing bot")
            return False
            
        # Check action frequency
        current_time = time.time()
        recent_actions = [a for a in self.action_history if current_time - a < 60]
        if len(recent_actions) > self.max_actions_per_minute:
            logger.warning("Action frequency too high - pausing bot")
            return False
            
        # Check for suspicious patterns
        if self._detect_suspicious_pattern(action):
            logger.warning("Suspicious pattern detected - pausing bot")
            return False
            
        return True
        
    def _detect_suspicious_pattern(self, action: BotAction) -> bool:
        """Detect suspicious or bot-like patterns"""
        if len(self.action_history) < 2:
            return False
            
        # Check for repetitive patterns
        recent_actions = self.action_history[-10:]
        if len(recent_actions) >= 5:
            # Check for exact coordinate repetition
            coords = [(a.x, a.y) for a in recent_actions if a.x is not None and a.y is not None]
            if len(set(coords)) <= 2 and len(coords) >= 5:
                return True
                
        # Check for unrealistic timing
        if len(recent_actions) >= 2:
            last_action = recent_actions[-1]
            if hasattr(last_action, 'timestamp'):
                time_diff = time.time() - last_action.timestamp
                if time_diff < 0.05:  # Less than 50ms between actions
                    return True
                    
        return False
        
    def update_human_input(self):
        """Update timestamp of last human input"""
        self.last_human_input = time.time()
        
    def emergency_stop_bot(self):
        """Emergency stop the bot"""
        self.emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED")

class HumanBehaviorSimulator:
    """Simulates human-like behavior patterns"""
    
    def __init__(self, behavior_analyzer: HumanBehaviorAnalyzer):
        self.analyzer = behavior_analyzer
        self.mouse_patterns = {}
        self.click_patterns = {}
        self.keyboard_patterns = {}
        self.scroll_patterns = {}
        
    def add_human_realism(self, action: BotAction) -> BotAction:
        """Add human-like realism to bot actions"""
        if action.action_type == 'move':
            action = self._add_mouse_realism(action)
        elif action.action_type == 'click':
            action = self._add_click_realism(action)
        elif action.action_type == 'key':
            action = self._add_keyboard_realism(action)
        elif action.action_type == 'scroll':
            action = self._add_scroll_realism(action)
            
        return action
        
    def _add_mouse_realism(self, action: BotAction) -> BotAction:
        """Add realistic mouse movement patterns"""
        # Add slight coordinate jitter
        if action.x is not None and action.y is not None:
            jitter_x = np.random.normal(0, 2)  # 2 pixel jitter
            jitter_y = np.random.normal(0, 2)
            action.x += jitter_x
            action.y += jitter_y
            
        # Add realistic movement delays
        action.delay_before = np.random.uniform(0.05, 0.15)
        action.delay_after = np.random.uniform(0.02, 0.08)
        
        return action
        
    def _add_click_realism(self, action: BotAction) -> BotAction:
        """Add realistic click patterns"""
        # Vary click timing
        action.delay_before = np.random.uniform(0.1, 0.3)
        action.delay_after = np.random.uniform(0.05, 0.15)
        
        # Sometimes double-click (rarely)
        if np.random.random() < 0.05:
            action.delay_after += 0.1
            
        return action
        
    def _add_keyboard_realism(self, action: BotAction) -> BotAction:
        """Add realistic keyboard patterns"""
        # Vary key press timing
        action.delay_before = np.random.uniform(0.05, 0.2)
        action.delay_after = np.random.uniform(0.02, 0.1)
        
        return action
        
    def _add_scroll_realism(self, action: BotAction) -> BotAction:
        """Add realistic scroll patterns"""
        # Vary scroll amount slightly
        if action.scroll_amount is not None:
            variation = np.random.randint(-1, 2)
            action.scroll_amount += variation
            
        # Add realistic delays
        action.delay_before = np.random.uniform(0.1, 0.25)
        action.delay_after = np.random.uniform(0.05, 0.15)
        
        return action

class ActionExecutor:
    """Executes bot actions on the system"""
    
    def __init__(self, safety_monitor: SafetyMonitor):
        self.safety_monitor = safety_monitor
        self.is_running = False
        self.action_queue = queue.Queue()
        self.execution_thread = None
        
        # Initialize control systems
        if CONTROL_AVAILABLE:
            self.mouse_controller = mouse.Controller()
            self.keyboard_controller = keyboard.Controller()
            # Disable pyautogui failsafe
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.01
        else:
            self.mouse_controller = None
            self.keyboard_controller = None
            
    def start_execution(self):
        """Start the action execution thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        logger.info("Action executor started")
        
    def stop_execution(self):
        """Stop the action execution thread"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join()
        logger.info("Action executor stopped")
        
    def queue_action(self, action: BotAction):
        """Queue an action for execution"""
        self.action_queue.put(action)
        
    def _execution_loop(self):
        """Main execution loop"""
        while self.is_running:
            try:
                # Get next action (non-blocking)
                try:
                    action = self.action_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                    
                # Execute action
                self._execute_action(action)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(0.1)
                
    def _execute_action(self, action: BotAction):
        """Execute a single action"""
        try:
            # Safety check
            if not self.safety_monitor.check_safety(action, None):
                logger.warning(f"Action blocked by safety monitor: {action}")
                return
                
            # Pre-action delay
            if action.delay_before > 0:
                time.sleep(action.delay_before)
                
            # Execute action
            if action.action_type == 'move':
                self._execute_mouse_move(action)
            elif action.action_type == 'click':
                self._execute_click(action)
            elif action.action_type == 'key':
                self._execute_key_action(action)
            elif action.action_type == 'scroll':
                self._execute_scroll(action)
                
            # Post-action delay
            if action.delay_after > 0:
                time.sleep(action.delay_after)
                
            logger.debug(f"Executed action: {action.action_type}")
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            
    def _execute_mouse_move(self, action: BotAction):
        """Execute mouse movement"""
        if not CONTROL_AVAILABLE or action.x is None or action.y is None:
            return
            
        # Move mouse to target position
        self.mouse_controller.position = (int(action.x), int(action.y))
        
    def _execute_click(self, action: BotAction):
        """Execute mouse click"""
        if not CONTROL_AVAILABLE or action.x is None or action.y is None:
            return
            
        # Move to position first
        self.mouse_controller.position = (int(action.x), int(action.y))
        
        # Determine button
        button = Button.left
        if action.button == 'right':
            button = Button.right
        elif action.button == 'middle':
            button = Button.middle
            
        # Execute click
        if action.key_action == 'press':
            self.mouse_controller.press(button)
        elif action.key_action == 'release':
            self.mouse_controller.release(button)
        else:
            self.mouse_controller.click(button)
            
    def _execute_key_action(self, action: BotAction):
        """Execute keyboard action"""
        if not CONTROL_AVAILABLE or action.key is None:
            return
            
        # Convert key string to Key object
        key_obj = self._string_to_key(action.key)
        if key_obj is None:
            return
            
        # Execute key action
        if action.key_action == 'press':
            self.keyboard_controller.press(key_obj)
        elif action.key_action == 'release':
            self.keyboard_controller.release(key_obj)
        else:
            self.keyboard_controller.press(key_obj)
            self.keyboard_controller.release(key_obj)
            
    def _execute_scroll(self, action: BotAction):
        """Execute scroll action"""
        if not CONTROL_AVAILABLE or action.scroll_amount is None:
            return
            
        # Execute scroll
        self.mouse_controller.scroll(0, action.scroll_amount)
        
    def _string_to_key(self, key_str: str):
        """Convert string to Key object"""
        try:
            # Try to get Key attribute
            if hasattr(Key, key_str):
                return getattr(Key, key_str)
            # Try to get character key
            elif len(key_str) == 1:
                return key_str
            else:
                return None
        except:
            return None

class ModelPredictor:
    """Generates predictions using trained model"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.data_config = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        try:
            # Load model architecture
            self.model = ImitationHybridModel(
                data_config={'gamestate_dim': 128, 'max_actions': 100, 'action_features': 7, 'temporal_window': 10, 'enum_sizes': {}, 'event_types': 4},
                hidden_dim=256,
                num_heads=8,
                num_layers=6
            )
            
            # Load trained weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    def predict_actions(self, gamestate_sequence: torch.Tensor, action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate action predictions from model"""
        if self.model is None:
            logger.error("Model not loaded")
            return {}
            
        try:
            with torch.no_grad():
                # Prepare inputs
                gamestate_sequence = gamestate_sequence.to(self.device)
                action_sequence = action_sequence.to(self.device)
                
                # Generate predictions
                predictions = self.model(gamestate_sequence, action_sequence)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {}

class AutomatedBotSystem:
    """Main bot system that coordinates all components"""
    
    def __init__(self, 
                 model_path: str,
                 behavior_data_path: str,
                 safety_enabled: bool = True):
        self.model_path = model_path
        self.behavior_data_path = behavior_data_path
        self.safety_enabled = safety_enabled
        
        # Initialize components
        self.safety_monitor = SafetyMonitor()
        self.model_predictor = ModelPredictor(model_path)
        self.behavior_analyzer = HumanBehaviorAnalyzer()
        self.behavior_simulator = HumanBehaviorSimulator(self.behavior_analyzer)
        self.action_executor = ActionExecutor(self.safety_monitor)
        self.enhanced_metrics = EnhancedBehavioralMetrics()
        
        # Bot state
        self.is_running = False
        self.current_game_state = None
        self.action_history = []
        self.performance_metrics = {}
        
        # Load human behavior patterns
        self._load_behavior_patterns()
        
        # Setup input listeners
        self._setup_input_listeners()
        
    def _load_behavior_patterns(self):
        """Load human behavior patterns for realism"""
        try:
            if os.path.exists(self.behavior_data_path):
                self.behavior_analyzer.load_analysis_results(self.behavior_data_path)
                logger.info("Human behavior patterns loaded")
            else:
                logger.warning("Behavior data not found, using default patterns")
        except Exception as e:
            logger.error(f"Error loading behavior patterns: {e}")
            
    def _setup_input_listeners(self):
        """Setup listeners for human input detection"""
        if not CONTROL_AVAILABLE:
            return
            
        def on_mouse_move(x, y):
            self.safety_monitor.update_human_input()
            
        def on_mouse_click(x, y, button, pressed):
            self.safety_monitor.update_human_input()
            
        def on_key_press(key):
            self.safety_monitor.update_human_input()
            # Emergency stop on F12
            if hasattr(key, 'char') and key.char == 'f12':
                self.emergency_stop()
                
        # Start listeners
        self.mouse_listener = MouseListener(
            on_move=on_mouse_move,
            on_click=on_mouse_click
        )
        self.keyboard_listener = KeyboardListener(
            on_press=on_key_press
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
    def start_bot(self):
        """Start the automated bot system"""
        if self.is_running:
            logger.warning("Bot is already running")
            return
            
        self.is_running = True
        self.action_executor.start_execution()
        logger.info("Automated bot system started")
        
    def stop_bot(self):
        """Stop the automated bot system"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.action_executor.stop_execution()
        logger.info("Automated bot system stopped")
        
    def emergency_stop(self):
        """Emergency stop the bot"""
        self.safety_monitor.emergency_stop_bot()
        self.stop_bot()
        logger.critical("EMERGENCY STOP - Bot system halted")
        
    def update_game_state(self, game_state: GameState):
        """Update current game state"""
        self.current_game_state = game_state
        
    def generate_andExecuteActions(self, gamestate_sequence: torch.Tensor, action_sequence: torch.Tensor):
        """Generate actions from model and execute them"""
        if not self.is_running:
            return
            
        try:
            # Generate predictions
            predictions = self.model_predictor.predict_actions(gamestate_sequence, action_sequence)
            
            # Convert predictions to bot actions
            actions = self._convert_predictions_to_actions(predictions)
            
            # Add human realism
            realistic_actions = []
            for action in actions:
                realistic_action = self.behavior_simulator.add_human_realism(action)
                realistic_actions.append(realistic_action)
                
            # Queue actions for execution
            for action in realistic_actions:
                self.action_executor.queue_action(action)
                
            # Update metrics
            self._update_performance_metrics(actions, predictions)
            
            # Log actions
            logger.info(f"Generated {len(actions)} actions")
            
        except Exception as e:
            logger.error(f"Error generating/executing actions: {e}")
            
    def _convert_predictions_to_actions(self, predictions: Dict[str, torch.Tensor]) -> List[BotAction]:
        """Convert model predictions to executable bot actions"""
        actions = []
        
        try:
            # Extract event predictions
            if 'event_logits' in predictions:
                event_probs = torch.softmax(predictions['event_logits'], dim=-1)
                predicted_events = event_probs.argmax(dim=-1)
                
                # Extract coordinate predictions
                x_pred = predictions.get('x_mu', torch.zeros(1, 100))
                y_pred = predictions.get('y_mu', torch.zeros(1, 100))
                
                # Extract timing predictions
                time_pred = predictions.get('time_q', torch.zeros(1, 3))
                
                # Convert to actions
                for batch_idx in range(predicted_events.shape[0]):
                    for action_idx in range(predicted_events.shape[1]):
                        event_type = predicted_events[batch_idx, action_idx].item()
                        confidence = event_probs[batch_idx, action_idx, event_type].item()
                        
                        # Create action based on event type
                        if event_type == 0:  # CLICK
                            action = BotAction(
                                action_type='click',
                                x=x_pred[batch_idx, action_idx].item() if x_pred.shape[1] > action_idx else None,
                                y=y_pred[batch_idx, action_idx].item() if y_pred.shape[1] > action_idx else None,
                                button='left',
                                confidence=confidence
                            )
                        elif event_type == 1:  # KEY
                            action = BotAction(
                                action_type='key',
                                key='space',  # Default key
                                key_action='press',
                                confidence=confidence
                            )
                        elif event_type == 2:  # SCROLL
                            action = BotAction(
                                action_type='scroll',
                                scroll_amount=1,
                                confidence=confidence
                            )
                        elif event_type == 3:  # MOVE
                            action = BotAction(
                                action_type='move',
                                x=x_pred[batch_idx, action_idx].item() if x_pred.shape[1] > action_idx else None,
                                y=y_pred[batch_idx, action_idx].item() if y_pred.shape[1] > action_idx else None,
                                confidence=confidence
                            )
                        else:
                            continue
                            
                        actions.append(action)
                        
        except Exception as e:
            logger.error(f"Error converting predictions to actions: {e}")
            
        return actions
        
    def _update_performance_metrics(self, actions: List[BotAction], predictions: Dict[str, torch.Tensor]):
        """Update performance metrics"""
        if not actions:
            return
            
        # Calculate metrics
        total_actions = len(actions)
        avg_confidence = np.mean([a.confidence for a in actions])
        
        # Event distribution
        event_counts = {}
        for action in actions:
            event_type = action.action_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        # Update metrics
        self.performance_metrics = {
            'total_actions': total_actions,
            'average_confidence': avg_confidence,
            'event_distribution': event_counts,
            'timestamp': time.time()
        }
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance report"""
        return {
            'is_running': self.is_running,
            'safety_status': {
                'emergency_stop': self.safety_monitor.emergency_stop,
                'last_human_input': self.safety_monitor.last_human_input
            },
            'performance_metrics': self.performance_metrics,
            'action_history_count': len(self.action_history)
        }
        
    def save_bot_state(self, filepath: str):
        """Save current bot state to file"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'performance_metrics': self.performance_metrics,
                'safety_status': {
                    'emergency_stop': self.safety_monitor.emergency_stop,
                    'last_human_input': self.safety_monitor.last_human_input
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Bot state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")

def create_bot_system(config: Dict[str, Any]) -> AutomatedBotSystem:
    """Factory function to create bot system with configuration"""
    required_keys = ['model_path', 'behavior_data_path']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
            
    return AutomatedBotSystem(
        model_path=config['model_path'],
        behavior_data_path=config['behavior_data_path'],
        safety_enabled=config.get('safety_enabled', True)
    )

if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': 'checkpoints/best.pt',
        'behavior_data_path': 'human_behavior_analysis/',
        'safety_enabled': True
    }
    
    try:
        bot = create_bot_system(config)
        print("Bot system created successfully!")
        print("Use bot.start_bot() to begin automation")
        print("Use bot.emergency_stop() to stop immediately")
        
    except Exception as e:
        print(f"Error creating bot system: {e}")
