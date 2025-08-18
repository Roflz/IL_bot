#!/usr/bin/env python3
"""
Basic OSRS Bot for Testing Trained Model
"""

import torch
import numpy as np
import time
import cv2
import pyautogui
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from model_architecture import OSRSImitationModel


class BasicOSRSBot:
    """
    Basic bot to test the trained imitation learning model.
    """
    
    def __init__(
        self,
        model_path: str = "best_model.pth",
        screenshot_region: Tuple[int, int, int, int] = None,
        device: str = None
    ):
        self.model_path = model_path
        self.screenshot_region = screenshot_region  # (x, y, width, height)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = self._load_model()
        
        # Bot state
        self.is_running = False
        self.action_history = []
        
        # Feature extraction (simplified)
        self.feature_extractor = SimplifiedFeatureExtractor()
        
        print(f"🤖 Basic OSRS Bot initialized")
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Screenshot region: {screenshot_region}")
    
    def _load_model(self) -> OSRSImitationModel:
        """Load the trained model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        model = OSRSImitationModel(
            input_features=128,
            sequence_length=10,
            hidden_size=256,
            num_layers=2,
            dropout=0.2
        )
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
    
    def take_screenshot(self) -> np.ndarray:
        """Take a screenshot of the OSRS window."""
        if self.screenshot_region:
            x, y, width, height = self.screenshot_region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
        else:
            # Take full screen screenshot
            screenshot = pyautogui.screenshot()
        
        # Convert to numpy array
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        return screenshot
    
    def extract_features(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Extract basic features from screenshot (simplified version).
        
        Args:
            screenshot: Screenshot as numpy array
            
        Returns:
            128-dimensional feature vector
        """
        return self.feature_extractor.extract_features(screenshot)
    
    def predict_action(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Use the trained model to predict the next action.
        
        Args:
            features: 128-dimensional feature vector
            
        Returns:
            Dictionary of predicted actions
        """
        # Create a sequence (repeat the same features 10 times for now)
        sequence = np.tile(features, (10, 1))  # (10, 128)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probabilities = self.model.get_action_probabilities(prediction)
        
        # Convert to numpy
        actions = {}
        for key, value in prediction.items():
            actions[key] = value.cpu().numpy()[0]
        
        return actions
    
    def execute_action(self, action: Dict[str, np.ndarray]) -> bool:
        """
        Execute the predicted action.
        
        Args:
            action: Dictionary of predicted actions
            
        Returns:
            True if action executed successfully
        """
        try:
            # Extract action components
            mouse_pos = action.get('mouse_position', [400, 300])
            mouse_click = action.get('mouse_click', [1, 0, 0])
            key_press = action.get('key_press', [0] * 50)
            scroll = action.get('scroll', [0, 0])
            
            # Execute mouse movement
            if 'mouse_position' in action:
                x, y = int(mouse_pos[0]), int(mouse_pos[1])
                print(f"🖱️  Moving mouse to ({x}, {y})")
                pyautogui.moveTo(x, y, duration=0.1)
            
            # Execute click
            if 'mouse_click' in action:
                click_type = np.argmax(mouse_click)
                if click_type == 1:  # Left click
                    print(f"🖱️  Left clicking at ({x}, {y})")
                    pyautogui.click(x, y, button='left')
                elif click_type == 2:  # Right click
                    print(f"🖱️  Right clicking at ({x}, {y})")
                    pyautogui.click(x, y, button='right')
            
            # Execute key press
            if 'key_press' in action:
                key_idx = np.argmax(key_press)
                if key_idx > 0:  # Skip "no key" (index 0)
                    # Map key index to actual key (simplified)
                    key = self._map_key_index(key_idx)
                    print(f"⌨️  Pressing key: {key}")
                    pyautogui.press(key)
            
            # Execute scroll
            if 'scroll' in action:
                dx, dy = scroll[0], scroll[1]
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    print(f"🖱️  Scrolling by ({dx:.1f}, {dy:.1f})")
                    pyautogui.scroll(int(dy * 3))  # Scale scroll amount
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing action: {e}")
            return False
    
    def _map_key_index(self, key_idx: int) -> str:
        """Map key index to actual key (simplified mapping)."""
        # Basic key mapping - you can expand this
        key_mapping = {
            1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e',
            6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
            11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
            16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
            21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
            26: 'z', 27: 'space', 28: 'enter', 29: 'escape',
            30: 'tab', 31: 'shift', 32: 'ctrl', 33: 'alt'
        }
        return key_mapping.get(key_idx, 'a')
    
    def run_bot(self, duration: int = 30, action_interval: float = 2.0):
        """
        Run the bot for a specified duration.
        
        Args:
            duration: How long to run the bot (seconds)
            action_interval: Time between actions (seconds)
        """
        print(f"🚀 Starting bot for {duration} seconds...")
        print(f"Action interval: {action_interval} seconds")
        print(f"Press Ctrl+C to stop early")
        
        self.is_running = True
        start_time = time.time()
        action_count = 0
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Take screenshot
                print(f"\n📸 Taking screenshot...")
                screenshot = self.take_screenshot()
                
                # Extract features
                print(f"🔍 Extracting features...")
                features = self.extract_features(screenshot)
                
                # Predict action
                print(f"🧠 Predicting action...")
                action = self.predict_action(features)
                
                # Execute action
                print(f"⚡ Executing action...")
                success = self.execute_action(action)
                
                # Record action
                action_record = {
                    'timestamp': time.time(),
                    'action': action,
                    'success': success,
                    'features': features
                }
                self.action_history.append(action_record)
                
                action_count += 1
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                print(f"✅ Action {action_count} completed. {remaining:.1f}s remaining")
                
                # Wait before next action
                if remaining > action_interval:
                    time.sleep(action_interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
        finally:
            self.is_running = False
            self._print_summary(action_count, time.time() - start_time)
    
    def _print_summary(self, action_count: int, total_time: float):
        """Print bot run summary."""
        print(f"\n📊 BOT RUN SUMMARY")
        print(f"=" * 40)
        print(f"Total actions: {action_count}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Actions per second: {action_count/total_time:.2f}")
        print(f"Average time per action: {total_time/action_count:.2f} seconds")
        
        # Save action history
        history_file = f"bot_run_{int(time.time())}.json"
        with open(history_file, 'w') as f:
            json.dump(self.action_history, f, indent=2, default=str)
        print(f"Action history saved to: {history_file}")
    
    def test_single_prediction(self):
        """Test a single prediction cycle."""
        print(f"🧪 Testing single prediction cycle...")
        
        # Take screenshot
        screenshot = self.take_screenshot()
        
        # Extract features
        features = self.extract_features(screenshot)
        print(f"Features extracted: {features.shape}")
        print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
        
        # Predict action
        action = self.predict_action(features)
        
        # Display prediction
        print(f"\n🎯 PREDICTED ACTIONS:")
        for key, value in action.items():
            if key == 'mouse_position':
                print(f"  Mouse position: ({value[0]:.1f}, {value[1]:.1f})")
            elif key == 'mouse_click':
                click_type = ['None', 'Left', 'Right'][np.argmax(value)]
                confidence = np.max(value)
                print(f"  Click type: {click_type} (confidence: {confidence:.3f})")
            elif key == 'key_press':
                key_idx = np.argmax(value)
                confidence = np.max(value)
                print(f"  Key press: {key_idx} (confidence: {confidence:.3f})")
            elif key == 'scroll':
                print(f"  Scroll: ({value[0]:.1f}, {value[1]:.1f})")
            elif key == 'confidence':
                print(f"  Confidence: {value[0]:.3f}")
        
        return action


class SimplifiedFeatureExtractor:
    """
    Simplified feature extractor for testing.
    Generates 128 features from screenshot (placeholder implementation).
    """
    
    def __init__(self):
        self.feature_count = 128
    
    def extract_features(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Extract 128 features from screenshot.
        
        Args:
            screenshot: Screenshot as numpy array
            
        Returns:
            128-dimensional feature vector
        """
        # Convert to grayscale for basic processing
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Extract basic features
        features = []
        
        # 1. Basic image statistics (16 features)
        features.extend([
            gray.mean(), gray.std(), gray.min(), gray.max(),
            np.percentile(gray, 25), np.percentile(gray, 50), np.percentile(gray, 75),
            gray.shape[0], gray.shape[1], gray.size,
            np.sum(gray > 128), np.sum(gray < 64), np.sum(gray > 192),
            np.var(gray), np.median(gray), np.mean(np.abs(np.diff(gray)))
        ])
        
        # 2. Edge detection features (16 features)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            edges.mean(), edges.std(), edges.sum(), np.sum(edges > 0),
            np.sum(edges > 50), np.sum(edges > 100), np.sum(edges > 150),
            np.sum(edges > 200), np.sum(edges > 250),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
            cv2.Sobel(gray, cv2.CV_64F, 1, 0).var(),
            cv2.Sobel(gray, cv2.CV_64F, 0, 1).var(),
            cv2.Sobel(gray, cv2.CV_64F, 2, 0).var(),
            cv2.Sobel(gray, cv2.CV_64F, 0, 2).var(),
            cv2.Sobel(gray, cv2.CV_64F, 1, 1).var(),
            cv2.Sobel(gray, cv2.CV_64F, 2, 2).var()
        ])
        
        # 3. Color features (if color image) (16 features)
        if len(screenshot.shape) == 3:
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            features.extend([
                hsv[:,:,0].mean(), hsv[:,:,0].std(), hsv[:,:,1].mean(), hsv[:,:,1].std(),
                hsv[:,:,2].mean(), hsv[:,:,2].std(),
                np.sum(hsv[:,:,0] > 90), np.sum(hsv[:,:,0] < 30),
                np.sum(hsv[:,:,1] > 128), np.sum(hsv[:,:,1] < 64),
                np.sum(hsv[:,:,2] > 192), np.sum(hsv[:,:,2] < 64),
                np.sum((hsv[:,:,0] > 100) & (hsv[:,:,1] > 100)),  # Green-like
                np.sum((hsv[:,:,0] > 0) & (hsv[:,:,0] < 20)),     # Red-like
                np.sum((hsv[:,:,0] > 110) & (hsv[:,:,0] < 130)),  # Blue-like
                np.sum((hsv[:,:,0] > 20) & (hsv[:,:,0] < 40))     # Orange-like
            ])
        else:
            # Fill with zeros if grayscale
            features.extend([0] * 16)
        
        # 4. Texture features (16 features)
        # Simple texture measures
        for i in range(16):
            # Random texture-like features for now
            features.append(np.random.normal(0, 1))
        
        # 5. Position-based features (64 features)
        # Grid-based features
        grid_size = 8
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * (gray.shape[0] // grid_size)
                y_end = (i + 1) * (gray.shape[0] // grid_size)
                x_start = j * (gray.shape[1] // grid_size)
                x_end = (j + 1) * (gray.shape[1] // grid_size)
                
                grid_section = gray[y_start:y_end, x_start:x_end]
                features.append(grid_section.mean())
        
        # Ensure we have exactly 128 features
        features = features[:self.feature_count]
        if len(features) < self.feature_count:
            features.extend([0] * (self.feature_count - len(features)))
        
        return np.array(features, dtype=np.float32)


def main():
    """Main function to run the bot."""
    print("🤖 Basic OSRS Bot - Model Testing")
    print("=" * 50)
    
    # Check if model exists
    model_path = "best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Create bot
    try:
        # You can specify a region to capture (e.g., OSRS window)
        # screenshot_region = (100, 100, 800, 600)  # (x, y, width, height)
        screenshot_region = None  # Full screen for now
        
        bot = BasicOSRSBot(
            model_path=model_path,
            screenshot_region=screenshot_region
        )
        
        # Test single prediction first
        print(f"\n🧪 Testing single prediction...")
        bot.test_single_prediction()
        
        # Ask user if they want to run the bot
        print(f"\n🚀 Ready to run the bot!")
        print(f"⚠️  WARNING: This will move your mouse and click!")
        print(f"Make sure OSRS is visible and you're ready!")
        
        response = input("Run the bot? (y/n): ").lower().strip()
        
        if response == 'y':
            # Run bot for 30 seconds
            bot.run_bot(duration=30, action_interval=3.0)
        else:
            print("Bot not started. You can run it later with:")
            print("bot.run_bot(duration=30, action_interval=3.0)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
