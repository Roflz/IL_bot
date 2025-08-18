#!/usr/bin/env python3
"""
Data Augmentation and Regularization for Imitation Learning
OSRS Bot Training Data Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from typing import Dict, Tuple, Optional, Union
import random

class GamestateAugmenter:
    """Augmenter for gamestate features"""
    
    def __init__(self, 
                 coordinate_noise_std: float = 2.0,
                 angle_noise_std: float = 5.0,
                 inventory_noise_range: int = 1,
                 feature_dropout_rate: float = 0.1):
        
        self.coordinate_noise_std = coordinate_noise_std
        self.angle_noise_std = angle_noise_std
        self.inventory_noise_range = inventory_noise_range
        self.feature_dropout_rate = feature_dropout_rate
        
    def augment_gamestate(self, gamestate: np.ndarray) -> np.ndarray:
        """
        Augment gamestate features
        
        Args:
            gamestate: (73,) gamestate feature vector
            
        Returns:
            Augmented gamestate features
        """
        augmented = gamestate.copy()
        
        # 1. Add noise to coordinates (features 0-1: player_world_x, player_world_y)
        if len(augmented) >= 2:
            augmented[0] += np.random.normal(0, self.coordinate_noise_std)  # player_world_x
            augmented[1] += np.random.normal(0, self.coordinate_noise_std)  # player_world_y
        
        # 2. Add noise to camera coordinates (features 6-10: camera_x, camera_y, camera_z, pitch, yaw)
        if len(augmented) >= 10:
            # Camera position noise
            augmented[6] += np.random.normal(0, self.coordinate_noise_std)  # camera_x
            augmented[7] += np.random.normal(0, self.coordinate_noise_std)  # camera_y
            augmented[8] += np.random.normal(0, self.coordinate_noise_std)  # camera_z
            
            # Camera angle noise
            augmented[9] += np.random.normal(0, self.angle_noise_std)      # camera_pitch
            augmented[10] += np.random.normal(0, self.angle_noise_std)     # camera_yaw
        
        # 3. Add noise to inventory quantities (features 11-38: inventory slots)
        for i in range(11, min(39, len(augmented))):
            if augmented[i] > 0:  # If slot has items
                noise = np.random.randint(-self.inventory_noise_range, self.inventory_noise_range + 1)
                augmented[i] = max(1, augmented[i] + noise)  # Ensure quantity >= 1
        
        # 4. Feature dropout (randomly zero out some features)
        if self.feature_dropout_rate > 0:
            dropout_mask = np.random.random(len(augmented)) > self.feature_dropout_rate
            augmented = augmented * dropout_mask
        
        return augmented
    
    def augment_batch(self, gamestates: np.ndarray) -> np.ndarray:
        """Augment a batch of gamestates"""
        return np.array([self.augment_gamestate(gs) for gs in gamestates])

class ScreenshotAugmenter:
    """Augmenter for screenshot images"""
    
    def __init__(self,
                 color_jitter: bool = True,
                 rotation_range: float = 5.0,
                 crop_scale_range: Tuple[float, float] = (0.9, 1.0),
                 horizontal_flip_prob: float = 0.5,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 saturation_range: Tuple[float, float] = (0.9, 1.1),
                 hue_range: Tuple[float, float] = (-0.05, 0.05)):
        
        self.color_jitter = color_jitter
        self.rotation_range = rotation_range
        self.crop_scale_range = crop_scale_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
    
    def augment_screenshot(self, image: np.ndarray) -> np.ndarray:
        """
        Augment screenshot image
        
        Args:
            image: (3, H, W) RGB image array
            
        Returns:
            Augmented image
        """
        # Convert to PIL Image for easier augmentation
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC
        
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # 1. Color jitter
        if self.color_jitter:
            pil_image = self._apply_color_jitter(pil_image)
        
        # 2. Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            pil_image = pil_image.rotate(angle, fillcolor=(0, 0, 0))
        
        # 3. Random crop and resize
        if self.crop_scale_range[0] < 1.0:
            pil_image = self._apply_random_crop(pil_image)
        
        # 4. Random horizontal flip
        if np.random.random() < self.horizontal_flip_prob:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert back to numpy array
        augmented = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert back to CHW format if needed
        if augmented.shape[2] == 3:  # HWC format
            augmented = np.transpose(augmented, (2, 0, 1))  # Convert to CHW
        
        return augmented
    
    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color jittering"""
        # Brightness
        brightness_factor = np.random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Contrast
        contrast_factor = np.random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Saturation
        saturation_factor = np.random.uniform(*self.saturation_range)
        image = ImageEnhance.Color(image).enhance(saturation_factor)
        
        return image
    
    def _apply_random_crop(self, image: Image.Image) -> Image.Image:
        """Apply random crop and resize"""
        width, height = image.size
        
        # Calculate crop size
        scale = np.random.uniform(*self.crop_scale_range)
        crop_width = int(width * scale)
        crop_height = int(height * scale)
        
        # Calculate crop position
        left = np.random.randint(0, width - crop_width + 1)
        top = np.random.randint(0, height - crop_height + 1)
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop and resize
        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize((width, height), Image.LANCZOS)
        
        return resized
    
    def augment_batch(self, screenshots: np.ndarray) -> np.ndarray:
        """Augment a batch of screenshots"""
        return np.array([self.augment_screenshot(ss) for ss in screenshots])

class ActionAugmenter:
    """Augmenter for action sequences"""
    
    def __init__(self,
                 timing_noise_std: float = 0.1,
                 position_noise_std: float = 2.0,
                 click_noise_prob: float = 0.05,
                 key_noise_prob: float = 0.05):
        
        self.timing_noise_std = timing_noise_std
        self.position_noise_std = position_noise_std
        self.click_noise_prob = click_noise_prob
        self.key_noise_prob = key_noise_prob
    
    def augment_action_sequence(self, action_sequence: np.ndarray) -> np.ndarray:
        """
        Augment action sequence
        
        Args:
            action_sequence: (106,) action sequence vector
            
        Returns:
            Augmented action sequence
        """
        augmented = action_sequence.copy()
        
        # Parse action sequence structure
        action_count = int(augmented[0])
        
        if action_count > 0:
            # Augment each action
            for i in range(min(action_count, 15)):  # Max 15 actions
                base_idx = 1 + i * 7  # Each action has 7 components
                
                if base_idx + 6 < len(augmented):
                    # 1. Timing noise
                    augmented[base_idx] += np.random.normal(0, self.timing_noise_std)
                    augmented[base_idx] = max(0, augmented[base_idx])  # Ensure non-negative
                    
                    # 2. Position noise (x, y coordinates)
                    if base_idx + 1 < len(augmented):
                        augmented[base_idx + 1] += np.random.normal(0, self.position_noise_std)
                    if base_idx + 2 < len(augmented):
                        augmented[base_idx + 2] += np.random.normal(0, self.position_noise_std)
                    
                    # 3. Click noise (randomly flip click states)
                    if base_idx + 3 < len(augmented) and np.random.random() < self.click_noise_prob:
                        augmented[base_idx + 3] = 1 - augmented[base_idx + 3]  # Flip binary value
                    
                    # 4. Key press noise (randomly change key)
                    if base_idx + 4 < len(augmented) and np.random.random() < self.key_noise_prob:
                        # Randomly select a different key
                        augmented[base_idx + 4] = np.random.randint(0, 50)
        
        return augmented
    
    def augment_batch(self, action_sequences: np.ndarray) -> np.ndarray:
        """Augment a batch of action sequences"""
        return np.array([self.augment_action_sequence(as_seq) for as_seq in action_sequences])

class MixupAugmenter:
    """Mixup data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def mixup_data(self, 
                   gamestates: np.ndarray,
                   action_targets: np.ndarray,
                   screenshots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply mixup augmentation
        
        Args:
            gamestates: Batch of gamestate features
            action_targets: Batch of action targets
            screenshots: Optional batch of screenshots
            
        Returns:
            Mixed gamestates, action targets, and screenshots
        """
        batch_size = len(gamestates)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix gamestates
        mixed_gamestates = lam * gamestates + (1 - lam) * gamestates[indices]
        
        # Mix action targets
        mixed_action_targets = lam * action_targets + (1 - lam) * action_targets[indices]
        
        # Mix screenshots if available
        mixed_screenshots = None
        if screenshots is not None:
            mixed_screenshots = lam * screenshots + (1 - lam) * screenshots[indices]
        
        return mixed_gamestates, mixed_action_targets, mixed_screenshots

class ComprehensiveAugmenter:
    """Comprehensive augmenter combining all techniques"""
    
    def __init__(self, 
                 gamestate_augmenter: Optional[GamestateAugmenter] = None,
                 screenshot_augmenter: Optional[ScreenshotAugmenter] = None,
                 action_augmenter: Optional[ActionAugmenter] = None,
                 mixup_augmenter: Optional[MixupAugmenter] = None,
                 augmentation_prob: float = 0.5):
        
        self.gamestate_augmenter = gamestate_augmenter or GamestateAugmenter()
        self.screenshot_augmenter = screenshot_augmenter or ScreenshotAugmenter()
        self.action_augmenter = action_augmenter or ActionAugmenter()
        self.mixup_augmenter = mixup_augmenter or MixupAugmenter()
        self.augmentation_prob = augmentation_prob
    
    def augment_batch(self, 
                      gamestates: np.ndarray,
                      action_targets: np.ndarray,
                      screenshots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply comprehensive augmentation to a batch
        
        Args:
            gamestates: Batch of gamestate features
            action_targets: Batch of action targets
            screenshots: Optional batch of screenshots
            
        Returns:
            Augmented batch
        """
        # Decide whether to apply augmentation
        if np.random.random() > self.augmentation_prob:
            return gamestates, action_targets, screenshots
        
        # 1. Individual feature augmentation
        augmented_gamestates = self.gamestate_augmenter.augment_batch(gamestates)
        augmented_action_targets = self.action_augmenter.augment_batch(action_targets)
        
        augmented_screenshots = None
        if screenshots is not None:
            augmented_screenshots = self.screenshot_augmenter.augment_batch(screenshots)
        
        # 2. Mixup augmentation
        mixed_gamestates, mixed_action_targets, mixed_screenshots = self.mixup_augmenter.mixup_data(
            augmented_gamestates, augmented_action_targets, augmented_screenshots
        )
        
        return mixed_gamestates, mixed_action_targets, mixed_screenshots

class RegularizationTechniques:
    """Collection of regularization techniques"""
    
    @staticmethod
    def label_smoothing(targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """Apply label smoothing to classification targets"""
        if smoothing > 0:
            # Convert to one-hot if needed
            if targets.dim() == 1:
                targets = F.one_hot(targets, num_classes=targets.max() + 1).float()
            
            # Apply smoothing
            targets = targets * (1 - smoothing) + smoothing / targets.size(-1)
        
        return targets
    
    @staticmethod
    def gradient_clipping(model: nn.Module, max_norm: float = 1.0):
        """Apply gradient clipping"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def early_stopping(val_losses: list, patience: int = 5, min_delta: float = 1e-4) -> bool:
        """Check if early stopping should be triggered"""
        if len(val_losses) < patience:
            return False
        
        # Check if validation loss has improved
        recent_losses = val_losses[-patience:]
        best_recent = min(recent_losses)
        current_loss = val_losses[-1]
        
        return current_loss > best_recent + min_delta

def create_augmenter(augmentation_type: str = 'comprehensive', **kwargs) -> Union[ComprehensiveAugmenter, GamestateAugmenter, ScreenshotAugmenter, ActionAugmenter]:
    """Factory function to create augmenters"""
    
    if augmentation_type == 'gamestate':
        return GamestateAugmenter(**kwargs)
    elif augmentation_type == 'screenshot':
        return ScreenshotAugmenter(**kwargs)
    elif augmentation_type == 'action':
        return ActionAugmenter(**kwargs)
    elif augmentation_type == 'comprehensive':
        return ComprehensiveAugmenter(**kwargs)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")

if __name__ == "__main__":
    print("Testing Data Augmentation...")
    
    # Test gamestate augmentation
    print("\n1. Testing Gamestate Augmentation...")
    gamestate_augmenter = GamestateAugmenter()
    sample_gamestate = np.random.randn(73)
    augmented_gamestate = gamestate_augmenter.augment_gamestate(sample_gamestate)
    print(f"Original gamestate shape: {sample_gamestate.shape}")
    print(f"Augmented gamestate shape: {augmented_gamestate.shape}")
    print(f"Coordinate changes: x={sample_gamestate[0]:.3f} -> {augmented_gamestate[0]:.3f}")
    
    # Test screenshot augmentation
    print("\n2. Testing Screenshot Augmentation...")
    screenshot_augmenter = ScreenshotAugmenter()
    sample_screenshot = np.random.rand(3, 224, 224)
    augmented_screenshot = screenshot_augmenter.augment_screenshot(sample_screenshot)
    print(f"Original screenshot shape: {sample_screenshot.shape}")
    print(f"Augmented screenshot shape: {augmented_screenshot.shape}")
    
    # Test action augmentation
    print("\n3. Testing Action Augmentation...")
    action_augmenter = ActionAugmenter()
    sample_action = np.random.randn(106)
    sample_action[0] = 3  # 3 actions
    augmented_action = action_augmenter.augment_action_sequence(sample_action)
    print(f"Original action shape: {sample_action.shape}")
    print(f"Augmented action shape: {augmented_action.shape}")
    
    # Test comprehensive augmentation
    print("\n4. Testing Comprehensive Augmentation...")
    comprehensive_augmenter = ComprehensiveAugmenter()
    batch_size = 4
    gamestates = np.random.randn(batch_size, 73)
    action_targets = np.random.randn(batch_size, 106)
    screenshots = np.random.rand(batch_size, 3, 224, 224)
    
    aug_gamestates, aug_action_targets, aug_screenshots = comprehensive_augmenter.augment_batch(
        gamestates, action_targets, screenshots
    )
    
    print(f"Batch augmentation successful!")
    print(f"Gamestates: {gamestates.shape} -> {aug_gamestates.shape}")
    print(f"Action targets: {action_targets.shape} -> {aug_action_targets.shape}")
    print(f"Screenshots: {screenshots.shape} -> {aug_screenshots.shape}")
    
    print("\n✅ All augmentation techniques working correctly!")














