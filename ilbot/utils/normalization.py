#!/usr/bin/env python3
"""
Shared Normalization Utilities

This module provides normalization methods that are consistent between
the training pipeline and the GUI display. It extracts the normalization
logic from OSRSDataset to ensure consistency.
"""

import numpy as np
import torch
from typing import Union, Optional, Dict, Any
import json
from pathlib import Path


class NormalizationUtils:
    """
    Utility class for consistent normalization across training and GUI.
    """
    
    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_scale = 10000.0
        self.camera_scale = 10000.0
        self.angle_scale = 2040.0  # Based on real max yaw value from experimentation
        self.camera_scale_max = 6000.0  # Based on real max scale value from experimentation
        self.time_scale = 1000.0  # Convert ms to seconds
    
    def normalize_gamestate_features(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize gamestate features using the same logic as OSRSDataset.
        
        Args:
            features: (T, G) or (N, G) tensor/array of gamestate features
            
        Returns:
            Normalized gamestate features with same type as input
        """
        is_torch = isinstance(features, torch.Tensor)
        if is_torch:
            normalized = features.clone()
        else:
            normalized = features.copy()
        
        # Player world coordinates (indices 0, 1)
        normalized[..., 0] = features[..., 0] / self.world_scale  # player_world_x
        normalized[..., 1] = features[..., 1] / self.world_scale  # player_world_y
        
        # Camera coordinates (indices 9, 10, 11)
        normalized[..., 9] = features[..., 9] / self.camera_scale   # camera_x
        normalized[..., 10] = features[..., 10] / self.camera_scale # camera_y
        normalized[..., 11] = features[..., 11] / self.camera_scale # camera_z
        
        # Camera angles (indices 12, 13)
        normalized[..., 12] = features[..., 12] / self.angle_scale  # camera_pitch
        normalized[..., 13] = features[..., 13] / self.angle_scale  # camera_yaw
        
        # Camera scale (index 14)
        normalized[..., 14] = features[..., 14] / self.camera_scale_max
        
        # Time features - normalize to seconds (divide by 1000)
        normalized[..., 8] = features[..., 8] / self.time_scale  # time_since_interaction
        
        # Phase timing features (indices 65, 66)
        normalized[..., 65] = features[..., 65] / self.time_scale  # phase_start_time
        normalized[..., 66] = features[..., 66] / self.time_scale  # phase_duration
        
        # Timestamp (index 128) - normalize to seconds
        if features.shape[-1] > 128:
            normalized[..., 128] = features[..., 128] / self.time_scale
        
        # Bank material coordinates (indices 47, 48, 52, 53, 57, 58, 62, 63) - screen coordinates
        bank_coord_indices = [47, 48, 52, 53, 57, 58, 62, 63]
        for i, coord_idx in enumerate(bank_coord_indices):
            if coord_idx < features.shape[-1]:
                if i % 2 == 0:  # X coordinates
                    normalized[..., coord_idx] = features[..., coord_idx] / self.screen_width
                else:  # Y coordinates
                    normalized[..., coord_idx] = features[..., coord_idx] / self.screen_height
        
        # Game object coordinates (indices 68-109) - world coordinates
        for i in range(68, 110, 3):  # Every 3rd feature starting from 68
            if i + 1 < features.shape[-1]:  # X coordinate
                normalized[..., i + 1] = features[..., i + 1] / self.world_scale
            if i + 2 < features.shape[-1]:  # Y coordinate
                normalized[..., i + 2] = features[..., i + 2] / self.world_scale
        
        # NPC coordinates (indices 110-124) - world coordinates
        for i in range(110, 125, 3):  # Every 3rd feature starting from 110
            if i + 1 < features.shape[-1]:  # X coordinate
                normalized[..., i + 1] = features[..., i + 1] / self.world_scale
            if i + 2 < features.shape[-1]:  # Y coordinate
                normalized[..., i + 2] = features[..., i + 2] / self.world_scale
        
        return normalized
    
    def normalize_action_input_features(self, action_features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize action input features using the same logic as OSRSDataset.
        
        Args:
            action_features: (T, A, Fin) or (A, Fin) tensor/array of action input features
            Format: [timestamp, x, y, click, key_action, key_id, scroll]
            
        Returns:
            Normalized action input features with same type as input
        """
        is_torch = isinstance(action_features, torch.Tensor)
        if is_torch:
            normalized = action_features.clone()
        else:
            normalized = action_features.copy()
        
        # Normalize timestamp (index 0) to seconds
        normalized[..., 0] = action_features[..., 0] / self.time_scale
        
        # Normalize X coordinates (index 1) to [0, 1]
        normalized[..., 1] = action_features[..., 1] / self.screen_width
        
        # Normalize Y coordinates (index 2) to [0, 1]
        normalized[..., 2] = action_features[..., 2] / self.screen_height
        
        # Other features (click, key_action, key_id, scroll) are already in appropriate ranges
        # and don't need normalization
        
        return normalized
    
    def normalize_action_target_features(self, action_targets: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize action target features using the same logic as OSRSDataset.
        
        Args:
            action_targets: (A, 7|8) tensor/array of action targets
            Format: [timestamp, x, y, click, key_action, key_id, scroll] or [timestamp, x, y, click, key_action, key_id, scroll, scroll_dy]
            
        Returns:
            Normalized action target features with same type as input
        """
        is_torch = isinstance(action_targets, torch.Tensor)
        if is_torch:
            normalized = action_targets.clone()
        else:
            normalized = action_targets.copy()
        
        # Normalize timestamp (index 0) to seconds
        normalized[..., 0] = action_targets[..., 0] / self.time_scale
        
        # Normalize X coordinates (index 1) to [0, 1]
        normalized[..., 1] = action_targets[..., 1] / self.screen_width
        
        # Normalize Y coordinates (index 2) to [0, 1]
        normalized[..., 2] = action_targets[..., 2] / self.screen_height
        
        # Other features are already in appropriate ranges
        
        return normalized
    
    def normalize_single_value(self, value: float, feature_index: int, feature_name: Optional[str] = None) -> float:
        """
        Normalize a single feature value based on its index and name.
        This is used by the GUI for individual value display.
        
        Args:
            value: The raw feature value
            feature_index: The feature index (0-128 for gamestate features)
            feature_name: Optional feature name for additional context
            
        Returns:
            Normalized value
        """
        # Action features (for action sequences) - prioritize feature names over indices
        if feature_name and feature_name.lower() in ["timestamp", "x", "y", "click", "key_action", "key_id", "scroll"]:
            if feature_name.lower() == "timestamp":
                return value / self.time_scale
            elif feature_name.lower() == "x":
                return value / self.screen_width
            elif feature_name.lower() == "y":
                return value / self.screen_height
        
        # Time features - normalize to seconds
        elif (feature_index == 8 or  # time_since_interaction
            feature_index == 65 or  # phase_start_time
            feature_index == 66 or  # phase_duration
            feature_index == 128 or  # timestamp
            (feature_name and ("timestamp" in feature_name.lower() or "time" in feature_name.lower()))):
            return value / self.time_scale
        
        # Player world coordinates
        elif feature_index in [0, 1]:  # player_world_x, player_world_y
            return value / self.world_scale
        
        # Camera coordinates
        elif feature_index in [9, 10, 11]:  # camera_x, camera_y, camera_z
            return value / self.camera_scale
        
        # Camera angles
        elif feature_index in [12, 13]:  # camera_pitch, camera_yaw
            return value / self.angle_scale
        
        # Camera scale
        elif feature_index == 14:  # camera_scale
            return value / self.camera_scale_max
        
        # Bank material coordinates (screen coordinates)
        elif feature_index in [47, 52, 57, 62]:  # X coordinates
            return value / self.screen_width
        elif feature_index in [48, 53, 58, 63]:  # Y coordinates
            return value / self.screen_height
        
        # Game object coordinates (world coordinates)
        elif 68 <= feature_index <= 109 and (feature_index - 68) % 3 == 1:  # X coordinates
            return value / self.world_scale
        elif 68 <= feature_index <= 109 and (feature_index - 68) % 3 == 2:  # Y coordinates
            return value / self.world_scale
        
        # NPC coordinates (world coordinates)
        elif 110 <= feature_index <= 124 and (feature_index - 110) % 3 == 1:  # X coordinates
            return value / self.world_scale
        elif 110 <= feature_index <= 124 and (feature_index - 110) % 3 == 2:  # Y coordinates
            return value / self.world_scale
        
        # Default: return original value (no normalization)
        return value


# Global instance for easy access
normalization_utils = NormalizationUtils()


def normalize_gamestate_features(features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for gamestate feature normalization."""
    return normalization_utils.normalize_gamestate_features(features)


def normalize_action_input_features(action_features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for action input feature normalization."""
    return normalization_utils.normalize_action_input_features(action_features)


def normalize_action_target_features(action_targets: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for action target feature normalization."""
    return normalization_utils.normalize_action_target_features(action_targets)


def normalize_single_value(value: float, feature_index: int, feature_name: Optional[str] = None) -> float:
    """Convenience function for single value normalization."""
    return normalization_utils.normalize_single_value(value, feature_index, feature_name)
