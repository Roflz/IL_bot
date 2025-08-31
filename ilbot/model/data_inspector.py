#!/usr/bin/env python3
"""
DataInspector class for auto-detecting model configuration from training data.
Automatically determines action features, gamestate dimensions, categorical sizes, etc.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


class DataInspector:
    """
    Automatically detects model configuration from training data files.
    No fallbacks - fails fast if data cannot be read or analyzed.
    """
    
    def __init__(self, action_input_path: str, gamestate_path: str, action_targets_path: str):
        """
        Initialize with paths to training data files.
        
        Args:
            action_input_path: Path to action_input_sequences.npy
            gamestate_path: Path to gamestate_sequences.npy  
            action_targets_path: Path to action_targets.npy
        """
        self.action_input_path = Path(action_input_path)
        self.gamestate_path = Path(gamestate_path)
        self.action_targets_path = Path(action_targets_path)
        
        # Validate paths exist
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all data files exist."""
        if not self.action_input_path.exists():
            raise FileNotFoundError(f"Action input sequences not found: {self.action_input_path}")
        if not self.gamestate_path.exists():
            raise FileNotFoundError(f"Gamestate sequences not found: {self.gamestate_path}")
        if not self.action_targets_path.exists():
            raise FileNotFoundError(f"Action targets not found: {self.action_targets_path}")
    
    def auto_detect(self) -> Dict[str, Any]:
        """
        Auto-detect all model configuration from data files.
        
        Returns:
            Dict containing all detected configuration values
            
        Raises:
            ValueError: If data has unexpected shapes or cannot be analyzed
        """
        # Load data files
        action_input_data = np.load(self.action_input_path)
        gamestate_data = np.load(self.gamestate_path)
        action_targets_data = np.load(self.action_targets_path)
        
        # Validate data shapes
        self._validate_data_shapes(action_input_data, gamestate_data, action_targets_data)
        
        # Extract dimensions
        config = {}
        
        # Action input sequences: (B, 10, 100, 7)
        config['batch_size'] = action_input_data.shape[0]
        config['temporal_window'] = action_input_data.shape[1]  # 10
        config['max_actions'] = action_input_data.shape[2]      # 100
        config['action_features'] = action_input_data.shape[3]  # 7
        
        # Gamestate sequences: (B, 10, 128)
        config['gamestate_dim'] = gamestate_data.shape[2]      # 128
        
        # Action targets: (B, 100, 7)
        config['action_target_features'] = action_targets_data.shape[2]  # 7
        
        # Validate consistency
        if config['action_features'] != config['action_target_features']:
            raise ValueError(f"Action features mismatch: input={config['action_features']}, targets={config['action_target_features']}")
        
        # Detect categorical sizes from actual data values
        config['categorical_sizes'] = self._detect_categorical_sizes(action_targets_data)
        
        # Detect event types (CLICK, KEY, SCROLL, MOVE)
        config['event_types'] = 4  # This is fixed by design, not data-driven
        
        return config
    
    def _validate_data_shapes(self, action_input_data: np.ndarray, gamestate_data: np.ndarray, action_targets_data: np.ndarray):
        """Validate that data has expected shapes and dimensions."""
        # Check action input: (B, 10, 100, 7)
        if len(action_input_data.shape) != 4:
            raise ValueError(f"Action input data must be 4D, got shape: {action_input_data.shape}")
        
        # Check gamestate: (B, 10, 128)
        if len(gamestate_data.shape) != 3:
            raise ValueError(f"Gamestate data must be 3D, got shape: {gamestate_data.shape}")
        
        # Check action targets: (B, 100, 7)
        if len(action_targets_data.shape) != 3:
            raise ValueError(f"Action targets data must be 3D, got shape: {action_targets_data.shape}")
        
        # Validate batch sizes match
        if action_input_data.shape[0] != gamestate_data.shape[0]:
            raise ValueError(f"Batch size mismatch: action_input={action_input_data.shape[0]}, gamestate={gamestate_data.shape[0]}")
        
        if action_input_data.shape[0] != action_targets_data.shape[0]:
            raise ValueError(f"Batch size mismatch: action_input={action_input_data.shape[0]}, action_targets={action_targets_data.shape[0]}")
        
        # Validate temporal windows match
        if action_input_data.shape[1] != gamestate_data.shape[1]:
            raise ValueError(f"Temporal window mismatch: action_input={action_input_data.shape[1]}, gamestate={gamestate_data.shape[1]}")
        
        # Validate max actions match
        if action_input_data.shape[2] != action_targets_data.shape[1]:
            raise ValueError(f"Max actions mismatch: action_input={action_input_data.shape[2]}, action_targets={action_targets_data.shape[1]}")
    
    def _detect_categorical_sizes(self, action_targets_data: np.ndarray) -> Dict[str, int]:
        """
        Detect categorical feature sizes from actual data values.
        
        Args:
            action_targets_data: Action targets array (B, 100, 7)
            
        Returns:
            Dict mapping feature names to their category counts
        """
        categorical_sizes = {}
        
        # Button: column 3 (index 3)
        button_values = action_targets_data[:, :, 3].flatten()
        button_unique = np.unique(button_values)
        categorical_sizes['button'] = len(button_unique)
        
        # Key action: column 4 (index 4)
        key_action_values = action_targets_data[:, :, 4].flatten()
        key_action_unique = np.unique(key_action_values)
        categorical_sizes['key_action'] = len(key_action_unique)
        
        # Key ID: column 5 (index 5)
        key_id_values = action_targets_data[:, :, 5].flatten()
        key_id_unique = np.unique(key_id_values)
        categorical_sizes['key_id'] = len(key_id_unique)
        
        # Scroll: column 6 (index 6)
        scroll_values = action_targets_data[:, :, 6].flatten()
        scroll_unique = np.unique(scroll_values)
        categorical_sizes['scroll'] = len(scroll_unique)
        
        return categorical_sizes
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get configuration suitable for model initialization.
        
        Returns:
            Dict with model configuration parameters
        """
        config = self.auto_detect()
        
        # Extract model-specific parameters
        model_config = {
            'gamestate_dim': config['gamestate_dim'],
            'max_actions': config['max_actions'],
            'action_features': config['action_features'],
            'temporal_window': config['temporal_window'],
            'enum_sizes': config['categorical_sizes'],
            'event_types': config['event_types']
        }
        
        return model_config
