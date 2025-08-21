#!/usr/bin/env python3
"""Feature pipeline service for processing gamestate data"""

import time
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import logging

# Import shared pipeline modules
try:
    from shared_pipeline.features import extract_features_from_gamestate, FeatureExtractor
    from shared_pipeline.feature_map import load_feature_mappings
    from shared_pipeline.actions import flatten_action_window, convert_raw_actions_to_tensors
    from shared_pipeline.normalize import normalize_features, normalize_action_data
    from shared_pipeline.encodings import ActionEncoder
except ImportError as e:
    logging.error(f"Failed to import shared pipeline modules: {e}")
    raise

LOG = logging.getLogger(__name__)


class FeaturePipeline:
    """Pipeline for processing gamestate data into features and actions"""
    
    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        
        # --- explicit state so first access never raises AttributeError
        self.window: Optional[np.ndarray] = None        # (10,128), T0 at row 0
        self._prev_window: Optional[np.ndarray] = None  # (10,128)
        self.feature_names: list[str] = []              # len 128
        self.feature_groups: list[str] = []             # len 128
        self._deque: deque[np.ndarray] = deque(maxlen=10)
        self._action_windows: deque[List[float]] = deque(maxlen=20)
        
        # Load feature mappings
        try:
            mappings_file = data_root / "05_mappings" / "feature_mappings.json"
            self.feature_mappings = load_feature_mappings(str(mappings_file))
            LOG.info(f"Loaded {len(self.feature_mappings)} feature mappings")
            
            # Validate exactly 128 features
            if len(self.feature_mappings) != 128:
                raise RuntimeError(f"Expected exactly 128 features, got {len(self.feature_mappings)}")
            
            # Populate feature names and groups from mappings
            self.feature_names = [mapping['feature_name'] for mapping in self.feature_mappings]
            self.feature_groups = [mapping['feature_group'] for mapping in self.feature_mappings]
            
        except Exception as e:
            LOG.exception("Failed to load feature mappings")
            raise
        
        # Action encoder
        self.action_encoder = ActionEncoder()
        
        # Feature extractor instance - CREATE ONCE and REUSE
        self.feature_extractor = FeatureExtractor()
        
        # Session timing management
        self.session_start_time = None
        self.session_timing_initialized = False
        self.live_mode_start_time = None  # When live mode started (for relative timing)
        
        # Action window processing utilities
        self._encoder = self.action_encoder
    
    def extract_window(self, gamestate: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract features and build window, returning window and metadata.
        
        Args:
            gamestate: Raw gamestate data
            
        Returns:
            Tuple of (window, feature_names, feature_groups)
            
        Raises:
            RuntimeError: If feature extraction fails or vector length != 128
            ValueError: If NaN/Inf values detected
        """
        try:
            # Initialize session timing before the first extraction
            if not self.session_timing_initialized:
                # For live mode, we want relative timestamps starting from 0
                # The first gamestate becomes time 0
                self.session_start_time = gamestate.get('timestamp', 0)
                self.live_mode_start_time = self.session_start_time
                
                # Initialize the feature extractor with this session timing
                self.feature_extractor.initialize_session_timing([gamestate])
                self.session_timing_initialized = True
            
            # Extract features using the properly initialized extractor
            features = self.feature_extractor.extract_features_from_gamestate(gamestate)
            
            if features is None or len(features) != 128:
                error_msg = f"Invalid features extracted: {len(features) if features is not None else 'None'}"
                LOG.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Convert to numpy array and validate
            feats = np.asarray(features, dtype=float)
            
            # Check for NaN/Inf values
            if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                raise ValueError("NaN or Inf values detected in extracted features")
            
            # Check vector length
            if feats.shape[0] != 128:
                raise RuntimeError(f"Feature vector wrong size: {feats.shape}, expected (128,)")
            
            # Time axis must be rows (10) and features columns (128)
            # Insert newest at row 0 and shift older rows down (toward 9)
            # window shape: (10, 128)  [time x features]
            # newest sample vector: feats shape (128,)
            
            # Store previous window before updating
            if self.window is not None:
                self._prev_window = self.window.copy()
            
            if self.window is None or self.window.shape != (10, 128):
                self.window = np.zeros((10, 128), dtype=float)
            
            # shift down (older gets larger t index), drop the last row
            self.window[1:] = self.window[:-1]
            # put newest at t0 (row 0)
            self.window[0, :] = feats
            
            # Save ID mappings to disk for persistence
            try:
                # Use absolute path to ensure correct location
                import os
                save_path = os.path.abspath("data/05_mappings/live_id_mappings.json")
                self.feature_extractor.save_id_mappings(save_path)
                
                # Hot-reload mappings so new live IDs are visible immediately
                try:
                    if hasattr(self.controller, "mapping_service") and self.controller.mapping_service:
                        self.controller.mapping_service.reload()
                except Exception:
                    pass
                    
            except Exception as e:
                pass
            

            
            return self.window, self.feature_names, self.feature_groups
            
        except Exception as e:
            LOG.exception("Failed to process gamestate")
            raise  # Re-raise to stop execution
    
    def diff_mask(self, window: np.ndarray) -> np.ndarray:
        """
        Compute change mask by comparing window to previous window.
        
        Args:
            window: Current window with shape (10,128)
            
        Returns:
            Boolean mask indicating changed cells
            
        Raises:
            RuntimeError: If window shape is invalid
        """
        # Validate window shape
        if window.shape != (10, 128):
            raise RuntimeError(f"Window must have shape (10,128), got {window.shape}")
        
        # First frame: all non-zero entries count as changed
        if self._prev_window is None:
            changed_mask = np.ones_like(window, dtype=bool)
        else:
            # Compare with previous window
            changed_mask = (window != self._prev_window)
        
        return changed_mask
    
    def push(self, gamestate: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extract features, build window, and compute change mask.
        
        Args:
            gamestate: Raw gamestate data
            
        Returns:
            Tuple of (window, changed_mask, feature_names, feature_groups)
        """
        # Extract window and metadata
        window, feature_names, feature_groups = self.extract_window(gamestate)
        
        # Compute change mask
        changed_mask = self.diff_mask(window)
        
        return window, changed_mask, feature_names, feature_groups
    
    def push_actions(self, actions: List[Dict[str, Any]], current_time_ms: Optional[float] = None) -> bool:
        """
        Process actions and add to actions buffer.
        
        Args:
            actions: List of action events
            current_time_ms: Current timestamp in milliseconds
            
        Returns:
            True if successfully processed, False otherwise
        """
        try:
            if not actions:
                # Add empty action frame
                empty_frame = np.zeros(1 + 8 * 0)  # [count=0]
                # Note: actions buffer removed in simplified version
                return True
            
            # Use current time if not provided
            if current_time_ms is None:
                current_time_ms = time.time() * 1000
            
            # Flatten actions into 600ms window using shared pipeline
            action_frame = flatten_action_window(actions, self.action_encoder)
            
            if action_frame is None:
                error_msg = "Failed to flatten action window"
                LOG.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Note: actions buffer removed in simplified version
            LOG.debug(f"Processed actions frame: {len(action_frame)} values")
            return True
            
        except Exception as e:
            LOG.exception("Failed to process actions")
            raise  # Re-raise to stop execution
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        deque_count = len(self._deque)
        window_shape = self.window.shape if self.window is not None else None
        
        return {
            'deque_count': deque_count,
            'window_shape': window_shape,
            'is_warm': deque_count >= 10,
            'session_timing_initialized': self.session_timing_initialized,
            'session_start_time': self.session_start_time,
            'live_mode_start_time': self.live_mode_start_time,
            'action_windows_count': len(self._action_windows)
        }
    
    def clear_buffers(self):
        """Clear all buffers"""
        self._deque.clear()
        self.window = None
        self._prev_window = None
        self.session_timing_initialized = False
        self.session_start_time = None
        self.live_mode_start_time = None
        LOG.info("Cleared feature buffers")
    
    def reset_session_timing(self):
        """Reset session timing - useful when switching between different data sources"""
        self.session_timing_initialized = False
        self.session_start_time = None
        self.live_mode_start_time = None
        LOG.info("Reset session timing")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names from mappings"""
        return self.feature_names
    
    def get_feature_groups(self) -> List[str]:
        """Get list of feature groups for each feature"""
        return self.feature_groups
    
    def get_unique_feature_groups(self) -> List[str]:
        """Get list of unique feature groups for the combo box"""
        groups = set()
        for mapping in self.feature_mappings:
            groups.add(mapping['feature_group'])
        return sorted(list(groups))
    
    def get_feature_info(self, index: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific feature"""
        if 0 <= index < len(self.feature_mappings):
            return self.feature_mappings[index]
        return None

    def build_action_frame(self, actions: List[Dict[str, Any]]) -> List[float]:
        # Use shared pipeline normalization workflow:
        # 1) Build raw_action_data structure for a single gamestate window
        # 2) Normalize via normalize_action_data
        # 3) Convert to tensors via convert_raw_actions_to_tensors
        raw_action_data = [{
            'mouse_movements': [
                {
                    'timestamp': a.get('timestamp', 0),
                    'x': a.get('x_in_window', 0),
                    'y': a.get('y_in_window', 0)
                }
                for a in actions if a.get('event_type') == 'move'
            ],
            'clicks': [
                {
                    'timestamp': a.get('timestamp', 0),
                    'x': a.get('x_in_window', 0),
                    'y': a.get('y_in_window', 0),
                    'button': a.get('btn', '')
                }
                for a in actions if a.get('event_type') == 'click'
            ],
            'key_presses': [
                {
                    'timestamp': a.get('timestamp', 0),
                    'key': a.get('key', '')
                }
                for a in actions if a.get('event_type') == 'key_press'
            ],
            'key_releases': [
                {
                    'timestamp': a.get('timestamp', 0),
                    'key': a.get('key', '')
                }
                for a in actions if a.get('event_type') == 'key_release'
            ],
            'scrolls': [
                {
                    'timestamp': a.get('timestamp', 0),
                    'dx': a.get('scroll_dx', 0),
                    'dy': a.get('scroll_dy', 0)
                }
                for a in actions if a.get('event_type') == 'scroll'
            ]
        }]

        # Normalize action data using shared pipeline (non-None gate)
        normalized_raw = normalize_action_data(raw_action_data, normalized_features=np.zeros((1,1)))

        # Convert to training tensors using shared pipeline
        tensors = convert_raw_actions_to_tensors(normalized_raw, self._encoder)

        frame = tensors[0] if tensors else [0]
        return frame

    def record_action_window_from_actions(self, actions: List[Dict[str, Any]]) -> None:
        frame = self.build_action_frame(actions)
        self._action_windows.append(frame)

    def get_last_action_windows(self, count: int = 10) -> List[List[float]]:
        if count <= 0:
            return []
        items = list(self._action_windows)[-count:]
        return list(reversed(items))
