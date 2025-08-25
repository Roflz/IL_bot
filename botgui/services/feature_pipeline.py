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
    
    def __init__(self, data_root: Path = Path("data"), actions_service=None):
        self.data_root = data_root
        self.actions_service = actions_service
        
        # --- explicit state so first access never raises AttributeError
        self.window: Optional[np.ndarray] = None        # (10,128), T0 at row 0
        self._prev_window: Optional[np.ndarray] = None  # (10,128)
        self.feature_names: list[str] = []              # len 128
        self.feature_groups: list[str] = []             # len 128
        self._deque: deque[np.ndarray] = deque(maxlen=10)
        self._action_windows: deque[List[float]] = deque(maxlen=20)
        
        # Gamestate storage for final 10 timesteps
        self._gamestate_windows: deque[Dict[str, Any]] = deque(maxlen=20)
        self._feature_windows: deque[np.ndarray] = deque(maxlen=20)
        
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
        # Wall-clock ↔ gamestate clock alignment
        self.session_wallclock_at_start_ms = None
        self.wallclock_to_gs_offset_ms = 0
        
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
                # Capture wall-clock at the moment we latched session_start_time
                import time as _time
                self.session_wallclock_at_start_ms = int(_time.time() * 1000)
                self.wallclock_to_gs_offset_ms = int(self.session_start_time) - int(self.session_wallclock_at_start_ms)
                
                # Initialize and PIN the feature extractor to this exact base
                self.feature_extractor.initialize_session_timing([gamestate])
                self.feature_extractor.session_start_time = self.session_start_time
                self.feature_extractor.session_start_time_initialized = True
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
            
            # DEBUG: Check if window is being updated correctly
            if self.window is not None:
                print(f"DEBUG: extract_window: Window updated")
                print(f"DEBUG: extract_window: Window shape: {self.window.shape}")
                print(f"DEBUG: extract_window: Newest row (T0) sample values: {self.window[0, :5]}...")
                print(f"DEBUG: extract_window: Newest row (T0) timestamp feature (127): {self.window[0, 127]}")
                
                if self.window.shape[0] > 1:
                    print(f"DEBUG: extract_window: Previous row (T1) sample values: {self.window[1, :5]}...")
                    print(f"DEBUG: extract_window: Previous row (T1) timestamp feature (127): {self.window[1, 127]}")
                    
                    # Check if rows are different
                    if np.array_equal(self.window[0], self.window[1]):
                        print("WARNING: extract_window: Newest and previous rows are identical!")
                        print(f"DEBUG: extract_window: T0 timestamp: {self.window[0, 127]}")
                        print(f"DEBUG: extract_window: T1 timestamp: {self.window[1, 127]}")
                    else:
                        print("DEBUG: extract_window: Newest and previous rows are different")
                
                # Check if ALL rows are identical (stale data issue)
                if self.window.shape[0] >= 2:
                    all_identical = True
                    for i in range(1, self.window.shape[0]):
                        if not np.array_equal(self.window[0], self.window[i]):
                            all_identical = False
                            break
                    
                    if all_identical:
                        print("CRITICAL ERROR: extract_window: ALL rows are identical - window is not shifting!")
                        print(f"DEBUG: extract_window: All rows have timestamp: {self.window[0, 127]}")
                    else:
                        print("DEBUG: extract_window: Window rows are different - shifting is working")
            
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
        try:
            print("🔍 DEBUG [feature_pipeline.py:230] FeaturePipeline.push called")
            print(f"🔍 DEBUG [feature_pipeline.py:235] Gamestate keys: {list(gamestate.keys()) if gamestate else 'None'}")
            print(f"🔍 DEBUG [feature_pipeline.py:240] Gamestate timestamp: {gamestate.get('timestamp', 'unknown') if gamestate else 'None'}")
            
            # Extract window and metadata
            print("🔍 DEBUG [feature_pipeline.py:245] About to call extract_window...")
            window, feature_names, feature_groups = self.extract_window(gamestate)
            print(f"🔍 DEBUG [feature_pipeline.py:250] extract_window returned - window: {type(window)}, names: {len(feature_names) if feature_names else 'None'}, groups: {len(feature_groups) if feature_groups else 'None'}")
            
            if window is not None:
                print(f"🔍 DEBUG [feature_pipeline.py:255] Window shape: {window.shape}")
            else:
                print("❌ ERROR [feature_pipeline.py:260] Window is None from extract_window")
                return None, None, None, None
            
            # Store gamestate and features for final 10 timesteps
            self._gamestate_windows.append(gamestate)
            self._feature_windows.append(window[-1].copy())  # Store a copy of the newest feature vector
            print("🔍 DEBUG [feature_pipeline.py:270] Stored gamestate and features in buffers")
            
            # Compute change mask
            print("🔍 DEBUG [feature_pipeline.py:275] About to compute change mask...")
            changed_mask = self.diff_mask(window)
            print(f"🔍 DEBUG [feature_pipeline.py:280] Change mask computed - shape: {changed_mask.shape if changed_mask is not None else 'None'}")
            
            print("🔍 DEBUG [feature_pipeline.py:285] FeaturePipeline.push completed successfully")
            return window, changed_mask, feature_names, feature_groups
            
        except Exception as e:
            print(f"❌ ERROR [feature_pipeline.py:290] in FeaturePipeline.push: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
        self.session_wallclock_at_start_ms = None
        self.wallclock_to_gs_offset_ms = 0
        # IMPORTANT: also reset the extractor's session timing
        try:
            self.feature_extractor.session_start_time = None
            self.feature_extractor.session_start_time_initialized = False
        except Exception:
            pass
        LOG.info("Cleared feature buffers")
    
    def reset_session_timing(self):
        """Reset session timing - useful when switching between different data sources"""
        self.session_timing_initialized = False
        self.session_start_time = None
        self.live_mode_start_time = None
        self.session_wallclock_at_start_ms = None
        self.wallclock_to_gs_offset_ms = 0
        try:
            self.feature_extractor.session_start_time = None
            self.feature_extractor.session_start_time_initialized = False
        except Exception:
            pass
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
    
    def _pad_action_sequences_to_fixed_length(self, action_tensors: List[List[float]], max_actions: int = 100) -> np.ndarray:
        """
        Pad action sequences to fixed length numpy array format expected by the model.
        
        Args:
            action_tensors: List of action tensors from convert_raw_actions_to_tensors
            max_actions: Maximum number of actions per timestep (default 100)
            
        Returns:
            Numpy array of shape (10, 101, 8) with action count at index 0, then up to 100 actions
        """
        timesteps = 10
        features_per_action = 8
        
        # Output: (10, 101, 8) - 10 timesteps, each with action count at index 0 + up to 100 actions
        action_array = np.zeros((timesteps, max_actions + 1, features_per_action), dtype=np.float32)
        
        # Process each of the 10 timesteps
        for i in range(min(timesteps, len(action_tensors))):
            timestep_tensor = action_tensors[i]  # One timestep's data
            
            if len(timestep_tensor) >= 1:
                action_count = int(timestep_tensor[0])
                
                # Store action count in the first row (index 0), first position
                action_array[i, 0, 0] = action_count
                
                # Parse remaining actions (starting from index 1)
                if len(timestep_tensor) > 1:
                    actions_data = timestep_tensor[1:]  # Skip action count
                    max_actions_found = min(len(actions_data) // features_per_action, max_actions)
                    
                    for j in range(max_actions_found):
                        start_idx = j * features_per_action
                        end_idx = start_idx + features_per_action
                        if end_idx <= len(actions_data):
                            action_features = actions_data[start_idx:end_idx]
                            # Store actions starting at index 1 (after the count row)
                            action_array[i, j + 1, :] = action_features
        
        return action_array

    def save_final_data(self, output_dir: Path = None) -> None:
        """
        Save the final 10 timesteps of data when live tracking stops.
        Creates all the files needed for the sample buttons.
        """
        try:
            # Use data_root/sample_data as default if no output_dir specified
            if output_dir is None:
                output_dir = self.data_root / "sample_data"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get the last 10 timesteps
            gamestates = list(self._gamestate_windows)[-10:]
            features = list(self._feature_windows)[-10:]
            
            if len(gamestates) < 10:
                LOG.warning(f"Only {len(gamestates)} timesteps available, expected 10")
                return
            
            # Save gamestate sequences
            # Non-normalized (raw features)
            raw_features_array = np.array(features)
            np.save(output_dir / "non_normalized_gamestate_sequence.npy", raw_features_array)
            
            # Normalized features
            mappings_file = self.data_root / "05_mappings" / "feature_mappings.json"
            normalized_features = normalize_features(raw_features_array, str(mappings_file))
            np.save(output_dir / "normalized_gamestate_sequence.npy", normalized_features)
            
            # Process actions using shared pipeline workflow
            # This is the key fix: use the existing pipeline method that properly syncs actions to gamestates
            raw_action_data = self._extract_raw_action_data_using_pipeline(gamestates)
            
            # Non-normalized action sequence
            non_normalized_actions = convert_raw_actions_to_tensors(raw_action_data, self._encoder)
            LOG.debug(f"DEBUG: Non-normalized actions shape: {len(non_normalized_actions)} tensors")
            LOG.debug(f"DEBUG: First tensor length: {len(non_normalized_actions[0]) if non_normalized_actions else 0}")
            LOG.debug(f"DEBUG: Last tensor length: {len(non_normalized_actions[-1]) if non_normalized_actions else 0}")
            
            # Save as list since tensors have variable lengths
            import pickle
            with open(output_dir / "non_normalized_action_sequence.pkl", 'wb') as f:
                pickle.dump(non_normalized_actions, f)
            
            # Also save as padded numpy array for model input
            padded_non_normalized_actions = self._pad_action_sequences_to_fixed_length(non_normalized_actions)
            np.save(output_dir / "non_normalized_action_sequence.npy", padded_non_normalized_actions)
            LOG.debug(f"DEBUG: Saved non-normalized actions as numpy array with shape: {padded_non_normalized_actions.shape}")
            
            # Normalized action sequence
            normalized_action_data = normalize_action_data(raw_action_data, normalized_features)
            normalized_actions = convert_raw_actions_to_tensors(normalized_action_data, self._encoder)
            LOG.debug(f"DEBUG: Normalized actions shape: {len(normalized_actions)} tensors")
            LOG.debug(f"DEBUG: First tensor length: {len(normalized_actions[0]) if normalized_actions else 0}")
            LOG.debug(f"DEBUG: Last tensor length: {len(normalized_actions[-1]) if normalized_actions else 0}")
            
            # Save as list since tensors have variable lengths
            with open(output_dir / "normalized_action_sequence.pkl", 'wb') as f:
                pickle.dump(normalized_actions, f)
            
            # Also save as padded numpy array for model input
            padded_normalized_actions = self._pad_action_sequences_to_fixed_length(normalized_actions)
            np.save(output_dir / "normalized_action_sequence.npy", padded_normalized_actions)
            LOG.debug(f"DEBUG: Saved normalized actions as numpy array with shape: {padded_normalized_actions.shape}")
            
            # Also save actions.csv for compatibility with existing tools
            actions_csv_path = output_dir / "actions.csv"
            self._save_actions_csv_from_pipeline_data(gamestates, raw_action_data, actions_csv_path)
            
            LOG.info(f"Saved final data to {output_dir}")
            
        except Exception as e:
            LOG.exception("Failed to save final data")
            raise
    
    def _extract_raw_action_data_using_pipeline(self, gamestates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract raw action data using the existing shared pipeline workflow.
        This method properly syncs actions to gamestates using the 600ms window logic.
        """
        # Import the shared pipeline method
        from shared_pipeline.actions import extract_raw_action_data
        
        # Get all actions from memory
        all_actions = self._get_all_actions_in_memory()
        LOG.info(f"Found {len(all_actions)} actions in memory")
        
        if not all_actions:
            LOG.warning("No actions found in memory")
            # Return empty action data for each gamestate
            return [{
                'mouse_movements': [],
                'clicks': [],
                'key_presses': [],
                'key_releases': [],
                'scrolls': []
            } for _ in gamestates]
        
        # Convert in-memory actions to the format expected by the pipeline
        # The pipeline expects actions.csv format, so we need to create a temporary CSV
        import tempfile
        import pandas as pd
        
        # Create temporary actions.csv with the in-memory actions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_csv:
            # Convert actions to DataFrame format
            actions_df_data = []
            for action in all_actions:
                actions_df_data.append({
                    'timestamp': action.get('timestamp', 0),
                    'event_type': action.get('event_type', ''),
                    'x_in_window': action.get('x_in_window', 0),
                    'y_in_window': action.get('y_in_window', 0),
                    'btn': action.get('btn', ''),
                    'key': action.get('key', ''),
                    'scroll_dx': action.get('scroll_dx', 0),
                    'scroll_dy': action.get('scroll_dy', 0)
                })
            
            # Write to temporary CSV
            df = pd.DataFrame(actions_df_data)
            df.to_csv(temp_csv.name, index=False)
            temp_csv_path = temp_csv.name
        
        try:
            # Use the existing pipeline method to extract action sequences
            # This will properly sync actions to gamestates using the 600ms window
            raw_action_data = extract_raw_action_data(gamestates, temp_csv_path)
            LOG.info(f"Pipeline extracted action data for {len(raw_action_data)} gamestates")
            return raw_action_data
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_csv_path)
            except:
                pass
    
    def _save_actions_csv_from_pipeline_data(self, gamestates: List[Dict[str, Any]], 
                                           raw_action_data: List[Dict[str, Any]], 
                                           csv_path: Path) -> None:
        """
        Save actions to CSV file using the processed pipeline data.
        This ensures the CSV matches what the pipeline actually used.
        """
        import pandas as pd
        
        # Collect all actions from the processed pipeline data
        all_actions = []
        
        for gamestate_idx, (gamestate, action_data) in enumerate(zip(gamestates, raw_action_data)):
            gamestate_timestamp = gamestate.get('timestamp', 0)
            
            # Process each action type from the pipeline data
            for move in action_data.get('mouse_movements', []):
                all_actions.append({
                    'timestamp': move.get('timestamp', 0),
                    'event_type': 'move',
                    'x_in_window': move.get('x', 0),
                    'y_in_window': move.get('y', 0),
                    'btn': '',
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0
                })
            
            for click in action_data.get('clicks', []):
                all_actions.append({
                    'timestamp': click.get('timestamp', 0),
                    'event_type': 'click',
                    'x_in_window': click.get('x', 0),
                    'y_in_window': click.get('y', 0),
                    'btn': click.get('button', ''),
                    'key': '',
                    'scroll_dx': 0,
                    'scroll_dy': 0
                })
            
            for key_press in action_data.get('key_presses', []):
                all_actions.append({
                    'timestamp': key_press.get('timestamp', 0),
                    'event_type': 'key_press',
                    'x_in_window': 0,
                    'y_in_window': 0,
                    'btn': '',
                    'key': key_press.get('key', ''),
                    'scroll_dx': 0,
                    'scroll_dy': 0
                })
            
            for key_release in action_data.get('key_releases', []):
                all_actions.append({
                    'timestamp': key_release.get('timestamp', 0),
                    'event_type': 'key_release',
                    'x_in_window': 0,
                    'y_in_window': 0,
                    'btn': '',
                    'key': key_release.get('key', ''),
                    'scroll_dx': 0,
                    'scroll_dy': 0
                })
            
            for scroll in action_data.get('scrolls', []):
                all_actions.append({
                    'timestamp': scroll.get('timestamp', 0),
                    'event_type': 'scroll',
                    'x_in_window': 0,
                    'y_in_window': 0,
                    'btn': '',
                    'key': '',
                    'scroll_dx': scroll.get('dx', 0),
                    'scroll_dy': scroll.get('dy', 0)
                })
        
        # Save to CSV
        df = pd.DataFrame(all_actions)
        df.to_csv(csv_path, index=False)
        LOG.info(f"Saved actions CSV to {csv_path} with {len(all_actions)} actions")
    
    def _get_all_actions_in_memory(self) -> List[Dict[str, Any]]:
        """Get all actions currently in memory from the actions service"""
        if self.actions_service:
            return self.actions_service.actions
        return []

    # --- Simple accessors for the live UI ---------------------------------
    def get_last_gamestate_timestamp(self) -> Optional[int]:
        """
        Absolute timestamp (ms) of the most recent gamestate used to build T0.
        Returns None if the buffer is empty.
        """
        try:
            if self._gamestate_windows:
                gs = self._gamestate_windows[-1]
                ts = gs.get("timestamp")
                return int(ts) if ts is not None else None
        except Exception:
            pass
        return None

    # ---------------------- Offline Prep (normalize & sequences) ----------------------
    def prepare_session_training_data(self, session_dir: Path, out_dir: Path | None = None) -> Path:
        """
        Normalize a session and build training sequences.
        Inputs:  <session_dir>/features.csv, <session_dir>/actions.csv, <session_dir>/gamestates/*.json
        Outputs: <out_dir>/
          - features_raw.npy             (N, 128)
          - features_norm.npy            (N, 128)
          - X_gs_norm.npy                (S, 10, 128)
          - y_actions_norm_padded.npy    (S, 101, 8)        # targets at NEXT timestep
          - action_input_sequences_padded.npy (S, 10, 101, 8)  # actions for last 10 steps
          - meta.json
        Returns: Path to the prepared directory.
        """
        from pathlib import Path as _Path
        import numpy as np, pandas as pd, json
        # Prefer shared_pipeline imports if available; otherwise fall back to local modules.
        try:
            from shared_pipeline.io_offline import load_gamestates
            from shared_pipeline.sequences import create_temporal_sequences
        except Exception:
            from .io_offline import load_gamestates
            from .sequences import create_temporal_sequences
        # normalize_features / normalize_action_data / convert_raw_actions_to_tensors were imported at top
        session_dir = _Path(session_dir)
        out_dir = _Path(out_dir) if out_dir is not None else session_dir / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load features.csv -> (N,128)
        fcsv = pd.read_csv(session_dir / "features.csv")
        if "timestamp" in fcsv.columns:
            feat_mat = fcsv.drop(columns=["timestamp"]).to_numpy(dtype=float)
        else:
            feat_mat = fcsv.to_numpy(dtype=float)
        np.save(out_dir / "features_raw.npy", feat_mat.astype(np.float32))

        # 2) Normalize features (uses your mappings in data/05_mappings or data/features)
        #    The function itself finds/uses its default mapping path if not given.
        feat_norm = normalize_features(feat_mat)
        np.save(out_dir / "features_norm.npy", feat_norm.astype(np.float32))

        gamestates = load_gamestates(str(session_dir / "gamestates"))
        LOG.info(f"[prep] Loaded {len(gamestates)} gamestates")
        if gamestates:
            LOG.info(f"[prep] First gamestate keys: {list(gamestates[0].keys())}")
            if "timestamp" in gamestates[0]:
                LOG.info(f"[prep] First gamestate timestamp: {gamestates[0]['timestamp']}")

        # --- Canonicalize actions timing & schema ---
        # We maintain BOTH absolute and session-relative timestamps,
        # but downstream extraction uses a guaranteed-good CSV.
        import sys
        import os
        # Add parent directory to path to find timebase module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from timebase import SessionClock, coerce_action_columns, infer_session_start_abs_ms, ABS_THRESHOLD  # type: ignore

        actions_csv = session_dir / "actions.csv"
        actions_path_for_extraction = str(actions_csv)
        try:
            import pandas as pd
            # Load existing files
            _adf = pd.read_csv(actions_csv) if actions_csv.exists() else pd.DataFrame()
            _fcsv = pd.read_csv(session_dir / "features.csv") if (session_dir / "features.csv").exists() else pd.DataFrame()
            
            LOG.info(f"[prep] Actions CSV: {len(_adf)} rows, columns: {list(_adf.columns)}")
            if not _adf.empty:
                LOG.info(f"[prep] Actions CSV timestamp range: {_adf['timestamp'].min()} to {_adf['timestamp'].max()}")
                LOG.info(f"[prep] Actions CSV event types: {_adf['event_type'].value_counts().to_dict()}")
            
            LOG.info(f"[prep] Actions CSV columns: {list(_adf.columns)}")
            LOG.info(f"[prep] Actions CSV shape: {_adf.shape}")
            if not _adf.empty and "timestamp" in _adf.columns:
                _ts_sample = pd.to_numeric(_adf["timestamp"], errors="coerce").dropna()
                if len(_ts_sample) > 0:
                    LOG.info(f"[prep] Legacy timestamp sample: min={_ts_sample.min()}, max={_ts_sample.max()}, 95th percentile={_ts_sample.quantile(0.95)}")

            # Infer absolute session start from meta/gamestates/features
            try:
                import json as _json
                _meta = None
                _meta_path = session_dir / "meta.json"
                if _meta_path.exists():
                    _meta = _json.loads(_meta_path.read_text())
            except Exception:
                _meta = None
            _session_start_abs = infer_session_start_abs_ms(gamestates, meta=_meta, features_df=_fcsv)
            LOG.info(f"[prep] Inferred session_start_abs_ms: {_session_start_abs}")
            if _session_start_abs is None:
                raise RuntimeError("Cannot determine session_start_abs_ms from meta/gamestates/features.")

            _clock = SessionClock(session_start_abs_ms=_session_start_abs)

            # Normalize names / ensure required columns
            _adf = coerce_action_columns(_adf)

            # Derive session-relative & absolute timestamp columns
            # First, check if we have existing timestamp data and determine what it represents
            _ts_rel = None
            _ts_abs = None
            
            # Check if we already have the new columns
            if "t_session_ms" in _adf.columns and pd.to_numeric(_adf["t_session_ms"], errors="coerce").max() > 0:
                _ts_rel = pd.to_numeric(_adf["t_session_ms"], errors="coerce")
                _ts_abs = _clock.to_absolute(_ts_rel)
            elif "timestamp_abs_ms" in _adf.columns and pd.to_numeric(_adf["timestamp_abs_ms"], errors="coerce").max() > 0:
                _ts_abs = pd.to_numeric(_adf["timestamp_abs_ms"], errors="coerce")
                _ts_rel = _clock.to_session(_ts_abs)
            elif "timestamp" in _adf.columns and pd.to_numeric(_adf["timestamp"], errors="coerce").max() > 0:
                # Legacy timestamp - assume it's session-relative if it's small numbers
                _ts_legacy = pd.to_numeric(_adf["timestamp"], errors="coerce")
                if _ts_legacy.quantile(0.95) < ABS_THRESHOLD:
                    # Small numbers, treat as session-relative
                    _ts_rel = _ts_legacy
                    _ts_abs = _clock.to_absolute(_ts_rel)
                else:
                    # Large numbers, treat as absolute
                    _ts_abs = _ts_legacy
                    _ts_rel = _clock.to_session(_ts_abs)
            
            # Ensure we have both columns
            if _ts_rel is not None:
                _adf["t_session_ms"] = _ts_rel.astype("int64")
                LOG.info(f"[prep] Using existing session-relative timestamps: min={_ts_rel.min()}, max={_ts_rel.max()}")
            else:
                _adf["t_session_ms"] = 0
                LOG.info(f"[prep] No session-relative timestamps found, using 0")
                
            if _ts_abs is not None:
                _adf["timestamp_abs_ms"] = _ts_abs.astype("int64")
                LOG.info(f"[prep] Using existing absolute timestamps: min={_ts_abs.min()}, max={_ts_abs.max()}")
            else:
                _adf["timestamp_abs_ms"] = _session_start_abs
                LOG.info(f"[prep] No absolute timestamps found, using session_start_abs={_session_start_abs}")

            # Duplicate columns some extractors expect
            if "x" not in _adf.columns and "x_in_window" in _adf.columns: _adf["x"] = _adf["x_in_window"]
            if "y" not in _adf.columns and "y_in_window" in _adf.columns: _adf["y"] = _adf["y_in_window"]
            if "dx" not in _adf.columns and "scroll_dx" in _adf.columns: _adf["dx"] = _adf["scroll_dx"]
            if "dy" not in _adf.columns and "scroll_dy" in _adf.columns: _adf["dy"] = _adf["scroll_dy"]

            # Save canonical (session-relative) & an absolute-timestamp temp CSV for current extractor
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "actions_canonical.csv").write_text(_adf.to_csv(index=False))
            _adf_export = _adf.copy()
            _adf_export["timestamp"] = _adf_export["timestamp_abs_ms"]
            LOG.info(f"[prep] Export dataframe timestamp range: {_adf_export['timestamp'].min()} to {_adf_export['timestamp'].max()}")
            LOG.info(f"[prep] Export dataframe columns: {list(_adf_export.columns)}")
            LOG.info(f"[prep] Export dataframe sample (first 3 rows):")
            LOG.info(f"[prep] {_adf_export.head(3).to_string()}")
            _tmp_path = out_dir / "actions_abs_tmp.csv"
            _adf_export.to_csv(_tmp_path, index=False)
            actions_path_for_extraction = str(_tmp_path)
            LOG.info(f"[prep] Actions path for extraction: {actions_path_for_extraction}")
            LOG.info(f"[prep] Temp CSV has {len(_adf_export)} rows")

            # Sanity log: actions per 600ms window before first few gamestates
            try:
                _win = 600
                _ts_series = pd.to_numeric(_adf["t_session_ms"], errors="coerce").fillna(0).astype("int64")
                LOG.info(f"[prep] Final actions dataframe: {len(_adf)} rows, columns: {list(_adf.columns)}")
                LOG.info(f"[prep] Final t_session_ms range: {_ts_series.min()} to {_ts_series.max()}")
                for _i, _gs in enumerate(gamestates[:5]):
                    _t0 = int(_gs.get("timestamp", 0))
                    _t0 = (_t0 - _session_start_abs) if _t0 > ABS_THRESHOLD else _t0
                    _n = ((_ts_series >= (_t0 - _win)) & (_ts_series < _t0)).sum()
                    LOG.info(f"[prep] gs#{_i:02d} @ t={_t0}ms -> {_n} actions in previous {_win}ms")
            except Exception as __e:
                LOG.debug(f"Sanity log failed: {__e}")
        except Exception as _e:
            LOG.warning(f"Could not canonicalize actions.csv: {_e}")

        # Build raw action windows (absolute timestamps expected by extractor)
        from shared_pipeline.actions import extract_action_sequences as _extract_actions  # try shared
        try:
            LOG.info(f"[prep] Extracting actions with shared pipeline from {actions_path_for_extraction}")
            raw_actions = _extract_actions(gamestates, actions_file=actions_path_for_extraction)
            LOG.info(f"[prep] Shared pipeline extracted {len(raw_actions)} action windows")
        except Exception as e:
            LOG.warning(f"[prep] Shared pipeline failed: {e}, trying local")
            from .actions import extract_action_sequences as _extract_actions_local
            raw_actions = _extract_actions_local(gamestates, actions_file=actions_path_for_extraction)
            LOG.info(f"[prep] Local pipeline extracted {len(raw_actions)} action windows")

        # Guardrail: log empty windows before normalization
        try:
            _empty = 0
            for _w in raw_actions:
                try:
                    _empty += (len(_w) == 0)
                except Exception:
                    pass
            if _empty:
                LOG.warning(f"[prep] {_empty} empty action windows detected before normalization")
            
            # Debug: log raw actions details
            LOG.info(f"[prep] Raw actions: {len(raw_actions)} windows")
            if raw_actions:
                try:
                    LOG.info(f"[prep] First window: {len(raw_actions[0])} actions")
                    if len(raw_actions[0]) > 0:
                        LOG.info(f"[prep] First action sample: {raw_actions[0][0]}")
                except (IndexError, KeyError, TypeError) as e:
                    LOG.info(f"[prep] First window structure: {type(raw_actions[0])} - {raw_actions[0]}")
        except Exception:
            pass

        # 4) Convert extracted actions to normalized format, then normalize and convert to tensors
        from shared_pipeline.actions import convert_extracted_actions_to_normalized_format
        normalized_format_actions = convert_extracted_actions_to_normalized_format(raw_actions)
        norm_action_data = normalize_action_data(normalized_format_actions, feat_norm)
        LOG.info(f"[prep] Normalized action data: {len(norm_action_data)} windows")
        if norm_action_data:
            try:
                LOG.info(f"[prep] First normalized window: {len(norm_action_data[0])} actions")
                if len(norm_action_data[0]) > 0:
                    LOG.info(f"[prep] First normalized action sample: {norm_action_data[0][0]}")
            except (IndexError, KeyError, TypeError) as e:
                LOG.info(f"[prep] First normalized window structure: {type(norm_action_data[0])} - {norm_action_data[0]}")
        else:
            LOG.warning(f"[prep] norm_action_data is empty!")
        
        action_tensors = convert_raw_actions_to_tensors(norm_action_data, self._encoder)
        LOG.info(f"[prep] Action tensors: {len(action_tensors)} windows")
        if action_tensors:
            try:
                LOG.info(f"[prep] First tensor window: {len(action_tensors[0])} actions")
                if len(action_tensors[0]) > 0:
                    LOG.info(f"[prep] First tensor action sample: {action_tensors[0][0]}")
            except (IndexError, KeyError, TypeError) as e:
                LOG.info(f"[prep] First tensor window structure: {type(action_tensors[0])} - {action_tensors[0]}")
        else:
            LOG.warning(f"[prep] action_tensors is empty!")

        # 5) Build temporal sequences (10-step)
        LOG.info(f"[prep] Creating temporal sequences with feat_norm.shape={feat_norm.shape}, action_tensors={len(action_tensors)}")
        X_seq, y_seq, action_input_seq = create_temporal_sequences(
            feat_norm, action_tensors, sequence_length=10
        )
        np.save(out_dir / "X_gs_norm.npy", X_seq.astype(np.float32))
        
        # Debug: log temporal sequence details
        LOG.info(f"[prep] Temporal sequences: X={X_seq.shape}, y={len(y_seq)}, action_input={len(action_input_seq)}")
        if y_seq:
            LOG.info(f"[prep] First y_seq: {len(y_seq[0]) if y_seq[0] else 0} elements")
            if y_seq[0] and len(y_seq[0]) > 0:
                LOG.info(f"[prep] First y_seq sample: {y_seq[0][:5] if len(y_seq[0]) >= 5 else y_seq[0]}")
        else:
            LOG.warning(f"[prep] y_seq is empty!")
        
        if action_input_seq:
            LOG.info(f"[prep] First action_input_seq: {len(action_input_seq[0]) if action_input_seq[0] else 0} elements")
            if action_input_seq[0] and len(action_input_seq[0]) > 0:
                LOG.info(f"[prep] First action_input_seq sample: {action_input_seq[0][:5] if len(action_input_seq[0]) >= 5 else action_input_seq[0]}")
        else:
            LOG.warning(f"[prep] action_input_seq is empty!")

        # ==== Correct padding (shapes) ====
        # Infer per-action event width (E)
        def _infer_event_width(example):
            # example is a flattened list: [count, e1_f1,...,e1_fE, e2_f1,...]
            try:
                cnt = int(example[0]) if len(example) > 0 else 0
                rest = max(0, len(example) - 1)
                result = (rest // max(1, cnt)) if cnt > 0 else 8
                LOG.debug(f"[prep] _infer_event_width: example={example[:5] if len(example) >= 5 else example}, cnt={cnt}, rest={rest}, result={result}")
                return result
            except Exception as e:
                LOG.debug(f"[prep] _infer_event_width: exception {e}, returning default 8")
                return 8
        E = 8
        for ex in (y_seq[0:1] or []):
            if ex:
                E = _infer_event_width(ex)
                LOG.info(f"[prep] Inferred event width: E={E}")
            else:
                LOG.info(f"[prep] No examples in y_seq to infer event width, using default E={E}")
        MAX_EVENTS = 100

        def pad_target(flat_vec, max_events=MAX_EVENTS, e=E):
            """Pad a single 'next-step' target window to (max_events+1, e)."""
            out = np.zeros((max_events + 1, e), dtype=np.float32)
            if flat_vec is None or len(flat_vec) == 0:
                LOG.debug(f"[prep] pad_target: flat_vec is None or empty")
                return out
            cnt = int(flat_vec[0]) if len(flat_vec) > 0 else 0
            LOG.debug(f"[prep] pad_target: count={cnt}, flat_vec length={len(flat_vec)}")
            out[0, 0] = float(cnt)
            if cnt <= 0:
                LOG.debug(f"[prep] pad_target: count <= 0, returning zeros")
                return out
            data = np.asarray(flat_vec[1:], dtype=np.float32)
            # Copy at most max_events actions
            m = min(cnt, max_events)
            needed = m * e
            if data.size < needed:
                # If encoder emitted fewer elements than expected, trim m
                m = data.size // e
                needed = m * e
            if m > 0:
                out[1:1 + m, :] = data[:needed].reshape(m, e)
                LOG.debug(f"[prep] pad_target: copied {m} actions, data shape={data[:needed].shape}")
            else:
                LOG.debug(f"[prep] pad_target: no actions to copy, m={m}")
            return out

        def pad_seq10(seq10, max_events=MAX_EVENTS, e=E):
            """Pad a 10-step list of flattened windows to (10, max_events+1, e)."""
            arr = np.zeros((10, max_events + 1, e), dtype=np.float32)
            if not seq10:
                LOG.debug(f"[prep] pad_seq10: seq10 is empty, returning zeros")
                return arr
            T = min(10, len(seq10))
            LOG.debug(f"[prep] pad_seq10: processing {T} timesteps")
            for t in range(T):
                arr[t] = pad_target(seq10[t], max_events, e)
            return arr

        # Targets (NEXT timestep): (S, 101, E)
        LOG.info(f"[prep] Padding {len(y_seq)} y_seq elements with E={E}, MAX_EVENTS={MAX_EVENTS}")
        y_actions_norm_padded = np.stack(
            [pad_target(v) for v in y_seq], axis=0
        )
        LOG.info(f"[prep] Final y_actions_norm_padded shape: {y_actions_norm_padded.shape}")
        LOG.info(f"[prep] Non-zero counts in first sequence: {np.count_nonzero(y_actions_norm_padded[0])}")
        np.save(out_dir / "y_actions_norm_padded.npy", y_actions_norm_padded)

        # Action inputs for last 10 steps: (S, 10, 101, E)
        LOG.info(f"[prep] Padding {len(action_input_seq)} action_input_seq elements")
        action_input_sequences_padded = np.stack(
            [pad_seq10(seq) for seq in action_input_seq], axis=0
        )
        LOG.info(f"[prep] Final action_input_sequences_padded shape: {action_input_sequences_padded.shape}")
        LOG.info(f"[prep] Non-zero counts in first sequence: {np.count_nonzero(action_input_sequences_padded[0])}")
        np.save(out_dir / "action_input_sequences_padded.npy", action_input_sequences_padded)

        # 6) Metadata
        meta = {
            "n_samples": int(feat_mat.shape[0]),
            "n_sequences": int(X_seq.shape[0]),
            "n_features": int(feat_mat.shape[1]),
            "session_start_time": int(self.session_start_time) if self.session_start_time else None,
            "paths": {
                "features_raw": str(out_dir / "features_raw.npy"),
                "features_norm": str(out_dir / "features_norm.npy"),
                "X_gs_norm": str(out_dir / "X_gs_norm.npy"),
                "y_actions_norm_padded": str(out_dir / "y_actions_norm_padded.npy"),
                "action_input_sequences_padded": str(out_dir / "action_input_sequences_padded.npy"),
            }
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        LOG.info(f"Prepared training data at {out_dir} (sequences={meta['n_sequences']})")
        return out_dir
