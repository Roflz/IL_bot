"""
Action Encodings Module for OSRS Bot Imitation Learning

This module derives action encodings from existing training data to ensure
consistency between offline training and live execution. It preserves the
exact encoding scheme used in the legacy phase1_data_preparation.py.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict


class ActionEncoder:
    """
    Encoder for action data that derives mappings from existing training targets.
    
    This class preserves the exact encoding scheme from the legacy code:
    - Action types: 0=move, 1=click, 2=key, 3=scroll
    - Button types: 0=none, 1=left, 2=right, 3=middle
    - Key encodings: Derived from KeyboardKeyMapper
    - Scroll deltas: Preserved as original pixel values
    """
    
    def __init__(self):
        # Action type encodings (fixed from legacy code)
        self.action_types = {
            'move': 0,
            'click': 1, 
            'key_press': 2,
            'key_release': 2,  # Both press and release use type 2
            'scroll': 3
        }
        
        # Button type encodings (fixed from legacy code)
        self.button_types = {
            '': 0,      # No button
            'left': 1,  # Left click
            'right': 2, # Right click
            'middle': 3 # Middle click
        }
        
        # Key encodings will be derived from data
        self.key_encodings = {}
        
        # Track all unique values seen in training data
        self.seen_action_types = set()
        self.seen_button_types = set()
        self.seen_key_values = set()
        self.seen_scroll_values = set()
    
    def derive_encodings_from_raw_actions(self, raw_action_data: List[Dict]) -> None:
        """
        Derive encodings from raw action data when training targets don't exist.
        
        Args:
            raw_action_data: List of raw action data dictionaries
        """
        print("Deriving action encodings from raw action data...")
        
        # Analyze raw action data to extract all unique values
        for gamestate_actions in raw_action_data:
            if not gamestate_actions:
                continue
            
            # Handle mouse movements
            for movement in gamestate_actions.get('mouse_movements', []):
                self.seen_action_types.add(self.action_types['move'])
            
            # Handle clicks
            for click in gamestate_actions.get('clicks', []):
                self.seen_action_types.add(self.action_types['click'])
                button = click.get('button', '')
                if button in self.button_types:
                    self.seen_button_types.add(self.button_types[button])
            
            # Handle key presses/releases
            for key_action in gamestate_actions.get('key_presses', []):
                self.seen_action_types.add(self.action_types['key_press'])
                key = key_action.get('key', '')
                if key:
                    if key not in self.key_encodings:
                        self.key_encodings[key] = len(self.key_encodings)
                    self.seen_key_values.add(self.key_encodings[key])
            
            for key_action in gamestate_actions.get('key_releases', []):
                self.seen_action_types.add(self.action_types['key_press'])  # Same encoding
                key = key_action.get('key', '')
                if key:
                    if key not in self.key_encodings:
                        self.key_encodings[key] = len(self.key_encodings)
                    self.seen_key_values.add(self.key_encodings[key])
            
            # Handle scrolls
            for scroll in gamestate_actions.get('scrolls', []):
                self.seen_action_types.add(self.action_types['scroll'])
                scroll_dx = scroll.get('dx', 0)
                scroll_dy = scroll.get('dy', 0)
                self.seen_scroll_values.add(scroll_dx)
                self.seen_scroll_values.add(scroll_dy)
        
        print(f"Derived encodings from raw actions:")
        print(f"  Action types: {sorted(list(self.seen_action_types))}")
        print(f"  Button types: {sorted(list(self.seen_button_types))}")
        print(f"  Key encodings: {len(self.key_encodings)} unique keys")
        if self.seen_scroll_values:
            print(f"  Scroll range: {min(self.seen_scroll_values)} to {max(self.seen_scroll_values)}")
        else:
            print(f"  Scroll range: No scroll actions found")

    def derive_encodings_from_data(self, action_targets_file: str = "data/training_data/action_targets.json") -> None:
        """
        Derive encodings from existing training targets to ensure consistency.
        
        Args:
            action_targets_file: Path to action_targets.json
            
        Raises:
            FileNotFoundError: If action targets file doesn't exist
        """
        targets_path = Path(action_targets_file)
        if not targets_path.exists():
            raise FileNotFoundError(f"Action targets file not found: {targets_path}")
        
        print("Deriving action encodings from existing training data...")
        
        with open(targets_path, 'r') as f:
            action_targets = json.load(f)
        
        # Analyze action targets to extract all unique values
        for target_sequence in action_targets:
            if len(target_sequence) < 2:  # Must have at least action count + 1 action
                continue
            
            action_count = int(target_sequence[0])
            
            # Each action has 8 features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
            for i in range(1, len(target_sequence), 8):
                if i + 7 < len(target_sequence):  # Ensure we have all 8 features
                    # Action type (index 1, 9, 17, etc.)
                    action_type = int(target_sequence[i + 1])
                    self.seen_action_types.add(action_type)
                    
                    # Button (index 4, 12, 20, etc.)
                    button = int(target_sequence[i + 4])
                    self.seen_button_types.add(button)
                    
                    # Key (index 5, 13, 21, etc.)
                    key = int(target_sequence[i + 5])
                    self.seen_key_values.add(key)
                    
                    # Scroll deltas (indices 6, 7, 14, 15, etc.)
                    scroll_dx = int(target_sequence[i + 6])
                    scroll_dy = int(target_sequence[i + 7])
                    self.seen_scroll_values.add(scroll_dx)
                    self.seen_scroll_values.add(scroll_dy)
        
        print(f"Derived encodings from {len(action_targets)} action sequences:")
        print(f"  - Action types: {sorted(self.seen_action_types)}")
        print(f"  - Button types: {sorted(self.seen_button_types)}")
        print(f"  - Key values: {sorted(self.seen_key_values)}")
        print(f"  - Scroll values: {sorted(self.seen_scroll_values)}")
        
        # Validate that our derived encodings match expected values
        self._validate_derived_encodings()
    
    def _validate_derived_encodings(self) -> None:
        """Validate that derived encodings match expected values from legacy code."""
        expected_action_types = {0, 1, 2, 3}
        if not self.seen_action_types.issubset(expected_action_types):
            print(f"Warning: Unexpected action types found: {self.seen_action_types - expected_action_types}")
        
        expected_button_types = {0, 1, 2, 3}
        if not self.seen_button_types.issubset(expected_button_types):
            print(f"Warning: Unexpected button types found: {self.seen_button_types - expected_button_types}")
        
        # Key values can vary widely due to KeyboardKeyMapper
        if not self.seen_key_values:
            print("Warning: No key values found in training data")
    
    def encode_action_type(self, event_type: str) -> int:
        """
        Encode action event type to integer.
        
        Args:
            event_type: String event type ('move', 'click', 'key_press', 'key_release', 'scroll')
            
        Returns:
            Encoded action type integer
            
        Raises:
            ValueError: If event_type is not recognized
        """
        if event_type not in self.action_types:
            raise ValueError(f"Unknown event type: {event_type}")
        
        return self.action_types[event_type]
    
    def encode_button(self, button: str) -> int:
        """
        Encode mouse button to integer.
        
        Args:
            button: String button type ('', 'left', 'right', 'middle')
            
        Returns:
            Encoded button integer
        """
        return self.button_types.get(button, 0)
    
    def encode_key(self, key: str) -> int:
        """
        Encode keyboard key to integer using KeyboardKeyMapper.
        
        Args:
            key: String key representation
            
        Returns:
            Encoded key integer
        """
        # Import here to avoid circular imports
        try:
            from utils.key_mapper import KeyboardKeyMapper
            return int(KeyboardKeyMapper.map_key_to_number(key))
        except ImportError:
            # Fallback encoding if key_mapper not available
            if not key:
                return 0
            return hash(key) % 10000
    
    def encode_action_sequence(self, actions: List[Dict]) -> List[float]:
        """
        Encode a sequence of actions to the training format.
        
        Args:
            actions: List of action dictionaries with keys:
                    - timestamp: float (relative to window start)
                    - event_type: str
                    - x: float (screen coordinate)
                    - y: float (screen coordinate)
                    - button: str (for clicks)
                    - key: str (for key events)
                    - scroll_dx: float (for scrolls)
                    - scroll_dy: float (for scrolls)
        
        Returns:
            Flattened action sequence in training format:
            [action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]
        """
        if not actions:
            return [0]  # No actions
        
        # Sort actions by timestamp
        sorted_actions = sorted(actions, key=lambda a: a.get('timestamp', 0))
        
        # Start with action count
        encoded_sequence = [len(sorted_actions)]
        
        # Encode each action
        for action in sorted_actions:
            # Timestamp (preserve as-is, already relative)
            encoded_sequence.append(float(action.get('timestamp', 0)))
            
            # Action type
            event_type = action.get('event_type', 'move')
            encoded_sequence.append(float(self.encode_action_type(event_type)))
            
            # Coordinates
            encoded_sequence.append(float(action.get('x', 0)))
            encoded_sequence.append(float(action.get('y', 0)))
            
            # Button
            button = action.get('button', '')
            encoded_sequence.append(float(self.encode_button(button)))
            
            # Key
            key = action.get('key', '')
            encoded_sequence.append(float(self.encode_key(key)))
            
            # Scroll deltas
            encoded_sequence.append(float(action.get('scroll_dx', 0)))
            encoded_sequence.append(float(action.get('scroll_dy', 0)))
        
        return encoded_sequence
    
    def decode_action_sequence(self, encoded_sequence: List[float]) -> List[Dict]:
        """
        Decode a training format action sequence back to structured format.
        
        Args:
            encoded_sequence: Flattened action sequence from training data
            
        Returns:
            List of action dictionaries
        """
        if len(encoded_sequence) < 1:
            return []
        
        action_count = int(encoded_sequence[0])
        if action_count == 0:
            return []
        
        actions = []
        
        # Each action has 8 features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
        for i in range(1, len(encoded_sequence), 8):
            if i + 7 < len(encoded_sequence):
                action = {
                    'timestamp': encoded_sequence[i],
                    'type': int(encoded_sequence[i + 1]),
                    'x': encoded_sequence[i + 2],
                    'y': encoded_sequence[i + 3],
                    'button': int(encoded_sequence[i + 4]),
                    'key': int(encoded_sequence[i + 5]),
                    'scroll_dx': encoded_sequence[i + 6],
                    'scroll_dy': encoded_sequence[i + 7]
                }
                actions.append(action)
        
        return actions
    
    def get_encoding_summary(self) -> Dict:
        """
        Get a summary of all encodings for debugging and validation.
        
        Returns:
            Dictionary containing encoding information
        """
        return {
            'action_types': self.action_types,
            'button_types': self.button_types,
            'seen_action_types': sorted(list(self.seen_action_types)),
            'seen_button_types': sorted(list(self.seen_button_types)),
            'seen_key_values': sorted(list(self.seen_key_values)),
            'seen_scroll_values': sorted(list(self.seen_scroll_values)),
            'encoding_scheme': {
                'description': 'Derived from existing training targets to ensure consistency',
                'action_format': '[count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]',
                'feature_count_per_action': 8,
                'coordinate_preservation': 'Screen coordinates preserved as original pixel values',
                'timestamp_preservation': 'Timestamps preserved as relative to window start'
            }
        }


def derive_encodings_from_raw_actions(raw_action_data: List[Dict]) -> ActionEncoder:
    """
    Convenience function to create and populate an ActionEncoder from raw action data.
    
    Args:
        raw_action_data: List of raw action data dictionaries
        
    Returns:
        Configured ActionEncoder instance
    """
    encoder = ActionEncoder()
    encoder.derive_encodings_from_raw_actions(raw_action_data)
    return encoder


def derive_encodings_from_data(action_targets_file: str = "data/training_data/action_targets.json") -> ActionEncoder:
    """
    Convenience function to create and populate an ActionEncoder from training data.
    
    Args:
        action_targets_file: Path to action_targets.json
        
    Returns:
        Configured ActionEncoder instance
    """
    encoder = ActionEncoder()
    encoder.derive_encodings_from_data(action_targets_file)
    return encoder


def validate_action_encodings(encoder: ActionEncoder, action_targets_file: str = "data/training_data/action_targets.json") -> bool:
    """
    Validate that an ActionEncoder can reproduce the exact encodings from training data.
    
    Args:
        encoder: ActionEncoder instance to validate
        action_targets_file: Path to action_targets.json
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        targets_path = Path(action_targets_file)
        if not targets_path.exists():
            print(f"Warning: Cannot validate encodings - file not found: {targets_path}")
            return False
        
        with open(targets_path, 'r') as f:
            action_targets = json.load(f)
        
        # Test encoding/decoding round-trip on a few samples
        for i, target_sequence in enumerate(action_targets[:5]):  # Test first 5 sequences
            if len(target_sequence) < 2:
                continue
            
            # Decode the training format
            decoded_actions = encoder.decode_action_sequence(target_sequence)
            
            # Re-encode
            re_encoded = encoder.encode_action_sequence(decoded_actions)
            
            # Compare (allowing for float precision differences)
            if len(re_encoded) != len(target_sequence):
                print(f"Validation failed: sequence {i} length mismatch")
                return False
            
            for j, (orig, re_enc) in enumerate(zip(target_sequence, re_encoded)):
                if abs(orig - re_enc) > 1e-10:
                    print(f"Validation failed: sequence {i}, position {j}: {orig} != {re_enc}")
                    return False
        
        print("Action encoding validation passed ✓")
        return True
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        return False
