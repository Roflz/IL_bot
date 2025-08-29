"""
Action Decoder Service

Wraps ActionEncoder to provide consistent action type/button/key labels.
"""

from typing import Dict, Any, List, Optional
from ilbot.pipeline.shared_pipeline.encodings import ActionEncoder


class ActionDecoder:
    """Service for decoding action data into human-readable labels."""
    
    def __init__(self, action_encoder: Optional[ActionEncoder] = None):
        """
        Initialize the action decoder.
        
        Args:
            action_encoder: Optional ActionEncoder instance, will create one if not provided
        """
        if action_encoder is None:
            self.action_encoder = ActionEncoder()
        else:
            self.action_encoder = action_encoder

        # Build reverse lookup tables for ids → names
        self._type_id_to_name = {}
        self._button_id_to_name = {}
        self._key_id_to_name = {}

        # Action types in ActionEncoder are name → id
        if isinstance(self.action_encoder.action_types, dict):
            for name, idx in self.action_encoder.action_types.items():
                self._type_id_to_name[int(idx)] = str(name)

        # Button types in ActionEncoder are name → id
        if isinstance(self.action_encoder.button_types, dict):
            for name, idx in self.action_encoder.button_types.items():
                self._button_id_to_name[int(idx)] = str(name)

        # Keys in ActionEncoder are key-string → id (may be empty initially)
        if isinstance(self.action_encoder.key_encodings, dict):
            for key_name, idx in self.action_encoder.key_encodings.items():
                self._key_id_to_name[int(idx)] = str(key_name)
    
    def get_action_type_name(self, action_type: int) -> str:
        """
        Get human-readable name for action type.
        
        Args:
            action_type: Action type integer
            
        Returns:
            Human-readable action type name
        """
        # Prefer reverse map if available
        if action_type in self._type_id_to_name:
            return self._type_id_to_name[action_type]
        # Fallback to common defaults
        default_map = {0: "move", 1: "click", 2: "key", 3: "scroll"}
        return default_map.get(action_type, f"Action_{action_type}")
    
    def get_button_name(self, button: int) -> str:
        """
        Get human-readable name for button.
        
        Args:
            button: Button integer
            
        Returns:
            Human-readable button name
        """
        if button in self._button_id_to_name:
            return self._button_id_to_name[button]
        default_map = {0: "", 1: "left", 2: "right", 3: "middle"}
        return default_map.get(button, f"Button_{button}")
    
    def get_key_name(self, key: int) -> str:
        """
        Get human-readable name for key.
        
        Args:
            key: Key integer
            
        Returns:
            Human-readable key name
        """
        if key in self._key_id_to_name:
            return self._key_id_to_name[key]
        return f"Key_{key}"
    
    def decode_action_tensor(self, action_tensor: List[float]) -> Dict[str, Any]:
        """
        Decode a flattened action tensor into structured data.
        
        Args:
            action_tensor: Flattened action tensor [count, Δt_ms, type, x, y, button, key, scroll_dx, scroll_dy, ...]
            
        Returns:
            Dictionary with decoded action information
        """
        if not action_tensor or len(action_tensor) < 1:
            return {"count": 0, "actions": []}
        
        action_count = int(action_tensor[0])
        actions = []
        
        for i in range(action_count):
            base_idx = 1 + i * 8  # Each action has 8 elements
            if base_idx + 7 < len(action_tensor):
                action = {
                    "timestamp": action_tensor[base_idx],
                    "type": self.get_action_type_name(int(action_tensor[base_idx + 1])),
                    "x": int(action_tensor[base_idx + 2]),
                    "y": int(action_tensor[base_idx + 3]),
                    "button": self.get_button_name(int(action_tensor[base_idx + 4])),
                    "key": self.get_key_name(int(action_tensor[base_idx + 5])),
                    "scroll_dx": int(action_tensor[base_idx + 6]),
                    "scroll_dy": int(action_tensor[base_idx + 7])
                }
                actions.append(action)
        
        return {
            "count": action_count,
            "actions": actions
        }
    
    def format_action_summary(self, action_tensor: List[float]) -> str:
        """
        Format action tensor into a human-readable summary.
        
        Args:
            action_tensor: Flattened action tensor
            
        Returns:
            Formatted action summary string
        """
        decoded = self.decode_action_tensor(action_tensor)
        
        if decoded["count"] == 0:
            return "No actions"
        
        summary_parts = [f"{decoded['count']} actions:"]
        
        for i, action in enumerate(decoded["actions"]):
            action_str = f"  {i+1}. {action['type']}"
            if action['x'] != 0 or action['y'] != 0:
                action_str += f" at ({action['x']}, {action['y']})"
            if action['button'] not in ("", "Button_0"):
                action_str += f" [{action['button']}]"
            if action['key'] not in ("Key_0", "", None):
                action_str += f" [{action['key']}]"
            if action['scroll_dx'] != 0 or action['scroll_dy'] != 0:
                action_str += f" scroll({action['scroll_dx']}, {action['scroll_dy']})"
            
            summary_parts.append(action_str)
        
        return "\n".join(summary_parts)
    
    def get_action_type_count(self, action_tensor: List[float]) -> Dict[str, int]:
        """
        Get count of each action type in the tensor.
        
        Args:
            action_tensor: Flattened action tensor
            
        Returns:
            Dictionary mapping action type names to counts
        """
        decoded = self.decode_action_tensor(action_tensor)
        type_counts = {}
        
        for action in decoded["actions"]:
            action_type = action["type"]
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
        
        return type_counts
