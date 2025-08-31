"""
Action Processing Module for OSRS Bot Imitation Learning

This module processes action sequences and converts them to training format,
preserving the exact behavior from legacy phase1_data_preparation.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .encodings import ActionEncoder











def extract_raw_action_data(gamestates, actions_csv_path, align_to_gamestates: bool = False):
    """
    Build per-gamestate action buckets. If align_to_gamestates=True, the "timestamp"
    field stored on each atomic action is made **relative to the earliest gamestate ts**,
    so features['timestamp'] and action['timestamp'] share the same zero-point.
    """
    import pandas as pd
    df = pd.read_csv(actions_csv_path)
    
    # Find session start time
    if align_to_gamestates:
        session_start_time = int(min(gs.get("timestamp", 0) for gs in gamestates))
    else:
        session_start_time = int(df["timestamp"].min())
    
    # Build action buckets for each gamestate
    buckets = []
    for i, gs in enumerate(gamestates):
        gs_ts = int(gs.get("timestamp", 0))
        window_start = gs_ts - 600  # 600ms before gamestate
        
        # Get actions in window
        relevant = df[(df["timestamp"] >= window_start) & (df["timestamp"] < gs_ts)]
        
        # Separate by action type
        mouse_movements = []
        clicks = []
        key_presses = []
        key_releases = []
        scrolls = []
        
        for _, row in relevant.iterrows():
            abs_ts = int(row.timestamp)
            rel_ts = abs_ts - session_start_time
            # store both, to make verification trivial later
            action_rec = {"timestamp": float(rel_ts),
                          "absolute_timestamp": abs_ts}
            
            if row["event_type"] == "move":
                action_rec.update({
                    "x": int(row.get("x_in_window", 0)),
                    "y": int(row.get("y_in_window", 0))
                })
                mouse_movements.append(action_rec)
            elif row["event_type"] == "click":
                action_rec.update({
                    "x": int(row.get("x_in_window", 0)),
                    "y": int(row.get("y_in_window", 0)),
                    "button": str(row.get("btn", ""))
                })
                clicks.append(action_rec)
            elif row["event_type"] == "key_press":
                action_rec.update({
                    "key": str(row.get("key", "")),
                    "x": int(row.get("x_in_window", 0)),
                    "y": int(row.get("y_in_window", 0))
                })
                key_presses.append(action_rec)
            elif row["event_type"] == "key_release":
                action_rec.update({
                    "key": str(row.get("key", "")),
                    "x": int(row.get("x_in_window", 0)),
                    "y": int(row.get("y_in_window", 0))
                })
                key_releases.append(action_rec)
            elif row["event_type"] == "scroll":
                action_rec.update({
                    "dy": int(row.get("scroll_dy", 0)),
                    "x": int(row.get("x_in_window", 0)),
                    "y": int(row.get("y_in_window", 0))
                })
                scrolls.append(action_rec)
        
        buckets.append({
            'mouse_movements': mouse_movements,
            'clicks': clicks,
            'key_presses': key_presses,
            'key_releases': key_releases,
            'scrolls': scrolls
        })
    
    # attach the source gamestate absolute ts to each bucket (verification + UI)
    for i, gs in enumerate(gamestates):
        for key in ("mouse_movements", "clicks", "key_presses", "key_releases", "scrolls"):
            bucket = buckets[i].get(key, [])
        buckets[i]["gamestate_timestamp"] = int(gs.get("timestamp", 0))
    return buckets




















def create_v2_actions_directly(raw_action_data: List[Dict], time_div: float = 1000.0, 
                              time_clip: Optional[float] = None, time_transform: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    """
    Create V2 actions directly from raw action data without going through legacy 8-feature format.
    
    Args:
        raw_action_data: List of action data dictionaries per gamestate
        time_div: Time division factor (default: 1000.0 for ms to seconds)
        time_clip: Optional time clipping in seconds
        time_transform: Time transformation ("none" or "log1p")
        
    Returns:
        Tuple of (actions_v2, valid_mask) where:
        - actions_v2: Array of shape (n_gamestates, max_actions, 7) with V2 format
        - valid_mask: Array of shape (n_gamestates, max_actions) indicating valid actions
    """
    import numpy as np
    
    n_gamestates = len(raw_action_data)
    max_actions = 100  # Fixed maximum actions per gamestate
    
    # Initialize output arrays
    actions_v2 = np.zeros((n_gamestates, max_actions, 7), dtype=np.float32)
    valid_mask = np.zeros((n_gamestates, max_actions), dtype=np.bool_)
    
    for gamestate_idx, action_bucket in enumerate(raw_action_data):
        # Collect all actions for this gamestate
        all_actions = []
        
        # Process mouse movements
        for action in action_bucket.get('mouse_movements', []):
            all_actions.append({
                'type': 'move',
                'timestamp': action.get('timestamp', 0.0),
                'x': action.get('x', 0),
                'y': action.get('y', 0),
                'button': 0,  # No button for moves
                'key_action': 0,  # No key action for moves
                'key_id': 0,  # No key ID for moves
                'scroll_y': 0  # No scroll for moves
            })
        
        # Process clicks
        for action in action_bucket.get('clicks', []):
            button_map = {'left': 1, 'right': 2, 'middle': 3, 'none': 0}
            button_value = action.get('button', 0)
            if isinstance(button_value, str):
                button_value = button_map.get(button_value.lower(), 0)
            
            all_actions.append({
                'type': 'click',
                'timestamp': action.get('timestamp', 0.0),
                'x': action.get('x', 0),
                'y': action.get('y', 0),
                'button': button_value,
                'key_action': 0,  # No key action for clicks
                'key_id': 0,  # No key ID for clicks
                'scroll_y': 0  # No scroll for clicks
            })
        
        # Process key presses
        for action in action_bucket.get('key_presses', []):
            # Convert key to numeric code if it's a string
            key_value = action.get('key', 0)
            if isinstance(key_value, str):
                key_map = {'w': 87, 'a': 65, 's': 83, 'd': 68, 'space': 32, 'enter': 13, 'escape': 27}
                key_value = key_map.get(key_value.lower(), 0)
            
            all_actions.append({
                'type': 'key_press',
                'timestamp': action.get('timestamp', 0.0),
                'x': action.get('x', 0),
                'y': action.get('y', 0),
                'button': 0,  # No button for keys
                'key_action': 1,  # Press
                'key_id': key_value,
                'scroll_y': 0  # No scroll for keys
            })
        
        # Process key releases
        for action in action_bucket.get('key_releases', []):
            # Convert key to numeric code if it's a string
            key_value = action.get('key', 0)
            if isinstance(key_value, str):
                key_map = {'w': 87, 'a': 65, 's': 83, 'd': 68, 'space': 32, 'enter': 13, 'escape': 27}
                key_value = key_map.get(key_value.lower(), 0)
            
            all_actions.append({
                'type': 'key_release',
                'timestamp': action.get('timestamp', 0.0),
                'x': action.get('x', 0),
                'y': action.get('y', 0),
                'button': 0,  # No button for keys
                'key_action': 2,  # Release
                'key_id': key_value,
                'scroll_y': 0  # No scroll for keys
            })
        
        # Process scrolls
        for action in action_bucket.get('scrolls', []):
            scroll_dy = action.get('dy', 0)
            # Map scroll direction: -1=down, 0=none, +1=up
            scroll_y = -1 if scroll_dy < 0 else (1 if scroll_dy > 0 else 0)
            
            all_actions.append({
                'type': 'scroll',
                'timestamp': action.get('timestamp', 0.0),
                'x': action.get('x', 0),
                'y': action.get('y', 0),
                'button': 0,  # No button for scrolls
                'key_action': 0,  # No key action for scrolls
                'key_id': 0,  # No key ID for scrolls
                'scroll_y': scroll_y
            })
        
        # Sort actions by timestamp
        all_actions.sort(key=lambda x: x['timestamp'])
        
        # Fill the actions array for this gamestate
        for action_idx, action in enumerate(all_actions[:max_actions]):
            # Calculate time delta (time until next action)
            if action_idx < len(all_actions) - 1:
                next_timestamp = all_actions[action_idx + 1]['timestamp']
                time_delta = max(0.0, next_timestamp - action['timestamp'])
            else:
                # For the last action, use a default delta
                time_delta = 0.1  # 100ms default
            
            # Convert time delta to seconds and apply transformations
            time_delta_sec = time_delta / time_div
            if time_clip is not None:
                time_delta_sec = min(time_delta_sec, time_clip)
            
            if time_transform == "log1p":
                time_delta_sec = np.log1p(time_delta_sec)
            
            # Store in V2 format: [time, x, y, click, key_action, key_id, scroll_y]
            actions_v2[gamestate_idx, action_idx] = [
                time_delta_sec,           # time delta in seconds
                float(action['x']),       # x coordinate
                float(action['y']),       # y coordinate
                float(action['button']),  # button code (0=none, 1=left, 2=right, 3=middle)
                float(action['key_action']), # key action (0=none, 1=press, 2=release)
                float(action['key_id']),  # key ID
                float(action['scroll_y']) # scroll direction (-1=down, 0=none, +1=up)
            ]
            
            # Mark this action as valid
            valid_mask[gamestate_idx, action_idx] = True
    
    return actions_v2, valid_mask
