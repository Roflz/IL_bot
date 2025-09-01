#!/usr/bin/env python3
"""
Human Behavior Analyzer for OSRS
Analyzes mouse movement, click patterns, keyboard patterns, and scroll patterns
Links them to game actions and provides insights for mechanical bot development
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

@dataclass
class MousePattern:
    """Represents a mouse movement pattern"""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    distance: float
    duration: float
    speed: float
    direction: float
    pattern_type: str  # "click", "drag", "hover", "move"

@dataclass
class ClickPattern:
    """Represents a click pattern"""
    x: float
    y: float
    button: int  # 1=left, 2=right, 3=middle
    timestamp: float
    context: str  # "bank", "inventory", "object", "ground"
    success: bool

@dataclass
class KeyboardPattern:
    """Represents a keyboard pattern"""
    key_id: int
    action: str  # "press", "release"
    timestamp: float
    duration: float
    context: str
    frequency: int

@dataclass
class ScrollPattern:
    """Represents a scroll pattern"""
    direction: int  # -1=up, 1=down
    timestamp: float
    context: str
    frequency: int

@dataclass
class GameAction:
    """Represents a complete game action with context"""
    timestamp: float
    action_type: str  # "click", "key", "scroll", "move"
    coordinates: Tuple[float, float]
    context: str
    player_state: Dict[str, Any]
    inventory_state: Dict[str, Any]
    nearby_objects: List[str]
    success_indicators: List[str]

class HumanBehaviorAnalyzer:
    """
    Analyzes human behavior patterns from OSRS gameplay data
    Provides insights for mechanical bot development
    """
    
    def __init__(self, data_dir: str = "data/recording_sessions"):
        self.data_dir = Path(data_dir)
        self.analysis_results = {}
        
        # Pattern storage
        self.mouse_patterns: List[MousePattern] = []
        self.click_patterns: List[ClickPattern] = []
        self.keyboard_patterns: List[KeyboardPattern] = []
        self.scroll_patterns: List[ScrollPattern] = []
        self.game_actions: List[GameAction] = []
        
        # Analysis results
        self.behavior_insights = {}
        self.pattern_clusters = {}
        self.success_correlations = {}
        
    def analyze_session(self, session_id: str) -> Dict:
        """
        Analyze a complete recording session
        """
        print(f"🔍 Analyzing session: {session_id}")
        print("=" * 60)
        
        session_dir = self.data_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        # Load session data
        session_data = self._load_session_data(session_dir)
        
        # Perform comprehensive analysis
        analysis = {
            'session_info': self._analyze_session_info(session_data),
            'mouse_analysis': self._analyze_mouse_patterns(session_data),
            'click_analysis': self._analyze_click_patterns(session_data),
            'keyboard_analysis': self._analyze_keyboard_patterns(session_data),
            'scroll_analysis': self._analyze_scroll_patterns(session_data),
            'game_context_analysis': self._analyze_game_context(session_data),
            'behavior_patterns': self._identify_behavior_patterns(session_data),
            'success_metrics': self._analyze_success_metrics(session_data),
            'bot_development_insights': self._generate_bot_insights(session_data)
        }
        
        # Save analysis results
        self._save_analysis_results(analysis, session_id)
        
        # Print key insights
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _load_session_data(self, session_dir: Path) -> Dict:
        """
        Load all relevant data from a session directory
        """
        data = {}
        
        # Load action targets
        action_targets_path = session_dir / "06_final_training_data" / "action_targets.npy"
        if action_targets_path.exists():
            data['action_targets'] = np.load(action_targets_path)
            print(f"✅ Loaded action targets: {data['action_targets'].shape}")
        
        # Load gamestate sequences
        gamestates_path = session_dir / "06_final_training_data" / "gamestate_sequences.npy"
        if gamestates_path.exists():
            data['gamestates'] = np.load(gamestates_path)
            print(f"✅ Loaded gamestates: {data['gamestates'].shape}")
        
        # Load metadata
        metadata_path = session_dir / "06_final_training_data" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
            print(f"✅ Loaded metadata")
        
        # Load slice info
        slice_info_path = session_dir / "06_final_training_data" / "slice_info.json"
        if slice_info_path.exists():
            with open(slice_info_path, 'r') as f:
                data['slice_info'] = json.load(f)
            print(f"✅ Loaded slice info")
        
        return data
    
    def _analyze_session_info(self, session_data: Dict) -> Dict:
        """
        Analyze basic session information
        """
        info = {}
        
        if 'metadata' in session_data:
            metadata = session_data['metadata']
            info['session_duration'] = metadata.get('duration', 'Unknown')
            info['total_actions'] = metadata.get('total_actions', 0)
            info['recording_start'] = metadata.get('start_time', 'Unknown')
            info['recording_end'] = metadata.get('end_time', 'Unknown')
        
        if 'action_targets' in session_data:
            action_targets = session_data['action_targets']
            info['data_shape'] = action_targets.shape
            info['total_sequences'] = action_targets.shape[0]
            info['max_actions_per_sequence'] = action_targets.shape[1]
            info['features_per_action'] = action_targets.shape[2]
        
        return info
    
    def _analyze_mouse_patterns(self, session_data: Dict) -> Dict:
        """
        Analyze mouse movement patterns
        """
        if 'action_targets' not in session_data:
            return {'error': 'No action targets available'}
        
        action_targets = session_data['action_targets']
        
        # Extract mouse coordinates and timing
        # Format: [time_delta, x, y, button, key_action, key_id, scroll]
        mouse_data = []
        
        for seq_idx in range(action_targets.shape[0]):
            for action_idx in range(action_targets.shape[1]):
                action = action_targets[seq_idx, action_idx]
                
                # Check if this is a valid action (not all zeros)
                if np.any(action != 0):
                    time_delta = action[0]
                    x, y = action[1], action[2]
                    button = action[3]
                    key_action = action[4]
                    scroll = action[6]
                    
                    # Determine if this is a mouse action
                    if button != 0 or (key_action == 0 and scroll == 0 and (x != 0 or y != 0)):
                        mouse_data.append({
                            'seq_idx': seq_idx,
                            'action_idx': action_idx,
                            'time_delta': time_delta,
                            'x': x,
                            'y': y,
                            'button': button,
                            'timestamp': seq_idx * 0.6 + action_idx * time_delta  # Approximate
                        })
        
        if not mouse_data:
            return {'error': 'No mouse actions found'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(mouse_data)
        
        # Analyze movement patterns
        movement_patterns = []
        for i in range(1, len(df)):
            prev_action = df.iloc[i-1]
            curr_action = df.iloc[i]
            
            # Calculate movement metrics
            dx = curr_action['x'] - prev_action['x']
            dy = curr_action['y'] - prev_action['y']
            distance = np.sqrt(dx**2 + dy**2)
            duration = curr_action['time_delta']
            speed = distance / duration if duration > 0 else 0
            direction = np.arctan2(dy, dx)
            
            # Classify pattern type
            if distance < 5:
                pattern_type = "hover"
            elif distance < 20:
                pattern_type = "small_move"
            elif distance < 100:
                pattern_type = "medium_move"
            else:
                pattern_type = "large_move"
            
            pattern = MousePattern(
                start_x=prev_action['x'],
                start_y=prev_action['y'],
                end_x=curr_action['x'],
                end_y=curr_action['y'],
                distance=distance,
                duration=duration,
                speed=speed,
                direction=direction,
                pattern_type=pattern_type
            )
            movement_patterns.append(pattern)
        
        # Analyze patterns
        analysis = {
            'total_mouse_actions': len(mouse_data),
            'total_movements': len(movement_patterns),
            'pattern_distribution': Counter([p.pattern_type for p in movement_patterns]),
            'distance_stats': {
                'mean': np.mean([p.distance for p in movement_patterns]),
                'std': np.std([p.distance for p in movement_patterns]),
                'min': np.min([p.distance for p in movement_patterns]),
                'max': np.max([p.distance for p in movement_patterns]),
                'median': np.median([p.distance for p in movement_patterns])
            },
            'speed_stats': {
                'mean': np.mean([p.speed for p in movement_patterns if p.speed > 0]),
                'std': np.std([p.speed for p in movement_patterns if p.speed > 0]),
                'max': np.max([p.speed for p in movement_patterns if p.speed > 0])
            },
            'coordinate_ranges': {
                'x_range': [df['x'].min(), df['x'].max()],
                'y_range': [df['y'].min(), df['y'].max()],
                'x_std': df['x'].std(),
                'y_std': df['y'].std()
            }
        }
        
        # Store patterns for further analysis
        self.mouse_patterns = movement_patterns
        
        return analysis
    
    def _analyze_click_patterns(self, session_data: Dict) -> Dict:
        """
        Analyze click patterns and their context
        """
        if 'action_targets' not in session_data:
            return {'error': 'No action targets available'}
        
        action_targets = session_data['action_targets']
        
        # Extract click actions
        click_data = []
        
        for seq_idx in range(action_targets.shape[0]):
            for action_idx in range(action_targets.shape[1]):
                action = action_targets[seq_idx, action_idx]
                
                # Check if this is a click action
                if action[3] != 0:  # Button column
                    time_delta = action[0]
                    x, y = action[1], action[2]
                    button = action[3]
                    
                    # Determine click context based on coordinates and gamestate
                    context = self._determine_click_context(x, y, session_data, seq_idx)
                    
                    click_pattern = ClickPattern(
                        x=x,
                        y=y,
                        button=button,
                        timestamp=seq_idx * 0.6 + action_idx * time_delta,
                        context=context,
                        success=True  # Assume success for now
                    )
                    click_data.append(click_pattern)
        
        if not click_data:
            return {'error': 'No click actions found'}
        
        # Analyze click patterns
        analysis = {
            'total_clicks': len(click_data),
            'button_distribution': Counter([c.button for c in click_data]),
            'context_distribution': Counter([c.context for c in click_data]),
            'coordinate_analysis': {
                'x_stats': {
                    'mean': np.mean([c.x for c in click_data]),
                    'std': np.std([c.x for c in click_data]),
                    'range': [min([c.x for c in click_data]), max([c.x for c in click_data])]
                },
                'y_stats': {
                    'mean': np.mean([c.y for c in click_data]),
                    'std': np.std([c.y for c in click_data]),
                    'range': [min([c.y for c in click_data]), max([c.y for c in click_data])]
                }
            },
            'click_frequency': self._analyze_click_frequency(click_data),
            'click_patterns': self._identify_click_patterns(click_data)
        }
        
        # Store patterns for further analysis
        self.click_patterns = click_data
        
        return analysis
    
    def _analyze_keyboard_patterns(self, session_data: Dict) -> Dict:
        """
        Analyze keyboard usage patterns
        """
        if 'action_targets' not in session_data:
            return {'error': 'No action targets available'}
        
        action_targets = session_data['action_targets']
        
        # Extract keyboard actions
        keyboard_data = []
        
        for seq_idx in range(action_targets.shape[0]):
            for action_idx in range(action_targets.shape[1]):
                action = action_targets[seq_idx, action_idx]
                
                # Check if this is a keyboard action
                if action[4] != 0:  # Key action column
                    time_delta = action[0]
                    key_action = action[4]
                    key_id = action[5]
                    
                    # Determine key context
                    context = self._determine_key_context(key_id, session_data, seq_idx)
                    
                    keyboard_pattern = KeyboardPattern(
                        key_id=key_id,
                        action="press" if key_action == 1 else "release",
                        timestamp=seq_idx * 0.6 + action_idx * time_delta,
                        duration=time_delta,
                        context=context,
                        frequency=1
                    )
                    keyboard_data.append(keyboard_pattern)
        
        if not keyboard_data:
            return {'error': 'No keyboard actions found'}
        
        # Analyze keyboard patterns
        analysis = {
            'total_key_actions': len(keyboard_data),
            'key_distribution': Counter([k.key_id for k in keyboard_data]),
            'action_type_distribution': Counter([k.action for k in keyboard_data]),
            'context_distribution': Counter([k.context for k in keyboard_data]),
            'key_frequency': self._analyze_key_frequency(keyboard_data),
            'key_patterns': self._identify_key_patterns(keyboard_data)
        }
        
        # Store patterns for further analysis
        self.keyboard_patterns = keyboard_data
        
        return analysis
    
    def _analyze_scroll_patterns(self, session_data: Dict) -> Dict:
        """
        Analyze scroll wheel usage patterns
        """
        if 'action_targets' not in session_data:
            return {'error': 'No action targets available'}
        
        action_targets = session_data['action_targets']
        
        # Extract scroll actions
        scroll_data = []
        
        for seq_idx in range(action_targets.shape[0]):
            for action_idx in range(action_targets.shape[1]):
                action = action_targets[seq_idx, action_idx]
                
                # Check if this is a scroll action
                if action[6] != 0:  # Scroll column
                    time_delta = action[0]
                    scroll_direction = action[6]
                    
                    # Determine scroll context
                    context = self._determine_scroll_context(scroll_direction, session_data, seq_idx)
                    
                    scroll_pattern = ScrollPattern(
                        direction=scroll_direction,
                        timestamp=seq_idx * 0.6 + action_idx * time_delta,
                        context=context,
                        frequency=1
                    )
                    scroll_data.append(scroll_pattern)
        
        if not scroll_data:
            return {'error': 'No scroll actions found'}
        
        # Analyze scroll patterns
        analysis = {
            'total_scrolls': len(scroll_data),
            'direction_distribution': Counter([s.direction for s in scroll_data]),
            'context_distribution': Counter([s.context for s in scroll_data]),
            'scroll_frequency': self._analyze_scroll_frequency(scroll_data),
            'scroll_patterns': self._identify_scroll_patterns(scroll_data)
        }
        
        # Store patterns for further analysis
        self.scroll_patterns = scroll_data
        
        return analysis
    
    def _analyze_game_context(self, session_data: Dict) -> Dict:
        """
        Analyze game context and correlate with actions
        """
        if 'gamestates' not in session_data:
            return {'error': 'No gamestate data available'}
        
        gamestates = session_data['gamestates']
        
        # Analyze player movement patterns
        player_movement = self._analyze_player_movement(gamestates)
        
        # Analyze inventory changes
        inventory_analysis = self._analyze_inventory_changes(gamestates)
        
        # Analyze bank interactions
        bank_analysis = self._analyze_bank_interactions(gamestates)
        
        # Correlate actions with game state
        action_correlations = self._correlate_actions_with_gamestate(session_data)
        
        return {
            'player_movement': player_movement,
            'inventory_analysis': inventory_analysis,
            'bank_analysis': bank_analysis,
            'action_correlations': action_correlations
        }
    
    def _identify_behavior_patterns(self, session_data: Dict) -> Dict:
        """
        Identify high-level behavior patterns
        """
        patterns = {}
        
        # Identify common action sequences
        action_sequences = self._identify_action_sequences(session_data)
        patterns['action_sequences'] = action_sequences
        
        # Identify skill-specific behaviors
        skill_behaviors = self._identify_skill_behaviors(session_data)
        patterns['skill_behaviors'] = skill_behaviors
        
        # Identify efficiency patterns
        efficiency_patterns = self._identify_efficiency_patterns(session_data)
        patterns['efficiency_patterns'] = efficiency_patterns
        
        # Identify error patterns
        error_patterns = self._identify_error_patterns(session_data)
        patterns['error_patterns'] = error_patterns
        
        return patterns
    
    def _analyze_success_metrics(self, session_data: Dict) -> Dict:
        """
        Analyze success metrics and correlations
        """
        metrics = {}
        
        # Action success rates
        success_rates = self._calculate_action_success_rates(session_data)
        metrics['action_success_rates'] = success_rates
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(session_data)
        metrics['efficiency_metrics'] = efficiency_metrics
        
        # Learning curves
        learning_curves = self._analyze_learning_curves(session_data)
        metrics['learning_curves'] = learning_curves
        
        return metrics
    
    def _generate_bot_insights(self, session_data: Dict) -> Dict:
        """
        Generate insights for mechanical bot development
        """
        insights = {}
        
        # Mouse movement insights
        mouse_insights = self._generate_mouse_insights()
        insights['mouse_insights'] = mouse_insights
        
        # Click pattern insights
        click_insights = self._generate_click_insights()
        insights['click_insights'] = click_insights
        
        # Keyboard insights
        keyboard_insights = self._generate_keyboard_insights()
        insights['keyboard_insights'] = keyboard_insights
        
        # Game context insights
        context_insights = self._generate_context_insights(session_data)
        insights['context_insights'] = context_insights
        
        # Bot implementation recommendations
        bot_recommendations = self._generate_bot_recommendations()
        insights['bot_recommendations'] = bot_recommendations
        
        return insights
    
    def _determine_click_context(self, x: float, y: float, session_data: Dict, seq_idx: int) -> str:
        """
        Determine the context of a click based on coordinates and gamestate
        """
        # This is a simplified context determination
        # In a real implementation, you'd analyze the gamestate to determine context
        
        # Simple coordinate-based context
        if 100 <= x <= 500 and 100 <= y <= 400:
            return "inventory"
        elif 600 <= x <= 800 and 100 <= y <= 400:
            return "bank"
        elif x > 800 or y > 400:
            return "game_world"
        else:
            return "ui_elements"
    
    def _determine_key_context(self, key_id: int, session_data: Dict, seq_idx: int) -> str:
        """
        Determine the context of a key press
        """
        # Map common key IDs to contexts
        key_contexts = {
            87: "camera_up",      # W
            83: "camera_down",    # S
            65: "camera_left",    # A
            68: "camera_right",   # D
            32: "space",          # Space
            27: "escape",         # Escape
            13: "enter",          # Enter
        }
        
        return key_contexts.get(key_id, "other")
    
    def _determine_scroll_context(self, direction: int, session_data: Dict, seq_idx: int) -> str:
        """
        Determine the context of a scroll action
        """
        if direction > 0:
            return "zoom_in"
        else:
            return "zoom_out"
    
    def _analyze_click_frequency(self, click_data: List[ClickPattern]) -> Dict:
        """
        Analyze click frequency patterns
        """
        if not click_data:
            return {}
        
        # Group clicks by time intervals
        timestamps = [c.timestamp for c in click_data]
        intervals = np.diff(sorted(timestamps))
        
        return {
            'total_clicks': len(click_data),
            'click_intervals': {
                'mean': np.mean(intervals),
                'std': np.std(intervals),
                'min': np.min(intervals),
                'max': np.max(intervals),
                'median': np.median(intervals)
            },
            'clicks_per_minute': len(click_data) / (max(timestamps) - min(timestamps)) * 60 if len(timestamps) > 1 else 0
        }
    
    def _identify_click_patterns(self, click_data: List[ClickPattern]) -> List[Dict]:
        """
        Identify common click patterns
        """
        patterns = []
        
        # Group clicks by context
        context_groups = defaultdict(list)
        for click in click_data:
            context_groups[click.context].append(click)
        
        # Analyze each context group
        for context, clicks in context_groups.items():
            if len(clicks) > 1:
                # Calculate spatial patterns
                x_coords = [c.x for c in clicks]
                y_coords = [c.y for c in clicks]
                
                pattern = {
                    'context': context,
                    'click_count': len(clicks),
                    'spatial_pattern': {
                        'x_center': np.mean(x_coords),
                        'y_center': np.mean(y_coords),
                        'x_spread': np.std(x_coords),
                        'y_spread': np.std(y_coords),
                        'area_coverage': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                    },
                    'temporal_pattern': {
                        'first_click': min([c.timestamp for c in clicks]),
                        'last_click': max([c.timestamp for c in clicks]),
                        'duration': max([c.timestamp for c in clicks]) - min([c.timestamp for c in clicks])
                    }
                }
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_key_frequency(self, keyboard_data: List[KeyboardPattern]) -> Dict:
        """
        Analyze key usage frequency
        """
        if not keyboard_data:
            return {}
        
        # Group by key ID
        key_groups = defaultdict(list)
        for key in keyboard_data:
            key_groups[key.key_id].append(key)
        
        # Analyze each key
        key_analysis = {}
        for key_id, keys in key_groups.items():
            key_analysis[key_id] = {
                'total_presses': len([k for k in keys if k.action == "press"]),
                'total_releases': len([k for k in keys if k.action == "release"]),
                'average_duration': np.mean([k.duration for k in keys if k.duration > 0]),
                'usage_contexts': Counter([k.context for k in keys])
            }
        
        return {
            'total_keys_used': len(key_groups),
            'key_analysis': key_analysis,
            'most_used_keys': sorted(key_analysis.items(), key=lambda x: x[1]['total_presses'], reverse=True)[:5]
        }
    
    def _identify_key_patterns(self, keyboard_data: List[KeyboardPattern]) -> List[Dict]:
        """
        Identify common key usage patterns
        """
        patterns = []
        
        # Group by context
        context_groups = defaultdict(list)
        for key in keyboard_data:
            context_groups[key.context].append(key)
        
        # Analyze each context
        for context, keys in context_groups.items():
            if len(keys) > 1:
                pattern = {
                    'context': context,
                    'key_count': len(keys),
                    'key_distribution': Counter([k.key_id for k in keys]),
                    'action_distribution': Counter([k.action for k in keys]),
                    'temporal_pattern': {
                        'first_key': min([k.timestamp for k in keys]),
                        'last_key': max([k.timestamp for k in keys]),
                        'duration': max([k.timestamp for k in keys]) - min([k.timestamp for k in keys])
                    }
                }
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_scroll_frequency(self, scroll_data: List[ScrollPattern]) -> Dict:
        """
        Analyze scroll usage frequency
        """
        if not scroll_data:
            return {}
        
        # Group by direction
        up_scrolls = [s for s in scroll_data if s.direction < 0]
        down_scrolls = [s for s in scroll_data if s.direction > 0]
        
        return {
            'total_scrolls': len(scroll_data),
            'up_scrolls': len(up_scrolls),
            'down_scrolls': len(down_scrolls),
            'scroll_ratio': len(up_scrolls) / len(scroll_data) if scroll_data else 0,
            'scrolls_per_minute': len(scroll_data) / (max([s.timestamp for s in scroll_data]) - min([s.timestamp for s in scroll_data])) * 60 if len(scroll_data) > 1 else 0
        }
    
    def _identify_scroll_patterns(self, scroll_data: List[ScrollPattern]) -> List[Dict]:
        """
        Identify common scroll patterns
        """
        patterns = []
        
        # Group by context
        context_groups = defaultdict(list)
        for scroll in scroll_data:
            context_groups[scroll.context].append(scroll)
        
        # Analyze each context
        for context, scrolls in context_groups.items():
            if len(scrolls) > 1:
                pattern = {
                    'context': context,
                    'scroll_count': len(scrolls),
                    'direction_distribution': Counter([s.direction for s in scrolls]),
                    'temporal_pattern': {
                        'first_scroll': min([s.timestamp for s in scrolls]),
                        'last_scroll': max([s.timestamp for s in scrolls]),
                        'duration': max([s.timestamp for s in scrolls]) - min([s.timestamp for s in scrolls])
                    }
                }
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_player_movement(self, gamestates: np.ndarray) -> Dict:
        """
        Analyze player movement patterns from gamestates
        """
        # Extract player position features (assuming first 2 features are x, y)
        player_positions = gamestates[:, :, :2]  # [B, T, 2]
        
        # Calculate movement between timesteps
        movements = np.diff(player_positions, axis=1)  # [B, T-1, 2]
        distances = np.linalg.norm(movements, axis=2)  # [B, T-1]
        
        return {
            'total_movements': int(np.sum(distances > 0)),
            'movement_stats': {
                'mean_distance': float(np.mean(distances[distances > 0])) if np.any(distances > 0) else 0,
                'std_distance': float(np.std(distances[distances > 0])) if np.any(distances > 0) else 0,
                'max_distance': float(np.max(distances)) if distances.size > 0 else 0,
                'min_distance': float(np.min(distances[distances > 0])) if np.any(distances > 0) else 0
            },
            'position_ranges': {
                'x_range': [float(np.min(player_positions[:, :, 0])), float(np.max(player_positions[:, :, 0]))],
                'y_range': [float(np.min(player_positions[:, :, 1])), float(np.max(player_positions[:, :, 1]))]
            }
        }
    
    def _analyze_inventory_changes(self, gamestates: np.ndarray) -> Dict:
        """
        Analyze inventory changes from gamestates
        """
        # This is a placeholder - you'd need to know which features represent inventory
        # For now, return basic stats
        return {
            'inventory_features_available': gamestates.shape[2] > 10,
            'total_gamestate_features': gamestates.shape[2]
        }
    
    def _analyze_bank_interactions(self, gamestates: np.ndarray) -> Dict:
        """
        Analyze bank interactions from gamestates
        """
        # This is a placeholder - you'd need to know which features represent bank state
        return {
            'bank_features_available': gamestates.shape[2] > 30,
            'total_gamestate_features': gamestates.shape[2]
        }
    
    def _correlate_actions_with_gamestate(self, session_data: Dict) -> Dict:
        """
        Correlate actions with game state changes
        """
        # This is a placeholder for more sophisticated correlation analysis
        return {
            'correlation_analysis': 'Placeholder - implement based on specific gamestate features'
        }
    
    def _identify_action_sequences(self, session_data: Dict) -> List[Dict]:
        """
        Identify common action sequences
        """
        # This is a placeholder for sequence analysis
        return [
            {
                'sequence_type': 'placeholder',
                'description': 'Implement sequence identification based on action patterns'
            }
        ]
    
    def _identify_skill_behaviors(self, session_data: Dict) -> List[Dict]:
        """
        Identify skill-specific behaviors
        """
        # This is a placeholder for skill behavior analysis
        return [
            {
                'skill': 'placeholder',
                'behaviors': 'Implement skill-specific behavior identification'
            }
        ]
    
    def _identify_efficiency_patterns(self, session_data: Dict) -> List[Dict]:
        """
        Identify efficiency patterns
        """
        # This is a placeholder for efficiency analysis
        return [
            {
                'pattern_type': 'placeholder',
                'description': 'Implement efficiency pattern identification'
            }
        ]
    
    def _identify_error_patterns(self, session_data: Dict) -> List[Dict]:
        """
        Identify error patterns
        """
        # This is a placeholder for error analysis
        return [
            {
                'error_type': 'placeholder',
                'description': 'Implement error pattern identification'
            }
        ]
    
    def _calculate_action_success_rates(self, session_data: Dict) -> Dict:
        """
        Calculate success rates for different actions
        """
        # This is a placeholder for success rate calculation
        return {
            'click_success_rate': 0.95,  # Placeholder
            'key_success_rate': 0.98,    # Placeholder
            'scroll_success_rate': 1.0   # Placeholder
        }
    
    def _calculate_efficiency_metrics(self, session_data: Dict) -> Dict:
        """
        Calculate efficiency metrics
        """
        # This is a placeholder for efficiency calculation
        return {
            'actions_per_minute': 0,     # Placeholder
            'successful_actions_per_minute': 0,  # Placeholder
            'efficiency_score': 0.0      # Placeholder
        }
    
    def _analyze_learning_curves(self, session_data: Dict) -> Dict:
        """
        Analyze learning curves over time
        """
        # This is a placeholder for learning curve analysis
        return {
            'learning_curve': 'Placeholder - implement time-based analysis'
        }
    
    def _generate_mouse_insights(self) -> List[str]:
        """
        Generate insights about mouse behavior for bot development
        """
        insights = []
        
        if self.mouse_patterns:
            # Analyze movement patterns
            distances = [p.distance for p in self.mouse_patterns]
            speeds = [p.speed for p in self.mouse_patterns if p.speed > 0]
            
            if distances:
                avg_distance = np.mean(distances)
                if avg_distance < 20:
                    insights.append("Mouse movements are typically small and precise - bot should use fine-grained movement")
                elif avg_distance < 100:
                    insights.append("Mouse movements are moderate - bot should balance speed and precision")
                else:
                    insights.append("Mouse movements are large - bot should prioritize speed over precision")
            
            if speeds:
                avg_speed = np.mean(speeds)
                if avg_speed < 50:
                    insights.append("Mouse movements are slow and deliberate - bot should use smooth, controlled movements")
                elif avg_speed < 200:
                    insights.append("Mouse movements are moderate speed - bot should use natural movement curves")
                else:
                    insights.append("Mouse movements are fast - bot should use direct, efficient paths")
        
        return insights
    
    def _generate_click_insights(self) -> List[str]:
        """
        Generate insights about click behavior for bot development
        """
        insights = []
        
        if self.click_patterns:
            # Analyze click patterns
            contexts = [c.context for c in self.click_patterns]
            context_counts = Counter(contexts)
            
            # Most common click context
            most_common_context = context_counts.most_common(1)[0][0]
            insights.append(f"Most clicks occur in '{most_common_context}' context - prioritize this area for bot automation")
            
            # Click frequency
            total_clicks = len(self.click_patterns)
            if total_clicks > 100:
                insights.append("High click frequency suggests active gameplay - bot should maintain similar activity levels")
            elif total_clicks > 50:
                insights.append("Moderate click frequency suggests balanced gameplay - bot should match this rhythm")
            else:
                insights.append("Low click frequency suggests passive gameplay - bot should avoid over-clicking")
        
        return insights
    
    def _generate_keyboard_insights(self) -> List[str]:
        """
        Generate insights about keyboard behavior for bot development
        """
        insights = []
        
        if self.keyboard_patterns:
            # Analyze key usage
            key_ids = [k.key_id for k in self.keyboard_patterns]
            key_counts = Counter(key_ids)
            
            # Most used keys
            most_used_keys = key_counts.most_common(3)
            insights.append(f"Most used keys: {[k[0] for k in most_used_keys]} - bot should prioritize these")
            
            # Key context analysis
            contexts = [k.context for k in self.keyboard_patterns]
            context_counts = Counter(contexts)
            
            most_common_context = context_counts.most_common(1)[0][0]
            insights.append(f"Most keyboard actions occur in '{most_common_context}' context")
        
        return insights
    
    def _generate_context_insights(self, session_data: Dict) -> List[str]:
        """
        Generate insights about game context for bot development
        """
        insights = []
        
        # This is a placeholder for context insights
        insights.append("Implement context analysis based on gamestate features")
        
        return insights
    
    def _generate_bot_recommendations(self) -> List[str]:
        """
        Generate specific recommendations for bot development
        """
        recommendations = []
        
        # Mouse recommendations
        if self.mouse_patterns:
            recommendations.append("Implement smooth mouse movement with natural acceleration curves")
            recommendations.append("Use variable movement speeds based on distance (fast for long, slow for short)")
        
        # Click recommendations
        if self.click_patterns:
            recommendations.append("Implement context-aware clicking (different behavior for inventory vs game world)")
            recommendations.append("Add small random delays between clicks to mimic human behavior")
        
        # Keyboard recommendations
        if self.keyboard_patterns:
            recommendations.append("Implement key press duration variation to mimic human typing patterns")
            recommendations.append("Add context-specific key bindings based on game state")
        
        # General recommendations
        recommendations.append("Implement action queuing system for smooth gameplay flow")
        recommendations.append("Add error recovery mechanisms for failed actions")
        recommendations.append("Use gamestate information to make context-aware decisions")
        
        return recommendations
    
    def _save_analysis_results(self, analysis: Dict, session_id: str):
        """
        Save analysis results to files
        """
        # Create output directory
        output_dir = Path(f"human_behavior_analysis/{session_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main analysis
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save individual pattern data
        if self.mouse_patterns:
            mouse_df = pd.DataFrame([asdict(p) for p in self.mouse_patterns])
            mouse_df.to_csv(output_dir / "mouse_patterns.csv", index=False)
        
        if self.click_patterns:
            click_df = pd.DataFrame([asdict(p) for p in self.click_patterns])
            click_df.to_csv(output_dir / "click_patterns.csv", index=False)
        
        if self.keyboard_patterns:
            keyboard_df = pd.DataFrame([asdict(p) for p in self.keyboard_patterns])
            keyboard_df.to_csv(output_dir / "keyboard_patterns.csv", index=False)
        
        if self.scroll_patterns:
            scroll_df = pd.DataFrame([asdict(p) for p in self.scroll_patterns])
            scroll_df.to_csv(output_dir / "scroll_patterns.csv", index=False)
        
        print(f"💾 Analysis results saved to: {output_dir}")
    
    def _print_analysis_summary(self, analysis: Dict):
        """
        Print a summary of the analysis results
        """
        print(f"\n📊 Human Behavior Analysis Summary:")
        print("=" * 60)
        
        # Session info
        if 'session_info' in analysis:
            info = analysis['session_info']
            print(f"📅 Session Duration: {info.get('session_duration', 'Unknown')}")
            print(f"🎯 Total Actions: {info.get('total_actions', 'Unknown')}")
            print(f"📊 Data Shape: {info.get('data_shape', 'Unknown')}")
        
        # Mouse analysis
        if 'mouse_analysis' in analysis and 'error' not in analysis['mouse_analysis']:
            mouse = analysis['mouse_analysis']
            print(f"\n🖱️  Mouse Analysis:")
            print(f"  • Total mouse actions: {mouse.get('total_mouse_actions', 0)}")
            print(f"  • Total movements: {mouse.get('total_movements', 0)}")
            if 'distance_stats' in mouse:
                dist_stats = mouse['distance_stats']
                print(f"  • Average movement distance: {dist_stats.get('mean', 0):.1f} pixels")
                print(f"  • Movement range: {dist_stats.get('min', 0):.1f} - {dist_stats.get('max', 0):.1f} pixels")
        
        # Click analysis
        if 'click_analysis' in analysis and 'error' not in analysis['click_analysis']:
            click = analysis['click_analysis']
            print(f"\n🖱️  Click Analysis:")
            print(f"  • Total clicks: {click.get('total_clicks', 0)}")
            if 'context_distribution' in click:
                contexts = click['context_distribution']
                print(f"  • Most common context: {contexts.most_common(1)[0][0] if contexts else 'Unknown'}")
        
        # Keyboard analysis
        if 'keyboard_analysis' in analysis and 'error' not in analysis['keyboard_analysis']:
            keyboard = analysis['keyboard_analysis']
            print(f"\n⌨️  Keyboard Analysis:")
            print(f"  • Total key actions: {keyboard.get('total_key_actions', 0)}")
            print(f"  • Unique keys used: {keyboard.get('total_keys_used', 0)}")
        
        # Scroll analysis
        if 'scroll_analysis' in analysis and 'error' not in analysis['scroll_analysis']:
            scroll = analysis['scroll_analysis']
            print(f"\n🖱️  Scroll Analysis:")
            print(f"  • Total scrolls: {scroll.get('total_scrolls', 0)}")
            print(f"  • Up/Down ratio: {scroll.get('scroll_ratio', 0):.1%}")
        
        # Bot insights
        if 'bot_development_insights' in analysis:
            insights = analysis['bot_development_insights']
            print(f"\n🤖 Bot Development Insights:")
            
            if 'mouse_insights' in insights:
                for insight in insights['mouse_insights'][:2]:  # Show first 2
                    print(f"  • {insight}")
            
            if 'click_insights' in insights:
                for insight in insights['click_insights'][:2]:  # Show first 2
                    print(f"  • {insight}")
            
            if 'bot_recommendations' in insights:
                print(f"  • Key recommendation: {insights['bot_recommendations'][0] if insights['bot_recommendations'] else 'None'}")
        
        print(f"\n💡 Use these insights to develop mechanical bots that mimic human behavior patterns!")
        print(f"📁 Detailed results saved to: human_behavior_analysis/{analysis.get('session_info', {}).get('session_id', 'unknown')}/")

def main():
    """
    Main function to demonstrate usage
    """
    analyzer = HumanBehaviorAnalyzer()
    
    # Example usage
    print("🔍 OSRS Human Behavior Analyzer")
    print("=" * 50)
    print("This tool analyzes mouse, keyboard, and scroll patterns from OSRS gameplay.")
    print("Use it to develop mechanical bots that mimic human behavior.")
    print("\nTo analyze a session, use:")
    print("analyzer.analyze_session('session_id')")
    print("\nExample:")
    print("analyzer.analyze_session('20250831_113719')")

if __name__ == "__main__":
    main()
