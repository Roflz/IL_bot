#!/usr/bin/env python3
"""
Enhanced Behavioral Metrics for OSRS Bot Analysis
Provides meaningful context for bot predictions and connects them to game state
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class GameAction:
    """Represents a single game action with full context"""
    timestamp: float
    event_type: str  # CLICK, KEY, SCROLL, MOVE
    x: float
    y: float
    confidence: float
    player_x: float
    player_y: float
    inventory_state: str
    bank_open: bool
    nearby_objects: List[str]
    action_context: str

@dataclass
class ActionSequence:
    """Represents a sequence of related actions"""
    start_time: float
    end_time: float
    actions: List[GameAction]
    sequence_type: str  # "banking", "crafting", "walking", "combat"
    success_rate: float

class EnhancedBehavioralMetrics:
    """
    Enhanced behavioral analysis that provides meaningful context
    for bot predictions and connects them to game state
    """
    
    def __init__(self, save_dir: str = "enhanced_behavioral_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store temporal sequences for analysis
        self.action_sequences: List[ActionSequence] = []
        self.game_contexts: Dict[str, Dict] = {}
        
        # Analysis results
        self.mouse_patterns = {}
        self.click_patterns = {}
        self.keyboard_patterns = {}
        self.scroll_patterns = {}
        self.game_state_correlations = {}
    
    def _safe_item(self, tensor, default=0.0):
        """Safely convert tensor to scalar, handling multi-element tensors"""
        try:
            if tensor.numel() == 0:
                return default
            elif tensor.numel() == 1:
                return tensor.item()
            else:
                # For multi-element tensors, return mean
                return tensor.mean().item()
        except:
            return default
        
    def analyze_epoch_predictions(self, 
                                model_outputs: Dict[str, torch.Tensor],
                                gamestates: torch.Tensor,
                                action_targets: torch.Tensor,
                                valid_mask: torch.Tensor,
                                epoch: int) -> Dict:
        """
        Enhanced analysis that provides meaningful context for predictions
        """
        print(f"\n🔍 Enhanced Behavioral Analysis (Epoch {epoch}):")
        print("=" * 60)
        
        analysis = {}
        
        try:
            # 1. Temporal Action Analysis
            temporal_analysis = self._analyze_temporal_patterns(
                model_outputs, gamestates, action_targets, valid_mask
            )
            analysis['temporal_patterns'] = temporal_analysis
            
            # 2. Game State Correlation Analysis
            game_correlation = self._analyze_game_state_correlations(
                model_outputs, gamestates, action_targets, valid_mask
            )
            analysis['game_correlations'] = game_correlation
            
            # 3. Mouse Movement Pattern Analysis
            mouse_patterns = self._analyze_mouse_movement_patterns(
                model_outputs, gamestates, valid_mask
            )
            analysis['mouse_patterns'] = mouse_patterns
            
            # 4. Action Context Analysis
            action_context = self._analyze_action_context(
                model_outputs, gamestates, action_targets, valid_mask
            )
            analysis['action_context'] = action_context
            
            # 5. Predictive Quality Analysis
            predictive_quality = self._analyze_predictive_quality(
                model_outputs, action_targets, valid_mask
            )
            analysis['predictive_quality'] = predictive_quality
            
            # Print meaningful insights
            self._print_enhanced_insights(analysis, epoch)
            
        except Exception as e:
            error_msg = str(e)
            # Truncate very long error messages (like tensor dumps)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"⚠️  Enhanced analysis failed: {error_msg}")
            analysis = {'error': error_msg}
        
        return analysis
    
    def _analyze_temporal_patterns(self, 
                                 model_outputs: Dict[str, torch.Tensor],
                                 gamestates: torch.Tensor,
                                 action_targets: torch.Tensor,
                                 valid_mask: torch.Tensor) -> Dict:
        """
        Analyze temporal patterns in predictions and actions
        """
        if 'time_q' not in model_outputs:
            return {'error': 'No timing predictions available'}
        
        # Get timing predictions
        time_predictions = model_outputs['time_q']  # [B, 3] - quantiles
        
        # Analyze timing distribution
        median_times = time_predictions[:, 1]  # q0.5
        time_ranges = time_predictions[:, 2] - time_predictions[:, 0]  # q0.9 - q0.1
        
        # Convert to meaningful time intervals
        time_stats = {
            'median_interval': self._safe_item(median_times.mean()),
            'min_interval': self._safe_item(median_times.min()),
            'max_interval': self._safe_item(median_times.max()),
            'time_uncertainty': self._safe_item(time_ranges.mean()),
            'fast_actions': self._safe_item((median_times < 0.5).float().mean()),  # Actions < 0.5s
            'slow_actions': self._safe_item((median_times > 2.0).float().mean()),  # Actions > 2s
        }
        
        # Analyze action sequences
        action_sequences = self._identify_action_sequences(
            time_predictions, gamestates, valid_mask
        )
        
        return {
            'timing_stats': time_stats,
            'action_sequences': action_sequences
        }
    
    def _analyze_game_state_correlations(self,
                                       model_outputs: Dict[str, torch.Tensor],
                                       gamestates: torch.Tensor,
                                       action_targets: torch.Tensor,
                                       valid_mask: torch.Tensor) -> Dict:
        """
        Correlate bot predictions with game state changes
        """
        # Extract key game state features
        B, T, F = gamestates.shape
        
        # Get event probabilities
        event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)
        
        # Player position over time
        player_positions = gamestates[:, :, :2]  # First 2 features are x, y
        
        # Inventory state (assuming features 10-20 are inventory related)
        inventory_features = gamestates[:, :, 10:20] if F > 20 else gamestates[:, :, 10:]
        
        # Bank state (assuming feature 30 indicates bank open/closed)
        bank_state = gamestates[:, :, 30] if F > 30 else gamestates[:, :, 0]
        
        # Analyze correlations
        correlations = {
            'player_movement': self._analyze_player_movement(player_positions),
            'inventory_changes': self._analyze_inventory_changes(inventory_features),
            'bank_interactions': self._analyze_bank_interactions(bank_state),
            'action_state_correlation': self._correlate_actions_with_state(
                event_probs, gamestates, valid_mask
            )
        }
        
        return correlations
    
    def _analyze_player_movement(self, player_positions):
        """Analyze player movement patterns"""
        if player_positions.numel() == 0:
            return {'error': 'No player position data'}
        
        # Calculate movement statistics
        if player_positions.shape[0] > 1:
            dx = player_positions[1:, 0] - player_positions[:-1, 0]
            dy = player_positions[1:, 1] - player_positions[:-1, 1]
            
            return {
                'total_distance': self._safe_item(torch.sqrt(dx**2 + dy**2).sum()),
                'average_step': self._safe_item(torch.sqrt(dx**2 + dy**2).mean()),
                'max_step': self._safe_item(torch.sqrt(dx**2 + dy**2).max()),
                'movement_direction': self._safe_item(torch.atan2(dy.mean(), dx.mean()))
            }
        else:
            return {'error': 'Insufficient position data for movement analysis'}
    
    def _analyze_inventory_changes(self, inventory_features):
        """Analyze inventory change patterns"""
        if inventory_features.numel() == 0:
            return {'error': 'No inventory data'}
        
        return {
            'inventory_variance': self._safe_item(inventory_features.var()),
            'inventory_mean': self._safe_item(inventory_features.mean()),
            'has_changes': (inventory_features.std() > 0.1).any().item()
        }
    
    def _analyze_bank_interactions(self, bank_state):
        """Analyze bank interaction patterns"""
        if bank_state.numel() == 0:
            return {'error': 'No bank data'}
        
        return {
            'bank_open_frequency': self._safe_item(bank_state.float().mean()),
            'bank_interactions': (bank_state.diff() != 0).sum().item()
        }
    
    def _correlate_actions_with_state(self, event_probs, gamestates, valid_mask):
        """Correlate actions with game state"""
        if not valid_mask.any():
            return {'error': 'No valid actions'}
        
        # Simple correlation analysis
        valid_events = event_probs[valid_mask]
        valid_gamestates = gamestates[valid_mask]
        
        return {
            'state_action_correlation': 0.0,  # Placeholder
            'correlation_strength': 'weak'
        }
    
    def _analyze_mouse_movement_patterns(self,
                                       model_outputs: Dict[str, torch.Tensor],
                                       gamestates: torch.Tensor,
                                       valid_mask: torch.Tensor) -> Dict:
        """
        Analyze mouse movement patterns and connect them to game objects
        """
        if 'x_mu' not in model_outputs or 'y_mu' not in model_outputs:
            return {'error': 'No mouse position predictions available'}
        
        # Get mouse predictions
        x_pred = model_outputs['x_mu']  # [B, A]
        y_pred = model_outputs['y_mu']  # [B, A]
        x_uncertainty = model_outputs['x_logsig'].exp()  # [B, A]
        y_uncertainty = model_outputs['y_logsig'].exp()  # [B, A]
        
        # Analyze mouse movement patterns
        mouse_stats = {
            'position_range': {
                'x_min': self._safe_item(x_pred.min()),
                'x_max': self._safe_item(x_pred.max()),
                'y_min': self._safe_item(y_pred.min()),
                'y_max': self._safe_item(y_pred.max()),
            },
            'uncertainty_stats': {
                'x_uncertainty_mean': self._safe_item(x_uncertainty.mean()),
                'y_uncertainty_mean': self._safe_item(y_uncertainty.mean()),
                'x_uncertainty_std': self._safe_item(x_uncertainty.std()),
                'y_uncertainty_std': self._safe_item(y_uncertainty.std()),
            },
            'movement_patterns': self._identify_mouse_patterns(x_pred, y_pred, gamestates)
        }
        
        return mouse_stats
    
    def _analyze_action_context(self,
                              model_outputs: Dict[str, torch.Tensor],
                              gamestates: torch.Tensor,
                              action_targets: torch.Tensor,
                              valid_mask: torch.Tensor) -> Dict:
        """
        Analyze the context in which actions are predicted
        """
        if 'event_logits' not in model_outputs:
            return {'error': 'No event predictions available'}
        
        # Get event predictions
        event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)  # [B, A, 4]
        
        # Analyze event distribution
        event_distribution = event_probs.mean(dim=(0, 1))  # Average across batch and actions
        
        # Identify action contexts
        action_contexts = {
            'event_distribution': {
                'CLICK': self._safe_item(event_distribution[0]),
                'KEY': self._safe_item(event_distribution[1]),
                'SCROLL': self._safe_item(event_distribution[2]),
                'MOVE': self._safe_item(event_distribution[3]),
            },
            'context_patterns': self._identify_action_contexts(
                event_probs, gamestates, valid_mask
            ),
            'prediction_confidence': self._analyze_prediction_confidence(event_probs)
        }
        
        return action_contexts
    
    def _analyze_predictive_quality(self,
                                  model_outputs: Dict[str, torch.Tensor],
                                  action_targets: torch.Tensor,
                                  valid_mask: torch.Tensor) -> Dict:
        """
        Analyze the quality and reliability of predictions
        """
        quality_metrics = {}
        
        # Event prediction quality
        if 'event_logits' in model_outputs:
            event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)
            quality_metrics['event_quality'] = self._assess_event_quality(
                event_probs, action_targets, valid_mask
            )
        
        # Coordinate prediction quality
        if 'x_mu' in model_outputs and 'y_mu' in model_outputs:
            quality_metrics['coordinate_quality'] = self._assess_coordinate_quality(
                model_outputs, action_targets, valid_mask
            )
        
        # Timing prediction quality
        if 'time_q' in model_outputs:
            quality_metrics['timing_quality'] = self._assess_timing_quality(
                model_outputs, action_targets, valid_mask
            )
        
        return quality_metrics
    
    def _identify_action_sequences(self,
                                 time_predictions: torch.Tensor,
                                 gamestates: torch.Tensor,
                                 valid_mask: torch.Tensor) -> List[Dict]:
        """
        Identify sequences of related actions
        """
        sequences = []
        
        # Group actions by time proximity
        B, T = gamestates.shape[:2]
        
        for batch_idx in range(B):
            # Get valid actions for this batch
            valid_actions = valid_mask[batch_idx]
            if not valid_actions.any():
                continue
            
            # Get timing for valid actions
            if time_predictions.dim() == 2 and time_predictions.shape[0] > batch_idx:
                batch_times = time_predictions[batch_idx]  # [3] - quantiles
            else:
                # Fallback: use the first batch or create dummy data
                batch_times = time_predictions[0] if time_predictions.numel() > 0 else torch.zeros(3)
            
            # Identify action clusters
            action_clusters = self._cluster_actions_by_time(batch_times, gamestates[batch_idx])
            
            for cluster in action_clusters:
                sequences.append({
                    'batch': batch_idx,
                    'start_time': cluster['start_time'],
                    'end_time': cluster['end_time'],
                    'action_count': cluster['action_count'],
                    'sequence_type': cluster['type']
                })
        
        return sequences
    
    def _cluster_actions_by_time(self, 
                               times: torch.Tensor,
                               gamestate: torch.Tensor) -> List[Dict]:
        """
        Cluster actions by temporal proximity
        """
        clusters = []
        
        # Simple temporal clustering
        if times.numel() > 0:
            # Handle both 1D and 2D tensors for timing
            if times.dim() == 1:
                median_time = self._safe_item(times[1])  # q0.5
            else:
                median_time = self._safe_item(times[:, 1].mean())  # Average q0.5 across batches
            
            # Classify sequence type based on timing and gamestate
            if median_time < 0.5:
                sequence_type = "rapid_actions"
            elif median_time < 1.0:
                sequence_type = "normal_actions"
            else:
                sequence_type = "slow_actions"
            
            clusters.append({
                'start_time': 0.0,
                'end_time': median_time,
                'action_count': 1,
                'type': sequence_type
            })
        
        return clusters
    
    def _identify_mouse_patterns(self,
                               x_pred: torch.Tensor,
                               y_pred: torch.Tensor,
                               gamestates: torch.Tensor) -> Dict:
        """
        Identify meaningful mouse movement patterns
        """
        patterns = {}
        
        # Calculate movement vectors
        if x_pred.size(0) > 1:
            dx = x_pred[1:] - x_pred[:-1]
            dy = y_pred[1:] - y_pred[:-1]
            
            # Movement statistics
            patterns['movement_stats'] = {
                'total_distance': self._safe_item(torch.sqrt(dx**2 + dy**2).sum()),
                'average_step': self._safe_item(torch.sqrt(dx**2 + dy**2).mean()),
                'max_step': self._safe_item(torch.sqrt(dx**2 + dy**2).max()),
                'movement_direction': self._safe_item(torch.atan2(dy.mean(), dx.mean()))
            }
            
            # Identify movement types
            patterns['movement_types'] = {
                'small_movements': self._safe_item((torch.sqrt(dx**2 + dy**2) < 10).float().mean()),
                'medium_movements': self._safe_item(((torch.sqrt(dx**2 + dy**2) >= 10) & 
                                   (torch.sqrt(dx**2 + dy**2) < 50)).float().mean()),
                'large_movements': self._safe_item((torch.sqrt(dx**2 + dy**2) >= 50).float().mean())
            }
        
        return patterns
    
    def _identify_action_contexts(self,
                                event_probs: torch.Tensor,
                                gamestates: torch.Tensor,
                                valid_mask: torch.Tensor) -> Dict:
        """
        Identify the context in which different actions are predicted
        """
        contexts = {}
        
        # Analyze when each event type is predicted
        for event_idx, event_name in enumerate(['CLICK', 'KEY', 'SCROLL', 'MOVE']):
            if event_probs.dim() == 3 and event_probs.shape[2] > event_idx:
                event_prob = event_probs[:, :, event_idx]  # [B, A]
            else:
                # Fallback: use the first event or create dummy data
                if event_probs.numel() > 0 and event_probs.dim() == 3:
                    event_prob = event_probs[:, :, 0]
                else:
                    event_prob = torch.zeros(2, 100)  # Default shape
            
            # Find high-confidence predictions
            high_conf_mask = event_prob > 0.7
            
            if high_conf_mask.any():
                # Analyze gamestate context for high-confidence predictions
                contexts[event_name] = self._analyze_event_context(
                    high_conf_mask, gamestates, event_name
                )
        
        return contexts
    
    def _analyze_event_context(self,
                             high_conf_mask: torch.Tensor,
                             gamestates: torch.Tensor,
                             event_name: str) -> Dict:
        """
        Analyze the game state context for high-confidence event predictions
        """
        context = {}
        
        # Get gamestates where this event is predicted with high confidence
        if high_conf_mask.any():
            # Analyze player position context
            if high_conf_mask.dtype == torch.bool and high_conf_mask.shape == gamestates.shape[:2]:
                player_positions = gamestates[high_conf_mask][:, :2]  # x, y coordinates
            else:
                # Fallback: use all gamestates
                player_positions = gamestates[:, :, :2]  # x, y coordinates
            
            if player_positions.numel() > 0:
                context['player_context'] = {
                    'position_range': {
                        'x_min': self._safe_item(player_positions[:, 0].min()),
                        'x_max': self._safe_item(player_positions[:, 0].max()),
                        'y_min': self._safe_item(player_positions[:, 1].min()),
                        'y_max': self._safe_item(player_positions[:, 1].max()),
                    },
                    'position_std': {
                        'x_std': self._safe_item(player_positions[:, 0].std()),
                        'y_std': self._safe_item(player_positions[:, 1].std()),
                    }
                }
        
        return context
    
    def _assess_event_quality(self,
                            event_probs: torch.Tensor,
                            action_targets: torch.Tensor,
                            valid_mask: torch.Tensor) -> Dict:
        """
        Assess the quality of event predictions
        """
        quality = {}
        
        # Calculate prediction confidence
        max_probs, predicted_events = event_probs.max(dim=-1)  # [B, A]
        
        # Confidence statistics
        if valid_mask.any():
            valid_max_probs = max_probs[valid_mask]
            quality['confidence_stats'] = {
                'mean_confidence': self._safe_item(valid_max_probs.mean()),
                'min_confidence': self._safe_item(valid_max_probs.min()),
                'max_confidence': self._safe_item(valid_max_probs.max()),
                'confidence_std': self._safe_item(valid_max_probs.std()),
            }
        else:
            quality['confidence_stats'] = {
                'mean_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'confidence_std': 0.0,
            }
        
        # Diversity of predictions
        if valid_mask.any():
            valid_predicted_events = predicted_events[valid_mask]
            unique_events = valid_predicted_events.unique()
            quality['prediction_diversity'] = {
                'unique_events': unique_events.numel(),
                'event_distribution': {
                    'CLICK': self._safe_item((valid_predicted_events == 0).float().mean()),
                    'KEY': self._safe_item((valid_predicted_events == 1).float().mean()),
                    'SCROLL': self._safe_item((valid_predicted_events == 2).float().mean()),
                    'MOVE': self._safe_item((valid_predicted_events == 3).float().mean()),
                }
            }
        else:
            quality['prediction_diversity'] = {
                'unique_events': 0,
                'event_distribution': {
                    'CLICK': 0.0,
                    'KEY': 0.0,
                    'SCROLL': 0.0,
                    'MOVE': 0.0,
                }
            }
        
        return quality
    
    def _assess_coordinate_quality(self,
                                 model_outputs: Dict[str, torch.Tensor],
                                 action_targets: torch.Tensor,
                                 valid_mask: torch.Tensor) -> Dict:
        """
        Assess the quality of coordinate predictions
        """
        quality = {}
        
        if 'x_mu' in model_outputs and 'y_mu' in model_outputs:
            x_pred = model_outputs['x_mu']  # [B, A]
            y_pred = model_outputs['y_mu']  # [B, A]
            x_uncertainty = model_outputs['x_logsig'].exp()  # [B, A]
            y_uncertainty = model_outputs['y_logsig'].exp()  # [B, A]
            
            # Uncertainty quality
            if valid_mask.any():
                valid_x_uncertainty = x_uncertainty[valid_mask]
                valid_y_uncertainty = y_uncertainty[valid_mask]
                valid_x_pred = x_pred[valid_mask]
                valid_y_pred = y_pred[valid_mask]
                
                quality['uncertainty_quality'] = {
                    'x_uncertainty_mean': self._safe_item(valid_x_uncertainty.mean()),
                    'y_uncertainty_mean': self._safe_item(valid_y_uncertainty.mean()),
                    'x_uncertainty_std': self._safe_item(valid_x_uncertainty.std()),
                    'y_uncertainty_std': self._safe_item(valid_y_uncertainty.std()),
                    'has_infinite_uncertainty': torch.isinf(x_uncertainty).any() or torch.isinf(y_uncertainty).any(),
                }
                
                # Prediction range quality
                quality['prediction_range'] = {
                    'x_range': [self._safe_item(valid_x_pred.min()), self._safe_item(valid_x_pred.max())],
                    'y_range': [self._safe_item(valid_y_pred.min()), self._safe_item(valid_y_pred.max())],
                    'prediction_std': {
                        'x_std': self._safe_item(valid_x_pred.std()),
                        'y_std': self._safe_item(valid_y_pred.std()),
                    }
                }
            else:
                quality['uncertainty_quality'] = {
                    'x_uncertainty_mean': 0.0,
                    'y_uncertainty_mean': 0.0,
                    'x_uncertainty_std': 0.0,
                    'y_uncertainty_std': 0.0,
                    'has_infinite_uncertainty': False,
                }
                
                quality['prediction_range'] = {
                    'x_range': [0.0, 0.0],
                    'y_range': [0.0, 0.0],
                    'prediction_std': {
                        'x_std': 0.0,
                        'y_std': 0.0,
                    }
                }
        
        return quality
    
    def _assess_timing_quality(self,
                             model_outputs: Dict[str, torch.Tensor],
                             action_targets: torch.Tensor,
                             valid_mask: torch.Tensor) -> Dict:
        """
        Assess the quality of timing predictions
        """
        quality = {}
        
        if 'time_q' in model_outputs:
            time_predictions = model_outputs['time_q']  # [B, 3] - quantiles
            
            # Timing uncertainty
            time_ranges = time_predictions[:, 2] - time_predictions[:, 0]  # q0.9 - q0.1
            
            quality['timing_quality'] = {
                'median_timing': self._safe_item(time_predictions[:, 1].mean()),  # q0.5
                'timing_uncertainty': self._safe_item(time_ranges.mean()),
                'timing_consistency': self._safe_item(time_ranges.std()),
                'has_negative_timing': (time_predictions < 0).any().item(),
                'timing_distribution': {
                    'fast_actions': self._safe_item((time_predictions[:, 1] < 0.5).float().mean()),
                    'normal_actions': self._safe_item(((time_predictions[:, 1] >= 0.5) & 
                                     (time_predictions[:, 1] < 2.0)).float().mean()),
                    'slow_actions': self._safe_item((time_predictions[:, 1] >= 2.0).float().mean()),
                }
            }
        
        return quality
    
    def _print_enhanced_insights(self, analysis: Dict, epoch: int):
        """
        Print meaningful insights from the enhanced analysis
        """
        print(f"\n🎯 Enhanced Insights for Epoch {epoch}:")
        print("=" * 50)
        
        # Temporal patterns
        if 'temporal_patterns' in analysis:
            temporal = analysis['temporal_patterns']
            if 'timing_stats' in temporal:
                stats = temporal['timing_stats']
                print(f"⏰ Timing Analysis:")
                print(f"  • Median action interval: {stats['median_interval']:.2f}s")
                print(f"  • Fast actions (<0.5s): {stats['fast_actions']:.1%}")
                print(f"  • Slow actions (>2s): {stats['slow_actions']:.1%}")
                print(f"  • Timing uncertainty: ±{stats['time_uncertainty']:.2f}s")
        
        # Mouse patterns
        if 'mouse_patterns' in analysis:
            mouse = analysis['mouse_patterns']
            if 'position_range' in mouse:
                pos_range = mouse['position_range']
                print(f"\n🖱️  Mouse Movement Analysis:")
                print(f"  • X range: {pos_range['x_min']:.0f} to {pos_range['x_max']:.0f}")
                print(f"  • Y range: {pos_range['y_min']:.0f} to {pos_range['y_max']:.0f}")
                
                if 'uncertainty_stats' in mouse:
                    unc_stats = mouse['uncertainty_stats']
                    print(f"  • X uncertainty: ±{unc_stats['x_uncertainty_mean']:.1f} pixels")
                    print(f"  • Y uncertainty: ±{unc_stats['y_uncertainty_mean']:.1f} pixels")
        
        # Action context
        if 'action_context' in analysis:
            context = analysis['action_context']
            if 'event_distribution' in context:
                event_dist = context['event_distribution']
                print(f"\n🎮 Action Context Analysis:")
                print(f"  • CLICK: {event_dist['CLICK']:.1%}")
                print(f"  • KEY: {event_dist['KEY']:.1%}")
                print(f"  • SCROLL: {event_dist['SCROLL']:.1%}")
                print(f"  • MOVE: {event_dist['MOVE']:.1%}")
        
        # Predictive quality
        if 'predictive_quality' in analysis:
            quality = analysis['predictive_quality']
            if 'event_quality' in quality:
                event_quality = quality['event_quality']
                if 'confidence_stats' in event_quality:
                    conf_stats = event_quality['confidence_stats']
                    print(f"\n📊 Prediction Quality:")
                    print(f"  • Mean confidence: {conf_stats['mean_confidence']:.1%}")
                    print(f"  • Confidence range: {conf_stats['min_confidence']:.1%} - {conf_stats['max_confidence']:.1%}")
    
    def generate_training_summary(self):
        """
        Generate final training summary with enhanced behavioral insights
        Compatible with BehavioralMetrics interface
        """
        print(f"\n🎉 Enhanced Behavioral Training Summary:")
        print("=" * 60)
        
        # Summary of all collected data
        if self.action_sequences:
            print(f"📊 Total Action Sequences Analyzed: {len(self.action_sequences)}")
            
            # Event distribution summary
            all_events = []
            for seq in self.action_sequences:
                all_events.extend(seq.events)
            
            if all_events:
                event_counts = Counter(all_events)
                total_events = sum(event_counts.values())
                print(f"\n🎮 Overall Event Distribution:")
                for event, count in event_counts.most_common():
                    percentage = (count / total_events) * 100
                    print(f"  • {event}: {count} ({percentage:.1f}%)")
        
        # Mouse movement summary
        if self.mouse_patterns:
            print(f"\n🖱️  Mouse Movement Patterns:")
            if 'position_range' in self.mouse_patterns:
                pos_range = self.mouse_patterns['position_range']
                print(f"  • X range: {pos_range['x_min']:.0f} to {pos_range['x_max']:.0f}")
                print(f"  • Y range: {pos_range['y_min']:.0f} to {pos_range['y_max']:.0f}")
        
        # Click patterns summary
        if self.click_patterns:
            print(f"\n🖱️  Click Patterns:")
            if 'click_frequency' in self.click_patterns:
                click_freq = self.click_patterns['click_frequency']
                print(f"  • Average clicks per sequence: {click_freq:.2f}")
        
        # Save final analysis
        final_analysis = {
            'action_sequences': len(self.action_sequences),
            'mouse_patterns': self.mouse_patterns,
            'click_patterns': self.click_patterns,
            'keyboard_patterns': self.keyboard_patterns,
            'scroll_patterns': self.scroll_patterns,
            'game_state_correlations': self.game_state_correlations
        }
        
        self.save_enhanced_analysis(final_analysis, 'final')
        print(f"\n💾 Final training summary saved to: {self.save_dir}")
    
    def save_enhanced_analysis(self, analysis: Dict, epoch: int):
        """
        Save enhanced analysis results
        """
        filename = os.path.join(self.save_dir, f"epoch_{epoch}_enhanced_analysis.json")
        
        # Convert tensors to lists for JSON serialization
        serializable_analysis = self._make_serializable(analysis)
        
        with open(filename, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        
        print(f"💾 Enhanced analysis saved to: {filename}")
    
    def _make_serializable(self, obj):
        """
        Convert tensors and other non-serializable objects to serializable format
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
