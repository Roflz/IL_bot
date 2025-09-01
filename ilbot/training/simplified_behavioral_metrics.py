#!/usr/bin/env python3
"""
Simplified Behavioral Metrics for analyzing bot intelligence
Avoids complex tensor indexing issues while providing essential analysis
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import json

class SimplifiedBehavioralMetrics:
    """
    Simplified behavioral analysis that provides basic insights without complex tensor operations
    """
    
    def __init__(self, save_dir: str = "simplified_behavioral_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.analysis_history = []
    
    def _safe_mean(self, tensor, default=0.0):
        """Safely compute mean of tensor"""
        try:
            if tensor.numel() == 0:
                return default
            return float(tensor.mean().item())
        except:
            return default
    
    def _safe_std(self, tensor, default=0.0):
        """Safely compute std of tensor"""
        try:
            if tensor.numel() == 0:
                return default
            return float(tensor.std().item())
        except:
            return default
    
    def _safe_min_max(self, tensor, default=(0.0, 0.0)):
        """Safely compute min and max of tensor"""
        try:
            if tensor.numel() == 0:
                return default
            return float(tensor.min().item()), float(tensor.max().item())
        except:
            return default
    
    def analyze_epoch_predictions(self, 
                                model_outputs: Dict[str, torch.Tensor],
                                gamestates: torch.Tensor,
                                action_targets: torch.Tensor,
                                valid_mask: torch.Tensor,
                                epoch: int) -> Dict:
        """
        Simplified behavioral analysis for epoch predictions
        """
        try:
            print(f"\n🔍 Simplified Behavioral Analysis (Epoch {epoch}):")
            print("=" * 60)
            
            analysis = {}
            
            # Basic timing analysis
            if 'time_q' in model_outputs:
                time_preds = model_outputs['time_q']  # [B, A, 3]
                # Apply valid mask to get only valid predictions
                valid_time_preds = time_preds[valid_mask]  # [N_valid, 3]
                
                # Get target timing data (column 0 is time delta)
                valid_time_targets = action_targets[valid_mask][:, 0]  # [N_valid]
                
                if valid_time_preds.numel() > 0 and valid_time_targets.numel() > 0:
                    analysis['timing'] = {
                        'median_timing': self._safe_mean(valid_time_preds[:, 1]),  # q0.5
                        'timing_uncertainty': self._safe_mean(valid_time_preds[:, 2] - valid_time_preds[:, 0]),  # q0.9 - q0.1
                        'timing_consistency': self._safe_std(valid_time_preds[:, 1]),
                        'target_median_timing': self._safe_mean(valid_time_targets),
                        'target_timing_consistency': self._safe_std(valid_time_targets),
                    }
                else:
                    analysis['timing'] = {
                        'median_timing': 0.0,
                        'timing_uncertainty': 0.0,
                        'timing_consistency': 0.0,
                        'target_median_timing': 0.0,
                        'target_timing_consistency': 0.0,
                    }
                
                print(f"⏱️  Timing Analysis:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ Median Timing (s)       │ {analysis['timing']['median_timing']:>10.2f} │ {analysis['timing']['target_median_timing']:>10.2f} │ {abs(analysis['timing']['median_timing'] - analysis['timing']['target_median_timing']):>10.2f} │")
                print(f"  │ Uncertainty (s)         │ {analysis['timing']['timing_uncertainty']:>10.2f} │ {'N/A':>10} │ {'N/A':>10} │")
                print(f"  │ Consistency (std dev)   │ {analysis['timing']['timing_consistency']:>10.2f} │ {analysis['timing']['target_timing_consistency']:>10.2f} │ {abs(analysis['timing']['timing_consistency'] - analysis['timing']['target_timing_consistency']):>10.2f} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Basic mouse position analysis
            if 'x_mu' in model_outputs and 'y_mu' in model_outputs:
                x_pred = model_outputs['x_mu']  # [B, A]
                y_pred = model_outputs['y_mu']  # [B, A]
                
                # Apply valid mask to get only valid predictions
                valid_x_pred = x_pred[valid_mask]  # [N_valid]
                valid_y_pred = y_pred[valid_mask]  # [N_valid]
                
                # Get target mouse position data (columns 1 and 2 are X and Y coordinates)
                valid_x_targets = action_targets[valid_mask][:, 1]  # [N_valid]
                valid_y_targets = action_targets[valid_mask][:, 2]  # [N_valid]
                
                if valid_x_pred.numel() > 0 and valid_y_pred.numel() > 0 and valid_x_targets.numel() > 0 and valid_y_targets.numel() > 0:
                    x_min, x_max = self._safe_min_max(valid_x_pred)
                    y_min, y_max = self._safe_min_max(valid_y_pred)
                    x_target_min, x_target_max = self._safe_min_max(valid_x_targets)
                    y_target_min, y_target_max = self._safe_min_max(valid_y_targets)
                    
                    analysis['mouse'] = {
                        'x_range': [x_min, x_max],
                        'y_range': [y_min, y_max],
                        'x_mean': self._safe_mean(valid_x_pred),
                        'y_mean': self._safe_mean(valid_y_pred),
                        'x_std': self._safe_std(valid_x_pred),
                        'y_std': self._safe_std(valid_y_pred),
                        'x_target_range': [x_target_min, x_target_max],
                        'y_target_range': [y_target_min, y_target_max],
                        'x_target_mean': self._safe_mean(valid_x_targets),
                        'y_target_mean': self._safe_mean(valid_y_targets),
                        'x_target_std': self._safe_std(valid_x_targets),
                        'y_target_std': self._safe_std(valid_y_targets),
                    }
                    
                    if 'x_logsig' in model_outputs and 'y_logsig' in model_outputs:
                        x_uncertainty = model_outputs['x_logsig'][valid_mask].exp()
                        y_uncertainty = model_outputs['y_logsig'][valid_mask].exp()
                        
                        analysis['mouse']['x_uncertainty'] = self._safe_mean(x_uncertainty)
                        analysis['mouse']['y_uncertainty'] = self._safe_mean(y_uncertainty)
                else:
                    analysis['mouse'] = {
                        'x_range': [0.0, 0.0],
                        'y_range': [0.0, 0.0],
                        'x_mean': 0.0,
                        'y_mean': 0.0,
                        'x_std': 0.0,
                        'y_std': 0.0,
                        'x_uncertainty': 0.0,
                        'y_uncertainty': 0.0,
                        'x_target_range': [0.0, 0.0],
                        'y_target_range': [0.0, 0.0],
                        'x_target_mean': 0.0,
                        'y_target_mean': 0.0,
                        'x_target_std': 0.0,
                        'y_target_std': 0.0,
                    }
                
                print(f"\n🖱️  Mouse Position Analysis:")
                
                # Denormalize coordinates for display (assuming 1920x1080 screen)
                screen_width, screen_height = 1920.0, 1080.0
                
                # Predicted coordinates
                x_screen_min = analysis['mouse']['x_range'][0] * screen_width
                x_screen_max = analysis['mouse']['x_range'][1] * screen_width
                y_screen_min = analysis['mouse']['y_range'][0] * screen_height
                y_screen_max = analysis['mouse']['y_range'][1] * screen_height
                x_screen_mean = analysis['mouse']['x_mean'] * screen_width
                y_screen_mean = analysis['mouse']['y_mean'] * screen_height
                x_screen_std = analysis['mouse']['x_std'] * screen_width
                y_screen_std = analysis['mouse']['y_std'] * screen_height
                
                # Target coordinates
                x_target_screen_min = analysis['mouse']['x_target_range'][0] * screen_width
                x_target_screen_max = analysis['mouse']['x_target_range'][1] * screen_width
                y_target_screen_min = analysis['mouse']['y_target_range'][0] * screen_height
                y_target_screen_max = analysis['mouse']['y_target_range'][1] * screen_height
                x_target_screen_mean = analysis['mouse']['x_target_mean'] * screen_width
                y_target_screen_mean = analysis['mouse']['y_target_mean'] * screen_height
                x_target_screen_std = analysis['mouse']['x_target_std'] * screen_width
                y_target_screen_std = analysis['mouse']['y_target_std'] * screen_height
                
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ X Mean (pixels)         │ {x_screen_mean:>10.0f} │ {x_target_screen_mean:>10.0f} │ {abs(x_screen_mean - x_target_screen_mean):>10.1f} │")
                print(f"  │ Y Mean (pixels)         │ {y_screen_mean:>10.0f} │ {y_target_screen_mean:>10.0f} │ {abs(y_screen_mean - y_target_screen_mean):>10.1f} │")
                print(f"  │ X Range (pixels)        │ {x_screen_max-x_screen_min:>10.0f} │ {x_target_screen_max-x_target_screen_min:>10.0f} │ {abs((x_screen_max-x_screen_min) - (x_target_screen_max-x_target_screen_min)):>10.0f} │")
                print(f"  │ Y Range (pixels)        │ {y_screen_max-y_screen_min:>10.0f} │ {y_target_screen_max-y_target_screen_min:>10.0f} │ {abs((y_screen_max-y_screen_min) - (y_target_screen_max-y_target_screen_min)):>10.0f} │")
                print(f"  │ X Spread (std dev)      │ {x_screen_std:>10.1f} │ {x_target_screen_std:>10.1f} │ {abs(x_screen_std - x_target_screen_std):>10.1f} │")
                print(f"  │ Y Spread (std dev)      │ {y_screen_std:>10.1f} │ {y_target_screen_std:>10.1f} │ {abs(y_screen_std - y_target_screen_std):>10.1f} │")
                if 'x_uncertainty' in analysis['mouse']:
                    x_uncertainty_screen = analysis['mouse']['x_uncertainty'] * screen_width
                    y_uncertainty_screen = analysis['mouse']['y_uncertainty'] * screen_height
                    print(f"  │ X Uncertainty (pixels)  │ {x_uncertainty_screen:>10.1f} │ {'N/A':>10} │ {'N/A':>10} │")
                    print(f"  │ Y Uncertainty (pixels)  │ {y_uncertainty_screen:>10.1f} │ {'N/A':>10} │ {'N/A':>10} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                
                # Add warning if mouse predictions seem off
                x_range = analysis['mouse']['x_range'][1] - analysis['mouse']['x_range'][0]
                y_range = analysis['mouse']['y_range'][1] - analysis['mouse']['y_range'][0]
                x_range_screen = (analysis['mouse']['x_range'][1] - analysis['mouse']['x_range'][0]) * screen_width
                y_range_screen = (analysis['mouse']['y_range'][1] - analysis['mouse']['y_range'][0]) * screen_height
                
                if x_range < 0.01 or y_range < 0.01:  # Very small variation in normalized coordinates
                    print(f"  ⚠️  WARNING: Mouse predictions have very little variation")
                    print(f"      X range: {x_range_screen:.0f} pixels (normalized: {x_range:.3f})")
                    print(f"      Y range: {y_range_screen:.0f} pixels (normalized: {y_range:.3f})")
                    print(f"      Expected: X ~200-1700 pixels, Y ~100-900 pixels (screen coordinate ranges)")
                    print(f"      This suggests the model isn't learning diverse mouse positions")
            
            # Basic event distribution analysis
            if 'event_logits' in model_outputs:
                event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)  # [B, A, 4]
                
                # Apply valid mask and average event probabilities across valid predictions only
                valid_event_probs = event_probs[valid_mask]  # [N_valid, 4]
                
                # Derive actual event types from action targets
                valid_actions = action_targets[valid_mask]  # [N_valid, 7]
                # action_targets format: [time, x, y, button, key_action, key_id, scroll]
                click_actions = (valid_actions[:, 3] > 0).float()  # button > 0
                key_actions = (valid_actions[:, 4] > 0).float()    # key_action > 0
                scroll_actions = (valid_actions[:, 6] != 0).float()  # scroll != 0
                move_actions = ((valid_actions[:, 3] == 0) & (valid_actions[:, 4] == 0) & (valid_actions[:, 6] == 0)).float()
                
                if valid_event_probs.numel() > 0:
                    mean_event_probs = valid_event_probs.mean(dim=0)  # Average across valid actions
                    
                    # Calculate target event distribution
                    total_valid = valid_actions.shape[0]
                    target_event_dist = {
                        'CLICK': float(click_actions.sum().item() / total_valid) if total_valid > 0 else 0.0,
                        'KEY': float(key_actions.sum().item() / total_valid) if total_valid > 0 else 0.0,
                        'SCROLL': float(scroll_actions.sum().item() / total_valid) if total_valid > 0 else 0.0,
                        'MOVE': float(move_actions.sum().item() / total_valid) if total_valid > 0 else 0.0,
                    }
                    
                    analysis['events'] = {
                        'CLICK': float(mean_event_probs[0].item()),
                        'KEY': float(mean_event_probs[1].item()),
                        'SCROLL': float(mean_event_probs[2].item()),
                        'MOVE': float(mean_event_probs[3].item()),
                    }
                    analysis['target_events'] = target_event_dist
                else:
                    analysis['events'] = {
                        'CLICK': 0.0,
                        'KEY': 0.0,
                        'SCROLL': 0.0,
                        'MOVE': 0.0,
                    }
                    analysis['target_events'] = {
                        'CLICK': 0.0,
                        'KEY': 0.0,
                        'SCROLL': 0.0,
                        'MOVE': 0.0,
                    }
                
                print(f"\n🎯 Event Distribution:")
                print(f"  ┌─────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Event Type  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────┼──────────────┼──────────────┼──────────────┤")
                event_names = ['CLICK', 'KEY', 'SCROLL', 'MOVE']
                for i, event_type in enumerate(event_names):
                    pred_prob = analysis['events'][event_type]
                    target_prob = analysis['target_events'][event_type]
                    accuracy = abs(pred_prob - target_prob)
                    print(f"  │ {event_type:<11} │ {pred_prob:>10.1%} │ {target_prob:>10.1%} │ {accuracy:>10.1%} │")
                print(f"  └─────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Basic gamestate analysis
            if gamestates.numel() > 0:
                # Player position analysis (assume first 2 features are x, y)
                player_positions = gamestates[:, :, :2]
                
                px_min, px_max = self._safe_min_max(player_positions[:, :, 0])
                py_min, py_max = self._safe_min_max(player_positions[:, :, 1])
                
                analysis['player'] = {
                    'position_range': {
                        'x_min': px_min,
                        'x_max': px_max,
                        'y_min': py_min,
                        'y_max': py_max,
                    },
                    'position_mean': {
                        'x': self._safe_mean(player_positions[:, :, 0]),
                        'y': self._safe_mean(player_positions[:, :, 1]),
                    }
                }
                
                print(f"\n🚶 Player Position Analysis:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ X Value      │ Y Value      │ Range        │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ Mean Position           │ {analysis['player']['position_mean']['x']:>10.0f} │ {analysis['player']['position_mean']['y']:>10.0f} │ {'N/A':>10} │")
                print(f"  │ Min Position            │ {px_min:>10.0f} │ {py_min:>10.0f} │ {'N/A':>10} │")
                print(f"  │ Max Position            │ {px_max:>10.0f} │ {py_max:>10.0f} │ {'N/A':>10} │")
                print(f"  │ Position Range          │ {px_max-px_min:>10.0f} │ {py_max-py_min:>10.0f} │ {'N/A':>10} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Valid mask analysis
            if valid_mask.numel() > 0:
                valid_ratio = float(valid_mask.float().mean().item())
                analysis['data_quality'] = {
                    'valid_action_ratio': valid_ratio,
                    'total_actions': int(valid_mask.numel()),
                    'valid_actions': int(valid_mask.sum().item()),
                }
                
                print(f"\n📊 Data Quality:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Count        │ Total        │ Percentage   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ Valid Actions           │ {analysis['data_quality']['valid_actions']:>10} │ {analysis['data_quality']['total_actions']:>10} │ {valid_ratio:>10.1%} │")
                print(f"  │ Padding Actions         │ {analysis['data_quality']['total_actions'] - analysis['data_quality']['valid_actions']:>10} │ {analysis['data_quality']['total_actions']:>10} │ {1-valid_ratio:>10.1%} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Store analysis
            analysis['epoch'] = epoch
            self.analysis_history.append(analysis)
            
            # Save to file
            self._save_analysis(analysis, epoch)
            
            print("=" * 60)
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            print(f"⚠️  Simplified analysis failed: {error_msg}")
            return {'error': error_msg, 'epoch': epoch}
    
    def _save_analysis(self, analysis: Dict, epoch: int):
        """Save analysis to file"""
        try:
            filepath = os.path.join(self.save_dir, f"epoch_{epoch:03d}_analysis.json")
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save analysis: {e}")
    
    def generate_training_summary(self, epoch: int) -> Dict:
        """Generate training summary (compatibility method)"""
        if self.analysis_history:
            latest = self.analysis_history[-1]
            return {
                'epoch': epoch,
                'summary': latest,
                'insights': [
                    f"Latest timing: {latest.get('timing', {}).get('median_timing', 0):.2f}s",
                    f"Mouse range: X={latest.get('mouse', {}).get('x_range', [0, 0])}",
                    f"Event distribution: {latest.get('events', {})}",
                ]
            }
        else:
            return {'epoch': epoch, 'summary': {}, 'insights': ['No analysis available']}
    
    def get_latest_insights(self) -> List[str]:
        """Get latest behavioral insights"""
        if not self.analysis_history:
            return ["No behavioral data available yet"]
        
        latest = self.analysis_history[-1]
        insights = []
        
        # Timing insights
        if 'timing' in latest:
            timing = latest['timing']
            if timing['median_timing'] > 2.0:
                insights.append("⚠️  Bot is predicting slow actions (>2s)")
            elif timing['median_timing'] < 0.5:
                insights.append("⚡ Bot is predicting fast actions (<0.5s)")
            else:
                insights.append("✅ Bot timing predictions look reasonable")
        
        # Mouse insights
        if 'mouse' in latest:
            mouse = latest['mouse']
            x_range = mouse['x_range'][1] - mouse['x_range'][0]
            y_range = mouse['y_range'][1] - mouse['y_range'][0]
            
            if x_range > 1000 or y_range > 1000:
                insights.append("📏 Bot predicts wide mouse movement range")
            else:
                insights.append("🎯 Bot predicts focused mouse movement")
        
        # Event insights
        if 'events' in latest:
            events = latest['events']
            dominant_event = max(events.items(), key=lambda x: x[1])
            
            if dominant_event[1] > 0.8:
                insights.append(f"⚠️  Bot heavily favors {dominant_event[0]} events ({dominant_event[1]:.1%})")
            else:
                insights.append("✅ Bot shows balanced event predictions")
        
        return insights
