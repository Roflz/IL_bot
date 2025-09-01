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
                
                if valid_time_preds.numel() > 0:
                    analysis['timing'] = {
                        'median_timing': self._safe_mean(valid_time_preds[:, 1]),  # q0.5
                        'timing_uncertainty': self._safe_mean(valid_time_preds[:, 2] - valid_time_preds[:, 0]),  # q0.9 - q0.1
                        'timing_consistency': self._safe_std(valid_time_preds[:, 1]),
                    }
                else:
                    analysis['timing'] = {
                        'median_timing': 0.0,
                        'timing_uncertainty': 0.0,
                        'timing_consistency': 0.0,
                    }
                
                print(f"⏱️  Timing Analysis:")
                print(f"  • Median timing: {analysis['timing']['median_timing']:.2f}s (q0.5 quantile)")
                print(f"  • Timing uncertainty: {analysis['timing']['timing_uncertainty']:.2f}s (q0.9 - q0.1 range)")
                print(f"  • Timing consistency: {analysis['timing']['timing_consistency']:.2f}s (std dev across predictions)")
            
            # Basic mouse position analysis
            if 'x_mu' in model_outputs and 'y_mu' in model_outputs:
                x_pred = model_outputs['x_mu']  # [B, A]
                y_pred = model_outputs['y_mu']  # [B, A]
                
                # Apply valid mask to get only valid predictions
                valid_x_pred = x_pred[valid_mask]  # [N_valid]
                valid_y_pred = y_pred[valid_mask]  # [N_valid]
                
                if valid_x_pred.numel() > 0 and valid_y_pred.numel() > 0:
                    x_min, x_max = self._safe_min_max(valid_x_pred)
                    y_min, y_max = self._safe_min_max(valid_y_pred)
                    
                    analysis['mouse'] = {
                        'x_range': [x_min, x_max],
                        'y_range': [y_min, y_max],
                        'x_mean': self._safe_mean(valid_x_pred),
                        'y_mean': self._safe_mean(valid_y_pred),
                        'x_std': self._safe_std(valid_x_pred),
                        'y_std': self._safe_std(valid_y_pred),
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
                    }
                
                print(f"\n🖱️  Mouse Position Analysis:")
                print(f"  • X range: {analysis['mouse']['x_range'][0]:.0f} to {analysis['mouse']['x_range'][1]:.0f} (min/max across valid predictions)")
                print(f"  • Y range: {analysis['mouse']['y_range'][0]:.0f} to {analysis['mouse']['y_range'][1]:.0f} (min/max across valid predictions)")
                print(f"  • Mean position: ({analysis['mouse']['x_mean']:.0f}, {analysis['mouse']['y_mean']:.0f}) (average across valid predictions)")
                print(f"  • Position spread: X={analysis['mouse']['x_std']:.1f}, Y={analysis['mouse']['y_std']:.1f} (std dev across valid predictions)")
                if 'x_uncertainty' in analysis['mouse']:
                    print(f"  • Position uncertainty: X={analysis['mouse']['x_uncertainty']:.1f}, Y={analysis['mouse']['y_uncertainty']:.1f} (predicted std dev from model)")
                
                # Add warning if mouse predictions seem off
                x_range = analysis['mouse']['x_range'][1] - analysis['mouse']['x_range'][0]
                y_range = analysis['mouse']['y_range'][1] - analysis['mouse']['y_range'][0]
                if x_range < 10 or y_range < 10:
                    print(f"  ⚠️  WARNING: Mouse predictions have very little variation (X range: {x_range:.1f}, Y range: {y_range:.1f})")
                    print(f"      Expected: X ~188-1708, Y ~7-860 (actual data ranges)")
                    print(f"      This suggests the model isn't learning proper mouse coordinate scales")
            
            # Basic event distribution analysis
            if 'event_logits' in model_outputs:
                event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)  # [B, A, 4]
                
                # Apply valid mask and average event probabilities across valid predictions only
                valid_event_probs = event_probs[valid_mask]  # [N_valid, 4]
                
                if valid_event_probs.numel() > 0:
                    mean_event_probs = valid_event_probs.mean(dim=0)  # Average across valid actions
                    
                    analysis['events'] = {
                        'CLICK': float(mean_event_probs[0].item()),
                        'KEY': float(mean_event_probs[1].item()),
                        'SCROLL': float(mean_event_probs[2].item()),
                        'MOVE': float(mean_event_probs[3].item()),
                    }
                else:
                    analysis['events'] = {
                        'CLICK': 0.0,
                        'KEY': 0.0,
                        'SCROLL': 0.0,
                        'MOVE': 0.0,
                    }
                
                print(f"\n🎯 Event Distribution:")
                for event_type, prob in analysis['events'].items():
                    print(f"  • {event_type}: {prob:.1%} (softmax probability)")
            
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
                print(f"  • Position range: X={px_min:.0f} to {px_max:.0f}, Y={py_min:.0f} to {py_max:.0f} (min/max across all timesteps)")
                print(f"  • Mean position: ({analysis['player']['position_mean']['x']:.0f}, {analysis['player']['position_mean']['y']:.0f}) (average across all timesteps)")
            
            # Valid mask analysis
            if valid_mask.numel() > 0:
                valid_ratio = float(valid_mask.float().mean().item())
                analysis['data_quality'] = {
                    'valid_action_ratio': valid_ratio,
                    'total_actions': int(valid_mask.numel()),
                    'valid_actions': int(valid_mask.sum().item()),
                }
                
                print(f"\n📊 Data Quality:")
                print(f"  • Valid actions: {analysis['data_quality']['valid_actions']}/{analysis['data_quality']['total_actions']} ({valid_ratio:.1%}) (non-padding action rows)")
            
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
