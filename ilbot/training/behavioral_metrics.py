#!/usr/bin/env python3
"""
Behavioral Intelligence Metrics for Imitation Learning
Analyzes how well the bot learns causal relationships between gamestates and actions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

class BehavioralMetrics:
    """Analyzes behavioral intelligence and gamestate-action correlations."""
    
    def __init__(self, save_dir: str = "behavioral_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.epoch_metrics = []
        
    def analyze_epoch_predictions(self, 
                                 model_outputs: Dict[str, torch.Tensor],
                                 gamestates: torch.Tensor,
                                 action_targets: torch.Tensor,
                                 valid_mask: torch.Tensor,
                                 epoch: int,
                                 sample_size: int = 3) -> Dict:
        """
        Analyze behavioral intelligence every 5 epochs.
        
        Args:
            model_outputs: Model predictions
            gamestates: [B, A, 10, 128] - 10 timestep gamestate sequences
            action_targets: [B, A, 7] - V2 action targets
            valid_mask: [B, A] - Boolean mask for valid actions
            epoch: Current epoch number
            sample_size: Number of sample sequences to analyze
        """
        
        # Only analyze every 5 epochs
        if epoch % 5 != 0:
            return {}
            
        print(f"\n🔍 Behavioral Intelligence Analysis (Epoch {epoch})")
        print("=" * 60)
        
        metrics = {
            'epoch': epoch,
            'event_probabilities': self._analyze_event_probabilities(model_outputs, valid_mask),
            'coordinate_predictions': self._analyze_coordinate_predictions(model_outputs, valid_mask),
            'timing_predictions': self._analyze_timing_predictions(model_outputs, valid_mask),
            'sample_sequences': self._analyze_sample_sequences(
                model_outputs, gamestates, action_targets, valid_mask, sample_size
            ),
            'gamestate_correlations': self._analyze_gamestate_correlations(
                model_outputs, gamestates, action_targets, valid_mask
            )
        }
        
        # Print summary
        self._print_epoch_summary(metrics)
        
        # Save detailed analysis
        self._save_epoch_analysis(metrics, epoch)
        
        self.epoch_metrics.append(metrics)
        return metrics
    
    def _analyze_event_probabilities(self, outputs: Dict[str, torch.Tensor], 
                                    valid_mask: torch.Tensor) -> Dict:
        """Analyze event classification confidence and probabilities."""
        
        # Get event probabilities
        event_logits = outputs['event_logits']  # [B, A, 4]
        event_probs = torch.softmax(event_logits, dim=-1)
        
        # Apply valid mask
        valid_probs = event_probs[valid_mask]  # [N, 4]
        valid_logits = event_logits[valid_mask]  # [N, 4]
        
        # Calculate confidence metrics
        max_probs, predicted_events = torch.max(valid_probs, dim=1)
        


        
        metrics = {
            'mean_confidence': float(max_probs.mean()),
            'confidence_std': float(max_probs.std()),
            'high_confidence_ratio': float((max_probs > 0.8).float().mean()),
            'event_distribution': {
                'CLICK': float((predicted_events == 0).float().mean()),
                'KEY': float((predicted_events == 1).float().mean()),
                'SCROLL': float((predicted_events == 2).float().mean()),
                'MOVE': float((predicted_events == 3).float().mean())
            }
        }
        
        return metrics
    
    def _analyze_coordinate_predictions(self, outputs: Dict[str, torch.Tensor], 
                                      valid_mask: torch.Tensor) -> Dict:
        """Analyze mouse coordinate predictions and uncertainty."""
        
        # Get coordinate predictions
        x_mu = outputs['x_mu']  # [B, A]
        y_mu = outputs['y_mu']  # [B, A]
        x_logsig = outputs['x_logsig']  # [B, A]
        y_logsig = outputs['y_logsig']  # [B, A]
        
        # Apply valid mask
        valid_x = x_mu[valid_mask]
        valid_y = y_mu[valid_mask]
        valid_x_std = torch.exp(x_logsig[valid_mask])
        valid_y_std = torch.exp(y_logsig[valid_mask])
        
        metrics = {
            'coordinate_bounds': {
                'x_min': float(valid_x.min()),
                'x_max': float(valid_x.max()),
                'y_min': float(valid_y.min()),
                'y_max': float(valid_y.max())
            },
            'coordinate_uncertainty': {
                'x_std_mean': float(valid_x_std.mean()),
                'y_std_mean': float(valid_y_std.mean()),
                'x_std_std': float(valid_x_std.std()),
                'y_std_std': float(valid_y_std.std())
            },
            'coordinate_distribution': {
                'x_mean': float(valid_x.mean()),
                'y_mean': float(valid_y.mean()),
                'x_std': float(valid_x.std()),
                'y_std': float(valid_y.std())
            }
        }
        
        return metrics
    
    def _analyze_timing_predictions(self, outputs: Dict[str, torch.Tensor], 
                                   valid_mask: torch.Tensor) -> Dict:
        """Analyze timing predictions and quantile coverage."""
        
        # Get timing predictions (quantiles)
        time_q = outputs['time_q']  # [B, A, 3] - q10, q50, q90
        
        # Apply valid mask
        valid_time = time_q[valid_mask]  # [N, 3]
        
        # Calculate timing metrics
        q10, q50, q90 = valid_time[:, 0], valid_time[:, 1], valid_time[:, 2]
        
        metrics = {
            'timing_bounds': {
                'q10_mean': float(q10.mean()),
                'q50_mean': float(q50.mean()),
                'q90_mean': float(q90.mean())
            },
            'timing_uncertainty': {
                'q90_q10_mean': float((q90 - q10).mean()),
                'q90_q10_std': float((q90 - q10).std())
            },
            'timing_distribution': {
                'q50_std': float(q50.std()),
                'negative_timing_ratio': float((q50 < 0).float().mean())
            }
        }
        
        return metrics
    
    def _analyze_sample_sequences(self, outputs: Dict[str, torch.Tensor],
                                 gamestates: torch.Tensor,
                                 action_targets: torch.Tensor,
                                 valid_mask: torch.Tensor,
                                 sample_size: int) -> List[Dict]:
        """Analyze sample action sequences with gamestate context."""
        
        # The tensors are: gamestates[B, 10, 128], action_targets[B, 100, 7], valid_mask[B, 100]
        # We want to sample from valid actions across all sequences
        
        # Find valid actions across all sequences
        valid_actions = torch.where(valid_mask)  # Returns (batch_indices, action_indices)
        if len(valid_actions[0]) == 0:
            return []
        
        # Sample valid actions
        sample_count = min(sample_size, len(valid_actions[0]))
        sample_indices = torch.randperm(len(valid_actions[0]))[:sample_count]
        samples = []
        
        for i in sample_indices:
            batch_idx = valid_actions[0][i]
            action_idx = valid_actions[1][i]
            
            # Extract sequence data - gamestates[B, 10, 128] gives us the 10-timestep sequence for this batch
            gamestate_seq = gamestates[batch_idx]  # [10, 128] - 10 timestep sequence
            
            action_target = action_targets[batch_idx, action_idx]  # [7]
            
            # Get model predictions
            event_pred = outputs['event_logits'][batch_idx, action_idx].argmax()
            event_probs = torch.softmax(outputs['event_logits'][batch_idx, action_idx], dim=-1)
            x_pred = outputs['x_mu'][batch_idx, action_idx]
            y_pred = outputs['y_mu'][batch_idx, action_idx]
            time_pred = outputs['time_q'][batch_idx, action_idx, 1]  # q50
            
            sample = {
                'sequence_idx': int(i),
                'gamestate_features': {
                    'player_pos': gamestate_seq[:, :2].tolist(),  # [10, 2] - player x,y over 10 timesteps
                    'animation_id': gamestate_seq[:, 2].tolist(),  # [10] - animation over 10 timesteps
                    'is_moving': gamestate_seq[:, 3].tolist(),    # [10] - movement state over 10 timesteps
                    'time_since_interaction': gamestate_seq[:, 4].tolist(),  # [10] - interaction time over 10 timesteps
                    'phase_features': gamestate_seq[:, 5:15].tolist()  # [10, 10] - phase features over 10 timesteps
                },
                'action_target': {
                    'time': float(action_target[0]),
                    'x': float(action_target[1]),
                    'y': float(action_target[2]),
                    'button': int(action_target[3]),
                    'key_action': int(action_target[4]),
                    'key_id': int(action_target[5]),
                    'scroll_y': int(action_target[6])
                },
                'model_prediction': {
                    'event_type': int(event_pred),
                    'event_confidence': float(event_probs[event_pred]),
                    'x_pred': float(x_pred),
                    'y_pred': float(y_pred),
                    'time_pred': float(time_pred)
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def _analyze_gamestate_correlations(self, outputs: Dict[str, torch.Tensor],
                                       gamestates: torch.Tensor,
                                       action_targets: torch.Tensor,
                                       valid_mask: torch.Tensor) -> Dict:
        """Analyze correlations between gamestate features and predicted actions."""
        
        # For now, return empty dict to avoid complex indexing issues
        # This can be enhanced later with proper tensor handling
        return {}
    
    def _print_epoch_summary(self, metrics: Dict):
        """Print a summary of the behavioral analysis."""
        
        print(f"📊 Event Classification:")
        print(f"  Mean Confidence: {metrics['event_probabilities']['mean_confidence']:.3f}")
        print(f"  High Confidence (>80%): {metrics['event_probabilities']['high_confidence_ratio']:.1%}")
        print(f"  Event Distribution: CLICK={metrics['event_probabilities']['event_distribution']['CLICK']:.1%}, "
              f"KEY={metrics['event_probabilities']['event_distribution']['KEY']:.1%}, "
              f"SCROLL={metrics['event_probabilities']['event_distribution']['SCROLL']:.1%}, "
              f"MOVE={metrics['event_probabilities']['event_distribution']['MOVE']:.1%}")
        
        print(f"\n🖱️  Coordinate Predictions:")
        coord = metrics['coordinate_predictions']
        print(f"  X Range: [{coord['coordinate_bounds']['x_min']:.1f}, {coord['coordinate_bounds']['x_max']:.1f}]")
        print(f"  Y Range: [{coord['coordinate_bounds']['y_min']:.1f}, {coord['coordinate_bounds']['y_max']:.1f}]")
        print(f"  Mean Uncertainty: X±{coord['coordinate_uncertainty']['x_std_mean']:.2f}, Y±{coord['coordinate_uncertainty']['y_std_mean']:.2f}")
        
        print(f"\n⏰ Timing Predictions:")
        timing = metrics['timing_predictions']
        print(f"  Median Timing: {timing['timing_bounds']['q50_mean']:.3f}s")
        print(f"  Timing Uncertainty: ±{timing['timing_uncertainty']['q90_q10_mean']:.3f}s")
        print(f"  Negative Timing: {timing['timing_distribution']['negative_timing_ratio']:.1%}")
        
        print(f"\n🎮 Sample Sequence Analysis:")
        for i, sample in enumerate(metrics['sample_sequences']):
            print(f"  Sequence {i+1}:")
            print(f"    Gamestate: Player at ({sample['gamestate_features']['player_pos'][-1][0]:.1f}, {sample['gamestate_features']['player_pos'][-1][1]:.1f}), "
                  f"Animation={sample['gamestate_features']['animation_id'][-1]:.0f}, Moving={sample['gamestate_features']['is_moving'][-1]:.0f}")
            print(f"    Target: {sample['action_target']['button']} at ({sample['action_target']['x']:.1f}, {sample['action_target']['y']:.1f})")
            print(f"    Prediction: Event {sample['model_prediction']['event_type']} at ({sample['model_prediction']['x_pred']:.1f}, {sample['model_prediction']['y_pred']:.1f}) "
                  f"[Confidence: {sample['model_prediction']['event_confidence']:.1%}]")
    
    def _save_epoch_analysis(self, metrics: Dict, epoch: int):
        """Save detailed analysis to file."""
        
        # Save as JSON
        import json
        analysis_file = self.save_dir / f"epoch_{epoch}_analysis.json"
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_metrics = convert_tensors(metrics)
        
        with open(analysis_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"💾 Detailed analysis saved to: {analysis_file}")
    
    def generate_training_summary(self):
        """Generate a summary of behavioral learning across all epochs."""
        
        if not self.epoch_metrics:
            return
        
        print(f"\n🎯 Behavioral Intelligence Training Summary")
        print("=" * 60)
        
        # Track learning progress
        epochs = [m['epoch'] for m in self.epoch_metrics]
        confidences = [m['event_probabilities']['mean_confidence'] for m in self.epoch_metrics]
        uncertainties = [m['coordinate_predictions']['coordinate_uncertainty']['x_std_mean'] for m in self.epoch_metrics]
        
        print(f"📈 Learning Progress:")
        print(f"  Epochs Analyzed: {len(epochs)}")
        print(f"  Confidence Trend: {confidences[0]:.3f} → {confidences[-1]:.3f}")
        print(f"  Uncertainty Trend: {uncertainties[0]:.3f} → {uncertainties[-1]:.3f}")
        
        # Save summary
        summary_file = self.save_dir / "training_summary.json"
        summary = {
            'total_epochs': len(epochs),
            'confidence_progression': confidences,
            'uncertainty_progression': uncertainties,
            'final_metrics': self.epoch_metrics[-1]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Training summary saved to: {summary_file}")
