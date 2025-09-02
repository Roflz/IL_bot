#!/usr/bin/env python3
"""
Clean and Simple Behavioral Metrics for analyzing bot intelligence
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

class SimplifiedBehavioralMetrics:
    """
    Clean behavioral analysis that provides essential insights
    """
    
    def __init__(self, save_dir: str = "behavioral_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def analyze_epoch_predictions(self, model_outputs: Dict, gamestates: torch.Tensor, 
                                action_targets: torch.Tensor, valid_mask: torch.Tensor, 
                                epoch: int, save_visualizations: bool = True, 
                                save_predictions: bool = True, save_timing_graphs: bool = True):
        """Analyze model predictions and generate clean metrics"""
        
        # Extract basic data
        batch_size = action_targets.shape[0]
        
        # Get valid actions
        valid_actions = action_targets[valid_mask]
        num_valid_actions = valid_actions.shape[0]
        
        # Save predictions if requested (save sample predictions from behavioral analysis)
        if save_predictions:
            self._save_predictions(model_outputs, action_targets, valid_mask, epoch)
            
            # Basic timing analysis
        if 'time_deltas' in model_outputs:
            time_deltas = model_outputs['time_deltas'][valid_mask]
            pred_timing_mean = float(time_deltas.mean().item())
            pred_timing_std = float(time_deltas.std().item())
        else:
            pred_timing_mean = 0.0
            pred_timing_std = 0.0
        
        # Target timing (from action_targets column 0)
        target_timing = valid_actions[:, 0]
        target_timing_mean = float(target_timing.mean().item())
        target_timing_std = float(target_timing.std().item())
        
        # Actions per gamestate
        actions_per_gamestate = valid_mask.sum(dim=1).float()
        target_actions_mean = float(actions_per_gamestate.mean().item())
        target_actions_std = float(actions_per_gamestate.std().item())
        
        # Predicted actions per gamestate (from sequence length)
        if 'sequence_length' in model_outputs:
            pred_actions = model_outputs['sequence_length'].squeeze()
            pred_actions_mean = float(pred_actions.mean().item())
        else:
            pred_actions_mean = 0.0
        
        # Event distribution analysis
        if 'event_logits' in model_outputs:
            event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)
            event_predictions = event_probs.argmax(dim=-1)
            
            # Count event types for valid actions
            valid_event_preds = event_predictions[valid_mask]
            event_counts = torch.bincount(valid_event_preds, minlength=4)
            event_percentages = (event_counts.float() / event_counts.sum() * 100).tolist()
        else:
            event_percentages = [0.0, 0.0, 0.0, 0.0]
        
        # Target event distribution (derive from action targets)
        target_events = self._derive_event_types(valid_actions)
        target_event_counts = torch.bincount(target_events, minlength=4)
        target_event_percentages = (target_event_counts.float() / target_event_counts.sum() * 100).tolist()
        
        # Print clean metrics
        self._print_metrics(epoch, pred_timing_mean, pred_timing_std, target_timing_mean, target_timing_std,
                          pred_actions_mean, target_actions_mean, target_actions_std,
                          event_percentages, target_event_percentages)
        
        # Save visualizations if requested
        if save_visualizations:
            self._save_visualizations(epoch, target_timing, time_deltas if 'time_deltas' in model_outputs else None,
                                    actions_per_gamestate, pred_actions if 'sequence_length' in model_outputs else None,
                                    save_timing_graphs=save_timing_graphs)
    
    def _derive_event_types(self, valid_actions: torch.Tensor) -> torch.Tensor:
        """Derive event types from action targets"""
        # Extract components
        button = valid_actions[:, 3]  # Button column
        key_action = valid_actions[:, 4]  # Key action column  
        scroll_y = valid_actions[:, 6]  # Scroll column
        
        # Initialize with MOVE (0)
        event_types = torch.zeros(valid_actions.shape[0], dtype=torch.long)
        
        # Priority: CLICK > KEY > SCROLL > MOVE
        # SCROLL: scroll_y != 0
        event_types[scroll_y != 0] = 2
        
        # KEY: key_action != 0 (overwrites SCROLL)
        event_types[key_action != 0] = 1
        
        # CLICK: button != 0 (overwrites KEY and SCROLL)
        event_types[button != 0] = 3
        
        return event_types
    
    def _print_metrics(self, epoch: int, pred_timing_mean: float, pred_timing_std: float,
                      target_timing_mean: float, target_timing_std: float,
                      pred_actions_mean: float, target_actions_mean: float, target_actions_std: float,
                      event_percentages: List[float], target_event_percentages: List[float]):
        """Print clean, organized metrics"""
        
        print(f"\n📊 Epoch {epoch} Analysis:")
        print("=" * 60)
        
        # Timing Analysis
        print(f"\n⏱️  Timing Analysis:")
        print(f"  Predicted: {pred_timing_mean:.3f}s ± {pred_timing_std:.3f}s")
        print(f"  Target:    {target_timing_mean:.3f}s ± {target_timing_std:.3f}s")
        print(f"  Difference: {pred_timing_mean - target_timing_mean:+.3f}s")
        
        # Actions per Gamestate
        print(f"\n🎯 Actions per Gamestate:")
        print(f"  Predicted: {pred_actions_mean:.1f}")
        print(f"  Target:    {target_actions_mean:.1f} ± {target_actions_std:.1f}")
        print(f"  Difference: {pred_actions_mean - target_actions_mean:+.1f}")
        
        # Event Distribution
        event_names = ["MOVE", "KEY", "SCROLL", "CLICK"]
        print(f"\n🎮 Event Distribution:")
        print(f"  {'Event':<8} {'Predicted':<10} {'Target':<10} {'Difference':<10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for i, name in enumerate(event_names):
            pred_pct = event_percentages[i]
            target_pct = target_event_percentages[i]
            diff = pred_pct - target_pct
            print(f"  {name:<8} {pred_pct:>8.1f}% {target_pct:>8.1f}% {diff:>+8.1f}%")
    
    def _save_visualizations(self, epoch: int, target_timing: torch.Tensor, pred_timing: torch.Tensor = None,
                           target_actions: torch.Tensor = None, pred_actions: torch.Tensor = None,
                           save_timing_graphs: bool = True):
        """Save clean visualizations"""
        
        # Timing distribution graph (only if requested)
        if save_timing_graphs:
            self._create_timing_graph(epoch, target_timing, pred_timing)
        
        # Actions per gamestate graph (always save when target_actions provided)
        if target_actions is not None:
            self._create_actions_graph(epoch, target_actions, pred_actions)
    
    def _create_timing_graph(self, epoch: int, target_timing: torch.Tensor, pred_timing: torch.Tensor = None):
        """Create timing distribution dot graphs with y-axis break - combined, target only, and predicted only"""
        
        # Get unique values and counts
        target_unique, target_counts = np.unique(target_timing.cpu().numpy(), return_counts=True)
        
        if pred_timing is not None:
            pred_unique, pred_counts = np.unique(pred_timing.cpu().numpy(), return_counts=True)
        else:
            pred_unique, pred_counts = np.array([]), np.array([])
        
        # Find the break point and max values
        all_counts = np.concatenate([target_counts, pred_counts]) if pred_timing is not None else target_counts
        y_max = np.max(all_counts)
        break_point = 20
        
        # Create the three graphs with y-axis break
        self._create_timing_graph_with_break(epoch, target_unique, target_counts, pred_unique, pred_counts, 
                                           y_max, break_point, "Combined", "timing_analysis")
        self._create_timing_graph_with_break(epoch, target_unique, target_counts, None, None, 
                                           y_max, break_point, "Target", "timing_analysis_targets")
        if pred_timing is not None:
            self._create_timing_graph_with_break(epoch, None, None, pred_unique, pred_counts, 
                                               y_max, break_point, "Predicted", "timing_analysis_predictions")
    
    def _create_timing_graph_with_break(self, epoch: int, target_unique, target_counts, pred_unique, pred_counts, 
                                      y_max, break_point, title_suffix, filename_suffix):
        """Create a single timing graph with y-axis break"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Transform y-coordinates for the break
        def transform_y(y_vals):
            """Transform y values to account for the break"""
            transformed = np.zeros_like(y_vals, dtype=float)
            # Bottom half: 0-20 maps to 0-0.5
            mask_low = y_vals <= break_point
            transformed[mask_low] = y_vals[mask_low] / (2 * break_point)
            # Top half: 20-max maps to 0.5-1.0
            mask_high = y_vals > break_point
            if y_max > break_point:
                transformed[mask_high] = 0.5 + 0.5 * (y_vals[mask_high] - break_point) / (y_max - break_point)
            return transformed
        
        # Plot target data
        if target_unique is not None and len(target_unique) > 0:
            target_y_transformed = transform_y(target_counts)
            ax.scatter(target_unique, target_y_transformed, alpha=0.7, color='blue', 
                      label='Target Timing', s=15)
        
        # Plot predicted data
        if pred_unique is not None and len(pred_unique) > 0:
            pred_y_transformed = transform_y(pred_counts)
            ax.scatter(pred_unique, pred_y_transformed, alpha=0.7, color='red', 
                      label='Predicted Timing', s=15)
        
        # Set up the broken y-axis
        ax.set_ylim(0, 1)
        
        # Create custom tick labels
        # Bottom half ticks (0-20)
        bottom_ticks = np.linspace(0, break_point, 6)  # 0, 4, 8, 12, 16, 20
        bottom_tick_positions = bottom_ticks / (2 * break_point)
        bottom_labels = [f'{int(tick)}' for tick in bottom_ticks]
        
        # Top half ticks (20-max)
        if y_max > break_point:
            # Create evenly spaced ticks for top half
            top_ticks = np.linspace(break_point, y_max, 5)  # 20, 40, 60, 80, max
            top_tick_positions = 0.5 + 0.5 * (top_ticks - break_point) / (y_max - break_point)
            top_labels = [f'{int(tick)}' for tick in top_ticks]
        else:
            top_tick_positions = []
            top_labels = []
        
        # Combine all ticks and labels
        all_ticks = np.concatenate([bottom_tick_positions, top_tick_positions])
        all_labels = bottom_labels + top_labels
        
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(all_labels)
        
        # Add break indicator
        ax.axhline(y=0.5, color='black', linestyle='-', linewidth=1)
        ax.text(0.02, 0.48, f'0-{break_point}', transform=ax.transAxes, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.text(0.02, 0.52, f'{break_point}-{int(y_max)}', transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Time Delta (seconds)')
        ax.set_ylabel('Count')
        ax.set_title(f'Epoch {epoch}: {title_suffix} Timing Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        filename = os.path.join(self.save_dir, f'epoch_{epoch:03d}_{filename_suffix}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 {title_suffix} timing graph saved: {filename}")
    
    def _create_actions_graph(self, epoch: int, target_actions: torch.Tensor, pred_actions: torch.Tensor = None):
        """Create actions per gamestate dot graph"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # X-axis: gamestate index
        gamestate_indices = np.arange(len(target_actions))
        
        # Plot target
        ax.scatter(gamestate_indices, target_actions.cpu().numpy(), alpha=0.7, color='blue', 
                  label='Target Actions', s=15)
        
        # Plot predicted if available
        if pred_actions is not None:
            ax.scatter(gamestate_indices, pred_actions.cpu().numpy(), alpha=0.7, color='red', 
                      label='Predicted Actions', s=15)
        
        ax.set_xlabel('Gamestate Index')
        ax.set_ylabel('Number of Actions')
        ax.set_title(f'Epoch {epoch}: Actions per Gamestate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        filename = os.path.join(self.save_dir, f'epoch_{epoch:03d}_actions_per_gamestate.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Actions graph saved: {filename}")
    
    def generate_training_summary(self, epoch: int = 0, train_loss: float = 0.0, val_loss: float = 0.0, 
                                best_val_loss: float = 0.0, is_best: bool = False):
        """Generate simple training summary"""
        
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"  🎯 Training Loss:   {train_loss:.3f}")
        print(f"  🔍 Validation Loss: {val_loss:.3f}")
        print(f"  🏆 Best Val Loss:   {best_val_loss:.3f}")
        if is_best:
            print(f"  ✨ New Best! Model saved")
        print()
    
    def _save_predictions(self, model_outputs: Dict, action_targets: torch.Tensor, 
                         valid_mask: torch.Tensor, epoch: int):
        """Save model predictions in the same format as action_targets.npy"""
        
        batch_size = action_targets.shape[0]
        max_actions = action_targets.shape[1]
        action_features = action_targets.shape[2]
        
        # Create prediction tensor with same shape as action_targets
        predictions = torch.zeros_like(action_targets)
        
        # Fill in predictions where valid
        if 'time_deltas' in model_outputs:
            predictions[:, :, 0] = model_outputs['time_deltas'].squeeze(-1)
        
        if 'x_mu' in model_outputs:
            predictions[:, :, 1] = model_outputs['x_mu'].squeeze(-1)
        
        if 'y_mu' in model_outputs:
            predictions[:, :, 2] = model_outputs['y_mu'].squeeze(-1)
        
        # For button, key_action, key_id, scroll_y - use argmax of logits
        if 'button_logits' in model_outputs:
            button_preds = torch.argmax(model_outputs['button_logits'], dim=-1)
            predictions[:, :, 3] = button_preds.float()
        
        if 'key_action_logits' in model_outputs:
            key_action_preds = torch.argmax(model_outputs['key_action_logits'], dim=-1)
            predictions[:, :, 4] = key_action_preds.float()
        
        if 'key_id_logits' in model_outputs:
            key_id_preds = torch.argmax(model_outputs['key_id_logits'], dim=-1)
            predictions[:, :, 5] = key_id_preds.float()
        
        if 'scroll_y_logits' in model_outputs:
            scroll_preds = torch.argmax(model_outputs['scroll_y_logits'], dim=-1)
            predictions[:, :, 6] = scroll_preds.float()
        
        # Zero out invalid predictions (keep only valid actions)
        predictions[~valid_mask] = 0.0
        
        # Save as numpy array
        predictions_np = predictions.cpu().numpy()
        filename = os.path.join(self.save_dir, f'epoch_{epoch:03d}_predictions.npy')
        np.save(filename, predictions_np)
        print(f"📊 Predictions saved: {filename} (shape: {predictions_np.shape})")
    
    def save_full_validation_predictions(self, all_predictions: torch.Tensor, epoch: int):
        """Save predictions from full validation set"""
        predictions_np = all_predictions.cpu().numpy()
        filename = os.path.join(self.save_dir, f'epoch_{epoch:03d}_full_validation_predictions.npy')
        np.save(filename, predictions_np)
        print(f"📊 Full validation predictions saved: {filename} (shape: {predictions_np.shape})")
    
    def create_full_validation_graphs(self, all_val_predictions, all_val_targets, all_val_masks, epoch: int):
        """Create timing and actions per gamestate graphs using the full validation set"""
        
        # Combine all timing data from full validation set
        all_target_timing = []
        all_pred_timing = []
        
        # Combine all actions per gamestate data from full validation set
        all_target_actions = []
        all_pred_actions = []
        
        for batch_idx, (pred_dict, targets, mask) in enumerate(zip(all_val_predictions, all_val_targets, all_val_masks)):
            # Get valid actions for this batch
            valid_mask = mask.bool()
            valid_targets = targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Target timing (column 0)
                target_timing = valid_targets[:, 0]
                all_target_timing.append(target_timing)
                
                # Predicted timing
                if pred_dict['time_deltas'] is not None:
                    pred_timing = pred_dict['time_deltas'][valid_mask].squeeze(-1)
                    all_pred_timing.append(pred_timing)
            
            # Actions per gamestate data
            # Target: count of valid actions per gamestate
            target_actions_per_gamestate = valid_mask.sum(dim=1).float()
            all_target_actions.append(target_actions_per_gamestate)
            
            # Predicted: sequence length predictions
            if pred_dict.get('sequence_length') is not None:
                pred_actions_per_gamestate = pred_dict['sequence_length'].squeeze()
                all_pred_actions.append(pred_actions_per_gamestate)
        
        # Combine all batches for timing
        if all_target_timing:
            combined_target_timing = torch.cat(all_target_timing, dim=0)
        else:
            combined_target_timing = torch.tensor([])
            
        if all_pred_timing:
            combined_pred_timing = torch.cat(all_pred_timing, dim=0)
        else:
            combined_pred_timing = None
        
        # Combine all batches for actions per gamestate
        if all_target_actions:
            combined_target_actions = torch.cat(all_target_actions, dim=0)
        else:
            combined_target_actions = torch.tensor([])
            
        if all_pred_actions:
            combined_pred_actions = torch.cat(all_pred_actions, dim=0)
        else:
            combined_pred_actions = None
        
        # Create comprehensive timing graphs with full validation data
        if len(combined_target_timing) > 0:
            self._create_timing_graph(epoch, combined_target_timing, combined_pred_timing)
            print(f"📊 Full validation timing graphs created using {len(combined_target_timing)} timing samples")
        else:
            print(f"⚠️  No valid timing data found in validation set")
        
        # Create comprehensive actions per gamestate graph with full validation data
        if len(combined_target_actions) > 0:
            self._create_actions_graph(epoch, combined_target_actions, combined_pred_actions)
            print(f"📊 Full validation actions per gamestate graph created using {len(combined_target_actions)} gamestates")
        else:
            print(f"⚠️  No valid actions per gamestate data found in validation set")
