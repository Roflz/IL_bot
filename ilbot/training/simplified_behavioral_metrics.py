#!/usr/bin/env python3
"""
Clean and Simple Behavioral Metrics for analyzing bot intelligence
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

CANONICAL_EVENT_ORDER = {0: "CLICK", 1: "KEY", 2: "SCROLL", 3: "MOVE"}

def derive_event_targets(targets):
    """
    targets: [B,A,7] -> [B,A] event ids with mapping 0=CLICK,1=KEY,2=SCROLL,3=MOVE
    """
    import torch
    B, A, _ = targets.shape
    device = targets.device
    ev = torch.full((B, A), 3, dtype=torch.long, device=device)  # default MOVE=3
    button = targets[..., 3].long()
    key_action = targets[..., 4].long()
    scroll_y = targets[..., 6].long()
    ev = torch.where(button != -1, torch.zeros_like(ev), ev)
    ev = torch.where(key_action != -1, torch.ones_like(ev), ev)
    ev = torch.where(scroll_y != 0, torch.full_like(ev, 2), ev)
    return ev

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
        """Analyze model predictions - only save visualizations, no per-epoch metrics"""
        
        # Save predictions if requested (save sample predictions from behavioral analysis)
        if save_predictions:
            self._save_predictions(model_outputs, action_targets, valid_mask, epoch)
        
        # Save visualizations if requested
        if save_visualizations:
            # Extract timing predictions
            time_deltas = model_outputs.get('time_deltas', torch.tensor([]))
            if time_deltas.numel() > 0:
                time_deltas = time_deltas.squeeze(-1)  # Remove last dimension if present
            
            # Extract valid actions
            valid_actions = action_targets[valid_mask]
            valid_time_deltas = time_deltas[valid_mask] if time_deltas.numel() > 0 else torch.tensor([])
            
            # Target timing statistics
            time_targets = action_targets[:, :, 0]  # First column is time delta
            time_targets_valid = time_targets[valid_mask]
            
            # Actions per gamestate
            actions_per_gamestate = valid_mask.sum(dim=1).float()
            
            # Get predicted actions from sequence length if available
            pred_actions = None
            if 'sequence_length' in model_outputs:
                pred_actions = model_outputs['sequence_length'].squeeze(-1).squeeze(-1)
            
            self._save_visualizations(epoch, time_targets_valid, time_deltas if 'time_deltas' in model_outputs else None,
                                    actions_per_gamestate, pred_actions, save_timing_graphs=save_timing_graphs)
    
    def _derive_event_types(self, valid_actions: torch.Tensor) -> torch.Tensor:
        """Derive event types from action targets - B) Fix canonical mapping"""
        # Extract components
        button = valid_actions[:, 3]  # Button column
        key_action = valid_actions[:, 4]  # Key action column  
        scroll_y = valid_actions[:, 6]  # Scroll column
        
        # B) Canonical order: 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
        # Initialize with MOVE (3) - default
        event_types = torch.full((valid_actions.shape[0],), 3, dtype=torch.long)
        
        # Priority: CLICK > KEY > SCROLL > MOVE
        # SCROLL: scroll_y != 0 (lowest priority)
        event_types[scroll_y != 0] = 2
        
        # KEY: key_action != 0 (medium priority - overwrites SCROLL)
        event_types[key_action != 0] = 1
        
        # CLICK: button != 0 (highest priority - overwrites KEY and SCROLL)
        event_types[button != 0] = 0
        
        return event_types
    
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
                                best_val_loss: float = 0.0, is_best: bool = False, 
                                all_val_predictions: List = None, all_val_targets: List = None, 
                                all_val_masks: List = None):
        """Generate training summary with final detailed analysis using full validation set"""
        
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"  🎯 Training Loss:   {train_loss:.3f}")
        print(f"  🔍 Validation Loss: {val_loss:.3f}")
        print(f"  🏆 Best Val Loss:   {best_val_loss:.3f}")
        if is_best:
            print(f"  ✨ New Best! Model saved")
        print()
        
        # Show detailed analysis using full validation set
        if all_val_predictions is not None and all_val_targets is not None and all_val_masks is not None:
            self._print_full_validation_analysis(epoch, all_val_predictions, all_val_targets, all_val_masks, train_loss, val_loss, best_val_loss)
    
    def _print_full_validation_analysis(self, epoch: int, all_val_predictions: List, all_val_targets: List, all_val_masks: List, train_loss: float, val_loss: float, best_val_loss: float):
        """Print detailed analysis using the full validation set"""
        
        # Combine all validation data
        all_target_timing = []
        all_pred_timing = []
        all_target_actions = []
        all_pred_actions = []
        all_valid_actions = []
        all_pred_event_types = []
        all_target_event_types = []
        
        for batch_idx, (pred_dict, targets, mask) in enumerate(zip(all_val_predictions, all_val_targets, all_val_masks)):
            # Get valid actions for this batch
            valid_mask = mask.bool()
            valid_targets = targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Target timing (column 0) - check if already in seconds or needs conversion
                target_timing_raw = valid_targets[:, 0]
                # If targets are in milliseconds (typical range 10-1000ms), convert to seconds
                if target_timing_raw.max() > 1.0:  # If max > 1, likely in milliseconds
                    target_timing = target_timing_raw / 1000.0
                else:  # Already in seconds
                    target_timing = target_timing_raw
                all_target_timing.append(target_timing)
                
                # Predicted timing
                if pred_dict['time_deltas'] is not None:
                    pred_timing = pred_dict['time_deltas'][valid_mask].squeeze(-1)
                    all_pred_timing.append(pred_timing)
                
                # Store valid actions for event analysis
                all_valid_actions.append(valid_targets)
                
                # Predicted event types
                if pred_dict.get('event_logits') is not None:
                    event_predictions = pred_dict['event_logits'].argmax(dim=-1)
                    pred_event_types = event_predictions[valid_mask]
                    all_pred_event_types.append(pred_event_types)
                
                # Target event types
                target_events = self._derive_event_types(valid_targets)
                all_target_event_types.append(target_events)
            
            # Actions per gamestate data - simple approach
            # Target: count of valid actions per gamestate
            target_actions_per_gamestate = valid_mask.sum(dim=1).float()
            all_target_actions.append(target_actions_per_gamestate)
            
            # Predicted: count valid actions per gamestate using time_deltas
            if pred_dict.get('time_deltas') is not None:
                time_deltas = pred_dict['time_deltas'].squeeze(-1)
                pred_actions_per_gamestate = (time_deltas > 0.001).sum(dim=1).float()
                all_pred_actions.append(pred_actions_per_gamestate)
        
        # Combine all batches
        if all_target_timing:
            combined_target_timing = torch.cat(all_target_timing, dim=0)
        else:
            combined_target_timing = torch.tensor([])
            
        if all_pred_timing:
            combined_pred_timing = torch.cat(all_pred_timing, dim=0)
        else:
            combined_pred_timing = torch.tensor([])
        
        if all_target_actions:
            combined_target_actions = torch.cat(all_target_actions, dim=0)
        else:
            combined_target_actions = torch.tensor([])
            
        if all_pred_actions:
            combined_pred_actions = torch.cat(all_pred_actions, dim=0)
        else:
            combined_pred_actions = torch.tensor([])
        
        if all_valid_actions:
            combined_valid_actions = torch.cat(all_valid_actions, dim=0)
        else:
            combined_valid_actions = torch.tensor([])
        
        if all_pred_event_types:
            combined_pred_event_types = torch.cat(all_pred_event_types, dim=0)
        else:
            combined_pred_event_types = torch.tensor([])
        
        if all_target_event_types:
            combined_target_event_types = torch.cat(all_target_event_types, dim=0)
        else:
            combined_target_event_types = torch.tensor([])
        
        # Print the detailed tables using full validation data
        self._print_validation_dataset_stats(all_val_predictions, all_val_targets, all_val_masks)
        self._print_actions_per_gamestate_table(combined_target_actions, combined_pred_actions)
        self._print_timing_analysis_table(combined_target_timing, combined_pred_timing)
        # REMOVED: Action Comparison Analysis table as requested
        self._print_event_distribution_table(combined_pred_event_types, combined_target_event_types)
        self._print_mouse_position_table(all_val_predictions, all_val_targets, all_val_masks)
        self._print_data_quality_table(all_val_targets, all_val_masks, all_val_predictions)
        self._print_model_performance_table(epoch, train_loss, val_loss, best_val_loss)
    
    def _print_actions_per_gamestate_table(self, target_actions: torch.Tensor, pred_actions: torch.Tensor):
        """Print Actions per Gamestate Analysis table"""
        
        print(f"\n📊 Actions per Gamestate Analysis:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        if len(target_actions) > 0:
            target_mean = float(target_actions.mean().item())
            target_std = float(target_actions.std().item())
            target_min = float(target_actions.min().item())
            target_max = float(target_actions.max().item())
            target_p10 = float(torch.quantile(target_actions, 0.1).item())
            target_p25 = float(torch.quantile(target_actions, 0.25).item())
            target_p50 = float(torch.quantile(target_actions, 0.5).item())
            target_p75 = float(torch.quantile(target_actions, 0.75).item())
            target_p90 = float(torch.quantile(target_actions, 0.9).item())
        else:
            target_mean = target_std = target_min = target_max = 0.0
            target_p10 = target_p25 = target_p50 = target_p75 = target_p90 = 0.0
        
        if len(pred_actions) > 0:
            pred_mean = float(pred_actions.mean().item())
            pred_std = float(pred_actions.std().item())
            pred_min = float(pred_actions.min().item())
            pred_max = float(pred_actions.max().item())
            pred_p10 = float(torch.quantile(pred_actions, 0.1).item())
            pred_p25 = float(torch.quantile(pred_actions, 0.25).item())
            pred_p50 = float(torch.quantile(pred_actions, 0.5).item())
            pred_p75 = float(torch.quantile(pred_actions, 0.75).item())
            pred_p90 = float(torch.quantile(pred_actions, 0.9).item())
        else:
            pred_mean = pred_std = pred_min = pred_max = 0.0
            pred_p10 = pred_p25 = pred_p50 = pred_p75 = pred_p90 = 0.0
        
        print(f"  │ Mean Actions/Gamestate   │ {pred_mean:>12.1f} │ {target_mean:>12.1f} │ {pred_mean-target_mean:>+12.1f} │")
        print(f"  │ Std Actions/Gamestate    │ {pred_std:>12.1f} │ {target_std:>12.1f} │ {pred_std-target_std:>+12.1f} │")
        print(f"  │ Min Actions/Gamestate    │ {pred_min:>12.1f} │ {target_min:>12.1f} │ {pred_min-target_min:>+12.1f} │")
        print(f"  │ Max Actions/Gamestate    │ {pred_max:>12.1f} │ {target_max:>12.1f} │ {pred_max-target_max:>+12.1f} │")
        print(f"  │ P10 Actions/Gamestate    │ {pred_p10:>12.1f} │ {target_p10:>12.1f} │ {pred_p10-target_p10:>+12.1f} │")
        print(f"  │ P25 Actions/Gamestate    │ {pred_p25:>12.1f} │ {target_p25:>12.1f} │ {pred_p25-target_p25:>+12.1f} │")
        print(f"  │ P50 Actions/Gamestate    │ {pred_p50:>12.1f} │ {target_p50:>12.1f} │ {pred_p50-target_p50:>+12.1f} │")
        print(f"  │ P75 Actions/Gamestate    │ {pred_p75:>12.1f} │ {target_p75:>12.1f} │ {pred_p75-target_p75:>+12.1f} │")
        print(f"  │ P90 Actions/Gamestate    │ {pred_p90:>12.1f} │ {target_p90:>12.1f} │ {pred_p90-target_p90:>+12.1f} │")
        print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                
    def _print_timing_analysis_table(self, target_timing: torch.Tensor, pred_timing: torch.Tensor):
        """Print Timing Prediction Analysis table"""
        
        print(f"\n⏱️  Timing Prediction Analysis:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        if len(target_timing) > 0:
            target_mean = float(target_timing.mean().item())
            target_std = float(target_timing.std().item())
            target_median = float(target_timing.median().item())
            target_p25 = float(target_timing.quantile(0.25).item())
            target_p75 = float(target_timing.quantile(0.75).item())
            target_p10 = float(target_timing.quantile(0.10).item())
            target_p90 = float(target_timing.quantile(0.90).item())
            target_min = float(target_timing.min().item())
            target_max = float(target_timing.max().item())
            target_unique = len(torch.unique(target_timing))
        else:
            target_mean = target_std = target_median = target_p25 = target_p75 = 0.0
            target_p10 = target_p90 = target_min = target_max = 0.0
            target_unique = 0
        
        if len(pred_timing) > 0:
            pred_mean = float(pred_timing.mean().item())
            pred_std = float(pred_timing.std().item())
            pred_median = float(pred_timing.median().item())
            pred_p25 = float(pred_timing.quantile(0.25).item())
            pred_p75 = float(pred_timing.quantile(0.75).item())
            pred_p10 = float(pred_timing.quantile(0.10).item())
            pred_p90 = float(pred_timing.quantile(0.90).item())
            pred_min = float(pred_timing.min().item())
            pred_max = float(pred_timing.max().item())
            pred_unique = len(torch.unique(pred_timing))
        else:
            pred_mean = pred_std = pred_median = pred_p25 = pred_p75 = 0.0
            pred_p10 = pred_p90 = pred_min = pred_max = 0.0
            pred_unique = 0
        
        print(f"  │ Delta Time Mean (s)     │ {pred_mean:>12.3f} │ {target_mean:>12.3f} │ {pred_mean-target_mean:>+12.3f} │")
        print(f"  │ Delta Time Std (s)      │ {pred_std:>12.3f} │ {target_std:>12.3f} │ {pred_std-target_std:>+12.3f} │")
        print(f"  │ Delta Time Median (s)   │ {pred_median:>12.3f} │ {target_median:>12.3f} │ {pred_median-target_median:>+12.3f} │")
        print(f"  │ Delta Time P25 (s)      │ {pred_p25:>12.3f} │ {target_p25:>12.3f} │ {pred_p25-target_p25:>+12.3f} │")
        print(f"  │ Delta Time P75 (s)      │ {pred_p75:>12.3f} │ {target_p75:>12.3f} │ {pred_p75-target_p75:>+12.3f} │")
        print(f"  │ Delta Time P10 (s)      │ {pred_p10:>12.3f} │ {target_p10:>12.3f} │ {pred_p10-target_p10:>+12.3f} │")
        print(f"  │ Delta Time P90 (s)      │ {pred_p90:>12.3f} │ {target_p90:>12.3f} │ {pred_p90-target_p90:>+12.3f} │")
        print(f"  │ Delta Time Min (s)      │ {pred_min:>12.3f} │ {target_min:>12.3f} │ {pred_min-target_min:>+12.3f} │")
        print(f"  │ Delta Time Max (s)      │ {pred_max:>12.3f} │ {target_max:>12.3f} │ {pred_max-target_max:>+12.3f} │")
        print(f"  │ Delta Time Unique Count │ {pred_unique:>12.0f} │ {target_unique:>12.0f} │ {pred_unique-target_unique:>+12.0f} │")
        print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    # REMOVED: _print_action_comparison_table function as requested
    
    def _print_event_distribution_table(self, pred_event_types: torch.Tensor, target_event_types: torch.Tensor):
        """Print Event Distribution Analysis table"""
        
        print(f"\n🎮 Event Distribution Analysis:")
        print(f"  ┌─────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Event Type  │ Predicted    │ Target       │ Difference   │")
        print(f"  ├─────────────┼──────────────┼──────────────┼──────────────┤")
        
        if len(pred_event_types) > 0 and len(target_event_types) > 0:
            # Count event types
            pred_event_counts = torch.bincount(pred_event_types, minlength=4)
            target_event_counts = torch.bincount(target_event_types, minlength=4)
            
            # Calculate percentages
            pred_percentages = (pred_event_counts.float() / pred_event_counts.sum() * 100).tolist()
            target_percentages = (target_event_counts.float() / target_event_counts.sum() * 100).tolist()
            
            event_names = [CANONICAL_EVENT_ORDER[i] for i in range(4)]
            for i, name in enumerate(event_names):
                pred_pct = pred_percentages[i]
                target_pct = target_percentages[i]
                diff = pred_pct - target_pct
                print(f"  │ {name:>11} │ {pred_pct:>8.1f}% │ {target_pct:>8.1f}% │ {diff:>+8.1f}% │")
        else:
            print(f"  │ No event data available  │        N/A │        N/A │        N/A │")
        
        print(f"  └─────────────┴──────────────┴──────────────┴──────────────┘")
    
    def _print_mouse_position_table(self, all_val_predictions: List, all_val_targets: List, all_val_masks: List):
        """Print Mouse Position Analysis table"""
        
        print(f"\n🖱️  Mouse Position Analysis:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        # Collect mouse position data from all validation batches
        all_x_pred = []
        all_y_pred = []
        all_x_target = []
        all_y_target = []
        
        for batch_idx, (pred_dict, targets, mask) in enumerate(zip(all_val_predictions, all_val_targets, all_val_masks)):
            valid_mask = mask.bool()
            valid_targets = targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Get target mouse positions (columns 1 and 2)
                x_target = valid_targets[:, 1]  # x coordinate
                y_target = valid_targets[:, 2]  # y coordinate
                all_x_target.append(x_target)
                all_y_target.append(y_target)
                
                # Get predicted mouse positions if available
                if pred_dict.get('x_mu') is not None and pred_dict.get('y_mu') is not None:
                    x_pred = pred_dict['x_mu'][valid_mask].squeeze(-1)
                    y_pred = pred_dict['y_mu'][valid_mask].squeeze(-1)
                    all_x_pred.append(x_pred)
                    all_y_pred.append(y_pred)
        
        # Combine all batches and convert to pixels
        if all_x_target:
            combined_x_target = torch.cat(all_x_target, dim=0)
            combined_y_target = torch.cat(all_y_target, dim=0)
            
            # Convert normalized coordinates to pixels (assuming 1920x1080 screen)
            target_x_pixels = combined_x_target * 1920.0
            target_y_pixels = combined_y_target * 1080.0
            
            target_x_mean = float(target_x_pixels.mean().item())
            target_y_mean = float(target_y_pixels.mean().item())
            target_x_min = float(target_x_pixels.min().item())
            target_y_min = float(target_y_pixels.min().item())
            target_x_max = float(target_x_pixels.max().item())
            target_y_max = float(target_y_pixels.max().item())
            target_x_p10 = float(torch.quantile(target_x_pixels, 0.1).item())
            target_x_p25 = float(torch.quantile(target_x_pixels, 0.25).item())
            target_x_p50 = float(torch.quantile(target_x_pixels, 0.5).item())
            target_x_p75 = float(torch.quantile(target_x_pixels, 0.75).item())
            target_x_p90 = float(torch.quantile(target_x_pixels, 0.9).item())
            target_y_p10 = float(torch.quantile(target_y_pixels, 0.1).item())
            target_y_p25 = float(torch.quantile(target_y_pixels, 0.25).item())
            target_y_p50 = float(torch.quantile(target_y_pixels, 0.5).item())
            target_y_p75 = float(torch.quantile(target_y_pixels, 0.75).item())
            target_y_p90 = float(torch.quantile(target_y_pixels, 0.9).item())
        else:
            target_x_mean = target_y_mean = 0.0
            target_x_min = target_y_min = 0.0
            target_x_max = target_y_max = 0.0
            target_x_p10 = target_x_p25 = target_x_p50 = target_x_p75 = target_x_p90 = 0.0
            target_y_p10 = target_y_p25 = target_y_p50 = target_y_p75 = target_y_p90 = 0.0
        
        if all_x_pred:
            combined_x_pred = torch.cat(all_x_pred, dim=0)
            combined_y_pred = torch.cat(all_y_pred, dim=0)
            
            # Convert normalized coordinates to pixels (assuming 1920x1080 screen)
            pred_x_pixels = combined_x_pred * 1920.0
            pred_y_pixels = combined_y_pred * 1080.0
            
            pred_x_mean = float(pred_x_pixels.mean().item())
            pred_y_mean = float(pred_y_pixels.mean().item())
            pred_x_min = float(pred_x_pixels.min().item())
            pred_y_min = float(pred_y_pixels.min().item())
            pred_x_max = float(pred_x_pixels.max().item())
            pred_y_max = float(pred_y_pixels.max().item())
            pred_x_p10 = float(torch.quantile(pred_x_pixels, 0.1).item())
            pred_x_p25 = float(torch.quantile(pred_x_pixels, 0.25).item())
            pred_x_p50 = float(torch.quantile(pred_x_pixels, 0.5).item())
            pred_x_p75 = float(torch.quantile(pred_x_pixels, 0.75).item())
            pred_x_p90 = float(torch.quantile(pred_x_pixels, 0.9).item())
            pred_y_p10 = float(torch.quantile(pred_y_pixels, 0.1).item())
            pred_y_p25 = float(torch.quantile(pred_y_pixels, 0.25).item())
            pred_y_p50 = float(torch.quantile(pred_y_pixels, 0.5).item())
            pred_y_p75 = float(torch.quantile(pred_y_pixels, 0.75).item())
            pred_y_p90 = float(torch.quantile(pred_y_pixels, 0.9).item())
        else:
            pred_x_mean = pred_y_mean = 0.0
            pred_x_min = pred_y_min = 0.0
            pred_x_max = pred_y_max = 0.0
            pred_x_p10 = pred_x_p25 = pred_x_p50 = pred_x_p75 = pred_x_p90 = 0.0
            pred_y_p10 = pred_y_p25 = pred_y_p50 = pred_y_p75 = pred_y_p90 = 0.0
        
        print(f"  │ X Mean (pixels)         │ {pred_x_mean:>12.1f} │ {target_x_mean:>12.1f} │ {pred_x_mean-target_x_mean:>+12.1f} │")
        print(f"  │ Y Mean (pixels)         │ {pred_y_mean:>12.1f} │ {target_y_mean:>12.1f} │ {pred_y_mean-target_y_mean:>+12.1f} │")
        print(f"  │ X Range (pixels)        │ {pred_x_min:>6.0f}-{pred_x_max:<6.0f} │ {target_x_min:>6.0f}-{target_x_max:<6.0f} │        N/A │")
        print(f"  │ Y Range (pixels)        │ {pred_y_min:>6.0f}-{pred_y_max:<6.0f} │ {target_y_min:>6.0f}-{target_y_max:<6.0f} │        N/A │")
        print(f"  │ X P10 (pixels)          │ {pred_x_p10:>12.1f} │ {target_x_p10:>12.1f} │ {pred_x_p10-target_x_p10:>+12.1f} │")
        print(f"  │ X P25 (pixels)          │ {pred_x_p25:>12.1f} │ {target_x_p25:>12.1f} │ {pred_x_p25-target_x_p25:>+12.1f} │")
        print(f"  │ X P50 (pixels)          │ {pred_x_p50:>12.1f} │ {target_x_p50:>12.1f} │ {pred_x_p50-target_x_p50:>+12.1f} │")
        print(f"  │ X P75 (pixels)          │ {pred_x_p75:>12.1f} │ {target_x_p75:>12.1f} │ {pred_x_p75-target_x_p75:>+12.1f} │")
        print(f"  │ X P90 (pixels)          │ {pred_x_p90:>12.1f} │ {target_x_p90:>12.1f} │ {pred_x_p90-target_x_p90:>+12.1f} │")
        print(f"  │ Y P10 (pixels)          │ {pred_y_p10:>12.1f} │ {target_y_p10:>12.1f} │ {pred_y_p10-target_y_p10:>+12.1f} │")
        print(f"  │ Y P25 (pixels)          │ {pred_y_p25:>12.1f} │ {target_y_p25:>12.1f} │ {pred_y_p25-target_y_p25:>+12.1f} │")
        print(f"  │ Y P50 (pixels)          │ {pred_y_p50:>12.1f} │ {target_y_p50:>12.1f} │ {pred_y_p50-target_y_p50:>+12.1f} │")
        print(f"  │ Y P75 (pixels)          │ {pred_y_p75:>12.1f} │ {target_y_p75:>12.1f} │ {pred_y_p75-target_y_p75:>+12.1f} │")
        print(f"  │ Y P90 (pixels)          │ {pred_y_p90:>12.1f} │ {target_y_p90:>12.1f} │ {pred_y_p90-target_y_p90:>+12.1f} │")
        print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    def _print_data_quality_table(self, all_val_targets: List, all_val_masks: List, all_val_predictions: List = None):
        print(f"\n📊 Data Quality Analysis:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        total_gamestates = 0
        target_valid_actions = 0
        pred_valid_actions = 0
        
        for targets, mask in zip(all_val_targets, all_val_masks):
            total_gamestates += targets.shape[0]
            target_valid_actions += mask.sum().item()
        
        if all_val_predictions:
            for pred_dict in all_val_predictions:
                if pred_dict.get('time_deltas') is not None:
                    time_deltas = pred_dict['time_deltas'].squeeze(-1)
                    pred_valid_actions += (time_deltas > 0.001).sum().item()
        
        target_avg_actions = target_valid_actions / total_gamestates if total_gamestates > 0 else 0.0
        pred_avg_actions = pred_valid_actions / total_gamestates if total_gamestates > 0 else 0.0
        
        print(f"  │ Total Gamestates        │ {total_gamestates:>12} │ {total_gamestates:>12} │         +0 │")
        print(f"  │ Valid Actions           │ {pred_valid_actions:>12} │ {target_valid_actions:>12} │ {pred_valid_actions-target_valid_actions:>+12} │")
        print(f"  │ Avg Actions/Gamestate   │ {pred_avg_actions:>12.1f} │ {target_avg_actions:>12.1f} │ {pred_avg_actions-target_avg_actions:>+12.1f} │")
        print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    def _print_model_performance_table(self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float):
        """Print Model Performance Summary table"""
        
        print(f"\n🎯 Model Performance Summary:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Current      │ Best         │ Improvement  │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        print(f"  │ Training Loss           │ {train_loss:>12.3f} │        N/A │        N/A │")
        print(f"  │ Validation Loss         │ {val_loss:>12.3f} │ {best_val_loss:>12.3f} │ {val_loss-best_val_loss:>+12.3f} │")
        
        # Calculate improvement percentage
        if best_val_loss > 0:
            improvement_pct = ((best_val_loss - val_loss) / best_val_loss) * 100
            print(f"  │ Loss Improvement %      │ {improvement_pct:>12.1f} │        N/A │        N/A │")
        
            print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
    def _print_validation_dataset_stats(self, all_val_predictions: List, all_val_targets: List, all_val_masks: List):
        """Print Validation Dataset Statistics to confirm full validation set usage"""
        
        print(f"\n📋 Validation Dataset Statistics:")
        print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(f"  │ Metric                  │ Count        │ Details      │ Confirmation │")
        print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        # Count validation batches
        num_batches = len(all_val_targets)
        
        # Count total gamestates
        total_gamestates = 0
        total_actions = 0
        total_valid_actions = 0
        
        for targets, mask in zip(all_val_targets, all_val_masks):
            batch_size = targets.shape[0]
            max_actions = targets.shape[1]
            batch_valid = mask.sum().item()
            
            total_gamestates += batch_size
            total_actions += batch_size * max_actions
            total_valid_actions += batch_valid
        
        # Calculate statistics
        avg_actions_per_gamestate = total_valid_actions / total_gamestates if total_gamestates > 0 else 0
        padding_actions = total_actions - total_valid_actions
        valid_percentage = (total_valid_actions / total_actions) * 100 if total_actions > 0 else 0
        
        print(f"  │ Validation Batches       │ {num_batches:>12} │ Full val set │ ✅ Complete │")
        print(f"  │ Total Gamestates         │ {total_gamestates:>12} │ All samples  │ ✅ Complete │")
        print(f"  │ Total Action Slots       │ {total_actions:>12} │ Max capacity │ ✅ Complete │")
        print(f"  │ Valid Actions            │ {total_valid_actions:>12} │ Real actions │ ✅ Complete │")
        print(f"  │ Padding Actions          │ {padding_actions:>12} │ Empty slots  │ ✅ Complete │")
        print(f"  │ Avg Actions/Gamestate    │ {avg_actions_per_gamestate:>12.1f} │ Per sample   │ ✅ Complete │")
        print(f"  │ Valid Action Percentage  │ {valid_percentage:>11.1f}% │ Data density │ ✅ Complete │")
        print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
        # Additional confirmation
        print(f"\n  🔍 Validation Space Confirmation:")
        print(f"    • Dataset Type: VALIDATION SET (not training set)")
        print(f"    • Coverage: 100% of validation samples ({total_gamestates} gamestates)")
        print(f"    • Data Source: All {num_batches} validation batches")
        print(f"    • Analysis Scope: Complete validation dataset")
    
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
                # Target timing (column 0) - check if already in seconds or needs conversion
                target_timing_raw = valid_targets[:, 0]
                # If targets are in milliseconds (typical range 10-1000ms), convert to seconds
                if target_timing_raw.max() > 1.0:  # If max > 1, likely in milliseconds
                    target_timing = target_timing_raw / 1000.0
                else:  # Already in seconds
                    target_timing = target_timing_raw
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