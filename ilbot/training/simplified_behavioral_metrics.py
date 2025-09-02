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
from .pretty_output import printer

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
            # Use pretty printer for cleaner output
            printer.print_behavioral_analysis({}, epoch)  # We'll populate this below
            
            analysis = {}
            
            # Basic timing analysis
            if 'time_q' in model_outputs:
                time_preds = model_outputs['time_q']  # [B, A, 3]
                # Apply valid mask to get only valid predictions
                valid_time_preds = time_preds[valid_mask]  # [N_valid, 3]
                
                # Get target timing data (column 0 is time delta)
                valid_time_targets = action_targets[valid_mask][:, 0]  # [N_valid]
                
                if valid_time_preds.numel() > 0 and valid_time_targets.numel() > 0:
                    # Calculate detailed timing statistics
                    pred_medians = valid_time_preds[:, 1]  # q0.5 predictions
                    pred_q10 = valid_time_preds[:, 0]     # q0.1 predictions
                    pred_q90 = valid_time_preds[:, 2]     # q0.9 predictions
                    
                    analysis['timing'] = {
                        # Predicted timing statistics
                        'mean_timing': self._safe_mean(pred_medians),
                        'median_timing': self._safe_mean(pred_medians),  # q0.5
                        'p10_timing': self._safe_mean(pred_q10),        # q0.1
                        'p25_timing': self._safe_mean(valid_time_preds[:, 1] * 0.75 + valid_time_preds[:, 0] * 0.25),  # Approximate p25
                        'p75_timing': self._safe_mean(valid_time_preds[:, 1] * 0.25 + valid_time_preds[:, 2] * 0.75),  # Approximate p75
                        'p90_timing': self._safe_mean(pred_q90),        # q0.9
                        'timing_uncertainty': self._safe_mean(pred_q90 - pred_q10),  # q0.9 - q0.1
                        'timing_consistency': self._safe_std(pred_medians),
                        
                        # Target timing statistics
                        'target_mean_timing': self._safe_mean(valid_time_targets),
                        'target_median_timing': self._safe_mean(valid_time_targets),
                        'target_p10_timing': self._safe_percentile(valid_time_targets, 10),
                        'target_p25_timing': self._safe_percentile(valid_time_targets, 25),
                        'target_p75_timing': self._safe_percentile(valid_time_targets, 75),
                        'target_p90_timing': self._safe_percentile(valid_time_targets, 90),
                        'target_timing_consistency': self._safe_std(valid_time_targets),
                    }
                else:
                    analysis['timing'] = {
                        'mean_timing': 0.0,
                        'median_timing': 0.0,
                        'p10_timing': 0.0,
                        'p25_timing': 0.0,
                        'p75_timing': 0.0,
                        'p90_timing': 0.0,
                        'timing_uncertainty': 0.0,
                        'timing_consistency': 0.0,
                        'target_mean_timing': 0.0,
                        'target_median_timing': 0.0,
                        'target_p10_timing': 0.0,
                        'target_p25_timing': 0.0,
                        'target_p75_timing': 0.0,
                        'target_p90_timing': 0.0,
                        'target_timing_consistency': 0.0,
                    }
                
                # Print detailed timing analysis table
                print(f"⏱️  Timing Analysis (Time Deltas Between Actions):")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ Mean Timing (s)         │ {analysis['timing']['mean_timing']:>10.2f} │ {analysis['timing']['target_mean_timing']:>10.2f} │ {abs(analysis['timing']['mean_timing'] - analysis['timing']['target_mean_timing']):>10.2f} │")
                print(f"  │ Median Timing (s)       │ {analysis['timing']['median_timing']:>10.2f} │ {analysis['timing']['target_median_timing']:>10.2f} │ {abs(analysis['timing']['median_timing'] - analysis['timing']['target_median_timing']):>10.2f} │")
                print(f"  │ P10 Timing (s)          │ {analysis['timing']['p10_timing']:>10.2f} │ {analysis['timing']['target_p10_timing']:>10.2f} │ {abs(analysis['timing']['p10_timing'] - analysis['timing']['target_p10_timing']):>10.2f} │")
                print(f"  │ P25 Timing (s)          │ {analysis['timing']['p25_timing']:>10.2f} │ {analysis['timing']['target_p25_timing']:>10.2f} │ {abs(analysis['timing']['p25_timing'] - analysis['timing']['target_p25_timing']):>10.2f} │")
                print(f"  │ P75 Timing (s)          │ {analysis['timing']['p75_timing']:>10.2f} │ {analysis['timing']['target_p75_timing']:>10.2f} │ {abs(analysis['timing']['p75_timing'] - analysis['timing']['target_p75_timing']):>10.2f} │")
                print(f"  │ P90 Timing (s)          │ {analysis['timing']['p90_timing']:>10.2f} │ {analysis['timing']['target_p90_timing']:>10.2f} │ {abs(analysis['timing']['p90_timing'] - analysis['timing']['target_p90_timing']):>10.2f} │")
                print(f"  │ Uncertainty (s)         │ {analysis['timing']['timing_uncertainty']:>10.2f} │ {'N/A':>10} │ {'N/A':>10} │")
                print(f"  │ Consistency (std dev)   │ {analysis['timing']['timing_consistency']:>10.2f} │ {analysis['timing']['target_timing_consistency']:>10.2f} │ {abs(analysis['timing']['timing_consistency'] - analysis['timing']['target_timing_consistency']):>10.2f} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                
                # Calculate actions per gamestate metrics
                batch_size = action_targets.shape[0]
                actions_per_gamestate_target = valid_mask.sum(dim=1).cpu().numpy()  # [B] - actions per gamestate
                
                # Calculate predicted actions per gamestate based on ACTUAL model predictions
                # The model should be predicting up to 100 actions per gamestate
                # We need to determine which actions are "active" based on model outputs
                
                # Method 1: Count actions where the model predicts non-zero timing
                time_preds = model_outputs['time_q']  # [B, A, 3]
                median_timing_preds = time_preds[:, :, 1]  # [B, A] - q0.5 predictions
                
                # Count actions where timing prediction > 0 (indicating an active action)
                # Use a more sophisticated threshold based on the model's confidence
                actions_per_gamestate_pred = []
                for i in range(batch_size):
                    # Count actions where timing prediction > threshold
                    # The threshold should be based on what constitutes a "real" action timing
                    # For now, use a small threshold but this could be learned
                    threshold = 0.001  # 1ms - very small threshold
                    active_actions = (median_timing_preds[i] > threshold).sum().item()
                    actions_per_gamestate_pred.append(active_actions)
                
                actions_per_gamestate_pred = np.array(actions_per_gamestate_pred)
                
                                # Method 2: Alternative approach using event predictions
                # Count actions where the model predicts any event type with confidence
                if 'event_logits' in model_outputs:
                    event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)  # [B, A, 4]
                    max_event_probs = event_probs.max(dim=-1)[0]  # [B, A] - max probability per action
                    
                    # Count actions with high event confidence (>0.1)
                    actions_per_gamestate_pred_alt = []
                    for i in range(batch_size):
                        confident_actions = (max_event_probs[i] > 0.1).sum().item()
                        actions_per_gamestate_pred_alt.append(confident_actions)
                    
                    actions_per_gamestate_pred_alt = np.array(actions_per_gamestate_pred_alt)
                    
                    # Method 3: Combined confidence approach
                    # An action is "predicted" if it has either good timing OR good event confidence
                    combined_confidence = []
                    for i in range(batch_size):
                        # Timing confidence: normalize timing predictions to [0, 1]
                        timing_confidence = torch.clamp(median_timing_preds[i] / 0.6, 0, 1)  # 0.6s = max gamestate time
                        
                        # Event confidence: already [0, 1]
                        event_confidence = max_event_probs[i]
                        
                        # Combined confidence: max of timing and event confidence
                        combined = torch.maximum(timing_confidence, event_confidence)
                        
                        # Count actions with combined confidence > threshold
                        confident_actions = (combined > 0.05).sum().item()  # 5% threshold
                        combined_confidence.append(confident_actions)
                    
                    combined_confidence = np.array(combined_confidence)
                    
                    # Use the maximum of all three methods
                    actions_per_gamestate_pred = np.maximum.reduce([
                        actions_per_gamestate_pred,      # timing-based
                        actions_per_gamestate_pred_alt,  # event-based
                        combined_confidence              # combined confidence
                    ])
                    
                    print(f"  - Combined confidence count: {combined_confidence}")
                
                # Debug: Show model confidence across all action slots for first gamestate
                if batch_size > 0:
                    print(f"\n🔍 Model Confidence Analysis (Gamestate 0):")
                    print(f"  - Timing predictions (all 100 slots): {median_timing_preds[0].cpu().numpy()}")
                    if 'event_logits' in model_outputs:
                        event_probs = torch.softmax(model_outputs['event_logits'], dim=-1)
                        max_probs = event_probs[0].max(dim=-1)[0].cpu().numpy()
                        print(f"  - Event max probs (all 100 slots): {max_probs}")
                        print(f"  - Actions with high event confidence (>0.1): {(max_probs > 0.1).sum()}")
                        print(f"  - Actions with medium event confidence (>0.05): {(max_probs > 0.05).sum()}")
                        print(f"  - Actions with low event confidence (>0.01): {(max_probs > 0.01).sum()}")
                
                # Method 4: Use sequence length prediction directly (most direct approach)
                if 'sequence_length' in model_outputs:
                    seq_lengths = model_outputs['sequence_length'].cpu().numpy()
                    # The model predicts how many actions should be in each gamestate
                    actions_per_gamestate_pred_seq = seq_lengths.astype(int)
                    
                    # Use the maximum of all four methods
                    actions_per_gamestate_pred = np.maximum.reduce([
                        actions_per_gamestate_pred,  # timing-based
                        actions_per_gamestate_pred_alt if 'event_logits' in model_outputs else actions_per_gamestate_pred,  # event-based
                        combined_confidence if 'event_logits' in model_outputs else actions_per_gamestate_pred,  # combined confidence
                        actions_per_gamestate_pred_seq  # sequence length-based
                    ])
                    
                    print(f"  - Sequence length-based count: {actions_per_gamestate_pred_seq}")
                
                # Calculate per-gamestate statistics
                target_actions_per_gamestate_mean = np.mean(actions_per_gamestate_target)
                target_actions_per_gamestate_std = np.std(actions_per_gamestate_target)
                pred_actions_per_gamestate_mean = np.mean(actions_per_gamestate_pred)
                pred_actions_per_gamestate_std = np.std(actions_per_gamestate_pred)
                
                # Print detailed actions per gamestate table
                print(f"\n🎯 Actions per Gamestate Analysis (600ms windows):")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                print(f"  │ Mean Actions/Gamestate  │ {pred_actions_per_gamestate_mean:>10.1f} │ {target_actions_per_gamestate_mean:>10.1f} │ {abs(pred_actions_per_gamestate_mean - target_actions_per_gamestate_mean):>10.1f} │")
                print(f"  │ Median Actions/Gamestate│ {np.median(actions_per_gamestate_pred):>10.0f} │ {np.median(actions_per_gamestate_target):>10.0f} │ {abs(np.median(actions_per_gamestate_pred) - np.median(actions_per_gamestate_target)):>10.0f} │")
                print(f"  │ P10 Actions/Gamestate   │ {np.percentile(actions_per_gamestate_pred, 10):>10.0f} │ {np.percentile(actions_per_gamestate_target, 10):>10.0f} │ {abs(np.percentile(actions_per_gamestate_pred, 10) - np.percentile(actions_per_gamestate_target, 10)):>10.0f} │")
                print(f"  │ P25 Actions/Gamestate   │ {np.percentile(actions_per_gamestate_pred, 25):>10.0f} │ {np.percentile(actions_per_gamestate_target, 25):>10.0f} │ {abs(np.percentile(actions_per_gamestate_pred, 25) - np.percentile(actions_per_gamestate_target, 25)):>10.0f} │")
                print(f"  │ P75 Actions/Gamestate   │ {np.percentile(actions_per_gamestate_pred, 75):>10.0f} │ {np.percentile(actions_per_gamestate_target, 75):>10.0f} │ {abs(np.percentile(actions_per_gamestate_pred, 75) - np.percentile(actions_per_gamestate_target, 75)):>10.0f} │")
                print(f"  │ P90 Actions/Gamestate   │ {np.percentile(actions_per_gamestate_pred, 90):>10.0f} │ {np.percentile(actions_per_gamestate_target, 90):>10.0f} │ {abs(np.percentile(actions_per_gamestate_pred, 90) - np.percentile(actions_per_gamestate_target, 90)):>10.0f} │")
                print(f"  │ Std Dev Actions/Gamestate│ {pred_actions_per_gamestate_std:>10.1f} │ {target_actions_per_gamestate_std:>10.1f} │ {abs(pred_actions_per_gamestate_std - target_actions_per_gamestate_std):>10.1f} │")
                print(f"  │ Min Actions/Gamestate   │ {np.min(actions_per_gamestate_pred):>10.0f} │ {np.min(actions_per_gamestate_target):>10.0f} │ {abs(np.min(actions_per_gamestate_pred) - np.min(actions_per_gamestate_target)):>10.0f} │")
                print(f"  │ Max Actions/Gamestate   │ {np.max(actions_per_gamestate_pred):>10.0f} │ {np.max(actions_per_gamestate_target):>10.0f} │ {abs(np.max(actions_per_gamestate_pred) - np.max(actions_per_gamestate_target)):>10.0f} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                
                # Store actions per gamestate analysis for pretty printer
                analysis['actions_per_gamestate'] = {
                    'mean_actions_pred': pred_actions_per_gamestate_mean,
                    'mean_actions_target': target_actions_per_gamestate_mean
                }
                
                # Debug info only shown occasionally
                if epoch % 5 == 0:  # Only show detailed debug every 5 epochs
                    printer.print_debug_info(f"Mean predicted actions: {pred_actions_per_gamestate_mean:.1f}, Target: {target_actions_per_gamestate_mean:.1f}")
                
                # Store sequence length info for pretty printer
                if 'sequence_length' in model_outputs:
                    seq_lengths = model_outputs['sequence_length'].cpu().numpy()
                    analysis['sequence_length'] = {
                        'mean_predicted': np.mean(seq_lengths),
                        'min_predicted': np.min(seq_lengths),
                        'max_predicted': np.max(seq_lengths)
                    }
                
                # Burst analysis - analyze action timing patterns
                if valid_time_targets.numel() > 0:
                    self._analyze_action_bursts(valid_time_targets, analysis, epoch)
            
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
                # Note: gamestates are normalized, so we need to denormalize for display
                player_positions_normalized = gamestates[:, :, :2]
                
                # Denormalize player positions (multiply by world_scale = 10000)
                world_scale = 10000.0
                player_positions = player_positions_normalized * world_scale
                
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
            
            # Create actions per gamestate visualization (commented out to avoid scope issues)
            # self._create_actions_per_gamestate_visualization(action_targets, valid_mask, analysis, epoch)
            
            # Store analysis
            analysis['epoch'] = epoch
            self.analysis_history.append(analysis)
            
            # Save to file
            self._save_analysis(analysis, epoch)
            
            # Use pretty printer to display the analysis
            printer.print_behavioral_analysis(analysis, epoch)
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            printer.print_debug_info(f"Simplified analysis failed: {error_msg}", "ERROR")
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
    
    def _safe_mean(self, tensor: torch.Tensor) -> float:
        """Safely compute mean of tensor, handling empty tensors"""
        if tensor.numel() == 0:
            return 0.0
        return float(tensor.mean().item())
    
    def _safe_std(self, tensor: torch.Tensor) -> float:
        """Safely compute std of tensor, handling empty tensors"""
        if tensor.numel() == 0:
            return 0.0
        return float(tensor.std().item())
    
    def _safe_min_max(self, tensor: torch.Tensor) -> Tuple[float, float]:
        """Safely compute min/max of tensor, handling empty tensors"""
        if tensor.numel() == 0:
            return 0.0, 0.0
        return float(tensor.min().item()), float(tensor.max().item())
    
    def _safe_percentile(self, tensor: torch.Tensor, percentile: float) -> float:
        """Safely compute percentile of tensor, handling empty tensors"""
        if tensor.numel() == 0:
            return 0.0
        # Convert percentile to quantile (0-1 range)
        quantile = percentile / 100.0
        return float(torch.quantile(tensor, quantile).item())
    
    def _analyze_action_bursts(self, time_targets: torch.Tensor, analysis: Dict, epoch: int):
        """Analyze action burst patterns in timing data"""
        try:
            # Convert to numpy for easier analysis
            times = time_targets.cpu().numpy()
            
            # Define burst thresholds
            fast_threshold = 0.1  # Actions within 0.1s are considered "fast"
            pause_threshold = 1.0  # Actions with >1s gap are considered "pauses"
            
            # Analyze burst patterns
            fast_actions = times[times <= fast_threshold]
            pause_actions = times[times >= pause_threshold]
            medium_actions = times[(times > fast_threshold) & (times < pause_threshold)]
            
            # Count burst sequences (consecutive fast actions)
            burst_sequences = []
            current_burst = []
            
            for i, time in enumerate(times):
                if time <= fast_threshold:
                    current_burst.append(time)
                else:
                    if len(current_burst) > 0:
                        burst_sequences.append(current_burst)
                        current_burst = []
            
            # Don't forget the last burst
            if len(current_burst) > 0:
                burst_sequences.append(current_burst)
            
            # Calculate burst statistics
            burst_lengths = [len(burst) for burst in burst_sequences]
            burst_durations = [sum(burst) for burst in burst_sequences]
            
            analysis['bursts'] = {
                'total_actions': len(times),
                'fast_actions': len(fast_actions),
                'medium_actions': len(medium_actions),
                'pause_actions': len(pause_actions),
                'fast_percentage': len(fast_actions) / len(times) * 100,
                'pause_percentage': len(pause_actions) / len(times) * 100,
                'num_bursts': len(burst_sequences),
                'avg_burst_length': np.mean(burst_lengths) if burst_lengths else 0,
                'max_burst_length': max(burst_lengths) if burst_lengths else 0,
                'avg_burst_duration': np.mean(burst_durations) if burst_durations else 0,
                'fast_timing_mean': np.mean(fast_actions) if len(fast_actions) > 0 else 0,
                'pause_timing_mean': np.mean(pause_actions) if len(pause_actions) > 0 else 0,
            }
            
            # Display burst analysis
            print(f"\n💥 Action Burst Analysis:")
            print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
            print(f"  │ Metric                  │ Count        │ Percentage   │ Avg Timing   │")
            print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
            print(f"  │ Fast Actions (≤0.1s)    │ {analysis['bursts']['fast_actions']:>10} │ {analysis['bursts']['fast_percentage']:>10.1f}% │ {analysis['bursts']['fast_timing_mean']:>10.3f}s │")
            print(f"  │ Medium Actions (0.1-1s) │ {analysis['bursts']['medium_actions']:>10} │ {100-analysis['bursts']['fast_percentage']-analysis['bursts']['pause_percentage']:>10.1f}% │ {'N/A':>10} │")
            print(f"  │ Pause Actions (≥1s)     │ {analysis['bursts']['pause_actions']:>10} │ {analysis['bursts']['pause_percentage']:>10.1f}% │ {analysis['bursts']['pause_timing_mean']:>10.3f}s │")
            print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
            print(f"\n🎯 Burst Sequences:")
            print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
            print(f"  │ Metric                  │ Count        │ Average      │ Maximum      │")
            print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
            print(f"  │ Number of Bursts        │ {analysis['bursts']['num_bursts']:>10} │ {'N/A':>10} │ {'N/A':>10} │")
            print(f"  │ Actions per Burst       │ {'N/A':>10} │ {analysis['bursts']['avg_burst_length']:>10.1f} │ {analysis['bursts']['max_burst_length']:>10} │")
            print(f"  │ Burst Duration (s)      │ {'N/A':>10} │ {analysis['bursts']['avg_burst_duration']:>10.3f} │ {'N/A':>10} │")
            print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Show some example burst sequences
            if burst_sequences:
                print(f"\n📋 Example Burst Sequences (first 5):")
                for i, burst in enumerate(burst_sequences[:5]):
                    burst_times = [f"{t:.3f}s" for t in burst]
                    print(f"  Burst {i+1}: {len(burst)} actions in {sum(burst):.3f}s - {', '.join(burst_times)}")
            
            # Create temporal action visualization
            self._create_temporal_visualization(time_targets, analysis, epoch)
            
            # Create actions per gamestate visualization (commented out to avoid scope issues)
            # self._create_actions_per_gamestate_visualization(action_targets, valid_mask, analysis, epoch)
                
        except Exception as e:
            print(f"⚠️  Burst analysis failed: {e}")
            analysis['bursts'] = {
                'total_actions': 0,
                'fast_actions': 0,
                'medium_actions': 0,
                'pause_actions': 0,
                'fast_percentage': 0,
                'pause_percentage': 0,
                'num_bursts': 0,
                'avg_burst_length': 0,
                'max_burst_length': 0,
                'avg_burst_duration': 0,
                'fast_timing_mean': 0,
                'pause_timing_mean': 0,
            }
    
    def _create_temporal_visualization(self, time_targets: torch.Tensor, analysis: Dict, epoch: int):
        """Create a temporal visualization showing actions per gamestate (600ms windows)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # The model predicts actions for each gamestate (every 600ms)
            # We need to count how many actions are predicted vs actual per gamestate
            
            target_times = time_targets.cpu().numpy()
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Timing distribution histogram
            ax1.hist(target_times, bins=50, alpha=0.7, color='blue', label='Target Timing Distribution')
            ax1.set_xlabel('Time Delta (s)')
            ax1.set_ylabel('Count')
            ax1.set_title(f'Epoch {epoch}: Action Timing Distribution')
            ax1.set_xlim(0, 0.5)  # Focus on the fast actions
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Cumulative time series showing burst patterns
            cumulative_times = np.cumsum(target_times)
            ax2.plot(cumulative_times, target_times, 'b-', alpha=0.7, linewidth=1, label='Target Timing')
            ax2.set_xlabel('Cumulative Time (s)')
            ax2.set_ylabel('Action Interval (s)')
            ax2.set_title('Action Timing Over Time (Burst Pattern)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Highlight fast actions
            fast_mask = target_times <= 0.1
            if np.any(fast_mask):
                fast_cumulative = cumulative_times[fast_mask]
                fast_times = target_times[fast_mask]
                ax2.scatter(fast_cumulative, fast_times, c='red', s=20, alpha=0.6, label='Fast Actions (≤0.1s)')
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_timing_analysis.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Timing analysis saved: {plot_path}")
            
            # Print summary statistics
            print(f"\n📈 Timing Analysis Summary:")
            print(f"  Total actions: {len(target_times)}")
            print(f"  Total session time: {cumulative_times[-1]:.2f}s")
            print(f"  Actions per second: {len(target_times) / cumulative_times[-1]:.1f}")
            print(f"  Fast actions (≤0.1s): {np.sum(fast_mask)} ({np.sum(fast_mask)/len(target_times)*100:.1f}%)")
            print(f"  Medium actions (0.1-1s): {np.sum((target_times > 0.1) & (target_times <= 1.0))} ({np.sum((target_times > 0.1) & (target_times <= 1.0))/len(target_times)*100:.1f}%)")
            print(f"  Slow actions (>1s): {np.sum(target_times > 1.0)} ({np.sum(target_times > 1.0)/len(target_times)*100:.1f}%)")
            print(f"  Mean timing: {np.mean(target_times):.3f}s ± {np.std(target_times):.3f}s")
            
        except ImportError:
            print("⚠️  matplotlib not available for temporal visualization")
        except Exception as e:
            print(f"⚠️  Temporal visualization failed: {e}")
    
    def _create_actions_per_gamestate_visualization(self, action_targets: torch.Tensor, valid_mask: torch.Tensor, 
                                                   analysis: Dict, epoch: int):
        """Create visualization showing actions per gamestate (600ms windows)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # action_targets shape: [B, A, 7] where B=batch_size, A=max_actions_per_timestep
            # valid_mask shape: [B, A] - indicates which actions are valid
            
            # Count valid actions per gamestate (per batch item)
            batch_size = action_targets.shape[0]
            actions_per_gamestate = valid_mask.sum(dim=1).cpu().numpy()  # [B] - actions per gamestate
            
            # Get prediction counts (we need to estimate this from the model outputs)
            # For now, let's use the mean prediction timing to estimate actions per gamestate
            pred_actions_per_gamestate = None
            if 'timing' in analysis and 'mean_timing' in analysis['timing']:
                # Estimate: if model predicts 0.447s average timing, how many actions in 600ms?
                pred_mean_timing = analysis['timing']['mean_timing']
                actions_in_600ms = 0.6 / pred_mean_timing  # 600ms / predicted_timing
                pred_actions_per_gamestate = np.full(batch_size, actions_in_600ms)
            
            # Create the plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # X-axis: gamestate index (every 600ms)
            gamestate_indices = np.arange(batch_size)
            
            # Plot target actions per gamestate
            ax.scatter(gamestate_indices, actions_per_gamestate, alpha=0.7, color='blue', 
                      label='Target Actions per Gamestate', s=30)
            
            # Plot predicted actions per gamestate (if available)
            if pred_actions_per_gamestate is not None:
                ax.scatter(gamestate_indices, pred_actions_per_gamestate, alpha=0.7, color='red', 
                          label='Predicted Actions per Gamestate', s=30)
            
            ax.set_xlabel('Gamestate Index (every 600ms)')
            ax.set_ylabel('Number of Actions')
            ax.set_title(f'Epoch {epoch}: Actions per Gamestate (600ms windows)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_actions_per_gamestate.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Actions per gamestate visualization saved: {plot_path}")
            
            # Print summary statistics
            print(f"\n📈 Actions per Gamestate Summary:")
            print(f"  Total gamestates: {batch_size}")
            print(f"  Target actions per gamestate: {np.mean(actions_per_gamestate):.1f} ± {np.std(actions_per_gamestate):.1f}")
            print(f"  Target actions range: {np.min(actions_per_gamestate)} - {np.max(actions_per_gamestate)}")
            if pred_actions_per_gamestate is not None:
                print(f"  Predicted actions per gamestate: {np.mean(pred_actions_per_gamestate):.1f} ± {np.std(pred_actions_per_gamestate):.1f}")
                print(f"  Predicted actions range: {np.min(pred_actions_per_gamestate):.1f} - {np.max(pred_actions_per_gamestate):.1f}")
            
        except ImportError:
            print("⚠️  matplotlib not available for actions per gamestate visualization")
        except Exception as e:
            print(f"⚠️  Actions per gamestate visualization failed: {e}")
