#!/usr/bin/env python3
"""
Loss function for OSRS action tensor imitation learning
Handles 8-feature action tensors: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from .. import config as CFG

class ActionTensorLoss(nn.Module):
    """
    Loss function for action tensor imitation learning
    """
    
    def __init__(self, weights: Dict[str, float] = None, event_cls_weights: List[float] = None):
        super().__init__()
        
        # Default loss weights
        self.weights = weights or {
            'action_count': 1.0,           # How many actions to expect
            'action_tensor': 2.0,           # Action tensor features (most important)
        }
        
        # Event classification weights for mutually exclusive events
        self.event_cls_weights = event_cls_weights
    
    def forward(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the total loss for action sequences.
        
        Args:
            predictions: Model predictions (batch_size, max_actions + 1, 8) - action count at index 0
            targets: Ground truth targets (batch_size, max_actions + 1, 8) - action count at index 0
            
        Returns:
            Total loss
        """
        # Extract action counts from index 0, position 0
        pred_counts = predictions[:, 0, 0]    # (batch_size,) - predicted action counts
        target_counts = targets[:, 0, 0]      # (batch_size,) - actual action counts
        
        # Extract action tensors (skip index 0 which contains count)
        pred_actions = predictions[:, 1:, :]  # (batch_size, max_actions, 8)
        target_actions = targets[:, 1:, :]    # (batch_size, max_actions, 8)
        
        total_loss = 0.0
        
        # Action count loss: MSE between predicted and actual counts
        count_loss = F.mse_loss(pred_counts, target_counts)
        total_loss += self.weights['action_count'] * count_loss
        
        # Action tensor loss: MSE for all actions
        # Compute loss only for valid actions (up to action_count)
        batch_loss = 0.0
        valid_batches = 0
        
        for i, count in enumerate(target_counts):
            count_int = int(count.item())
            if count_int > 0:
                # Only compute loss for actions that actually exist
                valid_actions = pred_actions[i, :count_int, :]  # (count, 8)
                target_valid_actions = target_actions[i, :count_int, :]  # (count, 8)
                
                # MSE loss for all 8 features
                action_loss = F.mse_loss(valid_actions, target_valid_actions)
                batch_loss += action_loss
                valid_batches += 1
        
        if valid_batches > 0:
            # Average across valid batches
            action_tensor_loss = batch_loss / valid_batches
            total_loss += self.weights['action_tensor'] * action_tensor_loss
        
        return total_loss


def _make_event_criterion(num_classes=4):
    """Create event classification criterion with optional class weights"""
    if CFG.event_cls_weights is None:
        return nn.CrossEntropyLoss()
    w = torch.tensor(CFG.event_cls_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=w)


def compute_tempered_event_weights(counts, power=None, use_log=True):
    """
    counts: list/np array of class counts in order [MOVE, CLICK, KEY, SCROLL]
    Returns weights normalized to weight[MOVE] == 1.0
    - use_log:  weights ∝ log(1 + inv_freq)
    - power:    alternatively weights ∝ inv_freq**power (e.g., 0.35–0.5)
    """
    import numpy as np
    c = np.asarray(counts, dtype=np.float64)
    freq = c / c.sum()
    inv = 1.0 / np.maximum(freq, 1e-12)
    if use_log:
        w = np.log1p(inv)
    else:
        assert power is not None
        w = inv ** float(power)
    w = w / w[0]  # normalize to MOVE=1.0
    return w.tolist()


# Create the event criterion instance
crit_event = _make_event_criterion()
