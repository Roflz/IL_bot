#!/usr/bin/env python3
"""
Loss function for OSRS action tensor imitation learning
Handles 8-feature action tensors: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ActionTensorLoss(nn.Module):
    """
    Loss function for action tensor imitation learning
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        
        # Default loss weights
        self.weights = weights or {
            'action_count': 1.0,           # How many actions to expect
            'action_tensor': 2.0,           # Action tensor features (most important)
        }
    
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
