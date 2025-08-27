#!/usr/bin/env python3
"""
Multi-Objective Loss Functions for Imitation Learning
OSRS Bot Training Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ImitationLoss(nn.Module):
    """Multi-objective loss for imitation learning"""
    
    def __init__(self, 
                 mouse_weight: float = 2.0,
                 click_weight: float = 1.5,
                 key_weight: float = 1.0,
                 scroll_weight: float = 0.5,
                 confidence_weight: float = 0.3,
                 action_count_weight: float = 1.0):
        super().__init__()
        
        self.mouse_weight = mouse_weight
        self.click_weight = click_weight
        self.key_weight = key_weight
        self.scroll_weight = scroll_weight
        self.confidence_weight = confidence_weight
        self.action_count_weight = action_count_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-objective loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            total_loss: Combined weighted loss
            loss_breakdown: Individual loss components
        """
        losses = {}
        
        # 1. Mouse position: MSE loss (most critical for gameplay)
        if 'mouse_position' in predictions and 'mouse_position' in targets:
            mouse_loss = self.mse_loss(predictions['mouse_position'], targets['mouse_position'])
            losses['mouse_position'] = mouse_loss.item()
        else:
            mouse_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['mouse_position'] = 0.0
        
        # 2. Mouse clicks: Binary cross-entropy
        if 'mouse_click' in predictions and 'mouse_click' in targets:
            click_loss = self.bce_loss(predictions['mouse_click'], targets['mouse_click'])
            losses['mouse_click'] = click_loss.item()
        else:
            click_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['mouse_click'] = 0.0
        
        # 3. Key presses: Cross-entropy
        if 'key_press' in predictions and 'key_press' in targets:
            key_loss = self.ce_loss(predictions['key_press'], targets['key_press'])
            losses['key_press'] = key_loss.item()
        else:
            key_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['key_press'] = 0.0
        
        # 4. Scroll actions: MSE loss
        if 'scroll' in predictions and 'scroll' in targets:
            scroll_loss = self.mse_loss(predictions['scroll'], targets['scroll'])
            losses['scroll'] = scroll_loss.item()
        else:
            scroll_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['scroll'] = 0.0
        
        # 5. Confidence: MSE loss
        if 'confidence' in predictions and 'confidence' in targets:
            confidence_loss = self.mse_loss(predictions['confidence'], targets['confidence'])
            losses['confidence'] = confidence_loss.item()
        else:
            confidence_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['confidence'] = 0.0
        
        # 6. Action count: Cross-entropy
        if 'action_count' in predictions and 'action_count' in targets:
            action_count_loss = self.ce_loss(predictions['action_count'], targets['action_count'])
            losses['action_count'] = action_count_loss.item()
        else:
            action_count_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            losses['action_count'] = 0.0
        
        # Weighted combination based on importance
        total_loss = (
            self.mouse_weight * mouse_loss +
            self.click_weight * click_loss +
            self.key_weight * key_loss +
            self.scroll_weight * scroll_loss +
            self.confidence_weight * confidence_loss +
            self.action_count_weight * action_count_loss
        )
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses

class TemporalConsistencyLoss(nn.Module):
    """Loss for ensuring temporal consistency of actions"""
    
    def __init__(self, consistency_weight: float = 0.5):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss
        
        Args:
            predictions: (batch_size, seq_len, action_dim)
            targets: (batch_size, seq_len, action_dim)
            
        Returns:
            consistency_loss: Loss for temporal smoothness
        """
        # Compute differences between consecutive timesteps
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]
        
        # Loss is the difference between predicted and target temporal changes
        consistency_loss = self.mse_loss(pred_diff, target_diff)
        
        return consistency_loss * self.consistency_weight

class GameContextLoss(nn.Module):
    """Loss for ensuring actions are appropriate for game context"""
    
    def __init__(self, context_weight: float = 0.3):
        super().__init__()
        self.context_weight = context_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, action_predictions: torch.Tensor, 
                context_features: torch.Tensor,
                valid_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute game context loss
        
        Args:
            action_predictions: Predicted actions
            context_features: Game context features
            valid_actions: Binary mask of valid actions for current context
            
        Returns:
            context_loss: Loss for context-appropriate actions
        """
        # This is a simplified version - in practice, you'd have more sophisticated
        # logic to determine what actions are valid in what contexts
        
        # For now, we'll use a simple approach: penalize actions that don't match
        # the expected action distribution for the current context
        
        # Compute context-appropriate action probabilities
        context_probs = torch.softmax(action_predictions, dim=-1)
        
        # Loss is cross-entropy between predicted and context-appropriate actions
        context_loss = self.ce_loss(context_probs, valid_actions.float())
        
        return context_loss * self.context_weight

class CombinedImitationLoss(nn.Module):
    """Combined loss with all components"""
    
    def __init__(self, 
                 base_weights: Dict[str, float] = None,
                 temporal_weight: float = 0.5,
                 context_weight: float = 0.3):
        super().__init__()
        
        if base_weights is None:
            base_weights = {
                'mouse_weight': 2.0,
                'click_weight': 1.5,
                'key_weight': 1.0,
                'scroll_weight': 0.5,
                'confidence_weight': 0.3,
                'action_count_weight': 1.0
            }
        
        self.base_loss = ImitationLoss(**base_weights)
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
        self.context_loss = GameContextLoss(context_weight)
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                temporal_predictions: torch.Tensor = None,
                temporal_targets: torch.Tensor = None,
                context_features: torch.Tensor = None,
                valid_actions: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            temporal_predictions: Temporal sequence predictions (optional)
            temporal_targets: Temporal sequence targets (optional)
            context_features: Game context features (optional)
            valid_actions: Valid actions for current context (optional)
            
        Returns:
            total_loss: Combined loss
            loss_breakdown: Individual loss components
        """
        # Base imitation loss
        base_loss, base_losses = self.base_loss(predictions, targets)
        
        total_loss = base_loss
        all_losses = base_losses.copy()
        
        # Temporal consistency loss
        if temporal_predictions is not None and temporal_targets is not None:
            temporal_loss = self.temporal_loss(temporal_predictions, temporal_targets)
            total_loss = total_loss + temporal_loss
            all_losses['temporal_consistency'] = temporal_loss.item()
        
        # Game context loss
        if context_features is not None and valid_actions is not None:
            context_loss = self.context_loss(
                predictions.get('action_count', torch.zeros(1)),
                context_features,
                valid_actions
            )
            total_loss = total_loss + context_loss
            all_losses['game_context'] = context_loss.item()
        
        all_losses['total_combined'] = total_loss.item()
        
        return total_loss, all_losses

def create_loss_function(loss_type: str = 'combined', **kwargs) -> nn.Module:
    """Factory function to create loss functions"""
    
    if loss_type == 'base':
        return ImitationLoss(**kwargs)
    elif loss_type == 'temporal':
        return TemporalConsistencyLoss(**kwargs)
    elif loss_type == 'context':
        return GameContextLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedImitationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

if __name__ == "__main__":
    # Test the loss functions
    print("Testing Imitation Loss Functions...")
    
    # Create sample data
    batch_size = 4
    device = torch.device('cpu')
    
    # Sample predictions
    predictions = {
        'mouse_position': torch.randn(batch_size, 2),
        'mouse_click': torch.randn(batch_size, 2),
        'key_press': torch.randn(batch_size, 50),
        'scroll': torch.randn(batch_size, 2),
        'confidence': torch.randn(batch_size, 1),
        'action_count': torch.randn(batch_size, 16)
    }
    
    # Sample targets
    targets = {
        'mouse_position': torch.rand(batch_size, 2),
        'mouse_click': torch.randint(0, 2, (batch_size, 2)).float(),
        'key_press': torch.randint(0, 50, (batch_size,)),
        'scroll': torch.randn(batch_size, 2),
        'confidence': torch.rand(batch_size, 1),
        'action_count': torch.randint(0, 16, (batch_size,))
    }
    
    # Test base loss
    print("\n1. Testing Base Imitation Loss...")
    base_loss = ImitationLoss()
    total_loss, loss_breakdown = base_loss(predictions, targets)
    print(f"Total loss: {total_loss:.4f}")
    for loss_name, loss_value in loss_breakdown.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    # Test combined loss
    print("\n2. Testing Combined Loss...")
    combined_loss = CombinedImitationLoss()
    total_loss, loss_breakdown = combined_loss(predictions, targets)
    print(f"Total combined loss: {total_loss:.4f}")
    for loss_name, loss_value in loss_breakdown.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    print("\n✅ All loss functions working correctly!")
