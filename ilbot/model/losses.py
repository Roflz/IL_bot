#!/usr/bin/env python3
"""
Loss functions for the unified event system model.
Implements comprehensive loss computation for event classification, time quantiles, and XY uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class PinballLoss(nn.Module):
    """Pinball (quantile) loss for robust time prediction."""
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pinball loss for quantile regression.
        
        Args:
            predictions: [B, A, Q] - predicted quantiles for each action
            targets: [B, A] - ground truth time values
            mask: [B, A] - optional mask for valid actions
            
        Returns:
            Average pinball loss across all valid predictions
        """
        B, A, Q = predictions.shape
        assert Q == len(self.quantiles), f"Expected {len(self.quantiles)} quantiles, got {Q}"
        
        # Expand targets to match predictions: [B, A] -> [B, A, Q]
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, Q)
        
        # Compute pinball loss for each quantile
        # L = max(q * (y - ŷ), (1-q) * (ŷ - y))
        diff = targets_expanded - predictions
        quantiles_expanded = self.quantiles.to(predictions.device).view(1, 1, -1)
        
        pinball_loss = torch.max(
            quantiles_expanded * diff,
            (1 - quantiles_expanded) * (-diff)
        )
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, Q)
            pinball_loss = pinball_loss * mask_expanded.float()
            valid_count = mask_expanded.sum().clamp_min(1.0)
            return pinball_loss.sum() / valid_count
        else:
            return pinball_loss.mean()


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for uncertainty-aware XY prediction."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 10.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss for heteroscedastic regression.
        
        Args:
            mu: [B, A] - predicted mean values
            log_sigma: [B, A] - predicted log-standard-deviation values
            targets: [B, A] - ground truth values
            mask: [B, A] - optional mask for valid actions
            
        Returns:
            Average Gaussian NLL loss across all valid predictions
        """
        # Convert log-sigma to sigma with bounds
        # Create tensors on the same device as log_sigma
        sigma_min_tensor = torch.tensor(self.sigma_min, device=log_sigma.device, dtype=log_sigma.dtype)
        sigma_max_tensor = torch.tensor(self.sigma_max, device=log_sigma.device, dtype=log_sigma.dtype)
        sigma = torch.exp(torch.clamp(log_sigma, 
                                    min=torch.log(sigma_min_tensor), 
                                    max=torch.log(sigma_max_tensor)))
        
        # Gaussian NLL: 0.5 * log(2πσ²) + 0.5 * ((x - μ)² / σ²)
        # Simplified: log(σ) + 0.5 * ((x - μ) / σ)² + 0.5 * log(2π)
        # We can ignore the constant 0.5 * log(2π) as it doesn't affect gradients
        
        squared_error = ((targets - mu) / sigma) ** 2
        nll_loss = torch.log(sigma) + 0.5 * squared_error
        
        # Apply mask if provided
        if mask is not None:
            nll_loss = nll_loss * mask.float()
            valid_count = mask.sum().clamp_min(1.0)
            return nll_loss.sum() / valid_count
        else:
            return nll_loss.mean()


class UnifiedEventLoss(nn.Module):
    """
    Comprehensive loss function for the unified event system.
    
    Handles:
    - Event classification (CLICK, KEY, SCROLL, MOVE)
    - Time quantile prediction
    - XY coordinate prediction with uncertainty
    - Event-specific auxiliary losses (button, key, scroll)
    """
    
    def __init__(self, 
                 data_config: Dict[str, Any] = None,
                 loss_weights: Dict[str, float] = None,
                 class_weights: Dict[str, torch.Tensor] = None,
                 time_quantiles: list = [0.1, 0.5, 0.9],
                 sigma_min: float = 0.01,
                 sigma_max: float = 10.0):
        super().__init__()
        
        # Data configuration for dynamic sizing
        self.data_config = data_config or {}
        self.event_types = data_config.get('event_types', 4) if data_config else 4
        
        # Default loss weights if none provided
        self.loss_weights = loss_weights or {
            'event': 1.0,      # Event classification
            'time': 1.0,       # Time quantiles
            'xy': 1.0,         # XY coordinates
            'button': 1.0,     # Button classification
            'key_action': 1.0, # Key action classification
            'key_id': 1.0,     # Key ID classification
            'scroll': 1.0      # Scroll classification
        }
        
        # Class weights for handling imbalanced distributions
        self.class_weights = class_weights or {}
        
        # Loss components - use dynamic event types
        self.event_loss = nn.CrossEntropyLoss(reduction='none')
        self.time_loss = PinballLoss(time_quantiles)
        self.xy_loss = GaussianNLLLoss(sigma_min, sigma_max)
        
        # Store quantiles for reference
        self.time_quantiles = torch.tensor(time_quantiles)
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                valid_mask: torch.Tensor,
                enum_sizes: Dict[str, Dict] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the complete unified event system loss.
        
        Args:
            predictions: Dictionary of model outputs
                - event_logits: [B, A, 4] - Event type logits
                - time_q: [B, A, 3] - Time quantiles
                - x_mu, x_logsig: [B, A] - X coordinate mean and log-sigma
                - y_mu, y_logsig: [B, A] - Y coordinate mean and log-sigma
                - button_logits: [B, A, 4] - Button classification logits
                - key_action_logits: [B, A, 3] - Key action classification logits
                - key_id_logits: [B, A, 151] - Key ID classification logits
                - scroll_y_logits: [B, A, 3] - Scroll classification logits
            targets: [B, A, 7] - Ground truth V2 action targets
                [time, x, y, button, key_action, key_id, scroll_y]
            valid_mask: [B, A] - Boolean mask for valid actions
            enum_sizes: Dictionary with enum information for auxiliary losses
            
        Returns:
            total_loss: Combined weighted loss
            loss_components: Dictionary of individual loss components
        """
        B, A, _ = targets.shape
        
        # Prepare mask
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.view(B, A)
        valid_mask = valid_mask.bool()
        
        # Extract target components
        time_target = targets[..., 0]      # [B, A] - time delta
        x_target = targets[..., 1]         # [B, A] - x coordinate
        y_target = targets[..., 2]         # [B, A] - y coordinate
        button_target = targets[..., 3]    # [B, A] - button type
        key_action_target = targets[..., 4] # [B, A] - key action type
        key_id_target = targets[..., 5]    # [B, A] - key ID
        scroll_y_target = targets[..., 6]  # [B, A] - scroll direction
        
        # Initialize loss components
        losses = {}
        
        # 1. Event Classification Loss
        # We need to derive event type from targets
        event_target = self._derive_event_target(button_target, key_action_target, scroll_y_target)
        event_loss = self._compute_event_loss(
            predictions['event_logits'], 
            event_target, 
            valid_mask
        )
        losses['event'] = event_loss
        
        # 2. Distribution Regularization Loss
        if hasattr(self, '_target_distribution'):
            dist_reg_loss = self._compute_distribution_regularization(
                predictions['event_logits'], valid_mask
            )
            losses['dist_reg'] = dist_reg_loss
        
        # 3. Time Quantile Loss
        time_loss = self._compute_time_loss(
            predictions['time_q'],
            time_target,
            valid_mask
        )
        losses['time'] = time_loss
        
        # 3. XY Coordinate Loss
        xy_loss = self._compute_xy_loss(
            predictions['x_mu'], predictions['x_logsig'],
            predictions['y_mu'], predictions['y_logsig'],
            x_target, y_target, valid_mask
        )
        losses['xy'] = xy_loss
        
        # 4. Auxiliary Losses (only for relevant event types)
        aux_losses = self._compute_auxiliary_losses(
            predictions, targets, valid_mask, event_target, enum_sizes
        )
        losses.update(aux_losses)
        
        # 5. Compute total weighted loss
        total_loss = sum(
            self.loss_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        
        return total_loss, losses
    
    def _derive_event_target(self, button_target: torch.Tensor, key_action_target: torch.Tensor, 
                           scroll_y_target: torch.Tensor) -> torch.Tensor:
        """
        Derive event type from action targets with proper priority.
        
        Priority order: CLICK > KEY > SCROLL > MOVE
        
        Returns:
            [B, A] tensor with event types: 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
        """
        B, A = button_target.shape
        event_target = torch.full((B, A), 3, dtype=torch.long, device=button_target.device)  # Default to MOVE
        
        # SCROLL: scroll_y != 0 (lowest priority)
        scroll_mask = scroll_y_target != 0
        event_target[scroll_mask] = 2
        
        # KEY: key_action != 0 (medium priority - overwrites SCROLL)
        key_mask = key_action_target != 0
        event_target[key_mask] = 1
        
        # CLICK: button != 0 (highest priority - overwrites KEY and SCROLL)
        click_mask = button_target != 0
        event_target[click_mask] = 0
        
        # Note: MOVE (3) is the default when no specific action is detected
        
        return event_target
    
    def _compute_event_loss(self, event_logits: torch.Tensor, event_target: torch.Tensor, 
                           valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute event classification loss with automatic class balancing."""
        # Get class weights if available, otherwise compute balanced weights
        weights = None
        if 'event' in self.class_weights:
            weights = self.class_weights['event'].to(event_logits.device)
        else:
            # Use global class weights if available, otherwise compute batch weights
            if hasattr(self, '_global_event_weights'):
                weights = self._global_event_weights.to(event_logits.device)
            else:
                # Automatically compute balanced class weights based on target distribution
                weights = self._compute_balanced_class_weights(event_target, valid_mask)
        
        # Compute cross-entropy loss with weights
        if weights is not None:
            # Use weighted cross-entropy loss
            event_loss = F.cross_entropy(
                event_logits.view(-1, 4), 
                event_target.view(-1), 
                weight=weights, 
                reduction='none'
            )
        else:
            # Fallback to unweighted loss
            event_loss = self.event_loss(event_logits.view(-1, 4), event_target.view(-1))
        
        # Apply mask and average
        event_loss = event_loss.view_as(valid_mask)
        event_loss = event_loss * valid_mask.float()
        valid_count = valid_mask.sum().clamp_min(1.0)
        
        return event_loss.sum() / valid_count
    
    def _compute_balanced_class_weights(self, event_target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute balanced class weights to handle imbalanced event distributions."""
        # Count occurrences of each event type in valid positions
        valid_events = event_target[valid_mask]
        
        if valid_events.numel() == 0:
            return None
        
        # Count each class
        class_counts = torch.bincount(valid_events, minlength=4)  # [4] for CLICK, KEY, SCROLL, MOVE
        
        # Avoid division by zero
        class_counts = class_counts.float().clamp(min=1.0)
        
        # Compute inverse frequency weights (more rare classes get higher weights)
        total_valid = class_counts.sum()
        class_weights = total_valid / (4.0 * class_counts)  # Normalize by number of classes
        
        # Normalize weights to sum to number of classes
        class_weights = class_weights / class_weights.mean()
        
        # Only print once per epoch to avoid spam
        if not hasattr(self, '_printed_weights_this_epoch'):
            print(f"🔍 Batch class weights: CLICK={class_weights[0]:.3f}, KEY={class_weights[1]:.3f}, SCROLL={class_weights[2]:.3f}, MOVE={class_weights[3]:.3f}")
            self._printed_weights_this_epoch = True
        
        return class_weights
    
    def set_global_event_weights(self, event_targets: torch.Tensor, valid_masks: torch.Tensor):
        """Set global class weights based on the entire dataset distribution."""
        # Collect all valid events from the dataset
        all_valid_events = []
        for event_target, valid_mask in zip(event_targets, valid_masks):
            valid_events = event_target[valid_mask]
            all_valid_events.append(valid_events)
        
        if not all_valid_events:
            return
        
        # Concatenate all valid events
        all_events = torch.cat(all_valid_events)
        
        # Count each class
        class_counts = torch.bincount(all_events, minlength=4)  # [4] for CLICK, KEY, SCROLL, MOVE
        
        # Avoid division by zero
        class_counts = class_counts.float().clamp(min=1.0)
        
        # Compute inverse frequency weights (more rare classes get higher weights)
        total_valid = class_counts.sum()
        class_weights = total_valid / (4.0 * class_counts)  # Normalize by number of classes
        
        # Normalize weights to sum to number of classes
        class_weights = class_weights / class_weights.mean()
        
        # Store global weights
        self._global_event_weights = class_weights
        
        # Store target distribution for regularization
        self._target_distribution = class_counts / class_counts.sum()
        
        print(f"🎯 Global class weights set: CLICK={class_weights[0]:.3f}, KEY={class_weights[1]:.3f}, SCROLL={class_weights[2]:.3f}, MOVE={class_weights[3]:.3f}")
        print(f"📊 Dataset distribution: CLICK={class_counts[0]:,}, KEY={class_counts[1]:,}, SCROLL={class_counts[2]:,}, MOVE={class_counts[3]:,}")
        print(f"🎯 Target distribution: CLICK={self._target_distribution[0]:.1%}, KEY={self._target_distribution[1]:.1%}, SCROLL={self._target_distribution[2]:.1%}, MOVE={self._target_distribution[3]:.1%}")
    
    def reset_epoch_flag(self):
        """Reset the epoch flag to allow printing weights again."""
        if hasattr(self, '_printed_weights_this_epoch'):
            delattr(self, '_printed_weights_this_epoch')
    
    def _compute_distribution_regularization(self, event_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute distribution regularization to encourage stable event predictions."""
        if not hasattr(self, '_target_distribution'):
            return torch.tensor(0.0, device=event_logits.device)
        
        # Get predicted probabilities
        event_probs = F.softmax(event_logits, dim=-1)  # [B, A, 4]
        
        # Average probabilities across batch and actions
        avg_probs = event_probs[valid_mask].mean(dim=0)  # [4]
        
        # Target distribution from dataset
        target_dist = self._target_distribution.to(event_probs.device)
        
        # KL divergence between predicted and target distribution
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        avg_probs = avg_probs.clamp(min=eps)
        target_dist = target_dist.clamp(min=eps)
        
        # KL divergence: KL(target || predicted)
        kl_loss = (target_dist * torch.log(target_dist / avg_probs)).sum()
        
        # Scale the regularization (stronger weight to force stable distributions)
        reg_weight = 1.0  # Increased from 0.1 to 1.0
        return reg_weight * kl_loss
    
    def _compute_time_loss(self, time_q: torch.Tensor, time_target: torch.Tensor, 
                          valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute time quantile loss."""
        return self.time_loss(time_q, time_target, valid_mask)
    
    def _compute_xy_loss(self, x_mu: torch.Tensor, x_logsig: torch.Tensor,
                         y_mu: torch.Tensor, y_logsig: torch.Tensor,
                         x_target: torch.Tensor, y_target: torch.Tensor,
                         valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute combined XY coordinate loss."""
        x_loss = self.xy_loss(x_mu, x_logsig, x_target, valid_mask)
        y_loss = self.xy_loss(y_mu, y_logsig, y_target, valid_mask)
        return x_loss + y_loss
    
    def _compute_auxiliary_losses(self, predictions: Dict[str, torch.Tensor],
                                 targets: torch.Tensor, valid_mask: torch.Tensor,
                                 event_target: torch.Tensor, enum_sizes: Dict) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for event-specific outputs."""
        aux_losses = {}
        
        # Button loss (only for CLICK events)
        if 'button_logits' in predictions:
            button_mask = valid_mask & (event_target == 0)  # CLICK events
            if button_mask.any():
                button_loss = self._compute_classification_loss(
                    predictions['button_logits'], targets[..., 3], button_mask, 'button'
                )
                aux_losses['button'] = button_loss
            else:
                aux_losses['button'] = torch.tensor(0.0, device=valid_mask.device)
        
        # Key action loss (only for KEY events)
        if 'key_action_logits' in predictions:
            key_action_mask = valid_mask & (event_target == 1)  # KEY events
            if key_action_mask.any():
                key_action_loss = self._compute_classification_loss(
                    predictions['key_action_logits'], targets[..., 4], key_action_mask, 'key_action'
                )
                aux_losses['key_action'] = key_action_loss
            else:
                aux_losses['key_action'] = torch.tensor(0.0, device=valid_mask.device)
        
        # Key ID loss (only for KEY events)
        if 'key_id_logits' in predictions:
            key_id_mask = valid_mask & (event_target == 1)  # KEY events
            if key_id_mask.any():
                key_id_loss = self._compute_classification_loss(
                    predictions['key_id_logits'], targets[..., 5], key_id_mask, 'key_id'
                )
                aux_losses['key_id'] = key_id_loss
            else:
                aux_losses['key_id'] = torch.tensor(0.0, device=valid_mask.device)
        
        # Scroll loss (only for SCROLL events)
        if 'scroll_y_logits' in predictions:
            scroll_mask = valid_mask & (event_target == 2)  # SCROLL events
            if scroll_mask.any():
                scroll_loss = self._compute_classification_loss(
                    predictions['scroll_y_logits'], targets[..., 6], scroll_mask, 'scroll'
                )
                aux_losses['scroll'] = scroll_loss
            else:
                aux_losses['scroll'] = torch.tensor(0.0, device=valid_mask.device)
        
        return aux_losses
    
    def _compute_classification_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                                   mask: torch.Tensor, class_name: str) -> torch.Tensor:
        """Compute classification loss for auxiliary outputs."""
        # Get class weights if available
        weights = None
        if class_name in self.class_weights:
            weights = self.class_weights[class_name].to(logits.device)
        
        # Compute cross-entropy loss
        num_classes = logits.shape[-1]
        targets_clamped = targets.long().clamp(0, num_classes - 1)
        
        # Apply mask and compute loss
        mask_flat = mask.view(-1)
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets_clamped.view(-1)
        
        if mask_flat.any():
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], weight=weights)
        else:
            loss = torch.tensor(0.0, device=logits.device)
        
        return loss


# Keep the existing CrossEntropyLoss for backward compatibility
CrossEntropyLoss = nn.CrossEntropyLoss
