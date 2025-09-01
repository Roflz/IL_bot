#!/usr/bin/env python3
"""
Advanced Loss Functions for OSRS Bot Training
Addresses event prediction issues with better class balancing and uncertainty handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in event classification
    Reduces the relative loss for well-classified examples and focuses on hard examples
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization in event classification
    Prevents the model from being overconfident in its predictions
    """
    
    def __init__(self, classes: int, smoothing: float = 0.1, dim: int = -1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-Aware Loss that considers prediction confidence
    Encourages the model to be uncertain when it should be uncertain
    """
    
    def __init__(self, base_loss: nn.Module, uncertainty_weight: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Base loss
        base_loss = self.base_loss(inputs, targets)
        
        # Uncertainty penalty - encourage appropriate uncertainty
        probs = F.softmax(inputs, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        
        # Penalize overconfidence when wrong, encourage confidence when right
        uncertainty_penalty = torch.mean(
            torch.where(
                targets == probs.argmax(dim=-1),
                (1 - max_probs) ** 2,  # Encourage confidence when correct
                max_probs ** 2          # Penalize confidence when wrong
            )
        )
        
        return base_loss + self.uncertainty_weight * uncertainty_penalty

class AdvancedUnifiedEventLoss(nn.Module):
    """
    Advanced Unified Event Loss with multiple improvements:
    1. Focal Loss for event classification
    2. Label smoothing for better generalization
    3. Uncertainty-aware coordinate prediction
    4. Temporal consistency regularization
    5. Action sequence coherence
    """
    
    def __init__(self, 
                 data_config: Dict,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 uncertainty_weight: float = 0.1,
                 temporal_weight: float = 0.05,
                 coherence_weight: float = 0.03):
        super().__init__()
        
        self.data_config = data_config
        self.enum_sizes = data_config.get('enum_sizes', {})
        self.event_types = data_config.get('event_types', 4)
        
        # Advanced loss components
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing_loss = LabelSmoothingLoss(classes=self.event_types, smoothing=label_smoothing)
        
        # Weights for different loss components
        self.uncertainty_weight = uncertainty_weight
        self.temporal_weight = temporal_weight
        self.coherence_weight = coherence_weight
        
        # Class weights for event classification
        self.register_buffer('event_class_weights', None)
        self.register_buffer('target_distribution', None)
        
        # Loss component tracking
        self.loss_components = {}
        
        # Debug flag
        self._debug_printed = False
        self._debug_derive_printed = False
    
    def set_event_class_weights(self, weights: torch.Tensor):
        """Set class weights for event classification"""
        self.register_buffer('event_class_weights', weights)
    
    def set_target_distribution(self, distribution: torch.Tensor):
        """Set target distribution for regularization"""
        self.register_buffer('target_distribution', distribution)
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute advanced unified event loss
        
        Args:
            predictions: Model output dictionary
            targets: Target tensor [B, A, 7]
            valid_mask: Valid action mask [B, A]
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        losses = {}
        
        # 1. Advanced Event Classification Loss
        event_loss = self._compute_advanced_event_loss(
            predictions['event_logits'], targets, valid_mask
        )
        losses['event'] = event_loss
        
        # 2. Uncertainty-Aware Coordinate Loss
        coord_loss = self._compute_uncertainty_aware_coordinate_loss(
            predictions, targets, valid_mask
        )
        losses['coordinates'] = coord_loss
        
        # 3. Temporal Consistency Loss
        temporal_loss = self._compute_temporal_consistency_loss(
            predictions, targets, valid_mask
        )
        losses['temporal'] = temporal_loss
        
        # 4. Action Sequence Coherence Loss
        coherence_loss = self._compute_action_coherence_loss(
            predictions, targets, valid_mask
        )
        losses['coherence'] = coherence_loss
        
        # 5. Distribution Regularization
        if self.target_distribution is not None:
            dist_reg_loss = self._compute_distribution_regularization(
                predictions['event_logits'], valid_mask
            )
            losses['distribution_reg'] = dist_reg_loss
        
        # 6. Uncertainty Regularization
        uncertainty_reg_loss = self._compute_uncertainty_regularization(
            predictions, valid_mask
        )
        losses['uncertainty_reg'] = uncertainty_reg_loss
        
        # Combine all losses
        total_loss = sum(losses.values())
        
        # Store loss components for monitoring
        self.loss_components = losses
        
        return total_loss, losses
    
    def _compute_advanced_event_loss(self,
                                   event_logits: torch.Tensor,
                                   targets: torch.Tensor,
                                   valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute advanced event classification loss using focal loss and label smoothing
        """
        # Derive event targets from action targets
        event_targets = self._derive_event_target(targets)
        
        # Debug: Print event target distribution
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"🔧 Debug - Event targets shape: {event_targets.shape}")
            print(f"🔧 Debug - Event targets unique: {event_targets.unique().tolist()}")
            print(f"🔧 Debug - Event targets distribution: {torch.bincount(event_targets.flatten(), minlength=self.event_types).tolist()}")
            self._debug_printed = True
        
        # Apply valid mask
        valid_event_logits = event_logits[valid_mask]
        valid_event_targets = event_targets[valid_mask]
        
        if valid_event_logits.numel() == 0:
            return torch.tensor(0.0, device=event_logits.device)
        
        # Combine focal loss and label smoothing
        focal_loss = self.focal_loss(valid_event_logits, valid_event_targets)
        smooth_loss = self.label_smoothing_loss(valid_event_logits, valid_event_targets)
        
        # Weighted combination
        event_loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        # Apply class weights if available
        if self.event_class_weights is not None:
            # Ensure class weights are on the same device as targets
            class_weights = self.event_class_weights.to(valid_event_targets.device)[valid_event_targets]
            event_loss = (event_loss * class_weights).mean()
        
        return event_loss
    
    def _compute_uncertainty_aware_coordinate_loss(self,
                                                 predictions: Dict[str, torch.Tensor],
                                                 targets: torch.Tensor,
                                                 valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware coordinate prediction loss
        """
        # Extract coordinate predictions and targets
        x_mu = predictions['x_mu']  # [B, A]
        y_mu = predictions['y_mu']  # [B, A]
        x_logsig = predictions['x_logsig']  # [B, A]
        y_logsig = predictions['y_logsig']  # [B, A]
        
        x_target = targets[..., 1]  # [B, A]
        y_target = targets[..., 2]  # [B, A]
        
        # Apply valid mask
        valid_x_mu = x_mu[valid_mask]
        valid_y_mu = y_mu[valid_mask]
        valid_x_logsig = x_logsig[valid_mask]
        valid_y_logsig = y_logsig[valid_mask]
        valid_x_target = x_target[valid_mask]
        valid_y_target = y_target[valid_mask]
        
        if valid_x_mu.numel() == 0:
            return torch.tensor(0.0, device=x_mu.device)
        
        # Gaussian NLL Loss for coordinates
        x_loss = F.gaussian_nll_loss(valid_x_mu, valid_x_target, torch.exp(valid_x_logsig))
        y_loss = F.gaussian_nll_loss(valid_y_mu, valid_y_target, torch.exp(valid_y_logsig))
        
        # Uncertainty penalty - encourage appropriate uncertainty
        x_uncertainty = torch.exp(valid_x_logsig)
        y_uncertainty = torch.exp(valid_y_logsig)
        
        # Penalize excessive uncertainty
        uncertainty_penalty = torch.mean(x_uncertainty + y_uncertainty)
        
        # Combine coordinate loss with uncertainty penalty
        coord_loss = x_loss + y_loss + self.uncertainty_weight * uncertainty_penalty
        
        return coord_loss
    
    def _compute_temporal_consistency_loss(self,
                                         predictions: Dict[str, torch.Tensor],
                                         targets: torch.Tensor,
                                         valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss to ensure smooth action sequences
        """
        if 'time_q' not in predictions:
            return torch.tensor(0.0, device=targets.device)
        
        time_predictions = predictions['time_q']  # [B, 3] - quantiles
        
        # Temporal consistency: consecutive actions should have reasonable timing
        B = time_predictions.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=time_predictions.device)
        
        # Get median timing predictions
        median_times = time_predictions[:, 1]  # [B] - q0.5
        
        # Compute timing differences between consecutive batches
        time_diffs = torch.diff(median_times)
        
        # Penalize unrealistic timing jumps
        # Actions shouldn't have huge time gaps unless there's a natural break
        timing_penalty = torch.mean(torch.clamp(time_diffs - 5.0, min=0.0) ** 2)
        
        return self.temporal_weight * timing_penalty
    
    def _compute_action_coherence_loss(self,
                                     predictions: Dict[str, torch.Tensor],
                                     targets: torch.Tensor,
                                     valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute action coherence loss to ensure logical action sequences
        """
        if 'event_logits' not in predictions:
            return torch.tensor(0.0, device=targets.device)
        
        event_probs = F.softmax(predictions['event_logits'], dim=-1)  # [B, A, 4]
        
        # Action coherence: consecutive actions should make sense together
        B, A = event_probs.shape[:2]
        if B < 2 or A < 2:
            return torch.tensor(0.0, device=event_probs.device)
        
        # Get most likely events for each action
        predicted_events = event_probs.argmax(dim=-1)  # [B, A]
        
        # Analyze event transitions within each batch
        coherence_penalty = 0.0
        
        for b in range(B):
            batch_events = predicted_events[b]  # [A]
            valid_actions = valid_mask[b]  # [A]
            
            if valid_actions.sum() < 2:
                continue
            
            # Get valid event sequence
            valid_events = batch_events[valid_actions]
            
            if len(valid_events) < 2:
                continue
            
            # Penalize unrealistic event transitions
            for i in range(1, len(valid_events)):
                prev_event = valid_events[i-1]
                curr_event = valid_events[i]
                
                # Define unrealistic transitions
                unrealistic_transitions = [
                    (0, 0),  # CLICK -> CLICK (double click)
                    (1, 1),  # KEY -> KEY (double key press)
                    (2, 2),  # SCROLL -> SCROLL (double scroll)
                ]
                
                if (prev_event.item(), curr_event.item()) in unrealistic_transitions:
                    coherence_penalty += 1.0
        
        # Normalize by total valid actions
        total_valid = valid_mask.sum()
        if total_valid > 0:
            coherence_penalty = coherence_penalty / total_valid
        
        return self.coherence_weight * coherence_penalty
    
    def _compute_distribution_regularization(self,
                                           event_logits: torch.Tensor,
                                           valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution regularization to maintain target event distribution
        """
        if self.target_distribution is None:
            return torch.tensor(0.0, device=event_logits.device)
        
        # Get predicted probabilities
        event_probs = F.softmax(event_logits, dim=-1)  # [B, A, 4]
        
        # Average probabilities across valid actions
        if valid_mask.any():
            valid_probs = event_probs[valid_mask]
            avg_probs = valid_probs.mean(dim=0)  # [4]
        else:
            avg_probs = event_probs.mean(dim=(0, 1))  # [4]
        
        # Target distribution
        target_dist = self.target_distribution.to(event_probs.device)
        
        # KL divergence between predicted and target distribution
        eps = 1e-8
        avg_probs = avg_probs.clamp(min=eps)
        target_dist = target_dist.clamp(min=eps)
        
        kl_loss = (target_dist * torch.log(target_dist / avg_probs)).sum()
        
        return kl_loss
    
    def _compute_uncertainty_regularization(self,
                                          predictions: Dict[str, torch.Tensor],
                                          valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty regularization to prevent overconfidence
        """
        uncertainty_penalty = 0.0
        
        # Coordinate uncertainty regularization
        if 'x_logsig' in predictions and 'y_logsig' in predictions:
            x_uncertainty = torch.exp(predictions['x_logsig'])
            y_uncertainty = torch.exp(predictions['y_logsig'])
            
            # Penalize infinite or extremely large uncertainty
            x_inf_penalty = torch.isinf(x_uncertainty).float().sum()
            y_inf_penalty = torch.isinf(y_uncertainty).float().sum()
            
            # Penalize extremely small uncertainty (overconfidence)
            x_overconf_penalty = torch.mean(torch.clamp(1.0 - x_uncertainty, min=0.0))
            y_overconf_penalty = torch.mean(torch.clamp(1.0 - y_uncertainty, min=0.0))
            
            uncertainty_penalty += x_inf_penalty + y_inf_penalty + x_overconf_penalty + y_overconf_penalty
        
        # Event uncertainty regularization
        if 'event_logits' in predictions:
            event_probs = F.softmax(predictions['event_logits'], dim=-1)
            max_probs = event_probs.max(dim=-1)[0]
            
            # Penalize overconfidence in event predictions
            overconfidence_penalty = torch.mean(torch.clamp(max_probs - 0.9, min=0.0) ** 2)
            uncertainty_penalty += overconfidence_penalty
        
        return uncertainty_penalty
    
    def _derive_event_target(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Derive event type from action targets with proper priority
        """
        B, A = targets.shape[:2]
        event_target = torch.full((B, A), 3, dtype=torch.long, device=targets.device)  # Default to MOVE
        
        # Extract action components
        button_target = targets[..., 3]  # Button column
        key_action_target = targets[..., 4]  # Key action column
        scroll_y_target = targets[..., 6]  # Scroll column
        
        # Debug: Print actual values to see what we're getting
        if hasattr(self, '_debug_derive_printed') and not self._debug_derive_printed:
            print(f"🔧 Debug - Button targets unique: {button_target.unique().tolist()}")
            print(f"🔧 Debug - Key action targets unique: {key_action_target.unique().tolist()}")
            print(f"🔧 Debug - Scroll targets unique: {scroll_y_target.unique().tolist()}")
            print(f"🔧 Debug - Sample button values: {button_target[:5, :5].tolist()}")
            self._debug_derive_printed = True
        
        # Priority order: CLICK > KEY > SCROLL > MOVE
        # SCROLL: scroll_y != 0 (lowest priority)
        scroll_mask = scroll_y_target != 0
        event_target[scroll_mask] = 2
        
        # KEY: key_action != 0 (medium priority - overwrites SCROLL)
        key_mask = key_action_target != 0
        event_target[key_mask] = 1
        
        # CLICK: button != 0 (highest priority - overwrites KEY and SCROLL)
        click_mask = button_target != 0
        event_target[click_mask] = 0
        
        return event_target
    
    def set_global_event_weights(self, event_targets: List[torch.Tensor], valid_masks: List[torch.Tensor]):
        """
        Set global event class weights based on dataset distribution
        Compatible with UnifiedEventLoss interface
        """
        # Flatten all event targets and valid masks
        all_events = []
        for events, mask in zip(event_targets, valid_masks):
            valid_events = events[mask]
            all_events.append(valid_events)
        
        if not all_events:
            return
        
        # Concatenate all valid events
        all_events = torch.cat(all_events, dim=0)
        
        # Count occurrences of each event type
        event_counts = torch.bincount(all_events, minlength=self.event_types)
        
        # Compute inverse frequency weights
        total_events = event_counts.sum()
        class_weights = total_events / (self.event_types * event_counts + 1e-8)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum()
        
        # Set the weights
        self.set_event_class_weights(class_weights)
        
        print(f"🎯 Set global event class weights: {class_weights.tolist()}")
    
    def reset_epoch_flag(self):
        """
        Reset epoch flag for loss function
        Compatible with UnifiedEventLoss interface
        """
        # Clear any epoch-specific state
        if hasattr(self, 'loss_components'):
            self.loss_components = {}
    
    def get_loss_breakdown(self) -> Dict[str, float]:
        """Get breakdown of loss components for monitoring"""
        return {k: v.item() if hasattr(v, 'item') else float(v) 
                for k, v in self.loss_components.items()}

class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that automatically adjusts weights based on training progress
    """
    
    def __init__(self, base_loss: nn.Module, adaptation_rate: float = 0.01):
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        
        # Track loss history for each component
        self.loss_history = {}
        self.weight_history = {}
        
        # Initialize adaptive weights
        self.adaptive_weights = nn.Parameter(torch.ones(1))
    
    def forward(self, *args, **kwargs):
        """Forward pass with adaptive weighting"""
        loss, components = self.base_loss(*args, **kwargs)
        
        # Update loss history
        for name, value in components.items():
            if name not in self.loss_history:
                self.loss_history[name] = []
            self.loss_history[name].append(value.item())
        
        # Adapt weights based on loss trends
        self._adapt_weights()
        
        # Apply adaptive weighting
        weighted_loss = loss * self.adaptive_weights
        
        return weighted_loss, components
    
    def _adapt_weights(self):
        """Adapt weights based on loss trends"""
        if len(self.loss_history) < 2:
            return
        
        # Calculate loss trends (simple moving average)
        trends = {}
        for name, history in self.loss_history.items():
            if len(history) >= 10:
                recent = history[-10:]
                trend = (recent[-1] - recent[0]) / len(recent)
                trends[name] = trend
        
        # Adjust weights based on trends
        if trends:
            # Increase weight for components with increasing loss
            # Decrease weight for components with decreasing loss
            total_trend = sum(trends.values())
            if total_trend != 0:
                adjustment = self.adaptation_rate * total_trend
                self.adaptive_weights.data += adjustment
                
                # Clamp weights to reasonable range
                self.adaptive_weights.data.clamp_(0.1, 10.0)
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Get weight adaptation history"""
        return self.weight_history
