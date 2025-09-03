"""
Evaluation metrics for imitation learning bot.
Provides tensor utilities with strict validation and masking support.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def per_class_prf1(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict:
    """
    Compute per-class precision, recall, F1 with macro and micro averages.
    
    Args:
        logits: [B, A, C] or [N, C] prediction logits
        targets: [B, A] or [N] target class indices
        mask: [B, A] or [N] boolean mask for valid predictions
    
    Returns:
        Dictionary with per_class, macro, and micro metrics
    """
    # Validate inputs
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains non-finite values")
    if not torch.isfinite(targets.float()).all():
        raise ValueError("targets contains non-finite values")
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be boolean tensor")
    
    # Flatten if needed
    if logits.dim() == 3:
        B, A, C = logits.shape
        logits = logits.view(-1, C)
        targets = targets.view(-1)
        mask = mask.view(-1)
    elif logits.dim() == 2:
        N, C = logits.shape
        if targets.dim() != 1 or mask.dim() != 1:
            raise ValueError("targets and mask must be 1D when logits is 2D")
    else:
        raise ValueError("logits must be 2D [N,C] or 3D [B,A,C]")
    
    if targets.size(0) != logits.size(0) or mask.size(0) != logits.size(0):
        raise ValueError("logits, targets, and mask must have same first dimension")
    
    # Apply mask
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    
    if valid_targets.numel() == 0:
        raise ValueError("No valid predictions after masking")
    
    num_classes = logits.size(-1)
    
    # Get predictions
    pred_classes = torch.argmax(valid_logits, dim=-1)
    
    # Compute confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=logits.device)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((pred_classes == i) & (valid_targets == j)).sum()
    
    # Per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for cls in range(num_classes):
        tp = cm[cls, cls].float()
        fp = cm[cls, :].sum().float() - tp
        fn = cm[:, cls].sum().float() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=cm.device)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=cm.device)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0, device=cm.device)
        
        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Macro averages
    macro_precision = torch.stack(precisions).mean().to(logits.device)
    macro_recall = torch.stack(recalls).mean().to(logits.device)
    macro_f1 = torch.stack(f1s).mean().to(logits.device)
    
    # Micro averages (overall accuracy)
    micro_accuracy = cm.diag().sum().float() / cm.sum().float()
    
    # Validate results are finite and in [0,1]
    for metric in [macro_precision, macro_recall, macro_f1, micro_accuracy]:
        if not torch.isfinite(metric):
            raise RuntimeError(f"Non-finite metric computed: {metric}")
        if not (0.0 <= metric <= 1.0):
            raise RuntimeError(f"Metric out of [0,1] range: {metric}")
    
    return {
        "per_class": per_class,
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1)
        },
        "micro": {
            "accuracy": float(micro_accuracy)
        }
    }


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """
    Compute confusion matrix for classification.
    
    Args:
        logits: [B, A, C] or [N, C] prediction logits
        targets: [B, A] or [N] target class indices
        mask: [B, A] or [N] boolean mask for valid predictions
        num_classes: Number of classes
    
    Returns:
        [num_classes, num_classes] confusion matrix
    """
    # Validate inputs
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains non-finite values")
    if not torch.isfinite(targets.float()).all():
        raise ValueError("targets contains non-finite values")
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be boolean tensor")
    
    # Flatten if needed
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        mask = mask.view(-1)
    
    if targets.size(0) != logits.size(0) or mask.size(0) != logits.size(0):
        raise ValueError("logits, targets, and mask must have same first dimension")
    
    # Apply mask
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    
    if valid_targets.numel() == 0:
        raise ValueError("No valid predictions after masking")
    
    # Get predictions
    pred_classes = torch.argmax(valid_logits, dim=-1)
    
    # Build confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=logits.device)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((pred_classes == i) & (valid_targets == j)).sum()
    
    # Validate matrix
    if cm.sum() != valid_targets.numel():
        raise RuntimeError("Confusion matrix sum doesn't match valid count")
    if (cm < 0).any():
        raise RuntimeError("Confusion matrix contains negative values")
    
    return cm


def timing_mae_median(time_q50: torch.Tensor, target_s: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute MAE for timing predictions using median quantile.
    
    Args:
        time_q50: [B, A] predicted median timing in seconds
        target_s: [B, A] target timing in seconds
        mask: [B, A] boolean mask for valid predictions
    
    Returns:
        Mean absolute error in seconds
    """
    # Validate inputs
    if not torch.isfinite(time_q50).all():
        raise ValueError("time_q50 contains non-finite values")
    if not torch.isfinite(target_s).all():
        raise ValueError("target_s contains non-finite values")
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be boolean tensor")
    
    if time_q50.shape != target_s.shape or time_q50.shape != mask.shape:
        raise ValueError("time_q50, target_s, and mask must have same shape")
    
    # Apply mask
    valid_time_q50 = time_q50[mask]
    valid_target_s = target_s[mask]
    
    if valid_time_q50.numel() == 0:
        raise ValueError("No valid timing predictions after masking")
    
    # Compute MAE
    mae = torch.abs(valid_time_q50 - valid_target_s).mean()
    
    if not torch.isfinite(mae):
        raise RuntimeError("Non-finite MAE computed")
    
    return float(mae)


def xy_mae(x_mu: torch.Tensor, y_mu: torch.Tensor, x_t: torch.Tensor, y_t: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    """
    Compute MAE for XY coordinate predictions.
    
    Args:
        x_mu, y_mu: [B, A] predicted coordinates (normalized or pixel)
        x_t, y_t: [B, A] target coordinates (same units as predictions)
        mask: [B, A] boolean mask for valid predictions
    
    Returns:
        (x_mae, y_mae) in same units as inputs
    """
    # Validate inputs
    for name, tensor in [("x_mu", x_mu), ("y_mu", y_mu), ("x_t", x_t), ("y_t", y_t)]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be boolean tensor")
    
    # Check shapes
    expected_shape = x_mu.shape
    for name, tensor in [("y_mu", y_mu), ("x_t", x_t), ("y_t", y_t), ("mask", mask)]:
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} doesn't match x_mu shape {expected_shape}")
    
    # Apply mask
    valid_x_mu = x_mu[mask]
    valid_y_mu = y_mu[mask]
    valid_x_t = x_t[mask]
    valid_y_t = y_t[mask]
    
    if valid_x_mu.numel() == 0:
        raise ValueError("No valid XY predictions after masking")
    
    # Compute MAE
    x_mae = torch.abs(valid_x_mu - valid_x_t).mean()
    y_mae = torch.abs(valid_y_mu - valid_y_t).mean()
    
    for name, mae in [("x_mae", x_mae), ("y_mae", y_mae)]:
        if not torch.isfinite(mae):
            raise RuntimeError(f"Non-finite {name} computed")
    
    return float(x_mae), float(y_mae)
