# ilbot/training/metrics.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def running_mean(value: float, state: float, alpha: float = 0.1) -> float:
    """Exponential moving average update."""
    return alpha * value + (1 - alpha) * state

def ema(value: float, state: float, alpha: float = 0.1) -> float:
    """Alias for running_mean for consistency."""
    return running_mean(value, state, alpha)

def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MAE with masking.
    pred, target: [B,A] or [B,A,K]; mask: [B,A] bool
    """
    assert pred.shape == target.shape
    assert mask.shape == pred.shape[:2]  # [B,A]
    
    diff = torch.abs(pred - target)
    if diff.dim() == 3:  # [B,A,K] - take mean over K
        diff = diff.mean(dim=-1)
    
    diff = diff.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom

def event_top1_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute top-1 accuracy for event classification.
    logits: [B,A,4] (CLICK,KEY,SCROLL,MOVE)
    target: [B,A] (0,1,2,3)
    mask: [B,A] bool
    """
    assert logits.shape[:2] == target.shape == mask.shape
    assert logits.shape[2] == 4, f"Expected 4 event classes, got {logits.shape[2]}"
    
    pred = logits.argmax(dim=-1)  # [B,A]
    correct = (pred == target) & mask
    denom = mask.sum().clamp_min(1)
    return correct.sum().float() / denom

def length_stats(seq_len_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for sequence lengths.
    seq_len_tensor: [B] or [B,A] - sequence lengths
    """
    if seq_len_tensor.dim() == 2:
        seq_len_tensor = seq_len_tensor.view(-1)
    
    return {
        "mean": float(seq_len_tensor.float().mean()),
        "std": float(seq_len_tensor.float().std()),
        "min": float(seq_len_tensor.min()),
        "max": float(seq_len_tensor.max()),
    }
