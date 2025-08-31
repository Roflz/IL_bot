#!/usr/bin/env python3
"""
Metrics and utility functions for OSRS imitation learning
"""

import torch
import numpy as np
from typing import Dict

# Add debugging imports
import logging
import torch
LOG = logging.getLogger(__name__)

def apply_masks_for_head(name, mask_dict, base_mask):
    # Check mask intersections per head
    m = base_mask.clone()
    head_m = mask_dict.get(name, None)
    if head_m is not None:
        m = m & head_m.bool()
    kept = int(m.sum().item())
    LOG.info("[DBG] mask[%s]: kept=%d / %d", name, kept, int(m.numel()))
    return m

def denorm_time(t_norm, stats):
    """Denormalize time values based on normalization mode"""
    mode = stats.get("time_norm", "z")  # "z" or "minmax" or "none"
    if mode == "z":
        return t_norm * stats["time_std"] + stats["time_mean"]
    if mode == "minmax":
        return t_norm * (stats["time_max"] - stats["time_min"]) + stats["time_min"]
    return t_norm

def build_masks(batch):
    """Build per-head masks based on event types"""
    # This function assumes you have event_idx in your batch
    # You may need to adapt this based on your actual data structure
    if "event_idx" not in batch:
        # Fallback: create a default mask that includes all rows
        return {
            "btn": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
            "ka": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
            "kid": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
            "sy": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
            "xy": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
            "time": torch.ones(batch["action_target"].shape[0], dtype=torch.bool),
        }
    
    evt = batch["event_idx"]
    MOVE, CLICK, KEY, SCROLL = 0, 1, 2, 3
    masks = {
        "btn":   (evt == CLICK),
        "ka":    (evt == KEY),
        "kid":   (evt == KEY),
        "sy":    (evt == SCROLL),
        "xy":    torch.ones_like(evt, dtype=torch.bool),   # or evt==MOVE if you only want MOVE
        "time":  torch.ones_like(evt, dtype=torch.bool),   # time usually defined for all rows
    }
    return masks


def clamp_nonneg(x):
    """Clamp tensor to non-negative values"""
    return x.clamp(min=0.0)


def topk_counts(arr, k=5):
    """
    arr: 1D int tensor/array of class ids (e.g., predictions or targets).
    returns: list of (class_id, count) sorted desc by count, up to k entries.
    """
    import numpy as np
    a = np.asarray(arr)
    uniques, counts = np.unique(a, return_counts=True)
    pairs = list(zip(uniques.tolist(), counts.tolist()))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs[:k]

def compute_event_metrics(outputs, targets):
    logits = outputs.get("event_logits")
    if logits is None:
        LOG.error("[DBG] event_logits missing; refusing to use scroll_y_logits fallback")
        return {}
    tgt = targets.get("event")
    if tgt is None:
        LOG.error("[DBG] targets.event missing")
        return {}
    
    pred = logits.argmax(dim=-1)
    LOG.info("[DBG] EVENT: pred counts=%s", torch.bincount(pred.view(-1).cpu(), minlength=logits.size(-1)).tolist())
    LOG.info("[DBG] EVENT: tgt  counts=%s", torch.bincount(tgt.view(-1).cpu(), minlength=logits.size(-1)).tolist())
    
    # Compute accuracy
    correct = (pred == tgt).float().mean()
    return {"event_accuracy": correct.item()}
