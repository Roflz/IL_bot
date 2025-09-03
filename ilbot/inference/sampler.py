"""
Inference sampler for imitation learning bot.
Provides greedy and sampling-based decoding with strict contracts.
"""

import json
import random
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ilbot.data.contracts import derive_event_targets_from_marks


def _load_norm_bounds(data_root: Path) -> Tuple[float, float]:
    """Load normalization bounds from norm.json with strict validation."""
    norm_path = data_root / "data_profile" / "norm.json"
    
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")
    
    try:
        with open(norm_path, 'r') as f:
            norm = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise RuntimeError(f"Failed to read norm.json: {e}")
    
    required_keys = {"x_max", "y_max"}
    if not required_keys.issubset(norm.keys()):
        missing = required_keys - set(norm.keys())
        raise ValueError(f"norm.json missing required keys: {missing}")
    
    x_max = float(norm["x_max"])
    y_max = float(norm["y_max"])
    
    if x_max <= 0 or y_max <= 0:
        raise ValueError(f"Invalid normalization bounds: x_max={x_max}, y_max={y_max}")
    
    return x_max, y_max


def denorm_xy(x_norm: torch.Tensor, y_norm: torch.Tensor, x_max: float, y_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denormalize XY coordinates from [0,1] to pixel coordinates.
    
    Args:
        x_norm, y_norm: Normalized coordinates in [0,1]
        x_max, y_max: Maximum pixel bounds from norm.json
    
    Returns:
        x_px, y_px: Pixel coordinates
    """
    # Strict validation: assert inputs are in [0,1] ± tolerance
    tolerance = 1e-6
    if not ((x_norm >= -tolerance).all() and (x_norm <= 1.0 + tolerance).all()):
        bad_x = torch.logical_or(x_norm < -tolerance, x_norm > 1.0 + tolerance)
        bad_count = int(bad_x.sum().item())
        raise ValueError(f"x_norm contains {bad_count} values outside [0,1]±{tolerance}")
    
    if not ((y_norm >= -tolerance).all() and (y_norm <= 1.0 + tolerance).all()):
        bad_y = torch.logical_or(y_norm < -tolerance, y_norm > 1.0 + tolerance)
        bad_count = int(bad_y.sum().item())
        raise ValueError(f"y_norm contains {bad_count} values outside [0,1]±{tolerance}")
    
    # Clamp to valid range and denormalize
    x_norm_clamped = torch.clamp(x_norm, 0.0, 1.0)
    y_norm_clamped = torch.clamp(y_norm, 0.0, 1.0)
    
    x_px = x_norm_clamped * x_max
    y_px = y_norm_clamped * y_max
    
    return x_px, y_px


def _set_deterministic_seed(seed: int):
    """Set deterministic seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def greedy_decode(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    max_actions: int,
    horizon_s: float,
    device: torch.device,
    seed: Optional[int] = None,
    data_root: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    Greedy decoding for action sequences.
    
    Args:
        model: Trained SequentialImitationModel
        batch: Input batch with keys: temporal_sequence, action_sequence, valid_mask
        max_actions: Maximum number of actions to generate
        horizon_s: Time horizon in seconds
        device: Device to run inference on
        seed: Optional seed for deterministic behavior
        data_root: Path to data root for norm.json (auto-detected if None)
    
    Returns:
        Dictionary with keys: event_logits, time_delta_q, x_mu, y_mu, x_logsigma, y_logsigma,
        event_id, time_s, x_norm, y_norm, x_px, y_px
    """
    if seed is not None:
        _set_deterministic_seed(seed)
    
    # Auto-detect data_root if not provided
    if data_root is None:
        # Assume we're in the repo root, find data directory
        current = Path.cwd()
        if current.name == "bot_runelite_IL":
            data_root = current / "data"
        else:
            # Try to find data directory by going up
            for parent in current.parents:
                if (parent / "data").exists():
                    data_root = parent / "data"
                    break
            else:
                raise RuntimeError("Could not auto-detect data root. Please provide data_root explicitly.")
    
    # Load normalization bounds
    x_max, y_max = _load_norm_bounds(data_root)
    
    # Move batch to device
    temporal_sequence = batch["temporal_sequence"].to(device)
    action_sequence = batch["action_sequence"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    
    B, T, A, Fa = action_sequence.shape
    
    # Forward pass through model
    model.eval()
    with torch.no_grad():
        outputs = model(temporal_sequence, action_sequence, valid_mask)
    
    # Validate model outputs are finite
    for head_name, tensor in outputs.items():
        if not torch.isfinite(tensor).all():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise RuntimeError(f"Non-finite values in model output '{head_name}': {bad_count} out of {tensor.numel()}")
    
    # Extract predictions
    event_logits = outputs["event_logits"]  # [B, A, 4]
    time_delta_q = outputs["time_delta_q"]  # [B, A, 3]
    x_mu = outputs["x_mu"]  # [B, A, 1]
    y_mu = outputs["y_mu"]  # [B, A, 1]
    x_logsigma = outputs["x_logsigma"]  # [B, A, 1]
    y_logsigma = outputs["y_logsigma"]  # [B, A, 1]
    
    # Greedy event selection
    event_id = torch.argmax(event_logits, dim=-1)  # [B, A]
    
    # Use median quantile for timing (index 1 of 3 quantiles)
    time_s = time_delta_q[:, :, 1]  # [B, A]
    
    # Extract normalized coordinates
    x_norm = x_mu.squeeze(-1)  # [B, A]
    y_norm = y_mu.squeeze(-1)  # [B, A]
    
    # Denormalize to pixel coordinates
    x_px, y_px = denorm_xy(x_norm, y_norm, x_max, y_max)
    
    # Validate all outputs are finite
    result = {
        "event_logits": event_logits,
        "time_delta_q": time_delta_q,
        "x_mu": x_mu,
        "y_mu": y_mu,
        "x_logsigma": x_logsigma,
        "y_logsigma": y_logsigma,
        "event_id": event_id,
        "time_s": time_s,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x_px": x_px,
        "y_px": y_px,
    }
    
    for key, tensor in result.items():
        if not torch.isfinite(tensor).all():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise RuntimeError(f"Non-finite values in result '{key}': {bad_count} out of {tensor.numel()}")
    
    return result


def sample_decode(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    max_actions: int,
    horizon_s: float,
    device: torch.device,
    temperature: float = 1.0,
    topk: Optional[int] = None,
    gumbel: bool = False,
    seed: Optional[int] = None,
    data_root: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    Sampling-based decoding for action sequences.
    
    Args:
        model: Trained SequentialImitationModel
        batch: Input batch with keys: temporal_sequence, action_sequence, valid_mask
        max_actions: Maximum number of actions to generate
        horizon_s: Time horizon in seconds
        device: Device to run inference on
        temperature: Sampling temperature (>0)
        topk: Optional top-k sampling (None for no limit)
        gumbel: Use Gumbel-Max sampling for events
        seed: Optional seed for deterministic behavior
        data_root: Path to data root for norm.json (auto-detected if None)
    
    Returns:
        Dictionary with same keys as greedy_decode
    """
    if seed is not None:
        _set_deterministic_seed(seed)
    
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Auto-detect data_root if not provided
    if data_root is None:
        # Assume we're in the repo root, find data directory
        current = Path.cwd()
        if current.name == "bot_runelite_IL":
            data_root = current / "data"
        else:
            # Try to find data directory by going up
            for parent in current.parents:
                if (parent / "data").exists():
                    data_root = parent / "data"
                    break
            else:
                raise RuntimeError("Could not auto-detect data root. Please provide data_root explicitly.")
    
    # Load normalization bounds
    x_max, y_max = _load_norm_bounds(data_root)
    
    # Move batch to device
    temporal_sequence = batch["temporal_sequence"].to(device)
    action_sequence = batch["action_sequence"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    
    # Forward pass through model
    model.eval()
    with torch.no_grad():
        outputs = model(temporal_sequence, action_sequence, valid_mask)
    
    # Validate model outputs are finite
    for head_name, tensor in outputs.items():
        if not torch.isfinite(tensor).all():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise RuntimeError(f"Non-finite values in model output '{head_name}': {bad_count} out of {tensor.numel()}")
    
    # Extract predictions
    event_logits = outputs["event_logits"]  # [B, A, 4]
    time_delta_q = outputs["time_delta_q"]  # [B, A, 3]
    x_mu = outputs["x_mu"]  # [B, A, 1]
    y_mu = outputs["y_mu"]  # [B, A, 1]
    x_logsigma = outputs["x_logsigma"]  # [B, A, 1]
    y_logsigma = outputs["y_logsigma"]  # [B, A, 1]
    
    # Sample events
    if gumbel:
        # Gumbel-Max sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(event_logits) + 1e-8) + 1e-8)
        event_id = torch.argmax(event_logits + gumbel_noise, dim=-1)
    else:
        # Temperature-scaled sampling
        scaled_logits = event_logits / temperature
        if topk is not None:
            # Top-k sampling
            topk_logits, topk_indices = torch.topk(scaled_logits, min(topk, scaled_logits.size(-1)), dim=-1)
            topk_probs = F.softmax(topk_logits, dim=-1)
            sampled_indices = torch.multinomial(topk_probs.view(-1, topk_probs.size(-1)), 1).view_as(event_id)
            event_id = torch.gather(topk_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        else:
            # Full sampling
            probs = F.softmax(scaled_logits, dim=-1)
            event_id = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(event_id)
    
    # Sample timing from quantile distribution
    # For now, use median quantile (could be extended to sample from full distribution)
    time_s = time_delta_q[:, :, 1]  # [B, A]
    
    # Sample XY coordinates from Gaussian distribution
    x_std = torch.exp(x_logsigma.squeeze(-1))  # [B, A]
    y_std = torch.exp(y_logsigma.squeeze(-1))  # [B, A]
    x_norm = x_mu.squeeze(-1) + x_std * torch.randn_like(x_std)  # [B, A]
    y_norm = y_mu.squeeze(-1) + y_std * torch.randn_like(y_std)  # [B, A]
    
    # Denormalize to pixel coordinates
    x_px, y_px = denorm_xy(x_norm, y_norm, x_max, y_max)
    
    # Validate all outputs are finite
    result = {
        "event_logits": event_logits,
        "time_delta_q": time_delta_q,
        "x_mu": x_mu,
        "y_mu": y_mu,
        "x_logsigma": x_logsigma,
        "y_logsigma": y_logsigma,
        "event_id": event_id,
        "time_s": time_s,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x_px": x_px,
        "y_px": y_px,
    }
    
    for key, tensor in result.items():
        if not torch.isfinite(tensor).all():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise RuntimeError(f"Non-finite values in result '{key}': {bad_count} out of {tensor.numel()}")
    
    return result
