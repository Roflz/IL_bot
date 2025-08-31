#!/usr/bin/env python3
"""
Training script for OSRS Imitation Learning Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import json
import os
from datetime import datetime
import time
import argparse
from pathlib import Path
from ilbot.training.setup import create_data_loaders, setup_model, setup_training, OSRSDataset
from ilbot.model.imitation_hybrid_model import ImitationHybridModel
from torch.optim.lr_scheduler import StepLR
import torch, os, numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from ilbot.utils.feature_spec import load_feature_spec

# Optional plotting (works headless)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False

# Enhanced loss configuration

# Add debugging imports and helpers
import logging
import torch
import math
LOG = logging.getLogger(__name__)

def _dbg_hist(name, t, mask=None, bins=10, max_items=5):
    try:
        if t is None:
            LOG.warning("[DBG] %s=None", name); return
        x = t.detach().float().cpu()
        if mask is not None:
            m = mask.detach().cpu().bool()
            if m.numel() == x.numel():
                x = x[m]
        if x.numel() == 0:
            LOG.warning("[DBG] %s: empty after mask", name); return
        vmin, vmax = float(x.min()), float(x.max())
        uniq = torch.unique(x)
        LOG.info("[DBG] %s: shape=%s min=%.4f max=%.4f uniq=%d%s",
                 name, tuple(t.shape), vmin, vmax, uniq.numel(),
                 f" e.g. {uniq[:max_items].tolist()}" if uniq.numel()<=max_items else "")
    except Exception as e:
        LOG.exception("[DBG] hist failed for %s: %s", name, e)

def _dbg_counts(name, t, mask=None, max_items=10):
    try:
        if t is None: LOG.warning("[DBG] %s=None", name); return
        x = t.detach().cpu().view(-1)
        if mask is not None:
            m = mask.detach().cpu().view(-1).bool()
            if m.shape == x.shape: x = x[m]
        vals, counts = torch.unique(x, return_counts=True)
        pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda p: -p[1])[:max_items]
        LOG.info("[DBG] %s counts: %s", name, pairs)
    except Exception as e:
        LOG.exception("[DBG] counts failed for %s: %s", name, e)


def _as_int(x):
    try:
        return int(x.item()) if hasattr(x, "item") else int(x)
    except Exception:
        return int(x)

def _percentiles_t(t: torch.Tensor):
    """Return dict of common percentiles and summary stats for a 1D tensor."""
    if t.numel() == 0:
        return {"min":0.0,"p25":0.0,"median":0.0,"p75":0.0,"max":0.0,"mean":0.0,"std":0.0,"uniq":0}
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], device=t.device))
    return {
        "min":   float(t.min().item()),
        "p25":   float(q[0].item()),
        "median":float(q[1].item()),
        "p75":   float(q[2].item()),
        "max":   float(t.max().item()),
        "mean":  float(t.mean().item()),
        "std":   float(t.std(unbiased=False).item()),
        "uniq":  int(t.unique().numel())
    }











# Add new imports for the updated run_training function
from ilbot.training.setup import OSRSDataset, create_data_loaders
from ilbot.model.imitation_hybrid_model import create_model
import random

try:
    # config flags (time_positive, debug_time, report_time_clamped_reference)
    from ilbot import config as CFG
except Exception:
    class _CFG:
        debug_time = False
        report_time_clamped_reference = False
    CFG = _CFG()

# ---- helpers ---------------------------------------------------------------
def _as_int(x):
    try:
        # Works for Python ints/floats and torch scalars
        return int(x.item()) if hasattr(x, "item") else int(x)
    except Exception:
        return int(x)

def _topk_counts_list(counts, k=10):
    # counts can be list[int] or list[torch.Tensor]; return sorted (idx, int_count)
    items = [(i, _as_int(c)) for i, c in enumerate(counts) if _as_int(c) > 0]
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:k]

# reproducibility
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# existing: train() that expects model returning dict of logits (v1)
def _allzero_mask(t: torch.Tensor) -> torch.Tensor:
    """(B,A,*) True where row is non-padding (any nonzero)."""
    return (t.abs().sum(dim=-1) > 0)

def _log1p_time_transform(t: torch.Tensor, div_ms: float = 1000.0) -> torch.Tensor:
    """Transform raw millisecond deltas: log1p(t/div_ms)."""
    return torch.log1p(t / div_ms)

def _inv_log1p_time_transform(x: torch.Tensor, div_ms: float = 1000.0) -> torch.Tensor:
    """Inverse transform back to ms."""
    return (torch.expm1(x) * div_ms)

def _masked_l1(pred, tgt, mask):
    p = pred[mask]; t = tgt[mask]
    return F.smooth_l1_loss(p, t, reduction="mean") if p.numel() else torch.zeros([], device=pred.device)

def _masked_ce(logits, tgt, mask, weight=None):
    L = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    T = tgt.reshape(-1)[mask.reshape(-1)]
    if L.numel() == 0:
        return torch.zeros([], device=logits.device)
    if weight is not None and weight.device != L.device:
        weight = weight.to(L.device)
    return F.cross_entropy(L, T, weight=weight, reduction="mean")

# (removed: legacy V1 loss)

# (removed: legacy V1 train function)

def _build_valid_mask(action_target: torch.Tensor) -> torch.Tensor:
    """
    Your rule: ONLY scroll {dx,dy} may be -1 in real rows.
    Padding rows have: type==button==key==-1  AND time==x==y==0  AND scroll_dx==scroll_dy==0.
    """
    t = action_target
    time_  = t[..., 0]; a_type = t[..., 1]; x_ = t[..., 2]; y_ = t[..., 3]
    button = t[..., 4]; key    = t[..., 5];  sx = t[..., 6]; sy = t[..., 7]
    pad_neg_non_scroll = (a_type == -1) & (button == -1) & (key == -1)
    pad_all_zero_cont  = (time_ == 0) & (x_ == 0) & (y_ == 0)
    pad_no_scroll      = (sx == 0) & (sy == 0)
    is_pad = pad_neg_non_scroll & pad_all_zero_cont & pad_no_scroll
    return ~is_pad

def _scroll_to_index(s: torch.Tensor) -> torch.Tensor:
    """Map {-1,0,1} -> {0,1,2} with clamping for strays."""
    return (s.long() + 1).clamp(0, 2)

def load_manifest(data_dir: Path):
    p = data_dir / "dataset_manifest.json"
    return json.load(open(p,"r",encoding="utf-8")) if p.exists() else None

def create_data_loaders_v2(data_dir: Path, targets_version: str = "v2", device=None):
    man = load_manifest(data_dir)
    tv = "v2"  # hard-lock to V2
    # load arrays
    gs = np.load(data_dir/"gamestate_sequences.npy")      # (N,T,G)
    ai = np.load(data_dir/"action_input_sequences.npy")   # (N,T,A,Fin)
    # V2 only
    _ = np.load(data_dir/"actions_v2.npy")                # (N,A,7)
    _ = np.load(data_dir/"valid_mask.npy")                # (N,A)
    
    # OSRSDataset expects a data_dir; let it discover file paths itself.
    dataset = OSRSDataset(data_dir=data_dir, targets_version="v2")
    # split to train/val; batch size logic unchanged…
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=32,
        disable_cuda_batch_opt=False,
        shuffle=True,
        device=device
    )
    return train_loader, val_loader, man, tv

def setup_model_v2(manifest, targets_version, device, data_dir: Path | None = None):
    """
    Build the hybrid model for V2 targets.
    """
    max_actions = int(manifest["max_actions"]) if manifest else 100
    head_version = targets_version
    enum_sizes = manifest["enums"]
    # dims inferred from data (not constants)
    seq_len, gamestate_dim = 10, 128
    action_in_dim = 8
    
    # derive indices & vocab from dataset artifacts
    spec = load_feature_spec(data_dir) if data_dir else {}
    model = ImitationHybridModel(
        gamestate_dim=gamestate_dim,
        action_dim=action_in_dim,
        sequence_length=seq_len,
        hidden_dim=256,
        num_attention_heads=8,
        max_actions=max_actions, 
        head_version=head_version, 
        enum_sizes=enum_sizes,
        feature_spec=spec
    )
    # Put the time head into a high-gradient region at init:
    # softplus(-3.7) ≈ 0.024 s, matching your mean target (~24 ms).
    try:
        with torch.no_grad():
            if hasattr(model, "time_head") and getattr(model.time_head, "bias", None) is not None:
                model.time_head.bias.fill_(-3.7)
    except Exception:
        pass
    return model

def estimate_class_weights(train_loader, targets_version="v2", enum_sizes=None):
    """
    Build softened inverse-frequency class weights with capping.
    This keeps rare classes helpful without letting weights explode.
    Also prints key_id usage stats so you can see the actual range present.
    """
    # V2 only
    n_btn  = int(enum_sizes["button"]["size"])
    n_ka   = int(enum_sizes["key_action"]["size"])
    n_kid  = int(enum_sizes["key_id"]["size"])
    n_sy   = int(enum_sizes["scroll_y"]["size"])
    btn_counts = torch.ones(n_btn); ka_counts = torch.ones(n_ka)
    kid_counts = torch.ones(n_kid); sy_counts = torch.ones(n_sy)
    for b in train_loader:
        tgt = b["action_target"]; m = b["valid_mask"]
        one = lambda t: torch.ones_like(t, dtype=torch.float, device=t.device)

        # button / key_action: clamp into vocab just in case
        btn_idx = tgt[...,3].long().clamp(0, n_btn-1)
        ka_idx  = tgt[...,4].long().clamp(0, n_ka-1)
        btn_counts.index_add_(0, btn_idx[m].view(-1), one(btn_idx[m].view(-1)))
        ka_counts.index_add_(0,  ka_idx[m].view(-1),  one(ka_idx[m].view(-1)))

        # key_id: gate by key_action != NONE and clamp into vocab
        none_idx_ka = int(enum_sizes["key_action"]["none_index"])
        ka_flat = tgt[...,4].long().view(-1)
        kid_idx = tgt[...,5].long().clamp(0, n_kid-1).view(-1)
        m_flat = m.view(-1)
        kid_mask = (ka_flat != none_idx_ka) & m_flat
        if kid_mask.any():
            kid_counts.index_add_(0, kid_idx[kid_mask], one(kid_idx[kid_mask]))

        # scroll_y: map {-1,0,+1} → {0,1,2}
        none_idx_sy = int(enum_sizes["scroll_y"]["none_index"])  # usually 1
        sy_idx = (tgt[...,6].long() + none_idx_sy).clamp(0, n_sy-1).view(-1)
        sy_counts.index_add_(0, sy_idx[m_flat], one(sy_idx[m_flat]))
    inv = lambda c: (1.0/(c+1e-9))
    def soften_cap(w: torch.Tensor) -> torch.Tensor:
        # soften very large ratios and cap extreme outliers
        w = torch.pow(w, 0.5)                      # sqrt inverse-frequency
        cap = 4.0 * w.mean().clamp_min(1e-6)       # cap at 4x mean
        return torch.clamp(w, max=cap)
    norm = lambda w: (w / w.mean().clamp_min(1e-6))
    w_btn = norm(soften_cap(inv(btn_counts)))
    w_ka  = norm(soften_cap(inv(ka_counts)))
    w_kid = norm(soften_cap(inv(kid_counts)))
    w_sy  = norm(soften_cap(inv(sy_counts)))

    # --- DEBUG: key_id usage evidence ---
    # counts were initialized at 1; "used" means >1 after accumulation
    used_idx = (kid_counts > 1).nonzero(as_tuple=False).view(-1)
    kid_used = int(used_idx.numel())
    kid_max_idx = int(used_idx.max().item()) if kid_used > 0 else -1
    # Top key_id indices by count (will mostly be true used ids)
    topk = min(10, n_kid)
    top_ids = torch.topk(kid_counts, k=topk).indices.cpu().tolist()
    print(f"[class_w] key_id used={kid_used}/{n_kid}, max_idx={kid_max_idx}, top_ids={top_ids}")

    return {"btn": w_btn, "ka": w_ka, "kid": w_kid, "sy": w_sy}

def clamp_time(t, time_div, time_clip, already_scaled=False):
    if already_scaled:  # V2 saved time already scaled
        return torch.clamp(t, 0.0, float(time_clip))
    return torch.clamp(t / float(time_div), 0.0, float(time_clip))

# ---------------------------
# Validation metrics helpers
# ---------------------------
def _init_val_agg(enum_sizes: Dict[str, Dict[str, int]]) -> Dict:
    def z(n): return torch.zeros(int(n), dtype=torch.long)
    return {
        "btn_pred": z(enum_sizes["button"]["size"]),
        "btn_tgt":  z(enum_sizes["button"]["size"]),
        "ka_pred":  z(enum_sizes["key_action"]["size"]),
        "ka_tgt":   z(enum_sizes["key_action"]["size"]),
        "kid_pred": z(enum_sizes["key_id"]["size"]),
        "kid_tgt":  z(enum_sizes["key_id"]["size"]),
        "sy_pred":  z(enum_sizes["scroll_y"]["size"]),
        "sy_tgt":   z(enum_sizes["scroll_y"]["size"]),
        "time_pred_sum": 0.0,              # post-activation (what the model outputs)
        "time_pred_sum_clamped": 0.0,      # post-activation but clamped to >= 0 (reference only)
        "time_mae_sum": 0.0,
        "time_tgt_sum":  0.0,
        "time_count":    0,
        "time_pred_neg": 0,                # from time_pre (pre-activation)
        "time_post_zero": 0,               # post-activation ~ 0 (after softplus)
        # full value captures for richer stats & plots (safe: ~6–10k rows)
        "time_pred_vals": [],
        "time_tgt_vals":  [],
        "x_pred_vals":    [],
        "x_tgt_vals":     [],
        "y_pred_vals":    [],
        "y_tgt_vals":     [],
        # event-type summaries
        "evt_pred": {"MOVE":0, "CLICK":0, "KEY":0, "SCROLL":0, "MULTI":0},
        "evt_tgt":  {"MOVE":0, "CLICK":0, "KEY":0, "SCROLL":0},
        # event confusion (rows=tgt, cols=pred) over {CLICK, KEY, SCROLL, MOVE}
        "evt_cm":   [[0,0,0,0] for _ in range(4)],
        # optional coordinate errors (we only print them; already in summary)
        "mae_x_sum": 0.0,
        "mae_y_sum": 0.0,
        # key_id top-k on eligible rows (where key_action != NONE)
        "kid_n": 0,
        "kid_top1": 0,
        "kid_top3": 0,
    }

def _update_val_agg(agg: Dict, heads: Dict[str, torch.Tensor], target: torch.Tensor,
                    valid_mask: torch.Tensor, enum_sizes: Dict, time_div: float, time_clip: float):
    """
    Accumulate per-head histograms & event-type summaries on masked rows.
    Does NOT modify loss or training behavior.
    """
    # Shapes
    B, A = target.shape[:2]
    vm2 = valid_mask.view(B, A)
    m = vm2.view(-1)

    # Sizes & none indices
    n_btn = int(enum_sizes["button"]["size"])
    n_ka  = int(enum_sizes["key_action"]["size"])
    n_kid = int(enum_sizes["key_id"]["size"])
    n_sy  = int(enum_sizes["scroll_y"]["size"])
    btn_none = int(enum_sizes["button"]["none_index"])
    ka_none  = int(enum_sizes["key_action"]["none_index"])
    sy_none  = int(enum_sizes["scroll_y"]["none_index"])

    # Flatten helpers
    def fl(x): return x.view(-1)

    # --- button ---
    btn_logits = heads["button_logits"].view(-1, n_btn)
    btn_pred = btn_logits.argmax(-1)
    btn_tgt  = fl(target[...,3]).long().clamp(0, n_btn-1)
    agg["btn_pred"] += torch.bincount(btn_pred[m], minlength=n_btn).cpu()
    agg["btn_tgt"]  += torch.bincount(btn_tgt[m],  minlength=n_btn).cpu()

    # --- key_action ---
    ka_logits = heads["key_action_logits"].view(-1, n_ka)
    ka_pred = ka_logits.argmax(-1)
    ka_tgt  = fl(target[...,4]).long().clamp(0, n_ka-1)
    agg["ka_pred"] += torch.bincount(ka_pred[m], minlength=n_ka).cpu()
    agg["ka_tgt"]  += torch.bincount(ka_tgt[m],  minlength=n_ka).cpu()

    # --- key_id (only when key_action != NONE) ---
    kid_logits = heads["key_id_logits"].view(-1, n_kid)
    kid_pred = kid_logits.argmax(-1)
    kid_tgt  = fl(target[...,5]).long().clamp(0, n_kid-1)
    kid_keep = m & (ka_tgt != ka_none)
    if kid_keep.any():
        agg["kid_pred"] += torch.bincount(kid_pred[kid_keep], minlength=n_kid).cpu()
        agg["kid_tgt"]  += torch.bincount(kid_tgt[kid_keep],  minlength=n_kid).cpu()
        # key_id top-k (eligible rows only)
        agg["kid_n"]     += int(kid_keep.sum().item())
        # top-1
        agg["kid_top1"]  += int((kid_pred[kid_keep] == kid_tgt[kid_keep]).sum().item())
        # top-3
        top3 = kid_logits.topk(k=3, dim=-1).indices[kid_keep]
        agg["kid_top3"]  += int((top3 == kid_tgt[kid_keep].unsqueeze(-1)).any(dim=-1).sum().item())

    # --- scroll_y mapping: raw {-1,0,+1} -> idx {0,1,2} (none=center) ---
    sy_raw = fl(target[...,6]).long()
    sy_tgt = (sy_raw + sy_none).clamp(0, n_sy-1)
    sy_logits = heads["scroll_y_logits"].view(-1, n_sy)
    sy_pred = sy_logits.argmax(-1)
    agg["sy_pred"] += torch.bincount(sy_pred[m], minlength=n_sy).cpu()
    agg["sy_tgt"]  += torch.bincount(sy_tgt[m],  minlength=n_sy).cpu()

    # --- time stats on masked rows ---
    # post-act (what we evaluate with) and pre-act (for negative-rate debug)
    t_pred = fl(heads["time_q"][..., 1])  # Use median quantile (q=0.5)
    t_tgt  = fl(clamp_time(target[...,0], time_div, time_clip, already_scaled=True))
    t_pred_m = t_pred[m]
    t_tgt_m  = t_tgt[m]
    agg["time_pred_sum"] += float(t_pred_m.sum().item())
    agg["time_pred_sum_clamped"] += float(t_pred_m.clamp_min(0.0).sum().item())
    agg["time_tgt_sum"]  += float(t_tgt_m.sum().item())
    agg["time_count"]    += int(t_pred_m.numel())
    agg["time_mae_sum"]  += float((t_pred_m - t_tgt_m).abs().sum().item())
    agg["time_post_zero"] += int((t_pred_m <= 1e-8).sum().item())
    # collect full vals for richer stats/plots
    if t_pred_m.numel() > 0:
        agg["time_pred_vals"].extend(t_pred_m.detach().flatten().cpu().tolist())
        agg["time_tgt_vals"].extend(t_tgt_m.detach().flatten().cpu().tolist())
    # If the model exposes pre-activation, count negatives there:
    if "time_q_pre" in heads:
        t_pre_m = fl(heads["time_q_pre"][..., 1])[m]  # Use median quantile (q=0.5)
        agg["time_pred_neg"] += int((t_pre_m < 0).sum().item())

    # --- optional: accumulate coordinate MAE (masked) so we can print inside detailed block ---
    # xy tensors (heteroscedastic means only)
    x_pred = fl(heads["x_mu"]); y_pred = fl(heads["y_mu"])
    x_tgt = fl(target[...,1]); y_tgt = fl(target[...,2])
    xpm = x_pred[m]; xtm = x_tgt[m]
    ypm = y_pred[m]; ytm = y_tgt[m]
    agg["mae_x_sum"] += float((xpm - xtm).abs().sum().item())
    agg["mae_y_sum"] += float((ypm - ytm).abs().sum().item())
    if xpm.numel() > 0:
        agg["x_pred_vals"].extend(xpm.detach().flatten().cpu().tolist())
        agg["x_tgt_vals"].extend(xtm.detach().flatten().cpu().tolist())
        agg["y_pred_vals"].extend(ypm.detach().flatten().cpu().tolist())
        agg["y_tgt_vals"].extend(ytm.detach().flatten().cpu().tolist())

    # --- event-type summaries (from event head) ---
    with torch.no_grad():
        n_btn = int(enum_sizes["button"]["size"])
        n_ka  = int(enum_sizes["key_action"]["size"])
        n_sy  = int(enum_sizes["scroll_y"]["size"])
        btn_none = int(enum_sizes["button"]["none_index"])
        ka_none  = int(enum_sizes["key_action"]["none_index"])
        sy_none  = int(enum_sizes["scroll_y"]["none_index"])

        # Define m_flat for both paths
        m_flat = m.view(-1)
        
        # --- event predictions: event head only ---
        evt_pred = heads["event_logits"].argmax(-1)  # (B,A)
        evt_pred_m = evt_pred.view(-1)[m]
        agg["evt_pred"]["CLICK"]  += int((evt_pred_m == 0).sum().item())
        agg["evt_pred"]["KEY"]    += int((evt_pred_m == 1).sum().item())
        agg["evt_pred"]["SCROLL"] += int((evt_pred_m == 2).sum().item())
        agg["evt_pred"]["MOVE"]   += int((evt_pred_m == 3).sum().item())

        # --- event TARGETS using canonical order (CLICK=0, KEY=1, SCROLL=2, MOVE=3) ---
        btn_tgt = target[...,3].long().view(-1)
        ka_tgt  = target[...,4].long().view(-1)
        sy_raw  = target[...,6].long().view(-1)  # {-1,0,+1}
        t_sy = (sy_raw != 0)
        t_btn = (btn_tgt != btn_none)
        t_ka  = (ka_tgt  != ka_none)
        evt_tgt = torch.where(t_btn, torch.tensor(0, device=sy_raw.device),   # CLICK idx 0
                              torch.where(t_ka,  torch.tensor(1, device=sy_raw.device),   # KEY idx 1
                              torch.where(t_sy,  torch.tensor(2, device=sy_raw.device),   # SCROLL idx 2
                                                   torch.tensor(3, device=sy_raw.device)))) # MOVE idx 3
        
        # Confusion matrix
        for t, p in zip(evt_tgt[m_flat].tolist(), evt_pred_m.tolist()):
            agg["evt_cm"][t][p] += 1

        # Target non-NONE flags using canonical order
        t_btn = (btn_tgt != btn_none)[m]
        t_ka  = (ka_tgt  != ka_none)[m]
        t_sy  = (sy_tgt  != sy_none)[m]
        t_is = t_btn.to(torch.int) + t_ka.to(torch.int) + t_sy.to(torch.int)
        agg["evt_tgt"]["CLICK"]  += int(t_btn.sum().item())
        agg["evt_tgt"]["KEY"]    += int(t_ka.sum().item())
        agg["evt_tgt"]["SCROLL"] += int(t_sy.sum().item())
        agg["evt_tgt"]["MOVE"]   += int((t_is == 0).sum().item())

def _topk_from_counts(counts: torch.Tensor, k: int = 10):
    arr = counts.tolist()
    order = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
    out = [(i, arr[i]) for i in order[:k] if arr[i] > 0]
    return out

def _print_val_agg(agg: Dict, enum_sizes: Dict, time_div: float, outdir: str = None, epoch_idx: int = None):
    def topk_counts(counts: List[int], k=10):
        return _topk_counts_list(counts, k)
    total_masked = agg["time_count"]  # same denominator for most per-row stats

    print("=== Detailed validation report ===")
    print(f"masked rows used: {total_masked}")
    # Button / Key Action: show both preds & tgts (plain ints)
    btn_pred_sum = sum(_as_int(c) for c in agg['btn_pred']); btn_tgt_sum = sum(_as_int(c) for c in agg['btn_tgt'])
    ka_pred_sum  = sum(_as_int(c) for c in agg['ka_pred']);  ka_tgt_sum  = sum(_as_int(c) for c in agg['ka_tgt'])
    print(f"[button]     pred total={btn_pred_sum}, top5={[(i,_as_int(c)) for i,c in topk_counts(agg['btn_pred'],5)]}, tgt top5={[(i,_as_int(c)) for i,c in topk_counts(agg['btn_tgt'],5)]}")
    print(f"[key_action] pred total={ka_pred_sum},  top5={[(i,_as_int(c)) for i,c in topk_counts(agg['ka_pred'],5)]},  tgt top5={[(i,_as_int(c)) for i,c in topk_counts(agg['ka_tgt'],5)]}")
    # Key ID: show both preds & tgts (only counted where key_action!=NONE)
    kid_pred_sum = sum(_as_int(c) for c in agg['kid_pred'])
    kid_tgt_sum  = sum(_as_int(c) for c in agg['kid_tgt'])
    print(f"[key_id]     pred total={kid_pred_sum}, top10={[(i,_as_int(c)) for i,c in topk_counts(agg['kid_pred'],10)]}")
    print(f"[key_id]     tgt  total={kid_tgt_sum},  top10={[(i,_as_int(c)) for i,c in topk_counts(agg['kid_tgt'],10)]}")
    if agg["kid_n"] > 0:
        top1 = 100.0 * agg["kid_top1"] / agg["kid_n"]
        top3 = 100.0 * agg["kid_top3"] / agg["kid_n"]
        print(f"[key_id]     top-1={top1:.1f}% | top-3={top3:.1f}% (n={agg['kid_n']})")
    # Scroll: show both preds & tgts
    sy_pred_sum = sum(_as_int(c) for c in agg['sy_pred']); sy_tgt_sum = sum(_as_int(c) for c in agg['sy_tgt'])
    sy_pred_pairs = [(i,_as_int(c)) for i,c in enumerate(agg['sy_pred']) if _as_int(c)>0]
    sy_tgt_pairs  = [(i,_as_int(c)) for i,c in enumerate(agg['sy_tgt'])  if _as_int(c)>0]
    print(f"[scroll_y]   pred total={sy_pred_sum}, counts={sy_pred_pairs}  (idx: 0=-1, 1=NONE, 2=+1)")
    print(f"[scroll_y]   tgt  total={sy_tgt_sum},  counts={sy_tgt_pairs}   (idx: 0=-1, 1=NONE, 2=+1)")
    mean_pred = agg["time_pred_sum"]/total_masked if total_masked>0 else 0.0
    mean_tgt  = agg["time_tgt_sum"]/total_masked if total_masked>0 else 0.0
    mae_t     = agg["time_mae_sum"]/total_masked if total_masked>0 else 0.0
    print(f"[time] mean_pred={mean_pred:.4f} ({mean_pred*time_div:.1f} ms) | mean_tgt={mean_tgt:.4f} ({mean_tgt*time_div:.1f} ms)")
    print(f"[time/ref] mean_pred_clamped={agg['time_pred_sum_clamped']/total_masked:.4f} ({(agg['time_pred_sum_clamped']/total_masked)*time_div:.1f} ms)")
    print(f"[time] mae={mae_t*time_div:.1f} ms")
    # richer distribution stats for time/x/y
    tp = torch.tensor(agg["time_pred_vals"])
    tt = torch.tensor(agg["time_tgt_vals"])
    xp = torch.tensor(agg["x_pred_vals"])
    xt = torch.tensor(agg["x_tgt_vals"])
    yp = torch.tensor(agg["y_pred_vals"])
    yt = torch.tensor(agg["y_tgt_vals"])
    if tp.numel() and tt.numel():
        sp = _percentiles_t(tp); st = _percentiles_t(tt)
        def fmt_ms(d): return {k:(v*time_div if k!="uniq" else v) for k,v in d.items()}
        spm, stm = fmt_ms(sp), fmt_ms(st)
        print(f"[time/dist] pred ms: min={spm['min']:.1f} p25={spm['p25']:.1f} med={spm['median']:.1f} p75={spm['p75']:.1f} max={spm['max']:.1f} | mean={spm['mean']:.1f} std={spm['std']:.1f} | uniq={spm['uniq']}")
        print(f"[time/dist]  tgt ms: min={stm['min']:.1f} p25={stm['p25']:.1f} med={stm['median']:.1f} p75={stm['p75']:.1f} max={stm['max']:.1f} | mean={stm['mean']:.1f} std={stm['std']:.1f} | uniq={stm['uniq']}")
    if xp.numel() and xt.numel():
        sxp = _percentiles_t(xp); sxt = _percentiles_t(xt)
        syp = _percentiles_t(yp); syt = _percentiles_t(yt)
        def fmt_xy(tag, spred, stgt):
            print(f"[{tag}/dist] pred: min={spred['min']:.1f} p25={spred['p25']:.1f} med={spred['median']:.1f} p75={spred['p75']:.1f} max={spred['max']:.1f} | mean={spred['mean']:.1f} std={spred['std']:.1f} | uniq={spred['uniq']}")
            print(f"[{tag}/dist]  tgt: min={stgt['min']:.1f} p25={stgt['p25']:.1f} med={stgt['median']:.1f} p75={stgt['p75']:.1f} max={stgt['max']:.1f} | mean={stgt['mean']:.1f} std={stgt['std']:.1f} | uniq={stgt['uniq']}")
        fmt_xy("x", sxp, sxt)
        fmt_xy("y", syp, syt)
    if getattr(CFG, "debug_time", False):
        def pct(n,d): return 100.0*n/max(1,d)
        print(f"[time/debug] post_zero_frac={pct(agg['time_post_zero'], total_masked):.2f}%")
        if 'time_pred_neg' in agg:
            print(f"[time/debug] pre_neg_frac={pct(agg['time_pred_neg'], total_masked):.2f}%")
    # Echo coordinate MAE in the detailed block too (rounded)
    if agg["mae_x_sum"] > 0 or agg["mae_y_sum"] > 0:
        print(f"[xy] mae_x={agg['mae_x_sum']/total_masked:.1f} px | mae_y={agg['mae_y_sum']/total_masked:.1f} px")
    # Event distributions + pretty confusion matrix
    ep = agg["evt_pred"]; et = agg["evt_tgt"]
    total_evt = sum(ep.values())  # same as total_masked
    def fr(v): return f"{v} ({100.0*v/max(1,total_evt):.1f}%)"
    print(f"[event(pred)] MOVE={fr(ep['MOVE'])}, CLICK={fr(ep['CLICK'])}, KEY={fr(ep['KEY'])}, SCROLL={fr(ep['SCROLL'])}, MULTI={fr(ep['MULTI'])}")
    print(f"[event(tgt)]  MOVE={fr(et['MOVE'])}, CLICK={fr(et['CLICK'])}, KEY={fr(et['KEY'])}, SCROLL={fr(et['SCROLL'])}")
    # Confusion (rows=tgt, cols=pred) with labels
    labels = ["CLICK","KEY","SCROLL","MOVE"]
    cm = [[_as_int(c) for c in row] for row in agg["evt_cm"]]
    # header
    header = " " * 12 + " | " + "  ".join(f"{lab:>6}" for lab in labels) + " |  total"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_total = max(1, sum(row))
        cells = "  ".join(f"{n:>6}" for n in row)
        pct  = "  ".join(f"{(n/row_total*100):>5.1f}%" for n in row)
        print(f"{labels[i]:>12} | {cells} | {row_total:>6}")
        print(f"{'':>12} | {pct} |")
    print("=== End detailed report ===")

    # -------- optional plots saved under checkpoints/<run>/val_plots/ --------
    if outdir and _MPL_OK:
        os.makedirs(outdir, exist_ok=True)
        eid = f"epoch_{int(epoch_idx):03d}" if epoch_idx is not None else "epoch"
        # time series overlay (downsample to first 2k points to keep files small)
        try:
            maxn = 2000
            if tp.numel() and tt.numel():
                n = min(int(tp.numel()), int(tt.numel()), maxn)
                x_axis = range(n)
                plt.figure(figsize=(10,3))
                plt.plot(x_axis, (tp[:n]*time_div).cpu().numpy(), label="pred ms")
                plt.plot(x_axis, (tt[:n]*time_div).cpu().numpy(), label="tgt ms", alpha=0.7)
                plt.legend(); plt.xlabel("masked row idx"); plt.ylabel("time (ms)"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_time_series.png")); plt.close()
                # residual histogram
                res = (tp - tt).cpu().numpy() * time_div
                plt.figure(figsize=(6,4)); plt.hist(res, bins=50)
                plt.xlabel("time residual (ms)"); plt.ylabel("count"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_time_residual_hist.png")); plt.close()
            if xp.numel() and xt.numel():
                n = min(int(xp.numel()), int(xt.numel()), maxn)
                xa = range(n)
                plt.figure(figsize=(10,3))
                plt.plot(xa, xp[:n].cpu().numpy(), label="x pred")
                plt.plot(xa, xt[:n].cpu().numpy(), label="x tgt", alpha=0.7)
                plt.legend(); plt.xlabel("masked row idx"); plt.ylabel("x (px)"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_x_series.png")); plt.close()
                rx = (xp - xt).cpu().numpy()
                plt.figure(figsize=(6,4)); plt.hist(rx, bins=50)
                plt.xlabel("x residual (px)"); plt.ylabel("count"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_x_residual_hist.png")); plt.close()
            if yp.numel() and yt.numel():
                n = min(int(yp.numel()), int(yt.numel()), maxn)
                xa = range(n)
                plt.figure(figsize=(10,3))
                plt.plot(xa, yp[:n].cpu().numpy(), label="y pred")
                plt.plot(xa, yt[:n].cpu().numpy(), label="y tgt", alpha=0.7)
                plt.legend(); plt.xlabel("masked row idx"); plt.ylabel("y (px)"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_y_series.png")); plt.close()
                ry = (yp - yt).cpu().numpy()
                plt.figure(figsize=(6,4)); plt.hist(ry, bins=50)
                plt.xlabel("y residual (px)"); plt.ylabel("count"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{eid}_y_residual_hist.png")); plt.close()
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

# (removed: legacy V1 IL loss)

import torch
import torch.nn.functional as F




def _masked_metrics(heads: dict, target: torch.Tensor, targets_version="v2"):
    """Simple masked metrics to track IL progress."""
    with torch.no_grad():
        # V2 format: target is (B,A,7) with [time_s, x, y, button, key_action, key_id, scroll_y]
        mask = (torch.abs(target).sum(dim=-1) > 0)
        mflat = mask.reshape(-1)
        # preds
        bt = heads["button_logits"].argmax(-1)
        ka = heads["key_action_logits"].argmax(-1)
        kid = heads["key_id_logits"].argmax(-1)
        sy = heads["scroll_y_logits"].argmax(-1)
        # tgts
        tb = target[...,3].long()
        tka = target[...,4].long()
        tkid = target[...,5].long()
        tsy = target[...,6].long()
        def m_acc(pred, tgt):
            p = pred.reshape(-1)[mflat]; t = tgt.reshape(-1)[mflat]
            return (p == t).float().mean().item() if p.numel() else 0.0
        def m_mae(pred, tgt):
            p = pred.reshape(-1)[mflat].float(); t = tgt.reshape(-1)[mflat].float()
            return (p - t).abs().mean().item() if p.numel() else 0.0
        valid_frac = mflat.float().mean().item() if mflat.numel() else 0.0
        return {
            "acc_button": m_acc(bt, tb),
            "acc_key_action": m_acc(ka, tka),
            "acc_key_id": m_acc(kid, tkid),
            "acc_scroll_y": m_acc(sy, tsy),
            "mae_time": m_mae(heads["time_q"][..., 1], target[...,0]),  # Use median quantile (q=0.5)
            "mae_x":    m_mae(heads["x_mu"],    target[...,1]),  # Use heteroscedastic mean
            "mae_y":    m_mae(heads["y_mu"],    target[...,2]),  # Use heteroscedastic mean
            "valid_frac": valid_frac,
        }

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cpu', scheduler=None,
                use_class_weights=True, loss_w=None, time_div=1.0,
                targets_version="v2", time_clip=3.0, enum_sizes=None):
    """Train the model"""
    
    print(f"Starting training on {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    
    if use_class_weights:
        print("Estimating class weights on train set (masked)…")
        if enum_sizes is not None:
            class_w = estimate_class_weights(train_loader, targets_version, enum_sizes)
            print("  Using dynamic class weights from enum_sizes")
        else:
            print("  Warning: enum_sizes not available, class weights disabled")
            class_w = None
        print("  class_w keys:", list(class_w.keys()) if class_w else "None")
    else:
        print("Class weights disabled via flag.")
        class_w = None

    # Early stopping / checkpoint
    best_val = float("inf")
    patience = 6
    patience_left = patience
    ckpt_dir = os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Create ActionTensorLoss criterion
    if enum_sizes is not None:
        from ilbot.model.losses import ActionTensorLoss
        criterion = ActionTensorLoss(enum_sizes)
        print(f"Created ActionTensorLoss with enum_sizes: {list(enum_sizes.keys())}")
    else:
        raise ValueError("enum_sizes is required for ActionTensorLoss")
    

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            temporal_sequence = batch['temporal_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            action_target = batch['action_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(temporal_sequence, action_sequence, return_logits=True)

            # --- DEBUG: shapes, masks, and basic stats
            tgt = {"event": action_target[..., 1], "scroll_y_idx": action_target[..., 6], "time_ms": action_target[..., 0], "xy_px": action_target[..., 2:4]}
            mask = batch.get("valid_mask", None)
            if mask is not None:
                valid_sum = int(mask.detach().sum().item())
                total = int(mask.numel())
                LOG.info("[dbg-train] batch_idx=%d valid_sum=%d / %d (%.2f%%)",
                         batch_idx, valid_sum, total, 100.0*valid_sum/max(1,total))
            _dbg_counts("tgt.event", tgt.get("event"), mask)
            _dbg_counts("tgt.scroll_y_idx", tgt.get("scroll_y_idx"), mask)
            _dbg_hist("tgt.time_ms", tgt.get("time_ms"), mask)
            # XY index audit - Replace your current y tap with this very explicit version:
            # Suppose action_target shape is (B, A, C)
            x_col, y_col = 1, 2  # <the exact indices you believe are x and y>
            x_t = action_target[..., x_col]
            y_t = action_target[..., y_col]
            print(f"[CHK] tgt.xy_px_x (masked) min={x_t[mask].min():.1f} max={x_t[mask].max():.1f}")
            print(f"[CHK] tgt.xy_px_y (masked) min={y_t[mask].min():.1f} max={y_t[mask].max():.1f}")
            
            _dbg_hist("tgt.xy_px_x", tgt.get("xy_px")[...,0] if tgt.get("xy_px") is not None else None, mask)
            _dbg_hist("tgt.xy_px_y", tgt.get("xy_px")[...,1] if tgt.get("xy_px") is not None else None, mask)

            # how many total train batches this epoch
            total_train_batches = len(train_loader)

            # LAST-BATCH diagnostics
            if batch_idx == total_train_batches - 1:
                vm_last = batch['valid_mask'].to(device)
                B, A = action_target.size(0), action_target.size(1)
                if vm_last.dim() == 1 or vm_last.shape[:2] != (B, A):
                    vm_last = vm_last.view(B, A)
                vcount = int(vm_last.sum().item())
                print(f"[dbg-last-train] batch_idx={batch_idx} B={B} A={A} valid_sum={vcount}")

            # ---------- one-time debug (place INSIDE the loop, after outputs) ----------
            if not hasattr(model, "_dbg_heads_once"):
                model._dbg_heads_once = True
                if isinstance(outputs, dict):
                    for k, v in outputs.items():
                        print(f"[dbg] head {k}: {tuple(v.shape)}")
                else:
                    print(f"[dbg] model output shape: {tuple(outputs.shape)}")

            # Prepare a 2D mask for diagnostics and optional skipping
            vm = batch['valid_mask'].to(device)
            B, A = action_target.size(0), action_target.size(1)
            if vm.dim() == 1 or vm.shape[:2] != (B, A):
                vm = vm.view(B, A)

            # Skip batches with zero valid rows to avoid NaNs in CE
            if vm.sum().item() == 0:
                # optionally: print once
                if not hasattr(model, "_dbg_skip_once"):
                    model._dbg_skip_once = True
                    print("[dbg] skipped a batch with zero valid rows")
                continue

            # One-time stats print
            if not hasattr(model, "_dbg_once"):
                model._dbg_once = True
                total_valid = int(vm.sum().item())
                frac = total_valid / (B * A)

                ka_none = int(enum_sizes["key_action"]["none_index"])
                ka = action_target[..., 4].long()
                kid_rows = int((vm & (ka != ka_none)).sum().item())

                sy_vals = action_target[..., 6].long()         # {-1,0,+1}
                sy_rows = int(vm.sum().item())                 # all valid rows participate
                sy_events = int((vm & (sy_vals != 0)).sum().item())

                print(f"[dbg] B={B} A={A} valid={total_valid} ({frac:.3f}) | "
                    f"kid_rows={kid_rows} | sy_rows={sy_rows} (nonzero={sy_events})")
                
                # Sanity check: verify model outputs match loss consumption
                if isinstance(outputs, dict):
                    expected_keys = {"event_logits", "time_q_pre", "time_q", "x_mu", "x_logsig", "y_mu", "y_logsig", 
                                   "button_logits", "key_action_logits", "key_id_logits", "scroll_y_logits"}
                    actual_keys = set(outputs.keys())
                    missing_keys = expected_keys - actual_keys
                    extra_keys = actual_keys - expected_keys
                    if missing_keys:
                        raise ValueError(f"Model missing required outputs: {missing_keys}")
                    if extra_keys:
                        print(f"[warn] Model has extra outputs: {extra_keys}")
                    print(f"[sanity] ✓ Model outputs match loss consumption ({len(expected_keys)} keys)")
            # ---------------------------------------------------------------------------

            
            if isinstance(outputs, dict):
                # Use ActionTensorLoss for V2 outputs
                valid_mask = batch['valid_mask'].to(device)
                loss, loss_details = criterion(outputs, action_target, valid_mask)
            else:
                # Back-compat with legacy tensor output + custom criterion
                loss = criterion(outputs, action_target)

            # --- DEBUG: check if any head is constant / NaN / wrong sized
            for k,v in outputs.items():
                if torch.is_tensor(v):
                    if torch.isnan(v).any():
                        LOG.error("[DBG] outputs.%s contains NaNs", k)
                    _dbg_hist(f"out.{k}", v)

            # --- DEBUG: individual loss terms
            try:
                for lk, lv in loss_details.items():
                    LOG.info("[dbg-train] loss[%s]=%.4f", lk, float(lv.detach().item()))
            except Exception:
                pass
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")


        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        print("Validating...")
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        # aggregate detailed metrics across VAL
        val_agg = _init_val_agg(enum_sizes) if enum_sizes is not None else None
        # V2 only
        vm = {"acc_button":0.0,"acc_key_action":0.0,"acc_key_id":0.0,"acc_scroll_y":0.0,
              "mae_time":0.0,"mae_x":0.0,"mae_y":0.0,"valid_frac":0.0}
        vm_n = 0
        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                outputs = model(temporal_sequence, action_sequence, return_logits=True)

                # --- DEBUG: validation targets and masks
                tgt = {"event": action_target[..., 1], "scroll_y_idx": action_target[..., 6], "time_ms": action_target[..., 0], "xy_px": action_target[..., 2:4]}
                mask = batch.get("valid_mask", None)
                if mask is not None:
                    LOG.info("[dbg-val] batch_idx=%d valid_sum=%d / %d",
                             val_idx, int(mask.detach().sum().item()), int(mask.numel()))
                _dbg_counts("VAL tgt.event", tgt.get("event"), mask)
                _dbg_counts("VAL tgt.scroll_y_idx", tgt.get("scroll_y_idx"), mask)
                _dbg_hist("VAL tgt.time_ms", tgt.get("time_ms"), mask)

                # how many total VAL batches this epoch
                n_val_batches = len(val_loader)

                # LAST-BATCH diagnostics  (requires: for val_idx, batch in enumerate(val_loader))
                if val_idx == n_val_batches - 1:
                    vm_last = batch['valid_mask'].to(device)
                    B, A = action_target.size(0), action_target.size(1)
                    if vm_last.dim() == 1 or vm_last.shape[:2] != (B, A):
                        vm_last = vm_last.view(B, A)
                    vcount = int(vm_last.sum().item())
                    print(f"[dbg-last-val] batch_idx={val_idx} B={B} A={A} valid_sum={vcount}")

                # one-time head shapes (val)
                if not hasattr(model, "_dbg_heads_val_once"):
                    model._dbg_heads_val_once = True
                    if isinstance(outputs, dict):
                        for k, v in outputs.items():
                            print(f"[dbg] head {k}: {tuple(v.shape)}")
                    else:
                        print(f"[dbg] model output shape: {tuple(outputs.shape)}")

                # Prepare a 2D mask for diagnostics and optional skipping
                vm_mask = batch['valid_mask'].to(device)
                B, A = action_target.size(0), action_target.size(1)
                if vm_mask.dim() == 1 or vm_mask.shape[:2] != (B, A):
                    vm_mask = vm_mask.view(B, A)

                # Skip VAL batches with zero valid rows to avoid NaNs in CE
                if vm_mask.sum().item() == 0:
                    if not hasattr(model, "_dbg_skip_val_once"):
                        model._dbg_skip_val_once = True
                        print("[dbg] skipped a VAL batch with zero valid rows")
                    continue

                # One-time stats print (val)
                if not hasattr(model, "_dbg_val_once"):
                    model._dbg_val_once = True
                    total_valid = int(vm_mask.sum().item())
                    frac = total_valid / (B * A)

                    ka_none = int(enum_sizes["key_action"]["none_index"])
                    ka = action_target[..., 4].long()
                    kid_rows = int((vm_mask & (ka != ka_none)).sum().item())

                    sy_vals = action_target[..., 6].long()   # {-1,0,+1}
                    sy_rows = int(vm_mask.sum().item())      # all valid rows participate
                    sy_events = int((vm_mask & (sy_vals != 0)).sum().item())

                    print(f"[dbg] (val) B={B} A={A} valid={total_valid} ({frac:.3f}) | "
                        f"kid_rows={kid_rows} | sy_rows={sy_rows} (nonzero={sy_events})")


                if isinstance(outputs, dict):
                    # Use ActionTensorLoss for V2 outputs
                    valid_mask = batch['valid_mask'].to(device)
                    vloss, vloss_details = criterion(outputs, action_target, valid_mask)
                    mm = _masked_metrics(outputs, action_target, "v2")
                    for k in vm: vm[k] += mm[k]
                    vm_n += 1
                    # accumulate detailed distributions
                    if val_agg is not None:
                        _update_val_agg(val_agg, outputs, action_target, vm_mask, enum_sizes, time_div, time_clip)
                else:
                    vloss = criterion(outputs, action_target)

                # --- DEBUG: validation loss terms
                try:
                    if isinstance(outputs, dict) and 'vloss_details' in locals():
                        for lk, lv in vloss_details.items():
                            LOG.info("[dbg-val] loss[%s]=%.4f", lk, float(lv.detach().item()))
                except Exception:
                    pass
                val_loss_sum += float(vloss.item())
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        if vm_n > 0:
            for k in vm: vm[k] /= vm_n
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            # V2 only
            print(f"  Val metrics (masked avg over {vm_n} batches): "
                  f"acc_btn={vm['acc_button']:.3f} acc_ka={vm['acc_key_action']:.3f} acc_kid={vm['acc_key_id']:.3f} "
                  f"acc_sy={vm['acc_scroll_y']:.3f} "
                  f"mae_time={vm['mae_time']:.1f} mae_x={vm['mae_x']:.1f} mae_y={vm['mae_y']:.1f} "
                  f"| valid_frac={vm['valid_frac']:.3f}")
            # print detailed per-head distributions and time stats
            if val_agg is not None:
                # Save plots to run checkpoint dir
                plot_dir = os.path.join(ckpt_dir, "val_plots")
                _print_val_agg(val_agg, enum_sizes, time_div, outdir=plot_dir, epoch_idx=epoch+1)
        else:
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Scheduler step (if any)
        if scheduler is not None:
            scheduler.step()
        # Early stopping + checkpoint
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            patience_left = patience
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1,
                        "val_loss": best_val}, best_path)
            print(f"  ↳ Saved best checkpoint: {best_path}")
        else:
            patience_left -= 1
            print(f"  ↳ No improvement. Patience left: {patience_left}")
            if patience_left <= 0:
                print("Early stopping triggered.")
                break
        
        # CUDA memory management
        if torch.cuda.is_available():
            print(f"  CUDA Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")
            # Clear cache to free up memory
            torch.cuda.empty_cache()
        
        print("-" * 40)
    
    return train_losses, val_losses

def save_training_results(model, train_losses, val_losses, config):
    """Save training results and model"""
    
    # Create results directory
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None
    }
    
    with open(results_dir / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), results_dir / "model_weights.pth")
    
    print(f"Training results saved to {results_dir}/")
    print(f"Model weights saved to {results_dir}/model_weights.pth")

def optimize_cuda_settings():
    """Optimize CUDA settings for better performance"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled")

def main():
    """Main training function"""
    
    print("OSRS Imitation Learning Model Training")
    print("=" * 60)
    
    # Optimize CUDA settings
    optimize_cuda_settings()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available, using CPU")
    print(f"Device: {device}")
    
    # Load training setup
    print("Loading training setup...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train OSRS Bot Imitation Learning Model')
    parser.add_argument('--data_dir', type=str, default="data/06_final_training_data",
                        help='Path to training data directory')
    # Run params
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32, help='base batch size')
    parser.add_argument('--disable_auto_batch', action='store_true',
                        help='force batch_size; do not auto-optimize for CUDA')
    # Scheduler
    parser.add_argument('--step_size', type=int, default=8, help='StepLR step_size')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR gamma')
    # Class weights toggle
    parser.add_argument('--no_class_weights', action='store_true', help='disable class weighting')
    # Loss head weights
    parser.add_argument('--lw_time', type=float, default=0.1)
    parser.add_argument('--lw_x',    type=float, default=1.0)
    parser.add_argument('--lw_y',    type=float, default=1.0)
    parser.add_argument('--lw_type', type=float, default=1.0)
    parser.add_argument('--lw_btn',  type=float, default=1.0)
    parser.add_argument('--lw_key',  type=float, default=1.0)
    parser.add_argument('--lw_sx',   type=float, default=1.0)
    parser.add_argument('--lw_sy',   type=float, default=1.0)
    # Time scaling (normalize inside loss only)
    parser.add_argument('--time_div', type=float, default=1.0,
                        help='divide time by this value in loss (e.g., 1000 for ms->s)')
    # V2 only
    parser.add_argument("--targets_version", default="v2")
    parser.add_argument("--time_clip", type=float, default=None,
                        help="Override time_clip; if None, use manifest.")
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    train_loader, val_loader, manifest, tv = create_data_loaders_v2(data_dir, targets_version="v2", device=device)
    enum_sizes = (manifest or {}).get("enums", {})
    time_div = manifest["time_div"] if (manifest and args.time_div is None) else (args.time_div or 1000.0)
    time_clip = manifest["time_clip"] if (manifest and args.time_clip is None) else (args.time_clip or 3.0)
    
    # Create model and setup training components
    model = setup_model_v2(manifest=manifest or {}, targets_version="v2", device=device, data_dir=data_dir)
    
    # Runtime safety check for feature spec
    spec = load_feature_spec(data_dir)
    print("[feature_spec] groups:", {k: len(v) for k, v in spec["group_indices"].items()})
    print("[feature_spec] cat fields:", [(f["name"], len(f["indices"]), f["vocab_size"]) for f in spec["cat_fields"]])
    print("[feature_spec] total_cat_vocab:", spec["total_cat_vocab"])
    
    # Estimate class weights
    class_w = estimate_class_weights(train_loader, targets_version="v2", enum_sizes=enum_sizes)
    
    criterion, optimizer = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    config = {
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': train_loader.batch_size,
        'device': str(device),
        'step_size': args.step_size,
        'gamma': args.gamma,
        'use_class_weights': not args.no_class_weights,
        'loss_weights': {
            'time': args.lw_time, 'x': args.lw_x, 'y': args.lw_y,
            'type': args.lw_type, 'btn': args.lw_btn, 'key': args.lw_key,
            'sx': args.lw_sx, 'sy': args.lw_sy
        }
    }
    
    # Train the model
    print("\nStarting training...")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        scheduler=scheduler,
        use_class_weights=(not args.no_class_weights),
        loss_w=config['loss_weights'],
        time_div=args.time_div,
        targets_version=tv,
        time_clip=time_clip,
        enum_sizes=enum_sizes
    )
    
    # Save results
    print("\nSaving training results...")
    save_training_results(model, train_losses, val_losses, config)
    
    print("\nTraining completed successfully!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    
    if torch.cuda.is_available():
        print(f"Final CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")

# main entrypoint called by setup.py
def run_training(config: dict):
    """
    config = {
      data_dir, targets_version, enum_sizes, epochs, lr, weight_decay, batch_size, disable_auto_batch,
      grad_clip, step_size, gamma, use_log1p_time, time_div_ms, time_clip_s, loss_weights{...},
      seed, device
    }
    """
    _seed_everything(config.get("seed", 1337))

    # 1) dataset & loaders
    ds = OSRSDataset(
        config["data_dir"],
        targets_version=config.get("targets_version"),   # may be None → auto from manifest
        use_log1p_time=config.get("use_log1p_time", True),
        time_div_ms=config.get("time_div_ms", 1000),
        time_clip_s=config.get("time_clip_s"),
        enum_sizes=config.get("enum_sizes"),             # optional
        device=config.get("device"),
    )
    train_loader, val_loader = create_data_loaders(
        dataset=ds,
        batch_size=config.get("batch_size", 32),
        disable_cuda_batch_opt=config.get("disable_auto_batch", False),
    )

    # 2) model
    # Build model using manifest/enums when v2. Avoid free var use.
    enum_sizes = (
        config.get("enum_sizes")
        or (ds.get_enums() if ds.targets_version == "v2" else {})
    )
    model = create_model({
        'gamestate_dim': ds.G,
        'action_dim': ds.Fin,
        'sequence_length': ds.T,
        'hidden_dim': 256,
        'num_heads': 8,
        'max_actions': ds.A,
        # trust what the dataset loaded (auto-detected from manifest/files)
        'head_version': ds.targets_version,
        'enum_sizes': enum_sizes,
        'feature_spec': load_feature_spec(Path(config["data_dir"])),
        'use_log1p_time': config.get("use_log1p_time", True),
        'time_div_ms': config.get("time_div_ms", 1000.0)
    })

    # 3) optimizer/scheduler
    lr = config.get("lr", 1e-4)  # Reduce learning rate for stability
    wd = config.get("weight_decay", 1e-4)
    step_size = config.get("step_size", 4)  # More frequent LR reduction
    gamma = config.get("gamma", 0.7)  # Gentler LR reduction
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Move model to device
    device = torch.device(config.get("device", "cuda"))
    model = model.to(device)

    # 4) train loop
    # V2-only training
    losses_train, losses_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=None,  # not used in V2 path
        optimizer=optimizer,
        num_epochs=config.get("epochs", 40),
        device=config.get("device", "cuda"),
        scheduler=scheduler,
        use_class_weights=True,
        loss_w=config.get("loss_weights", {}),
        time_div=config.get("time_div_ms", 1000.0),
        targets_version="v2",
        time_clip=config.get("time_clip_s", 3.0),
        enum_sizes=enum_sizes
    )
    history = {"train_losses": losses_train, "val_losses": losses_val}
    return history

if __name__ == "__main__":
    main()
