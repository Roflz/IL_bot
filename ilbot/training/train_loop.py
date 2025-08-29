#!/usr/bin/env python3
"""
Training script for OSRS Imitation Learning Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import json
import time
import argparse
from pathlib import Path
from ilbot.training.setup import create_data_loaders, setup_model, setup_training, OSRSDataset
from ilbot.model.imitation_hybrid_model import ImitationHybridModel
from torch.optim.lr_scheduler import StepLR
import torch, os, numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# Add new imports for the updated run_training function
from ilbot.training.setup import OSRSDataset, create_data_loaders
from ilbot.model.imitation_hybrid_model import create_model
import random

# ---- helpers ---------------------------------------------------------------
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

def compute_loss_v1(heads: Dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor,
                    class_w: Dict[str, torch.Tensor] | None = None,
                    use_log1p_time: bool = True, time_div_ms: float = 1000.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    target: (B,A,8) legacy layout
    mask:   (B,A) True where row is valid (non-zero)
    """
    B, A, _ = target.shape
    mflat = mask.reshape(-1)
    # splits
    if use_log1p_time:
        t_time = torch.log1p(target[..., 0].clamp_min(0) * (1000.0 / time_div_ms))
    else:
        t_time = target[..., 0]
    t_type = target[..., 1].long().clamp(0, 4)
    t_x    = target[..., 2]
    t_y    = target[..., 3]
    t_btn  = target[..., 4].long().clamp(0, 3)
    t_key  = target[..., 5].long().clamp(0, 150)
    t_sx   = (target[..., 6].long() + 1).clamp(0, 2)
    t_sy   = (target[..., 7].long() + 1).clamp(0, 2)
    # regression losses (masked SmoothL1)
    def masked_l1(pred, tgt):
        p = pred.reshape(-1)[mflat]; t = tgt.reshape(-1)[mflat]
        return F.smooth_l1_loss(p, t, reduction="mean") if p.numel() else torch.zeros([], device=pred.device)
    # Masked CE helper (optionally weighted)
    def masked_ce(logits, tgt, C, w=None):
        L = logits.reshape(-1, C)[mflat]
        T = tgt.reshape(-1)[mflat]
        if L.numel() == 0:
            return torch.zeros([], device=logits.device)
        if w is not None and w.device != L.device:
            w = w.to(L.device)
        return F.cross_entropy(L, T, weight=w, reduction="mean")
    # Per-head losses
    if use_log1p_time:
        loss_time = masked_l1(heads["time"], t_time)
    else:
        loss_time = masked_l1(heads["time"], t_time)
    loss_x    = masked_l1(heads["x"], t_x)
    loss_y    = masked_l1(heads["y"], t_y)
    w_type = class_w["type"] if (class_w and "type" in class_w) else None
    w_btn  = class_w["btn"]  if (class_w and "btn"  in class_w) else None
    w_key  = class_w["key"]  if (class_w and "key"  in class_w) else None
    w_sx   = class_w["sx"]   if (class_w and "sx"   in class_w) else None
    w_sy   = class_w["sy"]   if (class_w and "sy"   in class_w) else None
    loss_type = masked_ce(heads["action_type_logits"], t_type, 5,   w_type)
    loss_btn  = masked_ce(heads["button_logits"],      t_btn,  4,   w_btn)
    loss_key  = masked_ce(heads["key_logits"],         t_key,  151, w_key)
    # V1 doesn't have scroll_x_logits - scroll_x is always 0 in legacy format
    loss_sx   = torch.tensor(0.0, device=heads["time"].device)
    loss_sy   = masked_ce(heads["scroll_y_logits"],    t_sy,   3,   w_sy)
    # Optional term weights
    lw = {"time":0.1, "x":1.0, "y":1.0, "type":1.0, "btn":1.0, "key":1.0, "sx":1.0, "sy":1.0}
    total = (lw["time"]*loss_time + lw["x"]*loss_x + lw["y"]*loss_y +
             lw["type"]*loss_type + lw["btn"]*loss_btn + lw["key"]*loss_key +
             lw["sx"]*loss_sx + lw["sy"]*loss_sy)
    logs = dict(loss_time=float(loss_time.detach().cpu()),
                loss_x=float(loss_x.detach().cpu()),
                loss_y=float(loss_y.detach().cpu()),
                loss_type=float(loss_type.detach().cpu()),
                loss_button=float(loss_btn.detach().cpu()),
                loss_key=float(loss_key.detach().cpu()),
                loss_scroll_x=float(loss_sx.detach().cpu()),
                loss_scroll_y=float(loss_sy.detach().cpu()))
    return total, logs

def compute_loss_v2(heads: Dict[str, torch.Tensor], target_v2: torch.Tensor, valid_mask: torch.Tensor,
                    class_w: Dict[str, torch.Tensor] | None = None,
                    loss_weights: Dict[str, float] | None = None,
                    use_log1p_time: bool = True, time_div_ms: float = 1000.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V2 target layout: [time_s, x, y, button(4), key_action(3), key_id(151), scroll_y(3)]
    valid_mask: (B,A) True where row is a real event; padding rows are False.
    """
    B, A, _ = target_v2.shape
    m = valid_mask.bool()
    mflat = m.reshape(-1)
    # unpack
    t_time  = target_v2[..., 0]
    t_x     = target_v2[..., 1]
    t_y     = target_v2[..., 2]
    t_btn   = target_v2[..., 3].long()
    t_ka    = target_v2[..., 4].long()
    t_kid   = target_v2[..., 5].long()
    t_sy    = target_v2[..., 6].long()  # already encoded 0=down,1=none,2=up
    # time transform
    if use_log1p_time:
        t_time = torch.log1p(t_time.clamp_min(0) * (1000.0 / time_div_ms))
    # --- regression (masked SmoothL1) ---
    def masked_l1(pred, tgt, mask2d):
        return F.smooth_l1_loss(pred[mask2d], tgt[mask2d]) if mask2d.any() else pred.sum()*0
    loss_time = masked_l1(heads["time"], t_time, m)
    loss_x    = masked_l1(heads["x"], t_x, m)
    loss_y    = masked_l1(heads["y"], t_y, m)
    # --- classification (masked CE) ---
    def masked_ce(logits, tgt, mask2d, w=None):
        if not mask2d.any():
            return logits.sum()*0
        logits2 = logits.reshape(B*A, -1)[mflat]
        tgt2 = tgt.reshape(B*A)[mflat]
        return F.cross_entropy(logits2, tgt2, weight=w)
    w_btn = class_w.get("button") if class_w else None
    w_ka  = class_w.get("key_action") if class_w else None
    w_kid = class_w.get("key_id") if class_w else None
    w_sy  = class_w.get("scroll_y") if class_w else None
    loss_btn = masked_ce(heads["button_logits"], t_btn, m, w_btn)
    loss_ka  = masked_ce(heads["key_action_logits"], t_ka, m, w_ka)
    # key_id is only defined when key_action>0
    m_kid = m & (t_ka > 0)
    loss_kid = masked_ce(heads["key_id_logits"], t_kid, m_kid, w_kid)
    loss_sy  = masked_ce(heads["scroll_y_logits"], t_sy, m, w_sy)
    # weights (configurable from args, with sensible defaults)
    lw = loss_weights or {"time": 0.3, "x": 1.0, "y": 1.0, "button": 1.0, "key_action": 1.0, "key_id": 1.0, "scroll_y": 1.0}
    total = (lw["time"]*loss_time + lw["x"]*loss_x + lw["y"]*loss_y +
             lw["button"]*loss_btn + lw["key_action"]*loss_ka +
             lw["key_id"]*loss_kid + lw["scroll_y"]*loss_sy)
    logs = dict(loss_time=float(loss_time.detach().cpu()),
                loss_x=float(loss_x.detach().cpu()),
                loss_y=float(loss_y.detach().cpu()),
                loss_button=float(loss_btn.detach().cpu()),
                loss_key_action=float(loss_ka.detach().cpu()),
                loss_key_id=float(loss_kid.detach().cpu()),
                loss_scroll_y=float(loss_sy.detach().cpu()))
    return total, logs



def train(model,
          train_loader: DataLoader,
          num_epochs: int = 10,
          optimizer: torch.optim.Optimizer = None,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          device: torch.device = torch.device("cpu"),
          grad_clip: float = 1.0,
          loss_weights: Dict[str,float] = None,
          targets_version: str = "v1",
          use_log1p_time: bool = True):
    """Train the model with V1/V2 target support"""
    
    print(f"Starting training on {device}")
    print(f"Targets version: {targets_version}")
    
    history = {"train_losses": [], "val_losses": []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            temporal = batch["temporal_sequence"].to(device)
            actions  = batch["action_sequence"].to(device)
            target   = batch["action_target"].to(device)
            vmask    = batch.get("valid_mask", None)
            if vmask is not None: vmask = vmask.to(device)

            optimizer.zero_grad()
            out = model(temporal, actions, return_logits=True)
            if targets_version == "v2":
                loss_result = compute_loss_v2(out, target, vmask if vmask is not None else _allzero_mask(target),
                                       class_w=None, loss_weights=loss_weights,
                                       time_div_ms=1000.0, use_log1p_time=use_log1p_time)
                loss, logs = loss_result
            else:
                loss_result = compute_loss_v1(out, target, vmask if vmask is not None else _allzero_mask(target),
                                       class_w=None,
                                       time_div_ms=1000.0, use_log1p_time=use_log1p_time)
                loss, logs = loss_result
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)
        print(f"  Train Loss: {avg_train_loss:.4f}")
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        print("-" * 40)
    
    return history

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

def create_data_loaders_v2(data_dir: Path, targets_version: str = None, device=None):
    man = load_manifest(data_dir)
    tv = targets_version or (man["targets_version"] if man else "v1")
    # load arrays
    gs = np.load(data_dir/"gamestate_sequences.npy")      # (N,T,G)
    ai = np.load(data_dir/"action_input_sequences.npy")   # (N,T,A,Fin)
    if tv == "v2":
        at = np.load(data_dir/"actions_v2.npy")           # (N,A,7)
        vm = np.load(data_dir/"valid_mask.npy")           # (N,A)
    else:
        at = np.load(data_dir/"action_targets.npy")       # (N,A,8)
        vm = (np.abs(at).sum(axis=-1) > 0)
    
    # OSRSDataset expects a data_dir; let it discover file paths itself.
    dataset = OSRSDataset(
        data_dir=data_dir,
        targets_version=targets_version
    )
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

def setup_model_v2(manifest, targets_version, device):
    max_actions = int(manifest["max_actions"]) if manifest else 100
    head_version = targets_version
    enum_sizes = manifest["enums"] if manifest else {
        # safe fallbacks if training V1 without manifest
        "action_type":{"size":5}, "button":{"size":4}, "key_id":{"size":151}, "scroll_y":{"size":3}
    }
    # dims inferred from data (not constants)
    seq_len, gamestate_dim = 10, 128
    action_in_dim = 8
    model = ImitationHybridModel(
        gamestate_dim=gamestate_dim,
        action_dim=action_in_dim,
        sequence_length=seq_len,
        hidden_dim=256,
        num_attention_heads=8,
        max_actions=max_actions, 
        head_version=head_version, 
        enum_sizes=enum_sizes
    )
    return model

def estimate_class_weights(train_loader, targets_version="v2", enum_sizes=None):
    """
    Build inverse-frequency class weights with dynamic lengths.
    """
    if targets_version == "v1":
        n_type = int(enum_sizes.get("action_type",{}).get("size",5))
        n_btn  = int(enum_sizes.get("button",{}).get("size",4))
        n_key  = int(enum_sizes.get("key_id",{}).get("size",151))
        n_sy   = int(enum_sizes.get("scroll_y",{}).get("size",3))
        type_counts = torch.ones(n_type)
        btn_counts  = torch.ones(n_btn)
        key_counts  = torch.ones(n_key)
        sy_counts   = torch.ones(n_sy)
        for b in train_loader:
            tgt = b["action_target"]
            m   = b["valid_mask"]
            type_counts.index_add_(0, tgt[...,1][m].long(), torch.ones_like(tgt[...,1][m], dtype=torch.float))
            btn_counts.index_add_(0,  tgt[...,4][m].long().clamp(0,n_btn-1), torch.ones_like(tgt[...,4][m], dtype=torch.float))
            key_counts.index_add_(0,  tgt[...,5][m].long().clamp(0,n_key-1), torch.ones_like(tgt[...,4][m], dtype=torch.float))
            sy_counts.index_add_(0,   (tgt[...,7].long()+1).clamp(0,n_sy-1), torch.ones_like(tgt[...,7][m], dtype=torch.float))
        inv = lambda c: (1.0/(c+1e-9)); norm = lambda w: (w/w.mean())
        return {"type": norm(inv(type_counts)), "btn": norm(inv(btn_counts)),
                "key": norm(inv(key_counts)), "sy": norm(inv(sy_counts))}
    else:
        n_btn  = int(enum_sizes["button"]["size"])
        n_ka   = int(enum_sizes["key_action"]["size"])
        n_kid  = int(enum_sizes["key_id"]["size"])
        n_sy   = int(enum_sizes["scroll_y"]["size"])
        btn_counts = torch.ones(n_btn); ka_counts = torch.ones(n_ka)
        kid_counts = torch.ones(n_kid); sy_counts = torch.ones(n_sy)
        for b in train_loader:
            tgt = b["action_target"]; m = b["valid_mask"]
            btn_counts.index_add_(0, tgt[...,3][m].long(), torch.ones_like(tgt[...,3][m], dtype=torch.float))
            ka_counts.index_add_(0,  tgt[...,4][m].long(), torch.ones_like(tgt[...,4][m], dtype=torch.float))
            kid_counts.index_add_(0, tgt[...,5][m].long(), torch.ones_like(tgt[...,5][m], dtype=torch.float))
            sy_counts.index_add_(0,  tgt[...,6][m].long(), torch.ones_like(tgt[...,6][m], dtype=torch.float))
        inv = lambda c: (1.0/(c+1e-9)); norm = lambda w: (w/w.mean())
        return {"btn": norm(inv(btn_counts)), "ka": norm(inv(ka_counts)),
                "kid": norm(inv(kid_counts)), "sy": norm(inv(sy_counts))}

def clamp_time(t, time_div, time_clip, already_scaled=False):
    if already_scaled:  # V2 saved time already scaled
        return torch.clamp(t, 0.0, float(time_clip))
    return torch.clamp(t / float(time_div), 0.0, float(time_clip))

def compute_il_loss_v1(heads, target, class_w, loss_w, time_div, time_clip, enum_sizes):
    B, A, _ = target.shape
    mask = (target.abs().sum(dim=-1) > 0)
    m = mask.view(-1)
    n_type = heads["action_type_logits"].shape[-1]
    n_btn  = heads["button_logits"].shape[-1]
    n_key  = heads["key_logits"].shape[-1]
    n_sy   = heads["scroll_y_logits"].shape[-1]
    tt = clamp_time(target[...,0], time_div, time_clip, already_scaled=False)
    l_time = F.smooth_l1_loss(heads["time"][m], tt[m])
    l_x    = F.smooth_l1_loss(heads["x"][m],    target[...,2][m])
    l_y    = F.smooth_l1_loss(heads["y"][m],    target[...,3][m])
    l_type = F.cross_entropy(heads["action_type_logits"].view(-1,n_type)[m], target[...,1].long().clamp(0,n_type-1)[m], weight=class_w["type"])
    l_btn  = F.cross_entropy(heads["button_logits"].view(-1,n_btn)[m],      target[...,4].long().clamp(0,n_btn-1)[m],  weight=class_w["btn"])
    l_key  = F.cross_entropy(heads["key_logits"].view(-1,n_key)[m],         target[...,5].long().clamp(0,n_key-1)[m],  weight=class_w["key"])
    l_sy   = F.cross_entropy(heads["scroll_y_logits"].view(-1,n_sy)[m],      (target[...,7].long()+enum_sizes["scroll_y"].get("none_index",1)).clamp(0,n_sy-1)[m], weight=class_w["sy"])
    return (loss_w["time"]*l_time + loss_w["x"]*l_x + loss_w["y"]*l_y +
            loss_w["type"]*l_type + loss_w["btn"]*l_btn + loss_w["key"]*l_key + loss_w["sy"]*l_sy)

def compute_il_loss_v2(heads, target, valid_mask, class_w, loss_w, time_div, time_clip, enum_sizes):
    B, A, _ = target.shape
    m = valid_mask.view(-1)
    n_btn = heads["button_logits"].shape[-1]
    n_ka  = heads["key_action_logits"].shape[-1]
    n_kid = heads["key_id_logits"].shape[-1]
    n_sy  = heads["scroll_y_logits"].shape[-1]
    tt = clamp_time(target[...,0], time_div, time_clip, already_scaled=True)
    l_time = F.smooth_l1_loss(heads["time"][m], tt[m])
    l_x    = F.smooth_l1_loss(heads["x"][m],    target[...,1][m])
    l_y    = F.smooth_l1_loss(heads["y"][m],    target[...,2][m])
    l_btn  = F.cross_entropy(heads["button_logits"].view(-1,n_btn)[m],   target[...,3].long().clamp(0,n_btn-1)[m], weight=class_w["btn"])
    l_ka   = F.cross_entropy(heads["key_action_logits"].view(-1,n_ka)[m], target[...,4].long().clamp(0,n_ka-1)[m], weight=class_w["ka"])
    ka_mask = (target[...,4].view(-1) != enum_sizes["key_action"]["none_index"])
    if ka_mask.any():
        l_kid = F.cross_entropy(heads["key_id_logits"].view(-1,n_kid)[m][ka_mask], target[...,5].long().clamp(0,n_kid-1)[m][ka_mask], weight=class_w["kid"])
    else:
        l_kid = torch.tensor(0.0, device=target.device)
    l_sy   = F.cross_entropy(heads["scroll_y_logits"].view(-1,n_sy)[m],   target[...,6].long().clamp(0,n_sy-1)[m], weight=class_w["sy"])
    return (loss_w["time"]*l_time + loss_w["x"]*l_x + loss_w["y"]*l_y +
            loss_w["btn"]*l_btn + loss_w["ka"]*l_ka + loss_w["kid"]*l_kid + loss_w["sy"]*l_sy)

def _compute_class_weights_from_loader(train_loader, device):
    """
    Estimate inverse-frequency class weights under the valid-action mask.
    Returns tensors for: type(5), btn(4), key(151), sx(3), sy(3).
    """
    cnt_type, cnt_btn, cnt_key, cnt_sx, cnt_sy = Counter(), Counter(), Counter(), Counter(), Counter()
    total_type = total_btn = total_key = total_sx = total_sy = 0
    with torch.no_grad():
        for batch in train_loader:
            tgt = batch["action_target"].to(device)
            mask = _build_valid_mask(tgt).reshape(-1)
            def _count(col, C, counter):
                vals = col.reshape(-1)[mask].long().tolist()
                counter.update(vals)
                return len(vals)
            total_type += _count(tgt[...,1].clamp(0,4),   5,   cnt_type)
            total_btn  += _count(tgt[...,4].clamp(0,3),   4,   cnt_btn)
            total_key  += _count(tgt[...,5].clamp(0,150), 151, cnt_key)
            total_sx   += _count(_scroll_to_index(tgt[...,6]), 3, cnt_sx)
            total_sy   += _count(_scroll_to_index(tgt[...,7]), 3, cnt_sy)
    def _to_weights(counter, C):
        freq = torch.zeros(C, dtype=torch.float)
        for k,v in counter.items():
            if 0 <= k < C:
                freq[k] = v
        freq = freq + 1e-6
        w = 1.0 / freq
        w = w * (C / w.sum())  # normalize to mean ~1
        return w
    return {
        "type": _to_weights(cnt_type, 5).to(device),
        "btn":  _to_weights(cnt_btn,  4).to(device),
        "key":  _to_weights(cnt_key,  151).to(device),
        "sx":   _to_weights(cnt_sx,   3).to(device),
        "sy":   _to_weights(cnt_sy,   3).to(device),
    }

def compute_il_loss(heads, target, class_w=None, loss_w=None, time_div: float = 1.0) -> torch.Tensor:
    """
    heads: dict from ilbot.model(return_logits=True)
      - time: (B,100), x: (B,100), y:(B,100)
      - action_type_logits: (B,100,5)
      - button_logits: (B,100,4)
      - key_logits: (B,100,151)
      - scroll_x_logits/scroll_y_logits: (B,100,3)
    target: (B,100,8)  [time, type, x, y, button, key, scroll_dx, scroll_dy]
    """
    B, A, _ = target.shape
    mask = _build_valid_mask(target)                       # (B,100)
    mflat = mask.view(-1)

    # Targets
    t_time = target[..., 0]                          # (B,100)
    t_type = target[..., 1].long().clamp(0, 4)
    t_x    = target[..., 2]
    t_y    = target[..., 3]
    t_btn  = target[..., 4].long().clamp(0, 3)
    t_key  = target[..., 5].long().clamp(0, 150)
    t_sx   = _scroll_to_index(target[..., 6])
    t_sy   = _scroll_to_index(target[..., 7])

    # Regression losses (masked SmoothL1)
    def masked_l1(pred, tgt):
        p = pred.reshape(-1)[mflat]; t = tgt.reshape(-1)[mflat]
        return F.smooth_l1_loss(p, t, reduction="mean") if p.numel() else torch.zeros([], device=pred.device)
    # Masked CE helper (optionally weighted)
    def masked_ce(logits, tgt, C, w=None):
        L = logits.reshape(-1, C)[mflat]
        T = tgt.reshape(-1)[mflat]
        if L.numel() == 0:
            return torch.zeros([], device=logits.device)
        if w is not None and w.device != L.device:
            w = w.to(L.device)
        return F.cross_entropy(L, T, weight=w, reduction="mean")

    # Per-head losses (now actually defined)
    # Normalize time inside the loss only (e.g., ms->s with time_div=1000)
    if time_div != 1.0:
        loss_time = masked_l1(heads["time"] / time_div, t_time / time_div)
    else:
        loss_time = masked_l1(heads["time"], t_time)
    loss_x    = masked_l1(heads["x"],    t_x)
    loss_y    = masked_l1(heads["y"],    t_y)

    w_type = class_w["type"] if (class_w and "type" in class_w) else None
    w_btn  = class_w["btn"]  if (class_w and "btn"  in class_w) else None
    w_key  = class_w["key"]  if (class_w and "key"  in class_w) else None
    w_sx   = class_w["sx"]   if (class_w and "sx"   in class_w) else None
    w_sy   = class_w["sy"]   if (class_w and "sy"   in class_w) else None

    loss_type = masked_ce(heads["action_type_logits"], t_type, 5,   w_type)
    loss_btn  = masked_ce(heads["button_logits"],      t_btn,  4,   w_btn)
    loss_key  = masked_ce(heads["key_logits"],         t_key,  151, w_key)
    loss_sx   = masked_ce(heads["scroll_x_logits"],    t_sx,   3,   w_sx)
    loss_sy   = masked_ce(heads["scroll_y_logits"],    t_sy,   3,   w_sy)

    # Optional term weights (give coords more bite; deprioritize time a bit)
    lw = {"time":0.1, "x":1.0, "y":1.0, "type":1.0, "btn":1.0, "key":1.0, "sx":1.0, "sy":1.0}
    if loss_w: lw.update(loss_w)
    return (lw["time"]*loss_time + lw["x"]*loss_x + lw["y"]*loss_y +
            lw["type"]*loss_type + lw["btn"]*loss_btn + lw["key"]*loss_key +
            lw["sx"]*loss_sx + lw["sy"]*loss_sy)

def _masked_metrics(heads: dict, target: torch.Tensor, targets_version="v1"):
    """Simple masked metrics to track IL progress."""
    with torch.no_grad():
        if targets_version == "v1":
            mask = _build_valid_mask(target)
            mflat = mask.reshape(-1)
            # preds
            at = heads["action_type_logits"].argmax(-1)
            bt = heads["button_logits"].argmax(-1)
            ky = heads["key_logits"].argmax(-1)
            sx = heads["scroll_x_logits"].argmax(-1) - 1
            sy = heads["scroll_y_logits"].argmax(-1) - 1
            # tgts
            tt = target[...,1].long().clamp(0,4)
            tb = target[...,4].long().clamp(0,3)
            tk = target[...,5].long().clamp(0,150)
            tsx= target[...,6].long().clamp(-1,1)
            tsy= target[...,7].long().clamp(-1,1)
            def m_acc(pred, tgt):
                p = pred.reshape(-1)[mflat]; t = tgt.reshape(-1)[mflat]
                return (p == t).float().mean().item() if p.numel() else 0.0
            def m_mae(pred, tgt):
                p = pred.reshape(-1)[mflat].float(); t = tgt.reshape(-1)[mflat].float()
                return (p - t).abs().mean().item() if p.numel() else 0.0
            valid_frac = mflat.float().mean().item() if mflat.numel() else 0.0
            return {
                "acc_type": m_acc(at, tt),
                "acc_button": m_acc(bt, tb),
                "acc_key": m_acc(ky, tk),
                "acc_scroll_dx": m_acc(sx, tsx),
                "acc_scroll_dy": m_acc(sy, tsy),
                "mae_time": m_mae(heads["time"], target[...,0]),
                "mae_x":    m_mae(heads["x"],    target[...,2]),
                "mae_y":    m_mae(heads["y"],    target[...,3]),
                "valid_frac": valid_frac,
            }
        else:
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
                "mae_time": m_mae(heads["time"], target[...,0]),
                "mae_x":    m_mae(heads["x"],    target[...,1]),
                "mae_y":    m_mae(heads["y"],    target[...,2]),
                "valid_frac": valid_frac,
            }

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cpu', scheduler=None,
                use_class_weights=True, loss_w=None, time_div=1.0,
                targets_version="v1", time_clip=3.0, enum_sizes=None):
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
            class_w = _compute_class_weights_from_loader(train_loader, device)
            print("  Using legacy class weights")
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
            if isinstance(outputs, dict):
                # Use V1/V2 loss functions based on targets version
                if targets_version == "v1":
                    loss = compute_il_loss_v1(outputs, action_target, class_w, loss_w, time_div, time_clip, enum_sizes)
                else:
                    # Compute valid mask from action_target for V2
                    valid_mask = (torch.abs(action_target).sum(dim=-1) > 0)
                    loss = compute_il_loss_v2(outputs, action_target, valid_mask, class_w, loss_w, time_div, time_clip, enum_sizes)
            else:
                # Back-compat with legacy tensor output + custom criterion
                loss = criterion(outputs, action_target)
            
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
        if targets_version == "v1":
            vm = {"acc_type":0.0,"acc_button":0.0,"acc_key":0.0,"acc_scroll_dx":0.0,"acc_scroll_dy":0.0,
                  "mae_time":0.0,"mae_x":0.0,"mae_y":0.0,"valid_frac":0.0}
        else:
            vm = {"acc_button":0.0,"acc_key_action":0.0,"acc_key_id":0.0,"acc_scroll_y":0.0,
                  "mae_time":0.0,"mae_x":0.0,"mae_y":0.0,"valid_frac":0.0}
        vm_n = 0
        with torch.no_grad():
            for batch in val_loader:
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                outputs = model(temporal_sequence, action_sequence, return_logits=True)
                if isinstance(outputs, dict):
                    # Use V1/V2 loss functions based on targets version
                    if targets_version == "v1":
                        vloss = compute_il_loss_v1(outputs, action_target, class_w, loss_w, time_div, time_clip, enum_sizes)
                    else:
                        # Compute valid mask from action_target for V2
                        valid_mask = (torch.abs(action_target).sum(dim=-1) > 0)
                        vloss = compute_il_loss_v2(outputs, action_target, valid_mask, class_w, loss_w, time_div, time_clip, enum_sizes)
                    mm = _masked_metrics(outputs, action_target, targets_version)
                    for k in vm: vm[k] += mm[k]
                    vm_n += 1
                else:
                    vloss = criterion(outputs, action_target)
                val_loss_sum += float(vloss.item())
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        if vm_n > 0:
            for k in vm: vm[k] /= vm_n
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            if targets_version == "v1":
                print(f"  Val metrics (masked avg over {vm_n} batches): "
                      f"acc_type={vm['acc_type']:.3f} acc_btn={vm['acc_button']:.3f} acc_key={vm['acc_key']:.3f} "
                      f"acc_sx={vm['acc_scroll_dx']:.3f} acc_sy={vm['acc_scroll_dy']:.3f} "
                      f"mae_time={vm['mae_time']:.1f} mae_x={vm['mae_x']:.1f} mae_y={vm['mae_y']:.1f} "
                      f"| valid_frac={vm['valid_frac']:.3f}")
            else:
                print(f"  Val metrics (masked avg over {vm_n} batches): "
                      f"acc_btn={vm['acc_button']:.3f} acc_ka={vm['acc_key_action']:.3f} acc_kid={vm['acc_key_id']:.3f} "
                      f"acc_sy={vm['acc_scroll_y']:.3f} "
                      f"mae_time={vm['mae_time']:.1f} mae_x={vm['mae_x']:.1f} mae_y={vm['mae_y']:.1f} "
                      f"| valid_frac={vm['valid_frac']:.3f}")
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
    parser.add_argument("--targets_version", choices=["v1","v2"], default=None,
                        help="Override dataset_manifest.json; if None, use manifest.")
    parser.add_argument("--time_clip", type=float, default=None,
                        help="Override time_clip; if None, use manifest.")
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    train_loader, val_loader, manifest, tv = create_data_loaders_v2(data_dir, targets_version=args.targets_version, device=device)
    enum_sizes = (manifest or {}).get("enums", {})
    time_div = manifest["time_div"] if (manifest and args.time_div is None) else (args.time_div or 1000.0)
    time_clip = manifest["time_clip"] if (manifest and args.time_clip is None) else (args.time_clip or 3.0)
    
    # Create model and setup training components
    model = setup_model_v2(manifest=manifest or {}, targets_version=tv, device=device)
    
    # Estimate class weights
    class_w = estimate_class_weights(train_loader, targets_version=tv, enum_sizes=enum_sizes)
    
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
        'use_log1p_time': config.get("use_log1p_time", True),
        'time_div_ms': config.get("time_div_ms", 1000.0)
    })

    # 3) optimizer/scheduler
    lr = config.get("lr", 2.5e-4)
    wd = config.get("weight_decay", 1e-4)
    step_size = config.get("step_size", 8)
    gamma = config.get("gamma", 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Move model to device
    device = torch.device(config.get("device", "cuda"))
    model = model.to(device)

    # 4) train loop
    history = train(
        model=model,
        train_loader=train_loader,
        num_epochs=config.get("epochs", 40),
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get("device", "cuda"),
        grad_clip=config.get("grad_clip", 1.0),
        loss_weights=config.get("loss_weights", {}),
        targets_version=config.get("targets_version", "v1"),
        use_log1p_time=config.get("use_log1p_time", True),
    )
    return history

if __name__ == "__main__":
    main()
