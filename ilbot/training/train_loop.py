#!/usr/bin/env python3
"""
Training script for OSRS Imitation Learning Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import math
import json
import time
import argparse
from pathlib import Path
from ilbot.training.setup import create_data_loaders, setup_model, setup_training, OSRSDataset
from ilbot.model.imitation_hybrid_model import ImitationHybridModel
from torch.optim.lr_scheduler import StepLR
import os, numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from ilbot.utils.feature_spec import load_feature_spec

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
    return model

def estimate_class_weights(train_loader, targets_version="v2", enum_sizes=None):
    """
    Build inverse-frequency class weights with dynamic lengths.
    """
    # V2 only - new format: just the integer sizes
    n_btn  = int(enum_sizes["button"])
    n_ka   = int(enum_sizes["key_action"])
    n_kid  = int(enum_sizes["key_id"])
    n_sy   = int(enum_sizes["scroll"])
    none_idx_ka = 0  # Default none index for key_action
    none_idx_sy = 1  # Default none index for scroll_y
    
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
        ka_flat = tgt[...,4].long().view(-1)
        kid_idx = tgt[...,5].long().clamp(0, n_kid-1).view(-1)
        m_flat = m.view(-1)
        kid_mask = (ka_flat != none_idx_ka) & m_flat
        if kid_mask.any():
            kid_counts.index_add_(0, kid_idx[kid_mask], one(kid_idx[kid_mask]))

        # scroll_y: map {-1,0,+1} → {0,1,2}
        sy_idx = (tgt[...,6].long() + none_idx_sy).clamp(0, n_sy-1).view(-1)
        sy_counts.index_add_(0, sy_idx[m_flat], one(sy_idx[m_flat]))
    
    inv = lambda c: (1.0/(c+1e-9)); norm = lambda w: (w/w.mean())
    return {"btn": norm(inv(btn_counts)), "ka": norm(inv(ka_counts)),
            "kid": norm(inv(kid_counts)), "sy": norm(inv(sy_counts))}

def clamp_time(t, time_div, time_clip, already_scaled=False):
    if already_scaled:  # V2 saved time already scaled
        return torch.clamp(t, 0.0, float(time_clip))
    return torch.clamp(t / float(time_div), 0.0, float(time_clip))

# ---------------------------
# Validation metrics helpers
# ---------------------------
def _init_val_agg(enum_sizes: Dict[str, int]) -> Dict:
    def z(n): return torch.zeros(int(n), dtype=torch.long)
    return {
        "btn_pred": z(enum_sizes["button"]),
        "btn_tgt":  z(enum_sizes["button"]),
        "ka_pred":  z(enum_sizes["key_action"]),
        "ka_tgt":   z(enum_sizes["key_action"]),
        "kid_pred": z(enum_sizes["key_id"]),
        "kid_tgt":  z(enum_sizes["key_id"]),
        "sy_pred":  z(enum_sizes["scroll"]),
        "sy_tgt":   z(enum_sizes["scroll"]),
        "time_pred_sum": 0.0,
        "time_tgt_sum":  0.0,
        "time_count":    0,
        "time_pred_neg": 0,
        # event-type summaries
        "evt_pred": {"MOVE":0, "CLICK":0, "KEY":0, "SCROLL":0, "MULTI":0},
        "evt_tgt":  {"MOVE":0, "CLICK":0, "KEY":0, "SCROLL":0},
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
    n_btn = int(enum_sizes["button"])
    n_ka  = int(enum_sizes["key_action"])
    n_kid = int(enum_sizes["key_id"])
    n_sy  = int(enum_sizes["scroll"])
    btn_none = 0  # Default none index for button
    ka_none  = 0  # Default none index for key_action
    sy_none  = 1  # Default none index for scroll_y

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

    # --- scroll_y mapping: raw {-1,0,+1} -> idx {0,1,2} (none=center) ---
    sy_raw = fl(target[...,6]).long()
    sy_tgt = (sy_raw + sy_none).clamp(0, n_sy-1)
    sy_logits = heads["scroll_y_logits"].view(-1, n_sy)
    sy_pred = sy_logits.argmax(-1)
    agg["sy_pred"] += torch.bincount(sy_pred[m], minlength=n_sy).cpu()
    agg["sy_tgt"]  += torch.bincount(sy_tgt[m],  minlength=n_sy).cpu()

    # --- time stats on masked rows ---
    t_pred = fl(heads["time"])
    t_tgt  = fl(clamp_time(target[...,0], time_div, time_clip, already_scaled=True))
    t_pred_m = t_pred[m]
    t_tgt_m  = t_tgt[m]
    agg["time_pred_sum"] += float(t_pred_m.sum().item())
    agg["time_tgt_sum"]  += float(t_tgt_m.sum().item())
    agg["time_count"]    += int(t_pred_m.numel())
    agg["time_pred_neg"] += int((t_pred_m < 0).sum().item())

    # --- event-type summaries & contradictions (pred vs tgt) ---
    # Predicted non-NONE flags
    p_btn = (btn_pred != btn_none)
    p_ka  = (ka_pred  != ka_none)
    p_sy  = (sy_pred  != sy_none)
    p_btn = p_btn[m]; p_ka = p_ka[m]; p_sy = p_sy[m]
    # Target non-NONE flags
    t_btn = (btn_tgt != btn_none)[m]
    t_ka  = (ka_tgt  != ka_none)[m]
    t_sy  = (sy_tgt  != sy_none)[m]
    # Pred event type (MOVE if none fire; MULTI if more than one fire)
    nxt = p_btn.to(torch.int) + p_ka.to(torch.int) + p_sy.to(torch.int)
    if m.any():
        # counts
        agg["evt_pred"]["MULTI"] += int((nxt > 1).sum().item())
        # single-type
        only_btn = (nxt == 1) & p_btn
        only_ka  = (nxt == 1) & p_ka
        only_sy  = (nxt == 1) & p_sy
        only_mv  = (nxt == 0)
        agg["evt_pred"]["CLICK"]  += int(only_btn.sum().item())
        agg["evt_pred"]["KEY"]    += int(only_ka.sum().item())
        agg["evt_pred"]["SCROLL"] += int(only_sy.sum().item())
        agg["evt_pred"]["MOVE"]   += int(only_mv.sum().item())
        # target event type
        t_is = t_btn.to(torch.int) + t_ka.to(torch.int) + t_sy.to(torch.int)
        agg["evt_tgt"]["SCROLL"] += int((t_sy & m[m]).sum().item())  # m[m] is all True; kept for clarity
        agg["evt_tgt"]["CLICK"]  += int(((~t_sy) & t_btn).sum().item())
        agg["evt_tgt"]["KEY"]    += int(((~t_sy) & (~t_btn) & t_ka).sum().item())
        agg["evt_tgt"]["MOVE"]   += int((t_is == 0).sum().item())

def _topk_from_counts(counts: torch.Tensor, k: int = 10):
    arr = counts.tolist()
    order = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
    out = [(i, arr[i]) for i in order[:k] if arr[i] > 0]
    return out

def _print_val_agg(agg: Dict, enum_sizes: Dict, time_div: float):
    def pct(cnt, total): return (100.0*cnt/total) if total > 0 else 0.0
    print("=== Detailed validation report ===")
    total_masked = int(agg["time_count"])
    print(f"masked rows used: {total_masked}")
    # Button
    btn_tot = int(agg["btn_pred"].sum().item())
    print(f"[button] pred total={btn_tot}, top5={_topk_from_counts(agg['btn_pred'],5)}, tgt top5={_topk_from_counts(agg['btn_tgt'],5)}")
    # Key action
    ka_tot = int(agg["ka_pred"].sum().item())
    print(f"[key_action] pred total={ka_tot}, top5={_topk_from_counts(agg['ka_pred'],5)}, tgt top5={_topk_from_counts(agg['ka_tgt'],5)}")
    # Key id
    kid_tot = int(agg["kid_pred"].sum().item())
    print(f"[key_id] pred total={kid_tot}, top10={_topk_from_counts(agg['kid_pred'],10)}")
    # Scroll
    sy_tot = int(agg["sy_pred"].sum().item())
    print(f"[scroll_y] pred total={sy_tot}, counts={_topk_from_counts(agg['sy_pred'],3)} (idx order)")
    # Time
    if total_masked > 0:
        mean_pred = agg["time_pred_sum"]/total_masked
        mean_tgt  = agg["time_tgt_sum"]/total_masked
        print(f"[time] mean_pred={mean_pred:.4f} ({mean_pred*time_div:.1f} ms) | mean_tgt={mean_tgt:.4f} ({mean_tgt*time_div:.1f} ms) "
              f"| neg_pred_frac={pct(agg['time_pred_neg'], total_masked):.2f}%")
    # Event types & contradictions
    ep = agg["evt_pred"]; et = agg["evt_tgt"]
    pred_sum = sum(ep.values())
    tgt_sum  = sum(et.values())
    print(f"[event(pred)] MOVE={ep['MOVE']} ({pct(ep['MOVE'],pred_sum):.1f}%), "
          f"CLICK={ep['CLICK']} ({pct(ep['CLICK'],pred_sum):.1f}%), "
          f"KEY={ep['KEY']} ({pct(ep['KEY'],pred_sum):.1f}%), "
          f"SCROLL={ep['SCROLL']} ({pct(ep['SCROLL'],pred_sum):.1f}%), "
          f"MULTI={ep['MULTI']} ({pct(ep['MULTI'],pred_sum):.1f}%)")
    print(f"[event(tgt)]  MOVE={et['MOVE']} ({pct(et['MOVE'],tgt_sum):.1f}%), "
          f"CLICK={et['CLICK']} ({pct(et['CLICK'],tgt_sum):.1f}%), "
          f"KEY={et['KEY']} ({pct(et['KEY'],tgt_sum):.1f}%), "
          f"SCROLL={et['SCROLL']} ({pct(et['SCROLL'],tgt_sum):.1f}%)")
    print("=== End detailed report ===")

# (removed: legacy V1 IL loss)

import torch
import torch.nn.functional as F

def compute_unified_event_loss(predictions, targets, valid_mask, loss_fn, enum_sizes):
    """
    Compute unified event system loss using the new UnifiedEventLoss.
    
    Args:
        predictions: Model outputs from unified event system
        targets: [B, A, 7] V2 action targets [time, x, y, button, key_action, key_id, scroll_y]
        valid_mask: [B, A] Boolean mask for valid actions
        loss_fn: UnifiedEventLoss instance
        enum_sizes: Dictionary with categorical sizes for auxiliary losses
    
    Returns:
        total_loss: Combined weighted loss
        loss_components: Dictionary of individual loss components
    """

    # Ensure valid_mask is 2D boolean
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.view(targets.shape[0], targets.shape[1])
    valid_mask = valid_mask.bool()
    
    # Compute loss using UnifiedEventLoss
    total_loss, loss_components = loss_fn(predictions, targets, valid_mask, enum_sizes)
    
    return total_loss, loss_components

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
            "mae_time": m_mae(heads["time"], target[...,0]),
            "mae_x":    m_mae(heads["x"],    target[...,1]),
            "mae_y":    m_mae(heads["y"],    target[...,2]),
            "valid_frac": valid_frac,
        }

def train_model(model, train_loader, val_loader, loss_fn, optimizer,
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
            outputs = model(temporal_sequence, action_sequence)

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

                ka_none = 0  # Default none index for key_action
                ka = action_target[..., 4].long()
                kid_rows = int((vm & (ka != ka_none)).sum().item())

                sy_vals = action_target[..., 6].long()         # {-1,0,+1}
                sy_rows = int(vm.sum().item())                 # all valid rows participate
                sy_events = int((vm & (sy_vals != 0)).sum().item())

                print(f"[dbg] B={B} A={A} valid={total_valid} ({frac:.3f}) | "
                    f"kid_rows={kid_rows} | sy_rows={sy_rows} (nonzero={sy_events})")
            # ---------------------------------------------------------------------------

            
            if isinstance(outputs, dict):
                # V2 only - use unified event loss
                valid_mask = batch['valid_mask'].to(device)
                loss, loss_components = compute_unified_event_loss(outputs, action_target, valid_mask, loss_fn, enum_sizes)
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
                
                outputs = model(temporal_sequence, action_sequence)

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

                    ka_none = 0  # Default none index for key_action
                    ka = action_target[..., 4].long()
                    kid_rows = int((vm_mask & (ka != ka_none)).sum().item())

                    sy_vals = action_target[..., 6].long()   # {-1,0,+1}
                    sy_rows = int(vm_mask.sum().item())      # all valid rows participate
                    sy_events = int((vm_mask & (sy_vals != 0)).sum().item())

                    print(f"[dbg] (val) B={B} A={A} valid={total_valid} ({frac:.3f}) | "
                        f"kid_rows={kid_rows} | sy_rows={sy_rows} (nonzero={sy_events})")


                if isinstance(outputs, dict):
                    # V2 only - use unified event loss
                    valid_mask = batch['valid_mask'].to(device)
                    vloss, loss_components = compute_unified_event_loss(outputs, action_target, valid_mask, loss_fn, enum_sizes)
                    # TODO: Update metrics computation for unified event system
                    mm = _masked_metrics(outputs, action_target, "v2")
                    for k in vm: vm[k] += mm[k]
                    vm_n += 1
                    # accumulate detailed distributions
                    if val_agg is not None:
                        _update_val_agg(val_agg, outputs, action_target, vm_mask, enum_sizes, time_div, time_clip)
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
            # V2 only
            print(f"  Val metrics (masked avg over {vm_n} batches): "
                  f"acc_btn={vm['acc_button']:.3f} acc_ka={vm['acc_key_action']:.3f} acc_kid={vm['acc_key_id']:.3f} "
                  f"acc_sy={vm['acc_scroll_y']:.3f} "
                  f"mae_time={vm['mae_time']:.1f} mae_x={vm['mae_x']:.1f} mae_y={vm['mae_y']:.1f} "
                  f"| valid_frac={vm['valid_frac']:.3f}")
            # print detailed per-head distributions and time stats
            if val_agg is not None:
                _print_val_agg(val_agg, enum_sizes, time_div)
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

    # 1) Use DataInspector to auto-detect configuration first
    from ilbot.model.data_inspector import DataInspector
    
    data_inspector = DataInspector(
        action_input_path=Path(config["data_dir"]) / "action_input_sequences.npy",
        gamestate_path=Path(config["data_dir"]) / "gamestate_sequences.npy",
        action_targets_path=Path(config["data_dir"]) / "action_targets.npy"
    )
    
    data_config = data_inspector.get_model_config()
    print(f"Auto-detected data config: {data_config}")
    
    # Create data loaders using the auto-detected configuration
    from ilbot.training.setup import create_data_loaders
    
    # Create a proper dataset that loads the actual data
    class UnifiedEventDataset:
        def __init__(self, data_dir, data_config):
            self.data_dir = Path(data_dir)
            self.data_config = data_config
            
            # Load the actual data files
            self.gamestate_sequences = np.load(self.data_dir / "gamestate_sequences.npy")
            self.action_input_sequences = np.load(self.data_dir / "action_input_sequences.npy")
            self.action_targets = np.load(self.data_dir / "action_targets.npy")
            
            # Create valid mask (non-zero rows are valid)
            self.valid_mask = (np.abs(self.action_targets).sum(axis=-1) > 0)
            
            # Set dataset properties
            self.B, self.T, self.G = self.gamestate_sequences.shape
            _, _, self.A, self.Fin = self.action_input_sequences.shape
            self.n_sequences = self.B
        
        def __len__(self):
            return self.n_sequences
        
        def __getitem__(self, idx):
            return {
                "temporal_sequence": torch.from_numpy(self.gamestate_sequences[idx]).float(),
                "action_sequence": torch.from_numpy(self.action_input_sequences[idx]).float(),
                "action_target": torch.from_numpy(self.action_targets[idx]).float(),
                "valid_mask": torch.from_numpy(self.valid_mask[idx]).bool()
            }
    
    ds = UnifiedEventDataset(config["data_dir"], data_config)
    train_loader, val_loader = create_data_loaders(
        dataset=ds,
        batch_size=config.get("batch_size", 32),
        disable_cuda_batch_opt=config.get("disable_auto_batch", False),
    )

    # 2) model
    # Build model using auto-detected configuration
    model = create_model(
        data_config=data_config,
        model_config={
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6
        }
    )
    
    # Extract enum_sizes for compatibility
    enum_sizes = data_config['enum_sizes']

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

    # 4) Create unified event loss function
    from ilbot.model.losses import UnifiedEventLoss
    
    # Use the same data_config for the loss function
    # data_config is already created above from DataInspector
    
    loss_fn = UnifiedEventLoss(data_config=data_config)
    print(f"Created UnifiedEventLoss with data config: {data_config}")
    
    # 5) train loop
    # V2-only training with unified event system
    losses_train, losses_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,  # Pass the unified event loss function
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
