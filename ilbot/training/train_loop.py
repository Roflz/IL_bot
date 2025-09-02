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
from ilbot.training.setup import create_data_loaders, setup_training, OSRSDataset
from ilbot.model.imitation_hybrid_model import ImitationHybridModel
from ilbot.model.sequential_imitation_model import SequentialImitationModel
from torch.optim.lr_scheduler import StepLR
import os, numpy as np
from collections import Counter, defaultdict
from .pretty_output import printer
from datetime import datetime
from ilbot.utils.feature_spec import load_feature_spec
from .simplified_behavioral_metrics import SimplifiedBehavioralMetrics

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

def setup_model_v2(manifest, targets_version, device, data_dir: Path | None = None, use_sequential=False):
    """
    Build the hybrid model for V2 targets.
    """
    max_actions = int(manifest["max_actions"]) if manifest else 100
    head_version = targets_version
    # Convert enum_sizes from nested format to simple format expected by model
    raw_enums = manifest.get("enums", {})
    enum_sizes = {}
    for k, v in raw_enums.items():
        if isinstance(v, dict):
            enum_sizes[k] = int(v.get("size", v))
        else:
            enum_sizes[k] = int(v)
    # dims inferred from data (not constants)
    # Load actual dimensions from data files
    gamestate_sequences = np.load(Path(data_dir) / "gamestate_sequences.npy")
    action_input_sequences = np.load(Path(data_dir) / "action_input_sequences.npy")
    
    seq_len = gamestate_sequences.shape[1]  # temporal window
    gamestate_dim = gamestate_sequences.shape[2]  # number of features
    action_in_dim = action_input_sequences.shape[3]  # action features
    
    # derive indices & vocab from dataset artifacts
    spec = load_feature_spec(data_dir) if data_dir else {}
    
    # Create data_config for the model
    data_config = {
        'gamestate_dim': gamestate_dim,
        'max_actions': max_actions,
        'action_features': action_in_dim,
        'temporal_window': seq_len,
        'enum_sizes': enum_sizes,
        'event_types': 4
    }
    
    if use_sequential:
        model = SequentialImitationModel(
            data_config=data_config,
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            feature_spec=spec
        )
        printer.print_debug_info("Using SequentialImitationModel with SequentialActionDecoder", "SUCCESS")
    else:
        model = ImitationHybridModel(
            data_config=data_config,
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            feature_spec=spec
        )
        printer.print_debug_info("Using ImitationHybridModel with ActionSequenceDecoder", "INFO")
    
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
    if time_clip is None:
        time_clip = 3.0  # Default fallback
    if already_scaled:  # V2 saved time already scaled
        return torch.clamp(t, 0.0, float(time_clip))
    return torch.clamp(t / float(time_div), 0.0, float(time_clip))





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
    t_pred = fl(heads["time_q"][...,1])  # Use median (q50) from quantiles
    t_tgt  = fl(clamp_time(target[...,0], time_div, time_clip, already_scaled=True))
    t_pred_m = t_pred[m]
    t_tgt_m  = t_tgt[m]
    agg["time_pred_sum"] += float(t_pred_m.sum().item())
    agg["time_tgt_sum"]  += float(t_tgt_m.sum().item())
    agg["time_count"]    += int(t_pred_m.numel())
    agg["time_pred_neg"] += int((t_pred_m < 0).sum().item())

    # --- event-type summaries using actual event classification head ---
    # Get event predictions from the event classification head
    event_pred = heads["event_logits"].argmax(-1)  # [B, A] -> 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
    event_pred_flat = fl(event_pred)
    
    # Derive event targets from auxiliary heads (same logic as in _masked_metrics)
    button_gt = target[...,3].long()
    key_action_gt = target[...,4].long()
    scroll_y_gt = target[...,6].long()
    
    # Create event targets: 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
    event_target = torch.zeros_like(button_gt)
    event_target = torch.where(button_gt != 0, 0, event_target)      # CLICK
    event_target = torch.where(key_action_gt != 0, 1, event_target)  # KEY  
    event_target = torch.where(scroll_y_gt != 0, 2, event_target)    # SCROLL
    # MOVE is default (3) when no other events occur
    event_target_flat = fl(event_target)
    
    if m.any():
        # Count predicted events
        event_pred_masked = event_pred_flat[m]
        event_target_masked = event_target_flat[m]
        
        # Count each event type
        agg["evt_pred"]["CLICK"]  += int((event_pred_masked == 0).sum().item())
        agg["evt_pred"]["KEY"]    += int((event_pred_masked == 1).sum().item())
        agg["evt_pred"]["SCROLL"] += int((event_pred_masked == 2).sum().item())
        agg["evt_pred"]["MOVE"]   += int((event_pred_masked == 3).sum().item())
        agg["evt_pred"]["MULTI"]  = 0  # No more MULTI in unified event system
        
        # Count target events
        agg["evt_tgt"]["CLICK"]  += int((event_target_masked == 0).sum().item())
        agg["evt_tgt"]["KEY"]    += int((event_target_masked == 1).sum().item())
        agg["evt_tgt"]["SCROLL"] += int((event_target_masked == 2).sum().item())





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
          f"SCROLL={ep['SCROLL']} ({pct(ep['SCROLL'],pred_sum):.1f}%)")
    print(f"[event(tgt)]  MOVE={et['MOVE']} ({pct(et['MOVE'],tgt_sum):.1f}%), "
          f"CLICK={et['CLICK']} ({pct(et['CLICK'],tgt_sum):.1f}%), "
          f"KEY={et['KEY']} ({pct(et['KEY'],tgt_sum):.1f}%), "
          f"SCROLL={et['SCROLL']} ({pct(et['SCROLL'],tgt_sum):.1f}%)")


# (removed: legacy V1 IL loss)

import torch
import torch.nn.functional as F

def compute_unified_event_loss(predictions, targets, valid_mask, loss_fn, enum_sizes):
    """
    Compute advanced unified event system loss using the new AdvancedUnifiedEventLoss.
    
    Args:
        predictions: Model outputs from unified event system
        targets: [B, A, 7] V2 action targets [time, x, y, button, key_action, key_id, scroll_y]
        valid_mask: [B, A] Boolean mask for valid actions
        loss_fn: AdvancedUnifiedEventLoss instance
        enum_sizes: Dictionary with categorical sizes for auxiliary losses (not used by AdvancedUnifiedEventLoss)
    
    Returns:
        total_loss: Combined weighted loss
        loss_components: Dictionary of individual loss components
    """

    # Ensure valid_mask is 2D boolean
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.view(targets.shape[0], targets.shape[1])
    valid_mask = valid_mask.bool()
    
    # Compute loss using AdvancedUnifiedEventLoss (enum_sizes not needed)
    total_loss, loss_components = loss_fn(predictions, targets, valid_mask)
    
    return total_loss, loss_components




def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                num_epochs=10, device='cpu', scheduler=None,
                use_class_weights=True, loss_w=None, time_div=1.0,
                targets_version="v2", time_clip=3.0, enum_sizes=None, data_dir=None):
    """Train the model"""
    
    printer.print_header("OSRS Imitation Learning Training", f"Training on {device}")
    
    if torch.cuda.is_available():
        printer.print_debug_info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        printer.print_debug_info(f"Initial memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB allocated")
    
    printer.print_debug_info(f"Training samples: {len(train_loader.dataset)}")
    printer.print_debug_info(f"Validation samples: {len(val_loader.dataset)}")
    printer.print_debug_info(f"Epochs: {num_epochs}")
    
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
    # Create checkpoint directory in the session directory with run number
    if data_dir:
        session_dir = os.path.dirname(data_dir)  # Get the session directory (e.g., data/recording_sessions/20250831_113719)
        checkpoints_base_dir = os.path.join(session_dir, "checkpoints")
        
        # Find the next run number
        run_number = 1
        while True:
            ckpt_dir = os.path.join(checkpoints_base_dir, f"run_{run_number:02d}")
            if not os.path.exists(ckpt_dir):
                break
            run_number += 1
    else:
        # Fallback to old behavior if data_dir not provided
        ckpt_dir = os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"📁 Checkpoint directory: {ckpt_dir}")
    
    # Initialize enhanced behavioral intelligence metrics
    behavioral_metrics = SimplifiedBehavioralMetrics(save_dir=os.path.join(ckpt_dir, "behavioral_analysis"))
    

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        printer.print_epoch_start(epoch+1, num_epochs)
        
        # Reset epoch flag for loss function
        if hasattr(loss_fn, 'reset_epoch_flag'):
            loss_fn.reset_epoch_flag()
        
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

            # LAST-BATCH diagnostics (cleaner)
            if batch_idx == total_train_batches - 1:
                vm_last = batch['valid_mask'].to(device)
                B, A = action_target.size(0), action_target.size(1)
                if vm_last.dim() == 1 or vm_last.shape[:2] != (B, A):
                    vm_last = vm_last.view(B, A)
                vcount = int(vm_last.sum().item())
                print(f"  📊 Last batch: {vcount} valid actions")

            # Prepare a 2D mask for diagnostics and optional skipping
            vm = batch['valid_mask'].to(device)
            B, A = action_target.size(0), action_target.size(1)
            if vm.dim() == 1 or vm.shape[:2] != (B, A):
                vm = vm.view(B, A)

            # Skip batches with zero valid rows to avoid NaNs in CE
            if vm.sum().item() == 0:
                continue

            
            if isinstance(outputs, dict):
                # V2 only - use advanced unified event loss
                valid_mask = batch['valid_mask'].to(device)
                

                
                loss, loss_components = compute_unified_event_loss(outputs, action_target, valid_mask, loss_fn, enum_sizes)
            else:
                # Legacy loss computation (should not be used with V2)
                loss = torch.tensor(0.0, device=action_target.device)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress update with pretty printer
            printer.print_training_progress(batch_idx, len(train_loader), loss.item())


        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Clean Epoch Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  🎯 Training Loss: {avg_train_loss:.1f}")
        
        # Validation
        print("Validating...")
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        # Collect all validation predictions
        all_val_predictions = []
        all_val_targets = []
        all_val_masks = []

        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                outputs = model(temporal_sequence, action_sequence)
                
                # Collect predictions for full validation set
                if isinstance(outputs, dict):
                    # Store raw outputs and targets for full validation analysis
                    all_val_predictions.append({
                        'time_deltas': outputs.get('time_deltas', None),
                        'x_mu': outputs.get('x_mu', None),
                        'y_mu': outputs.get('y_mu', None),
                        'button_logits': outputs.get('button_logits', None),
                        'key_action_logits': outputs.get('key_action_logits', None),
                        'key_id_logits': outputs.get('key_id_logits', None),
                        'scroll_y_logits': outputs.get('scroll_y_logits', None),
                        'sequence_length': outputs.get('sequence_length', None),  # Add sequence length for actions per gamestate
                    })
                    all_val_targets.append(action_target.cpu())
                    all_val_masks.append(batch['valid_mask'].cpu())

                # how many total VAL batches this epoch
                n_val_batches = len(val_loader)

                # LAST-BATCH diagnostics (cleaner)
                if val_idx == n_val_batches - 1:
                    vm_last = batch['valid_mask'].to(device)
                    B, A = action_target.size(0), action_target.size(1)
                    if vm_last.dim() == 1 or vm_last.shape[:2] != (B, A):
                        vm_last = vm_last.view(B, A)
                    vcount = int(vm_last.sum().item())
                    print(f"    📊 Last val batch: {vcount} valid actions")

                # Prepare a 2D mask for diagnostics and optional skipping
                vm_mask = batch['valid_mask'].to(device)
                B, A = action_target.size(0), action_target.size(1)
                if vm_mask.dim() == 1 or vm_mask.shape[:2] != (B, A):
                    vm_mask = vm_mask.view(B, A)

                # Skip VAL batches with zero valid rows to avoid NaNs in CE
                if vm_mask.sum().item() == 0:
                    continue


                if isinstance(outputs, dict):
                    # V2 only - use advanced unified event loss
                    valid_mask = batch['valid_mask'].to(device)
                    vloss, loss_components = compute_unified_event_loss(outputs, action_target, valid_mask, loss_fn, enum_sizes)

                else:
                    # Legacy loss computation (should not be used with V2)
                    vloss = torch.tensor(0.0, device=action_target.device)
                val_loss_sum += float(vloss.item())
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Determine checkpoint saving logic
        is_best = avg_val_loss < best_val
        should_save_checkpoint = (
            epoch == 0 or  # First epoch
            epoch == num_epochs - 1 or  # Last epoch
            epoch % 5 == 0  # Every 5 epochs (0, 5, 10, 15, 20, ...)
        )
        
        # Save full validation predictions and create comprehensive graphs (only when checkpoints are saved)
        if all_val_predictions and should_save_checkpoint:
            # Combine all validation data for comprehensive analysis
            behavioral_metrics.create_full_validation_graphs(all_val_predictions, all_val_targets, all_val_masks, epoch)
        
        # Clean Event Distribution Display
        if val_batches > 0:
            # Get event predictions from validation
            sample_batch = next(iter(val_loader))
            sample_temporal = sample_batch['temporal_sequence'].to(device)
            sample_action = sample_batch['action_sequence'].to(device)
            sample_valid = sample_batch['valid_mask'].to(device)
            
            with torch.no_grad():
                sample_outputs = model(sample_temporal, sample_action)
                if isinstance(sample_outputs, dict):
                    event_logits = sample_outputs['event_logits']
                    event_probs = torch.softmax(event_logits, dim=-1)
                    
                    # Calculate average probabilities across all actions
                    avg_probs = event_probs.mean(dim=(0, 1))  # Average across batch and actions
                    
                    # Display detailed event distribution table
                    print(f"  🎮 Event Distribution Analysis:")
                    print(f"  ┌─────────────┬──────────────┬──────────────┬──────────────┐")
                    print(f"  │ Event Type  │ Predicted    │ Target       │ Difference   │")
                    print(f"  ├─────────────┼──────────────┼──────────────┼──────────────┤")
                    event_names = ['CLICK', 'KEY', 'SCROLL', 'MOVE']
                    # Calculate target distribution from validation data (only valid actions)
                    target_dist = torch.zeros(4)
                    if 'event_logits' in sample_outputs:
                        # Derive event targets from action targets (same logic as loss function)
                        targets = sample_batch['action_target']  # [B, A, 7]
                        valid_mask = sample_batch['valid_mask']  # [B, A]
                        
                        # Derive event type from action components
                        B, A = targets.shape[:2]
                        event_target = torch.full((B, A), 3, dtype=torch.long, device=targets.device)  # Default to MOVE
                        
                        # Extract action components
                        button_target = targets[..., 3]  # Button column
                        key_action_target = targets[..., 4]  # Key action column
                        scroll_y_target = targets[..., 6]  # Scroll column
                        
                        # Derive event types with priority
                        event_target = torch.where(button_target > 0, 0, event_target)  # CLICK
                        event_target = torch.where(key_action_target > 0, 1, event_target)  # KEY
                        event_target = torch.where(scroll_y_target != 0, 2, event_target)  # SCROLL
                        # MOVE is default (3)
                        
                        # Only count events for valid actions
                        valid_events = event_target[valid_mask.bool()]  # [N_valid]
                        
                        if valid_events.numel() > 0:
                            for j in range(4):
                                target_dist[j] = (valid_events == j).float().mean()
                    
                    for i, (name, prob) in enumerate(zip(event_names, avg_probs)):
                        target_prob = target_dist[i].item()
                        diff = prob.item() - target_prob
                        print(f"  │ {name:>11} │ {prob:.1%} │ {target_prob:.1%} │ {diff:+.1%} │")
                    print(f"  └─────────────┴──────────────┴──────────────┴──────────────┘")
        
        # Behavioral Intelligence Analysis (every 5 epochs)
        if val_batches > 0:
            # Get a sample batch for analysis
            sample_batch = next(iter(val_loader))
            sample_temporal = sample_batch['temporal_sequence'].to(device)
            sample_action = sample_batch['action_sequence'].to(device)
            sample_target = sample_batch['action_target'].to(device)
            sample_valid = sample_batch['valid_mask'].to(device)
            
            # Get model predictions on sample for detailed analysis
            with torch.no_grad():
                sample_outputs = model(sample_temporal, sample_action)
                
                # Debug: Show tensor shapes (but only once)
                if not hasattr(model, "_shapes_shown"):
                    model._shapes_shown = True
                    print(f"\n🔧 Model Output Shapes:")
                    if isinstance(sample_outputs, dict):
                        for k, v in sample_outputs.items():
                            print(f"  {k}: {tuple(v.shape)}")
                    print(f"  temporal_sequence: {sample_temporal.shape}")
                    print(f"  valid_mask: {sample_valid.shape}")
                    print("─" * 30)
            
            # Detailed Timing Analysis Table
            if isinstance(sample_outputs, dict) and 'time_q' in sample_outputs:
                try:
                    print(f"\n⏰ Timing Prediction Analysis:")
                    print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                    print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                    print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                    
                    # Get timing predictions (median quantile)
                    time_preds = sample_outputs['time_q'][:, 1]  # [B] - median timing
                    time_targets = sample_target[:, :, 0]  # [B, A] - target timing (timestamp column)
                    
                    # Calculate statistics
                    pred_mean = time_preds.mean().item()
                    pred_std = time_preds.std().item()
                    pred_min, pred_max = time_preds.min().item(), time_preds.max().item()
                    
                    # Target statistics (only for valid actions)
                    valid_times = time_targets[sample_valid.bool()]
                    target_mean = valid_times.mean().item() if valid_times.numel() > 0 else 0.0
                    target_std = valid_times.std().item() if valid_times.numel() > 0 else 0.0
                    target_min, target_max = valid_times.min().item(), valid_times.max().item() if valid_times.numel() > 0 else (0.0, 0.0)
                    
                    print(f"  │ Mean Timing (seconds)    │ {pred_mean:>12.3f} │ {target_mean:>12.3f} │ {pred_mean-target_mean:>+12.3f} │")
                    print(f"  │ Std Timing (seconds)     │ {pred_std:>12.3f} │ {target_std:>12.3f} │ {pred_std-target_std:>+12.3f} │")
                    print(f"  │ Min Timing (seconds)     │ {pred_min:>12.3f} │ {target_min:>12.3f} │ {pred_min-target_min:>+12.3f} │")
                    print(f"  │ Max Timing (seconds)     │ {pred_max:>12.3f} │ {target_max:>12.3f} │ {pred_max-target_max:>+12.3f} │")
                    print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                except Exception as e:
                    print(f"\n⏰ Timing Analysis: Could not analyze timing (error: {e})")
            
            # Actions per Gamestate Analysis Table
            try:
                print(f"\n📊 Actions per Gamestate Analysis:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                
                # Calculate predicted actions per gamestate
                if 'sequence_length' in sample_outputs:
                    pred_actions = sample_outputs['sequence_length'].squeeze().cpu().numpy()
                else:
                    # Fallback: count actions with timing > 0
                    time_preds = sample_outputs.get('time_q', torch.zeros(sample_temporal.size(0), 1))
                    pred_actions = (time_preds[:, 1] > 0.01).sum(dim=1).float().cpu().numpy()
                
                # Calculate target actions per gamestate
                target_actions = sample_valid.sum(dim=1).float().cpu().numpy()
                
                pred_mean = pred_actions.mean()
                pred_std = pred_actions.std()
                target_mean = target_actions.mean()
                target_std = target_actions.std()
                
                print(f"  │ Mean Actions/Gamestate   │ {pred_mean:>12.1f} │ {target_mean:>12.1f} │ {pred_mean-target_mean:>+12.1f} │")
                print(f"  │ Std Actions/Gamestate    │ {pred_std:>12.1f} │ {target_std:>12.1f} │ {pred_std-target_std:>+12.1f} │")
                print(f"  │ Min Actions/Gamestate    │ {pred_actions.min():>12.1f} │ {target_actions.min():>12.1f} │ {pred_actions.min()-target_actions.min():>+12.1f} │")
                print(f"  │ Max Actions/Gamestate    │ {pred_actions.max():>12.1f} │ {target_actions.max():>12.1f} │ {pred_actions.max()-target_actions.max():>+12.1f} │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            except Exception as e:
                print(f"\n📊 Actions per Gamestate Analysis: Could not analyze (error: {e})")
            
            # Timing Prediction Analysis Table
            try:
                if isinstance(sample_outputs, dict) and 'time_deltas' in sample_outputs and 'cumulative_times' in sample_outputs:
                    print(f"\n⏱️  Timing Prediction Analysis:")
                    print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                    print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                    print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                    
                    # Get timing predictions
                    time_deltas = sample_outputs['time_deltas']  # [B, A, 1]
                    cumulative_times = sample_outputs['cumulative_times']  # [B, A, 1]
                    
                    # Get targets
                    time_targets = sample_target[:, :, 0]  # [B, A] - target timing (timestamp column)
                    
                    # Calculate statistics (only for valid actions)
                    valid_mask = sample_valid.bool()
                    time_deltas_valid = time_deltas[valid_mask]  # [N_valid, 1]
                    cumulative_times_valid = cumulative_times[valid_mask]  # [N_valid, 1]
                    time_targets_valid = time_targets[valid_mask]  # [N_valid]
                    
                    if time_deltas_valid.numel() > 0:
                        # Delta timing statistics
                        delta_pred_mean = time_deltas_valid.mean().item()
                        delta_pred_std = time_deltas_valid.std().item()
                        delta_pred_min = time_deltas_valid.min().item()
                        delta_pred_max = time_deltas_valid.max().item()
                        delta_pred_median = time_deltas_valid.median().item()
                        delta_pred_p25 = time_deltas_valid.quantile(0.25).item()
                        delta_pred_p75 = time_deltas_valid.quantile(0.75).item()
                        delta_pred_p10 = time_deltas_valid.quantile(0.10).item()
                        delta_pred_p90 = time_deltas_valid.quantile(0.90).item()
                        delta_pred_unique = len(torch.unique(time_deltas_valid))
                        
                        # Cumulative timing statistics
                        cumul_pred_mean = cumulative_times_valid.mean().item()
                        cumul_pred_std = cumulative_times_valid.std().item()
                        cumul_pred_min = cumulative_times_valid.min().item()
                        cumul_pred_max = cumulative_times_valid.max().item()
                        
                        # Target timing statistics (data is already in delta time format)
                        if time_targets_valid.numel() > 0:
                            # Use the delta times directly (no torch.diff needed)
                            target_deltas = time_targets_valid
                            target_delta_mean = target_deltas.mean().item()
                            target_delta_std = target_deltas.std().item()
                            target_delta_min = target_deltas.min().item()
                            target_delta_max = target_deltas.max().item()
                            target_delta_median = target_deltas.median().item()
                            target_delta_p25 = target_deltas.quantile(0.25).item()
                            target_delta_p75 = target_deltas.quantile(0.75).item()
                            target_delta_p10 = target_deltas.quantile(0.10).item()
                            target_delta_p90 = target_deltas.quantile(0.90).item()
                            target_delta_unique = len(torch.unique(target_deltas))
                        else:
                            target_delta_mean = 0.0
                            target_delta_std = 0.0
                            target_delta_min = 0.0
                            target_delta_max = 0.0
                            target_delta_median = 0.0
                            target_delta_p25 = 0.0
                            target_delta_p75 = 0.0
                            target_delta_p10 = 0.0
                            target_delta_p90 = 0.0
                            target_delta_unique = 0
                        
                        print(f"  │ Delta Time Mean (s)     │ {delta_pred_mean:>12.3f} │ {target_delta_mean:>12.3f} │ {delta_pred_mean-target_delta_mean:>+12.3f} │")
                        print(f"  │ Delta Time Std (s)      │ {delta_pred_std:>12.3f} │ {target_delta_std:>12.3f} │ {delta_pred_std-target_delta_std:>+12.3f} │")
                        print(f"  │ Delta Time Median (s)   │ {delta_pred_median:>12.3f} │ {target_delta_median:>12.3f} │ {delta_pred_median-target_delta_median:>+12.3f} │")
                        print(f"  │ Delta Time P25 (s)      │ {delta_pred_p25:>12.3f} │ {target_delta_p25:>12.3f} │ {delta_pred_p25-target_delta_p25:>+12.3f} │")
                        print(f"  │ Delta Time P75 (s)      │ {delta_pred_p75:>12.3f} │ {target_delta_p75:>12.3f} │ {delta_pred_p75-target_delta_p75:>+12.3f} │")
                        print(f"  │ Delta Time P10 (s)      │ {delta_pred_p10:>12.3f} │ {target_delta_p10:>12.3f} │ {delta_pred_p10-target_delta_p10:>+12.3f} │")
                        print(f"  │ Delta Time P90 (s)      │ {delta_pred_p90:>12.3f} │ {target_delta_p90:>12.3f} │ {delta_pred_p90-target_delta_p90:>+12.3f} │")
                        print(f"  │ Delta Time Min (s)      │ {delta_pred_min:>12.3f} │ {target_delta_min:>12.3f} │ {delta_pred_min-target_delta_min:>+12.3f} │")
                        print(f"  │ Delta Time Max (s)      │ {delta_pred_max:>12.3f} │ {target_delta_max:>12.3f} │ {delta_pred_max-target_delta_max:>+12.3f} │")
                        print(f"  │ Delta Time Unique Count │ {delta_pred_unique:>12.0f} │ {target_delta_unique:>12.0f} │ {delta_pred_unique-target_delta_unique:>+12.0f} │")
                        print(f"  │ Cumulative Time Mean (s)│ {cumul_pred_mean:>12.3f} │        N/A │        N/A │")
                        print(f"  │ Cumulative Time Max (s) │ {cumul_pred_max:>12.3f} │        N/A │        N/A │")
                        print(f"  │ Actions > 600ms         │ {((cumulative_times_valid > 0.6).sum().item()):>12.0f} │        N/A │        N/A │")
                    else:
                        print(f"  │ No valid timing data    │        N/A │        N/A │        N/A │")
                    
                    print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            except Exception as e:
                print(f"\n⏱️  Timing Prediction Analysis: Could not analyze (error: {e})")
            
            # Mouse Position Analysis Table
            if isinstance(sample_outputs, dict) and 'x_mu' in sample_outputs and 'y_mu' in sample_outputs:
                try:
                    print(f"\n🖱️  Mouse Position Analysis:")
                    print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                    print(f"  │ Metric                  │ Predicted    │ Target       │ Difference   │")
                    print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                    
                    # Get predictions
                    x_pred = sample_outputs['x_mu']
                    y_pred = sample_outputs['y_mu']
                    x_std = sample_outputs['x_logsig'].exp()
                    y_std = sample_outputs['y_logsig'].exp()
                    
                    # Get targets
                    x_target = sample_target[:, :, 1]  # x coordinate
                    y_target = sample_target[:, :, 2]  # y coordinate
                    
                    # Calculate statistics (only for valid actions)
                    valid_mask = sample_valid.bool()
                    x_pred_valid = x_pred[valid_mask]
                    y_pred_valid = y_pred[valid_mask]
                    x_target_valid = x_target[valid_mask]
                    y_target_valid = y_target[valid_mask]
                    
                    if x_pred_valid.numel() > 0:
                        x_pred_mean = x_pred_valid.mean().item()
                        y_pred_mean = y_pred_valid.mean().item()
                        x_target_mean = x_target_valid.mean().item()
                        y_target_mean = y_target_valid.mean().item()
                        
                        print(f"  │ X Mean (normalized)     │ {x_pred_mean:>12.3f} │ {x_target_mean:>12.3f} │ {x_pred_mean-x_target_mean:>+12.3f} │")
                        print(f"  │ Y Mean (normalized)     │ {y_pred_mean:>12.3f} │ {y_target_mean:>12.3f} │ {y_pred_mean-y_target_mean:>+12.3f} │")
                        print(f"  │ X Uncertainty (pixels)  │ {x_std.mean().item():>12.0f} │        N/A │        N/A │")
                        print(f"  │ Y Uncertainty (pixels)  │ {y_std.mean().item():>12.0f} │        N/A │        N/A │")
                    else:
                        print(f"  │ No valid actions found  │        N/A │        N/A │        N/A │")
                    
                    print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
                except Exception as e:
                    print(f"\n🖱️  Mouse Position Analysis: Could not analyze (error: {e})")
            
            # Data Quality Analysis Table
            try:
                print(f"\n📊 Data Quality Analysis:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Count        │ Total        │ Percentage   │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                
                # Calculate data quality metrics
                total_actions = sample_valid.numel()
                valid_actions = sample_valid.sum().item()
                padding_actions = total_actions - valid_actions
                valid_percentage = (valid_actions / total_actions) * 100
                padding_percentage = (padding_actions / total_actions) * 100
                
                print(f"  │ Valid Actions           │ {valid_actions:>12} │ {total_actions:>12} │ {valid_percentage:>11.1f}% │")
                print(f"  │ Padding Actions         │ {padding_actions:>12} │ {total_actions:>12} │ {padding_percentage:>11.1f}% │")
                
                # Calculate actions per batch
                actions_per_batch = sample_valid.sum(dim=1).float()
                avg_actions_per_batch = actions_per_batch.mean().item()
                std_actions_per_batch = actions_per_batch.std().item()
                
                print(f"  │ Avg Actions/Batch       │ {avg_actions_per_batch:>12.1f} │        N/A │        N/A │")
                print(f"  │ Std Actions/Batch       │ {std_actions_per_batch:>12.1f} │        N/A │        N/A │")
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            except Exception as e:
                print(f"\n📊 Data Quality Analysis: Could not analyze (error: {e})")
            
            # Model Performance Summary Table
            try:
                print(f"\n🎯 Model Performance Summary:")
                print(f"  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
                print(f"  │ Metric                  │ Current      │ Best         │ Improvement  │")
                print(f"  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
                
                # Get current and best losses
                current_train_loss = avg_train_loss
                current_val_loss = avg_val_loss
                best_val_loss = best_val
                
                print(f"  │ Training Loss           │ {current_train_loss:>12.3f} │        N/A │        N/A │")
                print(f"  │ Validation Loss         │ {current_val_loss:>12.3f} │ {best_val_loss:>12.3f} │ {current_val_loss-best_val_loss:>+12.3f} │")
                
                # Calculate improvement percentage
                if best_val_loss > 0:
                    improvement_pct = ((best_val_loss - current_val_loss) / best_val_loss) * 100
                    print(f"  │ Loss Improvement %      │ {improvement_pct:>12.1f} │        N/A │        N/A │")
                
                print(f"  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
            except Exception as e:
                print(f"\n🎯 Model Performance Summary: Could not analyze (error: {e})")
            
        # Early stopping + checkpoint (logic already defined above)
            
        # Analyze behavioral intelligence (only save visualizations when checkpoints are saved)
        # Skip both timing graphs and actions per gamestate graph since they're already created from full validation set
        try:
            behavioral_metrics.analyze_epoch_predictions(
                model_outputs=sample_outputs,
                gamestates=sample_temporal,  # [B, 10, 128]
                action_targets=sample_target,  # [B, 100, 7]
                valid_mask=sample_valid,      # [B, 100]
                epoch=epoch,
                save_visualizations=False,  # Skip all visualizations - already created from full validation set
                save_timing_graphs=False  # Skip timing graphs - already created from full validation set
            )
        except Exception as e:
            print(f"⚠️  Behavioral analysis failed: {e}")

        # Scheduler step (if any)
        if scheduler is not None:
            scheduler.step()
            
        if is_best:
            best_val = avg_val_loss
            patience_left = patience
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1,
                        "val_loss": best_val}, best_path)
        
        # Save regular checkpoint if needed
        if should_save_checkpoint:
            checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1,
                        "val_loss": avg_val_loss}, checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                printer.print_debug_info("Early stopping triggered.", "WARNING")
                break
        
        # Print clean epoch summary
        printer.print_epoch_summary(epoch+1, avg_train_loss, avg_val_loss, best_val, patience_left, is_best)
        
        # Memory usage
        printer.print_memory_usage()
        
        # Clear cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate final behavioral intelligence summary
    printer.print_final_results(train_losses, val_losses, best_val)
    
    printer.print_debug_info("Generating Behavioral Intelligence Summary...")
    behavioral_metrics.generate_training_summary(epoch)
    
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
    # Model architecture
    parser.add_argument("--use_sequential", action='store_true',
                        help="Use SequentialImitationModel with SequentialActionDecoder")
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    
    # Create dataset first
    dataset = OSRSDataset(data_dir, targets_version=config.get("targets_version", "v1"))
    
    # Create data loaders from dataset
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=args.batch_size,
        shuffle=True,
        device=device
    )
    
    # Load manifest
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}
    
    tv = config.get("targets_version", "v1")
    enum_sizes = (manifest or {}).get("enums", {})
    time_div = manifest["time_div"] if (manifest and args.time_div is None) else (args.time_div or 1000.0)
    time_clip = manifest["time_clip"] if (manifest and args.time_clip is None) else (args.time_clip or 3.0)
    
    # Create model and setup training components
    model = setup_model_v2(manifest=manifest or {}, targets_version=config.get("targets_version", "v1"), device=device, data_dir=data_dir, use_sequential=args.use_sequential)
    
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
    
    # Use the existing OSRSDataset class instead of duplicating functionality
    ds = OSRSDataset(
        data_dir=config["data_dir"],
        targets_version=config.get("targets_version", "v1")
    )
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
        },
        data_dir=config["data_dir"],
        use_sequential=config.get("use_sequential", False)
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

    # 4) Create advanced unified event loss function
    from ilbot.model.advanced_losses import AdvancedUnifiedEventLoss
    
    # Use the same data_config for the loss function
    # data_config is already created above from DataInspector
    
    loss_fn = AdvancedUnifiedEventLoss(data_config=data_config)
    print(f"Created AdvancedUnifiedEventLoss with data config: {data_config}")
    
    # Set global class weights based on the entire dataset
    print("🎯 Computing global class weights from dataset...")
    all_event_targets = []
    all_valid_masks = []
    
    # Collect all event targets and valid masks
    for batch in train_loader:
        action_targets = batch["action_target"]
        valid_masks = batch["valid_mask"]
        
        # Derive event targets using the full action_targets tensor
        event_targets = loss_fn._derive_event_target(action_targets)
        
        all_event_targets.append(event_targets)
        all_valid_masks.append(valid_masks)
    
    # Set global weights
    loss_fn.set_global_event_weights(all_event_targets, all_valid_masks)
    
    # Debug: Print loss function info
    print(f"🔧 Loss function type: {type(loss_fn).__name__}")
    print(f"🔧 Loss function has focal loss: {hasattr(loss_fn, 'focal_loss')}")
    print(f"🔧 Loss function has label smoothing: {hasattr(loss_fn, 'label_smoothing_loss')}")
    print(f"🔧 Event class weights set: {loss_fn.event_class_weights is not None}")
    
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
        enum_sizes=enum_sizes,
        data_dir=config.get("data_dir")
    )
    history = {"train_losses": losses_train, "val_losses": losses_val}
    return history

if __name__ == "__main__":
    main()
