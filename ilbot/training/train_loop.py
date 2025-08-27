#!/usr/bin/env python3
"""
Training script for OSRS Imitation Learning Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
import argparse
from pathlib import Path
from ilbot.training.setup import create_data_loaders, setup_model, setup_training
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch, os, numpy as np
from collections import Counter
from datetime import datetime

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
    heads: dict from model(return_logits=True)
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

def _masked_metrics(heads: dict, target: torch.Tensor):
    """Simple masked metrics to track IL progress."""
    with torch.no_grad():
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

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cpu', scheduler=None,
                use_class_weights=True, loss_w=None, time_div=1.0):
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
        class_w = _compute_class_weights_from_loader(train_loader, device)
        print("  class_w[type]:", class_w["type"].tolist())
        print("  class_w[btn ]:", class_w["btn"].tolist())
        print("  class_w[key ]:", f"(151 dims; min={class_w['key'].min().item():.3g}, max={class_w['key'].max().item():.3g})")
        print("  class_w[sx  ]:", class_w["sx"].tolist(), " class_w[sy]:", class_w["sy"].tolist())
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
                loss = compute_il_loss(outputs, action_target, class_w=class_w, loss_w=loss_w, time_div=time_div)
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
        vm = {"acc_type":0.0,"acc_button":0.0,"acc_key":0.0,"acc_scroll_dx":0.0,"acc_scroll_dy":0.0,
              "mae_time":0.0,"mae_x":0.0,"mae_y":0.0,"valid_frac":0.0}
        vm_n = 0
        with torch.no_grad():
            for batch in val_loader:
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                outputs = model(temporal_sequence, action_sequence, return_logits=True)
                if isinstance(outputs, dict):
                    vloss = compute_il_loss(outputs, action_target, class_w=class_w, loss_w=loss_w, time_div=time_div)
                    mm = _masked_metrics(outputs, action_target)
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
            print(f"  Val metrics (masked avg over {vm_n} batches): "
                  f"acc_type={vm['acc_type']:.3f} acc_btn={vm['acc_button']:.3f} acc_key={vm['acc_key']:.3f} "
                  f"acc_sx={vm['acc_scroll_dx']:.3f} acc_sy={vm['acc_scroll_dy']:.3f} "
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
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    gamestate_file = data_dir / "gamestate_sequences.npy"
    action_input_file = data_dir / "action_input_sequences.npy"
    action_targets_file = data_dir / "action_targets.npy"
    
    from ilbot.training.setup import OSRSDataset
    dataset = OSRSDataset(
        gamestate_file=str(gamestate_file),
        action_input_file=str(action_input_file),
        action_targets_file=str(action_targets_file),
        sequence_length=10,
        max_actions=100
    )
    
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=args.batch_size,
        disable_cuda_batch_opt=args.disable_auto_batch,
        shuffle=True,
        device=device
    )
    
    # Create model and setup training components
    # Device already set above, reuse it
    model = setup_model(
        device=device,
        gamestate_dim=128,
        action_dim=8,
        sequence_length=10,
        hidden_dim=256,
        num_attention_heads=8
    )
    
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
        time_div=args.time_div
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

if __name__ == "__main__":
    main()
