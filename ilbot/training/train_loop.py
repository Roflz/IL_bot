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
from ilbot.training.setup import create_data_loaders, OSRSDataset
from ilbot.model.sequential_imitation_model import SequentialImitationModel
from torch.optim.lr_scheduler import StepLR
import os, numpy as np
from collections import Counter, defaultdict
from .pretty_output import printer
from datetime import datetime
from ilbot.utils.feature_spec import load_feature_spec
from .simplified_behavioral_metrics import SimplifiedBehavioralMetrics

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

def build_model_and_loss(manifest, device, data_dir: Path):
    # derive data_config from manifest + npy shapes (gamestate_dim, action_features, max_actions, temporal_window, enum_sizes, event_types)
    import numpy as np, json
    gs = np.load(data_dir / "gamestate_sequences.npy")
    ai = np.load(data_dir / "action_input_sequences.npy")
    dc = {
        "gamestate_dim": int(gs.shape[2]),
        "action_features": int(ai.shape[3]),
        "temporal_window": int(gs.shape[1]),
        "max_actions": 100,
        "enum_sizes": {k:int(v["size"] if isinstance(v,dict) else v) for k,v in manifest.get("enums",{}).items()},
        "event_types": 4,
    }
    model = SequentialImitationModel(dc).to(device)
    from ilbot.model.advanced_losses import AdvancedUnifiedEventLoss
    loss_fn = AdvancedUnifiedEventLoss(event_weight=1.0, timing_weight=0.5, xy_weight=0.1)
    return model, loss_fn, dc





















import torch
import torch.nn.functional as F
from typing import Dict

def _print_combined_loss_components_table(epoch: int, train_components: Dict[str, float], val_components: Dict[str, float]):
    """Print a combined table of training and validation loss components"""
    if not train_components and not val_components:
        return
    
    print(f"\n📊 Loss Components Comparison (Epoch {epoch}):")
    print(f"  ┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ Loss Component                      │ Train Value  │ Train %      │ Val Value    │ Val %        │")
    print(f"  ├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    
    # Get all unique component names
    all_components = set(train_components.keys()) | set(val_components.keys())
    
    # Calculate totals
    train_total = sum(train_components.values()) if train_components else 0
    val_total = sum(val_components.values()) if val_components else 0
    
    # Sort components by training value (descending), fallback to validation value
    sorted_components = sorted(all_components, key=lambda x: (
        train_components.get(x, 0), 
        val_components.get(x, 0)
    ), reverse=True)
    
    for component_name in sorted_components:
        train_value = train_components.get(component_name, 0)
        val_value = val_components.get(component_name, 0)
        
        # Calculate percentage based on absolute total to handle negative totals
        train_total_abs = sum(abs(v) for v in train_components.values()) if train_components else 0
        val_total_abs = sum(abs(v) for v in val_components.values()) if val_components else 0
        
        train_percentage = (train_value / train_total_abs * 100) if train_total_abs > 0 else 0
        val_percentage = (val_value / val_total_abs * 100) if val_total_abs > 0 else 0
        
        print(f"  │ {component_name:<35} │ {train_value:>12.6f} │ {train_percentage:>11.1f}% │ {val_value:>12.6f} │ {val_percentage:>11.1f}% │")
    
    print(f"  ├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"  │ {'Total':<35} │ {train_total:>12.6f} │ {'100.0':>11}% │ {val_total:>12.6f} │ {'100.0':>11}% │")
    print(f"  └─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")







def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                num_epochs=10, device='cpu', scheduler=None,
                use_class_weights=True, loss_w=None, time_div=1.0,
                time_clip=3.0, enum_sizes=None, data_dir=None):
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
    train_loss_components = []  # Store loss components for each epoch
    val_loss_components = []    # Store loss components for each epoch
    


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
        # Fallback behavior if data_dir not provided
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

            # G) Validate masks - ensure single valid_mask is computed once
            if vm.sum().item() == 0:
                continue

            # G) Log validation mask statistics on first batch
            if batch_idx == 0:
                total_slots = vm.numel()
                valid_slots = vm.sum().item()
                valid_percentage = (valid_slots / total_slots) * 100
                print(f"G) Validation mask: {valid_slots}/{total_slots} slots valid ({valid_percentage:.1f}%)")

            # Forward pass
            outputs = model(temporal_sequence, action_sequence, vm)
            
            if isinstance(outputs, dict):
                # Use advanced unified event loss
                valid_mask = batch['valid_mask'].to(device)
                

                
                loss, loss_components = loss_fn(outputs, action_target, valid_mask)
                
                # Store loss components for this batch
                if not hasattr(train_model, 'batch_loss_components'):
                    train_model.batch_loss_components = []
                train_model.batch_loss_components.append(loss_components)
            else:
                # Legacy loss computation (should not be used)
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
        
        # Average loss components for this epoch
        if hasattr(train_model, 'batch_loss_components') and train_model.batch_loss_components:
            epoch_train_components = {}
            for component_name in train_model.batch_loss_components[0].keys():
                component_values = [batch_components[component_name] for batch_components in train_model.batch_loss_components]
                epoch_train_components[component_name] = sum(component_values) / len(component_values)
            train_loss_components.append(epoch_train_components)
            # Clear batch components for next epoch
            train_model.batch_loss_components = []
        else:
            train_loss_components.append({})
        
        # Clean Epoch Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  🎯 Training Loss: {avg_train_loss:.1f}")
        
        # Store training components for combined table
        current_train_components = train_loss_components[-1] if train_loss_components and train_loss_components[-1] else {}
        
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

                # Forward pass
                outputs = model(temporal_sequence, action_sequence, vm_mask)

                if isinstance(outputs, dict):
                    # Use advanced unified event loss
                    valid_mask = batch['valid_mask'].to(device)
                    vloss, loss_components = loss_fn(outputs, action_target, valid_mask)
                    
                    # Store validation loss components for this batch
                    if not hasattr(train_model, 'val_batch_loss_components'):
                        train_model.val_batch_loss_components = []
                    train_model.val_batch_loss_components.append(loss_components)

                else:
                    # Legacy loss computation (should not be used)
                    vloss = torch.tensor(0.0, device=action_target.device)
                val_loss_sum += float(vloss.item())
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Average validation loss components for this epoch
        if hasattr(train_model, 'val_batch_loss_components') and train_model.val_batch_loss_components:
            epoch_val_components = {}
            for component_name in train_model.val_batch_loss_components[0].keys():
                component_values = [batch_components[component_name] for batch_components in train_model.val_batch_loss_components]
                epoch_val_components[component_name] = sum(component_values) / len(component_values)
            val_loss_components.append(epoch_val_components)
            # Clear validation batch components for next epoch
            train_model.val_batch_loss_components = []
        else:
            val_loss_components.append({})
        
        # Print combined loss components table
        current_val_components = val_loss_components[-1] if val_loss_components and val_loss_components[-1] else {}
        _print_combined_loss_components_table(epoch + 1, current_train_components, current_val_components)
        
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
    
    # Print final loss components table
    if train_loss_components and val_loss_components:
        print(f"\n🎯 Final Training Loss Components Summary:")
        _print_combined_loss_components_table("Final", train_loss_components[-1], val_loss_components[-1])
    
    behavioral_metrics.generate_training_summary(
        epoch=epoch+1,  # Convert 0-based epoch to 1-based for display
        train_loss=train_losses[-1] if train_losses else 0.0,
        val_loss=val_losses[-1] if val_losses else 0.0,
        best_val_loss=best_val,
        is_best=False,  # This is the final summary, not a best epoch
        all_val_predictions=all_val_predictions,
        all_val_targets=all_val_targets,
        all_val_masks=all_val_masks
    )
    
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

    parser.add_argument("--time_clip", type=float, default=None,
                        help="Override time_clip; if None, use manifest.")
    # Model architecture
    parser.add_argument("--use_sequential", action='store_true',
                        help="Use SequentialImitationModel with SequentialActionDecoder")
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    
    # Create dataset first
    dataset = OSRSDataset(data_dir)
    
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
    
    enum_sizes = (manifest or {}).get("enums", {})
    time_div = manifest["time_div"] if (manifest and args.time_div is None) else (args.time_div or 1000.0)
    time_clip = manifest["time_clip"] if (manifest and args.time_clip is None) else (args.time_clip or 3.0)
    
    # Create model and setup training components
    model, loss_fn, data_config = build_model_and_loss(manifest or {}, device, data_dir)
    
    # Runtime safety check for feature spec
    spec = load_feature_spec(data_dir)
    print("[feature_spec] groups:", {k: len(v) for k, v in spec["group_indices"].items()})
    print("[feature_spec] cat fields:", [(f["name"], len(f["indices"]), f["vocab_size"]) for f in spec["cat_fields"]])
    print("[feature_spec] total_cat_vocab:", spec["total_cat_vocab"])
    
    # 3) optimizer/scheduler
    lr = args.lr
    wd = args.weight_decay
    step_size = args.step_size
    gamma = args.gamma
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5) train loop
    losses_train, losses_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler
    )
    
    print("\nTraining completed successfully!")
    print(f"Final training loss: {losses_train[-1]:.4f}")
    print(f"Final validation loss: {losses_val[-1]:.4f}")
    print(f"Best validation loss: {min(losses_val):.4f}")
    
    if torch.cuda.is_available():
        print(f"Final CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")




def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                num_epochs=10, device='cpu', scheduler=None, behavioral_metrics=None,
                best_val_loss=float('inf'), patience=10, patience_counter=0):
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
    train_loss_components = []
    val_loss_components = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        printer.print_epoch_start(epoch + 1, num_epochs)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            temporal_sequence = batch['temporal_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            action_target = batch['action_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Prepare a 2D mask for diagnostics and optional skipping
            vm = batch['valid_mask'].to(device)
            B, A = action_target.size(0), action_target.size(1)
            if vm.dim() == 1 or vm.shape[:2] != (B, A):
                vm = vm.view(B, A)

            # Skip batches with zero valid rows to avoid NaNs in CE
            if vm.sum().item() == 0:
                continue

            # Forward pass
            outputs = model(temporal_sequence, action_sequence, vm)
            
            if isinstance(outputs, dict):
                # Use unified event loss
                valid_mask = batch['valid_mask'].to(device)
                loss, loss_components = loss_fn(outputs, action_target, valid_mask)
                
                # Store loss components for this batch
                if not hasattr(train_model, 'batch_loss_components'):
                    train_model.batch_loss_components = []
                train_model.batch_loss_components.append(loss_components)
            else:
                # Legacy loss computation (should not be used)
                loss = torch.tensor(0.0, device=action_target.device)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress update
            printer.print_training_progress(batch_idx, len(train_loader), loss.item())

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Average training loss components for this epoch
        if hasattr(train_model, 'batch_loss_components') and train_model.batch_loss_components:
            epoch_train_components = {}
            for component_name in train_model.batch_loss_components[0].keys():
                component_values = [batch_components[component_name] for batch_components in train_model.batch_loss_components]
                epoch_train_components[component_name] = sum(component_values) / len(component_values)
            train_loss_components.append(epoch_train_components)
            # Clear batch components for next epoch
            train_model.batch_loss_components = []
        else:
            train_loss_components.append({})

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_val_predictions = []
        all_val_targets = []
        all_val_masks = []
        
        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                # Prepare a 2D mask for diagnostics and optional skipping
                vm_mask = batch['valid_mask'].to(device)
                B, A = action_target.size(0), action_target.size(1)
                if vm_mask.dim() == 1 or vm_mask.shape[:2] != (B, A):
                    vm_mask = vm_mask.view(B, A)

                # Skip VAL batches with zero valid rows to avoid NaNs in CE
                if vm_mask.sum().item() == 0:
                    continue

                # Forward pass
                outputs = model(temporal_sequence, action_sequence, vm_mask)

                if isinstance(outputs, dict):
                    # Use unified event loss
                    valid_mask = batch['valid_mask'].to(device)
                    vloss, loss_components = loss_fn(outputs, action_target, valid_mask)
                    
                    # Store validation loss components for this batch
                    if not hasattr(train_model, 'val_batch_loss_components'):
                        train_model.val_batch_loss_components = []
                    train_model.val_batch_loss_components.append(loss_components)
                    
                    # Store predictions for behavioral analysis
                    if behavioral_metrics is not None:
                        all_val_predictions.append(outputs)
                        all_val_targets.append(action_target)
                        all_val_masks.append(valid_mask)
                else:
                    # Legacy loss computation (should not be used)
                    vloss = torch.tensor(0.0, device=action_target.device)
                val_loss_sum += float(vloss.item())
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Average validation loss components for this epoch
        if hasattr(train_model, 'val_batch_loss_components') and train_model.val_batch_loss_components:
            epoch_val_components = {}
            for component_name in train_model.val_batch_loss_components[0].keys():
                component_values = [batch_components[component_name] for batch_components in train_model.val_batch_loss_components]
                epoch_val_components[component_name] = sum(component_values) / len(component_values)
            val_loss_components.append(epoch_val_components)
            # Clear validation batch components for next epoch
            train_model.val_batch_loss_components = []
        else:
            val_loss_components.append({})

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping and behavioral metrics
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        # Generate behavioral metrics summary
        if behavioral_metrics is not None and all_val_predictions:
            behavioral_metrics.generate_training_summary(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                best_val_loss=best_val_loss,
                is_best=is_best,
                all_val_predictions=all_val_predictions,
                all_val_targets=all_val_targets,
                all_val_masks=all_val_masks
            )

        # Print epoch summary
        best_val = min(val_losses) if val_losses else avg_val_loss
        printer.print_epoch_summary(epoch + 1, avg_train_loss, avg_val_loss, best_val, 0)

        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience: {patience})")
            break

    return train_losses, val_losses


def main():
    """Main entry point for training"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=8, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma')
    parser.add_argument('--use_sequential', action='store_true', help='Use sequential model')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Set up training
    _seed_everything(1337)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1) data loaders
    data_dir = Path(args.data_dir)
    dataset = OSRSDataset(data_dir)
    train_loader, val_loader = create_data_loaders(dataset, batch_size=args.batch_size)
    
    # Load manifest
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}
    
    # 2) model and loss
    model, loss_fn, data_config = build_model_and_loss(manifest or {}, device, data_dir)
    
    # 3) optimizer/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5) train loop with early stopping and behavioral metrics
    from ilbot.training.simplified_behavioral_metrics import SimplifiedBehavioralMetrics
    
    behavioral_metrics = SimplifiedBehavioralMetrics()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    losses_train, losses_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        behavioral_metrics=behavioral_metrics,
        best_val_loss=best_val_loss,
        patience=patience,
        patience_counter=patience_counter
    )
    
    print("\nTraining completed successfully!")
    print(f"Final training loss: {losses_train[-1]:.4f}")
    print(f"Final validation loss: {losses_val[-1]:.4f}")
    print(f"Best validation loss: {min(losses_val):.4f}")
    
    if torch.cuda.is_available():
        print(f"Final CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")


if __name__ == "__main__":
    main()
