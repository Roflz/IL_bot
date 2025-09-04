# ilbot/training/train_loop.py
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Tuple
import json
from torch import amp
from torch.amp import autocast
from torch.amp import GradScaler

from ilbot.config import Config
from ilbot.data.dataset import make_loaders
from ilbot.model.model import SequentialImitationModel
from ilbot.training.losses import AdvancedUnifiedEventLoss
from ilbot.training.losses import LOGSIG_MIN, LOGSIG_MAX  # print clamp band
from ilbot.training.metrics import ema, masked_mae, event_top1_acc, length_stats
from ilbot.data.contracts import derive_event_targets_from_marks

def _print_run_header(cfg, data_config: Dict[str, Any], model: nn.Module, loss_fn: AdvancedUnifiedEventLoss, device: torch.device, train_loader, val_loader) -> None:
    print("\n==================== TRAIN SETUP ====================")
    total_steps = cfg.epochs * len(train_loader)
    print(f"Device: {device.type} | AMP: {cfg.amp}")
    print(f"Seed: {cfg.seed}")
    print(f"Data dims: Dg={data_config.get('gamestate_dim')}  T={data_config.get('temporal_window')}  "
          f"A={data_config.get('max_actions')}  Fa={data_config.get('action_features')}  events={data_config.get('event_types')}")
    # norm.json (if present via dataset contract)
    try:
        data_root = Path(cfg.data_dir).resolve().parents[2]
        norm_path = data_root / "data_profile" / "norm.json"
        norm = json.loads(norm_path.read_text())
        print(f"Normalization bounds: x_max={norm['x_max']}  y_max={norm['y_max']}  (coords normalized to [0,1])")
    except Exception:
        print("Normalization bounds: (unavailable in header; dataset already enforced it)")
    # loss weights & clamps
    print(f"Loss weights: event={loss_fn.event_weight}  timing={loss_fn.timing_weight}  xy={loss_fn.xy_weight}  qc={getattr(loss_fn,'qc_weight',0.0)}")
    print(f"XY logσ clamp: [{cfg.logsig_min}, {cfg.logsig_max}]  (σ≈[{math.exp(cfg.logsig_min):.3f}, {math.exp(cfg.logsig_max):.3f}])")
    # model params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    # new visibility
    print(f"Optimizer: AdamW | lr={cfg.lr} | weight_decay={cfg.weight_decay} | grad_clip={cfg.grad_clip}")
    print(f"Scheduler: CosineLR | total_steps={total_steps} | steps/epoch={len(train_loader)}")
    # --- Snapshot the first batch of each loader for mask stats
    def _snapshot(name: str, loader):
        try:
            b = next(iter(loader))
            vm = b["valid_mask"]                     # [B, A] bool
            per_seq = vm.sum(dim=1)                  # [B]
            zero_seq = int((per_seq == 0).sum().item())
            total_seq = int(per_seq.numel())
            print(f"{name:>6} mask: zero_seq={zero_seq}/{total_seq} | "
                  f"per-seq mean={per_seq.float().mean().item():.2f} "
                  f"min={int(per_seq.min().item())} max={int(per_seq.max().item())}")
        except StopIteration:
            print(f"{name:>6} mask: <empty loader>")
    _snapshot("train", train_loader)
    _snapshot("  val", val_loader)
    print("====================================================\n")

def _stats_str(x: torch.Tensor, name: str) -> str:
    x_ = x.detach()
    nan = (~torch.isfinite(x_)).sum().item()
    if x_.numel() == 0:
        return f"{name}: empty"
    finite = x_[torch.isfinite(x_)]
    if finite.numel() == 0:
        return f"{name}: ALL non-finite (nan/inf) count={nan}"
    return (f"{name}: shape={tuple(x_.shape)} "
            f"min={finite.min().item():.6g} max={finite.max().item():.6g} "
            f"mean={finite.mean().item():.6g} nonfinite={nan}")

def _dump_first_batch_inputs(gs, ai, tg, vm):
    # gs: [B,T,Dg], ai: [B,T,A,Fa], tg: [B,A,7], vm: [B,A]
    print("\n----- FIRST BATCH INPUT SNAPSHOT -----")
    print(_stats_str(gs, "gs"))
    print(_stats_str(ai, "ai"))
    print(_stats_str(tg[...,0], "tg.time_s"))
    print(_stats_str(tg[...,1], "tg.x_norm"))
    print(_stats_str(tg[...,2], "tg.y_norm"))
    print(_stats_str(vm.float(), "valid_mask"))
    # Quick per-seq valid counts
    per_seq = vm.sum(dim=1).detach().cpu().numpy()
    if per_seq.size > 0:
        print(f"valid per-seq: mean={per_seq.mean():.2f} min={per_seq.min()} max={per_seq.max()} zero_seq={(per_seq==0).sum()}/{len(per_seq)}")
    print("--------------------------------------\n")

def _dump_heads(pred):
    print("\n----- FIRST STEP PREDICTION SNAPSHOT -----")
    for k in ("event_logits","time_delta_q","x_mu","y_mu","x_logsigma","y_logsigma"):
        if k in pred and isinstance(pred[k], torch.Tensor):
            print(_stats_str(pred[k], f"pred.{k}"))
    print("------------------------------------------\n")

def _seed_everything(seed: int):
    """Set seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _validate_batch(batch: Dict[str, torch.Tensor], data_config: Dict[str, Any]):
    """Validate batch structure and types."""
    required_keys = ["temporal_sequence", "action_sequence", "targets", "valid_mask"]
    for key in required_keys:
        assert key in batch, f"Missing batch key: {key}"
    
    # Validate shapes
    B, T, Dg = batch["temporal_sequence"].shape
    assert batch["action_sequence"].shape == (B, T, data_config["max_actions"], data_config["action_features"])
    assert batch["targets"].shape == (B, data_config["max_actions"], 7)
    assert batch["valid_mask"].shape == (B, data_config["max_actions"])
    assert batch["valid_mask"].dtype == torch.bool, f"valid_mask must be bool, got {batch['valid_mask'].dtype}"

def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
    hparams: Dict[str, Any],
    data_bounds: Dict[str, float],
    run_dir: Path,
    is_best: bool = False
):
    """Save checkpoint with strict schema."""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "sched_state": scheduler.state_dict(),
        "best_val": best_val,
        "hparams": hparams,
        "data_bounds": data_bounds,
    }
    
    # Save last checkpoint
    last_path = run_dir / "last.pt"
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = run_dir / "best.pt"
        torch.save(checkpoint, best_path)

def _load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    checkpoint_path: Path
) -> Tuple[int, float]:
    """Load checkpoint and return epoch and best_val."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    if "sched_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["sched_state"])
    
    return checkpoint["epoch"], checkpoint["best_val"]

def _write_metrics(metrics: Dict[str, float], epoch: int, split: str, run_dir: Path):
    """Append metrics to CSV file."""
    metrics_path = run_dir / "metrics.csv"
    
    # Write header if file doesn't exist
    if not metrics_path.exists():
        header = "epoch,split,total,event_ce,timing_pinball,x_nll,y_nll,event_top1,timing_mae,x_mae,y_mae,lr\n"
        metrics_path.write_text(header)
    
    # Fill missing metrics with 0.0 for training
    if split == "train":
        metrics = {
            "total": metrics.get("total", 0.0),
            "event_ce": metrics.get("event_ce", 0.0),
            "timing_pinball": metrics.get("timing_pinball", 0.0),
            "x_nll": metrics.get("x_nll", 0.0),
            "y_nll": metrics.get("y_nll", 0.0),
            "event_top1": 0.0,  # Not computed for training
            "timing_mae": 0.0,  # Not computed for training
            "x_mae": 0.0,       # Not computed for training
            "y_mae": 0.0,       # Not computed for training
            "lr": metrics.get("lr", 0.0),
        }
    
    # Append metrics
    row = f"{epoch},{split},{metrics['total']:.6f},{metrics['event_ce']:.6f},{metrics['timing_pinball']:.6f},{metrics['x_nll']:.6f},{metrics['y_nll']:.6f},{metrics['event_top1']:.6f},{metrics['timing_mae']:.6f},{metrics['x_mae']:.6f},{metrics['y_mae']:.6f},{metrics['lr']:.8f}\n"
    with open(metrics_path, "a") as f:
        f.write(row)


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _pretty_log_epoch(split: str, m: Dict[str, float], epoch: int, total_epochs: int):
    print(f"Epoch {epoch:03d}/{total_epochs:03d} | {split:5}: "
          f"total={m['total']:.4f}, event={m['event_ce']:.4f}, time={m['timing_pinball']:.4f}, "
          f"x_nll={m['x_nll']:.4f}, y_nll={m['y_nll']:.4f}"
          + (f", top1={m['top1']:.4f}" if 'top1' in m else ""))


def _compute_event_weights(train_loader) -> torch.Tensor:
    """Inverse-frequency (or inv-sqrt) weights from the *masked* training targets."""
    counts = Counter()
    for batch in train_loader:
        targets = batch["targets"]
        mask = batch["valid_mask"].bool()
        y = derive_event_targets_from_marks(targets)  # [B,A]
        for c in range(4):
            counts[c] += int(((y == c) & mask).sum().item())
    tot = sum(counts.values())
    if tot == 0:
        raise RuntimeError("No valid events in training data; cannot compute class weights.")
    return torch.tensor([counts[i] for i in range(4)], dtype=torch.float32), counts


def _epoch_table(
    epoch: int,
    n_epochs: int,
    train: Dict[str, float],
    val: Dict[str, float],
    lr: float,
    improved: bool,
) -> str:
    """Create a readable table format for epoch summary."""
    star = "★" if improved else " "
    
    # Header
    header = f"Epoch {epoch:03d}/{n_epochs:03d} | LR: {lr:.2e} | {star}"
    
    # Table rows
    rows = [
        "┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐",
        "│    Metric   │    Total    │    Event    │   Timing    │     XY      │",
        "├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤",
        f"│ Train       │ {_fmt(train['total']):>11} │ {_fmt(train['event_ce']):>11} │ {_fmt(train['timing_pinball']):>11} │ {_fmt(train['x_nll']):>5},{_fmt(train['y_nll']):>5} │",
        f"│ Val         │ {_fmt(val['total']):>11} │ {_fmt(val['event_ce']):>11} │ {_fmt(val['timing_pinball']):>11} │ {_fmt(val['x_nll']):>5},{_fmt(val['y_nll']):>5} │",
        f"│ Val Top1    │ {_fmt(val.get('event_top1', float('nan'))):>11} │             │             │             │",
        "└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘"
    ]
    
    return "\n".join([header] + rows)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int,
    data_config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # EMA tracking
    ema_state = {
        "total": 0.0, "event_ce": 0.0, "timing_pinball": 0.0,
        "x_nll": 0.0, "y_nll": 0.0, "grad_norm": 0.0
    }
    amp_overflows = 0
    
    for step, batch in enumerate(train_loader):
        gs = batch["temporal_sequence"].to(device)
        ai = batch["action_sequence"].to(device)
        tg = batch["targets"].to(device)
        vm = batch["valid_mask"].to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=(cfg.amp and device.type == "cuda")):
            outputs = model(gs, ai, vm)
            # loss in FP32 even under AMP
            outputs = {k: v.float() for k, v in outputs.items()}
            
            # Strict pre-loss checks (fail-fast at the true source)
            # Dump the *very first* batch for maximum visibility
            if epoch == 0 and step == 0:
                _dump_first_batch_inputs(gs, ai, tg, vm)
                _dump_heads(outputs)

            # Assert per-head finiteness before computing loss (labels the culprit head)
            for head_name in ("event_logits","time_delta_q","x_mu","y_mu","x_logsigma","y_logsigma"):
                if head_name in outputs:
                    tens = outputs[head_name]
                    if not torch.isfinite(tens).all():
                        # Print detailed stats and crash *right here*
                        print(_stats_str(tens, f"NON-FINITE {head_name}"))
                        raise RuntimeError(f"Non-finite values in model output head '{head_name}' at step {step}")

            loss_out = loss_fn(outputs, tg, vm)
            total_loss = (
                loss_out["event_ce"] * loss_fn.event_weight
                + loss_out["timing_pinball"] * loss_fn.timing_weight
                + (loss_out["x_gaussian_nll"] + loss_out["y_gaussian_nll"]) * 0.5 * loss_fn.xy_weight
            )
        
        # Backward pass
        if cfg.amp and device.type == "cuda":
            scaler.scale(total_loss).backward()
            # Unscale before clipping and grad checks
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()
        
        # Grad clip
        if cfg.grad_clip and cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        
        # Fail-fast non-finite gradient detection
        if not (cfg.amp and device.type == "cuda"):
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"Non-finite gradient in {name} at step {step}")
        
        # Optim step (+ scheduler) - Correct order: optimizer.step() BEFORE scheduler.step()
        if cfg.amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Update EMA
        ema_state["total"] = ema(float(total_loss), ema_state["total"])
        ema_state["event_ce"] = ema(float(loss_out["event_ce"]), ema_state["event_ce"])
        ema_state["timing_pinball"] = ema(float(loss_out["timing_pinball"]), ema_state["timing_pinball"])
        ema_state["x_nll"] = ema(float(loss_out["x_gaussian_nll"]), ema_state["x_nll"])
        ema_state["y_nll"] = ema(float(loss_out["y_gaussian_nll"]), ema_state["y_nll"])
        ema_state["grad_norm"] = ema(float(grad_norm), ema_state["grad_norm"])
        
        # Logging
        if (step % cfg.log_interval == 0) or (step == 0):
            print(
                f"[{epoch:03d}/{cfg.epochs:03d}] "
                f"step {step:04d} | "
                f"loss={total_loss.item():.4f} "
                f"(evt={loss_out['event_ce']:.4f} "
                f"time={loss_out['timing_pinball']:.4f} "
                f"xNLL={loss_out['x_gaussian_nll']:.4f} "
                f"yNLL={loss_out['y_gaussian_nll']:.4f}) | "
                f"grad={grad_norm:.2f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Fail-fast checks
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {total_loss}\n"
                               f"  components={{'event_ce': {float(loss_out['event_ce']) if torch.isfinite(loss_out['event_ce']) else 'nan'}, "
                               f"'timing_pinball': {float(loss_out['timing_pinball']) if torch.isfinite(loss_out['timing_pinball']) else 'nan'}, "
                               f"'x_nll': {float(loss_out['x_gaussian_nll']) if torch.isfinite(loss_out['x_gaussian_nll']) else 'nan'}, "
                               f"'y_nll': {float(loss_out['y_gaussian_nll']) if torch.isfinite(loss_out['y_gaussian_nll']) else 'nan'}}}")
        
        # Check for NaNs in parameters
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(f"Non-finite parameter {name} at step {step}")
    
    out = {
        "total": ema_state["total"],
        "event_ce": ema_state["event_ce"],
        "timing_pinball": ema_state["timing_pinball"],
        "x_nll": ema_state["x_nll"],
        "y_nll": ema_state["y_nll"],
        "grad_norm": ema_state["grad_norm"],
        "lr": optimizer.param_groups[0]["lr"]
    }
    out["amp_overflows"] = torch.tensor(amp_overflows)
    return out

def generate_predictions(
    model: nn.Module,
    val_loader,
    device: torch.device,
    data_config: Dict[str, Any],
    data_bounds: Dict[str, float]
) -> np.ndarray:
    """Generate predictions on validation set in action_targets format [N, A, 7]."""
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                temporal_sequence=batch["temporal_sequence"],
                action_sequence=batch["action_sequence"],
                valid_mask=batch["valid_mask"]
            )
            
            # Convert predictions to action_targets format [B, A, 7]
            B, A = batch["targets"].shape[:2]
            predictions = torch.zeros(B, A, 7, device=device, dtype=torch.float32)
            
            # [0] time_ms: convert from seconds back to milliseconds
            time_pred = outputs["time_delta_q"][:, :, 1]  # median quantile [B, A]
            predictions[:, :, 0] = time_pred * 1000.0  # seconds -> ms
            
            # [1] x_px: convert from normalized [0,1] back to pixels
            x_pred = outputs["x_mu"].squeeze(-1)  # [B, A]
            predictions[:, :, 1] = x_pred * data_bounds["x_max"]
            
            # [2] y_px: convert from normalized [0,1] back to pixels  
            y_pred = outputs["y_mu"].squeeze(-1)  # [B, A]
            predictions[:, :, 2] = y_pred * data_bounds["y_max"]
            
            # [3] button: get argmax from button logits
            button_pred = outputs["button_logits"].argmax(dim=-1)  # [B, A]
            predictions[:, :, 3] = button_pred.float()
            
            # [4] key_action: get argmax from key_action logits
            key_action_pred = outputs["key_action_logits"].argmax(dim=-1)  # [B, A]
            predictions[:, :, 4] = key_action_pred.float()
            
            # [5] key_id: get argmax from key_id logits
            key_id_pred = outputs["key_id_logits"].argmax(dim=-1)  # [B, A]
            predictions[:, :, 5] = key_id_pred.float()
            
            # [6] scroll_y: get argmax from scroll_y logits
            scroll_pred = outputs["scroll_y_logits"].argmax(dim=-1)  # [B, A]
            predictions[:, :, 6] = scroll_pred.float()
            
            # Only keep predictions where valid_mask is True
            valid_mask = batch["valid_mask"]  # [B, A]
            predictions = predictions.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all batches: [N, A, 7]
    return np.concatenate(all_predictions, axis=0).astype(np.float32)

def validate(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device,
    data_config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_losses = []
    loss_components = {"event_ce": [], "timing_pinball": [], "x_gaussian_nll": [], "y_gaussian_nll": []}
    metrics = {"event_top1": [], "timing_mae": [], "x_mae": [], "y_mae": []}
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Validate batch
            _validate_batch(batch, data_config)
            
            # Forward pass
            outputs = model(
                temporal_sequence=batch["temporal_sequence"],
                action_sequence=batch["action_sequence"],
                valid_mask=batch["valid_mask"]
            )
            loss_out = loss_fn(outputs, batch["targets"], batch["valid_mask"])
            total_loss = (
                loss_out["event_ce"] * loss_fn.event_weight
                + loss_out["timing_pinball"] * loss_fn.timing_weight
                + (loss_out["x_gaussian_nll"] + loss_out["y_gaussian_nll"]) * 0.5 * loss_fn.xy_weight
            )
            
            # Store losses
            total_losses.append(float(total_loss))
            for key in loss_components:
                loss_components[key].append(float(loss_out[key]))
            
            # Compute metrics
            mask = batch["valid_mask"]
            
            # Event accuracy
            event_targets = derive_event_targets_from_marks(batch["targets"])
            event_acc = event_top1_acc(outputs["event_logits"], event_targets, mask)
            metrics["event_top1"].append(float(event_acc))
            
            # Timing MAE (median quantile)
            timing_mae = masked_mae(outputs["time_delta_q"][:, :, 1], batch["targets"][:, :, 0], mask)
            metrics["timing_mae"].append(float(timing_mae))
            
            # XY MAE
            x_mae = masked_mae(outputs["x_mu"].squeeze(-1), batch["targets"][:, :, 1], mask)
            y_mae = masked_mae(outputs["y_mu"].squeeze(-1), batch["targets"][:, :, 2], mask)
            metrics["x_mae"].append(float(x_mae))
            metrics["y_mae"].append(float(y_mae))
    
    # Aggregate metrics
    result = {
        "total": float(np.mean(total_losses)),
        "event_ce": float(np.mean(loss_components["event_ce"])),
        "timing_pinball": float(np.mean(loss_components["timing_pinball"])),
        "x_nll": float(np.mean(loss_components["x_gaussian_nll"])),
        "y_nll": float(np.mean(loss_components["y_gaussian_nll"])),
        "event_top1": float(np.mean(metrics["event_top1"])),
        "timing_mae": float(np.mean(metrics["timing_mae"])),
        "x_mae": float(np.mean(metrics["x_mae"])),
        "y_mae": float(np.mean(metrics["y_mae"])),
        "lr": 0.0  # Not applicable for validation
    }
    
    # Fail-fast check
    for key, value in result.items():
        if key != "lr" and not math.isfinite(value):
            raise RuntimeError(f"Non-finite validation metric {key}: {value}")
    
    return result

def run_training(cfg: Config) -> Dict[str, Any]:
    """Main training loop."""
    # Set up determinism
    _seed_everything(cfg.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build data loaders
    train_loader, val_loader, data_config = make_loaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    model = SequentialImitationModel(data_config, hidden_dim=128, horizon_s=cfg.horizon_s, logsig_min=cfg.logsig_min, logsig_max=cfg.logsig_max).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss function
    loss_fn = AdvancedUnifiedEventLoss(
        xy_weight=cfg.xy_weight,
        focal_gamma=cfg.focal_gamma,
    ).to(device)
    
    # ---- Optional: build event class weights
    if cfg.event_weighting != "none":
        class_counts_t, class_counts = _compute_event_weights(train_loader)
        if cfg.event_weighting == "inverse":
            w = 1.0 / class_counts_t.clamp_min(1.0)
        elif cfg.event_weighting == "inv_sqrt":
            w = 1.0 / torch.sqrt(class_counts_t.clamp_min(1.0))
        else:
            raise ValueError(f"Unknown event_weighting={cfg.event_weighting}")
        # normalize to sum=num_classes (nice scale)
        w = (w / w.sum()) * 4.0
        loss_fn.set_event_class_weights(w.to(device))
        print(f"Event weights ({cfg.event_weighting}): {w.tolist()} | counts={dict(class_counts)}")
    
    # Build optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # AMP scaler
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    
    # Set up checkpointing
    data_root = Path(cfg.data_dir).resolve().parents[2]
    if data_root.name != "data":
        raise RuntimeError(f"Expected data root to be .../data, got {data_root}")
    
    # Extract recording session folder from data_dir path
    # data_dir: data/recording_sessions/20250831_113719/06_final_training_data
    # We want to get: 20250831_113719
    data_dir_path = Path(cfg.data_dir).resolve()
    session_folder = data_dir_path.parts[-2]  # Get the session folder name directly from path parts
    
    # Create checkpoints directory within the recording session folder
    # data_root is already data/, so we need to go to recording_sessions/session_folder
    session_dir = data_root / "recording_sessions" / session_folder
    ckpt_root = session_dir / "checkpoints"
    
    # Generate run name: run_XX format
    if cfg.run_name:
        run_name = cfg.run_name
    else:
        # Find next available run number
        existing_runs = []
        if ckpt_root.exists():
            for item in ckpt_root.iterdir():
                if item.is_dir() and item.name.startswith("run_"):
                    try:
                        run_num = int(item.name.split("_")[1])
                        existing_runs.append(run_num)
                    except (ValueError, IndexError):
                        continue
        
        next_run_num = max(existing_runs, default=0) + 1
        run_name = f"run_{next_run_num:02d}"
    
    run_dir = ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {run_dir}")
    _print_run_header(cfg, data_config, model, loss_fn, device, train_loader, val_loader)
    
    # Also show a tiny stats snapshot for raw normalized inputs on the first train batch
    batch0 = next(iter(train_loader))
    gs0 = batch0["temporal_sequence"].to(device)
    ai0 = batch0["action_sequence"].to(device)
    tg0 = batch0["targets"].to(device)
    vm0 = batch0["valid_mask"].to(device)
    _dump_first_batch_inputs(gs0, ai0, tg0, vm0)
    
    # Save hyperparameters
    hparams = {
        "data_dir": cfg.data_dir,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "amp": cfg.amp,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "log_interval": cfg.log_interval,
        "data_config": data_config,
    }
    
    # Get data bounds from norm.json
    norm_path = data_root / "data_profile" / "norm.json"
    with open(norm_path) as f:
        norm_data = json.load(f)
    data_bounds = {"x_max": norm_data["x_max"], "y_max": norm_data["y_max"]}
    
    hparams_path = run_dir / "hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    
    # Training loop
    best_val = float("inf")
    patience_counter = 0

    # --- Resume if last.pt exists
    start_epoch = 1
    last_ckpt = run_dir / "last.pt"
    if last_ckpt.exists():
        loaded_epoch, loaded_best = _load_checkpoint(model, optimizer, scaler, scheduler, last_ckpt)
        best_val = float(loaded_best)
        start_epoch = int(loaded_epoch) + 1
        if start_epoch > cfg.epochs:
            print(f"No epochs to run (start_epoch={start_epoch} > epochs={cfg.epochs}). Returning previous summary.")
            return {
                "final_train_loss": math.nan,
                "final_val_loss": math.nan,
                "best_val_loss": best_val,
                "epochs_run": loaded_epoch,
                "run_dir": str(run_dir),
                "data_bounds": data_bounds,
            }
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, cfg, epoch, data_config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, data_config)
        
        # Logging
        print(
            _epoch_table(
                epoch, cfg.epochs, train_metrics, val_metrics,
                scheduler.get_last_lr()[0], val_metrics["total"] < best_val
            )
        )
        
        # Save metrics
        _write_metrics(train_metrics, epoch, "train", run_dir)
        _write_metrics(val_metrics, epoch, "val", run_dir)
        
        # Checkpointing
        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]
            patience_counter = 0
        else:
            patience_counter += 1
        
        _save_checkpoint(
            model, optimizer, scaler, scheduler, epoch, best_val, hparams, data_bounds, run_dir, is_best
        )
        
        # Early stopping
        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
            break
    
    # Generate and save sample predictions on validation set
    print("Generating sample predictions on validation set...")
    predictions = generate_predictions(model, val_loader, device, data_config, data_bounds)
    predictions_path = run_dir / "sample_predictions.npy"
    np.save(predictions_path, predictions)
    print(f"Saved sample predictions: {predictions_path}")
    print(f"Predictions shape: {predictions.shape} (matches action_targets format [N, A, 7])")
    
    # Final summary
    summary = {
        "final_train_loss": train_metrics["total"],
        "final_val_loss": val_metrics["total"],
        "best_val_loss": best_val,
        "epochs_run": epoch,
        "run_dir": str(run_dir),
        "data_bounds": data_bounds,
        "predictions_path": str(predictions_path),
    }
    
    return summary
