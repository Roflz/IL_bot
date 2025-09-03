#!/usr/bin/env python3
"""
Inference and evaluation CLI for imitation learning bot.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ilbot.config import Config
from ilbot.data.dataset import make_loaders
from ilbot.model.model import SequentialImitationModel
from ilbot.inference.sampler import greedy_decode, sample_decode
from ilbot.eval.metrics import per_class_prf1, confusion_matrix, timing_mae_median, xy_mae
from ilbot.data.contracts import derive_event_targets_from_marks, EVENT_ID_TO_NAME


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint with strict schema validation."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Validate checkpoint schema
    required_keys = {"model_state", "epoch", "best_val"}
    if not required_keys.issubset(checkpoint.keys()):
        missing = required_keys - set(checkpoint.keys())
        raise ValueError(f"Checkpoint missing required keys: {missing}")
    
    # Load model state
    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception as e:
        raise RuntimeError(f"Failed to load model state: {e}")
    
    return checkpoint


def write_predictions_jsonl(predictions: Dict[str, torch.Tensor], targets: torch.Tensor, valid_mask: torch.Tensor, output_path: Path):
    """Write predictions to JSONL format."""
    B, A = predictions["event_id"].shape
    
    with open(output_path, 'w') as f:
        for b in range(B):
            for a in range(A):
                if valid_mask[b, a]:
                    pred = {
                        "batch_idx": b,
                        "action_idx": a,
                        "event_id": int(predictions["event_id"][b, a].item()),
                        "event_name": EVENT_ID_TO_NAME[int(predictions["event_id"][b, a].item())],
                        "time_s": float(predictions["time_s"][b, a].item()),
                        "x_norm": float(predictions["x_norm"][b, a].item()),
                        "y_norm": float(predictions["y_norm"][b, a].item()),
                        "x_px": float(predictions["x_px"][b, a].item()),
                        "y_px": float(predictions["y_px"][b, a].item()),
                        "target_event_id": int(derive_event_targets_from_marks(targets)[b, a].item()),
                        "target_time_s": float(targets[b, a, 0].item()),
                        "target_x_norm": float(targets[b, a, 1].item()),
                        "target_y_norm": float(targets[b, a, 2].item()),
                    }
                    f.write(json.dumps(pred) + '\n')


def compute_evaluation_metrics(predictions: Dict[str, torch.Tensor], targets: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    # Ensure all tensors are on the same device
    device = predictions["event_logits"].device
    targets = targets.to(device)
    valid_mask = valid_mask.to(device)
    
    # Event classification metrics
    event_logits = predictions["event_logits"]
    event_targets = derive_event_targets_from_marks(targets)
    
    prf1_metrics = per_class_prf1(event_logits, event_targets, valid_mask)
    cm = confusion_matrix(event_logits, event_targets, valid_mask, num_classes=4)
    
    # Timing metrics
    time_mae = timing_mae_median(predictions["time_s"], targets[:, :, 0], valid_mask)
    
    # XY metrics
    x_mae, y_mae = xy_mae(
        predictions["x_norm"], predictions["y_norm"],
        targets[:, :, 1], targets[:, :, 2],
        valid_mask
    )
    
    # Counts
    total_valid = int(valid_mask.sum().item())
    per_class_counts = {}
    for cls in range(4):
        per_class_counts[cls] = int((event_targets == cls).sum().item())
    
    return {
        "event_classification": prf1_metrics,
        "confusion_matrix": cm.tolist(),
        "timing_mae": time_mae,
        "xy_mae": {"x": x_mae, "y": y_mae},
        "counts": {
            "total_valid": total_valid,
            "per_class_targets": per_class_counts
        }
    }


def plot_confusion_matrix(cm: torch.Tensor, output_path: Path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy for plotting
    cm_np = cm.cpu().numpy()
    
    # Plot heatmap
    im = ax.imshow(cm_np, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    class_names = [EVENT_ID_TO_NAME[i] for i in range(4)]
    ax.set(xticks=np.arange(cm_np.shape[1]),
           yticks=np.arange(cm_np.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm_np.max() / 2.
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            ax.text(j, i, format(cm_np[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm_np[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_xy_scatter(predictions: Dict[str, torch.Tensor], targets: torch.Tensor, valid_mask: torch.Tensor, output_path: Path):
    """Plot XY scatter plot (normalized coordinates)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract valid predictions and targets
    valid_x_pred = predictions["x_norm"][valid_mask].cpu().numpy()
    valid_y_pred = predictions["y_norm"][valid_mask].cpu().numpy()
    valid_x_tgt = targets[:, :, 1][valid_mask].cpu().numpy()
    valid_y_tgt = targets[:, :, 2][valid_mask].cpu().numpy()
    
    # X coordinates
    ax1.scatter(valid_x_tgt, valid_x_pred, alpha=0.6, s=1)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax1.set_xlabel('Target X (normalized)')
    ax1.set_ylabel('Predicted X (normalized)')
    ax1.set_title('X Coordinate Predictions')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Y coordinates
    ax2.scatter(valid_y_tgt, valid_y_pred, alpha=0.6, s=1)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax2.set_xlabel('Target Y (normalized)')
    ax2.set_ylabel('Predicted Y (normalized)')
    ax2.set_title('Y Coordinate Predictions')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference and evaluation for imitation learning bot")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--out", type=str, required=True, help="Output directory for results")
    parser.add_argument("--mode", type=str, choices=["greedy", "sample"], default="greedy", help="Inference mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inference")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (for sample mode)")
    parser.add_argument("--topk", type=int, help="Top-k sampling (for sample mode)")
    parser.add_argument("--gumbel", action="store_true", help="Use Gumbel-Max sampling (for sample mode)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config and loaders
    cfg = Config(data_dir=args.data_dir)
    train_loader, val_loader, data_config = make_loaders(cfg)
    
    # Select loader
    loader = val_loader if args.split == "val" else train_loader
    
    # Build model
    model = SequentialImitationModel(
        data_config=data_config,
        hidden_dim=128,  # Match the checkpoint architecture
        horizon_s=cfg.horizon_s,
        enum_sizes={"button": 3, "key_action": 3, "key_id": 505, "scroll": 3}
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.ckpt)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss {checkpoint['best_val']:.4f}")
    
    # Run inference
    all_predictions = []
    all_targets = []
    all_valid_masks = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_items and batch_idx * cfg.batch_size >= args.max_items:
                break
            
            # Run inference
            if args.mode == "greedy":
                predictions = greedy_decode(
                    model, batch,
                    max_actions=data_config["max_actions"],
                    horizon_s=cfg.horizon_s,
                    device=device,
                    seed=args.seed
                )
            else:  # sample mode
                predictions = sample_decode(
                    model, batch,
                    max_actions=data_config["max_actions"],
                    horizon_s=cfg.horizon_s,
                    device=device,
                    temperature=args.temperature,
                    topk=args.topk,
                    gumbel=args.gumbel,
                    seed=args.seed
                )
            
            all_predictions.append(predictions)
            all_targets.append(batch["targets"])
            all_valid_masks.append(batch["valid_mask"])
    
    # Concatenate all batches
    if not all_predictions:
        raise RuntimeError("No predictions generated - check data loading")
    
    # Combine predictions
    combined_predictions = {}
    for key in all_predictions[0].keys():
        combined_predictions[key] = torch.cat([pred[key] for pred in all_predictions], dim=0)
    
    combined_targets = torch.cat(all_targets, dim=0)
    combined_valid_masks = torch.cat(all_valid_masks, dim=0)
    
    print(f"Generated predictions for {len(all_predictions)} batches")
    print(f"Total valid actions: {int(combined_valid_masks.sum().item())}")
    
    # Write predictions
    predictions_path = output_dir / "predictions.jsonl"
    write_predictions_jsonl(combined_predictions, combined_targets, combined_valid_masks, predictions_path)
    print(f"Wrote predictions to {predictions_path}")
    
    # Compute metrics
    metrics = compute_evaluation_metrics(combined_predictions, combined_targets, combined_valid_masks)
    
    # Write evaluation report
    report_path = output_dir / "eval_report.json"
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote evaluation report to {report_path}")
    
    # Generate plots
    cm = torch.tensor(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_xy_scatter(combined_predictions, combined_targets, combined_valid_masks, output_dir / "xy_scatter_norm.png")
    print(f"Generated plots in {output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Event Classification:")
    print(f"  Macro F1: {metrics['event_classification']['macro']['f1']:.4f}")
    print(f"  Micro Accuracy: {metrics['event_classification']['micro']['accuracy']:.4f}")
    print(f"Timing MAE: {metrics['timing_mae']:.4f}s")
    print(f"XY MAE: x={metrics['xy_mae']['x']:.4f}, y={metrics['xy_mae']['y']:.4f}")
    print(f"Total valid actions: {metrics['counts']['total_valid']}")
    print("="*50)


if __name__ == "__main__":
    main()
