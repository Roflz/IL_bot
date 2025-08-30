#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from ilbot.training.train_loop import run_training

def main():
    p = argparse.ArgumentParser("OSRS Imitation Learning Training")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--targets_version", default=None, choices=[None, "v1", "v2"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--disable_auto_batch", action="store_true")
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--step_size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--use_log1p_time", type=lambda s: s.lower()!="false", default=True)
    p.add_argument("--time_div_ms", type=float, default=1000.0)
    p.add_argument("--time_clip_s", type=float, default=None)
    # shared loss weights
    p.add_argument("--lw_time", type=float, default=0.3)
    p.add_argument("--lw_x", type=float, default=2.0)
    p.add_argument("--lw_y", type=float, default=2.0)
    # v1 heads (kept for compatibility)
    p.add_argument("--lw_type", type=float, default=1.0)
    p.add_argument("--lw_btn", type=float, default=1.0)
    p.add_argument("--lw_key", type=float, default=1.0)
    p.add_argument("--lw_sx", type=float, default=1.0)
    p.add_argument("--lw_sy", type=float, default=1.0)
    # v2 heads
    p.add_argument("--lw_button", type=float, default=1.0)
    p.add_argument("--lw_key_action", type=float, default=1.0)
    p.add_argument("--lw_key_id", type=float, default=1.0)
    p.add_argument("--lw_scroll_y", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # auto-detect targets_version + enums from manifest
    manifest_path = Path(args.data_dir) / "dataset_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    targets_version = args.targets_version or manifest.get("targets_version", "v1")
    
    # Normalize enum_sizes to ensure consistent dict format
    raw_enums = manifest.get("enums") or manifest.get("enum_sizes") or {}
    enum_sizes = {}
    for k, v in raw_enums.items():
        if isinstance(v, dict):
            size = int(v.get("size", v))
            none_idx = int(v.get("none_index", 1 if k == "scroll_y" else 0))
        else:
            size = int(v)
            none_idx = 1 if k == "scroll_y" else 0
        enum_sizes[k] = {"size": size, "none_index": none_idx}
    
    if targets_version != "v2":
        enum_sizes = {}

    config = {
        "data_dir": args.data_dir,
        "targets_version": targets_version,
        "enum_sizes": enum_sizes,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "disable_auto_batch": args.disable_auto_batch,
        "grad_clip": args.grad_clip,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "use_log1p_time": args.use_log1p_time,
        "time_div_ms": args.time_div_ms,
        "time_clip_s": args.time_clip_s,
        "loss_weights": {
            "time": args.lw_time, "x": args.lw_x, "y": args.lw_y,
            # v1 (legacy) heads
            "type": args.lw_type, "btn": args.lw_btn, "key": args.lw_key, "sx": args.lw_sx, "sy": args.lw_sy,
            # v2 heads
            "button": args.lw_button, "key_action": args.lw_key_action, "key_id": args.lw_key_id, "scroll_y": args.lw_scroll_y,
        },
        "seed": args.seed,
        "device": args.device,
    }

    print("OSRS Imitation Learning Training")
    print("============================================================")
    print(f"Data directory: {args.data_dir}")
    print(f"Targets version: {targets_version or 'auto'}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("============================================================")

    run_training(config)

if __name__ == "__main__":
    main()
