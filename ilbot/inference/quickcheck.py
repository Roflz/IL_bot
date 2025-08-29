#!/usr/bin/env python3
# inference_quickcheck.py
import argparse, os, json, glob
from pathlib import Path
import numpy as np
import torch

# Use your existing helpers
from ilbot.training.setup import OSRSDataset, create_data_loaders, setup_model

def load_manifest(data_dir: Path):
    p = data_dir/"dataset_manifest.json"
    return json.load(open(p,"r",encoding="utf-8")) if p.exists() else None

@torch.no_grad()
def _masked_metrics_from_heads(heads, target, time_div: float = 1.0):
    """
    heads: dict of model outputs with logits/reals (return_logits=True)
    target: (B,100,8) [time, type, x, y, button, key, scroll_dx, scroll_dy]
    Mask: any row whose absolute-sum == 0 is padding (same rule as training).
    """
    B, A, _ = target.shape
    mask = (target.abs().sum(dim=-1) > 0).view(-1)  # (B*100,)

    # Categorical preds decoded from logits
    at = heads["action_type_logits"].argmax(-1)            # (B,100)
    bt = heads["button_logits"].argmax(-1)                 # (B,100)
    ky = heads["key_logits"].argmax(-1)                    # (B,100)
    sx = heads["scroll_x_logits"].argmax(-1) - 1           # {0,1,2}->{-1,0,1}
    sy = heads["scroll_y_logits"].argmax(-1) - 1

    def _acc(pred, tgt):
        p = pred.view(-1)[mask]
        t = tgt.view(-1)[mask]
        return (p == t).float().mean().item() if p.numel() else 0.0

    def _mae(pred, tgt):
        p = pred.view(-1).float()[mask]
        t = tgt.view(-1).float()[mask]
        return (p - t).abs().mean().item() if p.numel() else 0.0

    # time head was trained against (target_time / time_div); unscale predictions for metrics in ms
    pred_time_ms = heads["time"] * time_div
    metrics = {
        "acc_type": _acc(at, target[..., 1].long().clamp(0, 4)),
        "acc_btn":  _acc(bt, target[..., 4].long().clamp(0, 3)),
        "acc_key":  _acc(ky, target[..., 5].long().clamp(0, 150)),
        "acc_sx":   _acc(sx, target[..., 6].long().clamp(-1, 1)),
        "acc_sy":   _acc(sy, target[..., 7].long().clamp(-1, 1)),
        "mae_time": _mae(pred_time_ms, target[..., 0]),   # compare in real ms
        "mae_x":    _mae(heads["x"],    target[..., 2]),
        "mae_y":    _mae(heads["y"],    target[..., 3]),
        "valid_frac": float(mask.float().mean().item()),
    }
    return metrics

def _latest_best():
    cands = sorted(glob.glob(os.path.join("checkpoints", "*", "best.pt")), key=os.path.getmtime)
    return cands[-1] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing gamestate_sequences.npy, action_input_sequences.npy, action_targets.npy")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to best.pt (optional; auto-pick latest if omitted)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=2, help="How many validation batches to run")
    ap.add_argument("--val_batch", type=int, default=8, help="Validation batch size for quickcheck")
    ap.add_argument("--out_dir", type=str, default="quickcheck_out")
    ap.add_argument("--time_div", type=float, default=1.0, help="Same scaling used in training (e.g., 1000)")
    ap.add_argument("--disable_auto_batch", action="store_true",
                    help="Force val batch size; skip CUDA auto batch optimization")
    ap.add_argument("--targets_version", choices=["v1","v2"], default=None,
                    help="If omitted, use dataset_manifest.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and determine configuration
    man = load_manifest(Path(args.data_dir))
    tv  = args.targets_version or (man["targets_version"] if man else "v1")
    time_div  = man["time_div"]  if (man and args.time_div is None) else (args.time_div or 1000.0)
    time_clip = man["time_clip"] if (man and args.time_clip is None) else (args.time_clip or 3.0)
    enum_sizes = (man or {}).get("enums", {})

    # Dataset - loaders: if tv=='v2' load actions_v2 + valid_mask else legacy, batch sizes from args
    if tv == "v2":
        ds = OSRSDataset(
            gamestate_file=str(data_dir / "gamestate_sequences.npy"),
            action_input_file=str(data_dir / "action_input_sequences.npy"),
            action_targets_file=str(data_dir / "actions_v2.npy"),
            sequence_length=10,
            max_actions=100
        )
    else:
        ds = OSRSDataset(
            gamestate_file=str(data_dir / "gamestate_sequences.npy"),
            action_input_file=str(data_dir / "action_input_sequences.npy"),
            action_targets_file=str(data_dir / "action_targets.npy"),
            sequence_length=10,
            max_actions=100
        )

    # Loaders (call with dataset=..., not data_dir=...)
    device = torch.device(args.device)
    train_loader, val_loader = create_data_loaders(
        dataset=ds,
        train_split=0.8,
        batch_size=args.val_batch,
        shuffle=False,
        device=device,
        disable_cuda_batch_opt=args.disable_auto_batch  # keep your chosen batch size
    )

    # Model - model: pass max_actions and enum sizes from manifest (fallback to inferred)
    inferred_A = 100  # fallback if no manifest
    model = setup_model(device=device, max_actions=(man["max_actions"] if man else inferred_A),
                       head_version=tv, enum_sizes=enum_sizes)
    ckpt_path = args.checkpoint or _latest_best()
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint found. Provide --checkpoint or ensure checkpoints/**/best.pt exists.")
    print(f"Using checkpoint: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location=device)
    state = blob["model"] if (isinstance(blob, dict) and "model" in blob) else blob
    model.load_state_dict(state)
    model.eval()

    # Run a few val batches
    metrics_all = []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= args.batches: break
            temporal = batch["temporal_sequence"].to(device)
            actions  = batch["action_sequence"].to(device)
            target   = batch["action_target"].to(device)

            heads = model(temporal, actions, return_logits=True)
            # decoding
            if tv == "v1":
                decoded = model.action_decoder.decode_v1(heads)
            else:
                decoded = model.action_decoder.decode_v2_as_legacy8(heads)
            decoded = decoded.cpu().numpy()
            # put time back in milliseconds for the .npy dump
            decoded[..., 0] *= time_div
            tgt_np  = target.cpu().numpy()

            np.save(out_dir / f"decoded_batch{bi}.npy", decoded)
            np.save(out_dir / f"target_batch{bi}.npy",  tgt_np)

            m = _masked_metrics_from_heads(heads, target, time_div=time_div)
            metrics_all.append(m)
            print(f"[batch {bi}] {m}")

    if metrics_all:
        keys = metrics_all[0].keys()
        summary = {k: float(sum(d[k] for d in metrics_all)/len(metrics_all)) for k in keys}
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("== Summary ==", summary)
        print(f"Saved arrays + summary to {out_dir}/")

if __name__ == "__main__":
    main()
