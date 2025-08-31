#!/usr/bin/env python3
import argparse, os, glob, json
from pathlib import Path
import torch
import numpy as np

from setup_training import OSRSDataset, create_data_loaders, setup_model  # uses your existing API

@torch.no_grad()
def masked_metrics_from_heads(heads, target):
    # replicate the metric logic used in train script
    def build_mask(t):
        time_  = t[...,0]; a_type=t[...,1]; x_=t[...,2]; y_=t[...,3]
        button = t[...,4]; key=t[...,5]; sx=t[...,6]; sy=t[...,7]
        is_pad = (a_type==-1)&(button==-1)&(key==-1) & (time_==0)&(x_==0)&(y_==0) & (sx==0)&(sy==0)
        return ~is_pad
    m = build_mask(target).reshape(-1)

    at = heads["action_type_logits"].argmax(-1)
    bt = heads["button_logits"].argmax(-1)
    ky = heads["key_logits"].argmax(-1)
    sx = heads["scroll_x_logits"].argmax(-1) - 1
    sy = heads["scroll_y_logits"].argmax(-1) - 1

    def acc(p,t): 
        p = p.reshape(-1)[m]; t=t.reshape(-1)[m]
        return (p==t).float().mean().item() if p.numel() else 0.0
    def mae(p,t):
        p = p.reshape(-1)[m].float(); t=t.reshape(-1)[m].float()
        return (p-t).abs().mean().item() if p.numel() else 0.0

    return {
        "acc_type": acc(at, target[...,1].long().clamp(0,4)),
        "acc_btn":  acc(bt, target[...,4].long().clamp(0,3)),
        "acc_key":  acc(ky, target[...,5].long().clamp(0,150)),
        "acc_sx":   acc(sx, target[...,6].long().clamp(-1,1)),
        "acc_sy":   acc(sy, target[...,7].long().clamp(-1,1)),
        "mae_time": mae(heads["time"], target[...,0]),
        "mae_x":    mae(heads["x"],    target[...,2]),
        "mae_y":    mae(heads["y"],    target[...,3]),
    }

def latest_best():
    cands = sorted(glob.glob(os.path.join("checkpoints","*","best.pt")), key=os.path.getmtime)
    return cands[-1] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="path to best.pt (optional)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args()

    ckpt_path = args.ckpt or latest_best()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoints/**/best.pt found and --ckpt not provided.")
    print(f"Using checkpoint: {ckpt_path}")

    data_dir = Path(args.data_dir)
    ds = OSRSDataset(
        gamestate_file=str(data_dir/"gamestate_sequences.npy"),
        action_input_file=str(data_dir/"action_input_sequences.npy"),
        action_targets_file=str(data_dir/"action_targets.npy"),
        sequence_length=10, max_actions=100
    )
    train_loader, val_loader = create_data_loaders(ds, train_split=0.8, batch_size=8, shuffle=False, device=torch.device(args.device))
    model = setup_model(device=torch.device(args.device))

    # Load weights (support both raw state_dict and {"model":...})
    blob = torch.load(ckpt_path, map_location=args.device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.eval()

    out_dir = Path("eval_outputs"); out_dir.mkdir(exist_ok=True)
    metrics_agg = []
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= args.batches: break
        temporal = batch["temporal_sequence"].to(args.device)
        actions  = batch["action_sequence"].to(args.device)
        target   = batch["action_target"].to(args.device)

        heads = model(temporal, actions)
        # For unified event system, we work with the heads directly
        # Extract predictions from the unified event system outputs
        event_pred = heads["event_logits"].argmax(-1)  # [B, A] - 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
        time_pred = heads["time_q"][:, :, 1]  # [B, A] - median quantile (q=0.5)
        x_pred = heads["x_mu"]  # [B, A] - X mean
        y_pred = heads["y_mu"]  # [B, A] - Y mean
        
        # Create a simple output format for compatibility
        B, A = event_pred.shape
        decoded = torch.zeros(B, A, 8, device=args.device)  # [B, A, 8] for compatibility
        
        # Map unified events to legacy format for output
        # [time, type, x, y, button, key, scroll_dx, scroll_y]
        decoded[:, :, 0] = time_pred  # time
        decoded[:, :, 1] = event_pred.float()  # event type
        decoded[:, :, 2] = x_pred  # x
        decoded[:, :, 3] = y_pred  # y
        
        # Set button, key, scroll based on event type
        button_pred = heads["button_logits"].argmax(-1)
        key_action_pred = heads["key_action_logits"].argmax(-1)
        key_id_pred = heads["key_id_logits"].argmax(-1)
        scroll_y_pred = heads["scroll_y_logits"].argmax(-1) - 1  # Convert to {-1, 0, 1}
        
        # Only set values for the relevant event type
        decoded[:, :, 4] = torch.where(event_pred == 0, button_pred.float(), 0)  # button for CLICK
        decoded[:, :, 5] = torch.where(event_pred == 1, key_id_pred.float(), 0)  # key_id for KEY
        decoded[:, :, 6] = 0  # scroll_dx always 0
        decoded[:, :, 7] = torch.where(event_pred == 2, scroll_y_pred.float(), 0)  # scroll_y for SCROLL
        
        decoded = decoded.cpu().numpy()
        tgt_np  = target.cpu().numpy()
        np.save(out_dir/f"decoded_batch{i}.npy", decoded)
        np.save(out_dir/f"target_batch{i}.npy",  tgt_np)

        m = masked_metrics_from_heads(heads, target)
        metrics_agg.append(m); n += 1
        print(f"[batch {i}] ", m)

    # write summary
    if n:
        keys = metrics_agg[0].keys()
        summary = {k: float(sum(m[k] for m in metrics_agg)/n) for k in keys}
        with open(out_dir/"summary.json","w") as f:
            json.dump(summary, f, indent=2)
        print("== Summary ==", summary)
        print(f"Saved arrays + summary to {out_dir}/")

if __name__ == "__main__":
    main()
