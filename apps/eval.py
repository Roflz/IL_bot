#!/usr/bin/env python3
"""
Enhanced evaluation script for OSRS imitation learning model
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from ilbot.model.metrics import denorm_time, build_masks, clamp_nonneg, topk_counts

def _print_time_stats(pred_time_log1p, tgt_time_log1p, args):
    """Print detailed time statistics including negative examples"""
    # raw (unclamped) seconds
    pred_sec_raw = torch.expm1(pred_time_log1p) * (args.time_div/1000.0)
    tgt_sec = torch.expm1(tgt_time_log1p.clamp_min(0)) * (args.time_div/1000.0)
    neg_frac = float((pred_time_log1p < 0).float().mean().cpu()) * 100.0
    if getattr(args, "time_clamp_nonneg", False):
        pred_for_mean = pred_sec_raw.clamp_min(0)
    else:
        pred_for_mean = pred_sec_raw
    mean_pred = float(torch.nanmean(pred_for_mean).cpu())
    mean_tgt  = float(torch.nanmean(tgt_sec).cpu())
    print(f"[time] mean_pred={mean_pred:.4f} (s) | mean_tgt={mean_tgt:.4f} (s) | neg_pred_frac={neg_frac:.2f}%")

    # show some raw negative examples (log1p domain and seconds)
    if neg_frac > 0:
        idx = (pred_time_log1p < 0).nonzero(as_tuple=False)
        take = idx[: min(len(idx), max(3, getattr(args, 'report_examples', 6)))]
        lines = []
        for b,a in take:
            pz = float(pred_time_log1p[b,a].cpu())
            ps = float(pred_sec_raw[b,a].cpu())
            tz = float(tgt_time_log1p[b,a].cpu())
            ts = float(tgt_sec[b,a].cpu())
            lines.append(f"    b={int(b)} a={int(a)}  pred_log1p={pz:.4f} pred_s={ps:.4f}  tgt_log1p={tz:.4f} tgt_s={ts:.4f}")
        if lines:
            print("[time] examples with negative pred (raw):")
            print("\n".join(lines))

def _print_topk(name, logits, tgt, k=5):
    """Print top-k predictions and targets for a given head"""
    # predictions
    pred = logits.argmax(dim=-1).view(-1)
    tgt  = tgt.view(-1)
    total = int((tgt >= 0).sum().item())
    topk = torch.bincount(pred, minlength=logits.shape[-1]).topk(k)
    print(f"[{name}] pred total={total}, top{k}={[ (int(i), int(c)) for c,i in zip(topk.values.tolist(), topk.indices.tolist()) ]}", end='')
    # targets
    tgt_counts = torch.bincount(tgt.clamp_min(0), minlength=logits.shape[-1])
    tk = tgt_counts.topk(min(k, logits.shape[-1]))
    print(f", tgt top{k}={[ (int(i), int(c)) for c,i in zip(tk.indices.tolist(), tk.values.tolist()) ]}")

def validate_step(model, batch, args):
    """Validate a single batch with enhanced reporting"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(batch['temporal_sequence'], batch['action_sequence'], return_logits=True)
        
        # Get targets
        tgt = batch['action_target']
        
        # button & key_action (top-k for pred+target)
        _print_topk("button", outputs["button_logits"], tgt["button"], k=args.topk_print)
        _print_topk("key_action", outputs["key_action_logits"], tgt["key_action"], k=args.topk_print)
        
        # key_id: show top-k for pred and target
        kid_logits = outputs["key_id_logits"]
        kid_tgt = tgt["key_id"]
        pred_kid = kid_logits.argmax(dim=-1).view(-1)
        tgt_kid  = kid_tgt.view(-1)
        pk = torch.bincount(pred_kid, minlength=kid_logits.shape[-1]).topk(min(args.topk_print, kid_logits.shape[-1]))
        tk = torch.bincount(tgt_kid.clamp_min(0), minlength=kid_logits.shape[-1]).topk(min(args.topk_print, kid_logits.shape[-1]))
        print(f"[key_id] pred total={int((kid_tgt>=0).sum())}, top{args.topk_print}={[ (int(i), int(c)) for c,i in zip(pk.indices.tolist(), pk.values.tolist()) ]} | tgt top{args.topk_print}={[ (int(i), int(c)) for c,i in zip(tk.indices.tolist(), tk.values.tolist()) ]}")
        
        # scroll_y: simple counts + target counts
        sy_logits = outputs["scroll_y_logits"]
        sy_tgt = tgt["scroll_y"]
        pred_sy = sy_logits.argmax(dim=-1).view(-1)
        tgt_sy  = sy_tgt.view(-1)
        pred_counts = torch.bincount(pred_sy, minlength=sy_logits.shape[-1]).tolist()
        tgt_counts  = torch.bincount(tgt_sy.clamp_min(0), minlength=sy_logits.shape[-1]).tolist()
        print(f"[scroll_y] pred total={int((sy_tgt>=0).sum())}, pred_counts={list(enumerate(pred_counts))} | tgt_counts={list(enumerate(tgt_counts))}")
        
        # event histogram (pred vs tgt)
        ev_pred = outputs["event_logits"].argmax(dim=-1)  # (B,A)
        ev_tgt = tgt["event"]
        
        # Count events
        n_move = int((ev_pred == 0).sum().item())
        n_click = int((ev_pred == 1).sum().item())
        n_key = int((ev_pred == 2).sum().item())
        n_scroll = int((ev_pred == 3).sum().item())
        total = n_move + n_click + n_key + n_scroll
        
        if total > 0:
            pct_move = (100.0 * n_move / total)
            pct_click = (100.0 * n_click / total)
            pct_key = (100.0 * n_key / total)
            pct_scroll = (100.0 * n_scroll / total)
        else:
            pct_move = pct_click = pct_key = pct_scroll = 0.0
        
        # optionally gate sub-heads by event to make MULTI impossible in reporting
        if getattr(args, "exclusive_event", False):
            pred_event = ev_pred  # one-hot choice per (B,A)
            # NB: we only change reporting masks; training loss stays as-is
            # Example: for CLICK rows, ignore key/scroll metrics; for KEY rows, ignore button/scroll, etc.
            masks = build_masks({"event_idx": pred_event})  # returns dict of per-task boolean masks
        else:
            masks = None
        
        # print event(pred) without MULTI when exclusive_event=True
        print(f"[event(pred)] MOVE={n_move} ({pct_move:.1f}%), CLICK={n_click} ({pct_click:.1f}%), KEY={n_key} ({pct_key:.1f}%), SCROLL={n_scroll} ({pct_scroll:.1f}%)" + ("" if not getattr(args,"exclusive_event",False) else ", MULTI=0 (0.0%)"))
        
        # time stats — we always show negatives if they exist
        _print_time_stats(outputs["time"], tgt["time"], args)
        
        # raw example preds (a few rows to eyeball)
        if args.report_examples > 0:
            print("=== Raw example predictions ===")
            shown = 0
            B, A = outputs["time"].shape
            for b in range(B):
                for a in range(A):
                    if shown >= args.report_examples:
                        break
                    ev = int(ev_pred[b,a].cpu())
                    tlog = float(outputs["time"][b,a].cpu())
                    line = f"  b={b} a={a}  event={ev}  time_log1p={tlog:.4f}"
                    if "button_logits" in outputs:
                        line += f"  btn={int(outputs['button_logits'][b,a].argmax().cpu())}"
                    if "key_action_logits" in outputs:
                        line += f"  ka={int(outputs['key_action_logits'][b,a].argmax().cpu())}"
                    if "key_id_logits" in outputs:
                        line += f"  kid={int(outputs['key_id_logits'][b,a].argmax().cpu())}"
                    if "scroll_y_logits" in outputs:
                        line += f"  sy={int(outputs['scroll_y_logits'][b,a].argmax().cpu())}"
                    print(line)
                    shown += 1

def main():
    """Main evaluation function"""
    # This would be called from your main training/evaluation script
    pass

if __name__ == "__main__":
    main()
