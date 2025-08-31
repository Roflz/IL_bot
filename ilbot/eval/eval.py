#!/usr/bin/env python3
"""
Evaluation functions for OSRS imitation learning model
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from ..model.metrics import denorm_time, build_masks, clamp_nonneg, topk_counts
from .. import config as CFG

# Add debugging imports
import logging, torch
LOG = logging.getLogger(__name__)

def _build_evt_tgt(btn_tgt, ka_tgt, sy_tgt, btn_none=0, ka_none=0, sy_none=1):
    # 0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE
    is_click  = (btn_tgt != btn_none)
    is_key    = (ka_tgt  != ka_none)
    is_scroll = (sy_tgt  != sy_none)
    evt = torch.where(is_click, torch.tensor(0, device=btn_tgt.device),
              torch.where(is_key,   torch.tensor(1, device=btn_tgt.device),
              torch.where(is_scroll,torch.tensor(2, device=btn_tgt.device),
                                       torch.tensor(3, device=btn_tgt.device))))
    return evt

def detailed_report(outputs, targets, **kwargs):
    # EVENT (do NOT fall back to scroll)
    event_logits = outputs.get("event_logits")
    if event_logits is None:
        LOG.error("[DBG] detailed_report: event_logits missing; report will be wrong if we continue.")
        return {"event_accuracy": 0.0, "scroll_accuracy": 0.0}  # short-circuit
    event_tgt = targets.get("event")
    
    # SCROLL_Y
    scroll_logits = outputs.get("scroll_y_logits")
    scroll_tgt = targets.get("scroll_y_idx")
    if scroll_tgt is not None:
        # Reveal pre/post mapping ranges in the report run
        t = scroll_tgt.detach().cpu().view(-1)
        LOG.info("[DBG] report.scroll_tgt raw: min=%d max=%d uniq=%d",
                 int(t.min()), int(t.max()), int(torch.unique(t).numel()))
    
    # Return basic report structure
    return {
        "event_accuracy": 0.0,  # Placeholder
        "scroll_accuracy": 0.0,  # Placeholder
    }

def evaluate(model, val_loader, stats, args):
    """Evaluate model on validation data with detailed metrics"""
    model.eval()
    
    # Initialize counters and accumulators
    valid_counts = {"btn": 0, "ka": 0, "kid": 0, "sy": 0, "time": 0, "kid_eval_rows": 0}
    all_batch_sizes = []
    all_evt_pred = []
    all_evt_tgt = []
    
    # Per-head counters for detailed analysis
    kid_pred_ctr = Counter()
    kid_tgt_ctr = Counter()
    sy_pred_ctr = Counter()
    sy_tgt_ctr = Counter()
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            device = next(model.parameters()).device
            temporal_sequence = batch['temporal_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            action_target = batch['action_target'].to(device)
            
            # Forward pass
            outputs = model(temporal_sequence, action_sequence, return_logits=True)
            
            # Get predictions
            out_btn = outputs["button_logits"]
            out_ka = outputs["key_action_logits"]
            out_kid = outputs["key_id_logits"]
            out_sy = outputs["scroll_y_logits"]
            # XY (use heteroscedastic means from new heads)
            out_x = outputs["x_mu"]; out_y = outputs["y_mu"]
            
            # Define none indices for event mapping
            btn_none = 0  # Assuming button none index is 0
            ka_none = 0   # Assuming key_action none index is 0
            sy_none = 1   # Assuming scroll_y none index is 1
            
            # EVENT PRED (CLICK, KEY, SCROLL, MOVE) from event head ONLY
            evt_pred_indices = outputs["event_logits"].argmax(-1)  # (B,A) indices 0..3
            
            # Build per-head masks
            masks = build_masks(batch)
            
            # Update valid counts
            valid_counts["btn"] += int(masks["btn"].sum())
            valid_counts["ka"] += int(masks["ka"].sum())
            valid_counts["kid"] += int(masks["kid"].sum())
            valid_counts["sy"] += int(masks["sy"].sum())
            valid_counts["time"] += int(masks["time"].sum())
            
            # Time: use median quantile (q=0.5) from time_q
            k_med = CFG.time_quantiles.index(0.5)
            t_pred_pre = denorm_time(outputs["time_q_pre"][..., k_med].detach().cpu(), stats)
            t_pred_post = denorm_time(outputs["time_q"][..., k_med].detach().cpu(), stats)
            t_tgt = denorm_time(batch["time"].detach().cpu(), stats)
            t_pred_ms_pre = (t_pred_pre.numpy() * 1000.0)
            t_pred_ms_post = (t_pred_post.numpy() * 1000.0)
            t_tgt_ms = (t_tgt.numpy() * 1000.0)
            
            # Report negative predictions before clamp (using pre-activation)
            neg_pred_frac = float((t_pred_ms_pre < 0).mean() * 100.0)
            
            # Optional clamped reference (display only if you explicitly opt in)
            if CFG.report_time_clamped_reference:
                t_pred_ms_clamped = clamp_nonneg(t_pred_ms_post)
                mean_pred_ms_clamped = float(t_pred_ms_clamped.mean())
            else:
                mean_pred_ms_clamped = None
            
            # Debug time reporting if enabled
            if getattr(CFG, "debug_time", False):
                zeros_pct = float((t_pred_ms_post == 0.0).mean() * 100.0)
                pre_min  = float(np.nanmin(t_pred_ms_pre))
                pre_mean = float(np.nanmean(t_pred_ms_pre))
                pre_max  = float(np.nanmax(t_pred_ms_pre))
                print(f"[time/debug] pre(ms): min={pre_min:.2f} mean={pre_mean:.2f} max={pre_max:.2f} "
                      f"neg%={neg_pred_frac:.2f}")
                post_min  = float(np.nanmin(t_pred_ms_post))
                post_max  = float(np.nanmax(t_pred_ms_post))
                print(f"[time/debug] post(ms): min={post_min:.2f} mean={float(np.nanmean(t_pred_ms_post)):.2f} max={post_max:.2f} "
                      f"zeros%={zeros_pct:.2f}")
            
            # Key-ID evaluation is always on TARGET key rows (we log the count explicitly)
            kid_keep = masks["kid"]
            valid_counts["kid_eval_rows"] += int(kid_keep.sum())
            if kid_keep.any():
                pred_kid = out_kid[kid_keep].argmax(-1).cpu().tolist()
                tgt_kid = batch["key_id"][kid_keep].cpu().tolist()
                kid_pred_ctr.update(pred_kid)
                kid_tgt_ctr.update(tgt_kid)
            
            # scroll_y: top-k for both pred and target (on scroll rows)
            if masks["sy"].any():
                pred_sy = out_sy[masks["sy"]].argmax(-1).cpu().tolist()
                tgt_sy = batch["scroll_y"][masks["sy"]].cpu().tolist()
                sy_pred_ctr.update(pred_sy)
                sy_tgt_ctr.update(tgt_sy)
            
            # Collect examples
            if len(examples) < CFG.report_examples:
                take = min(5, CFG.report_examples - len(examples))
                idx = torch.randint(0, action_target.shape[0], (take,))
                for i in idx.tolist():
                    ep = int(evt_pred_indices[i].item())
                    # Derive event target from batch labels for this example using canonical order
                    et = int(_build_evt_tgt(batch["button"][i:i+1], batch["key_action"][i:i+1], batch["scroll_y"][i:i+1]).item())
                    row = {
                        "evt_pred": ep, "evt_tgt": et,
                        "btn_pred": int(out_btn[i].argmax().item()) if masks["btn"][i] or ep==0 else None,
                        "btn_tgt": int(batch["button"][i].item()) if masks["btn"][i] or et==0 else None,
                        "ka_pred": int(out_ka[i].argmax().item()) if masks["ka"][i] or ep==1 else None,
                        "ka_tgt": int(batch["key_action"][i].item()) if masks["ka"][i] or et==1 else None,
                        "kid_pred": int(out_kid[i].argmax().item()) if masks["kid"][i] or ep==1 else None,
                        "kid_tgt": int(batch["key_id"][i].item()) if masks["kid"][i] or et==1 else None,
                        "sy_pred": int(out_sy[i].argmax().item()) if masks["sy"][i] or ep==2 else None,
                        "sy_tgt": int(batch["scroll_y"][i].item()) if masks["sy"][i] or et==2 else None,
                        "time_ms_pred": float(t_pred_ms_post[i]),
                        "time_ms_tgt": float(t_tgt_ms[i]),
                        "x_pred": float(out_x[i].item()), "x_tgt": float(batch["x"][i].item()),
                        "y_pred": float(out_y[i].item()), "y_tgt": float(batch["y"][i].item()),
                    }
                    examples.append(row)
            
            # Store batch info for final summary
            all_batch_sizes.append(action_target.shape[0])
            all_evt_pred.extend(evt_pred_indices.cpu().tolist())
            # Derive event targets from batch labels using canonical order
            evt_tgt = _build_evt_tgt(batch["button"].view(-1), batch["key_action"].view(-1), batch["scroll_y"].view(-1)).cpu().tolist()
            all_evt_tgt.extend(evt_tgt)
    
    # End-of-epoch print (new bits)
    k = CFG.report_k_top
    
    # key_id histogram with optional original key code translation
    def print_key_id_hist(pred, tgt, m, topk=10, manifest=None):
        counts = Counter(pred)
        total = sum(counts.values())
        top = counts.most_common(topk)
        # If a vocab map exists, translate dense indices back to original key codes for readability.
        if manifest is not None:
            try:
                to_orig = manifest.get("vocab_maps", {}).get("key_id", {}).get("to_orig", {})
                if to_orig:
                    top_pretty = [(int(to_orig.get(int(k), k)), v) for (k, v) in top]
                    print(f"[key_id] pred total={total}, top{topk} (orig codes)={top_pretty}")
                else:
                    print(f"[key_id] pred total={total}, top{topk}={top}")
            except Exception:
                print(f"[key_id] pred total={total}, top{topk}={top}")
        else:
            print(f"[key_id] pred total={total}, top{topk}={top}")
    
    # Use the helper function for key_id reporting
    print_key_id_hist(list(kid_pred_ctr.elements()), list(kid_tgt_ctr.elements()), None, k, None)
    print(f"[key_id]     eval rows (tgt key rows) = {valid_counts['kid_eval_rows']}")
    print(f"[scroll_y] pred total={sum(sy_pred_ctr.values())}, top{k}={topk_counts(list(sy_pred_ctr.elements()), k)}, tgt top{k}={topk_counts(list(sy_tgt_ctr.elements()), k)}")
    
    # Event confusion matrix
    if all_evt_pred and all_evt_tgt:
        cm = torch.zeros(4, 4, dtype=torch.long)
        for p, t in zip(all_evt_pred, all_evt_tgt):
            cm[t, p] += 1
        print(f"[event] confusion (rows=tgt, cols=pred) - Class order: {CFG.EVENT_ORDER}:\n{cm.tolist()}")
    
    # Per-head valid fraction
    total_rows = sum(all_batch_sizes)
    print("valid rows per head:",
          f"btn={valid_counts['btn']}/{total_rows}  ka={valid_counts['ka']}/{total_rows}  kid={valid_counts['kid']}/{total_rows}  sy={valid_counts['sy']}/{total_rows}  time={valid_counts['time']}/{total_rows}")
    
    # Raw example predictions
    print("=== Raw example predictions (truncated) ===")
    for j, ex in enumerate(examples[:CFG.report_examples]):
        print(f"  ex[{j}]: {ex}")
    print("=== End detailed report ===")
    
    return {
        "valid_counts": valid_counts,
        "examples": examples,
        "neg_pred_frac": neg_pred_frac
    }
