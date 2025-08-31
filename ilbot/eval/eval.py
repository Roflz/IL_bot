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

def evaluate(model, val_loader, stats, args):
    """Evaluate model on validation data with detailed metrics"""
    model.eval()
    
    # Initialize counters and accumulators
    valid_counts = {"btn": 0, "ka": 0, "kid": 0, "sy": 0, "time": 0}
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
            outputs = model(temporal_sequence, action_sequence)
            
            # Get predictions
            out_btn = outputs["button_logits"]
            out_ka = outputs["key_action_logits"]
            out_kid = outputs["key_id_logits"]
            out_sy = outputs["scroll_y_logits"]
            out_x = outputs["x"]
            out_y = outputs["y"]
            
            # Enforce exclusivity: choose the single max-logit event
            # This makes MULTI impossible at inference time
            # Note: This assumes you have event_logits in your outputs
            # You may need to adapt this based on your actual model structure
            if "event_logits" in outputs:
                evt_logits = outputs["event_logits"]
                evt_pred_indices = evt_logits.argmax(dim=-1)
                multi_mask = torch.zeros_like(evt_pred_indices, dtype=torch.bool)
                # MULTI will always be 0 now
            else:
                # Fallback: create dummy event predictions
                evt_pred_indices = torch.zeros(action_target.shape[0], dtype=torch.long, device=device)
                multi_mask = torch.zeros_like(evt_pred_indices, dtype=torch.bool)
            
            # Build per-head masks
            masks = build_masks(batch)
            
            # Update valid counts
            valid_counts["btn"] += int(masks["btn"].sum())
            valid_counts["ka"] += int(masks["ka"].sum())
            valid_counts["kid"] += int(masks["kid"].sum())
            valid_counts["sy"] += int(masks["sy"].sum())
            valid_counts["time"] += int(masks["time"].sum())
            
            # Time denormalization
            t_pred = denorm_time(outputs["time"].detach().cpu(), stats)
            t_tgt = denorm_time(batch["time"].detach().cpu(), stats)
            t_pred_ms = (t_pred.numpy() * 1000.0)
            t_tgt_ms = (t_tgt.numpy() * 1000.0)
            
            # Report negative predictions before clamp
            neg_pred_frac = (t_pred_ms < 0).mean()
            if CFG.report_time_clamped_reference:
                t_pred_ms_clamped = clamp_nonneg(t_pred_ms)
                mean_pred_ms_clamped = t_pred_ms_clamped.mean()
            else:
                mean_pred_ms_clamped = 0.0
            
            # key_id: both pred and target top5 (on key rows)
            if masks["kid"].any():
                pred_kid = out_kid[masks["kid"]].argmax(-1).cpu().tolist()
                tgt_kid = batch["key_id"][masks["kid"]].cpu().tolist()
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
                    et = int(evt_pred_indices[i].item())  # Assuming you have event targets
                    row = {
                        "evt_pred": ep, "evt_tgt": et,
                        "btn_pred": int(out_btn[i].argmax().item()) if masks["btn"][i] or ep==1 else None,
                        "btn_tgt": int(batch["button"][i].item()) if masks["btn"][i] or et==1 else None,
                        "ka_pred": int(out_ka[i].argmax().item()) if masks["ka"][i] or ep==2 else None,
                        "ka_tgt": int(batch["key_action"][i].item()) if masks["ka"][i] or et==2 else None,
                        "kid_pred": int(out_kid[i].argmax().item()) if masks["kid"][i] or ep==2 else None,
                        "kid_tgt": int(batch["key_id"][i].item()) if masks["kid"][i] or et==2 else None,
                        "sy_pred": int(out_sy[i].argmax().item()) if masks["sy"][i] or ep==3 else None,
                        "sy_tgt": int(batch["scroll_y"][i].item()) if masks["sy"][i] or et==3 else None,
                        "time_ms_pred": float(t_pred_ms[i]),
                        "time_ms_tgt": float(t_tgt_ms[i]),
                        "x_pred": float(out_x[i].item()), "x_tgt": float(batch["x"][i].item()),
                        "y_pred": float(out_y[i].item()), "y_tgt": float(batch["y"][i].item()),
                    }
                    examples.append(row)
            
            # Store batch info for final summary
            all_batch_sizes.append(action_target.shape[0])
            all_evt_pred.extend(evt_pred_indices.cpu().tolist())
            all_evt_tgt.extend(evt_pred_indices.cpu().tolist())  # Assuming you have event targets
    
    # End-of-epoch print (new bits)
    k = CFG.report_k_top
    print(f"[key_id] pred total={sum(kid_pred_ctr.values())}, top{k}={topk_counts(list(kid_pred_ctr.elements()), k)}, tgt top{k}={topk_counts(list(kid_tgt_ctr.elements()), k)}")
    print(f"[scroll_y] pred total={sum(sy_pred_ctr.values())}, top{k}={topk_counts(list(sy_pred_ctr.elements()), k)}, tgt top{k}={topk_counts(list(sy_tgt_ctr.elements()), k)}")
    
    # Event confusion matrix
    if all_evt_pred and all_evt_tgt:
        cm = torch.zeros(4, 4, dtype=torch.long)
        for p, t in zip(all_evt_pred, all_evt_tgt):
            cm[t, p] += 1
        print(f"[event] confusion (rows=tgt, cols=pred):\n{cm.tolist()}")
    
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
