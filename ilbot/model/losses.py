#!/usr/bin/env python3
"""
Loss function for IL (classification + quantile time + heteroscedastic xy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from .. import config as CFG
import math

# Add debugging imports and helpers
import logging
import torch
LOG = logging.getLogger(__name__)

def _dbg_minmax_counts(name, t):
    if t is None: 
        LOG.warning("[DBG] %s=None", name); return
    x = t.detach().cpu().view(-1)
    LOG.info("[DBG] %s: min=%s max=%s uniq=%d", name, x.min().item(), x.max().item(), torch.unique(x).numel())

def _pinball(pred_q: torch.Tensor, tgt: torch.Tensor, taus: List[float]) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    # pred_q: (N, K), tgt: (N,)
    diff = tgt.unsqueeze(-1) - pred_q
    tau = pred_q.new_tensor(taus).view(1, -1)
    return torch.maximum(tau*diff, (tau-1.0)*diff).mean()

def _gauss_nll(mu: torch.Tensor, logsig: torch.Tensor, tgt: torch.Tensor, min_sigma: float) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL."""
    # Independent 1D Gaussian NLL
    sigma = torch.clamp_min(torch.exp(logsig), min_sigma)
    z = (tgt - mu) / sigma
    return (0.5*z*z + logsig + math.log(math.sqrt(2*math.pi))).mean()

def clamp_time(t, time_div, time_clip, already_scaled=False):
    """Clamp time values to reasonable range."""
    if not already_scaled:
        t = t / time_div
    return torch.clamp(t, 0.0, time_clip)

class ActionTensorLoss(nn.Module):
    """Loss function for IL (classification + quantile time + heteroscedastic xy)."""
    def __init__(self, enum_sizes: Dict[str, Dict[str, int]]):
        super().__init__()
        self.enum = enum_sizes
        self.ce_btn = nn.CrossEntropyLoss()
        self.ce_ka  = nn.CrossEntropyLoss()
        self.ce_kid = nn.CrossEntropyLoss(ignore_index=-100)  # masked when ka==NONE
        self.ce_sy  = nn.CrossEntropyLoss()
        w = torch.tensor(CFG.event_cls_weights, dtype=torch.float32)
        self.ce_event = nn.CrossEntropyLoss(weight=w)

    def forward(self, heads: Dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        m = mask
        losses = {}
        total = 0.0
        
        # --- classification heads ---
        btn_logits = heads["button_logits"][m]
        ka_logits  = heads["key_action_logits"][m]
        kid_logits = heads["key_id_logits"][m]
        sy_logits  = heads["scroll_y_logits"][m]
        btn_tgt = target[...,3].long()[m]
        ka_tgt  = target[...,4].long()[m]
        kid_tgt = target[...,5].long()[m]
        sy_tgt  = target[...,6].long()[m]
        
        # Validate and clamp classification targets to valid ranges
        if torch.any(btn_tgt < 0) or torch.any(btn_tgt >= self.enum["button"]["size"]):
            print(f"[WARN] Button targets out of range: min={btn_tgt.min()}, max={btn_tgt.max()}, size={self.enum['button']['size']}")
            btn_tgt = torch.clamp(btn_tgt, 0, self.enum["button"]["size"] - 1)
        if torch.any(ka_tgt < 0) or torch.any(ka_tgt >= self.enum["key_action"]["size"]):
            print(f"[WARN] Key action targets out of range: min={ka_tgt.min()}, max={ka_tgt.max()}, size={self.enum['key_action']['size']}")
            ka_tgt = torch.clamp(ka_tgt, 0, self.enum["key_action"]["size"] - 1)
        if torch.any(kid_tgt < 0) or torch.any(kid_tgt >= self.enum["key_id"]["size"]):
            print(f"[WARN] Key ID targets out of range: min={kid_tgt.min()}, max={kid_tgt.max()}, size={self.enum['key_id']['size']}")
            kid_tgt = torch.clamp(kid_tgt, 0, self.enum["key_id"]["size"] - 1)
        if torch.any(sy_tgt < 0) or torch.any(sy_tgt >= self.enum["scroll_y"]["size"]):
            print(f"[WARN] Scroll Y targets out of range: min={sy_tgt.min()}, max={sy_tgt.max()}, size={self.enum['scroll_y']['size']}")
            sy_tgt = torch.clamp(sy_tgt, 0, self.enum["scroll_y"]["size"] - 1)
        # key_id only when ka != NONE
        ka_none = self.enum["key_action"]["none_index"]
        kid_mask = (ka_tgt != ka_none)
        kid_logits = kid_logits[kid_mask]
        kid_tgt    = kid_tgt[kid_mask]
        if kid_logits.numel() == 0:
            kid_loss = btn_logits.sum()*0.0
        else:
            kid_loss = self.ce_kid(kid_logits, kid_tgt)
        losses["btn"] = self.ce_btn(btn_logits, btn_tgt)
        losses["ka"]  = self.ce_ka(ka_logits,  ka_tgt)
        losses["kid"] = kid_loss
        
        # Scroll-Y: CE over 3 classes (map raw {-1,0,+1} -> {0,1,2})
        if "scroll_y_logits" in heads:
            logits = heads["scroll_y_logits"]  # [B, A, 3]
            raw = target[..., 6].long() # may be {-1,0,+1} OR already {0,1,2}

            # DEBUG: reveal if values are {-1,0} only (bug) or full {0,1,2}
            _dbg_minmax_counts("tgt.scroll_y_idx(raw)", raw)

            if raw.min() < 0 or raw.max() <= 1:
                # Heuristic remap {-1,0,+1} -> {0,1,2}
                # If values are only {-1,0}, +1 is simply missing in this batch.
                mapped = raw + 1  # -1->0, 0->1, 1->2
            else:
                mapped = raw

            # Mask invalid (optional separate mask)
            m = mask if 'mask' in locals() else None
            if m is not None:
                # Flatten and set ignore_index where invalid
                ignore_tgt = mapped.clone()
                ignore_tgt[~m.bool()] = -100  # self.ignore_index
                tgt_ce = ignore_tgt
            else:
                tgt_ce = mapped

            _dbg_minmax_counts("tgt.scroll_y_idx(mapped)", tgt_ce.masked_select(tgt_ce != -100))

            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 tgt_ce.view(-1),
                                 ignore_index=-100)
            losses["sy"] = ce
        else:
            losses["sy"] = self.ce_sy(sy_logits, sy_tgt)
            
        total = total + losses["btn"] + losses["ka"] + losses["kid"] + losses["sy"]

        # Build event targets (CLICK=0, KEY=1, SCROLL=2, MOVE=3) - canonical order
        btn_none = self.enum["button"]["none_index"]
        ka_none = self.enum["key_action"]["none_index"]
        sy_none = self.enum["scroll_y"]["none_index"]
        
        # Debug: check target ranges before event construction
        if torch.any(btn_tgt < 0) or torch.any(btn_tgt >= self.enum["button"]["size"]):
            print(f"[ERROR] Button targets out of range: min={btn_tgt.min()}, max={btn_tgt.max()}, size={self.enum['button']['size']}")
        if torch.any(ka_tgt < 0) or torch.any(ka_tgt >= self.enum["key_action"]["size"]):
            print(f"[ERROR] Key action targets out of range: min={ka_tgt.min()}, max={ka_tgt.max()}, size={self.enum['key_action']['size']}")
        if torch.any(sy_tgt < 0) or torch.any(sy_tgt >= self.enum["scroll_y"]["size"]):
            print(f"[ERROR] Scroll Y targets out of range: min={sy_tgt.min()}, max={sy_tgt.max()}, size={self.enum['scroll_y']['size']}")
        
        evt_tgt = torch.where(btn_tgt != btn_none, torch.zeros_like(btn_tgt),
                   torch.where(ka_tgt != ka_none, torch.ones_like(ka_tgt),
                   torch.where(sy_tgt != sy_none, torch.full_like(sy_tgt, 2), torch.full_like(sy_tgt, 3))))
        
        # Validate event targets are in valid range [0, 3]
        if torch.any(evt_tgt < 0) or torch.any(evt_tgt > 3):
            print(f"[ERROR] Event targets out of range: min={evt_tgt.min()}, max={evt_tgt.max()}")
            print(f"[ERROR] Button targets: {btn_tgt}")
            print(f"[ERROR] Key action targets: {ka_tgt}")
            print(f"[ERROR] Scroll Y targets: {sy_tgt}")
            print(f"[ERROR] None indices: btn={btn_none}, ka={ka_none}, sy={sy_none}")
            raise ValueError(f"Invalid event targets detected: {evt_tgt.min()} to {evt_tgt.max()}")
        
        # Immediately after computing the masks
        bad_gate = torch.any((evt_pred := heads["event_logits"].argmax(-1))[mask] != evt_tgt[mask])
        print(f"[CHK] gating uses TARGETS, not predictions. If this prints True and you use evt_pred in a mask, fix it! -> {bool(bad_gate)}")

        # Event head CE (CLICK, KEY, SCROLL, MOVE) - canonical order
        ev = heads["event_logits"][m]
        
        # Right before ce_event = F.cross_entropy(...)
        with torch.no_grad():
            tgt_u, tgt_c = torch.unique(evt_tgt, return_counts=True)
            pred_top1 = ev.argmax(dim=-1)
            pred_u, pred_c = torch.unique(pred_top1, return_counts=True)

            cm = torch.zeros(4, 4, dtype=torch.long, device=ev.device)
            for t, p in zip(evt_tgt.view(-1), pred_top1.view(-1)):
                cm[t.long(), p.long()] += 1

            print(f"[CHK] evt_tgt counts: {list(zip(tgt_u.tolist(), tgt_c.tolist()))}")
            print(f"[CHK] evt_pred counts: {list(zip(pred_u.tolist(), pred_c.tolist()))}")
            print(f"[CHK] evt confmat (rows=tgt, cols=pred):\n{cm.cpu().numpy()}")

        # If using class weights:
        # Make sure 'weight' is aligned to the *target class index order*
        # For canonical ["CLICK","KEY","SCROLL","MOVE"]:
        # weight = torch.tensor([w_click, w_key, w_scroll, w_move], device=ev.device)
        w = None if CFG.event_cls_weights is None else torch.tensor(
            CFG.event_cls_weights, device=ev.device, dtype=ev.dtype
        )
        losses["event_ce"] = F.cross_entropy(ev, evt_tgt, weight=w) * CFG.event_loss_weight
        losses["event"] = losses["event_ce"] * CFG.loss_weights["event"]

        # Time: check mask/scale/clamp behavior explicitly
        if "time_q" in heads:
            tq = heads["time_q"][m]                        # (N, K)
            t_tgt = clamp_time(target[...,0], CFG.time_div, CFG.time_clip, already_scaled=True)[m]
            
            # DEBUG: check if predictions have NaNs
            if torch.isnan(tq).any():
                LOG.error("[DBG] time_q has NaNs")
            LOG.info("[DBG] time_q stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                     float(tq.detach().mean()), float(tq.detach().std()),
                     float(tq.detach().min()), float(tq.detach().max()))
            LOG.info("[DBG] time_tgt stats:  mean=%.4f std=%.4f min=%.4f max=%.4f uniq=%d",
                     float(t_tgt.detach().mean()), float(t_tgt.detach().std()),
                     float(t_tgt.detach().min()), float(t_tgt.detach().max()),
                     int(torch.unique(t_tgt.detach()).numel()))

            # Ensure mask is not wiping all time samples
            kept = int(m.detach().sum().item())
            LOG.info("[DBG] time_mask keeps %d / %d samples", kept, int(m.numel()))
            if kept == 0:
                LOG.warning("[DBG] time head: mask drops ALL samples — time loss = 0, no learning")
                # fall through without indexing so you still see pred/tgt stats below
            # Note: tq and t_tgt are already masked by 'm' above, so we don't need to mask again
            
            # After you build time_mask and before the pinball loss
            time_mask = m
            tq_pre = heads.get("time_q_pre", None)
            if tq_pre is not None:
                tq_pre = tq_pre[m]
            
            def s(x):  # compact stats
                return (float(x.min()), float(x.mean()), float(x.max()))

            print(f"[CHK] time stats pre: {s(tq_pre) if tq_pre is not None else 'None'}  post: {s(tq)}  tgt: {s(t_tgt)}  (N={int(time_mask.sum())})")
                
            losses["time_pinball"] = _pinball(tq, t_tgt, CFG.time_quantiles)
            total = total + losses["time_pinball"] * CFG.loss_weights["time"]

        # XY regression (pixels)
        if "x_mu" in heads and "y_mu" in heads:
            x_mu, y_mu = heads["x_mu"][m], heads["y_mu"][m]
            x_ls, y_ls = heads["x_logsig"][m], heads["y_logsig"][m]
            
            # Clamp log_sigma to prevent numerical instability
            x_ls = torch.clamp(x_ls, -10.0, 10.0)  # Prevent extreme values
            y_ls = torch.clamp(y_ls, -10.0, 10.0)  # Prevent extreme values
            
            x_sig = (x_ls.exp()).clamp_min(CFG.xy_min_sigma)
            y_sig = (y_ls.exp()).clamp_min(CFG.xy_min_sigma)
            
            x = target[...,1][m]; y = target[...,2][m]
            
            LOG.info("[DBG] XY: pred range x=[%.1f, %.1f] y=[%.1f, %.1f] | tgt x=[%.1f, %.1f] y=[%.1f, %.1f]",
                     float(x_mu.min()), float(x_mu.max()),
                     float(y_mu.min()), float(y_mu.max()),
                     float(x.min()), float(x.max()),
                     float(y.min()), float(y.max()))
        
        # Add numerical stability checks
        if torch.any(torch.isnan(x_mu)) or torch.any(torch.isnan(y_mu)):
            print(f"[WARN] NaN detected in x_mu or y_mu")
            x_mu = torch.nan_to_num(x_mu, nan=0.0)
            y_mu = torch.nan_to_num(y_mu, nan=0.0)
        
        if torch.any(torch.isnan(x_sig)) or torch.any(torch.isnan(y_sig)):
            print(f"[WARN] NaN detected in x_sig or y_sig")
            x_sig = torch.nan_to_num(x_sig, nan=CFG.xy_min_sigma)
            y_sig = torch.nan_to_num(y_sig, nan=CFG.xy_min_sigma)
        
        nll_x = 0.5*(((x - x_mu)/x_sig)**2 + 2*x_ls + math.log(2*math.pi))
        nll_y = 0.5*(((y - y_mu)/y_sig)**2 + 2*y_ls + math.log(2*math.pi))
        
        # Check for NaN in NLL computation
        if torch.any(torch.isnan(nll_x)) or torch.any(torch.isnan(nll_y)):
            print(f"[WARN] NaN detected in NLL computation")
            nll_x = torch.nan_to_num(nll_x, nan=0.0)
            nll_y = torch.nan_to_num(nll_y, nan=0.0)
        
        losses["xy_nll"] = nll_x.mean() + nll_y.mean()
        total = total + losses["xy_nll"] * CFG.loss_weights["xy"]
        
        # Final NaN check on total loss
        if torch.isnan(total):
            print(f"[ERROR] Total loss is NaN! Individual losses: {losses}")
            # Return a safe fallback loss
            total = torch.tensor(1.0, device=total.device, requires_grad=True)
        
        return total, {k: float(v.item()) for k,v in losses.items()}



