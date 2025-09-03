# ilbot/training/losses.py
from typing import Dict, Tuple, Optional
import math
LOGSIG_MIN = -3.0   # σ >= ~0.05 in normalized [0,1] space
LOGSIG_MAX =  0.0   # σ <= 1.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from ilbot.data.contracts import derive_event_targets_from_marks

class PinballLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.q = torch.tensor(list(quantiles), dtype=torch.float32)

    def forward(self, pred_q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred_q: [B,A,3], target: [B,A] (seconds), mask: [B,A] bool
        """
        assert pred_q.ndim == 3 and pred_q.size(-1) == 3
        assert target.shape == pred_q.shape[:-1]
        assert mask.shape == target.shape and mask.dtype == torch.bool

        q = self.q.to(pred_q.device).view(1,1,3)
        diff = target.unsqueeze(-1) - pred_q
        loss = torch.maximum(q*diff, (1 - q)*(-diff))  # [B,A,3]
        loss = loss.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum().clamp_min(1)
        return loss.sum() / denom

def gaussian_nll(mu: torch.Tensor, logsig: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mu, logsig, target: [B,A]; mask: [B,A] bool
    """
    assert mu.shape == logsig.shape == target.shape == mask.shape
    # Fail-fast sanity for inputs
    if not (torch.isfinite(mu).all() and torch.isfinite(logsig).all() and torch.isfinite(target).all()):
        raise RuntimeError("Non-finite input to gaussian_nll (mu/logsig/target)")
    
    # Hard guard: re-clamp logsig here to avoid overflow under AMP, even if decoder regresses
    logsig = torch.clamp(logsig, LOGSIG_MIN, LOGSIG_MAX)
    var = torch.exp(2.0 * logsig) + 1e-8
    nll = 0.5 * ((target - mu) ** 2 / var) + logsig + 0.5 * math.log(2.0 * math.pi)
    if not torch.isfinite(nll).all():
        raise RuntimeError("Non-finite nll inside gaussian_nll (after clamp)")
    # mask: [B,A] -> broadcast to heads if needed
    if nll.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    nll = torch.where(mask, nll, torch.zeros_like(nll))
    denom = mask.sum()
    if denom.item() == 0:
        raise RuntimeError("gaussian_nll: valid_mask has zero true entries (division by zero).")
    return nll.sum() / denom

class AdvancedUnifiedEventLoss(nn.Module):
    def __init__(self, event_weight=1.0, timing_weight=0.5, xy_weight=0.1,
                 quantiles=(0.1,0.5,0.9), qc_weight: float = 0.0):
        super().__init__()
        self.event_weight = float(event_weight)
        self.timing_weight = float(timing_weight)
        self.xy_weight = float(xy_weight)
        self.qc_weight = float(qc_weight)
        self.pinball = PinballLoss(quantiles)
        self.register_buffer('event_class_weights', None)

    def set_event_class_weights(self, weights: torch.Tensor):
        # weights: [4] on correct device/dtype
        self.register_buffer('event_class_weights', weights)

    def forward(self, pred: Dict[str, torch.Tensor], targets: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Pre-checks: label precisely which head is bad
        for k in ("event_logits","time_delta_q","x_mu","y_mu","x_logsigma","y_logsigma"):
            if k in pred and not torch.isfinite(pred[k]).all():
                raise RuntimeError(f"Non-finite tensor handed to loss: pred['{k}']")
        if not (torch.isfinite(targets).all() and torch.isfinite(valid_mask.float()).all()):
            raise RuntimeError("Non-finite targets or valid_mask handed to loss")
        
        mask = valid_mask
        vc = int(mask.sum().item())
        if vc == 0:
            # Special case: if all components are zero, return zero loss (for testing)
            zero_loss = torch.tensor(0.0, device=targets.device, dtype=targets.dtype)
            zero_comps = {
                "event_ce": zero_loss,
                "timing_pinball": zero_loss, 
                "x_gaussian_nll": zero_loss,
                "y_gaussian_nll": zero_loss,
                "total": zero_loss,
            }
            return zero_comps
        
        comps: Dict[str, torch.Tensor] = {}

        # 1) Event CE
        ev_logits = pred["event_logits"]      # [B,A,4]
        ev_tgt = derive_event_targets_from_marks(targets)  # [B,A] (0=CLICK,1=KEY,2=SCROLL,3=MOVE)
        ce = F.cross_entropy(
            ev_logits.view(-1, ev_logits.size(-1)),
            ev_tgt.view(-1),
            weight=self.event_class_weights,
            reduction='none',
        ).view_as(ev_tgt)
        ce = ce.masked_fill(~mask, 0.0)
        denom = mask.sum().clamp_min(1)
        comps["event_ce"] = ce.sum() / denom

        # 2) Timing pinball on Δt (seconds)
        dt_pred = pred["time_delta_q"]        # [B,A,3]
        dt_tgt_s = targets[...,0]                   # already in seconds
        comps["timing_pinball"] = self.pinball(dt_pred, dt_tgt_s, mask)

        # 3) XY Gaussian NLL on normalized coords [0,1]
        x_mu  = pred["x_mu"].squeeze(-1)
        x_ls  = pred["x_logsigma"].squeeze(-1)
        y_mu  = pred["y_mu"].squeeze(-1)
        y_ls  = pred["y_logsigma"].squeeze(-1)
        x_tgt = targets[...,1]
        y_tgt = targets[...,2]
        comps["x_gaussian_nll"] = gaussian_nll(x_mu, x_ls, x_tgt, mask)
        comps["y_gaussian_nll"] = gaussian_nll(y_mu, y_ls, y_tgt, mask)

        total = (
            self.event_weight * comps["event_ce"]
            + self.timing_weight * comps["timing_pinball"]
            + self.xy_weight * (comps["x_gaussian_nll"] + comps["y_gaussian_nll"])
        )
        # Optional quantile-crossing penalty: enforce q10 ≤ q50 ≤ q90
        if self.qc_weight > 0:
            dq = pred["time_delta_q"]
            q10, q50, q90 = dq[...,0], dq[...,1], dq[...,2]
            qc = (F.relu(q10 - q50) + F.relu(q50 - q90)).masked_fill(~mask, 0.0)
            denom = mask.sum().clamp_min(1)
            comps["quantile_penalty"] = qc.sum() / denom
            total = total + self.qc_weight * comps["quantile_penalty"]
        comps["total"] = total
        return comps
