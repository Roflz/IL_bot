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

        # force FP32 math even under autocast for numerical stability
        pred_q = pred_q.float()
        target = target.float()
        mask = mask.bool()
        q = self.q.to(pred_q.device).view(1,1,3).float()
        diff = target.unsqueeze(-1) - pred_q
        loss = torch.maximum(q*diff, (1 - q)*(-diff))  # [B,A,3]
        loss = loss.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom

def gaussian_nll(mu: torch.Tensor, logsig: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mu, logsig, target: [B,A]; mask: [B,A] bool
    """
    assert mu.shape == logsig.shape == target.shape == mask.shape
    # force FP32 math even under autocast for numerical stability
    mu = mu.float()
    logsig = logsig.float()
    target = target.float()
    mask = mask.bool()
    var = (2.0 * logsig).exp().clamp_min(1e-12)
    nll = 0.5 * ((target - mu) ** 2) / var + logsig + 0.5 * math.log(2.0 * math.pi)
    # mask: [B,A] -> broadcast to heads if needed
    if nll.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    nll = nll.masked_fill(~mask, 0.0)
    denom = mask.float().sum().clamp_min(1.0)
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

        # ensure FP32 for all loss math
        event_logits = pred["event_logits"].float()
        time_delta_q = pred["time_delta_q"].float()
        x_mu, y_mu = pred["x_mu"].float(), pred["y_mu"].float()
        x_logsig, y_logsig = pred["x_logsigma"].float(), pred["y_logsigma"].float()
        mask = valid_mask.bool()

        # 1) Event CE
        ev_logits = event_logits      # [B,A,4]
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
        dt_pred = time_delta_q        # [B,A,3]
        dt_tgt_s = targets[...,0]                   # already in seconds
        comps["timing_pinball"] = self.pinball(dt_pred, dt_tgt_s, mask)

        # 3) XY Gaussian NLL on normalized coords [0,1]
        x_mu_sq = x_mu.squeeze(-1)
        x_ls_sq = x_logsig.squeeze(-1)
        y_mu_sq = y_mu.squeeze(-1)
        y_ls_sq = y_logsig.squeeze(-1)
        x_tgt = targets[...,1]
        y_tgt = targets[...,2]
        comps["x_gaussian_nll"] = gaussian_nll(x_mu_sq, x_ls_sq, x_tgt, mask)
        comps["y_gaussian_nll"] = gaussian_nll(y_mu_sq, y_ls_sq, y_tgt, mask)

        total = (
            self.event_weight * comps["event_ce"]
            + self.timing_weight * comps["timing_pinball"]
            + self.xy_weight * (comps["x_gaussian_nll"] + comps["y_gaussian_nll"])
        )
        # Optional quantile-crossing penalty: enforce q10 ≤ q50 ≤ q90
        if self.qc_weight > 0:
            dq = time_delta_q
            q10, q50, q90 = dq[...,0], dq[...,1], dq[...,2]
            qc = (F.relu(q10 - q50) + F.relu(q50 - q90)).masked_fill(~mask, 0.0)
            denom = mask.sum().clamp_min(1)
            comps["quantile_penalty"] = qc.sum() / denom
            total = total + self.qc_weight * comps["quantile_penalty"]
        comps["total"] = total
        return comps
