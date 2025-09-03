# ilbot/training/losses.py
from typing import Dict, Tuple, Optional
import math
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
    var = torch.exp(2.0 * logsig).clamp_min(1e-6)
    # Full Normal NLL: 0.5*((x-μ)^2/σ^2) + log σ + 0.5*log(2π)
    nll = 0.5 * ((target - mu)**2 / var) + logsig + 0.5 * math.log(2.0 * math.pi)
    nll = nll.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp_min(1)
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

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        targets = batch["targets"]            # [B,A,7]
        mask = batch["valid_mask"]            # [B,A] bool
        comps: Dict[str, torch.Tensor] = {}

        # 1) Event CE
        ev_logits = predictions["event_logits"]      # [B,A,4]
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
        dt_pred = predictions["time_delta_q"]        # [B,A,3]
        dt_tgt_s = targets[...,0]                   # already in seconds
        comps["timing_pinball"] = self.pinball(dt_pred, dt_tgt_s, mask)

        # 3) XY Gaussian NLL on normalized coords [0,1]
        x_mu  = predictions["x_mu"].squeeze(-1)
        x_ls  = predictions["x_logsigma"].squeeze(-1)
        y_mu  = predictions["y_mu"].squeeze(-1)
        y_ls  = predictions["y_logsigma"].squeeze(-1)
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
            dq = predictions["time_delta_q"]
            q10, q50, q90 = dq[...,0], dq[...,1], dq[...,2]
            qc = (F.relu(q10 - q50) + F.relu(q50 - q90)).masked_fill(~batch["valid_mask"], 0.0)
            denom = batch["valid_mask"].sum().clamp_min(1)
            comps["quantile_penalty"] = qc.sum() / denom
            total = total + self.qc_weight * comps["quantile_penalty"]
        comps["total"] = total
        return total, comps
