#!/usr/bin/env python3
"""
Unified Loss — Single Workflow
Computes:
- Event classification (weighted CE)
- Timing pinball loss on Δt quantiles (seconds)
- XY Gaussian NLL (pixels)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PinballLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.q = torch.tensor(list(quantiles), dtype=torch.float32)

    def forward(self, pred_q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred_q: [B, A, 3], target: [B, A], mask: [B, A] (bool)
        """
        q = self.q.to(pred_q.device).view(1, 1, 3)
        diff = target.unsqueeze(-1) - pred_q
        loss = torch.maximum(q * diff, (1 - q) * (-diff))  # [B,A,3]
        if mask is not None:
            loss = loss.masked_fill(~mask.unsqueeze(-1), 0.0)
            denom = mask.sum().clamp_min(1)
            return loss.sum() / denom
        return loss.mean()


def gaussian_nll(mu: torch.Tensor, logsig: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mu, logsig, target: [B,A] each; mask: [B,A] bool
    """
    var = torch.exp(2.0 * logsig)
    nll = 0.5 * ((target - mu) ** 2 / var + 2.0 * logsig)
    nll = nll.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp_min(1)
    return nll.sum() / denom


class AdvancedUnifiedEventLoss(nn.Module):
    def __init__(
        self,
        event_weight: float = 1.0,
        timing_weight: float = 0.5,
        xy_weight: float = 0.1,
        quantiles=(0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.event_weight = float(event_weight)
        self.timing_weight = float(timing_weight)
        self.xy_weight = float(xy_weight)
        self.pinball = PinballLoss(quantiles)
        self.register_buffer('event_class_weights', None)

    def set_event_class_weights(self, weights: torch.Tensor):
        self.register_buffer('event_class_weights', weights)

    @staticmethod
    def build_event_targets(targets: torch.Tensor) -> torch.Tensor:
        """
        targets: [B,A,7] -> event_target [B,A] with mapping:
        0=CLICK, 1=KEY, 2=SCROLL, 3=MOVE (default)
        """
        B, A, _ = targets.shape
        device = targets.device
        ev = torch.full((B, A), 3, dtype=torch.long, device=device)  # MOVE default
        button = targets[..., 3].long()
        key_action = targets[..., 4].long()
        key_id = targets[..., 5].long()
        scroll_y = targets[..., 6].long()

        ev = torch.where(button != -1, torch.zeros_like(ev), ev)               # CLICK
        ev = torch.where(key_action != -1, torch.ones_like(ev), ev)            # KEY
        ev = torch.where(scroll_y != 0, torch.full_like(ev, 2), ev)            # SCROLL
        return ev

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,               # [B,A,7]
        valid_mask: torch.Tensor,            # [B,A] bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        losses = {}

        # 1) Event CE
        ev_logits = predictions['event_logits']                 # [B,A,4]
        ev_tgt = self.build_event_targets(targets)              # [B,A]
        ce = F.cross_entropy(
            ev_logits.view(-1, ev_logits.size(-1)),
            ev_tgt.view(-1),
            weight=self.event_class_weights,
            reduction='none',
        ).view_as(ev_tgt)
        ce = ce.masked_fill(~valid_mask, 0.0)
        denom = valid_mask.sum().clamp_min(1)
        losses['event_ce'] = ce.sum() / denom

        # 2) Timing pinball on Δt (seconds)
        dt_pred = predictions['time_delta_q']                   # [B,A,3]
        dt_tgt = targets[..., 0] / 1000.0                       # [B,A] ms -> s
        losses['timing_pinball'] = self.pinball(dt_pred, dt_tgt, valid_mask)

        # 3) XY Gaussian NLL (pixels)
        x_mu  = predictions['x_mu'].squeeze(-1)
        x_ls  = predictions['x_logsigma'].squeeze(-1)
        y_mu  = predictions['y_mu'].squeeze(-1)
        y_ls  = predictions['y_logsigma'].squeeze(-1)
        x_tgt = targets[..., 1]
        y_tgt = targets[..., 2]

        losses['x_gaussian_nll'] = gaussian_nll(x_mu, x_ls, x_tgt, valid_mask)
        losses['y_gaussian_nll'] = gaussian_nll(y_mu, y_ls, y_tgt, valid_mask)

        total = (
            self.event_weight * losses['event_ce']
            + self.timing_weight * losses['timing_pinball']
            + self.xy_weight * (losses['x_gaussian_nll'] + losses['y_gaussian_nll'])
        )
        losses['total'] = total
        return total, losses

