#!/usr/bin/env python3
"""
Sequential Action Decoder — Single Workflow
- Generates actions step-by-step until a time horizon or max_actions
- Predicts Δt quantiles in seconds; accumulates median for stopping
- Outputs per-step heads and padded tensors with shape [B, A, ...]
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialActionDecoder(nn.Module):
    def __init__(self, input_dim: int, max_actions: int, enum_sizes: dict, event_types: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_actions = int(max_actions)
        self.enum_sizes = dict(enum_sizes)
        self.event_types = int(event_types)

        # Simple step function
        hidden = self.input_dim
        self.step_net = nn.Sequential(
            nn.Linear(self.input_dim + 4, hidden),  # + timing context (current_time, remaining, step_frac, bias)
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # Heads
        self.time_head = nn.Linear(hidden, 3)               # Δt quantiles q10,q50,q90 (seconds)
        self.event_head = nn.Linear(hidden, self.event_types)

        self.x_mu_head = nn.Linear(hidden, 1)
        self.x_logsig_head = nn.Linear(hidden, 1)
        self.y_mu_head = nn.Linear(hidden, 1)
        self.y_logsig_head = nn.Linear(hidden, 1)

        self.button_head = nn.Linear(hidden, self.enum_sizes['button'])
        self.key_action_head = nn.Linear(hidden, self.enum_sizes['key_action'])
        self.key_id_head = nn.Linear(hidden, self.enum_sizes['key_id'])
        self.scroll_y_head = nn.Linear(hidden, self.enum_sizes['scroll'])

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _pad_to_max(steps_list, pad_shape, device, pad_value=0.0):
        """
        steps_list: list of [B, ...] tensors (length S <= A)
        returns [B, A, ...]
        """
        B = pad_shape[0]
        A = pad_shape[1]
        if len(steps_list) == 0:
            return torch.full(pad_shape, pad_value, device=device)

        elems = torch.stack(steps_list, dim=1)              # [B, S, ...]
        if elems.size(1) < A:
            pad_dims = list(pad_shape)
            pad_dims[0] = B
            pad_dims[1] = A - elems.size(1)
            pad = torch.full([*pad_dims], pad_value, device=device)
            elems = torch.cat([elems, pad], dim=1)
        return elems

    def forward(
        self,
        context: torch.Tensor,                    # [B, H]
        action_history: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        max_time: float = 0.6,
    ) -> Dict[str, torch.Tensor]:
        device = context.device
        B = context.size(0)
        A = self.max_actions

        # running time (seconds)
        current_time = torch.zeros(B, 1, device=device)

        # storage
        ev_logits, tdq_list, tcq_list = [], [], []
        x_mu_l, x_ls_l, y_mu_l, y_ls_l = [], [], [], []
        btn_l, ka_l, kid_l, sy_l = [], [], [], []

        # unroll
        for step in range(A):
            step_frac = torch.full((B, 1), float(step) / float(A), device=device)
            remain = torch.clamp(max_time - current_time, min=0.0)
            timing_ctx = torch.cat([current_time, remain, step_frac, torch.ones_like(step_frac)], dim=-1)  # [B,4]

            h = self.step_net(torch.cat([context, timing_ctx], dim=-1))  # [B,H]

            time_delta_q = F.softplus(self.time_head(h)) + 1e-4          # [B,3] seconds
            delta_med = time_delta_q[:, 1:2]                              # [B,1]
            next_time = current_time + delta_med                          # [B,1]

            # heads
            ev = self.event_head(h)
            x_mu = torch.sigmoid(self.x_mu_head(h)) * 1920.0
            y_mu = torch.sigmoid(self.y_mu_head(h)) * 1080.0
            x_ls = torch.clamp(self.x_logsig_head(h), -5.0, 2.0)
            y_ls = torch.clamp(self.y_logsig_head(h), -5.0, 2.0)

            btn = self.button_head(h)
            ka = self.key_action_head(h)
            kid = self.key_id_head(h)
            sy = self.scroll_y_head(h)

            # store
            ev_logits.append(ev)
            tdq_list.append(time_delta_q)                  # [B,3]
            tcq_list.append(next_time.expand(-1, 3))       # [B,3] cumulative (median replicated)
            x_mu_l.append(x_mu); x_ls_l.append(x_ls)
            y_mu_l.append(y_mu); y_ls_l.append(y_ls)
            btn_l.append(btn); ka_l.append(ka); kid_l.append(kid); sy_l.append(sy)

            # advance / stopping
            current_time = next_time
            if (current_time > max_time).all():
                break

        S = len(tdq_list)
        seq_len = torch.full((B,), S, device=device)

        preds = {
            'event_logits': self._pad_to_max(ev_logits, (B, A, self.event_types), device),
            'time_delta_q': self._pad_to_max(tdq_list, (B, A, 3), device),
            'time_cum_q':   self._pad_to_max(tcq_list, (B, A, 3), device),
            'x_mu':         self._pad_to_max(x_mu_l, (B, A, 1), device),
            'x_logsigma':   self._pad_to_max(x_ls_l, (B, A, 1), device),
            'y_mu':         self._pad_to_max(y_mu_l, (B, A, 1), device),
            'y_logsigma':   self._pad_to_max(y_ls_l, (B, A, 1), device),
            'button_logits':     self._pad_to_max(btn_l, (B, A, self.enum_sizes['button']), device),
            'key_action_logits': self._pad_to_max(ka_l, (B, A, self.enum_sizes['key_action']), device),
            'key_id_logits':     self._pad_to_max(kid_l, (B, A, self.enum_sizes['key_id']), device),
            'scroll_y_logits':   self._pad_to_max(sy_l, (B, A, self.enum_sizes['scroll']), device),
            'sequence_length':   seq_len,
        }
        return preds

