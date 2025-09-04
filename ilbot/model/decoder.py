# ilbot/model/decoder.py
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequentialActionDecoder(nn.Module):
    def __init__(self, input_dim: int, max_actions: int, enum_sizes: dict, event_types: int, horizon_s: float = 0.6, logsig_min: float = -3.0, logsig_max: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_actions = int(max_actions)
        self.enum_sizes = dict(enum_sizes)
        self.event_types = int(event_types)
        self.horizon_s = float(horizon_s)
        self.logsig_min = float(logsig_min)
        self.logsig_max = float(logsig_max)

        hidden = self.input_dim
        self.step = nn.Sequential(
            nn.Linear(self.input_dim + 4, hidden),  # context + timing ctx (current, remaining, step_frac, bias)
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # Heads
        self.time_head = nn.Linear(hidden, 3)               # Δt quantiles (seconds)
        self.event_head = nn.Linear(hidden, self.event_types)
        self.x_mu = nn.Linear(hidden, 1)
        self.x_ls = nn.Linear(hidden, 1)
        self.y_mu = nn.Linear(hidden, 1)
        self.y_ls = nn.Linear(hidden, 1)
        self.button_head = nn.Linear(hidden, self.enum_sizes['button'])
        self.key_action_head = nn.Linear(hidden, self.enum_sizes['key_action'])
        self.key_id_head = nn.Linear(hidden, self.enum_sizes['key_id'])
        self.scroll_y_head = nn.Linear(hidden, self.enum_sizes['scroll'])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # Start σ ≈ 0.1 so XY NLL gradients are moderate from step 0
        if self.x_ls.bias is not None:
            self.x_ls.bias.data.fill_(math.log(0.1))
        if self.y_ls.bias is not None:
            self.y_ls.bias.data.fill_(math.log(0.1))

    @staticmethod
    def _pad(list_of_B_tensors, pad_shape, device, pad_value=0.0):
        if len(list_of_B_tensors) == 0:
            return torch.full(pad_shape, pad_value, device=device)
        elems = torch.stack(list_of_B_tensors, dim=1)  # [B,S,...]
        if elems.size(1) < pad_shape[1]:
            pad_dims = [*pad_shape]
            pad_dims[1] = pad_shape[1] - elems.size(1)
            pad = torch.full(pad_dims, pad_value, device=device)
            elems = torch.cat([elems, pad], dim=1)
        return elems

    def forward(self, context: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        device = context.device
        B = context.size(0)
        A = self.max_actions
        current_time = torch.zeros(B, 1, device=device)

        evs, tdq, tcq = [], [], []
        xmu, xls, ymu, yls = [], [], [], []
        btn, ka, kid, sy = [], [], [], []

        for step in range(A):
            step_frac = torch.full((B, 1), float(step)/float(A), device=device)
            remaining = torch.clamp(self.horizon_s - current_time, min=0.0)
            tctx = torch.cat([current_time, remaining, step_frac, torch.ones_like(step_frac)], dim=-1)  # [B,4]

            h = self.step(torch.cat([context, tctx], dim=-1))

            time_delta_q = F.softplus(self.time_head(h)) + 1e-4  # [B,3] seconds
            delta_med = time_delta_q[:, 1:2]                     # [B,1]
            next_time = current_time + delta_med                 # [B,1]

            evs.append(self.event_head(h))
            tdq.append(time_delta_q)
            tcq.append(next_time.expand(-1, 3))                  # cumulative median replicated

            xmu.append(torch.sigmoid(self.x_mu(h)))
            ymu.append(torch.sigmoid(self.y_mu(h)))
            xls.append(torch.clamp(self.x_ls(h), self.logsig_min, self.logsig_max))
            yls.append(torch.clamp(self.y_ls(h), self.logsig_min, self.logsig_max))
            btn.append(self.button_head(h))
            ka.append(self.key_action_head(h))
            kid.append(self.key_id_head(h))
            sy.append(self.scroll_y_head(h))

            current_time = next_time
            if (current_time > self.horizon_s).all():  # stop early if everyone passed horizon
                break

        S = len(tdq)
        # derive per-sample length: count steps whose cumulative median <= horizon
        if S == 0:
            seq_len = torch.zeros(B, device=device, dtype=torch.long)
        else:
            cum = torch.stack(tcq, dim=1)[..., 0]  # [B,S], any channel is same (replicated)
            seq_len = (cum <= self.horizon_s).sum(dim=1).to(torch.long)
        
        preds = {
            "event_logits":      self._pad(evs,  (B, A, self.event_types), device),
            "time_delta_q":      self._pad(tdq,  (B, A, 3), device),
            "time_cum_q":        self._pad(tcq,  (B, A, 3), device),
            "x_mu":              self._pad(xmu,  (B, A, 1), device),
            "x_logsigma":        self._pad(xls,  (B, A, 1), device),
            "y_mu":              self._pad(ymu,  (B, A, 1), device),
            "y_logsigma":        self._pad(yls,  (B, A, 1), device),
            "button_logits":     self._pad(btn,  (B, A, self.enum_sizes['button']), device),
            "key_action_logits": self._pad(ka,   (B, A, self.enum_sizes['key_action']), device),
            "key_id_logits":     self._pad(kid,  (B, A, self.enum_sizes['key_id']), device),
            "scroll_y_logits":   self._pad(sy,   (B, A, self.enum_sizes['scroll']), device),
            "sequence_length":   seq_len,
        }
        return preds
