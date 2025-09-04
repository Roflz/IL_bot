# ilbot/model/model.py
from typing import Dict, Optional
import torch
import torch.nn as nn
from ilbot.model.decoder import SequentialActionDecoder

class SequentialImitationModel(nn.Module):
    """
    Minimal, single-workflow model:
    - Encode last gamestate [B,Dg] → [B,H]
    - Encode pooled action history [B,Fa] → [B,H/2]
    - Fuse → decoder
    """
    def __init__(self, data_config: dict, hidden_dim: int = 256, horizon_s: float = 0.6, enum_sizes: dict | None = None, logsig_min: float = -3.0, logsig_max: float = 0.0):
        super().__init__()
        self.Dg = int(data_config["gamestate_dim"])
        self.Fa = int(data_config["action_features"])
        self.A  = int(data_config["max_actions"])
        self.T  = int(data_config["temporal_window"])
        self.event_types = int(data_config.get("event_types", 4))
        enum_sizes = enum_sizes or {"button":3, "key_action":3, "key_id":505, "scroll":3}

        # Pre-normalize *inputs* before the first Linear to avoid FP16 overflow
        self.gs_enc = nn.Sequential(
            nn.LayerNorm(self.Dg),
            nn.Linear(self.Dg, hidden_dim),
            nn.ReLU(),
        )
        self.act_enc = nn.Sequential(
            nn.LayerNorm(self.Fa),
            nn.Linear(self.Fa, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim + hidden_dim // 2),
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = SequentialActionDecoder(input_dim=hidden_dim, max_actions=self.A, enum_sizes=enum_sizes, event_types=self.event_types, horizon_s=horizon_s, logsig_min=logsig_min, logsig_max=logsig_max)

    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = temporal_sequence.size(0)
        gs_last = temporal_sequence[:, -1, :]              # [B,Dg]
        gs_feat = self.gs_enc(gs_last)                     # [B,H]
        
        # Action history encoder (pooled)  [B, Fa] -> [B, H//2]
        # SAFE POOLING: avoid division-by-zero when a sequence has no valid actions
        # Use last timestep actions and pool across A with the provided valid_mask.
        B, T, A, F = action_sequence.shape
        last_actions = action_sequence[:, -1, :, :]                 # [B, A, Fa]
        maskA = valid_mask.to(last_actions.dtype).unsqueeze(-1)     # [B, A, 1]
        num = (last_actions * maskA).sum(dim=1)                     # [B, Fa]
        den = maskA.sum(dim=1).clamp_min(1.0)                       # [B, 1]
        pooled = num / den                                          # [B, Fa]
        # Fail fast if anything went non-finite
        if not torch.isfinite(pooled).all():
            bad = ~torch.isfinite(pooled)
            raise RuntimeError(
                f"Action pooling produced non-finite values: count={int(bad.sum())}. "
                f"Hint: some sequences may have zero valid actions; pooling now guards via clamp_min(1.0)."
            )
        act_feat = self.act_enc(pooled)                  # [B,H/2]
        fused = self.fuse(torch.cat([gs_feat, act_feat], dim=-1))  # [B,H]
        return self.decoder(fused, valid_mask)
