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
    def __init__(self, data_config: dict, hidden_dim: int = 256, horizon_s: float = 0.6, enum_sizes: dict | None = None):
        super().__init__()
        self.Dg = int(data_config["gamestate_dim"])
        self.Fa = int(data_config["action_features"])
        self.A  = int(data_config["max_actions"])
        self.T  = int(data_config["temporal_window"])
        self.event_types = int(data_config.get("event_types", 4))
        enum_sizes = enum_sizes or {"button":3, "key_action":3, "key_id":505, "scroll":3}

        self.gs_enc = nn.Sequential(nn.Linear(self.Dg, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.act_enc = nn.Sequential(nn.Linear(self.Fa, hidden_dim//2), nn.ReLU(), nn.LayerNorm(hidden_dim//2))
        self.fuse = nn.Sequential(nn.Linear(hidden_dim + hidden_dim//2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

        self.decoder = SequentialActionDecoder(input_dim=hidden_dim, max_actions=self.A, enum_sizes=enum_sizes, event_types=self.event_types, horizon_s=horizon_s)

    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = temporal_sequence.size(0)
        gs_last = temporal_sequence[:, -1, :]              # [B,Dg]
        gs_feat = self.gs_enc(gs_last)                     # [B,H]
        act_mean = action_sequence.mean(dim=(1,2))         # [B,Fa]
        act_feat = self.act_enc(act_mean)                  # [B,H/2]
        fused = self.fuse(torch.cat([gs_feat, act_feat], dim=-1))  # [B,H]
        return self.decoder(fused, valid_mask)
