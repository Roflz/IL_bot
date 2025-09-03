#!/usr/bin/env python3
"""
OSRS Imitation Learning — Single Workflow Model
- Standalone SequentialImitationModel (no legacy inheritance)
- Encodes last gamestate + pooled action history
- Uses SequentialActionDecoder for auto-regressive action generation
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequential_action_decoder import SequentialActionDecoder


class SequentialImitationModel(nn.Module):
    def __init__(self, data_config: dict, hidden_dim: int = 256, **kwargs):
        super().__init__()
        self.max_actions: int = int(data_config['max_actions'])
        self.enum_sizes: Dict[str, int] = dict(data_config['enum_sizes'])
        self.event_types: int = int(data_config.get('event_types', 4))
        self.temporal_window: int = int(data_config.get('temporal_window', 10))
        self.gamestate_dim: int = int(data_config['gamestate_dim'])
        self.action_feat_dim: int = int(data_config.get('action_features', 7))

        # Encoders
        self.gs_encoder = nn.Sequential(
            nn.Linear(self.gamestate_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.act_encoder = nn.Sequential(
            nn.Linear(self.action_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Decoder
        self.action_decoder = SequentialActionDecoder(
            input_dim=hidden_dim,
            max_actions=self.max_actions,
            enum_sizes=self.enum_sizes,
            event_types=self.event_types,
        )

    def forward(
        self,
        temporal_sequence: torch.Tensor,        # [B, T, Dg]
        action_sequence: torch.Tensor,          # [B, T, A, Fa]
        valid_mask: Optional[torch.Tensor] = None,
    ):
        B = temporal_sequence.size(0)

        # Gamestate: use last step features
        gs_last = temporal_sequence[:, -1, :]                    # [B, Dg]
        gs_feat = self.gs_encoder(gs_last)                       # [B, H]

        # Action history: mean over time and action slots
        assert action_sequence.dim() == 4, "action_sequence must be [B,T,A,F]"
        act_mean = action_sequence.mean(dim=(1, 2))              # [B, Fa]
        act_feat = self.act_encoder(act_mean)                    # [B, H/2]

        fused = self.fuse(torch.cat([gs_feat, act_feat], dim=-1))# [B, H]
        return self.action_decoder(fused, action_sequence, valid_mask)