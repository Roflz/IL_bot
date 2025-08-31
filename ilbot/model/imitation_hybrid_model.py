#!/usr/bin/env python3
"""
Imitation Learning Hybrid Model: Transformer + CNN + LSTM
OSRS Bot that learns to play like you
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Literal
from .. import config as CFG

# Add debugging imports
import logging
import torch
import torch.nn.functional as F
LOG = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for gamestate features"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        return output

class GamestateEncoder(nn.Module):
    """Encoder for gamestate features with feature-type-specific encoding and self-attention"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_heads: int = 8, feature_spec: dict | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_spec = feature_spec or {}
        groups = (self.feature_spec.get("group_indices") or {})
        self.idx_cat   = list(feature_spec["group_indices"].get("categorical", []))
        self.idx_cont  = list(feature_spec["group_indices"].get("continuous", []))
        self.idx_bool  = list(feature_spec["group_indices"].get("boolean", []))
        self.idx_count = list(feature_spec["group_indices"].get("counts", []))
        self.idx_angle = list(feature_spec["group_indices"].get("angles", []))
        self.idx_time  = list(feature_spec["group_indices"].get("time", []))

        n_cont  = len(self.idx_cont)
        n_bool  = len(self.idx_bool)
        n_count = len(self.idx_count)
        n_angle = len(self.idx_angle)
        n_time  = len(self.idx_time)


        # Categorical embedding: one shared table with per-column offsets
        total_vocab = int(self.feature_spec.get("total_cat_vocab", 0))
        emb_dim_cat = max(hidden_dim // 16, 8)
        self.has_cat = bool(self.idx_cat) and total_vocab > 0
        if self.has_cat:
            self.cat_offsets = torch.tensor(self.feature_spec["cat_offsets"], dtype=torch.long)  # len = n_cat_cols
            self.cat_unknowns = torch.tensor(self.feature_spec["unknown_index_per_field"], dtype=torch.long)
            self.categorical_emb = nn.Embedding(total_vocab, emb_dim_cat)

        # small MLPs per group (reuse your existing heads if you already had them)
        emb_dim_cont = max(hidden_dim // 8, 16)
        emb_dim_bool = max(hidden_dim // 16, 8)
        emb_dim_count = max(hidden_dim // 16, 8)
        emb_dim_angle = max(hidden_dim // 16, 8)
        emb_dim_time = max(hidden_dim // 16, 8)
        
        # Continuous/Coordinate features
        self._cont_mlp = nn.Sequential(
            nn.Linear(max(1, n_cont), emb_dim_cont),
            nn.LayerNorm(emb_dim_cont),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Boolean features
        self._bool_mlp = nn.Sequential(
            nn.Linear(max(1, n_bool), emb_dim_bool),
            nn.LayerNorm(emb_dim_bool),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Count features
        self._count_mlp = nn.Sequential(
            nn.Linear(max(1, n_count), emb_dim_count),
            nn.LayerNorm(emb_dim_count),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Time features (timestamps, durations)
        self._time_mlp = nn.Sequential(
            nn.Linear(max(1, n_time), emb_dim_time),
            nn.LayerNorm(emb_dim_time),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Angle features (camera pitch/yaw)
        self._angle_mlp = nn.Sequential(
            nn.Linear(max(1, n_angle), emb_dim_angle),
            nn.LayerNorm(emb_dim_angle),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature combiner
        in_dims = []
        if self.has_cat:    in_dims.append(emb_dim_cat * len(self.idx_cat))
        if self.idx_cont:   in_dims.append(emb_dim_cont)
        if self.idx_bool:   in_dims.append(emb_dim_bool)
        if self.idx_count:  in_dims.append(emb_dim_count)
        if self.idx_angle:  in_dims.append(emb_dim_angle)
        if self.idx_time:   in_dims.append(emb_dim_time)
        fused = sum(in_dims) if in_dims else hidden_dim
        self.fuse = nn.Linear(fused, hidden_dim)
        
        # Self-attention layers
        self.attention1 = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.attention2 = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        B, T, D = x.shape
        chunks = []
        
        if self.has_cat and self.idx_cat:
            # Gather categorical columns
            cats = x[..., self.idx_cat].long()                        # (B,T,n_cat)
            # Map raw ids → field-local ids if you pre-remap offline; otherwise, cap negatives to UNKNOWN
            cats = torch.where(cats < 0, torch.zeros_like(cats), cats)
            # Apply per-column offsets into shared table
            device = cats.device
            offsets = self.cat_offsets.to(device)                     # (n_cat,)
            cats_off = cats + offsets.view(1,1,-1)                    # (B,T,n_cat)
            # Unknown handling: any id >= field vocab becomes UNKNOWN (last bin in that field)
            # Build per-column upper bounds: offset + (vocab_size-1) -- unknown index
            unknowns = self.cat_unknowns.to(device) + offsets.to(device)
            # mask out-of-range (>= unknown index) to UNKNOWN (exact index)
            cats_off = torch.minimum(cats_off, unknowns.view(1,1,-1))
            cat_emb = self.categorical_emb(cats_off)                   # (B,T,n_cat,emb)
            chunks.append(cat_emb.reshape(B, T, -1))

        # Continuous
        if self.idx_cont:
            cont = x[..., self.idx_cont].float()
            # normalize if you already had stats; else passthrough
            cont = self._cont_mlp(cont)  # define this mlp as you already had
            chunks.append(cont)
        
        # Bool
        if self.idx_bool:
            boo = x[..., self.idx_bool].float()
            boo = self._bool_mlp(boo)
            chunks.append(boo)
        
        # Counts
        if self.idx_count:
            cnt = x[..., self.idx_count].float()
            cnt = self._count_mlp(cnt)
            chunks.append(cnt)
        
        # Angles
        if self.idx_angle:
            ang = x[..., self.idx_angle].float()
            ang = self._angle_mlp(ang)
            chunks.append(ang)
        
        # Time
        if self.idx_time:
            tim = x[..., self.idx_time].float()
            tim = self._time_mlp(tim)
            chunks.append(tim)

        h = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
        x_encoded = torch.relu(self.fuse(h))
        
        # Self-attention with residual connection
        attn_out = self.attention1(x_encoded, x_encoded, x_encoded)
        x_encoded = self.norm1(x_encoded + attn_out)
        
        # Second attention layer
        attn_out = self.attention2(x_encoded, x_encoded, x_encoded)
        x_encoded = self.norm1(x_encoded + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x_encoded)
        x_encoded = self.norm2(x_encoded + ffn_out)
        
        return x_encoded

class ScreenshotEncoder(nn.Module):
    """CNN encoder for screenshot features"""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Initial conv: 224x224 -> 112x112
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            
            # Second conv: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            
            # Third conv: 14x14 -> 7x7
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global average pooling and flatten
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Final projection to output dimension
        self.final_proj = nn.Linear(256, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.final_proj(x)
        return x

class TemporalEncoder(nn.Module):
    """LSTM encoder for temporal gamestate sequences"""
    
    def __init__(self, input_dim: int = 73, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional output to single dimension
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # Concatenate last layer bidirectional
        output = self.output_proj(last_hidden)
        
        return output

class CrossAttention(nn.Module):
    """Cross-attention between gamestate and screenshot features"""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
        
    def forward(self, gamestate_features: torch.Tensor, screenshot_features: torch.Tensor) -> torch.Tensor:
        # gamestate_features: (batch_size, 1, embed_dim)
        # screenshot_features: (batch_size, 1, embed_dim)
        
        # Cross-attention: gamestate attends to screenshot
        cross_features = self.attention(gamestate_features, screenshot_features, screenshot_features)
        return cross_features

class ActionDecoder(nn.Module):
    """V1/V2 decoder with dynamic sizes taken from manifest/config."""
    def __init__(self, input_dim, *,
                 max_actions: int,
                 head_version: str,
                 enum_sizes: dict,
                 use_log1p_time: bool = True,
                 time_div_ms: int = 1000,
                 time_positive: bool = False,
                 args=None):
        super().__init__()
        self.max_actions = max_actions
        self.head_version = head_version
        self.enum_sizes = enum_sizes  # dict of sizes & indices
        self.use_log1p_time = use_log1p_time
        self.time_div_ms = time_div_ms
        # remember flags
        self.time_positive = bool(getattr(args, 'time_positive', False)) if args else time_positive


        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # Expand to learn richer features
            nn.GELU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(input_dim * 2, input_dim * 2),  # Deeper representation
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),  # Project back to original dimension
        )
        # Common regressors (no hardcoding on A)

        
        # --- new heads ---
        # 4-way event: [CLICK, KEY, SCROLL, MOVE] (canonical order)
        self.event_head = nn.Linear(input_dim, max_actions * 4)
        # Time quantiles: K outputs
        self.time_q_head = nn.Linear(input_dim, max_actions * len(CFG.time_quantiles))
        # Heteroscedastic XY: predict μ and log σ for each
        self.xy_mu_head   = nn.Linear(input_dim, max_actions * 2)   # [μx, μy]
        self.xy_lsig_head = nn.Linear(input_dim, max_actions * 2)   # [log σx, log σy]
        

        if head_version == "v1":
            n_type = int(enum_sizes.get("action_type", {"size": 5})["size"])
            n_btn  = int(enum_sizes.get("button", {"size": 4})["size"])
            n_key  = int(enum_sizes.get("key_id", {"size": 151})["size"])
            n_sy   = int(enum_sizes.get("scroll_y", {"size": 3})["size"])
            self.action_type_head = nn.Linear(input_dim, max_actions * n_type)
            self.button_head      = nn.Linear(input_dim, max_actions * n_btn)
            self.key_head         = nn.Linear(input_dim, max_actions * n_key)
            self.scroll_y_head    = nn.Linear(input_dim, max_actions * n_sy)
            self.scroll_x_head    = nn.Linear(input_dim, max_actions * 3)  # ignored in loss; legacy buffer
        else:
            def _get_size(d, k, default):
                v = d.get(k, default)
                return int(v.get("size", v)) if isinstance(v, dict) else int(v)
            def _get_none(d, k, default):
                v = d.get(k, {})
                return int(v.get("none_index", default)) if isinstance(v, dict) else int(default)
            
            n_btn = _get_size(enum_sizes, "button", 4)
            n_ka  = _get_size(enum_sizes, "key_action", 3)
            n_kid = _get_size(enum_sizes, "key_id", 151)
            n_sy  = _get_size(enum_sizes, "scroll_y", 3)
            self.sy_none_index = _get_none(enum_sizes, "scroll_y", 1)  # store for inference mapping
            
            self.button_head      = nn.Linear(input_dim, max_actions * n_btn)
            self.key_action_head  = nn.Linear(input_dim, max_actions * n_ka)
            self.key_id_head      = nn.Linear(input_dim, max_actions * n_kid)
            self.scroll_y_head    = nn.Linear(input_dim, max_actions * n_sy)
        
        # Initialize weights properly to prevent collapse
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent collapse and improve training stability"""
        # Event head: small random weights to break symmetry
        nn.init.xavier_uniform_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)
        
        # Time head: small positive bias to prevent all-zeros
        nn.init.xavier_uniform_(self.time_q_head.weight)
        nn.init.constant_(self.time_q_head.bias, 0.1)
        
        # XY heads: small weights to prevent extreme outputs
        nn.init.xavier_uniform_(self.xy_mu_head.weight)
        nn.init.zeros_(self.xy_mu_head.bias)
        nn.init.xavier_uniform_(self.xy_lsig_head.weight)
        nn.init.constant_(self.xy_lsig_head.bias, -2.0)  # Start with small sigma
        
        # Legacy classification heads - handle both v1 and v2
        if self.head_version == "v1":
            legacy_heads = [self.button_head, self.key_head, self.scroll_y_head]
            if hasattr(self, 'action_type_head'):
                legacy_heads.append(self.action_type_head)
        else:
            legacy_heads = [self.button_head, self.key_action_head, self.key_id_head, self.scroll_y_head]
        
        for head in legacy_heads:
            if hasattr(head, 'weight'):
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
    
    def forward(self, x):
        # Add residual connection for better gradient flow
        s = self.shared(x) + x  # Residual connection
        B = x.size(0)
        if self.head_version == "v1":
            n_type = int(self.enum_sizes.get("action_type", {"size": 5})["size"])
            n_btn  = int(self.enum_sizes.get("button", {"size": 4})["size"])
            n_key  = int(self.enum_sizes.get("key_id", {"size": 151})["size"])
            n_sy   = int(self.enum_sizes.get("scroll_y", {"size": 3})["size"])
            # Event logits
            event_logits = self.event_head(s).view(B, self.max_actions, 4)
            
            # Time quantiles
            tq_pre = self.time_q_head(s).view(B, self.max_actions, len(CFG.time_quantiles))
            time_q = F.softplus(tq_pre + CFG.time_head_bias_default)
            # Use the median quantile for scalar time metrics/plots
            k_med = CFG.time_quantiles.index(0.5)
            time_output = time_q[..., k_med]
            
            # XY heteroscedastic
            xy_mu    = self.xy_mu_head(s).view(B, self.max_actions, 2)
            xy_lsig  = self.xy_lsig_head(s).view(B, self.max_actions, 2)
            x_mu, y_mu = xy_mu[..., 0], xy_mu[..., 1]
            x_ls, y_ls = xy_lsig[..., 0], xy_lsig[..., 1]
            
            out = {
                "time_q_pre": tq_pre,
                "time_q": time_q,
                "x_mu": x_mu,
                "y_mu": y_mu,
                "x_logsig": x_ls,
                "y_logsig": y_ls,
                "action_type_logits": self.action_type_head(s).view(B, self.max_actions, n_type),
                "button_logits": self.button_head(s).view(B, self.max_actions, n_btn),
                "key_logits": self.key_head(s).view(B, self.max_actions, n_key),
                "scroll_y_logits": self.scroll_y_head(s).view(B, self.max_actions, n_sy),
            }
            
            # Add event logits
            out["event_logits"] = event_logits
            
            return out
        else:
            n_btn = int(self.enum_sizes["button"]["size"])
            n_ka  = int(self.enum_sizes["key_action"]["size"])
            n_kid = int(self.enum_sizes["key_id"]["size"])
            n_sy  = int(self.enum_sizes["scroll_y"]["size"])
            # Event logits
            event_logits = self.event_head(s).view(B, self.max_actions, 4)
            
            # Time quantiles
            tq_pre = self.time_q_head(s).view(B, self.max_actions, len(CFG.time_quantiles))
            time_q = F.softplus(tq_pre + CFG.time_head_bias_default)
            # Use the median quantile for scalar time metrics/plots
            k_med = CFG.time_quantiles.index(0.5)
            time_output = time_q[..., k_med]
            
            # XY heteroscedastic
            xy_mu    = self.xy_mu_head(s).view(B, self.max_actions, 2)
            xy_lsig  = self.xy_lsig_head(s).view(B, self.max_actions, 2)
            x_mu, y_mu = xy_mu[..., 0], xy_mu[..., 1]
            x_ls, y_ls = xy_lsig[..., 0], xy_lsig[..., 1]
            
            out = {
                "time_q_pre": tq_pre,
                "time_q": time_q,
                "x_mu": x_mu,
                "y_mu": y_mu,
                "x_logsig": x_ls,
                "y_logsig": y_ls,
                "button_logits": self.button_head(s).view(B, self.max_actions, n_btn),
                "key_action_logits": self.key_action_head(s).view(B, self.max_actions, n_ka),
                "key_id_logits": self.key_id_head(s).view(B, self.max_actions, n_kid),
                "scroll_y_logits": self.scroll_y_head(s).view(B, self.max_actions, n_sy),
            }
            
            # Add event logits
            out["event_logits"] = event_logits
            
            return out

    @torch.no_grad()
    def _invert_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map model time back to seconds (float)."""
        if self.use_log1p_time:
            return torch.expm1(t) * (self.time_div_ms / 1000.0)
        return t

    @torch.no_grad()
    def decode_v2_as_legacy8(self, heads) -> torch.Tensor:
        """
        Inference-only: discretize heads to the legacy (B,100,8) tensor.
        Synthesizes type when head_version=='v2'.
        """
        B, A = heads["time"].shape
        x = heads["x"].round()
        y = heads["y"].round()
        time = self._invert_time(heads["time"])
        if self.head_version == "v2":
            btn = heads["button_logits"].argmax(-1)            # 0=None,1=L,2=R,3=M
            ka  = heads["key_action_logits"].argmax(-1)        # 0=None,1=Press,2=Release
            kid = heads["key_id_logits"].argmax(-1)            # 0=None,1..K
            sy  = heads["scroll_y_logits"].argmax(-1) - self.sy_none_index      # {0,1,2}->{-1,0,1}
            # synthesize legacy type: 1=click, 2=press, 3=release, 4=scroll, else 0
            type_ = torch.zeros_like(btn)
            type_ = torch.where(btn > 0, torch.ones_like(type_), type_)
            type_ = torch.where((type_==0) & (ka==1), torch.full_like(type_, 2), type_)
            type_ = torch.where((type_==0) & (ka==2), torch.full_like(type_, 3), type_)
            type_ = torch.where((type_==0) & (sy!=0), torch.full_like(type_, 4), type_)
            sx = torch.zeros_like(sy)  # legacy scroll_dx = 0
            return torch.stack([time, type_.float(), x, y,
                                btn.float(), kid.float(), sx.float(), sy.float()], dim=-1)
        else:
            at = heads["action_type_logits"].argmax(-1).float()
            btn = heads["button_logits"].argmax(-1).float()
            key = heads["key_logits"].argmax(-1).float()
            sx = heads["scroll_x_logits"].argmax(-1) - 1
            sy = heads["scroll_y_logits"].argmax(-1) - 1
            return torch.stack([time, at, x, y, btn, key, sx.float(), sy.float()], dim=-1)

class ImitationHybridModel(nn.Module):
    """Complete hybrid model combining Transformer + CNN + LSTM with action sequence input"""
    
    def __init__(self, gamestate_dim: int, action_dim: int, sequence_length: int,
                 hidden_dim: int = 256, num_heads: int = 8, max_actions: int = 100,
                 head_version: str = "v1",
                 enum_sizes: dict | None = None,
                 use_log1p_time: bool = True,
                 time_div_ms: float = 1000.0,
                 feature_spec: dict | None = None,
                 **kwargs):
        # --- Backward-compat for older callers --------------------------------
        # Some older code passed `num_attention_heads`; if present, prefer it.
        if "num_attention_heads" in kwargs and kwargs["num_attention_heads"] is not None:
            num_heads = int(kwargs.pop("num_attention_heads"))
        super().__init__()
        
        self.gamestate_dim = gamestate_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_actions = max_actions
        self.head_version = head_version
        self.enum_sizes = enum_sizes or {"button": {"size": 4}, "key_action": {"size": 3}, "key_id": {"size": 151}, "scroll_y": {"size": 3}}
        self.use_log1p_time = use_log1p_time
        self.time_div_ms = float(time_div_ms)
        
        # 1. Gamestate Feature Encoder (128 -> 256)
        self.gamestate_encoder = GamestateEncoder(
            input_dim=gamestate_dim,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            feature_spec=feature_spec
        )
        
        # 2. Action Sequence Encoder - Feature-type-specific encoding (8 -> 128)
        # Input: (batch_size, 10, 100, 8) -> Output: (batch_size, 10, 100, hidden_dim // 2)
        # Features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
        
        # Timestamp encoder (continuous) - raw output, no activation
        self.timestamp_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Action type encoder (categorical: 0,1,2,3,4) - 5 classes: move, click, key_press, key_release, scroll
        self.action_type_embedding = nn.Embedding(5, hidden_dim // 16)  # 5 categories -> 16 dims
        self.action_type_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Coordinate encoders (continuous) - sigmoid + rounding for discrete integers
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # x or y -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Button encoder (categorical: 4)
        self.button_embedding = nn.Embedding(4, hidden_dim // 16)  # 4 categories -> 16 dims
        self.button_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Key encoder (categorical: 151 classes including "no key")
        self.key_embedding = nn.Embedding(151, hidden_dim // 16)  # 151 categories -> 16 dims
        self.key_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Scroll encoder (categorical: -1, 0, 1) - tanh + sign for discrete values
        self.scroll_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # dx or dy -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combine all encoded features
        self.feature_combiner = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),  # 8 * 16 = 128 -> 128
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature preprocessing for proper scaling
        self.feature_preprocessor = nn.Sequential(
            nn.LayerNorm(8),  # Normalize across the 8 features
            nn.Dropout(0.05)  # Light dropout for regularization
        )
        
        # 2b. Action Sequence LSTM - Process actions within each timestep
        # Input: (batch_size, 101, hidden_dim//2) -> Output: (batch_size, 101, hidden_dim)
        self.action_sequence_encoder = nn.LSTM(
            input_size=hidden_dim // 2,  # Encoded action features
            hidden_size=hidden_dim // 2,  # Output per action
            num_layers=2,                 # Deep enough to capture complex patterns
            bidirectional=True,           # Consider both forward/backward action context
            dropout=0.1,
            batch_first=True
        )
        
        # 3. Temporal Context Encoder (LSTM) - Updated to handle 128 features
        self.temporal_encoder = TemporalEncoder(
            input_dim=gamestate_dim,  # Now 128
            hidden_dim=hidden_dim,    # LSTM output will be doubled due to bidirectional, then projected to hidden_dim
            num_layers=2
        )
        
        # 4. Multi-Modal Fusion (Gamestate + Action + Temporal)
        # Input: gamestate(256) + action_sequence(10, 256) + temporal(256)
        # We need to process the action sequence before fusion
        
        # Action sequence processor - convert (10, 256) to (256)
        self.action_sequence_processor = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim),  # Flatten 10 timesteps and process
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer: gamestate(256) + processed_actions(256) + temporal(256) = 768
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim * 2),  # gamestate + processed_actions + temporal
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 5. Action Decoder (Multi-Head)
        self.action_decoder = ActionDecoder(
            input_dim=hidden_dim,
            max_actions=max_actions,
            head_version=head_version,
            enum_sizes=self.enum_sizes,
            use_log1p_time=self.use_log1p_time,
            time_div_ms=self.time_div_ms,
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor,
                return_logits: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Returns a dict of heads when return_logits=True (for training).
        Returns decoded legacy (B,100,8) tensor when return_logits=False (for inference/back-compat).
        """
        batch_size = temporal_sequence.size(0)
        
        # 1. Encode current gamestate features (last timestep of temporal sequence)
        current_gamestate = temporal_sequence[:, -1, :]  # (batch_size, 128)
        gamestate_encoded = self.gamestate_encoder(current_gamestate.unsqueeze(1))  # Add sequence dimension
        gamestate_encoded = gamestate_encoded.squeeze(1)  # Remove sequence dimension
        
        # 2. Encode action sequence (10 timesteps, 100 actions, 8 features -> 10 timesteps, 100 actions, hidden_dim//2)
        # action_sequence shape: (batch_size, 10, 100, 8)
        batch_size, seq_len, num_actions, action_features = action_sequence.shape
        
        # Reshape to process all actions: (batch_size * 10 * 100, 8)
        action_sequence_flat = action_sequence.view(-1, action_features)
        
        # IMPORTANT: do NOT normalize before extracting categorical IDs
        
        # Feature-type-specific encoding: (batch_size * 10 * 100, 8) -> (batch_size * 10 * 100, hidden_dim//2)
        # Features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
        
        # Extract individual features
        timestamp_features = action_sequence_flat[:, 0:1]      # (batch*10*100, 1)
        action_type_features = action_sequence_flat[:, 1:2]    # (batch*10*100, 1)
        x_coord_features = action_sequence_flat[:, 2:3]       # (batch*10*100, 1)
        y_coord_features = action_sequence_flat[:, 3:4]       # (batch*10*100, 1)
        button_features = action_sequence_flat[:, 4:5]         # (batch*10*100, 1)
        key_features = action_sequence_flat[:, 5:6]            # (batch*10*100, 1)
        scroll_dx_features = action_sequence_flat[:, 6:7]      # (batch*10*100, 1)
        scroll_dy_features = action_sequence_flat[:, 7:8]      # (batch*10*100, 1)
        
        # Encode each feature type
        timestamp_encoded = self.timestamp_encoder(timestamp_features)           # (batch*10*100, 16)
        
        # Categorical features: use embeddings
        action_type_features_int = action_type_features.squeeze(-1).long().clamp(0, 4)  # 5 categories
        action_type_embedded = self.action_type_embedding(action_type_features_int)  # (batch*10*100, 16)
        action_type_encoded = self.action_type_encoder(action_type_embedded)  # (batch*10*100, 16)
        
        # Encode coordinates separately to maintain consistent dimensions
        x_coord_encoded = self.coordinate_encoder(x_coord_features)  # (batch*10*100, 16)
        y_coord_encoded = self.coordinate_encoder(y_coord_features)  # (batch*10*100, 16)
        
        button_features_int = button_features.squeeze(-1).long().clamp(0, 3)  # 4 categories
        button_embedded = self.button_embedding(button_features_int)  # (batch*10*100, 16)
        button_encoded = self.button_encoder(button_embedded)  # (batch*10*100, 16)
        
        key_features_int = key_features.squeeze(-1).long().clamp(0, 150)  # 151 categories
        key_embedded = self.key_embedding(key_features_int)  # (batch*10*100, 16)
        key_encoded = self.key_encoder(key_embedded)  # (batch*10*100, 16)
        
        # Encode scroll features separately to maintain consistent dimensions
        scroll_dx_encoded = self.scroll_encoder(scroll_dx_features)  # (batch*10*100, 16)
        scroll_dy_encoded = self.scroll_encoder(scroll_dy_features)  # (batch*10*100, 16)
        
        # Combine all encoded features
        combined_features = torch.cat([
            timestamp_encoded, action_type_encoded, x_coord_encoded, y_coord_encoded,
            button_encoded, key_encoded, scroll_dx_encoded, scroll_dy_encoded
        ], dim=1)  # (batch*10*100, 128) - 8 features * 16 dims each
        
        # Final feature combination
        action_encoded_flat = self.feature_combiner(combined_features)  # (batch*10*100, hidden_dim//2)
        
        # Reshape back: (batch_size, 10, 100, hidden_dim//2)
        action_encoded = action_encoded_flat.view(batch_size, seq_len, num_actions, -1)
        
        # NEW: Process actions within each timestep using LSTM to preserve action detail
        # Process each timestep's actions through the action sequence LSTM
        action_encoded_timesteps = []
        for i in range(seq_len):
            # Extract actions for this timestep: (batch_size, 100, hidden_dim//2)
            timestep_actions = action_encoded[:, i, :, :]
            
            # Process through LSTM: (batch_size, 100, hidden_dim) - bidirectional doubles the size
            timestep_actions_encoded, _ = self.action_sequence_encoder(timestep_actions)
            
            # Take the last action's output as representation for this timestep: (batch_size, hidden_dim)
            timestep_representation = timestep_actions_encoded[:, -1, :]
            action_encoded_timesteps.append(timestep_representation)
        
        # Stack timestep representations: (batch_size, 10, hidden_dim)
        action_encoded = torch.stack(action_encoded_timesteps, dim=1)
        
        # Keep all 10 timesteps of action context instead of just the last one
        # action_encoded shape: (batch_size, 10, hidden_dim)
        
        # 3. Encode temporal sequence
        temporal_encoded = self.temporal_encoder(temporal_sequence)  # (batch_size, hidden_dim)
        
        # 4. Process action sequence and Multi-Modal Fusion (Gamestate + Action + Temporal)
        
        # Process action sequence: (batch_size, 10, 256) -> (batch_size, 256)
        batch_size, seq_len, action_features = action_encoded.shape
        action_encoded_flat = action_encoded.view(batch_size, -1)  # Flatten to (batch_size, 10*256)
        processed_actions = self.action_sequence_processor(action_encoded_flat)  # (batch_size, 256)
        
        # Fuse all features
        fused_features = torch.cat([
            gamestate_encoded,        # (batch_size, hidden_dim)
            processed_actions,        # (batch_size, hidden_dim) - processed from 10 timesteps
            temporal_encoded          # (batch_size, hidden_dim)
        ], dim=-1)  # (batch_size, hidden_dim + hidden_dim + hidden_dim = 768)
        
        fused_output = self.fusion_layer(fused_features)
        
        # 5. Decode actions
        heads = self.action_decoder(fused_output)  # dict of heads for training
        
        # --- DEBUG: confirm the heads and time activation
        if return_logits:
            # event head
            if "event_logits" in heads:
                if heads["event_logits"].detach().abs().mean() < 1e-6:
                    LOG.warning("[DBG] event_logits ~ constant (%.2e)", float(heads["event_logits"].detach().abs().mean()))
            # scroll_y head
            if "scroll_y_logits" in heads:
                if heads["scroll_y_logits"].detach().abs().mean() < 1e-6:
                    LOG.warning("[DBG] scroll_y_logits ~ constant (%.2e)", float(heads["scroll_y_logits"].detach().abs().mean()))
            # time head
            if "time" in heads:
                t = heads["time"]
                t_act = F.softplus(t) if getattr(self, "time_softplus", True) else t
                time_ms_pred = t_act * getattr(self, "time_scale", 1.0)
                if time_ms_pred.detach().abs().mean() < 1e-6:
                    LOG.warning("[DBG] time_ms_pred ~ 0; check mask/scale/clip path")
            # xy head
            if "x" in heads and "y" in heads:
                xy = torch.stack([heads["x"], heads["y"]], dim=-1)
                xy_scaled = xy * getattr(self, "xy_scale", 1.0)
                # Quick range check
                LOG.info("[DBG] fwd ranges: event|scroll logits mean=%.3f|%.3f, time_ms mean=%.3f, xy mean=%.3f",
                         float(heads.get("event_logits", torch.zeros(1)).detach().abs().mean()),
                         float(heads.get("scroll_y_logits", torch.zeros(1)).detach().abs().mean()),
                         float(time_ms_pred.detach().mean()) if 'time_ms_pred' in locals() else 0.0,
                         float(xy_scaled.detach().abs().mean()))
            
            # quick ranges (cheap)
            with torch.no_grad():
                # Get xy_abs safely - only if xy heads exist
                xy_abs_mean = 0.0
                if "x" in heads and "y" in heads:
                    xy = torch.stack([heads["x"], heads["y"]], dim=-1)
                    xy_scaled = xy * getattr(self, "xy_scale", 1.0)
                    xy_abs_mean = float(xy_scaled.detach().abs().mean())
                
                LOG.info("[DBG] fwd means: event|scroll=%.3f|%.3f  time_ms=%.3f  xy_abs=%.3f",
                         float(heads.get("event_logits", torch.zeros(1)).abs().mean()),
                         float(heads.get("scroll_y_logits", torch.zeros(1)).abs().mean()),
                         float(time_ms_pred.detach().mean()) if 'time_ms_pred' in locals() else 0.0,
                         xy_abs_mean)
            
            return heads
        return self.decode_legacy(heads)
    
    @torch.no_grad()
    def decode_legacy(self, heads: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference-only: discretize heads to the legacy (B,100,8) tensor.
        Synthesizes type when head_version=='v2'.
        """
        B, A = heads["time"].shape
        x = heads["x"].round()
        y = heads["y"].round()
        time = self._invert_time(heads["time"])
        if self.head_version == "v2":
            btn = heads["button_logits"].argmax(-1)            # 0=None,1=L,2=R,3=M
            ka  = heads["key_action_logits"].argmax(-1)        # 0=None,1=Press,2=Release
            kid = heads["key_id_logits"].argmax(-1)            # 0=None,1=None,1..K
            sy  = heads["scroll_y_logits"].argmax(-1) - self.action_decoder.sy_none_index      # {0,1,2}->{-1,0,1}
            # synthesize legacy type: 1=click, 2=press, 3=release, 4=scroll, else 0
            type_ = torch.zeros_like(btn)
            type_ = torch.where(btn > 0, torch.ones_like(type_), type_)
            type_ = torch.where((type_==0) & (ka==1), torch.full_like(type_, 2), type_)
            type_ = torch.where((type_==0) & (ka==2), torch.full_like(type_, 3), type_)
            type_ = torch.where((type_==0) & (sy!=0), torch.full_like(type_, 4), type_)
            sx = torch.zeros_like(sy)  # legacy scroll_dx = 0
            return torch.stack([time, type_.float(), x, y,
                                btn.float(), kid.float(), sx.float(), sy.float()], dim=-1)
        else:
            at = heads["action_type_logits"].argmax(-1).float()
            btn = heads["button_logits"].argmax(-1).float()
            key = heads["key_logits"].argmax(-1).float()
            sx = heads["scroll_x_logits"].argmax(-1) - 1
            sy = heads["scroll_y_logits"].argmax(-1) - 1
            return torch.stack([time, at, x, y, btn, key, sx.float(), sy.float()], dim=-1)
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model information and parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'gamestate_dim': self.gamestate_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim
        }

def create_model(config: Dict = None) -> ImitationHybridModel:
    """Factory function to create the model with default or custom config"""
    if config is None:
        config = {
            'gamestate_dim': 128,  # Updated to match your data pipeline
            'action_dim': 8,       # Action features per timestep
            'sequence_length': 10,
            'hidden_dim': 256,
            'num_heads': 8
        }
    
    # Provide default feature_spec if not present
    if 'feature_spec' not in config:
        config['feature_spec'] = {
            'group_indices': {
                'categorical': [],
                'continuous': list(range(128)),  # All features as continuous
                'boolean': [],
                'counts': [],
                'angles': [],
                'time': []
            },
            'total_cat_vocab': 0,
            'cat_offsets': [],
            'unknown_index_per_field': []
        }
    
    model = ImitationHybridModel(**config)
    return model

if __name__ == "__main__":
    # Test the model
    print("Testing ImitationHybridModel...")
    
    # Create model
    model = create_model()
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model created successfully!")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Test forward pass
    batch_size = 4
    temporal_sequence = torch.randn(batch_size, 10, 128)
    action_sequence = torch.randn(batch_size, 10, 100, 8)
    
    with torch.no_grad():
        # Test inference mode (default)
        output = model(temporal_sequence, action_sequence)
        print(f"\nInference mode forward pass successful!")
        print(f"Input shapes: temporal={temporal_sequence.shape}, action={action_sequence.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test training mode (logits)
        logits_output = model(temporal_sequence, action_sequence, return_logits=True)
        print(f"\nTraining mode forward pass successful!")
        print(f"Logits output keys: {list(logits_output.keys())}")
        for key, value in logits_output.items():
            print(f"  {key}: {value.shape}")
