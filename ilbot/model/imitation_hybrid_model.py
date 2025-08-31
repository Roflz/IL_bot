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
        self.idx_cat   = list(groups.get("categorical", []))
        self.idx_cont  = list(groups.get("continuous", []))
        self.idx_bool  = list(groups.get("boolean", []))
        self.idx_count = list(groups.get("counts", []))
        self.idx_angle = list(groups.get("angles", []))
        self.idx_time  = list(groups.get("time", []))

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
        
        # Fallback MLP for when no feature_spec is provided (treats all features as continuous)
        self._fallback_mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),  # 128 is the default gamestate_dim, output full hidden_dim
            nn.LayerNorm(hidden_dim),
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

        # If no chunks were added (no feature_spec provided), treat all features as continuous
        if len(chunks) == 0:
            # Treat all features as continuous using fallback MLP
            cont = x.float()
            cont = self._fallback_mlp(cont)
            chunks.append(cont)

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
    """Unified event system decoder with exclusive event classification."""
    
    def __init__(self, input_dim, *, max_actions: int, enum_sizes: dict):
        super().__init__()
        self.max_actions = max_actions
        self.enum_sizes = enum_sizes
        
        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        
        # Unified event classification head: [CLICK, KEY, SCROLL, MOVE]
        self.event_head = nn.Linear(input_dim, 4)
        
        # Time quantile head: [q10, q50, q90] for robust time prediction
        self.time_quantile_head = nn.Linear(input_dim, 3)
        
        # Heteroscedastic XY heads: mean + uncertainty for cursor positions
        self.x_mu_head = nn.Linear(input_dim, 1)      # X coordinate mean
        self.x_logsig_head = nn.Linear(input_dim, 1)  # X coordinate log(std)
        self.y_mu_head = nn.Linear(input_dim, 1)      # Y coordinate mean
        self.y_logsig_head = nn.Linear(input_dim, 1)  # Y coordinate log(std)
        
        # Event-specific detail heads (only used when that event wins)
        n_btn = int(enum_sizes["button"]["size"])
        n_ka = int(enum_sizes["key_action"]["size"])
        n_kid = int(enum_sizes["key_id"]["size"])
        n_sy = int(enum_sizes["scroll_y"]["size"])
        
        self.button_head = nn.Linear(input_dim, n_btn)
        self.key_action_head = nn.Linear(input_dim, n_ka)
        self.key_id_head = nn.Linear(input_dim, n_kid)
        self.scroll_y_head = nn.Linear(input_dim, n_sy)
    
    def forward(self, x):
        """
        Forward pass producing unified event system outputs.
        
        Returns:
            dict with keys:
            - event_logits: [B, A, 4] - logits for [CLICK, KEY, SCROLL, MOVE]
            - time_q: [B, A, 3] - time quantiles [q10, q50, q90]
            - x_mu, x_logsig: [B, A] - X coordinate mean + log(std)
            - y_mu, y_logsig: [B, A] - Y coordinate mean + log(std)
            - button_logits: [B, A, 4] - button classification (conditional)
            - key_action_logits: [B, A, 3] - key action classification (conditional)
            - key_id_logits: [B, A, vocab_size] - key ID classification (conditional)
            - scroll_y_logits: [B, A, 3] - scroll direction (conditional)
        """
        s = self.shared(x)
        B, A = x.size(0), x.size(1)
        
        # Process each action position individually
        event_logits = []
        time_q_raw = []
        x_mu_list = []
        x_logsig_list = []
        y_mu_list = []
        y_logsig_list = []
        button_logits = []
        key_action_logits = []
        key_id_logits = []
        scroll_y_logits = []
        
        for i in range(A):
            # Get features for this action position
            action_features = s[:, i, :]  # [B, F]
            
            # Process through each head
            event_logits.append(self.event_head(action_features))           # [B, 4]
            time_q_raw.append(self.time_quantile_head(action_features))     # [B, 3]
            x_mu_list.append(self.x_mu_head(action_features))              # [B, 1]
            x_logsig_list.append(self.x_logsig_head(action_features))      # [B, 1]
            y_mu_list.append(self.y_mu_head(action_features))              # [B, 1]
            y_logsig_list.append(self.y_logsig_head(action_features))      # [B, 1]
            button_logits.append(self.button_head(action_features))         # [B, n_btn]
            key_action_logits.append(self.key_action_head(action_features)) # [B, n_ka]
            key_id_logits.append(self.key_id_head(action_features))         # [B, n_kid]
            scroll_y_logits.append(self.scroll_y_head(action_features))     # [B, n_sy]
        
        # Stack all outputs along action dimension
        event_logits = torch.stack(event_logits, dim=1)           # [B, A, 4]
        time_q_raw = torch.stack(time_q_raw, dim=1)              # [B, A, 3]
        x_mu = torch.stack(x_mu_list, dim=1).squeeze(-1)        # [B, A]
        x_logsig = torch.stack(x_logsig_list, dim=1).squeeze(-1) # [B, A]
        y_mu = torch.stack(y_mu_list, dim=1).squeeze(-1)        # [B, A]
        y_logsig = torch.stack(y_logsig_list, dim=1).squeeze(-1) # [B, A]
        button_logits = torch.stack(button_logits, dim=1)         # [B, A, n_btn]
        key_action_logits = torch.stack(key_action_logits, dim=1) # [B, A, n_ka]
        key_id_logits = torch.stack(key_id_logits, dim=1)         # [B, A, n_kid]
        scroll_y_logits = torch.stack(scroll_y_logits, dim=1)     # [B, A, n_sy]
        
        # Apply time positivity constraint
        time_q = F.softplus(time_q_raw) + 0.1  # Small bias to avoid 0
        
        return {
            "event_logits": event_logits,           # [B, A, 4]
            "time_q": time_q,                       # [B, A, 3]
            "x_mu": x_mu,                          # [B, A]
            "x_logsig": x_logsig,                  # [B, A]
            "y_mu": y_mu,                          # [B, A]
            "y_logsig": y_logsig,                  # [B, A]
            "button_logits": button_logits,        # [B, A, n_btn]
            "key_action_logits": key_action_logits, # [B, A, n_ka]
            "key_id_logits": key_id_logits,        # [B, A, n_kid]
            "scroll_y_logits": scroll_y_logits,    # [B, A, n_sy]
        }

    @torch.no_grad()
    def _invert_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map model time back to seconds (float)."""
        if self.use_log1p_time:
            return torch.expm1(t) * (self.time_div_ms / 1000.0)
        return t


class ImitationHybridModel(nn.Module):
    """Complete hybrid model combining Transformer + CNN + LSTM with action sequence input"""
    
    def __init__(self, gamestate_dim: int, action_dim: int, sequence_length: int,
                 hidden_dim: int = 256, num_heads: int = 8, max_actions: int = 100,
                 enum_sizes: dict | None = None,
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
        self.enum_sizes = enum_sizes or {"button": {"size": 4}, "key_action": {"size": 3}, "key_id": {"size": 151}, "scroll_y": {"size": 3}}
        
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
        
        # NEW: Key action encoder (categorical: 3 classes - None, Press, Release)
        self.key_action_embedding = nn.Embedding(3, hidden_dim // 16)  # 3 categories -> 16 dims
        self.key_action_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # NEW: Key ID encoder (categorical: 151 classes including "no key")
        self.key_id_embedding = nn.Embedding(151, hidden_dim // 16)  # 151 categories -> 16 dims
        self.key_id_encoder = nn.Sequential(
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
        
        # Combine all encoded features (7 features * (hidden_dim//16) dims each)
        feature_combiner_input_dim = 7 * (hidden_dim // 16)  # 7 * 8 = 56 for hidden_dim=128
        self.feature_combiner = nn.Sequential(
            nn.Linear(feature_combiner_input_dim, hidden_dim // 2),  # 56 -> 64 for hidden_dim=128
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature preprocessing for proper scaling
        self.feature_preprocessor = nn.Sequential(
            nn.LayerNorm(7),  # Normalize across the 7 V2 features
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
        
                # 5. Action Decoder (Unified Event System)
        self.action_decoder = ActionDecoder(
            input_dim=hidden_dim,
            max_actions=max_actions,
            enum_sizes=self.enum_sizes,
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
    
    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for the unified event system model.
        
        Args:
            temporal_sequence: (B, T, D) - Batch of temporal gamestate sequences
            action_sequence: (B, T, A, F) - Batch of action sequences (F=7 for V2 actions)
        
        Returns:
            dict of unified event system outputs
        """
        batch_size = temporal_sequence.size(0)
        
        # 1. Encode current gamestate features (last timestep of temporal sequence)
        current_gamestate = temporal_sequence[:, -1, :]  # (batch_size, 128)
        gamestate_encoded = self.gamestate_encoder(current_gamestate.unsqueeze(1))  # Add sequence dimension
        gamestate_encoded = gamestate_encoded.squeeze(1)  # Remove sequence dimension
        
        # 2. Encode action sequence (10 timesteps, 100 actions, 7 features -> 10 timesteps, 100 actions, hidden_dim//2)
        # action_sequence shape: (batch_size, 10, 100, 7) for V2 actions
        batch_size, seq_len, num_actions, action_features = action_sequence.shape
        
        # Validate input: ensure we have 7 features for V2 actions
        if action_features != 7:
            raise ValueError(f"Expected 7 action features for V2 actions, got {action_features}")
        
        # Reshape to process all actions: (batch_size * 10 * 100, 7)
        action_sequence_flat = action_sequence.view(-1, action_features)
        
        # Feature-type-specific encoding: (batch_size * 10 * 100, 7) -> (batch_size * 10 * 100, hidden_dim//2)
        # V2 Features: [timestamp, x, y, button, key_action, key_id, scroll_y]
        
        # Extract individual features
        timestamp_features = action_sequence_flat[:, 0:1]      # (batch*10*100, 1)
        x_coord_features = action_sequence_flat[:, 1:2]       # (batch*10*100, 1)
        y_coord_features = action_sequence_flat[:, 2:3]       # (batch*10*100, 1)
        button_features = action_sequence_flat[:, 3:4]         # (batch*10*100, 1)
        key_action_features = action_sequence_flat[:, 4:5]     # (batch*10*100, 1)
        key_id_features = action_sequence_flat[:, 5:6]         # (batch*10*100, 1)
        scroll_y_features = action_sequence_flat[:, 6:7]       # (batch*10*100, 1)
        
        # Encode each feature type
        timestamp_encoded = self.timestamp_encoder(timestamp_features)           # (batch*10*100, 16)
        
        # Encode coordinates separately to maintain consistent dimensions
        x_coord_encoded = self.coordinate_encoder(x_coord_features)  # (batch*10*100, 16)
        y_coord_encoded = self.coordinate_encoder(y_coord_features)  # (batch*10*100, 16)
        
        # Categorical features: use embeddings
        button_features_int = button_features.squeeze(-1).long().clamp(0, 3)  # 4 categories
        button_embedded = self.button_embedding(button_features_int)  # (batch*10*100, 16)
        button_encoded = self.button_encoder(button_embedded)  # (batch*10*100, 16)
        
        key_action_features_int = key_action_features.squeeze(-1).long().clamp(0, 2)  # 3 categories
        key_action_embedded = self.key_action_embedding(key_action_features_int)  # (batch*10*100, 16)
        key_action_encoded = self.key_action_encoder(key_action_embedded)  # (batch*10*100, 16)
        
        key_id_features_int = key_id_features.squeeze(-1).long().clamp(0, 150)  # 151 categories
        key_id_embedded = self.key_id_embedding(key_id_features_int)  # (batch*10*100, 16)
        key_id_encoded = self.key_id_encoder(key_id_embedded)  # (batch*10*100, 16)
        
        # Encode scroll_y feature (scroll_dx no longer exists in V2)
        scroll_y_encoded = self.scroll_encoder(scroll_y_features)  # (batch*10*100, 16)
        
        # Combine all encoded features (7 features * 16 dims each = 112)
        combined_features = torch.cat([
            timestamp_encoded, x_coord_encoded, y_coord_encoded,
            button_encoded, key_action_encoded, key_id_encoded, scroll_y_encoded
        ], dim=1)  # (batch*10*100, 112)
        
        # Final feature combination (112 -> hidden_dim//2)
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
        
        # Expand fused output to cover all action positions
        # fused_output: [B, hidden_dim] -> [B, max_actions, hidden_dim]
        fused_output = fused_output.unsqueeze(1).expand(-1, self.max_actions, -1)
        
        # 5. Return unified event system outputs
        return self.action_decoder(fused_output)  # dict of heads
    

    
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
            'action_dim': 7,       # V2 action features per timestep
            'sequence_length': 10,
            'hidden_dim': 256,
            'num_heads': 8
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
    action_sequence = torch.randn(batch_size, 10, 100, 7)
    
    with torch.no_grad():
        # Test forward pass
        output = model(temporal_sequence, action_sequence)
        print(f"\nForward pass successful!")
        print(f"Input shapes: temporal={temporal_sequence.shape}, action={action_sequence.shape}")
        print(f"Output keys: {list(output.keys())}")
        for key, value in output.items():
            print(f"  {key}: {value.shape}")
