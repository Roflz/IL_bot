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

        # Ensure we always have proper feature specification
        if len(chunks) == 0:
            raise ValueError("No feature specification provided. The model requires a proper feature_spec to determine how to process gamestate features.")

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

class ActionSequenceDecoder(nn.Module):
    """
    New decoder that can predict variable-length action sequences.
    
    Key improvements:
    1. Each action slot gets different predictions (no more .expand())
    2. Sequence length prediction (how many actions in this gamestate?)
    3. Attention-based sequence modeling
    4. Proper masking for variable-length sequences
    """
    
    def __init__(self, input_dim, *, max_actions: int, enum_sizes: dict, event_types: int):
        super().__init__()
        self.max_actions = max_actions
        self.enum_sizes = enum_sizes
        self.event_types = event_types
        
        # 1. Sequence length predictor (how many actions in this gamestate?)
        self.sequence_length_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Output: [0, 1] -> multiply by max_actions
        )
        
        # Initialize sequence length head to output reasonable values
        # We want it to start predicting around 1-5 actions per gamestate initially
        # This means sigmoid output should be around 0.01-0.05 (1-5/100)
        with torch.no_grad():
            # Set the final linear layer bias to output ~0.03 (3 actions per gamestate)
            # The last layer is nn.Linear, not nn.Sigmoid
            self.sequence_length_head[-2].bias.data.fill_(-3.5)  # sigmoid(-3.5) ≈ 0.03
        
        # 2. Action position embeddings (learn different representations for each action slot)
        self.action_position_embeddings = nn.Embedding(max_actions, input_dim // 4)
        
        # 3. Cross-attention for each action slot to attend to different parts of context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 4. Action-specific processing for each slot
        self.action_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(max_actions)
        ])
        
        # 5. Output heads (same as before, but now each action slot is processed individually)
        self.event_head = nn.Linear(input_dim, event_types)
        self.time_quantile_head = nn.Linear(input_dim, 3)
        
        # Coordinate heads
        self.x_mu_head = nn.Linear(input_dim, 1)
        self.x_logsig_head = nn.Linear(input_dim, 1)
        self.y_mu_head = nn.Linear(input_dim, 1)
        self.y_logsig_head = nn.Linear(input_dim, 1)
        
        # Event-specific heads
        self.button_head = nn.Linear(input_dim, self.enum_sizes['button'])
        self.key_action_head = nn.Linear(input_dim, self.enum_sizes['key_action'])
        self.key_id_head = nn.Linear(input_dim, self.enum_sizes['key_id'])
        self.scroll_y_head = nn.Linear(input_dim, self.enum_sizes['scroll'])
    
    def forward(self, context: torch.Tensor, action_history: Optional[torch.Tensor] = None, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for action sequence prediction.
        
        Args:
            context: [B, input_dim] - fused context from gamestate + action history + temporal
            action_history: [B, T, A, F] - optional action history for temporal context (T=10, A=100, F=7)
            valid_mask: [B, max_actions] - optional mask indicating which actions are valid vs padded
            
        Returns:
            Dict with action predictions for each slot
        """
        batch_size = context.size(0)
        
        # 1. Predict sequence length (how many actions in this gamestate?)
        sequence_length_logits = self.sequence_length_head(context)  # [B, 1] - sigmoid output [0,1]
        predicted_length = (sequence_length_logits * self.max_actions).round().long()  # [B, 1]
        
        # Debug: Print sequence length predictions during training
        if self.training and torch.rand(1).item() < 0.01:  # 1% of the time during training
            print(f"🔍 Sequence Length Debug:")
            print(f"  - Sigmoid output (0-1): {sequence_length_logits[:3].detach().cpu().numpy()}")
            print(f"  - Predicted length: {predicted_length[:3].detach().cpu().numpy()}")
            print(f"  - Max actions: {self.max_actions}")
        
        # 2. Process action history for temporal context and delta understanding
        temporal_context = context
        if action_history is not None:
            # action_history: [B, T, A, F] where T=10, A=100, F=7
            # Extract the most recent action history (last timestep)
            recent_actions = action_history[:, -1, :, :]  # [B, A, F]
            
            # Extract timing information for delta understanding
            # Column 0 is timestamp (delta time from previous action within this gamestate)
            action_timings = recent_actions[:, :, 0]  # [B, A] - timing deltas
            
            # Create per-gamestate cumulative timing context (0ms to 600ms max)
            # For each action slot, compute cumulative time from start of THIS gamestate
            cumulative_times = torch.cumsum(action_timings, dim=1)  # [B, A]
            
            # Ensure cumulative times don't exceed gamestate duration (600ms = 0.6s)
            # This prevents actions from spanning multiple gamestates
            cumulative_times = torch.clamp(cumulative_times, 0.0, 0.6)
            
            # Create timing context for each action slot
            timing_context = torch.zeros(batch_size, self.max_actions, 2).to(context.device)  # [B, A, 2]
            timing_context[:, :, 0] = action_timings  # Delta time (relative to previous action)
            timing_context[:, :, 1] = cumulative_times  # Cumulative time (0ms to 600ms within gamestate)
            
            # Process action history through a simple MLP to get temporal context
            # Flatten action history: [B, A, F] -> [B, A*F]
            action_history_flat = recent_actions.view(batch_size, -1)  # [B, A*F]
            
            # Project to context dimension
            action_history_proj = nn.Linear(action_history_flat.size(-1), context.size(-1) // 4).to(context.device)
            action_history_encoded = action_history_proj(action_history_flat)  # [B, input_dim//4]
            
            # Process timing context
            timing_flat = timing_context.view(batch_size, -1)  # [B, A*2]
            timing_proj = nn.Linear(timing_flat.size(-1), context.size(-1) // 8).to(context.device)
            timing_encoded = timing_proj(timing_flat)  # [B, input_dim//8]
            
            # Combine with main context
            temporal_context = torch.cat([context, action_history_encoded, timing_encoded], dim=-1)  # [B, input_dim + input_dim//4 + input_dim//8]
            
            # Project back to original context dimension
            context_proj = nn.Linear(temporal_context.size(-1), context.size(-1)).to(context.device)
            temporal_context = context_proj(temporal_context)  # [B, input_dim]
        
        # 3. Create action position embeddings for each slot with boundary awareness
        action_positions = torch.arange(self.max_actions, device=context.device)  # [max_actions]
        position_embeddings = self.action_position_embeddings(action_positions)  # [max_actions, input_dim//4]
        
        # Add boundary awareness: mark the last action slot as a boundary
        boundary_info = torch.zeros(batch_size, self.max_actions, 1).to(context.device)  # [B, A, 1]
        if valid_mask is not None:
            # Mark the last valid action in each sequence as a boundary
            for b in range(batch_size):
                valid_indices = torch.where(valid_mask[b])[0]
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1].item()
                    boundary_info[b, last_valid_idx, 0] = 1.0  # Mark as boundary
        else:
            # If no valid mask, mark the last slot as boundary
            boundary_info[:, -1, 0] = 1.0
        
        # 4. Expand temporal context and combine with position embeddings and boundary info
        context_expanded = temporal_context.unsqueeze(1).expand(-1, self.max_actions, -1)  # [B, max_actions, input_dim]
        position_emb_expanded = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, max_actions, input_dim//4]
        
        # Pad position embeddings to match context dimension
        position_emb_padded = F.pad(position_emb_expanded, (0, temporal_context.size(-1) - position_emb_expanded.size(-1)))
        
        # Add boundary information to context
        boundary_expanded = boundary_info.expand(-1, -1, temporal_context.size(-1))  # [B, max_actions, input_dim]
        
        # 5. Combine context with position information and boundary awareness
        combined_context = context_expanded + position_emb_padded + boundary_expanded  # [B, max_actions, input_dim]
        
        # 6. Cross-attention: let each action slot attend to different parts of the context
        # Query: each action slot, Key/Value: the combined context
        
        # Create attention mask for padding awareness
        attention_mask = None
        if valid_mask is not None:
            # Create attention mask: [B, max_actions, max_actions]
            # Mask out padded slots so they don't attend to anything
            attention_mask = valid_mask.unsqueeze(1).expand(-1, self.max_actions, -1)  # [B, max_actions, max_actions]
            # Also mask out padded slots from being attended to
            attention_mask = attention_mask & valid_mask.unsqueeze(2).expand(-1, -1, self.max_actions)
            
            # Convert to the format expected by MultiheadAttention
            # MultiheadAttention expects [B*num_heads, seq_len, seq_len] for 3D mask
            num_heads = self.cross_attention.num_heads
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [B, num_heads, max_actions, max_actions]
            attention_mask = attention_mask.reshape(-1, self.max_actions, self.max_actions)  # [B*num_heads, max_actions, max_actions]
        
        attended_context, _ = self.cross_attention(
            query=combined_context,  # [B, max_actions, input_dim]
            key=combined_context,    # [B, max_actions, input_dim]
            value=combined_context,  # [B, max_actions, input_dim]
            attn_mask=attention_mask  # [B, max_actions, max_actions] - NEW: padding awareness
        )
        
        # 7. Process each action slot individually
        action_features = []
        for i in range(self.max_actions):
            slot_context = attended_context[:, i, :]  # [B, input_dim]
            processed = self.action_processors[i](slot_context)  # [B, input_dim]
            action_features.append(processed)
        
        # Stack all action features: [B, max_actions, input_dim]
        action_features = torch.stack(action_features, dim=1)
        
        # 8. Generate predictions for each action slot
        outputs = {}
        
        # Event classification
        outputs['event_logits'] = self.event_head(action_features)  # [B, max_actions, event_types]
        
        # Time quantiles
        time_q_raw = self.time_quantile_head(action_features)  # [B, max_actions, 3]
        outputs['time_q'] = F.softplus(time_q_raw) + 0.001  # Ensure positivity
        
        # Coordinates
        x_mu = self.x_mu_head(action_features).squeeze(-1)  # [B, max_actions]
        x_logsig = self.x_logsig_head(action_features).squeeze(-1)  # [B, max_actions]
        y_mu = self.y_mu_head(action_features).squeeze(-1)  # [B, max_actions]
        y_logsig = self.y_logsig_head(action_features).squeeze(-1)  # [B, max_actions]
        
        # Apply sigmoid to coordinate predictions to ensure [0, 1] range
        outputs['x_mu'] = torch.sigmoid(x_mu.clone())  # [B, max_actions] - normalized [0, 1]
        outputs['x_logsig'] = x_logsig  # [B, max_actions]
        outputs['y_mu'] = torch.sigmoid(y_mu.clone())  # [B, max_actions] - normalized [0, 1]
        outputs['y_logsig'] = y_logsig  # [B, max_actions]
        
        # Event-specific details
        outputs['button_logits'] = self.button_head(action_features)  # [B, max_actions, button_types]
        outputs['key_action_logits'] = self.key_action_head(action_features)  # [B, max_actions, key_action_types]
        outputs['key_id_logits'] = self.key_id_head(action_features)  # [B, max_actions, key_id_types]
        outputs['scroll_y_logits'] = self.scroll_y_head(action_features)  # [B, max_actions, scroll_types]
        
        # Add sequence length prediction
        outputs['sequence_length'] = predicted_length.squeeze(-1)  # [B]
        
        # 9. Apply padding mask to outputs (ensure padded slots produce neutral predictions)
        if valid_mask is not None:
            # Create mask for invalid slots
            invalid_mask = ~valid_mask  # [B, max_actions]
            
            # Mask out invalid slots in all outputs
            for key, value in outputs.items():
                if key == 'sequence_length':
                    continue  # Don't mask sequence length prediction
                
                if value.dim() == 2:  # [B, max_actions]
                    # For 2D outputs, create new tensor with masked values
                    if key in ['x_mu', 'y_mu']:
                        masked_value = value.clone()
                        masked_value[invalid_mask] = 0.5  # Neutral position (center of screen)
                        outputs[key] = masked_value
                    elif key in ['x_logsig', 'y_logsig']:
                        masked_value = value.clone()
                        masked_value[invalid_mask] = -2.0  # High uncertainty
                        outputs[key] = masked_value
                    else:
                        masked_value = value.clone()
                        masked_value[invalid_mask] = 0.0  # Neutral value
                        outputs[key] = masked_value
                elif value.dim() == 3:  # [B, max_actions, features]
                    # For 3D outputs, handle differently based on content
                    if key == 'time_q':
                        # Don't mask time predictions - let the model learn to predict 0 for invalid slots
                        # This allows the model to learn which slots should have actions
                        pass
                    else:
                        # For logits, set invalid slots to very negative values
                        masked_value = value.clone()
                        masked_value[invalid_mask] = -1e9  # Very negative logits (no prediction)
                        outputs[key] = masked_value
        
        return outputs

    @torch.no_grad()
    def _invert_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map model time back to seconds (float)."""
        if self.use_log1p_time:
            return torch.expm1(t) * (self.time_div_ms / 1000.0)
        return t


class ImitationHybridModel(nn.Module):
    """Complete hybrid model combining Transformer + CNN + LSTM with action sequence input"""
    
    def __init__(self, data_config: dict, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 6,
                 feature_spec: dict | None = None, **kwargs):
        """
        Initialize the ImitationHybridModel with data-driven configuration.
        
        Args:
            data_config: Configuration dict from DataInspector.auto_detect()
            hidden_dim: Hidden dimension for the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            feature_spec: Feature specification for gamestate encoding
        """
        super().__init__()
        
        # Extract configuration from data
        self.gamestate_dim = data_config['gamestate_dim']
        self.max_actions = data_config['max_actions']
        self.action_features = data_config['action_features']
        self.temporal_window = data_config['temporal_window']
        self.sequence_length = data_config['temporal_window']  # Alias for compatibility
        self.enum_sizes = data_config['enum_sizes']
        self.event_types = data_config['event_types']
        
        # Store model parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 1. Gamestate Feature Encoder (128 -> 256)
        self.gamestate_encoder = GamestateEncoder(
            input_dim=self.gamestate_dim,
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
        
        # Button encoder (categorical: dynamic size from data)
        self.button_embedding = nn.Embedding(self.enum_sizes['button'], hidden_dim // 16)
        self.button_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Key action encoder (categorical: dynamic size from data)
        self.key_action_embedding = nn.Embedding(self.enum_sizes['key_action'], hidden_dim // 16)
        self.key_action_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Key ID encoder (categorical: dynamic size from data)
        self.key_id_embedding = nn.Embedding(self.enum_sizes['key_id'], hidden_dim // 16)
        self.key_id_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Scroll encoder (categorical: dynamic size from data)
        self.scroll_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),
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
            input_dim=self.gamestate_dim,  # Now 128
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
        
                # 5. Action Sequence Decoder (New architecture for variable-length sequences)
        self.action_decoder = ActionSequenceDecoder(
            input_dim=hidden_dim,
            max_actions=self.max_actions,
            enum_sizes=self.enum_sizes,
            event_types=self.event_types,
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
    
    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Forward pass for the unified event system model.
        
        Args:
            temporal_sequence: (B, T, D) - Batch of temporal gamestate sequences
            action_sequence: (B, T, A, F) - Batch of action sequences (F=7 for V2 actions)
            valid_mask: (B, A) - Optional mask indicating which actions are valid vs padded
        
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
        button_features_int = button_features.squeeze(-1).long().clamp(0, self.enum_sizes['button'] - 1)
        button_embedded = self.button_embedding(button_features_int)  # (batch*10*100, 16)
        button_encoded = self.button_encoder(button_embedded)  # (batch*10*100, 16)
        
        key_action_features_int = key_action_features.squeeze(-1).long().clamp(0, self.enum_sizes['key_action'] - 1)
        key_action_embedded = self.key_action_embedding(key_action_features_int)  # (batch*10*100, 16)
        key_action_encoded = self.key_action_encoder(key_action_embedded)  # (batch*10*100, 16)
        
        key_id_features_int = key_id_features.squeeze(-1).long().clamp(0, self.enum_sizes['key_id'] - 1)
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
        
        # 5. Return action sequence predictions using new decoder
        # The new decoder takes [B, hidden_dim] context and produces [B, max_actions, ...] outputs
        return self.action_decoder(fused_output, action_sequence, valid_mask)  # dict of heads
    

    
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

def create_model(data_config: Dict = None, model_config: Dict = None, data_dir: str = None, feature_spec: Dict = None) -> ImitationHybridModel:
    """Factory function to create the model with data-driven configuration"""
    if data_config is None:
        # Create default data config for testing
        data_config = {
            'gamestate_dim': 128,
            'max_actions': 100,
            'action_features': 7,
            'temporal_window': 10,
            'enum_sizes': {'button': 3, 'key_action': 3, 'key_id': 6, 'scroll': 3},
            'event_types': 4
        }
    
    if model_config is None:
        model_config = {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6
        }
    
    # Load feature specification if data_dir is provided or use provided feature_spec
    if feature_spec is None:
        feature_spec = {}
        if data_dir:
            try:
                from pathlib import Path
                from ilbot.utils.feature_spec import load_feature_spec
                feature_spec = load_feature_spec(Path(data_dir))
                # Successfully loaded feature spec
            except Exception as e:
                print(f"WARNING: Could not load feature spec: {e}")
                feature_spec = {}
    
    model = ImitationHybridModel(data_config=data_config, feature_spec=feature_spec, **model_config)
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
