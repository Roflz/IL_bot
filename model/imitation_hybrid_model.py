#!/usr/bin/env python3
"""
Imitation Learning Hybrid Model: Transformer + CNN + LSTM
OSRS Bot that learns to play like you
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

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
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature-type-specific encoders based on feature_mappings.json
        # Group features by data type for optimal encoding
        
        # Continuous/Coordinate features (world_coord, camera_coord, screen_coord, angles, time)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Categorical features (item_ids, animation_ids, slot_ids, object_ids, hashed_strings)
        # Use embeddings for proper categorical encoding
        # Assume vocabulary size of 10000 for categorical features (can be tuned)
        self.categorical_embedding = nn.Embedding(10000, hidden_dim // 16)
        self.categorical_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Boolean features
        self.boolean_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Count features
        self.count_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Time features (timestamps, durations)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Angle features (camera pitch/yaw)
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 16),  # 1 -> 16
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature grouping based on feature_mappings.json - CORRECTED INDICES
        # These indices are based on the actual feature structure analysis
        self.continuous_indices = [0, 1, 9, 10, 11, 46, 47, 51, 52, 56, 57, 61, 62, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123]  # world_coord, camera_coord, screen_coord
        self.categorical_indices = [2, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 50, 55, 60, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 125, 126, 63]  # item_ids, animation_ids, object_ids, npc_ids, slot_ids, hashed_strings, etc.
        self.boolean_indices = [3, 42, 43, 48, 53, 58]  # boolean flags
        self.count_indices = [44, 49, 54, 59, 66]  # counts
        self.time_indices = [8, 64, 65, 127]  # timestamps, durations
        self.angle_indices = [12, 13]  # camera pitch/yaw
        
        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(2048, hidden_dim),  # 51*16 + 60*16 + 6*16 + 5*16 + 4*16 + 2*16 = 2048 -> 256
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature preprocessing for proper scaling
        self.feature_preprocessor = nn.Sequential(
            nn.LayerNorm(input_dim),  # Normalize across the 128 features
            nn.Dropout(0.05)  # Light dropout for regularization
        )
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("testing")
        print(x)
        print(x.shape)
        batch_size, seq_len, features = x.shape
        
        # Preprocess features for better training stability
        x = self.feature_preprocessor(x)
        
        # Reshape to process all features: (batch_size * seq_len, features)
        x_flat = x.view(-1, features)
        
        # Encode features by type
        continuous_features = x_flat[:, self.continuous_indices]
        categorical_features = x_flat[:, self.categorical_indices]
        boolean_features = x_flat[:, self.boolean_indices]
        count_features = x_flat[:, self.count_indices]
        time_features = x_flat[:, self.time_indices]
        angle_features = x_flat[:, self.angle_indices]
        
        # Encode each feature type
        continuous_encoded = self.continuous_encoder(continuous_features.unsqueeze(-1))  # (batch*seq, n_continuous, 16)
        
        # Categorical features: convert to embeddings then encode
        categorical_features_int = categorical_features.long().clamp(0, 9999)  # Ensure valid indices
        categorical_embedded = self.categorical_embedding(categorical_features_int)  # (batch*seq, n_categorical, 16)
        categorical_encoded = self.categorical_encoder(categorical_embedded)  # (batch*seq, n_categorical, 16)
        
        boolean_encoded = self.boolean_encoder(boolean_features.unsqueeze(-1))  # (batch*seq, n_boolean, 16)
        count_encoded = self.count_encoder(count_features.unsqueeze(-1))  # (batch*seq, n_count, 16)
        time_encoded = self.time_encoder(time_features.unsqueeze(-1))  # (batch*seq, n_time, 16)
        angle_encoded = self.angle_encoder(angle_features.unsqueeze(-1))  # (batch*seq, n_angle, 16)
        
        # Flatten the encoded features: (batch*seq, n_features, 16) -> (batch*seq, n_features * 16)
        continuous_flat = continuous_encoded.view(continuous_encoded.size(0), -1)  # (batch*seq, n_continuous * 16)
        categorical_flat = categorical_encoded.view(categorical_encoded.size(0), -1)  # (batch*seq, n_categorical * 16)
        boolean_flat = boolean_encoded.view(boolean_encoded.size(0), -1)  # (batch*seq, n_boolean * 16)
        count_flat = count_encoded.view(count_encoded.size(0), -1)  # (batch*seq, n_count * 16)
        time_flat = time_encoded.view(time_encoded.size(0), -1)  # (batch*seq, n_time * 16)
        angle_flat = angle_encoded.view(angle_encoded.size(0), -1)  # (batch*seq, n_angle * 16)
        
        # Combine all encoded features
        combined_features = torch.cat([
            continuous_flat, categorical_flat, boolean_flat, count_flat, time_flat, angle_flat
        ], dim=1)  # (batch*seq, total_encoded_features)
        
        # Final feature combination
        x_encoded = self.feature_combiner(combined_features)  # (batch*seq, hidden_dim)
        
        # Reshape back: (batch_size, seq_len, hidden_dim)
        x_encoded = x_encoded.view(batch_size, seq_len, -1)
        
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
    """Decoder for OSRS action tensors with discrete categorical outputs"""
    
    def __init__(self, input_dim: int = 256, max_actions: int = 100, 
                 screen_width: int = 1920, screen_height: int = 1080):
        super().__init__()
        self.max_actions = max_actions
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Split into separate heads for different feature types
        # Features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
        
        # Time head (continuous) - keep as linear
        self.time_head = nn.Linear(input_dim, max_actions)
        
        # Action type head (categorical: 0,1,2,3) - 4 categories
        self.action_type_head = nn.Linear(input_dim, max_actions * 4)
        
        # Coordinate heads (discrete integers) - use sigmoid + rounding
        self.x_coord_head = nn.Linear(input_dim, max_actions)
        self.y_coord_head = nn.Linear(input_dim, max_actions)
        
        # Button head (categorical: 0,1,2,3) - 4 categories  
        self.button_head = nn.Linear(input_dim, max_actions * 4)
        
        # Key head (categorical) - use key categories + 0 for "no key"
        # Based on key_mapper.py: ~150 key categories + 1 for "no key" = 151 total
        self.key_head = nn.Linear(input_dim, max_actions * 151)
        
        # Scroll heads (categorical: -1,0,1) - use tanh + sign
        self.scroll_x_head = nn.Linear(input_dim, max_actions)
        self.scroll_y_head = nn.Linear(input_dim, max_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_features(x)
        batch_size = x.size(0)
        
        # Decode each feature type separately
        
        # 1. Time (continuous) - keep as is
        time_output = self.time_head(shared)  # (batch_size, max_actions)
        
        # 2. Action type (categorical: 0,1,2,3)
        action_type_logits = self.action_type_head(shared)  # (batch_size, max_actions * 4)
        action_type_logits = action_type_logits.view(batch_size, self.max_actions, 4)
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        action_type_output = torch.argmax(action_type_probs, dim=-1).float()  # (batch_size, max_actions)
        
        # 3. Coordinates (discrete integers) - use sigmoid + rounding
        x_coord_raw = torch.sigmoid(self.x_coord_head(shared))  # (batch_size, max_actions)
        y_coord_raw = torch.sigmoid(self.y_coord_head(shared))  # (batch_size, max_actions)
        
        # Scale to screen dimensions and round to integers
        x_coord_output = torch.round(x_coord_raw * self.screen_width).float()  # (batch_size, max_actions)
        y_coord_output = torch.round(y_coord_raw * self.screen_height).float()  # (batch_size, max_actions)
        
        # 4. Button (categorical: 0,1,2,3)
        button_logits = self.button_head(shared)  # (batch_size, max_actions * 4)
        button_logits = button_logits.view(batch_size, self.max_actions, 4)
        button_probs = F.softmax(button_logits, dim=-1)
        button_output = torch.argmax(button_probs, dim=-1).float()  # (batch_size, max_actions)
        
        # 5. Key (categorical) - use key categories + 0 for "no key"
        key_logits = self.key_head(shared)  # (batch_size, max_actions * 151)
        key_logits = key_logits.view(batch_size, self.max_actions, 151)
        key_probs = F.softmax(key_logits, dim=-1)
        key_output = torch.argmax(key_probs, dim=-1).float()  # (batch_size, max_actions)
        
        # 6. Scroll (categorical: -1,0,1) - use tanh + sign
        scroll_x_raw = torch.tanh(self.scroll_x_head(shared))  # (batch_size, max_actions)
        scroll_y_raw = torch.tanh(self.scroll_y_head(shared))  # (batch_size, max_actions)
        
        # Convert to -1, 0, 1
        scroll_x_output = torch.sign(scroll_x_raw)  # (batch_size, max_actions)
        scroll_y_output = torch.sign(scroll_y_raw)  # (batch_size, max_actions)
        
        # Stack all features: [time, type, x, y, button, key, scroll_x, scroll_y]
        action_tensor = torch.stack([
            time_output,        # (batch_size, max_actions)
            action_type_output, # (batch_size, max_actions)
            x_coord_output,     # (batch_size, max_actions)
            y_coord_output,     # (batch_size, max_actions)
            button_output,      # (batch_size, max_actions)
            key_output,         # (batch_size, max_actions)
            scroll_x_output,    # (batch_size, max_actions)
            scroll_y_output     # (batch_size, max_actions)
        ], dim=-1)  # (batch_size, max_actions, 8)
        
        return action_tensor

class ImitationHybridModel(nn.Module):
    """Complete hybrid model combining Transformer + CNN + LSTM with action sequence input"""
    
    def __init__(self, 
                 gamestate_dim: int = 128,  # Updated to match your data
                 action_dim: int = 8,       # Action features per timestep
                 sequence_length: int = 10,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8,
                 screen_width: int = 1920,  # Screen width for coordinate scaling
                 screen_height: int = 1080): # Screen height for coordinate scaling
        super().__init__()
        
        self.gamestate_dim = gamestate_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 1. Gamestate Feature Encoder (128 -> 256)
        self.gamestate_encoder = GamestateEncoder(
            input_dim=gamestate_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads
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
        
        # Button encoder (categorical: 0,1,2,3) - 4 classes: none, left, right, middle
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
        self.action_decoder = ActionDecoder(input_dim=hidden_dim, max_actions=100, screen_width=screen_width, screen_height=screen_height)
        
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
    
    def forward(self, 
                temporal_sequence: torch.Tensor,
                action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model with action sequence input
        
        Args:
            temporal_sequence: (batch_size, 10, 128) - Sequence of 10 gamestates
            action_sequence: (batch_size, 10, 100, 8) - Sequence of 10 timesteps, each with up to 100 actions, 8 features per action
            
        Returns:
            Dictionary of action predictions
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
        
        # Preprocess features for better training stability
        action_sequence_flat = self.feature_preprocessor(action_sequence_flat)
        
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
        action_type_features_int = action_type_features.squeeze(-1).long().clamp(0, 3)  # 4 categories
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
        action_tensor = self.action_decoder(fused_output)
        
        return action_tensor  # (batch_size, max_actions, 8)
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model information and parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'gamestate_dim': self.gamestate_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height
        }

def create_model(config: Dict = None) -> ImitationHybridModel:
    """Factory function to create the model with default or custom config"""
    if config is None:
        config = {
            'gamestate_dim': 128,  # Updated to match your data pipeline
            'action_dim': 8,       # Action features per timestep
            'sequence_length': 10,
            'hidden_dim': 256,
            'num_attention_heads': 8,
            'screen_width': 1920,  # Default screen width
            'screen_height': 1080  # Default screen height
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
        output = model(temporal_sequence, action_sequence)
    
    print(f"\nForward pass successful!")
    print(f"Input shapes: temporal={temporal_sequence.shape}, action={action_sequence.shape}")
    print(f"Output shape: {output.shape}")
