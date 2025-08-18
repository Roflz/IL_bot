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
    """Encoder for gamestate features with self-attention"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        # Feature projection
        x = self.feature_proj(x)
        
        # Self-attention with residual connection
        attn_out = self.attention1(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Second attention layer
        attn_out = self.attention2(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

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
    """Decoder for OSRS action tensors with action count at index 0"""
    
    def __init__(self, input_dim: int = 256, max_actions: int = 100):
        super().__init__()
        self.max_actions = max_actions
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Single action tensor decoder - outputs 8 features per action including count at index 0
        # Features: [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
        # Index 0: [action_count, 0, 0, 0, 0, 0, 0, 0] (count + padding)
        # Index 1+: actual actions with 8 features each
        self.action_tensor_head = nn.Linear(input_dim, (max_actions + 1) * 8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_features(x)
        
        # Decode action tensor (8 features per action, including count at index 0)
        action_tensor_flat = self.action_tensor_head(shared)
        action_tensor = action_tensor_flat.view(-1, self.max_actions + 1, 8)
        
        return action_tensor  # (batch_size, max_actions + 1, 8)

class ImitationHybridModel(nn.Module):
    """Complete hybrid model combining Transformer + CNN + LSTM with action sequence input"""
    
    def __init__(self, 
                 gamestate_dim: int = 128,  # Updated to match your data
                 action_dim: int = 8,       # Action features per timestep
                 sequence_length: int = 10,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8):
        super().__init__()
        
        self.gamestate_dim = gamestate_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 1. Gamestate Feature Encoder (128 -> 256)
        self.gamestate_encoder = GamestateEncoder(
            input_dim=gamestate_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads
        )
        
        # 2. Action Sequence Encoder (8 -> 128)
        # Input: (batch_size, 10, 101, 8) -> Output: (batch_size, 10, 101, hidden_dim // 2)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        self.action_decoder = ActionDecoder(input_dim=hidden_dim, max_actions=100)
        
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
            action_sequence: (batch_size, 10, 101, 8) - Sequence of 10 timesteps, each with action count at index 0 + up to 100 actions, 8 features per action
            
        Returns:
            Dictionary of action predictions
        """
        batch_size = temporal_sequence.size(0)
        
        # 1. Encode current gamestate features (last timestep of temporal sequence)
        current_gamestate = temporal_sequence[:, -1, :]  # (batch_size, 128)
        gamestate_encoded = self.gamestate_encoder(current_gamestate.unsqueeze(1))  # Add sequence dimension
        gamestate_encoded = gamestate_encoded.squeeze(1)  # Remove sequence dimension
        
        # 2. Encode action sequence (10 timesteps, 101 actions, 8 features -> 10 timesteps, 101 actions, hidden_dim//2)
        # action_sequence shape: (batch_size, 10, 101, 8)
        batch_size, seq_len, num_actions, action_features = action_sequence.shape
        
        # Reshape to process all actions: (batch_size * 10 * 101, 8)
        action_sequence_flat = action_sequence.view(-1, action_features)
        
        # Encode each action: (batch_size * 10 * 101, hidden_dim//2)
        action_encoded_flat = self.action_encoder(action_sequence_flat)
        
        # Reshape back: (batch_size, 10, 101, hidden_dim//2)
        action_encoded = action_encoded_flat.view(batch_size, seq_len, num_actions, -1)
        
        # NEW: Process actions within each timestep using LSTM to preserve action detail
        # Process each timestep's actions through the action sequence LSTM
        action_encoded_timesteps = []
        for i in range(seq_len):
            # Extract actions for this timestep: (batch_size, 101, hidden_dim//2)
            timestep_actions = action_encoded[:, i, :, :]
            
            # Process through LSTM: (batch_size, 101, hidden_dim) - bidirectional doubles the size
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
        
        return action_tensor  # (batch_size, max_actions + 1, 8)
    
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
            'num_attention_heads': 8
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
    action_sequence = torch.randn(batch_size, 10, 8)
    
    with torch.no_grad():
        output = model(temporal_sequence, action_sequence)
    
    print(f"\nForward pass successful!")
    print(f"Input shapes: temporal={temporal_sequence.shape}, action={action_sequence.shape}")
    print(f"Output shape: {output.shape}")
