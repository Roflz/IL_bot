#!/usr/bin/env python3
"""
Sequential Imitation Learning Model

This model uses the SequentialActionDecoder to generate actions sequentially
with cumulative timing, addressing the fundamental issues with parallel generation.

Key Features:
1. Preserves excellent temporal processing from the original model
2. Uses SequentialActionDecoder for sequential action generation
3. Natural sequence length emergence from timing constraints
4. Temporal causality between actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .imitation_hybrid_model import ImitationHybridModel
from .sequential_action_decoder import SequentialActionDecoder


class SequentialImitationModel(ImitationHybridModel):
    """
    Sequential Imitation Learning Model that extends the original model
    with sequential action generation capabilities.
    
    This model preserves all the excellent temporal processing from the original
    ImitationHybridModel but replaces the ActionSequenceDecoder with the
    SequentialActionDecoder for better timing and sequence length handling.
    """
    
    def __init__(self, data_config: dict, hidden_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, feature_spec: dict | None = None, **kwargs):
        # Initialize the parent model (preserves all temporal processing)
        super().__init__(data_config, hidden_dim, num_heads, num_layers, feature_spec, **kwargs)
        
        # Replace the ActionSequenceDecoder with SequentialActionDecoder
        self.action_decoder = SequentialActionDecoder(
            input_dim=hidden_dim,
            max_actions=self.max_actions,
            enum_sizes=self.enum_sizes,
            event_types=self.event_types
        )
        
        print("🚀 SequentialImitationModel initialized with SequentialActionDecoder")
        print(f"   - Preserves temporal processing: ✅")
        print(f"   - Sequential action generation: ✅")
        print(f"   - Cumulative timing: ✅")
        print(f"   - Natural sequence length: ✅")
    
    def forward(self, temporal_sequence: torch.Tensor, action_sequence: torch.Tensor, 
                valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sequential action generation.
        
        This method preserves all the excellent temporal processing from the parent
        model but uses sequential action generation instead of parallel generation.
        
        Args:
            temporal_sequence: (B, T, D) - Batch of temporal gamestate sequences
            action_sequence: (B, T, A, F) - Batch of action sequences (F=7 for V2 actions)
            valid_mask: (B, A) - Optional mask indicating which actions are valid vs padded
        
        Returns:
            Dict of action predictions with cumulative timing
        """
        # Use the parent model's excellent temporal processing
        batch_size = temporal_sequence.size(0)
        
        # 1. Encode current gamestate features (last timestep of temporal sequence)
        current_gamestate = temporal_sequence[:, -1, :]  # (batch_size, 128)
        gamestate_encoded = self.gamestate_encoder(current_gamestate.unsqueeze(1))  # Add sequence dimension
        gamestate_encoded = gamestate_encoded.squeeze(1)  # Remove sequence dimension
        
        # 2. Encode action sequence (10 timesteps, 100 actions, 7 features -> 10 timesteps, 100 actions, hidden_dim//2)
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
        
        # Process actions within each timestep using LSTM to preserve action detail
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
        
        # 5. PHASE 2: Use SequentialActionDecoder for sequential generation
        return self.action_decoder(fused_output, action_sequence, valid_mask)
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model information including sequential decoder info"""
        info = super().get_model_info()
        info['decoder_type'] = 'SequentialActionDecoder'
        info['generation_method'] = 'Sequential with Cumulative Timing'
        return info
