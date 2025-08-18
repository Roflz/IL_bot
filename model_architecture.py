#!/usr/bin/env python3
"""
OSRS Imitation Learning Model Architecture
Phase 2: Enhanced Model for Variable-Length Action Sequences (600ms windows)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class OSRSImitationModel(nn.Module):
    """
    Enhanced imitation learning model for OSRS gameplay.
    
    Input: Sequence of 10 gamestates (128 features each)
    Output: Variable-length sequence of actions for next 600ms
    """
    
    def __init__(
        self,
        input_features: int = 128,
        sequence_length: int = 10,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_actions_per_window: int = 15,  # Maximum actions in 600ms
        action_heads: bool = True
    ):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_actions_per_window = max_actions_per_window
        self.action_heads = action_heads
        
        # 1. Gamestate Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Temporal Sequence Encoder (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            dropout=dropout if num_layers > 1 else 0,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 3. Action Sequence Generator
        if action_heads:
            # Action count predictor (how many actions in 600ms)
            self.action_count_head = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, max_actions_per_window + 1)  # 0 to max_actions
            )
            
            # Action sequence decoder (LSTM for generating action sequences)
            self.action_sequence_decoder = nn.LSTM(
                input_size=hidden_size + 4,  # hidden + action embedding
                hidden_size=hidden_size // 2,
                num_layers=2,
                dropout=dropout,
                batch_first=True
            )
            
            # Action type predictor for each position
            self.action_type_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 4)  # none, mouse_click, key_press, scroll
            )
            
            # Mouse position predictor (x, y coordinates)
            self.mouse_position_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2)  # (x, y)
            )
            
            # Mouse click type predictor (left, right, none)
            self.mouse_click_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3)  # (none, left, right)
            )
            
            # Key press predictor (most common keys)
            self.key_press_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 50)  # 50 most common keys
            )
            
            # Scroll action predictor (dx, dy)
            self.scroll_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2)  # (dx, dy)
            )
            
            # Timing predictor (when each action occurs in 600ms window)
            self.timing_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),  # Relative time in 600ms window (0-600)
                nn.Sigmoid()  # Normalize to 0-1, multiply by 600
            )
            
            # Action confidence predictor
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_size // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 0-1 confidence
            )
    
    def forward(
        self, 
        gamestate_sequence: torch.Tensor,
        max_actions: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            gamestate_sequence: (batch_size, sequence_length, input_features)
            max_actions: Maximum number of actions to generate (default: predicted count)
            
        Returns:
            Dictionary containing action sequence predictions
        """
        batch_size, seq_len, feat_dim = gamestate_sequence.shape
        
        # 1. Encode individual gamestate features
        # (batch_size, sequence_length, 256)
        encoded_features = self.feature_encoder(gamestate_sequence)
        
        # 2. Process temporal sequence with LSTM
        # (batch_size, sequence_length, hidden_size)
        lstm_out, (hidden, cell) = self.temporal_encoder(encoded_features)
        
        # 3. Use final timestep output for action prediction
        # (batch_size, hidden_size)
        final_output = lstm_out[:, -1, :]
        
        # 4. Generate action sequence predictions
        if self.action_heads:
            # Predict how many actions in the 600ms window
            action_count_logits = self.action_count_head(final_output)
            action_count_probs = F.softmax(action_count_logits, dim=-1)
            
            # Use provided max_actions or sample from predicted distribution
            if max_actions is None:
                # Sample action count from predicted distribution
                action_count = torch.multinomial(action_count_probs, 1).squeeze(-1)
            else:
                action_count = torch.full((batch_size,), max_actions, dtype=torch.long)
            
            # Generate action sequence
            actions = self._generate_action_sequence(
                final_output, action_count, batch_size
            )
            
            # Add action count to output
            actions['action_count'] = action_count
            actions['action_count_probs'] = action_count_probs
            
        else:
            # For feature extraction only
            actions = {
                'features': final_output
            }
        
        return actions
    
    def _generate_action_sequence(
        self, 
        context: torch.Tensor, 
        action_count: torch.Tensor, 
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Generate variable-length action sequence using LSTM decoder.
        
        Args:
            context: Context vector from gamestate encoder (batch_size, hidden_size)
            action_count: Number of actions to generate per batch item (batch_size,)
            batch_size: Batch size
            
        Returns:
            Dictionary of action sequence predictions
        """
        max_actions_in_batch = action_count.max().item()
        
        # Initialize action sequence tensors
        action_types = torch.zeros(batch_size, max_actions_in_batch, 4, device=context.device)
        mouse_positions = torch.zeros(batch_size, max_actions_in_batch, 2, device=context.device)
        mouse_clicks = torch.zeros(batch_size, max_actions_in_batch, 3, device=context.device)
        key_presses = torch.zeros(batch_size, max_actions_in_batch, 50, device=context.device)
        scrolls = torch.zeros(batch_size, max_actions_in_batch, 2, device=context.device)
        timings = torch.zeros(batch_size, max_actions_in_batch, 1, device=context.device)
        confidences = torch.zeros(batch_size, max_actions_in_batch, 1, device=context.device)
        
        # Initialize decoder hidden state
        decoder_hidden = (context.unsqueeze(0).expand(2, -1, -1),  # 2 layers
                         context.unsqueeze(0).expand(2, -1, -1))
        
        # Generate actions sequentially
        for action_idx in range(max_actions_in_batch):
            # Create input for this action position
            # Concatenate context with action position embedding
            action_pos_embedding = torch.full((batch_size, 1), action_idx, dtype=torch.float32, device=context.device)
            action_pos_embedding = action_pos_embedding / max_actions_in_batch  # Normalize to 0-1
            
            # Add some additional context for action generation
            action_context = torch.cat([
                context,
                action_pos_embedding.squeeze(-1),
                torch.zeros(batch_size, 3, device=context.device)  # Placeholder for previous action
            ], dim=-1)
            
            # Decode this action
            decoder_input = action_context.unsqueeze(1)  # Add sequence dimension
            decoder_output, decoder_hidden = self.action_sequence_decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.squeeze(1)  # Remove sequence dimension
            
            # Predict action components
            action_types[:, action_idx] = self.action_type_head(decoder_output)
            mouse_positions[:, action_idx] = self.mouse_position_head(decoder_output)
            mouse_clicks[:, action_idx] = self.mouse_click_head(decoder_output)
            key_presses[:, action_idx] = self.key_press_head(decoder_output)
            scrolls[:, action_idx] = self.scroll_head(decoder_output)
            timings[:, action_idx] = self.timing_head(decoder_output)
            confidences[:, action_idx] = self.confidence_head(decoder_output)
        
        return {
            'action_types': action_types,
            'mouse_positions': mouse_positions,
            'mouse_clicks': mouse_clicks,
            'key_presses': key_presses,
            'scrolls': scrolls,
            'timings': timings,
            'confidences': confidences
        }
    
    def get_action_probabilities(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert raw model outputs to probabilities.
        
        Args:
            actions: Raw model outputs
            
        Returns:
            Dictionary with probability distributions
        """
        probs = {}
        
        # Action count: Already softmax
        if 'action_count_probs' in actions:
            probs['action_count_probs'] = actions['action_count_probs']
        
        # Action types: Softmax over action types
        if 'action_types' in actions:
            probs['action_types'] = F.softmax(actions['action_types'], dim=-1)
        
        # Mouse position: Keep as-is (regression)
        if 'mouse_positions' in actions:
            probs['mouse_positions'] = actions['mouse_positions']
        
        # Mouse click: Softmax over click types
        if 'mouse_clicks' in actions:
            probs['mouse_clicks'] = F.softmax(actions['mouse_clicks'], dim=-1)
        
        # Key press: Softmax over key options
        if 'key_presses' in actions:
            probs['key_presses'] = F.softmax(actions['key_presses'], dim=-1)
        
        # Scroll: Keep as-is (regression)
        if 'scrolls' in actions:
            probs['scrolls'] = actions['scrolls']
        
        # Timing: Already sigmoid (0-1), multiply by 600 for ms
        if 'timings' in actions:
            probs['timings'] = actions['timings'] * 600.0
        
        # Confidence: Already sigmoid (0-1)
        if 'confidences' in actions:
            probs['confidences'] = actions['confidences']
        
        return probs
    
    def predict_action_sequence(self, 
                              gamestate_sequence: torch.Tensor,
                              temperature: float = 1.0,
                              max_actions: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate action sequence with temperature sampling for more diverse outputs.
        
        Args:
            gamestate_sequence: Input gamestate sequence
            temperature: Sampling temperature (higher = more random)
            max_actions: Maximum actions to generate
            
        Returns:
            Sampled action sequence
        """
        # Get raw predictions
        raw_predictions = self.forward(gamestate_sequence, max_actions)
        
        # Apply temperature to logits
        if temperature != 1.0:
            for key in ['action_types', 'mouse_clicks', 'key_presses']:
                if key in raw_predictions:
                    raw_predictions[key] = raw_predictions[key] / temperature
        
        # Convert to probabilities
        probabilities = self.get_action_probabilities(raw_predictions)
        
        # Sample from distributions
        sampled_actions = {}
        
        # Sample action count
        if 'action_count_probs' in probabilities:
            sampled_actions['action_count'] = torch.multinomial(
                probabilities['action_count_probs'], 1
            ).squeeze(-1)
        
        # Sample action types
        if 'action_types' in probabilities:
            sampled_actions['action_types'] = torch.multinomial(
                probabilities['action_types'].view(-1, 4), 1
            ).view(probabilities['action_types'].shape[:2])
        
        # Sample mouse clicks
        if 'mouse_clicks' in probabilities:
            sampled_actions['mouse_clicks'] = torch.multinomial(
                probabilities['mouse_clicks'].view(-1, 3), 1
            ).view(probabilities['mouse_clicks'].shape[:2])
        
        # Sample key presses
        if 'key_presses' in probabilities:
            sampled_actions['key_presses'] = torch.multinomial(
                probabilities['key_presses'].view(-1, 50), 1
            ).view(probabilities['key_presses'].shape[:2])
        
        # Keep continuous values as-is
        for key in ['mouse_positions', 'scrolls', 'timings', 'confidences']:
            if key in raw_predictions:
                sampled_actions[key] = raw_predictions[key]
        
        return sampled_actions


class OSRSDataLoader:
    """
    Data loader for OSRS training data.
    """
    
    def __init__(
        self,
        input_sequences_path: str,
        action_data_path: str,
        batch_size: int = 16,
        shuffle: bool = True
    ):
        self.input_sequences_path = input_sequences_path
        self.action_data_path = action_data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load data
        self.input_sequences = np.load(input_sequences_path)
        with open(action_data_path, 'r') as f:
            self.action_data = json.load(f)
        
        self.n_sequences = len(self.input_sequences)
        self.sequence_length = self.input_sequences.shape[1]
        self.n_features = self.input_sequences.shape[2]
        
        print(f"Loaded {self.n_sequences} sequences")
        print(f"Sequence shape: {self.input_sequences.shape}")
    
    def __len__(self):
        return (self.n_sequences + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        # Create indices
        indices = list(range(self.n_sequences))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield batches
        for i in range(0, self.n_sequences, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self._get_batch(batch_indices)
    
    def _get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of data."""
        batch_sequences = []
        batch_actions = []
        
        for idx in indices:
            # Get input sequence
            sequence = self.input_sequences[idx]  # (10, 128)
            batch_sequences.append(sequence)
            
            # Get target actions (simplified for now)
            # TODO: Implement proper action target extraction
            target_action = self._extract_target_action(idx)
            batch_actions.append(target_action)
        
        return {
            'input_sequences': torch.FloatTensor(np.array(batch_sequences)),
            'target_actions': torch.FloatTensor(np.array(batch_actions))
        }
    
    def _extract_target_action(self, sequence_idx: int) -> np.ndarray:
        """Extract target action for a sequence (placeholder)."""
        # TODO: Implement proper action target extraction from action_data
        # For now, return a dummy target
        return np.array([400.0, 300.0, 0, 0, 0, 0.5])  # (x, y, click_type, key, scroll, confidence)


def create_model(
    input_features: int = 128,
    sequence_length: int = 10,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2
) -> OSRSImitationModel:
    """
    Create and configure the OSRS imitation learning model.
    
    Args:
        input_features: Number of input features per gamestate
        sequence_length: Number of timesteps in sequence
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        
    Returns:
        Configured model
    """
    model = OSRSImitationModel(
        input_features=input_features,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: (batch_size, {sequence_length}, {input_features})")
    print(f"Hidden size: {hidden_size}")
    print(f"LSTM layers: {num_layers}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing OSRS Imitation Learning Model...")
    
    # Create model
    model = create_model(
        input_features=128,
        sequence_length=10,
        hidden_size=256,
        num_layers=2,
        dropout=0.2
    )
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 10, 128)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"\nTest forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    for key, value in output.items():
        print(f"Output {key}: {value.shape}")
    
    # Test action probabilities
    probs = model.get_action_probabilities(output)
    print(f"\nAction probabilities:")
    for key, value in probs.items():
        print(f"  {key}: {value.shape}")
