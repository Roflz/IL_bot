#!/usr/bin/env python3
"""
Sequential Action Decoder for OSRS Imitation Learning

This decoder generates actions sequentially, where each action depends on previous actions
and their timing. This addresses the fundamental issues with parallel action generation.

Key Features:
1. Sequential generation with cumulative timing
2. Natural sequence length emergence from timing constraints
3. Temporal causality between actions
4. Preserves existing temporal processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class SequentialActionDecoder(nn.Module):
    """
    Sequential Action Decoder that generates actions one at a time.
    
    This decoder addresses the core issues with the current architecture:
    - Timing predictions are cumulative and sequential
    - Sequence length emerges naturally from timing constraints
    - Each action depends on previous actions and their timing
    - Preserves the excellent temporal processing from the main model
    """
    
    def __init__(self, input_dim: int, max_actions: int, enum_sizes: dict, event_types: int):
        super().__init__()
        self.max_actions = max_actions
        self.enum_sizes = enum_sizes
        self.event_types = event_types
        self.input_dim = input_dim
        
        # Single action generator (called repeatedly for each action)
        # Calculate actual timing context size: 4 basic + 2 * (input_dim//4) + 1 density = 5 + input_dim//2
        timing_context_size = 5 + input_dim // 2
        self.action_generator = nn.Sequential(
            nn.Linear(input_dim + timing_context_size, input_dim),  # +timing_context_size for enhanced timing context
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal context processor (leverages the rich 6-second history)
        self.temporal_context_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
        
        # Action history processor (for temporal causality)
        self.action_history_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
        
        # Output heads for single action
        self.event_head = nn.Linear(input_dim, event_types)
        self.time_quantile_head = nn.Linear(input_dim, 3)  # Quantile timing predictions (q0.1, q0.5, q0.9)
        
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, context: torch.Tensor, action_history: Optional[torch.Tensor] = None, 
                valid_mask: Optional[torch.Tensor] = None, max_time: float = 0.6) -> Dict[str, torch.Tensor]:
        """
        Sequential action generation with cumulative timing.
        
        Args:
            context: [B, input_dim] - fused context from gamestate + action history + temporal
            action_history: [B, T, A, F] - 10 timesteps of action history for temporal awareness
            valid_mask: [B, A] - valid action mask (for compatibility)
            max_time: Maximum time window (0.6s)
            
        Returns:
            Dict of action predictions with cumulative timing
        """
        batch_size = context.size(0)
        device = context.device
        
        # Initialize storage for all actions
        all_actions = {
            'event_logits': [],
            'time_q': [],
            'time_deltas': [],
            'cumulative_times': [],
            'x_mu': [],
            'x_logsig': [],
            'y_mu': [],
            'y_logsig': [],
            'button_logits': [],
            'key_action_logits': [],
            'key_id_logits': [],
            'scroll_y_logits': []
        }
        
        # Initialize timing state
        current_time = torch.zeros(batch_size, 1, device=device)
        continue_generation = torch.ones(batch_size, 1, device=device)
        
        # Process temporal context to understand action patterns
        temporal_context = self.temporal_context_processor(context)  # [B, input_dim//4]
        
        # Process action history for temporal awareness
        action_history_context = self._process_action_history(action_history, context)  # [B, input_dim//4]
        
        # Sequential generation
        for step in range(self.max_actions):
            # Check if any batch elements should continue
            if continue_generation.sum() == 0:
                break
            
            # Create enhanced timing context for this step
            timing_context = self._create_timing_context(
                current_time, max_time, step, temporal_context, action_history_context, device
            )
            
            # Combine context with timing
            step_context = torch.cat([context, timing_context], dim=1)  # [B, input_dim + timing_context_size]
            
            # Generate single action
            action_features = self.action_generator(step_context)  # [B, input_dim]
            
            # Predict timing for this action (quantiles)
            time_q_raw = self.time_quantile_head(action_features)  # [B, 3]
            # Scale timing predictions to match target distribution (10-13ms range)
            # Target timing deltas are in milliseconds, so convert to seconds: ms/1000
            time_q_deltas = F.softplus(time_q_raw) * 0.005 + 0.010  # [B, 3] - scale to ~10-15ms range (0.010-0.015s)
            delta_time = time_q_deltas[:, 1:2]  # [B, 1] - use median (q0.5) for generation logic
            
            # Check if we exceed 600ms window
            next_time = current_time + delta_time
            step_continue = (next_time <= max_time).float() * continue_generation
            
            # Predict action features
            event_logits = self.event_head(action_features)  # [B, event_types]
            x_mu = self.x_mu_head(action_features)  # [B, 1]
            x_logsig = self.x_logsig_head(action_features)  # [B, 1]
            y_mu = self.y_mu_head(action_features)  # [B, 1]
            y_logsig = self.y_logsig_head(action_features)  # [B, 1]
            
            # Event-specific predictions
            button_logits = self.button_head(action_features)  # [B, button_types]
            key_action_logits = self.key_action_head(action_features)  # [B, key_action_types]
            key_id_logits = self.key_id_head(action_features)  # [B, key_id_types]
            scroll_y_logits = self.scroll_y_head(action_features)  # [B, scroll_types]
            
            # Store action predictions
            all_actions['event_logits'].append(event_logits)
            all_actions['time_q'].append(time_q_deltas)
            all_actions['time_deltas'].append(delta_time)
            all_actions['cumulative_times'].append(next_time)
            all_actions['x_mu'].append(x_mu)
            all_actions['x_logsig'].append(x_logsig)
            all_actions['y_mu'].append(y_mu)
            all_actions['y_logsig'].append(y_logsig)
            all_actions['button_logits'].append(button_logits)
            all_actions['key_action_logits'].append(key_action_logits)
            all_actions['key_id_logits'].append(key_id_logits)
            all_actions['scroll_y_logits'].append(scroll_y_logits)
            
            # Update timing state
            current_time = next_time * step_continue + current_time * (1 - step_continue)
            continue_generation = step_continue
        
        # Pad sequences to max_actions length
        outputs = self._pad_sequences(all_actions, batch_size, device)
        
        # Compute cumulative timing from quantiles (for loss computation)
        if valid_mask is not None:
            outputs['time_q'] = self._compute_cumulative_timing(outputs['time_q'], valid_mask)
        else:
            # Create a default valid mask if none provided
            batch_size = outputs['time_q'].shape[0]
            max_actions = outputs['time_q'].shape[1]
            default_valid_mask = torch.ones(batch_size, max_actions, device=outputs['time_q'].device)
            outputs['time_q'] = self._compute_cumulative_timing(outputs['time_q'], default_valid_mask)
        
        # Add sequence length (derived from timing)
        outputs['sequence_length'] = self._compute_sequence_length(outputs['cumulative_times'], max_time)
        
        return outputs
    
    def _create_timing_context(self, current_time: torch.Tensor, max_time: float, step: int,
                              temporal_context: torch.Tensor, action_history_context: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        """
        Create enhanced timing context for sequential generation.
        
        Args:
            current_time: [B, 1] - current cumulative time
            max_time: Maximum time window
            step: Current step number
            temporal_context: [B, input_dim//4] - temporal context from history
            action_history_context: [B, input_dim//4] - action history context
            device: Device for tensor creation
            
        Returns:
            timing_context: [B, 5 + input_dim//2] - enhanced timing context
        """
        batch_size = current_time.size(0)
        
        # Basic timing information
        remaining_time = max_time - current_time
        step_progress = torch.tensor([step / self.max_actions], device=device).expand(batch_size, 1)
        max_steps_indicator = torch.ones(batch_size, 1, device=device)
        
        # Action density from historical data
        action_density = self._compute_action_density(current_time, device)
        
        # Combine all timing context
        timing_context = torch.cat([
            current_time,                    # [B, 1] - current cumulative time
            remaining_time,                  # [B, 1] - remaining time
            step_progress,                   # [B, 1] - step progress
            max_steps_indicator,             # [B, 1] - max steps indicator
            temporal_context,                # [B, input_dim//4] - temporal context
            action_history_context,          # [B, input_dim//4] - action history context
            action_density                   # [B, 1] - action density
        ], dim=1)  # [B, 5 + input_dim//2]
        
        return timing_context
    
    def _process_action_history(self, action_history: Optional[torch.Tensor], 
                               context: torch.Tensor) -> torch.Tensor:
        """
        Process action history for temporal awareness.
        
        Args:
            action_history: [B, T, A, F] - 10 timesteps of action history
            context: [B, input_dim] - main context
            
        Returns:
            action_history_context: [B, input_dim//4] - processed action history
        """
        if action_history is None:
            return torch.zeros(context.size(0), self.input_dim // 4, device=context.device)
        
        # Simple processing: extract timing patterns from action history
        batch_size = action_history.size(0)
        
        # Extract timing information from action history
        action_timings = action_history[:, :, :, 0]  # [B, T, A] - timing deltas
        
        # Compute action density patterns
        # Count actions in different time windows
        early_actions = (action_timings < 0.1).sum(dim=(1, 2)).float()  # < 100ms
        mid_actions = ((action_timings >= 0.1) & (action_timings < 0.3)).sum(dim=(1, 2)).float()  # 100-300ms
        late_actions = (action_timings >= 0.3).sum(dim=(1, 2)).float()  # > 300ms
        
        # Normalize by total possible actions
        total_possible = action_timings.size(1) * action_timings.size(2)
        early_density = early_actions / total_possible
        mid_density = mid_actions / total_possible
        late_density = late_actions / total_possible
        
        # Create action density context
        action_density_context = torch.stack([early_density, mid_density, late_density], dim=1)  # [B, 3]
        
        # Project to required dimension
        if action_density_context.size(1) < self.input_dim // 4:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.input_dim // 4 - 3, device=context.device)
            action_history_context = torch.cat([action_density_context, padding], dim=1)
        else:
            # Truncate
            action_history_context = action_density_context[:, :self.input_dim // 4]
        
        return action_history_context
    
    def _compute_action_density(self, current_time: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute action density based on current timing context.
        
        Args:
            current_time: [B, 1] - current cumulative time
            device: Device for tensor creation
            
        Returns:
            action_density: [B, 1] - action density indicator
        """
        batch_size = current_time.size(0)
        
        # Simple heuristic: higher density in early time windows
        # This encourages the model to generate more actions early in the 600ms window
        early_window = (current_time < 0.2).float()  # First 200ms
        mid_window = ((current_time >= 0.2) & (current_time < 0.4)).float()  # 200-400ms
        late_window = (current_time >= 0.4).float()  # 400-600ms
        
        # Action density decreases over time
        action_density = early_window * 0.8 + mid_window * 0.5 + late_window * 0.2
        
        return action_density
    
    def _pad_sequences(self, all_actions: Dict[str, list], batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Pad sequences to max_actions length.
        
        Args:
            all_actions: Dict of action lists
            batch_size: Batch size
            device: Device for tensor creation
            
        Returns:
            outputs: Dict of padded tensors
        """
        outputs = {}
        
        for key, action_list in all_actions.items():
            if not action_list:
                # Create empty tensor if no actions
                if key in ['x_mu', 'x_logsig', 'y_mu', 'y_logsig', 'time_deltas', 'cumulative_times']:
                    outputs[key] = torch.zeros(batch_size, self.max_actions, 1, device=device)
                elif key == 'time_q':
                    outputs[key] = torch.zeros(batch_size, self.max_actions, 3, device=device)
                else:
                    outputs[key] = torch.zeros(batch_size, self.max_actions, action_list[0].size(-1), device=device)
                continue
            
            # Stack actions
            stacked = torch.stack(action_list, dim=1)  # [B, num_actions, features]
            
            # Pad to max_actions
            if stacked.size(1) < self.max_actions:
                padding_size = self.max_actions - stacked.size(1)
                if stacked.dim() == 3:
                    padding = torch.zeros(batch_size, padding_size, stacked.size(2), device=device)
                else:
                    padding = torch.zeros(batch_size, padding_size, device=device)
                stacked = torch.cat([stacked, padding], dim=1)
            
            outputs[key] = stacked
        
        return outputs
    
    def _compute_cumulative_timing(self, time_deltas: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative timing from delta times.
        
        Args:
            time_deltas: [B, A, 3] - quantile timing deltas
            valid_mask: [B, A] - valid action mask
            
        Returns:
            cumulative_times: [B, A, 3] - cumulative timing quantiles
        """
        batch_size, max_actions, num_quantiles = time_deltas.shape
        device = time_deltas.device
        
        # Initialize cumulative times
        cumulative_times = torch.zeros_like(time_deltas)
        
        # Compute cumulative sum for each quantile
        for q in range(num_quantiles):
            # Get delta times for this quantile
            deltas = time_deltas[:, :, q]  # [B, A]
            
            # Apply valid mask (set invalid actions to 0)
            deltas = deltas * valid_mask
            
            # Compute cumulative sum
            cumulative = torch.cumsum(deltas, dim=1)  # [B, A]
            
            # Store in output tensor
            cumulative_times[:, :, q] = cumulative
        
        return cumulative_times
    
    def _compute_sequence_length(self, cumulative_times: torch.Tensor, max_time: float) -> torch.Tensor:
        """
        Compute sequence length from cumulative timing.
        
        Args:
            cumulative_times: [B, max_actions, 1] - cumulative timing predictions
            max_time: Maximum time window
            
        Returns:
            sequence_lengths: [B, 1] - sequence lengths
        """
        # Count actions that fit within the time window AND have non-zero timing
        # Actions with 0 cumulative time are padding and shouldn't be counted
        valid_actions = ((cumulative_times > 0) & (cumulative_times <= max_time)).sum(dim=1, keepdim=True).float()  # [B, 1]
        
        # Ensure minimum sequence length of 1
        sequence_lengths = torch.clamp(valid_actions, 1.0, self.max_actions)
        
        return sequence_lengths
