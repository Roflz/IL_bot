# Deep Learning Analysis: OSRS Imitation Learning Model Architecture

## Current Architecture Overview

### **Problem Statement**
The model needs to predict the next 600ms of actions given:
- **Input**: Last 6 seconds of actions (10 x 600ms windows) + gamestate sequences
- **Output**: Next 600ms of actions (variable length sequence)
- **Key Challenge**: Timing, sequence length, and event classification are fundamentally intertwined

### **Current Architecture Issues**

#### 1. **Disconnected Prediction Heads**
The current model treats timing, sequence length, and event classification as separate, independent predictions:

```python
# Current approach - SEPARATE heads
self.sequence_length_head = nn.Sequential(...)  # Predicts how many actions
self.time_quantile_head = nn.Linear(...)        # Predicts timing for each action
self.event_head = nn.Linear(...)                # Predicts event type for each action
```

**Problem**: This ignores the fundamental relationship where:
- Timing determines sequence length (actions stop when cumulative time > 600ms)
- Sequence length affects timing (fewer actions = longer intervals)
- Event classification depends on timing context

#### 2. **Incorrect Timing Understanding**
Current model predicts timing as independent deltas:
```python
# Current: Each action has independent timing prediction
time_predictions = self.time_quantile_head(action_context)  # [B, A, 3]
```

**Problem**: Timing should be **cumulative and sequential**:
- Action 1: 50ms from start
- Action 2: 120ms from start (70ms after action 1)
- Action 3: 200ms from start (80ms after action 2)
- When cumulative time > 600ms, sequence ends

#### 3. **Sequence Length Prediction is Redundant**
The model predicts sequence length separately, but this is derivable from timing:
```python
# Redundant prediction
sequence_length = self.sequence_length_head(context)  # [B, 1]
```

**Problem**: Sequence length should emerge naturally from timing predictions, not be predicted separately.

#### 4. **No Temporal Causality**
The model processes all action slots independently:
```python
# Current: All actions processed in parallel
for i in range(max_actions):
    action_context = self.action_processors[i](combined_context)
```

**Problem**: Actions should be generated **sequentially** where each action depends on previous actions and their timing.

## **Current Temporal Data Processing Analysis**

### **How Current Model Processes Temporal Data**

The current model does a **good job** processing the temporal sequences:

#### **Action Sequence Processing (10 x 600ms windows)**
```python
# Current approach - WELL DESIGNED
# 1. Encode each action within each timestep: (B, 10, 100, 7) -> (B, 10, 100, hidden_dim//2)
action_encoded = self.action_encoder(action_sequence)  # Feature encoding

# 2. Process actions within each timestep using LSTM
for i in range(seq_len):  # For each of the 10 timesteps
    timestep_actions = action_encoded[:, i, :, :]  # (B, 100, hidden_dim//2)
    timestep_actions_encoded, _ = self.action_sequence_encoder(timestep_actions)  # LSTM
    timestep_representation = timestep_actions_encoded[:, -1, :]  # Last action as summary

# 3. Stack all 10 timestep representations: (B, 10, hidden_dim)
action_encoded = torch.stack(action_encoded_timesteps, dim=1)

# 4. Flatten and process: (B, 10*256) -> (B, 256)
processed_actions = self.action_sequence_processor(action_encoded_flat)
```

#### **Gamestate Sequence Processing (10 x 128 features)**
```python
# Current approach - WELL DESIGNED
# 1. Process all 10 timesteps through bidirectional LSTM
temporal_encoded = self.temporal_encoder(temporal_sequence)  # (B, 10, 128) -> (B, 256)

# 2. Uses last hidden state from both directions
last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # Concatenate bidirectional
output = self.output_proj(last_hidden)  # (B, 256)
```

#### **Multi-Modal Fusion**
```python
# Current approach - WELL DESIGNED
fused_features = torch.cat([
    gamestate_encoded,        # (B, 256) - current gamestate
    processed_actions,        # (B, 256) - 10 timesteps of action history
    temporal_encoded          # (B, 256) - 10 timesteps of gamestate history
], dim=-1)  # (B, 768)

fused_output = self.fusion_layer(fused_features)  # (B, 256)
```

### **Assessment: Temporal Processing is SOLID**

✅ **What's Working Well:**
1. **Action History**: Properly processes all 10 timesteps of action sequences
2. **Gamestate History**: Uses bidirectional LSTM to capture temporal patterns
3. **Multi-Modal Fusion**: Combines current gamestate + action history + gamestate history
4. **Rich Context**: Model has access to 6 seconds of historical context

❌ **What Needs Fixing:**
1. **Timing Predictions**: Still independent deltas instead of cumulative
2. **Sequence Length**: Still predicted separately instead of derived from timing
3. **Temporal Causality**: Actions generated in parallel instead of sequentially

## **Proposed Architecture Solutions**

### **Solution 1: Sequential Action Generation with Cumulative Timing (ENHANCED)**

#### **Core Concept**
Generate actions one at a time, where each action's timing is cumulative from the start of the 600ms window. **KEEP the existing temporal processing** but make action generation sequential.

#### **Enhanced Architecture (Preserves Temporal Processing)**
```python
class SequentialActionDecoder(nn.Module):
    def __init__(self, input_dim, max_actions, enum_sizes, event_types):
        super().__init__()
        self.max_actions = max_actions
        
        # Single action generator (called repeatedly) - ENHANCED with temporal awareness
        self.action_generator = nn.Sequential(
            nn.Linear(input_dim + 6, input_dim),  # +6 for enhanced timing context
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
        
        # Output heads for single action
        self.event_head = nn.Linear(input_dim, event_types)
        self.time_head = nn.Linear(input_dim, 1)  # Single timing prediction
        self.coord_heads = nn.ModuleDict({
            'x_mu': nn.Linear(input_dim, 1),
            'x_logsig': nn.Linear(input_dim, 1),
            'y_mu': nn.Linear(input_dim, 1),
            'y_logsig': nn.Linear(input_dim, 1)
        })
        # ... other heads
    
    def forward(self, context, action_history=None, max_time=0.6):
        """
        Enhanced forward pass that leverages temporal context
        
        Args:
            context: [B, input_dim] - fused context from gamestate + action history + temporal
            action_history: [B, T, A, F] - 10 timesteps of action history for temporal awareness
            max_time: Maximum time window (0.6s)
        """
        batch_size = context.size(0)
        actions = []
        current_time = torch.zeros(batch_size, 1).to(context.device)
        
        # Process temporal context to understand action patterns
        temporal_context = self.temporal_context_processor(context)  # [B, input_dim//4]
        
        for step in range(self.max_actions):
            # Enhanced timing context: [current_time, remaining_time, step, max_steps, temporal_context, action_density]
            timing_context = torch.cat([
                current_time,  # Current cumulative time
                max_time - current_time,  # Remaining time
                torch.tensor([step / self.max_actions]).expand(batch_size, 1).to(context.device),  # Step progress
                torch.tensor([1.0]).expand(batch_size, 1).to(context.device),  # Max steps indicator
                temporal_context,  # Temporal context from 6-second history
                self._compute_action_density(action_history, current_time)  # Action density in similar time windows
            ], dim=1)
            
            # Combine context with enhanced timing
            step_context = torch.cat([context, timing_context], dim=1)
            
            # Generate single action
            action_features = self.action_generator(step_context)
            
            # Predict timing for this action (delta from current time)
            # Use temporal context to inform timing predictions
            delta_time = F.softplus(self.time_head(action_features)) + 0.001  # [B, 1]
            
            # Check if we exceed 600ms window
            next_time = current_time + delta_time
            continue_generation = (next_time <= max_time).float()
            
            # Predict action features
            event_logits = self.event_head(action_features)
            coords = {k: head(action_features) for k, head in self.coord_heads.items()}
            
            # Store action
            action = {
                'event_logits': event_logits,
                'delta_time': delta_time,
                'cumulative_time': next_time,
                'continue': continue_generation,
                **coords
            }
            actions.append(action)
            
            # Update current time
            current_time = next_time * continue_generation + current_time * (1 - continue_generation)
            
            # Stop if no batch elements continue
            if continue_generation.sum() == 0:
                break
        
        return self._format_output(actions)
    
    def _compute_action_density(self, action_history, current_time):
        """Compute action density from historical data at similar time points"""
        if action_history is None:
            return torch.zeros(action_history.size(0), 1).to(current_time.device)
        
        # Analyze action density in similar time windows from history
        # This helps the model understand typical action patterns
        batch_size = action_history.size(0)
        density = torch.zeros(batch_size, 1).to(current_time.device)
        
        # Simple heuristic: count actions in similar time windows
        for b in range(batch_size):
            # Count actions in the first 200ms of each historical window
            early_actions = (action_history[b, :, :, 0] < 0.2).sum().float()
            density[b, 0] = early_actions / (action_history.size(1) * action_history.size(2))
        
        return density
```

#### **Key Benefits**
1. **Natural Sequence Length**: Emerges from timing, not predicted separately
2. **Cumulative Timing**: Each action knows its position in the 600ms window
3. **Temporal Causality**: Each action depends on previous actions
4. **Early Stopping**: Generation stops when 600ms is exceeded
5. **Preserves Temporal Processing**: Keeps the excellent 6-second history processing
6. **Enhanced Context**: Uses action density and temporal patterns to inform predictions
7. **Backward Compatible**: Can be implemented as a drop-in replacement for current decoder

### **Solution 2: Attention-Based Sequential Generation**

#### **Core Concept**
Use transformer-style attention where each action slot attends to previous actions and their timing.

#### **Architecture**
```python
class AttentionBasedDecoder(nn.Module):
    def __init__(self, input_dim, max_actions, enum_sizes, event_types):
        super().__init__()
        self.max_actions = max_actions
        
        # Position embeddings for action slots
        self.position_embeddings = nn.Embedding(max_actions, input_dim // 4)
        
        # Timing embeddings (discretized time slots)
        self.timing_embeddings = nn.Embedding(600, input_dim // 4)  # 600ms = 600 slots
        
        # Self-attention for action sequence
        self.action_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-attention to context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Action processors
        self.action_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(max_actions)
        ])
        
        # Output heads
        self.event_head = nn.Linear(input_dim, event_types)
        self.time_head = nn.Linear(input_dim, 1)
        # ... other heads
    
    def forward(self, context, valid_mask=None):
        batch_size = context.size(0)
        
        # Create action slot embeddings
        action_positions = torch.arange(self.max_actions, device=context.device)
        pos_embeddings = self.position_embeddings(action_positions)  # [max_actions, input_dim//4]
        
        # Create timing embeddings (discretized)
        timing_slots = torch.arange(600, device=context.device)  # 0-599ms
        timing_embeddings = self.timing_embeddings(timing_slots)  # [600, input_dim//4]
        
        # Initialize action representations
        action_reprs = torch.zeros(batch_size, self.max_actions, context.size(-1)).to(context.device)
        
        # Sequential generation with attention
        for step in range(self.max_actions):
            # Get current action representation
            current_action = action_reprs[:, step:step+1, :]  # [B, 1, input_dim]
            
            # Self-attention to previous actions
            if step > 0:
                prev_actions = action_reprs[:, :step, :]  # [B, step, input_dim]
                attended, _ = self.action_attention(current_action, prev_actions, prev_actions)
                current_action = current_action + attended
            
            # Cross-attention to context
            context_attended, _ = self.context_attention(current_action, context.unsqueeze(1), context.unsqueeze(1))
            current_action = current_action + context_attended
            
            # Process action
            processed_action = self.action_processors[step](current_action.squeeze(1))  # [B, input_dim]
            
            # Predict timing (cumulative from start)
            cumulative_time = F.softplus(self.time_head(processed_action)) * 0.6  # [B, 1] - scale to 600ms
            
            # Check if we should continue
            if step > 0:
                prev_cumulative = action_reprs[:, step-1, -1:1]  # Get previous cumulative time
                delta_time = cumulative_time - prev_cumulative
                continue_generation = (cumulative_time <= 0.6).float()
            else:
                delta_time = cumulative_time
                continue_generation = torch.ones(batch_size, 1).to(context.device)
            
            # Update action representation
            action_reprs[:, step, :] = processed_action
            
            # Stop if no batch elements continue
            if continue_generation.sum() == 0:
                break
        
        return self._generate_outputs(action_reprs)
```

### **Solution 3: Hybrid Approach with Timing-Aware Loss**

#### **Core Concept**
Keep current architecture but add timing-aware loss functions that enforce the 600ms constraint.

#### **Key Components**
1. **Cumulative Timing Loss**: Ensure predicted times sum to ≤ 600ms
2. **Sequence Length Consistency Loss**: Penalize when predicted length doesn't match timing-derived length
3. **Temporal Coherence Loss**: Ensure actions are temporally consistent

```python
class TimingAwareLoss(nn.Module):
    def __init__(self, max_time=0.6):
        super().__init__()
        self.max_time = max_time
    
    def forward(self, predictions, targets, valid_mask):
        # Extract timing predictions
        time_predictions = predictions['time_q']  # [B, A, 3]
        median_times = time_predictions[:, :, 1]  # [B, A] - q0.5
        
        # Apply valid mask
        valid_times = median_times * valid_mask.float()
        
        # Cumulative timing loss
        cumulative_times = torch.cumsum(valid_times, dim=1)  # [B, A]
        exceed_mask = cumulative_times > self.max_time  # [B, A]
        
        # Penalize exceeding 600ms
        exceed_penalty = torch.sum((cumulative_times - self.max_time) * exceed_mask.float())
        
        # Sequence length consistency
        predicted_lengths = predictions['sequence_length']  # [B, 1]
        timing_derived_lengths = (cumulative_times <= self.max_time).sum(dim=1, keepdim=True).float()
        length_consistency_loss = F.mse_loss(predicted_lengths, timing_derived_lengths)
        
        return exceed_penalty + length_consistency_loss
```

## **Temporal Data Utilization Assessment**

### **✅ What's Already Working Well**

Your current model **IS** properly incorporating the temporal data:

1. **Action History Processing**: 
   - Processes all 10 timesteps of action sequences (6 seconds total)
   - Uses LSTM to capture temporal patterns within each 600ms window
   - Summarizes each timestep using the last action's representation

2. **Gamestate History Processing**:
   - Processes all 10 timesteps of gamestate features (128 features each)
   - Uses bidirectional LSTM to capture forward/backward temporal dependencies
   - Combines both directions for rich temporal understanding

3. **Multi-Modal Fusion**:
   - Combines current gamestate + action history + gamestate history
   - Creates rich 768-dimensional context (256 + 256 + 256)
   - Provides comprehensive understanding of game state and player behavior

### **❌ What Needs Fixing (The Core Issues)**

The problem is **NOT** with temporal data processing - that's working well. The issues are:

1. **Timing Prediction Logic**: Independent deltas instead of cumulative timing
2. **Sequence Length Logic**: Predicted separately instead of derived from timing
3. **Action Generation Logic**: Parallel instead of sequential generation

### **🎯 The Fix Strategy**

**Keep the excellent temporal processing** and only fix the action generation logic:

```python
# CURRENT (GOOD): Temporal processing
fused_features = torch.cat([
    gamestate_encoded,        # (B, 256) - current gamestate
    processed_actions,        # (B, 256) - 10 timesteps of action history  
    temporal_encoded          # (B, 256) - 10 timesteps of gamestate history
], dim=-1)  # (B, 768)

# CURRENT (GOOD): Rich context creation
fused_output = self.fusion_layer(fused_features)  # (B, 256)

# FIX (BAD): Action generation logic
# OLD: Parallel generation with independent timing
# NEW: Sequential generation with cumulative timing
return self.sequential_action_decoder(fused_output, action_sequence, valid_mask)
```

## **Recommended Implementation Strategy**

### **Phase 1: Fix Current Architecture (Quick Wins)**
1. **Add Timing-Aware Loss**: Implement cumulative timing constraints
2. **Fix Sequence Length**: Derive from timing instead of predicting separately
3. **Improve Timing Predictions**: Use cumulative timing instead of independent deltas

### **Phase 2: Sequential Generation (Medium Term)**
1. **Implement SequentialActionDecoder**: Generate actions one at a time
2. **Add Temporal Causality**: Each action depends on previous actions
3. **Natural Sequence Length**: Emerges from timing constraints

### **Phase 3: Advanced Architecture (Long Term)**
1. **Attention-Based Generation**: Use transformer-style attention
2. **Multi-Scale Timing**: Handle both fast actions (< 100ms) and slow actions (> 100ms)
3. **Context-Aware Generation**: Actions depend on game state and history

## **Expected Benefits**

1. **Realistic Timing**: Actions will have proper cumulative timing
2. **Natural Sequence Length**: No more predicting 100 actions when only 5 are needed
3. **Temporal Coherence**: Actions will be temporally consistent
4. **Better Learning**: Model will learn the true relationship between timing and sequence length
5. **Improved Performance**: More realistic action sequences will lead to better bot behavior

## **Implementation Priority**

1. **High Priority**: Fix timing predictions to be cumulative
2. **High Priority**: Add timing-aware loss functions
3. **Medium Priority**: Implement sequential generation
4. **Low Priority**: Advanced attention-based architecture

This analysis shows that the current architecture has fundamental flaws in how it handles timing and sequence length. The proposed solutions address these issues by making timing cumulative, sequential, and naturally constraining sequence length.
