# OSRS Imitation Learning Model Plan
## Professional Approach to Creating an AI Bot that Plays Like You

**Version:** 1.0  
**Date:** December 8, 2025  
**Project:** OSRS Learner - Sapphire Ring Crafting Bot  
**Author:** AI Assistant  

---

## Executive Summary

This document outlines a comprehensive 6-week plan to create an imitation learning model that can play Old School RuneScape (OSRS) by learning from your gameplay data. The model will interpret gamestates, screenshots, and action sequences to predict the next appropriate action, effectively mimicking your playing style.

**Goal:** Create a RuneScape bot that plays exactly like you, with natural mouse movements, appropriate clicks, and intelligent decision-making based on game context.

---

## Current Data Analysis

### Data Assets Available
- **216 gamestate samples** collected at 0.6-second intervals
- **73 features per gamestate** including:
  - Player state (position, animation, movement)
  - Camera information (position, pitch, yaw)
  - Inventory contents (28 slots)
  - Bank state and materials
  - Game objects and NPCs
  - UI tabs and skills
  - Enhanced interaction tracking
- **3,450 action records** capturing:
  - Mouse movements with timestamps
  - Left/right clicks with coordinates
  - Key presses and releases
  - Scroll actions
  - Modifier key states
- **216 screenshots** (one per gamestate) for visual context
- **0.6-second temporal resolution** for real-time gameplay

### Data Quality Assessment
- **Temporal Alignment:** Gamestates and screenshots are perfectly synchronized
- **Feature Richness:** Comprehensive coverage of game state (73 features)
- **Action Granularity:** High-resolution action tracking (millisecond precision)
- **Sample Size:** 216 samples provide sufficient training data for initial model

---

## Phase 1: Data Preparation & Alignment ✅ COMPLETED

### 1.1 Temporal Alignment & Sequence Creation

#### Sequence Structure
```python
# Create sliding window sequences for temporal learning
sequence_length = 10  # 6 seconds of context (10 * 0.6s)
stride = 1           # 0.6s step between sequences

# Each sequence: [gamestate_0, gamestate_1, ..., gamestate_9] -> action_9
# This captures the temporal context leading up to each action
```

#### Implementation Details ✅ IMPLEMENTED
- **Input Sequence:** 10 consecutive gamestates (6 seconds of gameplay context)
- **Target:** Next 600ms of actions (multiple actions with timing, coordinates, types)
- **Overlap:** 90% overlap between sequences for dense temporal learning
- **Total Sequences:** 206 training sequences (216 - 10 + 1)
- **Action Vector:** 106-dimensional vector encoding up to 15 actions per window

### 1.2 Action Vectorization

#### Action Space Definition ✅ IMPLEMENTED
```python
# Multi-action sequence vectorization (106 dimensions)
action_vector = [
    action_count,                    # How many actions in next 600ms
    timing_1, type_1, x_1, y_1, click_1, key_1, scroll_1,  # Action 1
    timing_2, type_2, x_2, y_2, click_2, key_2, scroll_2,  # Action 2
    ...                             # Up to 15 actions
]
# Total: 106-dimensional action space (1 + 15 * 7)
```

#### Action Processing
- **Mouse Position:** Normalize to [0, 1] range for consistent training
- **Click Types:** One-hot encode left/right/middle clicks
- **Key Presses:** Map to 50 most common keys + special keys
- **Scroll Actions:** Normalize scroll deltas to reasonable ranges
- **Confidence:** Model's confidence in the predicted action

### 1.3 Multi-Modal Data Fusion

#### Input Channel Architecture
```python
# Combine gamestate features + screenshots + actions
input_channels = {
    'gamestate_features': (73,),        # 1D feature vector
    'screenshot_features': (224, 224, 3), # 2D CNN features
    'temporal_context': (10, 73),      # 2D sequence features
}
```

#### Data Synchronization
- **Timestamp Matching:** Align gamestates, screenshots, and actions by timestamp
- **Interpolation:** Handle any missing data points with linear interpolation
- **Validation:** Ensure all modalities are properly synchronized

---

## Phase 2: Model Architecture Design (Week 2-3)

### 2.1 Hybrid Architecture: Transformer + CNN + LSTM

#### Core Architecture Components
```python
class ImitationHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Gamestate Feature Encoder (73 -> 256)
        self.gamestate_encoder = nn.Sequential(
            nn.Linear(73, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # 2. Screenshot Feature Encoder (CNN)
        self.screenshot_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 112x112
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 56x56
            nn.Conv2d(64, 128, 5, stride=2, padding=2), # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 14x14
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),               # 1x1
            nn.Flatten()                                # 256
        )
        
        # 3. Temporal Context Encoder (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=73,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # 4. Multi-Modal Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 5. Action Decoder (Multi-Head)
        self.mouse_position_head = nn.Linear(256, 2)      # (x, y)
        self.mouse_click_head = nn.Linear(256, 2)         # (left, right)
        self.key_press_head = nn.Linear(256, 50)          # 50 most common keys
        self.scroll_head = nn.Linear(256, 2)              # (dx, dy)
        self.confidence_head = nn.Linear(256, 1)          # action confidence
```

### 2.2 Attention Mechanism for Game Context

#### Self-Attention Implementation
```python
# Self-attention over gamestate features to focus on relevant information
self.gamestate_attention = nn.MultiheadAttention(
    embed_dim=256,
    num_heads=8,
    dropout=0.1
)

# Cross-attention between gamestate and screenshot features
self.cross_attention = nn.MultiheadAttention(
    embed_dim=256,
    num_heads=8,
    dropout=0.1
)
```

#### Attention Benefits
- **Feature Selection:** Automatically focus on relevant gamestate features
- **Visual-Game Alignment:** Connect visual elements with game state
- **Context Understanding:** Learn which features matter for different actions

### 2.3 Model Complexity & Parameters

#### Parameter Count Estimation
- **Gamestate Encoder:** ~25K parameters
- **Screenshot Encoder:** ~500K parameters
- **Temporal Encoder:** ~100K parameters
- **Fusion & Decoder:** ~200K parameters
- **Total:** ~825K parameters (manageable for real-time inference)

---

## Phase 3: Training Strategy (Week 3-4)

### 3.1 Loss Functions

#### Multi-Objective Loss Design
```python
class ImitationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Mouse position: MSE loss
        mouse_pos_loss = F.mse_loss(pred['mouse_position'], target['mouse_position'])
        
        # Mouse clicks: Binary cross-entropy
        click_loss = F.binary_cross_entropy_with_logits(
            pred['mouse_click'], target['mouse_click']
        )
        
        # Key presses: Cross-entropy
        key_loss = F.cross_entropy(pred['key_press'], target['key_press'])
        
        # Scroll: MSE loss
        scroll_loss = F.mse_loss(pred['scroll'], target['scroll'])
        
        # Confidence: MSE loss
        confidence_loss = F.mse_loss(pred['confidence'], target['confidence'])
        
        # Weighted combination based on importance
        total_loss = (
            2.0 * mouse_pos_loss +      # Most important for gameplay
            1.5 * click_loss +          # Critical for interaction
            1.0 * key_loss +            # Important for efficiency
            0.5 * scroll_loss +         # Less critical
            0.3 * confidence_loss       # Training stability
        )
        
        return total_loss
```

#### Loss Weighting Rationale
- **Mouse Position (2.0x):** Most critical for accurate gameplay
- **Click Actions (1.5x):** Essential for game interaction
- **Key Presses (1.0x):** Important for efficiency and shortcuts
- **Scroll Actions (0.5x):** Less critical for core gameplay
- **Confidence (0.3x):** Helps with training stability

### 3.2 Training Loop with Curriculum Learning

#### Phase-Based Training Strategy
```python
def train_with_curriculum(model, dataloader, epochs=100):
    # Phase 1: Train on simple actions (mouse movements only)
    for epoch in range(20):
        train_phase1(model, dataloader, action_types=['mouse_move'])
        print(f"Phase 1, Epoch {epoch+1}/20: Mouse movement training")
    
    # Phase 2: Add clicks and basic interactions
    for epoch in range(30):
        train_phase2(model, dataloader, action_types=['mouse_move', 'click'])
        print(f"Phase 2, Epoch {epoch+1}/30: Adding click training")
    
    # Phase 3: Full action space training
    for epoch in range(50):
        train_phase3(model, dataloader, action_types=['all'])
        print(f"Phase 3, Epoch {epoch+1}/50: Full action training")
```

#### Curriculum Benefits
- **Progressive Complexity:** Start simple, build up to complex actions
- **Stable Training:** Avoid overwhelming the model initially
- **Better Convergence:** Each phase builds on previous learning
- **Faster Training:** Early phases converge quickly

### 3.3 Training Hyperparameters

#### Optimizer Configuration
```python
# Adam optimizer with learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
```

#### Training Parameters
- **Batch Size:** 16 (balanced between memory and stability)
- **Learning Rate:** 0.001 (Adam default, good for this task)
- **Weight Decay:** 1e-4 (moderate regularization)
- **Dropout:** 0.2-0.3 (prevent overfitting)
- **Patience:** 5 epochs (early stopping threshold)

---

## Phase 4: Data Augmentation & Regularization (Week 4-5)

### 4.1 Gamestate Augmentation

#### Feature Perturbation
```python
def augment_gamestate(gamestate):
    # Add noise to coordinates (±2 pixels)
    if 'player' in gamestate:
        gamestate['player']['world_x'] += np.random.normal(0, 2)
        gamestate['player']['world_y'] += np.random.normal(0, 2)
    
    # Vary camera angles slightly (±5 degrees)
    gamestate['camera_pitch'] += np.random.normal(0, 5)
    gamestate['camera_yaw'] += np.random.normal(0, 5)
    
    # Add noise to inventory quantities (±1)
    if 'inventory' in gamestate:
        for item in gamestate['inventory']:
            if item.get('quantity', 0) > 0:
                item['quantity'] = max(1, item['quantity'] + np.random.randint(-1, 2))
    
    return gamestate
```

#### Augmentation Rationale
- **Coordinate Noise:** Simulates slight variations in player positioning
- **Camera Variation:** Accounts for different viewing angles
- **Inventory Noise:** Handles slight variations in item counts
- **Realistic Variation:** Maintains game logic while adding diversity

### 4.2 Screenshot Augmentation

#### Visual Perturbation
```python
def augment_screenshot(image):
    # Color jitter (brightness, contrast, saturation)
    image = transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.1,
        hue=0.05
    )(image)
    
    # Slight rotation (±5 degrees)
    angle = np.random.uniform(-5, 5)
    image = transforms.functional.rotate(image, angle, fill=0)
    
    # Random crop and resize (maintains aspect ratio)
    image = transforms.RandomResizedCrop(
        224, 
        scale=(0.9, 1.0), 
        ratio=(0.9, 1.1)
    )(image)
    
    # Random horizontal flip (50% probability)
    if np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
    
    return image
```

#### Visual Augmentation Benefits
- **Color Robustness:** Handles different lighting conditions
- **Rotation Tolerance:** Accounts for slight camera tilts
- **Crop Variation:** Simulates different viewing windows
- **Flip Augmentation:** Increases training data diversity

### 4.3 Regularization Techniques

#### Additional Regularization
```python
# Label smoothing for classification tasks
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixup data augmentation
def mixup_data(x1, x2, y1, y2, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    
    return mixed_x, mixed_y

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Phase 5: Evaluation & Deployment (Week 5-6)

### 5.1 Action Prediction Metrics

#### Comprehensive Evaluation Framework
```python
def evaluate_model(model, test_loader):
    metrics = {
        'mouse_position_accuracy': [],    # Within 5 pixels
        'click_accuracy': [],            # Correct click type
        'key_accuracy': [],              # Correct key pressed
        'action_sequence_accuracy': [],  # Correct action sequence
        'temporal_consistency': [],      # Actions make sense in sequence
        'game_context_accuracy': []      # Actions appropriate for game state
    }
    
    # Evaluate on test set
    for batch in test_loader:
        predictions = model(batch['gamestate'], batch['screenshot'])
        
        # Calculate individual metrics
        mouse_acc = calculate_mouse_accuracy(predictions, batch['target'])
        click_acc = calculate_click_accuracy(predictions, batch['target'])
        key_acc = calculate_key_accuracy(predictions, batch['target'])
        
        # Store metrics
        metrics['mouse_position_accuracy'].append(mouse_acc)
        metrics['click_accuracy'].append(click_acc)
        metrics['key_accuracy'].append(key_acc)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

#### Metric Definitions
- **Mouse Position Accuracy:** Percentage of predictions within 5 pixels of target
- **Click Accuracy:** Percentage of correct click type predictions
- **Key Accuracy:** Percentage of correct key press predictions
- **Action Sequence Accuracy:** Percentage of correct action sequences
- **Temporal Consistency:** Actions that make logical sense in sequence
- **Game Context Accuracy:** Actions appropriate for current game state

### 5.2 Real-Time Inference Pipeline

#### Production-Ready Bot Implementation
```python
class RealTimeBot:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.action_history = []
        self.gamestate_buffer = []
        self.last_action_time = time.time()
        
    def predict_next_action(self, current_gamestate, screenshot):
        # Update gamestate buffer
        self.gamestate_buffer.append(current_gamestate)
        if len(self.gamestate_buffer) > 10:
            self.gamestate_buffer.pop(0)
        
        # Prepare input tensors
        gamestate_features = self.extract_features(current_gamestate)
        screenshot_features = self.preprocess_screenshot(screenshot)
        temporal_context = np.array(self.gamestate_buffer)
        
        # Predict action
        with torch.no_grad():
            action = self.model(gamestate_features, screenshot_features, temporal_context)
        
        # Post-process action
        processed_action = self.post_process_action(action)
        
        # Validate action
        is_valid, validation_msg = self.validate_action(processed_action, current_gamestate)
        
        if not is_valid:
            # Generate fallback action
            processed_action = self.generate_fallback_action(current_gamestate)
        
        return processed_action
    
    def post_process_action(self, raw_action):
        # Convert model outputs to executable actions
        mouse_pos = raw_action['mouse_position'].cpu().numpy()
        click_type = torch.sigmoid(raw_action['mouse_click']).cpu().numpy()
        key_press = torch.softmax(raw_action['key_press'], dim=-1).cpu().numpy()
        
        return {
            'mouse_position': mouse_pos,
            'click_type': click_type,
            'key_press': key_press,
            'confidence': raw_action['confidence'].cpu().numpy()
        }
```

#### Performance Optimization
- **Model Quantization:** Convert to INT8 for faster inference
- **Batch Processing:** Process multiple gamestates simultaneously
- **GPU Acceleration:** Utilize CUDA for real-time performance
- **Memory Management:** Efficient tensor handling and cleanup

---

## Phase 6: Safety & Validation (Week 6)

### 6.1 Action Validation

#### Comprehensive Safety Checks
```python
def validate_action(action, gamestate):
    # Check if action is within reasonable bounds
    if action['mouse_position'][0] < 0 or action['mouse_position'][0] > 800:
        return False, "Mouse X out of bounds"
    
    if action['mouse_position'][1] < 0 or action['mouse_position'][1] > 600:
        return False, "Mouse Y out of bounds"
    
    # Check if action makes sense given current game state
    if gamestate.get('bank_open', False) and action['click_type'][0] > 0.8:
        # High confidence left click when bank is open - validate target
        return validate_bank_interaction(action, gamestate)
    
    # Check for dangerous actions (e.g., dropping valuable items)
    if is_dangerous_action(action, gamestate):
        return False, "Potentially dangerous action detected"
    
    # Validate temporal consistency
    if not is_temporally_consistent(action, gamestate):
        return False, "Action lacks temporal consistency"
    
    return True, "Action validated"
```

#### Validation Categories
- **Boundary Checks:** Ensure actions are within game window
- **Context Validation:** Verify actions make sense for current game state
- **Safety Checks:** Prevent dangerous or destructive actions
- **Temporal Consistency:** Ensure actions follow logical sequences
- **Game Logic Validation:** Verify actions follow OSRS game rules

### 6.2 Fallback Mechanisms

#### Intelligent Fallback System
```python
def safe_action_execution(action, gamestate):
    # Try predicted action first
    success = execute_action(action)
    
    if not success:
        # Generate rule-based fallback action
        fallback_action = generate_fallback_action(gamestate)
        success = execute_action(fallback_action)
        
        if not success:
            # Emergency fallback: safe default action
            emergency_action = generate_emergency_action(gamestate)
            success = execute_action(emergency_action)
    
    return success

def generate_fallback_action(gamestate):
    # Rule-based action generation based on game state
    if gamestate.get('bank_open', False):
        return generate_banking_action(gamestate)
    elif gamestate.get('inventory_full', False):
        return generate_banking_action(gamestate)
    else:
        return generate_crafting_action(gamestate)

def generate_emergency_action(gamestate):
    # Safe default action when all else fails
    return {
        'mouse_position': [400, 300],  # Center of screen
        'click_type': [0, 0],          # No click
        'key_press': [0] * 50,         # No key press
        'confidence': 0.0               # No confidence
    }
```

#### Fallback Hierarchy
1. **Predicted Action:** Model's primary prediction
2. **Rule-Based Fallback:** Logic-based action generation
3. **Emergency Fallback:** Safe default actions
4. **Complete Stop:** Halt bot if all fallbacks fail

---

## Implementation Timeline

### Week-by-Week Breakdown

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1-2 | Data Preparation | Data alignment, sequence creation, action vectorization | Aligned training dataset, data loaders |
| 2-3 | Model Architecture | Design and implement hybrid model | Complete model architecture, training pipeline |
| 3-4 | Training Strategy | Implement loss functions, curriculum learning | Trained model, training metrics |
| 4-5 | Augmentation & Regularization | Data augmentation, regularization techniques | Augmented dataset, regularized model |
| 5-6 | Evaluation & Deployment | Model evaluation, real-time pipeline | Performance metrics, production-ready bot |
| 6 | Safety & Validation | Action validation, fallback mechanisms | Safe bot implementation, validation framework |

### Critical Milestones
- **Week 2:** Complete data pipeline and model architecture
- **Week 4:** First working model with basic training
- **Week 6:** Production-ready bot with safety measures

---

## Expected Outcomes

### Performance Metrics
1. **Action Prediction Accuracy:** 
   - Mouse movements: 85-90%
   - Click actions: 80-85%
   - Key presses: 75-80%
   - Overall accuracy: 80-85%

2. **Temporal Consistency:** 
   - Actions that make logical sense in sequence: 90%+
   - Proper action timing: 85%+

3. **Real-Time Performance:** 
   - Inference time: <100ms
   - Memory usage: <2GB
   - CPU usage: <30%

4. **Safety & Validation:** 
   - Action validation rate: 99%+
   - False positive rate: <1%
   - Fallback success rate: 95%+

### Success Criteria
- **Gameplay Quality:** Bot performs actions indistinguishable from human player
- **Efficiency:** Bot maintains or improves upon human efficiency
- **Safety:** Zero destructive or dangerous actions
- **Reliability:** 99%+ uptime with graceful error handling

---

## Risk Assessment & Mitigation

### Technical Risks

#### Risk: Insufficient Training Data
- **Probability:** Medium
- **Impact:** High
- **Mitigation:** Data augmentation, transfer learning, active learning

#### Risk: Model Overfitting
- **Probability:** High
- **Impact:** Medium
- **Mitigation:** Regularization, early stopping, cross-validation

#### Risk: Real-Time Performance Issues
- **Probability:** Medium
- **Impact:** High
- **Mitigation:** Model optimization, GPU acceleration, efficient inference

### Game-Specific Risks

#### Risk: Game Rule Violations
- **Probability:** Low
- **Impact:** High
- **Mitigation:** Comprehensive validation, rule-based fallbacks

#### Risk: Detection by Anti-Bot Systems
- **Probability:** Medium
- **Impact:** High
- **Mitigation:** Human-like behavior patterns, randomized delays

---

## Future Enhancements

### Advanced Features
1. **Multi-Task Learning:** Learn multiple skills simultaneously
2. **Meta-Learning:** Adapt to new game scenarios quickly
3. **Reinforcement Learning:** Optimize actions based on rewards
4. **Human-in-the-Loop:** Continuous learning from human feedback

### Scalability Improvements
1. **Distributed Training:** Scale to larger datasets
2. **Model Compression:** Reduce model size for deployment
3. **Edge Computing:** Deploy on resource-constrained devices
4. **Cloud Integration:** Centralized model updates and monitoring

---

## Conclusion

This comprehensive plan provides a roadmap for creating a sophisticated imitation learning model that can play RuneScape just like you. By leveraging your rich multimodal data and implementing state-of-the-art deep learning techniques, we can create a bot that:

- **Understands Game Context:** Interprets gamestates, screenshots, and temporal sequences
- **Makes Intelligent Decisions:** Predicts appropriate actions based on learned patterns
- **Maintains Safety:** Validates all actions and provides fallback mechanisms
- **Performs Naturally:** Mimics human-like behavior and timing

The 6-week timeline ensures systematic development with clear milestones, while the modular architecture allows for iterative improvements and future enhancements.

**Next Steps:** Begin with Phase 1 (Data Preparation) and establish the foundational data pipeline that will support the entire model development process.

---

## Appendices

### Appendix A: Technical Specifications
- Detailed model architecture diagrams
- Training hyperparameter configurations
- Data preprocessing pipelines

### Appendix B: Evaluation Metrics
- Comprehensive evaluation criteria
- Benchmark datasets and comparisons
- Performance analysis tools

### Appendix C: Deployment Guide
- Production deployment checklist
- Monitoring and logging setup
- Troubleshooting guide

---

**Document Version Control**
- **v1.0** (2025-12-08): Initial comprehensive plan
- **Future Updates:** Will be versioned as implementation progresses
