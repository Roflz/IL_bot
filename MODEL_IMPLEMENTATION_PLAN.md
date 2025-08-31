# OSRS Imitation Learning Model Implementation Plan

## Overview

This document outlines the plan to implement a new single-sequence model architecture for OSRS imitation learning. The model will output exactly one event type per timestep with associated parameters, replacing the current multi-head approach with a unified, exclusive event system.

## Current State Analysis

### What We Have ✅

1. **Infrastructure**: Complete data pipeline with gamestate sequences, action sequences, and training infrastructure
2. **Data Format**: V2 targets with proper enum sizes and vocabularies
3. **Model Framework**: Hybrid model with gamestate encoder, action sequence encoder, and temporal encoder
4. **Training Loop**: Functional training with loss computation and validation
5. **Configuration**: Event class weights and basic settings

### What's Missing ❌

1. **Event Head**: No unified event classification head for [CLICK, KEY, SCROLL, MOVE]
2. **Quantile Time Head**: Current time head is simple regression, not quantile-based
3. **Heteroscedastic XY**: No uncertainty modeling for cursor positions
4. **Unified Loss**: Current loss is per-head, not a single unified objective
5. **Event Gating**: No conditional computation of event-specific details

## Target Architecture

### Model Outputs (Per Timestep)

```
event_logits: [B, A, 4] → ["CLICK", "KEY", "SCROLL", "MOVE"]
time_q: [B, A, 3] → [q10, q50, q90] quantiles of Δt
x_mu, x_logsig: [B, A] → cursor X mean + log(std)
y_mu, y_logsig: [B, A] → cursor Y mean + log(std)

Conditional outputs (only when event wins):
button_logits: [B, A, 4] → when event == CLICK
key_action_logits: [B, A, 3] → when event == KEY  
key_id_logits: [B, A, vocab_size] → when event == KEY
scroll_y_logits: [B, A, 3] → when event == SCROLL
```

### Key Design Principles

1. **Exclusive Events**: Only one event type per timestep (no MULTI)
2. **Quantile Time**: Robust time prediction with uncertainty bands
3. **Heteroscedastic XY**: Model can express confidence in cursor predictions
4. **Conditional Computation**: Event-specific heads only computed when relevant
5. **Unified Loss**: Single loss function combining all objectives

## Implementation Steps

### Phase 1: Model Architecture Updates

#### 1.1 Update ActionDecoder Class
- **File**: `ilbot/model/imitation_hybrid_model.py`
- **Changes**:
  - Add `event_head` for 4-class classification
  - Replace time regression with `time_quantile_head` (3 outputs)
  - Add `x_mu_head`, `x_logsig_head`, `y_mu_head`, `y_logsig_head`
  - Keep existing button/key/scroll heads for conditional output
  - Update forward pass to return new head structure

#### 1.2 Update Model Forward Pass
- **File**: `ilbot/model/imitation_hybrid_model.py`
- **Changes**:
  - Modify `forward()` method to return new head structure
  - Ensure `return_logits=True` returns all heads
  - Update `decode_legacy()` to handle new structure

### Phase 2: Loss Function Implementation

#### 2.1 Create New Unified Loss Class
- **File**: `ilbot/model/losses.py`
- **New Class**: `UnifiedEventLoss`
- **Components**:
  - Event CE with class weights: `[1.0, 8.0, 6.0, 12.0]`
  - Time pinball loss on quantiles
  - XY Gaussian NLL (heteroscedastic)
  - Masked auxiliary CE losses

#### 2.2 Implement Loss Components
```python
def compute_event_loss(self, event_logits, event_targets, class_weights):
    """Event classification with class weighting"""
    
def compute_time_loss(self, time_q, time_targets, valid_mask):
    """Pinball loss on quantiles [0.1, 0.5, 0.9]"""
    
def compute_xy_loss(self, x_mu, x_logsig, y_mu, y_logsig, x_target, y_target, valid_mask):
    """Gaussian NLL for heteroscedastic XY prediction"""
    
def compute_auxiliary_losses(self, heads, targets, event_targets, valid_mask):
    """Masked CE losses for event-specific heads"""
```

### Phase 3: Training Loop Updates

#### 3.1 Update Training Loss Computation
- **File**: `ilbot/training/train_loop.py`
- **Changes**:
  - Replace `compute_il_loss_v2()` with new unified loss
  - Update validation metrics to use event-based masking
  - Add quantile time metrics and XY uncertainty reporting

#### 3.2 Update Validation Metrics
- **File**: `ilbot/eval/eval.py`
- **Changes**:
  - Event confusion matrix (4x4)
  - Time quantile statistics (q10, q50, q90)
  - XY MAE + uncertainty statistics
  - Per-event head accuracies with proper masking

### Phase 4: Configuration & Data Handling

#### 4.1 Update Configuration
- **File**: `ilbot/config.py`
- **New Settings**:
  - Loss weights for all components
  - Time quantile configuration
  - XY uncertainty bounds
  - Event class weights (already exists)

#### 4.2 Data Target Updates
- **Current**: V2 targets with 7 features
- **New**: Need to derive event targets from existing data
- **Process**: 
  - CLICK: when button != 0
  - KEY: when key_action != 0  
  - SCROLL: when scroll_y != 0
  - MOVE: else case

### Phase 5: Inference & Sampling

#### 5.1 Event Selection
```python
def sample_event(event_logits, temperature=1.0):
    """Sample event type with optional temperature"""
    
def sample_time(time_q):
    """Sample time from quantile-based distribution"""
    
def sample_xy(x_mu, x_logsig, y_mu, y_logsig):
    """Sample cursor position with uncertainty"""
```

#### 5.2 Safety Layers
- Cooldown management per event type
- Bounds checking for cursor positions
- Time floor enforcement
- Key press/release consistency

## Data Requirements

### Current Data Structure
- **Gamestate sequences**: [B, 10, 128] features
- **Action sequences**: [B, 10, 100, 8] features  
- **Targets**: [B, 100, 7] V2 format

### Required Changes
1. **Event Target Derivation**: Create event classification targets
2. **Time Delta Calculation**: Compute Δt between consecutive actions
3. **Valid Mask Updates**: Ensure proper masking for new head structure

## Metrics & Validation

### Event Metrics
- Confusion matrix (4x4)
- Per-event accuracy
- Class distribution analysis
- Zero MULTI enforcement

### Time Metrics  
- Quantile statistics (q10, q50, q90)
- MAE on median prediction
- Distribution plots
- Residual analysis

### XY Metrics
- MAE on mean predictions
- Uncertainty calibration
- Residual distributions
- Bounds compliance

## Testing Strategy

### Unit Tests
1. **Loss Components**: Test each loss function independently
2. **Model Forward Pass**: Verify output shapes and types
3. **Event Gating**: Test conditional computation logic

### Integration Tests
1. **Training Loop**: End-to-end training with new loss
2. **Validation**: Metrics computation and reporting
3. **Inference**: Sampling and safety layer behavior

### Data Validation
1. **Target Derivation**: Verify event target creation
2. **Masking**: Ensure proper loss masking
3. **Vocabulary**: Confirm key ID mapping

## Migration Plan

### Phase 1: Parallel Implementation
- Keep existing model functional
- Implement new architecture alongside
- Share data loading and training infrastructure

### Phase 2: Gradual Replacement
- Test new model on subset of data
- Compare performance metrics
- Validate inference behavior

### Phase 3: Full Deployment
- Replace training pipeline
- Update evaluation scripts
- Deploy new inference code

## Success Criteria

### Functional Requirements
- [ ] Model outputs exactly 4 event types (no MULTI)
- [ ] Time predictions use quantile approach
- [ ] XY predictions include uncertainty
- [ ] Event-specific heads properly gated
- [ ] Single unified loss function

### Performance Requirements
- [ ] Event accuracy > 80% on validation
- [ ] Time MAE < 50ms on median
- [ ] XY MAE < 200px on mean
- [ ] Training stability maintained
- [ ] Inference latency < 100ms

### Quality Requirements
- [ ] No legacy code paths
- [ ] Comprehensive error handling
- [ ] Clear documentation
- [ ] Unit test coverage > 90%
- [ ] Performance benchmarks

## Timeline Estimate

- **Phase 1 (Architecture)**: 1-2 weeks
- **Phase 2 (Loss)**: 1 week  
- **Phase 3 (Training)**: 1 week
- **Phase 4 (Config)**: 0.5 weeks
- **Phase 5 (Inference)**: 1 week
- **Testing & Validation**: 1-2 weeks

**Total**: 5-7 weeks for complete implementation

## Risk Assessment

### Technical Risks
1. **Loss Stability**: New loss function may cause training instability
2. **Event Gating**: Conditional computation complexity
3. **Performance**: Additional heads may impact inference speed

### Mitigation Strategies
1. **Gradual Rollout**: Implement and test incrementally
2. **Extensive Testing**: Comprehensive validation before deployment
3. **Performance Monitoring**: Track metrics throughout development
4. **Rollback Plan**: Keep existing implementation as backup

## Next Steps

1. **Immediate**: Review and approve this plan
2. **Week 1**: Begin Phase 1 (Model Architecture Updates)
3. **Week 2**: Implement Phase 2 (Loss Function)
4. **Week 3**: Update training loop and validation
5. **Week 4**: Configuration and data handling
6. **Week 5**: Inference implementation and testing
7. **Week 6**: Integration testing and validation
8. **Week 7**: Deployment and monitoring

## Conclusion

This plan provides a comprehensive roadmap for implementing the new model architecture. The approach maintains backward compatibility while introducing the unified event system, quantile time prediction, and heteroscedastic XY modeling. The phased implementation allows for incremental validation and reduces risk.

The new architecture will provide more interpretable outputs, better uncertainty quantification, and cleaner event classification, ultimately leading to more human-like bot behavior.

