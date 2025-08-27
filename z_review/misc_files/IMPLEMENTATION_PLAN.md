# Implementation Plan: Hybrid Data Extraction & Feature Creation for IL Training

## **Project Overview**
This document outlines the implementation plan for creating a hybrid imitation learning system that combines rules-based phase detection with learning-based action generation for OSRS sapphire ring crafting automation.

## **Current System State**
- **Feature Extraction**: 90 features (80 original + 10 action features)
- **Data Collection**: RuneLite plugin with gamestate and actions.csv
- **Feature Interpretation**: Comprehensive mapping and analysis tools
- **Action Features**: Currently too generic (single most recent action)

## **Target System**
- **Hybrid Approach**: Rules-based phase detection + Learning-based action generation
- **Comprehensive Input State**: Capture ALL nearby actions (±300ms) with timestamps
- **Phase Context**: Automatic phase identification and timing
- **Behavioral Learning**: Model learns to replicate exact mouse/keyboard patterns

---

## **Phase 1: Enhanced Data Collection (Week 1)**

### **1.1 Modify StateExporterPlugin.java**

#### **Add Phase Detection Logic**
```java
// Add these fields to the plugin class
private String currentPhase = "unknown";
private long phaseStartTime = 0;
private long sessionStartTime = 0;
private int completedCycles = 0;

// Add phase detection method
private String detectCurrentPhase() {
    // Check bank status
    boolean bankOpen = isBankOpen();
    
    // Check crafting interface
    boolean craftingOpen = isCraftingInterfaceOpen();
    
    // Check inventory contents
    boolean hasMaterials = hasCraftingMaterials();
    
    if (bankOpen) {
        return "banking";
    } else if (craftingOpen && hasMaterials) {
        return "crafting";
    } else if (!bankOpen && !craftingOpen && hasMaterials) {
        return "moving_to_furnace";
    } else if (!bankOpen && !craftingOpen && !hasMaterials) {
        return "moving_to_bank";
    }
    
    return "unknown";
}
```

#### **Add Phase Context to Gamestates**
```java
// Add to each gamestate JSON in onGameTick()
Map<String, Object> phaseContext = new HashMap<>();
phaseContext.put("cycle_phase", currentPhase);
phaseContext.put("phase_progress", calculatePhaseProgress());
phaseContext.put("session_start_time", sessionStartTime);
phaseContext.put("completed_cycles", completedCycles);
phaseContext.put("phase_duration", System.currentTimeMillis() - phaseStartTime);

state.put("phase_context", phaseContext);
```

#### **Enhanced Action Recording**
```java
// Modify actions.csv structure to include:
// timestamp, event_type, x_in_window, y_in_window, btn, key, scroll_dx, scroll_dy, modifiers, active_keys, phase, relative_timestamp, action_sequence_id

// Add phase column to each action
action.put("phase", currentPhase);
action.put("relative_timestamp", System.currentTimeMillis() - sessionStartTime);
action.put("action_sequence_id", generateSequenceId());
```

### **1.2 Phase Detection Rules**
```java
// Banking Phase
// - Trigger: bank_open = true
// - End: bank_open = false

// Moving to Furnace Phase  
// - Trigger: bank closes (bank_open = false)
// - End: crafting interface opens OR furnace interaction

// Crafting Phase
// - Trigger: crafting interface opens
// - End: no more materials in inventory (gold bars or sapphires)

// Moving to Bank Phase
// - Trigger: crafting ends (no materials)
// - End: bank opens (bank_open = true)
```

---

## **Phase 2: Comprehensive Feature Extraction (Week 2)**

### **2.1 Replace Current Action Features (80-89)**

#### **New Feature Structure**
```python
# Feature 80: Complete Input State Hash
# Combines ALL nearby actions (±300ms) with timestamps
# Format: "action1|timestamp1|action2|timestamp2|..."

# Feature 81: Phase Context  
# Current phase (0.0-1.0) + phase progress (0.0-1.0)

# Feature 82: Session Time
# Relative timestamp from session start (0.0 to max_session_time)

# Feature 83: Phase Transition Signal
# Boolean: 1 if phase just changed, 0 if same phase

# Features 84-89: Reserved for future expansion
```

### **2.2 Implement ±300ms Action Window**

#### **Comprehensive Input State Extraction**
```python
def extract_comprehensive_input_state(self, gamestate: Dict) -> float:
    """Extract ALL actions within ±300ms of gamestate timestamp."""
    gamestate_timestamp = gamestate.get('timestamp', 0)
    
    # Find actions within ±300ms window
    nearby_actions = actions_df[
        (actions_df['timestamp'] >= gamestate_timestamp - 300) & 
        (actions_df['timestamp'] <= gamestate_timestamp + 300)
    ]
    
    # Combine ALL action data into comprehensive string
    action_strings = []
    for _, action in nearby_actions.iterrows():
        action_str = f"{action['event_type']}|{action['x_in_window']}|{action['y_in_window']}|{action['btn']}|{action['key']}|{action['scroll_dx']}|{action['scroll_dy']}|{action['timestamp']}"
        action_strings.append(action_str)
    
    # Hash the complete action sequence
    complete_sequence = "||".join(action_strings)
    return self.stable_hash(complete_sequence)
```

#### **Phase Context Features**
```python
def extract_phase_features(self, gamestate: Dict) -> List[float]:
    """Extract phase context and timing features."""
    features = []
    
    # Feature 81: Phase context (0.0-1.0)
    phase = gamestate.get('phase_context', {}).get('cycle_phase', 'unknown')
    phase_mapping = {
        'banking': 0.0,
        'moving_to_furnace': 0.25, 
        'crafting': 0.5,
        'moving_to_bank': 0.75
    }
    phase_value = phase_mapping.get(phase, 0.0)
    features.append(phase_value)
    
    # Feature 82: Session time (normalized 0.0-1.0)
    session_start = gamestate.get('phase_context', {}).get('session_start_time', 0)
    current_time = gamestate.get('timestamp', 0)
    session_duration = current_time - session_start
    normalized_time = min(session_duration / 60000, 1.0)  # Normalize to 1 minute
    features.append(normalized_time)
    
    # Feature 83: Phase transition signal
    previous_phase = self.get_previous_phase(gamestate)
    phase_changed = 1.0 if phase != previous_phase else 0.0
    features.append(phase_changed)
    
    return features
```

### **2.3 Update Feature Groups**
```python
self.feature_groups = {
    "Player State": (0, 5),           # 5 features
    "Enhanced Interactions": (5, 6),   # 1 feature
    "Camera": (6, 11),                # 5 features
    "Inventory": (11, 39),            # 28 features
    "Bank": (39, 41),                 # 2 features
    "Game Objects": (41, 55),         # 14 features
    "NPCs": (55, 60),                 # 5 features
    "Tabs": (60, 73),                 # 13 features
    "Chatbox": (73, 78),              # 5 features
    "Skills": (78, 80),               # 2 features
    "Actions & Phase": (80, 90)       # 10 features (NEW: comprehensive input state + phase context)
}
```

---

## **Phase 3: Data Collection Campaign (Week 3)**

### **3.1 Structured Data Collection Requirements**

#### **Session Structure**
- **Duration**: 10+ complete crafting cycles
- **Variation**: Different mouse paths, timing variations, approach styles
- **Quality**: Clean, consistent gameplay without interruptions
- **Metadata**: Record session start/end times, total cycles completed

#### **Data Quality Checklist**
- [ ] Phase detection accuracy >95%
- [ ] All phases represented in data
- [ ] Smooth phase transitions captured
- [ ] Action data covers full ±300ms windows
- [ ] Timestamps align between gamestates and actions

### **3.2 Collection Protocol**
```bash
# 1. Start RuneLite with StateExporter plugin enabled
# 2. Begin at bank with materials
# 3. Execute 10+ complete cycles:
#    - Bank → Withdraw materials
#    - Walk to furnace
#    - Craft rings
#    - Walk back to bank
#    - Deposit rings, withdraw materials
# 4. Vary approach each cycle
# 5. End session at bank
```

---

## **Phase 4: Feature Validation & Analysis (Week 4)**

### **4.1 Run Enhanced Feature Extraction**
```bash
# Run the updated feature extraction
python extract_features.py

# Expected output:
# - Total samples: [number] gamestates
# - Features per sample: 90
# - All mapping files saved successfully
```

### **4.2 Feature Analysis Commands**
```bash
# Check overall feature statistics
python feature_interpreter.py --feature-stats

# Analyze specific new features
python feature_interpreter.py --feature-index 80  # Input state hash
python feature_interpreter.py --feature-index 81  # Phase context
python feature_interpreter.py --feature-index 82  # Session timing
python feature_interpreter.py --feature-index 83  # Phase transition
```

### **4.3 Data Quality Assessment**

#### **Phase Distribution Analysis**
```python
# Expected phase values in Feature 81:
# - 0.0 (banking): ~25% of samples
# - 0.25 (moving_to_furnace): ~25% of samples  
# - 0.5 (crafting): ~25% of samples
# - 0.75 (moving_to_bank): ~25% of samples
```

#### **Action Coverage Validation**
```python
# Feature 80 (input state hash) should show:
# - High variety: >1000 unique values
# - Good distribution: no single value >20%
# - Meaningful patterns: similar game states should have similar action hashes
```

#### **Temporal Coherence Check**
```python
# Feature 82 (session timing) should show:
# - Progression from 0.0 to 1.0 across session
# - Smooth transitions between phases
# - Consistent timing patterns
```

---

## **Phase 5: Training Data Preparation (Week 5)**

### **5.1 Training Dataset Structure**
```python
# Input Features (X): 90 total
# - Game State: 80 features (player, inventory, bank, objects, etc.)
# - Input Context: 1 feature (comprehensive action hash)
# - Phase Context: 1 feature (current phase)
# - Session Timing: 1 feature (relative time)
# - Reserved: 7 features for future expansion

# Output Target (Y): Next Action Sequence
# - Format: Same as Feature 80 (comprehensive action hash)
# - Represents: The action sequence that should follow this game state
# - Learning Goal: Model learns to predict correct next actions
```

### **5.2 Data Validation Requirements**
- [ ] All 90 features show meaningful variation
- [ ] No features are constant or have extreme outliers
- [ ] Phase transitions align with game state changes
- [ ] Input state features capture sufficient variety
- [ ] Session timing progresses logically

---

## **Phase 6: Model Training Preparation (Week 6)**

### **6.1 Training Strategy**

#### **Phase-Aware Training**
```python
# Option 1: Single model with phase context
# - Include phase features in input
# - Model learns phase-specific behaviors
# - Simpler architecture, single training process

# Option 2: Separate models per phase
# - Train 4 separate models (one per phase)
# - Each model specializes in phase-specific actions
# - More complex but potentially better performance
```

#### **Sequence Learning Implementation**
```python
# Use LSTM/GRU architecture to capture temporal patterns
# Input: Sequence of gamestates with actions
# Output: Next action sequence
# Loss: Difference between predicted and actual action hashes
```

### **6.2 Training Data Requirements**
```python
# Minimum viable dataset:
# - 100+ gamestate-action pairs per phase
# - 10+ complete crafting cycles
# - Varied approaches and timing
# - Clean, consistent gameplay data

# Optimal dataset:
# - 500+ gamestate-action pairs per phase
# - 50+ complete crafting cycles
# - Multiple players/approaches
# - Edge cases and error recovery
```

---

## **Implementation Priority Order**

### **Week 1: Foundation**
1. Modify StateExporterPlugin.java for phase detection
2. Add phase context to gamestates
3. Enhance action recording with phase information

### **Week 2: Feature Engineering**
1. Implement comprehensive input state extraction
2. Add phase context features
3. Update feature groups and metadata

### **Week 3: Data Collection**
1. Execute structured data collection campaign
2. Collect 10+ complete crafting cycles
3. Ensure data quality and consistency

### **Week 4: Validation**
1. Run enhanced feature extraction
2. Analyze feature distributions and quality
3. Validate phase detection accuracy

### **Week 5: Preparation**
1. Prepare training dataset
2. Validate data quality
3. Structure data for model training

### **Week 6: Training Setup**
1. Choose training strategy (single vs. separate models)
2. Implement sequence learning architecture
3. Begin model training process

---

## **Key Success Metrics**

### **Phase Detection**
- **Accuracy**: >95% correct phase identification
- **Consistency**: Phases detected consistently across sessions
- **Transitions**: Smooth phase changes captured

### **Action Coverage**
- **Variety**: Feature 80 should have >1000 unique values
- **Distribution**: No single action pattern >20% of samples
- **Meaning**: Similar game states should have similar action hashes

### **Data Quality**
- **Completeness**: All 90 features populated correctly
- **Consistency**: Timestamps align between gamestates and actions
- **Variation**: Sufficient variety in all features for learning

### **Training Readiness**
- **Dataset Size**: 100+ samples per phase minimum
- **Feature Quality**: All features show meaningful variation
- **Phase Balance**: Roughly equal representation of all phases

---

## **Technical Notes**

### **File Modifications Required**
1. `StateExporterPlugin.java` - Add phase detection and context
2. `extract_features.py` - Replace action features with comprehensive input state
3. `feature_interpreter.py` - Update for new feature structure
4. `actions.csv` - Add phase and timing columns

### **Dependencies**
- RuneLite client with StateExporter plugin
- Python 3.7+ with pandas, numpy
- Sufficient storage for gamestate and action data
- Consistent gameplay sessions for data collection

### **Risk Mitigation**
- **Phase Detection Errors**: Implement fallback logic and manual validation
- **Data Quality Issues**: Regular validation checks during collection
- **Feature Extraction Problems**: Comprehensive error handling and logging
- **Training Data Imbalance**: Ensure equal phase representation

---

## **Next Steps**

1. **Review and approve** this implementation plan
2. **Begin Phase 1** - Modify StateExporterPlugin.java
3. **Set up development environment** for Java and Python modifications
4. **Create test data** to validate phase detection logic
5. **Execute plan** following the weekly phases outlined above

This hybrid approach provides the best of both worlds: reliable phase detection through rules and personalized action learning through imitation learning.
