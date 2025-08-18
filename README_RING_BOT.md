# 🏗️ Sapphire Ring Crafting Bot - Complete Guide

This guide explains how to build and train an AI bot that can craft sapphire rings in Old School RuneScape using imitation learning.

## 🎯 **What This Bot Does**

The bot learns to craft sapphire rings by watching you do it manually. It observes:
- **Game State**: Your position, camera, inventory, bank status, nearby objects
- **Your Actions**: Mouse clicks, movements, key presses, timing
- **Results**: How the game state changes after each action

Then it learns to predict what action to take next based on the current game state.

## 🏗️ **System Architecture**

```
Raw Data Collection → Feature Extraction → Training → Bot Execution
       ↓                    ↓              ↓           ↓
   Gamestates +         Numerical      Neural      Real-time
   Actions +           Features       Network     Inference
   Screenshots                        Training
```

### **Data Flow:**
1. **RuneLite Plugin** → Collects game state every tick
2. **Python Recorder** → Records your mouse/keyboard inputs
3. **Feature Extractor** → Converts data to numerical features
4. **Training Script** → Trains neural network on state→action pairs
5. **Bot Executor** → Uses trained model to play the game

## 📊 **Feature Breakdown**

### **State Features (205 total):**
- **Player (12)**: Position, health, animation_id, animation_name, is_moving, movement_direction, run energy, prayer, special attack, last_action, last_target
- **Camera (5)**: X, Y, Z, pitch, yaw
- **Inventory (84)**: 28 slots × 3 (item ID, quantity, name_hash)
- **Bank State (7)**: Bank open/closed, quantity mode (1/5/10/X/All), custom quantity
- **Game Objects (60)**: 15 closest objects × 4 (id, distance, x, name_hash)
- **Bank Booths (20)**: 4 closest booths × 5 (id, distance, x, y, name_hash)
- **Furnaces (15)**: 3 closest furnaces × 5 (id, distance, x, y, name_hash)
- **Skills (2)**: Crafting level and XP

### **Action Features (10 total):**
- **Event Type (4)**: One-hot encoded (mouse_click, mouse_move, key_press, key_release)
- **Mouse Coordinates (2)**: X, Y coordinates (for mouse events)
- **Button/Key Info (2)**: Button hash and pressed state (for mouse) or key hash and 0 (for keyboard)
- **Active Keys (1)**: Hash of active keys string
- **Modifiers (1)**: Hash of modifiers string

## 🚀 **Getting Started**

### **Prerequisites:**
- Python 3.8+ with PyTorch
- RuneLite client with StateExporterPlugin
- OSRS account with crafting materials

### **Installation:**
```bash
cd bot_runelite_IL
pip install torch numpy matplotlib
```

## 📝 **Step 1: Data Collection**

### **A. Start RuneLite with Plugin:**
1. Launch RuneLite with the StateExporterPlugin enabled
2. Make sure the plugin is collecting data to `../data/gamestates/`
3. Verify screenshots are being saved to `../data/runelite_screenshots/`

### **B. Record Training Session:**
```bash
# Start a 5-minute training session
python collect_training_data.py --session_name "session_1" --duration 300
```

**During recording:**
1. Go to a bank (e.g., Al Kharid, Varrock)
2. Withdraw sapphires and gold bars
3. Walk to a furnace
4. Craft sapphire rings
5. Return to bank and repeat

**The script will:**
- Monitor gamestate files from RuneLite
- Monitor actions.csv from the recorder
- Align timestamps automatically
- Save organized training data

### **C. Collect Multiple Sessions:**
```bash
# Session 1: Basic crafting
python collect_training_data.py --session_name "basic_crafting" --duration 300

# Session 2: Different bank location
python collect_training_data.py --session_name "varrock_bank" --duration 300

# Session 3: Different furnace location
python collect_training_data.py --session_name "edgeville_furnace" --duration 300
```

**Aim for:**
- 5-10 sessions total
- 3-5 minutes each
- Different locations and scenarios
- At least 1000 state-action pairs

## 🧠 **Step 2: Training the Bot**

### **A. Basic Training:**
```bash
cd model
python train_ring_bot.py
```

**Training Parameters:**
- **Epochs**: 100 (adjust based on data size)
- **Batch Size**: 32 (adjust based on memory)
- **Learning Rate**: 1e-4 (adjust if loss doesn't decrease)
- **Sequence Length**: 5 (how many previous states to consider)

### **B. Monitor Training:**
The script will show:
- Training progress every 10 epochs
- Loss values and learning rate
- Checkpoints every 25 epochs
- Training loss plot at the end

### **C. Training Tips:**
- **Loss should decrease** over time
- **If loss plateaus**: Increase learning rate or add more data
- **If loss spikes**: Decrease learning rate or check data quality
- **Save checkpoints** to resume training later

## 🤖 **Step 3: Bot Execution (Coming Soon)**

Once training is complete, the bot will be able to:
1. **Read current game state** from RuneLite
2. **Predict next action** using the trained model
3. **Execute actions** (mouse clicks, key presses)
4. **Monitor results** and adjust behavior

## 📁 **File Structure**

```
bot_runelite_IL/
├── model/
│   ├── feature_extraction_rings.py    # Feature extraction logic
│   ├── train_ring_bot.py             # Training script
│   └── README_RING_BOT.md            # This guide
├── collect_training_data.py           # Data collection script
├── data/
│   ├── gamestates/                    # RuneLite gamestate files
│   ├── runelite_screenshots/          # Game screenshots
│   └── actions.csv                    # Recorded inputs
└── training_data/                     # Organized training sessions
    ├── session_1/
    │   ├── gamestates/                # Session-specific gamestates
    │   ├── actions.csv                # Session-specific actions
    │   ├── features/                  # Extracted features
    │   └── metadata.json              # Session information
    └── session_2/
        └── ...
```

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. No gamestates collected:**
- Check RuneLite plugin is enabled
- Verify plugin configuration
- Check file permissions

**2. No actions recorded:**
- Ensure main_app.py is running
- Check window focus detection
- Verify actions.csv is being written

**3. Training fails:**
- Check data quality and quantity
- Verify feature extraction works
- Check PyTorch installation

**4. Poor training results:**
- Collect more training data
- Try different locations/scenarios
- Adjust training parameters
- Check data alignment

### **Debug Commands:**
```bash
# Check gamestate collection
ls -la data/gamestates/ | wc -l

# Check action recording
wc -l data/actions.csv

# Test feature extraction
python -c "from model.feature_extraction_rings import extract_ring_crafting_features; print('Feature extraction works!')"

# Check training data
python collect_training_data.py --session_name "test" --duration 10
```

## 📈 **Performance Metrics**

### **Training Success Indicators:**
- **Loss decreases** from ~1.0 to <0.1
- **Training completes** without errors
- **Checkpoints saved** regularly
- **Loss plot shows** downward trend

### **Data Quality Indicators:**
- **Gamestates**: 100+ per session
- **Actions**: 50+ per session
- **Features**: 190 state + 8 action dimensions
- **Alignment**: Timestamps match between states and actions

## 🎯 **Next Steps**

### **Immediate:**
1. ✅ Collect training data (multiple sessions)
2. ✅ Train the model
3. 🔄 Test feature extraction
4. 🔄 Validate training data

### **Short-term:**
1. Create bot execution system
2. Add safety mechanisms
3. Test in controlled environment
4. Iterate on training data

### **Long-term:**
1. Expand to other crafting tasks
2. Add combat capabilities
3. Implement multi-account support
4. Create GUI for bot control

## 🚨 **Important Notes**

### **Safety:**
- **Never train on real gameplay** without supervision
- **Test in controlled environments** first
- **Follow OSRS rules** and community guidelines
- **Use at your own risk**

### **Limitations:**
- **Requires good training data** to work well
- **May not handle edge cases** perfectly
- **Needs regular retraining** for game updates
- **Not a replacement** for learning the game

### **Best Practices:**
- **Record diverse scenarios** for robust training
- **Validate data quality** before training
- **Start with simple tasks** and expand gradually
- **Monitor bot behavior** closely during testing

## 🤝 **Getting Help**

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages and logs
3. Verify all prerequisites are met
4. Check file paths and permissions
5. Ensure data collection is working

## 📚 **Additional Resources**

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Imitation Learning Papers**: Search for "behavioral cloning" and "imitation learning"
- **OSRS Wiki**: https://oldschool.runescape.wiki/
- **RuneLite API**: https://github.com/runelite/runelite

---

**Happy Bot Building! 🎮🤖**
