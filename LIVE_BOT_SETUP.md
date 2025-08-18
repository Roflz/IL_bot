# 🚀 Live Gamestate Bot Setup Guide

This guide explains how to set up and use the **Live Gamestate Bot** system that reads real OSRS data from your RuneLite plugin.

## **🎯 What This System Does**

Instead of using generic screenshot analysis, the bot now:
1. **Reads live gamestate data** from your RuneLite plugin
2. **Extracts the same 128 features** your model was trained on
3. **Makes predictions in real-time** based on actual game state
4. **Shows you exactly what the bot sees** and what it wants to do

## **📋 Prerequisites**

- ✅ RuneLite with StateExporter plugin installed
- ✅ Trained model (`best_model.pth`) available
- ✅ Python environment with required packages

## **🔧 Setup Steps**

### **Step 1: Configure RuneLite Plugin**

1. **Open RuneLite plugin settings**
2. **Find "StateExporter" plugin**
3. **Set "Bot Mode" to "Bot1"** (or Bot2/Bot3)
4. **Enable "Enable Screenshots"** (optional, for visual debugging)
5. **Set output path** to your bot directory

### **Step 2: Start the Bot Controller**

```bash
cd bot_runelite_IL
python bot_controller_gui.py
```

### **Step 3: Load Your Model**

1. **Click "Load Model"** button
2. **Wait for confirmation** that model is loaded
3. **Verify model status** shows "✅ Model: Loaded"

### **Step 4: Enable Live Gamestate Mode**

1. **Check "Live Gamestate Mode"** checkbox
2. **Look for confirmation** in logs:
   ```
   🔄 Switching to Live Gamestate Mode
   This mode reads real OSRS data from RuneLite plugin
   Make sure to set 'Bot1' mode in RuneLite plugin settings
   📊 Gamestate monitoring thread started
   ```

## **🎮 Using the Live Bot**

### **Real-Time Monitoring**

Once enabled, the bot will:
- **Monitor gamestate data** every 100ms
- **Extract features** from live game state
- **Make predictions** using your trained model
- **Log key information** every 10 predictions:
  ```
  🎯 Live Prediction #10: Mouse: (400.0, 300.0), Click: Left (0.850)
  👤 Player: (3000, 3000) - WALKING
  🔄 Phase: crafting
  🏦 Bank: Closed
  📦 Inventory: 2/28 slots filled
  ```

### **Test Predictions**

Click **"Test Prediction"** to:
- **Get current gamestate** from RuneLite
- **Extract features** and make prediction
- **See detailed results** without executing actions

### **Key Information Displayed**

- **Player Position**: Current world coordinates
- **Animation State**: Walking, crafting, banking, etc.
- **Phase Context**: Current crafting phase
- **Bank Status**: Open/closed, material positions
- **Inventory**: Number of filled slots
- **Game Objects**: Nearby furnaces, bank booths, NPCs

## **📁 File Structure**

```
bot_runelite_IL/
├── data/
│   ├── bot1/                          # When "Bot1" mode active
│   │   └── runelite_gamestate.json    # Live gamestate data
│   ├── bot2/                          # When "Bot2" mode active
│   │   └── runelite_gamestate.json
│   └── bot3/                          # When "Bot3" mode active
│       └── runelite_gamestate.json
├── bot_controller_gui.py              # Main bot controller
├── live_feature_extractor.py          # Live feature extraction
└── model_architecture.py              # Your trained model
```

## **🔍 Troubleshooting**

### **"No gamestate data available"**
- **Check RuneLite plugin** is set to Bot1/Bot2/Bot3 mode
- **Verify output path** is correct
- **Check file permissions** in bot data directory

### **"Failed to extract features"**
- **Verify gamestate JSON** is valid
- **Check feature extractor** matches your training data
- **Look for missing fields** in gamestate data

### **Model predictions seem wrong**
- **Verify model file** is the correct trained model
- **Check feature extraction** matches training pipeline
- **Compare live features** with training data features

## **🎯 Next Steps**

1. **Test the system** with your trained model
2. **Observe predictions** in different game situations
3. **Fine-tune the model** if needed
4. **Enable actual actions** when ready (remove safe mode)

## **💡 Benefits of Live Gamestate Mode**

- **🎯 Accurate Predictions**: Uses same features as training
- **⚡ Real-Time Operation**: No screenshot delays
- **🔍 Better Debugging**: See exact game state
- **📊 Rich Information**: Player position, inventory, bank status
- **🎮 Game-Aware**: Understands crafting phases, interactions

## **🚨 Safety Notes**

- **Currently in safe mode** - bot only shows predictions
- **No actual mouse/keyboard actions** are performed
- **Test thoroughly** before enabling real actions
- **Monitor predictions** to ensure they make sense

---

**🎉 You're now ready to see your bot in action with real OSRS data!**
