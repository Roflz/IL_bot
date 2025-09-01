# 🤖 OSRS Automated Bot System

A comprehensive automated bot system that connects trained imitation learning models to actual mouse and keyboard control, providing human-like behavior patterns for OSRS automation.

## 🎯 Features

### 🧠 **AI-Powered Automation**
- **Trained Model Integration**: Connects your trained imitation learning models directly to bot control
- **Real-time Prediction**: Generates actions based on current game state and model predictions
- **Uncertainty Handling**: Advanced loss functions with uncertainty-aware predictions

### 🎭 **Human Behavior Simulation**
- **Realistic Mouse Movements**: Natural mouse jitter and timing variations
- **Human-like Click Patterns**: Varied click timing and occasional double-clicks
- **Behavioral Analysis**: Uses your actual gameplay data to create realistic patterns

### 🔒 **Safety & Monitoring**
- **Emergency Stop**: Press F12 to immediately halt all bot activity
- **Human Input Detection**: Automatically pauses when you move mouse/keyboard
- **Pattern Detection**: Identifies and stops suspicious bot-like behavior
- **Real-time Monitoring**: Live status updates and performance metrics

### ⚙️ **Flexible Configuration**
- **Scenario Presets**: Pre-configured for woodcutting, fishing, combat, and banking
- **Custom Configurations**: Full control over safety thresholds and behavior patterns
- **Runtime Adjustment**: Modify settings without restarting the bot

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd bot_runelite_IL

# Run the setup script
python setup_bot_system.py
```

### 2. **Launch the Bot**
```bash
# Interactive mode (recommended for first use)
python -m ilbot.bot.bot_launcher --interactive

# Or use the generated launch script
# Windows: launch_bot.bat
# Linux/Mac: ./launch_bot.sh
```

### 3. **Select Your Scenario**
- **Woodcutting**: Slow, natural woodcutting behavior
- **Fishing**: Very slow, patient fishing behavior  
- **Combat**: Fast, precise combat behavior
- **Banking**: Medium speed banking operations
- **Custom**: Use your own configuration file

## 📁 System Architecture

```
ilbot/
├── bot/                          # Bot control system
│   ├── automated_bot_system.py  # Main bot orchestrator
│   ├── bot_config.py            # Configuration management
│   └── bot_launcher.py          # User interface and launcher
├── model/                       # AI models
│   ├── imitation_hybrid_model.py
│   └── advanced_losses.py
├── training/                    # Training components
│   └── enhanced_behavioral_metrics.py
└── analysis/                    # Behavior analysis
    └── human_behavior_analyzer.py
```

## 🔧 Configuration

### **Safety Settings**
```json
{
  "safety": {
    "max_actions_per_minute": 120,
    "max_click_frequency": 10,
    "human_input_detection_enabled": true,
    "emergency_stop_key": "f12"
  }
}
```

### **Behavior Settings**
```json
{
  "behavior": {
    "mouse_jitter_enabled": true,
    "mouse_jitter_std": 2.0,
    "timing_variation_enabled": true,
    "double_click_probability": 0.05
  }
}
```

### **Game Settings**
```json
{
  "game": {
    "game_window_title": "Old School RuneScape",
    "screen_resolution": [1920, 1080],
    "coordinate_system": "absolute"
  }
}
```

## 🎮 Usage Examples

### **Basic Bot Control**
```python
from ilbot.bot.bot_launcher import BotLauncher

# Create launcher
launcher = BotLauncher()

# Setup bot for woodcutting
launcher.setup_bot(scenario="woodcutting")

# Start automation
launcher.start_bot()

# Monitor and control
while True:
    command = input("Command (start/stop/status/quit): ")
    if command == "quit":
        launcher.stop_bot()
        break
```

### **Custom Configuration**
```python
from ilbot.bot.bot_config import ConfigManager

# Create custom configuration
config_manager = ConfigManager()
custom_config = config_manager.create_scenario_config("combat")

# Modify settings
custom_config.behavior.mouse_jitter_std = 1.0  # More precise
custom_config.safety.max_actions_per_minute = 150  # Faster

# Save configuration
config_manager.save_config(custom_config, "my_combat_config.json")
```

### **Advanced Bot Control**
```python
from ilbot.bot.automated_bot_system import AutomatedBotSystem

# Create bot with custom config
bot = AutomatedBotSystem(
    model_path="checkpoints/my_model.pt",
    behavior_data_path="my_behavior_data/",
    safety_enabled=True
)

# Start bot
bot.start_bot()

# Generate and execute actions
gamestate_sequence = torch.randn(1, 10, 128)
action_sequence = torch.randn(1, 10, 100, 7)

bot.generateAndExecuteActions(gamestate_sequence, action_sequence)

# Get performance report
report = bot.get_performance_report()
print(f"Actions executed: {report['performance_metrics']['total_actions']}")
```

## 🛡️ Safety Features

### **Emergency Controls**
- **F12 Key**: Immediate emergency stop
- **Mouse Movement**: Pauses bot when you move mouse
- **Keyboard Input**: Pauses bot when you press keys
- **Pattern Detection**: Stops suspicious repetitive behavior

### **Safety Thresholds**
- **Action Frequency**: Configurable maximum actions per minute
- **Click Frequency**: Limits rapid clicking
- **Coordinate Jumps**: Detects unrealistic mouse movements
- **Timing Patterns**: Identifies bot-like timing

### **Monitoring & Logging**
- **Real-time Status**: Live bot status updates
- **Performance Metrics**: Action counts and confidence levels
- **Safety Logs**: Detailed safety event logging
- **Error Handling**: Graceful error recovery

## 📊 Performance Monitoring

### **Real-time Metrics**
- **Action Count**: Total actions executed
- **Confidence Levels**: Model prediction confidence
- **Event Distribution**: Breakdown of action types
- **Safety Status**: Current safety monitoring state

### **Performance Reports**
```python
# Get comprehensive performance report
report = bot.get_performance_report()

print(f"Bot Status: {report['is_running']}")
print(f"Actions Executed: {report['performance_metrics']['total_actions']}")
print(f"Average Confidence: {report['performance_metrics']['average_confidence']:.2f}")

# Event distribution
event_dist = report['performance_metrics']['event_distribution']
for event_type, count in event_dist.items():
    print(f"{event_type}: {count}")
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Bot Not Starting**
```bash
# Check dependencies
pip install -r requirements_bot.txt

# Verify model path
ls checkpoints/

# Check configuration
python -c "from ilbot.bot.bot_config import ConfigManager; ConfigManager().load_config()"
```

#### **Import Errors**
```bash
# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/bot_runelite_IL"

# Or use setup script
python setup_bot_system.py
```

#### **Control Issues**
```bash
# Check system permissions
# Windows: Run as Administrator
# Linux: Check X11 permissions

# Verify control libraries
python -c "import pyautogui, pynput; print('Control libraries OK')"
```

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python -m ilbot.bot.bot_launcher --interactive --debug
```

## 📚 Advanced Usage

### **Custom Behavior Patterns**
```python
from ilbot.bot.automated_bot_system import HumanBehaviorSimulator

# Create custom behavior simulator
simulator = HumanBehaviorSimulator(behavior_analyzer)

# Customize mouse realism
def custom_mouse_realism(action):
    # Add your custom logic
    action.delay_before = 0.2  # Fixed delay
    return action

simulator._add_mouse_realism = custom_mouse_realism
```

### **Integration with Training Pipeline**
```python
# After training, automatically start bot
from ilbot.training.train_loop import train_model

# Train model
history = train_model(...)

# Start bot with trained model
bot = AutomatedBotSystem(
    model_path="checkpoints/latest.pt",
    behavior_data_path="human_behavior_analysis/"
)
bot.start_bot()
```

### **Multi-Bot Coordination**
```python
# Coordinate multiple bots for complex tasks
bots = {}

# Woodcutting bot
bots['woodcutter'] = AutomatedBotSystem(
    model_path="checkpoints/woodcutting.pt",
    behavior_data_path="woodcutting_behavior/"
)

# Banking bot  
bots['banker'] = AutomatedBotSystem(
    model_path="checkpoints/banking.pt", 
    behavior_data_path="banking_behavior/"
)

# Start all bots
for name, bot in bots.items():
    bot.start_bot()
    print(f"Started {name} bot")
```

## 🚨 Important Notes

### **Legal & Ethical Considerations**
- **Terms of Service**: Ensure compliance with game terms of service
- **Account Safety**: Use on test accounts, not main accounts
- **Fair Play**: Don't use for competitive advantages
- **Monitoring**: Always supervise bot operation

### **Technical Limitations**
- **Model Quality**: Bot performance depends on model training quality
- **Game Updates**: May need retraining after game updates
- **System Resources**: Requires adequate CPU/GPU for real-time inference
- **Network Stability**: Requires stable internet connection

### **Best Practices**
- **Start Small**: Begin with simple tasks and short sessions
- **Monitor Closely**: Watch bot behavior for unexpected actions
- **Regular Testing**: Test in safe environments before production use
- **Backup Models**: Keep multiple model checkpoints
- **Log Analysis**: Regularly review logs for issues

## 🤝 Contributing

### **Development Setup**
```bash
# Clone repository
git clone <repo-url>
cd bot_runelite_IL

# Install development dependencies
pip install -r requirements_bot.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Code formatting
black ilbot/
flake8 ilbot/
```

### **Adding New Features**
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Implement changes** with tests
4. **Submit pull request** with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OSRS Community**: For gameplay insights and testing
- **PyTorch Team**: For the excellent deep learning framework
- **Open Source Contributors**: For the libraries that make this possible

## 📞 Support

### **Getting Help**
- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Wiki**: Check the project wiki for additional resources

### **Community**
- **Discord**: Join our community server
- **Reddit**: r/OSRSBotting (if applicable)
- **Forums**: OSRS community forums

---

**⚠️ Disclaimer**: This bot system is for educational and research purposes. Users are responsible for ensuring compliance with game terms of service and applicable laws. The developers are not responsible for any consequences of using this software.

**🤖 Happy Botting!**
