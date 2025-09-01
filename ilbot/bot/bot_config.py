#!/usr/bin/env python3
"""
Configuration system for the Automated Bot System
Provides flexible configuration management for different bot scenarios
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class SafetyConfig:
    """Safety monitoring configuration"""
    max_actions_per_minute: int = 120
    max_click_frequency: int = 10
    suspicious_coordinate_jumps: int = 500
    human_input_detection_enabled: bool = True
    emergency_stop_key: str = 'f12'
    suspicious_pattern_detection: bool = True
    repetitive_action_threshold: int = 5
    unrealistic_timing_threshold: float = 0.05

@dataclass
class BehaviorConfig:
    """Human behavior simulation configuration"""
    mouse_jitter_enabled: bool = True
    mouse_jitter_std: float = 2.0
    timing_variation_enabled: bool = True
    double_click_probability: float = 0.05
    scroll_variation_enabled: bool = True
    key_timing_variation: bool = True
    
    # Timing ranges (seconds)
    mouse_delay_before: tuple = (0.05, 0.15)
    mouse_delay_after: tuple = (0.02, 0.08)
    click_delay_before: tuple = (0.1, 0.3)
    click_delay_after: tuple = (0.05, 0.15)
    key_delay_before: tuple = (0.05, 0.2)
    key_delay_after: tuple = (0.02, 0.1)
    scroll_delay_before: tuple = (0.1, 0.25)
    scroll_delay_after: tuple = (0.05, 0.15)

@dataclass
class ModelConfig:
    """Model prediction configuration"""
    model_path: str = "checkpoints/best.pt"
    device: str = "cuda"
    batch_size: int = 1
    prediction_threshold: float = 0.5
    max_predictions_per_batch: int = 10
    enable_uncertainty_handling: bool = True
    fallback_actions: bool = True

@dataclass
class GameConfig:
    """Game-specific configuration"""
    game_window_title: str = "Old School RuneScape"
    screen_resolution: tuple = (1920, 1080)
    game_region: tuple = (0, 0, 1920, 1080)  # x, y, width, height
    coordinate_system: str = "absolute"  # absolute or relative
    click_offset: tuple = (0, 0)  # x, y offset for clicks
    
    # Game state detection
    inventory_detection_enabled: bool = True
    bank_detection_enabled: bool = True
    player_position_detection: bool = True
    nearby_object_detection: bool = True

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = "INFO"
    log_file: str = "bot_system.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_rotation: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Performance monitoring
    enable_performance_logging: bool = True
    performance_log_interval: int = 60  # seconds
    enable_action_logging: bool = True
    enable_safety_logging: bool = True

@dataclass
class BotConfig:
    """Main bot configuration"""
    # Core settings
    bot_name: str = "OSRS_Automated_Bot"
    version: str = "1.0.0"
    enabled: bool = True
    
    # Component configurations
    safety: SafetyConfig = None
    behavior: BehaviorConfig = None
    model: ModelConfig = None
    game: GameConfig = None
    logging: LoggingConfig = None
    
    # Paths
    behavior_data_path: str = "human_behavior_analysis/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    config_dir: str = "config/"
    
    # Advanced settings
    enable_adaptive_behavior: bool = True
    enable_safety_learning: bool = True
    enable_performance_optimization: bool = True
    max_runtime_hours: int = 24
    auto_restart_enabled: bool = False
    restart_interval_hours: int = 4
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.behavior is None:
            self.behavior = BehaviorConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.game is None:
            self.game = GameConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

class ConfigManager:
    """Manages bot configuration loading, saving, and validation"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.default_config_file = self.config_dir / "default_config.json"
        self.user_config_file = self.config_dir / "user_config.json"
        
    def create_default_config(self) -> BotConfig:
        """Create and save default configuration"""
        config = BotConfig()
        self.save_config(config, self.default_config_file)
        return config
        
    def load_config(self, config_file: Optional[str] = None) -> BotConfig:
        """Load configuration from file"""
        if config_file is None:
            config_file = self.user_config_file
            
        config_path = Path(config_file)
        
        if not config_path.exists():
            # Create default config if user config doesn't exist
            logger.info("User config not found, creating default configuration")
            return self.create_default_config()
            
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Create config object from data
            config = self._dict_to_config(config_data)
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            return self.create_default_config()
            
    def save_config(self, config: BotConfig, config_file: Optional[str] = None):
        """Save configuration to file"""
        if config_file is None:
            config_file = self.user_config_file
            
        config_path = Path(config_file)
        config_path.parent.mkdir(exist_ok=True)
        
        try:
            config_data = self._config_to_dict(config)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
                
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def _config_to_dict(self, config: BotConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        config_dict = asdict(config)
        
        # Handle nested dataclasses
        for key, value in config_dict.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = asdict(value)
                
        return config_dict
        
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> BotConfig:
        """Convert dictionary to config object"""
        # Handle nested configurations
        if 'safety' in config_dict:
            config_dict['safety'] = SafetyConfig(**config_dict['safety'])
        if 'behavior' in config_dict:
            config_dict['behavior'] = BehaviorConfig(**config_dict['behavior'])
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'game' in config_dict:
            config_dict['game'] = GameConfig(**config_dict['game'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
            
        return BotConfig(**config_dict)
        
    def validate_config(self, config: BotConfig) -> bool:
        """Validate configuration for errors"""
        errors = []
        
        # Check required paths
        if not os.path.exists(config.model.model_path):
            errors.append(f"Model path does not exist: {config.model.model_path}")
            
        if not os.path.exists(config.behavior_data_path):
            errors.append(f"Behavior data path does not exist: {config.behavior_data_path}")
            
        # Check safety thresholds
        if config.safety.max_actions_per_minute <= 0:
            errors.append("max_actions_per_minute must be positive")
            
        if config.safety.max_click_frequency <= 0:
            errors.append("max_click_frequency must be positive")
            
        # Check timing ranges
        for timing_config in [config.behavior.mouse_delay_before, config.behavior.mouse_delay_after,
                            config.behavior.click_delay_before, config.behavior.click_delay_after]:
            if timing_config[0] < 0 or timing_config[1] < 0:
                errors.append("Delay times must be non-negative")
            if timing_config[0] >= timing_config[1]:
                errors.append("Delay before must be less than delay after")
                
        # Check game configuration
        if config.game.screen_resolution[0] <= 0 or config.game.screen_resolution[1] <= 0:
            errors.append("Screen resolution must be positive")
            
        if len(errors) > 0:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
            
        logger.info("Configuration validation passed")
        return True
        
    def create_scenario_config(self, scenario_name: str, **kwargs) -> BotConfig:
        """Create configuration for specific game scenarios"""
        base_config = self.load_config()
        
        # Scenario-specific modifications
        if scenario_name == "woodcutting":
            base_config.game.game_window_title = "Woodcutting Bot"
            base_config.behavior.mouse_jitter_std = 3.0  # More natural for woodcutting
            base_config.safety.max_actions_per_minute = 60  # Slower for woodcutting
            
        elif scenario_name == "fishing":
            base_config.game.game_window_title = "Fishing Bot"
            base_config.behavior.click_delay_before = (0.2, 0.4)  # Slower for fishing
            base_config.safety.max_actions_per_minute = 40  # Very slow for fishing
            
        elif scenario_name == "combat":
            base_config.game.game_window_title = "Combat Bot"
            base_config.behavior.mouse_jitter_std = 1.5  # Precise for combat
            base_config.safety.max_actions_per_minute = 150  # Fast for combat
            
        elif scenario_name == "banking":
            base_config.game.game_window_title = "Banking Bot"
            base_config.behavior.click_delay_before = (0.15, 0.25)
            base_config.safety.max_actions_per_minute = 80
            
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
            elif hasattr(base_config.safety, key):
                setattr(base_config.safety, key, value)
            elif hasattr(base_config.behavior, key):
                setattr(base_config.behavior, key, value)
            elif hasattr(base_config.model, key):
                setattr(base_config.model, key, value)
            elif hasattr(base_config.game, key):
                setattr(base_config.game, key, value)
            elif hasattr(base_config.logging, key):
                setattr(base_config.logging, key, value)
                
        return base_config
        
    def get_config_summary(self, config: BotConfig) -> str:
        """Get human-readable configuration summary"""
        summary = f"""
🤖 Bot Configuration Summary
============================
Bot Name: {config.bot_name}
Version: {config.version}
Enabled: {config.enabled}

🔒 Safety Settings:
  • Max actions per minute: {config.safety.max_actions_per_minute}
  • Max click frequency: {config.safety.max_click_frequency}
  • Human input detection: {config.safety.human_input_detection_enabled}
  • Emergency stop key: {config.safety.emergency_stop_key}

🎭 Behavior Settings:
  • Mouse jitter: {config.behavior.mouse_jitter_enabled}
  • Timing variation: {config.behavior.timing_variation_enabled}
  • Double click probability: {config.behavior.double_click_probability}

🧠 Model Settings:
  • Model path: {config.model.model_path}
  • Device: {config.model.device}
  • Prediction threshold: {config.model.prediction_threshold}

🎮 Game Settings:
  • Window title: {config.game.game_window_title}
  • Screen resolution: {config.game.screen_resolution}
  • Coordinate system: {config.game.coordinate_system}

📝 Logging Settings:
  • Log level: {config.logging.log_level}
  • Log file: {config.logging.log_file}
  • Performance logging: {config.logging.enable_performance_logging}
"""
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create config manager
    config_manager = ConfigManager()
    
    # Create default configuration
    default_config = config_manager.create_default_config()
    
    # Create scenario-specific configurations
    woodcutting_config = config_manager.create_scenario_config("woodcutting")
    fishing_config = config_manager.create_scenario_config("fishing")
    combat_config = config_manager.create_scenario_config("combat")
    
    # Validate configurations
    config_manager.validate_config(default_config)
    config_manager.validate_config(woodcutting_config)
    
    # Print summaries
    print(config_manager.get_config_summary(default_config))
    print("\n" + "="*50 + "\n")
    print("Woodcutting Scenario:")
    print(config_manager.get_config_summary(woodcutting_config))
