#!/usr/bin/env python3
"""
Main Bot Launcher for OSRS Automated Bot System
Provides user interface for starting, stopping, and monitoring the bot
"""

import sys
import time
import threading
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ilbot.bot.automated_bot_system import AutomatedBotSystem, create_bot_system
from ilbot.bot.bot_config import ConfigManager, BotConfig
from ilbot.analysis.human_behavior_analyzer import HumanBehaviorAnalyzer

class BotLauncher:
    """Main bot launcher with user interface"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.bot: Optional[AutomatedBotSystem] = None
        self.config: Optional[BotConfig] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        
    def setup_bot(self, scenario: Optional[str] = None, config_file: Optional[str] = None):
        """Setup bot with configuration"""
        try:
            # Load configuration
            if config_file:
                self.config = self.config_manager.load_config(config_file)
            elif scenario:
                self.config = self.config_manager.create_scenario_config(scenario)
            else:
                self.config = self.config_manager.load_config()
                
            # Validate configuration
            if not self.config_manager.validate_config(self.config):
                print("❌ Configuration validation failed. Please check your settings.")
                return False
                
            # Create bot system
            bot_config = {
                'model_path': self.config.model.model_path,
                'behavior_data_path': self.config.behavior_data_path,
                'safety_enabled': self.config.safety is not None
            }
            
            self.bot = create_bot_system(bot_config)
            
            # Print configuration summary
            print(self.config_manager.get_config_summary(self.config))
            print("✅ Bot system setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up bot: {e}")
            return False
            
    def start_bot(self):
        """Start the automated bot"""
        if not self.bot:
            print("❌ Bot not set up. Run setup_bot() first.")
            return False
            
        try:
            print("🚀 Starting automated bot system...")
            self.bot.start_bot()
            
            # Start monitoring thread
            self.start_monitoring()
            
            print("✅ Bot started successfully!")
            print("📋 Controls:")
            print("  • Press F12 for emergency stop")
            print("  • Move mouse/keyboard to pause bot")
            print("  • Use stop_bot() to stop gracefully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting bot: {e}")
            return False
            
    def stop_bot(self):
        """Stop the automated bot gracefully"""
        if not self.bot:
            print("❌ Bot not running.")
            return False
            
        try:
            print("🛑 Stopping bot system...")
            self.bot.stop_bot()
            self.stop_monitoring()
            print("✅ Bot stopped successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping bot: {e}")
            return False
            
    def emergency_stop(self):
        """Emergency stop the bot"""
        if not self.bot:
            print("❌ Bot not running.")
            return False
            
        try:
            print("🚨 EMERGENCY STOP ACTIVATED!")
            self.bot.emergency_stop()
            self.stop_monitoring()
            print("✅ Bot emergency stopped!")
            return True
            
        except Exception as e:
            print(f"❌ Error during emergency stop: {e}")
            return False
            
    def start_monitoring(self):
        """Start monitoring thread for bot status"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            
    def _monitoring_loop(self):
        """Monitor bot status and performance"""
        while self.is_monitoring and self.bot and self.bot.is_running:
            try:
                # Get performance report
                report = self.bot.get_performance_report()
                
                # Print status update
                self._print_status_update(report)
                
                # Wait before next update
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(5)
                
    def _print_status_update(self, report: dict):
        """Print formatted status update"""
        if not report:
            return
            
        status = "🟢 RUNNING" if report.get('is_running', False) else "🔴 STOPPED"
        safety_status = report.get('safety_status', {})
        
        print(f"\n📊 Bot Status Update - {time.strftime('%H:%M:%S')}")
        print(f"Status: {status}")
        
        if safety_status.get('emergency_stop', False):
            print("🚨 SAFETY: Emergency stop active")
            
        if 'performance_metrics' in report:
            metrics = report['performance_metrics']
            if metrics:
                print(f"Actions executed: {metrics.get('total_actions', 0)}")
                print(f"Average confidence: {metrics.get('average_confidence', 0):.2f}")
                
                event_dist = metrics.get('event_distribution', {})
                if event_dist:
                    print("Event distribution:")
                    for event_type, count in event_dist.items():
                        print(f"  • {event_type}: {count}")
                        
    def interactive_mode(self):
        """Run bot in interactive mode with user commands"""
        print("🤖 OSRS Automated Bot System - Interactive Mode")
        print("=" * 50)
        
        # Setup bot
        print("\n📋 Available scenarios:")
        print("  • woodcutting - Slow, natural woodcutting behavior")
        print("  • fishing - Very slow, patient fishing behavior")
        print("  • combat - Fast, precise combat behavior")
        print("  • banking - Medium speed banking operations")
        print("  • custom - Use custom configuration file")
        
        scenario = input("\n🎯 Select scenario (or press Enter for default): ").strip().lower()
        
        if scenario == "custom":
            config_file = input("📁 Enter config file path: ").strip()
            if not self.setup_bot(config_file=config_file):
                return
        elif scenario:
            if not self.setup_bot(scenario=scenario):
                return
        else:
            if not self.setup_bot():
                return
                
        # Main command loop
        print("\n💡 Available commands:")
        print("  • start - Start the bot")
        print("  • stop - Stop the bot gracefully")
        print("  • emergency - Emergency stop")
        print("  • status - Show bot status")
        print("  • config - Show configuration")
        print("  • quit - Exit the program")
        
        while True:
            try:
                command = input("\n🤖 Bot> ").strip().lower()
                
                if command == "start":
                    self.start_bot()
                elif command == "stop":
                    self.stop_bot()
                elif command == "emergency":
                    self.emergency_stop()
                elif command == "status":
                    if self.bot:
                        report = self.bot.get_performance_report()
                        self._print_status_update(report)
                    else:
                        print("❌ Bot not set up")
                elif command == "config":
                    if self.config:
                        print(self.config_manager.get_config_summary(self.config))
                    else:
                        print("❌ No configuration loaded")
                elif command == "quit":
                    if self.bot and self.bot.is_running:
                        print("🛑 Stopping bot before exit...")
                        self.stop_bot()
                    print("👋 Goodbye!")
                    break
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                if self.bot and self.bot.is_running:
                    self.stop_bot()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="OSRS Automated Bot System")
    parser.add_argument("--scenario", "-s", 
                       choices=["woodcutting", "fishing", "combat", "banking"],
                       help="Game scenario to run")
    parser.add_argument("--config", "-c", 
                       help="Path to configuration file")
    parser.add_argument("--auto-start", "-a", 
                       action="store_true",
                       help="Automatically start bot after setup")
    parser.add_argument("--interactive", "-i", 
                       action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = BotLauncher()
    
    try:
        if args.interactive:
            # Interactive mode
            launcher.interactive_mode()
        else:
            # Command line mode
            if not launcher.setup_bot(scenario=args.scenario, config_file=args.config):
                sys.exit(1)
                
            if args.auto_start:
                if not launcher.start_bot():
                    sys.exit(1)
                    
                # Keep running until interrupted
                try:
                    print("🤖 Bot is running. Press Ctrl+C to stop.")
                    while launcher.bot and launcher.bot.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping bot...")
                    launcher.stop_bot()
                    
            else:
                print("✅ Bot setup complete. Use --auto-start to begin automation.")
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
