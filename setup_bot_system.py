#!/usr/bin/env python3
"""
Setup script for OSRS Automated Bot System
Installs dependencies and sets up the bot environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🤖 OSRS Automated Bot System - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
        
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements file
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_bot.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "checkpoints",
        "logs", 
        "config",
        "human_behavior_analysis",
        "bot_data",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created: {directory}/")
        
    return True

def create_default_config():
    """Create default configuration files"""
    print("\n⚙️  Creating default configuration...")
    
    try:
        # Import config manager
        sys.path.append(str(Path(__file__).parent / "ilbot" / "bot"))
        from bot_config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.create_default_config()
        
        print("✅ Default configuration created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def check_system_compatibility():
    """Check system compatibility"""
    print("\n💻 Checking system compatibility...")
    
    system = platform.system()
    print(f"   Operating System: {system}")
    
    if system == "Windows":
        print("   ✅ Windows - Fully supported")
    elif system == "Linux":
        print("   ✅ Linux - Fully supported")
    elif system == "Darwin":
        print("   ⚠️  macOS - Limited support (may need additional setup)")
    else:
        print("   ❌ Unknown system - May not work properly")
        
    # Check for display
    try:
        import tkinter
        print("   ✅ Display support available")
    except ImportError:
        print("   ❌ No display support - GUI features may not work")
        
    return True

def setup_environment():
    """Setup environment variables and paths"""
    print("\n🌍 Setting up environment...")
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    python_path = os.environ.get('PYTHONPATH', '')
    
    if str(current_dir) not in python_path:
        os.environ['PYTHONPATH'] = f"{current_dir}{os.pathsep}{python_path}"
        print("   ✅ Added current directory to PYTHONPATH")
        
    # Create .env file
    env_file = current_dir / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(f"PYTHONPATH={current_dir}\n")
            f.write("BOT_CONFIG_DIR=config\n")
            f.write("BOT_LOG_DIR=logs\n")
            f.write("BOT_DATA_DIR=bot_data\n")
        print("   ✅ Created .env file")
        
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        "torch",
        "numpy", 
        "cv2",
        "pyautogui",
        "pynput",
        "matplotlib",
        "pandas"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
            
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
        
    print("✅ All modules imported successfully")
    return True

def create_launch_scripts():
    """Create launch scripts for different platforms"""
    print("\n🚀 Creating launch scripts...")
    
    current_dir = Path(__file__).parent.absolute()
    
    # Windows batch file
    if platform.system() == "Windows":
        batch_file = current_dir / "launch_bot.bat"
        with open(batch_file, 'w') as f:
            f.write(f"@echo off\n")
            f.write(f"cd /d {current_dir}\n")
            f.write(f"python -m ilbot.bot.bot_launcher --interactive\n")
            f.write(f"pause\n")
        print("   ✅ Created: launch_bot.bat")
        
    # Unix shell script
    else:
        shell_file = current_dir / "launch_bot.sh"
        with open(shell_file, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"cd {current_dir}\n")
            f.write(f"python3 -m ilbot.bot.bot_launcher --interactive\n")
        os.chmod(shell_file, 0o755)
        print("   ✅ Created: launch_bot.sh")
        
    return True

def print_setup_complete():
    """Print setup completion message"""
    print("\n" + "=" * 60)
    print("🎉 Bot System Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Train your model and save it to the 'checkpoints/' directory")
    print("2. Collect human behavior data and place it in 'human_behavior_analysis/'")
    print("3. Customize configuration in 'config/user_config.json'")
    print("4. Launch the bot system:")
    
    if platform.system() == "Windows":
        print("   • Double-click 'launch_bot.bat'")
        print("   • Or run: python -m ilbot.bot.bot_launcher --interactive")
    else:
        print("   • Run: ./launch_bot.sh")
        print("   • Or run: python3 -m ilbot.bot.bot_launcher --interactive")
        
    print()
    print("🔒 Safety Features:")
    print("   • Press F12 for emergency stop")
    print("   • Move mouse/keyboard to pause bot")
    print("   • Automatic suspicious pattern detection")
    print()
    print("📚 Documentation:")
    print("   • Check 'README.md' for detailed usage instructions")
    print("   • Review 'config/default_config.json' for configuration options")
    print()
    print("⚠️  Important:")
    print("   • Always test in a safe environment first")
    print("   • Monitor the bot while it's running")
    print("   • Keep your trained models secure")
    print()
    print("🤖 Happy Botting!")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
        
    # Check system compatibility
    check_system_compatibility()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        print("   Try running: pip install -r requirements_bot.txt manually")
        sys.exit(1)
        
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Create default config
    create_default_config()
    
    # Test imports
    if not test_imports():
        print("\n❌ Some modules failed to import")
        print("   Check the error messages above and reinstall dependencies if needed")
        sys.exit(1)
        
    # Create launch scripts
    create_launch_scripts()
    
    # Setup complete
    print_setup_complete()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
