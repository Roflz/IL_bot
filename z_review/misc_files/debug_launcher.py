#!/usr/bin/env python3
"""Debug launcher for Bot GUI - handles imports and launches the application"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for debugging"""
    try:
        print("🔍 DEBUG: Starting Bot GUI debug launcher...")
        print(f"🔍 DEBUG: Current directory: {os.getcwd()}")
        print(f"🔍 DEBUG: Python path: {sys.path[:3]}...")
        
        # Test imports
        print("🔍 DEBUG: Testing imports...")
        
        try:
            from botgui.app import main as botgui_main
            print("✅ DEBUG: BotGUI imports successful")
        except ImportError as e:
            print(f"❌ DEBUG: BotGUI import failed: {e}")
            print("🔍 DEBUG: Trying alternative import...")
            
            # Try alternative import path
            sys.path.insert(0, str(current_dir / "botgui"))
            from app import main as botgui_main
            print("✅ DEBUG: Alternative import successful")
        
        # Set debug environment
        os.environ["DEBUG"] = "1"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        print("🔍 DEBUG: Environment set for debugging")
        print("🔍 DEBUG: Launching BotGUI...")
        
        # Launch the application
        botgui_main()
        
    except Exception as e:
        print(f"❌ DEBUG: Failed to launch: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
