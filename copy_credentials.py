#!/usr/bin/env python3
"""
Copy credentials.properties to credentials folder as unnamed_character_XX or custom name.
This script should be run when RuneLite starts.

Usage:
    python copy_credentials.py                    # Uses unnamed_character_XX
    python copy_credentials.py my_character      # Uses my_character.properties
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    credentials_file = Path("C:/Users/mahnr/.runelite/credentials.properties")
    
    # Check if credentials.properties exists
    if not credentials_file.exists():
        print(f"ERROR: credentials.properties not found at {credentials_file}")
        return 1
    
    # Create credentials directory if it doesn't exist
    credentials_dir = script_dir / "credentials"
    credentials_dir.mkdir(exist_ok=True)
    
    # Check if custom filename was provided
    if len(sys.argv) > 1:
        # Use custom filename
        custom_name = sys.argv[1]
        # Clean up the name (remove invalid filename characters)
        import re
        custom_name = re.sub(r'[<>:"/\\|?*]', '_', custom_name)
        filename = f"{custom_name}.properties"
        destination_file = credentials_dir / filename
    else:
        # Find the next available filename with default naming
        counter = 0
        while True:
            filename = f"unnamed_character_{counter:02d}.properties"
            destination_file = credentials_dir / filename
            
            if not destination_file.exists():
                break
            counter += 1
    
    # Copy the file
    try:
        shutil.copy2(credentials_file, destination_file)
        print(f"SUCCESS: Copied credentials to {destination_file}")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to copy credentials: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
