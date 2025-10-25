#!/usr/bin/env python3
"""
Simple Jagex account creation script
Run this from the simple_recorder directory
"""

import sys
import os
import time

# Add the ui-bot src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui-bot', 'src'))

# Import the required modules
from selenium_driver import make_driver
from flows.flow_create_jagex_account import navigate_to_jagex_account_creation

def main():
    print("🚀 Starting Jagex account creation navigation...")
    
    driver = make_driver()
    try:
        # Navigate to Jagex account creation page
        navigate_to_jagex_account_creation(driver)
        
        print("✅ Browser opened to Jagex account creation page")
        print("📝 You can now manually create the RuneScape account.")
        print("⏳ Keeping browser open for 60 seconds to allow manual account creation...")
        
        # Keep browser open longer for manual account creation
        time.sleep(60)
        
    except Exception as e:
        print(f"❌ Jagex account creation navigation failed: {e}")
        print("🔧 Check the browser for any issues")
        time.sleep(10)
    finally:
        driver.quit()
        print("🔚 Browser closed")

if __name__ == "__main__":
    main()








