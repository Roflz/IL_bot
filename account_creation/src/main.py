
import time
import os
import random
import string

import win32gui
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from .flows.flow_create_gmail_account import create_gmail_account_pyautogui, open_gmail_signup_in_browser

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def generate_random_name():
    """Generate a random first and last name"""
    first_names = ["Charlie", "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Quinn", "Blake"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return random.choice(first_names), random.choice(last_names)

def generate_random_username():
    """Generate a random username"""
    adjectives = ["cool", "smart", "fast", "bright", "happy", "lucky", "brave", "wise", "kind", "bold"]
    nouns = ["user", "player", "gamer", "coder", "hacker", "ninja", "master", "pro", "ace", "star"]
    numbers = random.randint(100, 9999)
    return f"{random.choice(adjectives)}{random.choice(nouns)}{numbers}"

def generate_random_password():
    """Generate a random strong password"""
    length = random.randint(12, 16)
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))

def oauth_worker(flow, token_path):
    """Worker function to run OAuth and save token"""
    try:
        print("🔐 Starting OAuth2 authorization flow...")
        creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        print("✅ OAuth2 authorization completed and token.json saved!")
        return True
    except Exception as e:
        print(f"❌ OAuth2 failed: {e}")
        return False

def setup_gmail_oauth():
    """Setup Gmail OAuth2 authorization and generate token.json"""
    print("🔐 Setting up Gmail OAuth2 authorization...")
    
    token_path = "/account_creation/auth/gmail_oauth/token.json"
    credentials_path = "/account_creation/auth/gmail_oauth/credentials.json"
    
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            pass
        else:
            print("🌐 Opening browser for OAuth2 authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            
            # Start the OAuth flow in a separate thread so PyAutoGUI can take control
            import threading
            oauth_thread = threading.Thread(target=oauth_worker, args=(flow, token_path))
            oauth_thread.start()
            
            # Wait a moment for browser to open
            time.sleep(3)
            
            # Now PyAutoGUI can take control of the browser window
            print("🎯 PyAutoGUI taking control of OAuth browser window...")
            return flow, oauth_thread
    
    return None, None

def main():
    print("🚀 Starting automated Gmail account creation with PyAutoGUI...")
    
    # Setup Gmail OAuth2 authorization first
    # flow, oauth_thread = setup_gmail_oauth()
    open_gmail_signup_in_browser()

    def print_all_windows():
        def enum_handler(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:
                    windows.append(window_title)
            return True

        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        for title in windows:
            print(title)

    print_all_windows()
    
    # if flow and oauth_thread:
    #     # PyAutoGUI will now work in the OAuth browser window
    #     print("🎮 PyAutoGUI will now create the Gmail account in the OAuth browser...")
    
    try:
        # Generate random account creation parameters
        first_name, last_name = generate_random_name()
        base_username = generate_random_username()
        password = generate_random_password()
        recovery_email = None  # Optional
        
        print(f"🎲 Generated random account details:")
        print(f"   Name: {first_name} {last_name}")
        print(f"   Username: {base_username}")
        print(f"   Password: {password}")
        
        # Create the Gmail account using PyAutoGUI
        created_email = create_gmail_account_pyautogui(
            first_name=first_name,
            last_name=last_name,
            base_username=base_username,
            password=password,
            recovery_email=recovery_email
        )
        
        print(f"✅ Account created: {created_email}")
        
        # Handle OAuth authorization using PyAutoGUI
        # oauth_success = handle_oauth_authorization_pyautogui()
        
        if oauth_success:
            print("🎉 Complete! Gmail account created and OAuth authorized.")
            print(f"📧 New account: {created_email}")
            print("🔑 You can now use this account for automation!")
        else:
            print("⚠️ Account created but OAuth authorization may need manual completion")
        
        # Keep browser open to see final result
        print("⏳ Keeping browser open for 30 seconds to observe...")
        time.sleep(30)
        
    except Exception as e:
        print(f"❌ Gmail account creation failed: {e}")
        print("🔧 Check the browser for any manual steps needed")
        time.sleep(10)
    finally:
        print("🔚 Automation completed")

if __name__ == "__main__":
    main()
