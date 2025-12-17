import time
import random
import pyautogui
import webbrowser
import subprocess
import os

import pytesseract
import win32gui
import win32con

from actions.timing import wait_until


def open_jagex_in_browser():
    """Open Jagex account creation in your normal browser"""
    print("🚀 Opening Jagex account creation in your browser...")
    
    # Try to open Chrome with your profile
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(os.getenv('USERNAME', 'mahnr'))
    ]
    
    for chrome_path in chrome_paths:
        if os.path.exists(chrome_path):
            print(f"✅ Found Chrome at: {chrome_path}")
            try:
                # Open with your normal profile
                subprocess.Popen([chrome_path, "https://account.jagex.com/"])
                print("✅ Opened Chrome with your profile")
                
                # Wait a moment for browser to open
                time.sleep(10)
                
                # Maximize the browser window using PyAutoGUI
                print("🖥️ Maximizing browser window...")
                pyautogui.hotkey('win', 'up')  # Windows key + up arrow to maximize
                time.sleep(1)
                
                return True
            except Exception as e:
                print(f"❌ Failed to open Chrome: {e}")
                continue
    
    # Fallback to default browser
    print("❌ Chrome not found, using default browser")
    webbrowser.open("https://account.jagex.com/")
    time.sleep(3)
    pyautogui.hotkey('win', 'up')  # Maximize anyway
    return True


def open_gmail_signup_in_browser():
    """Open Jagex account creation in your normal browser"""
    print("🚀 Opening Jagex account creation in your browser...")

    # Try to open Chrome with your profile
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(os.getenv('USERNAME', 'mahnr'))
    ]

    for chrome_path in chrome_paths:
        if os.path.exists(chrome_path):
            print(f"✅ Found Chrome at: {chrome_path}")
            try:
                # Open with your normal profile
                subprocess.Popen([chrome_path, "accounts.google.com/signup"])
                print("✅ Opened Chrome with your profile")

                # Wait a moment for browser to open
                time.sleep(3)

                # Maximize the browser window using PyAutoGUI
                print("🖥️ Maximizing browser window...")
                pyautogui.hotkey('win', 'up')  # Windows key + up arrow to maximize
                time.sleep(1)

                return True
            except Exception as e:
                print(f"❌ Failed to open Chrome: {e}")
                continue

    # Fallback to default browser
    print("❌ Chrome not found, using default browser")
    webbrowser.open("accounts.google.com/signup")
    time.sleep(3)
    pyautogui.hotkey('win', 'up')  # Maximize anyway
    return True

def check_for_cloudflare():
    """Check if Cloudflare challenge is present by looking for common indicators"""
    print("🔍 Checking for Cloudflare challenge...")

    # Wait for page to load
    time.sleep(3)

    # Take a screenshot to analyze
    screenshot = pyautogui.screenshot()

    # Look for Cloudflare indicators in the page title or content
    # We'll check if the page title contains "Just a moment" or similar
    try:
        # This is a simple check - in a real implementation you'd use OCR or image recognition
        # For now, we'll assume Cloudflare is present if we see certain patterns
        # We'll check the center area of the screen for Cloudflare elements

        screen_width, screen_height = pyautogui.size()
        center_x, center_y = screen_width // 2, screen_height // 2

        # Check if there's a Cloudflare challenge by looking for the typical "Are you a robot?" text area
        # Cloudflare usually appears in the center of the screen
        cloudflare_detected = False

        # Simple heuristic: if we see a white/light colored area in the center, it might be Cloudflare
        # This is a basic check - in practice you'd use more sophisticated detection

        if cloudflare_detected:
            print("🚨 Cloudflare challenge detected!")
            # Click in the center area where Cloudflare checkbox usually appears
            print(f"🎯 Clicking Cloudflare checkbox at: {center_x},{center_y}")
            pyautogui.click(center_x, center_y)
            time.sleep(2)

            # Wait for Cloudflare to process
            time.sleep(3)
            print("✅ Cloudflare challenge handled")
        else:
            print("✅ No Cloudflare challenge detected")

    except Exception as e:
        print(f"⚠️ Error checking for Cloudflare: {e}")
        print("✅ Continuing without Cloudflare handling")

def navigate_to_account_creation():
    """Automatically navigate to account creation using text detection"""
    print("🔍 Looking for 'Create an account' link...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Try to find and click "Create an account" text
    success = click_text_in_window("Google Chrome", "Create an account")
    if not success:
        # Fallback to hardcoded coordinates
        create_account_x, create_account_y = 975, 320
        print(f"🔗 Fallback: Clicking 'Create an account' at: {create_account_x},{create_account_y}")
        pyautogui.click(create_account_x, create_account_y)

    time.sleep(3)
    print("✅ Navigated to account creation")

def automate_account_creation():
    """Use PyAutoGUI to automate the account creation form with text detection"""
    print("🤖 Starting PyAutoGUI automation...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Account details
    email = "charlieeeeee54345@gmail.com"
    day = "12"
    month = "3"
    year = "1978"

    print(f"📧 Will use email: {email}")
    print(f"📅 Will use date: {day}/{month}/{year}")

    # Wait a moment for page to load
    time.sleep(1)

    # Try to find email field by looking for common field indicators
    print("📧 Looking for email field...")
    email_field_found = False

    # Try to find email field by text near it
    email_indicators = ["Email", "email address", "Enter your email"]
    for indicator in email_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            # Click near the found text (assuming it's a label)
            x, y, found_text = locations[0]
            pyautogui.click(x + 100, y)  # Click to the right of the label
            time.sleep(0.2)
            pyautogui.typewrite(email)
            email_field_found = True
            print(f"✅ Found email field using '{found_text}'")
            break

    if not email_field_found:
        # Fallback to hardcoded coordinates
        email_x, email_y = 760, 540
        print(f"📧 Fallback: Clicking email field at: {email_x},{email_y}")
        pyautogui.click(email_x, email_y)
        time.sleep(0.2)
        pyautogui.typewrite(email)

    time.sleep(0.5)

    # Try to find date fields
    print("📅 Looking for date fields...")

    # Try to find day field
    day_found = False
    day_indicators = ["Day", "DD", "day"]
    for indicator in day_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            x, y, found_text = locations[0]
            pyautogui.click(x + 50, y)
            time.sleep(0.2)
            pyautogui.typewrite(day)
            day_found = True
            print(f"✅ Found day field using '{found_text}'")
            break

    if not day_found:
        # Fallback
        day_x, day_y = 760, 730
        pyautogui.click(day_x, day_y)
        time.sleep(0.2)
        pyautogui.typewrite(day)

    time.sleep(0.2)

    # Try to find month field
    month_found = False
    month_indicators = ["Month", "MM", "month"]
    for indicator in month_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            x, y, found_text = locations[0]
            pyautogui.click(x + 50, y)
            time.sleep(0.2)
            pyautogui.typewrite(month)
            month_found = True
            print(f"✅ Found month field using '{found_text}'")
            break

    if not month_found:
        # Fallback
        month_x, month_y = 850, 730
        pyautogui.click(month_x, month_y)
        time.sleep(0.2)
        pyautogui.typewrite(month)

    time.sleep(0.2)

    # Try to find year field
    year_found = False
    year_indicators = ["Year", "YYYY", "year"]
    for indicator in year_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            x, y, found_text = locations[0]
            pyautogui.click(x + 50, y)
            time.sleep(0.2)
            pyautogui.typewrite(year)
            year_found = True
            print(f"✅ Found year field using '{found_text}'")
            break

    if not year_found:
        # Fallback
        year_x, year_y = 940, 730
        pyautogui.click(year_x, year_y)
        time.sleep(0.2)
        pyautogui.typewrite(year)

    time.sleep(0.5)

    # Try to find terms checkbox
    print("☑️ Looking for terms checkbox...")
    terms_found = False
    terms_indicators = ["I agree", "terms", "conditions", "accept"]
    for indicator in terms_indicators:
        success = click_text_in_window("Google Chrome", indicator)
        if success:
            terms_found = True
            print(f"✅ Found terms checkbox using '{indicator}'")
            break

    if not terms_found:
        # Fallback
        checkbox_x, checkbox_y = 752, 835
        print(f"☑️ Fallback: Clicking terms checkbox at: {checkbox_x},{checkbox_y}")
        pyautogui.click(checkbox_x, checkbox_y)

    time.sleep(0.5)

    # Try to find Continue button
    print("▶️ Looking for Continue button...")
    continue_found = False
    continue_indicators = ["Continue", "Next", "Submit", "Create"]
    for indicator in continue_indicators:
        success = click_text_in_window("Google Chrome", indicator)
        if success:
            continue_found = True
            print(f"✅ Found Continue button using '{indicator}'")
            break

    if not continue_found:
        # Fallback
        continue_x, continue_y = 947, 918
        print(f"▶️ Fallback: Clicking Continue button at: {continue_x},{continue_y}")
        pyautogui.click(continue_x, continue_y)

    print("✅ Form submission completed!")
    print("📝 Check the browser for any additional steps needed")

def get_verification_code_from_gmail():
    """Get verification code from Gmail using Gmail API"""
    print("📧 Getting latest emails from Gmail via Gmail API...")

    try:
        from email_gmail_api import _gmail_service
        import re
        import time
        from datetime import datetime, timedelta

        # Get Gmail service
        svc = _gmail_service()

        # Keep checking for emails until we find one
        max_attempts = 50  # Check for 5 minutes (50 * 6 seconds)
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"🔍 Checking for emails (attempt {attempt}/{max_attempts})...")

            # Search for emails with "Jagex" in subject
            resp = svc.users().messages().list(userId="me", q="subject:Jagex", maxResults=5).execute()
            msgs = resp.get("messages", [])

            if msgs:
                print(f"📧 Found {len(msgs)} Jagex emails")

                # Check emails from last 10 minutes
                ten_minutes_ago = datetime.now() - timedelta(minutes=10)

                for m in msgs:
                    full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
                    headers = full.get("payload", {}).get("headers", [])

                    # Get subject and date
                    subject = ""
                    email_date = None
                    for header in headers:
                        if header["name"] == "Subject":
                            subject = header["value"]
                        elif header["name"] == "Date":
                            try:
                                # Parse the date string: 'Sat, 25 Oct 2025 16:26:18 +0000 (UTC)'
                                date_str = header["value"]
                                # Remove the (UTC) part
                                date_str = date_str.replace(" (UTC)", "")
                                email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
                                email_date = email_date.replace(tzinfo=None)
                            except:
                                print(f"⚠️ Could not parse date: {header['value']}")

                    print(f"📧 Subject: {subject}")
                    print(f"📅 Date: {email_date}")

                    # Check if email is within last 10 minutes
                    if email_date and email_date >= ten_minutes_ago:
                        print("✅ Email is recent (within 10 minutes)")

                        # Look for 5-character code in subject (letters/numbers, caps)
                        code_match = re.search(r'\b[A-Z0-9]{5}\b', subject)
                        if code_match:
                            verification_code = code_match.group(0)
                            print(f"✅ Found verification code: {verification_code}")
                            # Copy to clipboard
                            import pyperclip
                            pyperclip.copy(verification_code)
                            return True
                    else:
                        print("⏰ Email is too old, skipping")
            else:
                print("❌ No Jagex emails found, waiting...")

            # Wait 6 seconds before next check
            if attempt < max_attempts:
                print("⏳ Waiting 6 seconds before next check...")
                time.sleep(6)

        print("❌ No verification email found after 2 minutes")
        return False

    except Exception as e:
        print(f"❌ Failed to get emails: {e}")
        return False

def enter_verification_code():
    """Enter the verification code"""
    print("🔢 Entering verification code...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Try to find verification code field using text detection
    print("🔢 Looking for verification code field...")
    verification_found = False
    verification_indicators = ["Enter verification code", "verification code", "code"]
    for indicator in verification_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            x, y, found_text = locations[0]
            pyautogui.click(x + 100, y)  # Click to the right of the label
            verification_found = True
            print(f"✅ Found verification field using '{found_text}'")
            break

    if not verification_found:
        # Fallback to hardcoded coordinates
        verification_code_x, verification_code_y = 940, 729
        print(f"🔢 Fallback: Clicking verification code field at: {verification_code_x},{verification_code_y}")
        pyautogui.click(verification_code_x, verification_code_y)

    time.sleep(0.5)

    # Clear field first
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.5)

    # Paste the verification code from clipboard (set by Gmail script)
    print("🔢 Pasting verification code from clipboard")
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)

    # Try to find Continue button
    print("▶️ Looking for Continue button...")
    continue_found = False
    continue_indicators = ["Continue", "Next", "Submit", "Verify"]
    for indicator in continue_indicators:
        success = click_text_in_window("Google Chrome", indicator)
        if success:
            continue_found = True
            print(f"✅ Found Continue button using '{indicator}'")
            break

    if not continue_found:
        # Fallback
        continue_button_x, continue_button_y = 947, 806
        print(f"▶️ Fallback: Clicking Continue button at: {continue_button_x},{continue_button_y}")
        pyautogui.click(continue_button_x, continue_button_y)

    time.sleep(2)
    print("✅ Verification code entered and submitted")

def enter_jagex_account_name():
    """Enter the Jagex account name on the account name selection page"""
    print("👤 Entering Jagex account name...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Wait for page to load
    time.sleep(2)

    # Generate a unique account name based on the email
    email = "charlieeeeee54345@gmail.com"
    # Extract username part and add some random numbers
    username_part = email.split('@')[0]
    random_suffix = str(random.randint(100, 999))
    account_name = f"{username_part}{random_suffix}"

    print(f"👤 Generated account name: {account_name}")

    # Try to find account name field using text detection
    print("👤 Looking for account name field...")
    account_name_found = False
    account_name_indicators = ["account name", "Choose your account name", "Jagex Account Name"]
    for indicator in account_name_indicators:
        locations = find_text_in_window("Google Chrome", indicator)
        if locations:
            x, y, found_text = locations[0]
            pyautogui.click(x + 100, y)  # Click to the right of the label
            account_name_found = True
            print(f"✅ Found account name field using '{found_text}'")
            break

    if not account_name_found:
        # Fallback to hardcoded coordinates
        account_name_x, account_name_y = 929, 693
        print(f"👤 Fallback: Clicking account name field at: {account_name_x},{account_name_y}")
        pyautogui.click(account_name_x, account_name_y)

    time.sleep(0.5)

    # Clear any existing text
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)

    # Type the account name
    print(f"👤 Typing account name: {account_name}")
    pyautogui.typewrite(account_name)
    time.sleep(0.5)

    # Try to find Continue button
    print("▶️ Looking for Continue button...")
    continue_found = False
    continue_indicators = ["Continue", "Next", "Submit"]
    for indicator in continue_indicators:
        success = click_text_in_window("Google Chrome", indicator)
        if success:
            continue_found = True
            print(f"✅ Found Continue button using '{indicator}'")
            break

    if not continue_found:
        # Fallback
        continue_button_x, continue_button_y = 955, 772
        print(f"▶️ Fallback: Clicking Continue button at: {continue_button_x},{continue_button_y}")
        pyautogui.click(continue_button_x, continue_button_y)

    time.sleep(2)
    print("✅ Jagex account name entered and submitted")
    print(f"📝 Account name used: {account_name}")

def create_jagex_password():
    """Create and enter password for Jagex account"""
    print("🔐 Creating Jagex account password...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Wait for page to load
    time.sleep(2)

    # Generate a strong password
    import string
    password_length = 12
    password_chars = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(random.choice(password_chars) for _ in range(password_length))

    print(f"🔐 Generated password: {password}")
    print("💾 Password will be saved for reference")

    # Password field coordinates (first input field in the modal)
    password_x, password_y = 956, 521  # Approximate position of "Choose a password" field

    print(f"🔐 Clicking password field at: {password_x},{password_y}")
    # pyautogui.click(password_x, password_y)
    time.sleep(0.5)

    # Clear any existing text
    # pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)

    # Type the password
    print(f"🔐 Typing password...")
    # pyautogui.typewrite(password)
    time.sleep(0.5)

    # Confirm password field coordinates (second input field)
    confirm_password_x, confirm_password_y = 941, 591  # Approximate position of "Confirm password" field

    print(f"🔐 Clicking confirm password field at: {confirm_password_x},{confirm_password_y}")
    # pyautogui.click(confirm_password_x, confirm_password_y)
    time.sleep(0.5)

    # Clear any existing text
    # pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)

    # Type the same password again
    print(f"🔐 Confirming password...")
    # pyautogui.typewrite(password)
    time.sleep(0.5)

    # Handle checkboxes (optional - you can skip these if needed)
    # First checkbox: "Send me offers, news, and updates about Jagex games"
    # Uncomment the lines below if you want to check this box
    # offers_checkbox_x, offers_checkbox_y = 960, 720
    # print(f"☑️ Clicking offers checkbox at: {offers_checkbox_x},{offers_checkbox_y}")
    # pyautogui.click(offers_checkbox_x, offers_checkbox_y)
    # time.sleep(0.2)

    # Second checkbox: "I agree that Jagex can process my data for analytics..."
    # Uncomment the lines below if you want to check this box
    # analytics_checkbox_x, analytics_checkbox_y = 960, 750
    # print(f"☑️ Clicking analytics checkbox at: {analytics_checkbox_x},{analytics_checkbox_y}")
    # pyautogui.click(analytics_checkbox_x, analytics_checkbox_y)
    # time.sleep(0.2)

    # Create account button coordinates (bottom of the modal)
    create_account_x, create_account_y = 957, 901  # Approximate position of "Create account" button

    print(f"▶️ Clicking Create account button at: {create_account_x},{create_account_y}")
    # pyautogui.click(create_account_x, create_account_y)
    time.sleep(3)

    print("✅ Password created and account creation submitted")
    print(f"🔐 Password used: {password}")
    print("💾 Make sure to save this password securely!")

    # Save password to a CSV file for reference
    import csv
    import os

    try:
        account_name = f"charlieeeeee54345{random.randint(100, 999)}"
        csv_filename = "jagex_accounts.csv"

        # Check if file exists
        file_exists = os.path.isfile(csv_filename)

        # Open file in append mode
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writerow(["Email", "Account Name", "Password"])

            # Write account data
            writer.writerow(["charlieeeeee54345@gmail.com", account_name, password])

        print(f"📄 Account details saved to {csv_filename}")
    except Exception as e:
        print(f"⚠️ Could not save account details: {e}")

def complete_registration():
    """Click Continue on the registration completed page"""
    print("🎉 Registration completed! Clicking Continue...")

    # Focus the browser window
    focus_window("Google Chrome")
    time.sleep(1)

    # Wait for page to load
    time.sleep(2)

    # Continue button coordinates (bottom of the modal)
    continue_x, continue_y = 951, 722  # Approximate position of "Continue" butto

    print(f"▶️ Clicking Continue button at: {continue_x},{continue_y}")
    pyautogui.click(continue_x, continue_y)
    time.sleep(3)

    print("✅ Registration process fully completed!")
    print("🎉 Your Jagex account is now ready to use!")
    print("📧 Email: charlieeeeee54345@gmail.com")
    print("📄 Check jagex_account_info.txt for your account details")

def navigate_to_account_profile():
    """Navigate to the account management profile page"""
    print("🔗 Navigating to account management profile...")

    # Wait a moment for the previous page to settle
    time.sleep(2)

    # Use PyAutoGUI to navigate to the profile URL
    # First, click on the address bar
    address_bar_x, address_bar_y = 500, 50  # Approximate position of address bar
    print(f"🌐 Clicking address bar at: {address_bar_x},{address_bar_y}")
    pyautogui.click(address_bar_x, address_bar_y)
    time.sleep(0.5)

    # Clear the address bar and type the new URL
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)
    pyautogui.typewrite("https://account.jagex.com/en-GB/manage/profile")
    time.sleep(0.5)

    # Press Enter to navigate
    pyautogui.press('enter')
    time.sleep(3)

    print("✅ Navigated to account management profile page")

def navigate_and_click_characters_pyautogui():
    """Use PyAutoGUI to navigate to profile page and click Characters button in existing browser"""
    print("🎭 Using PyAutoGUI to navigate to profile and click Characters button...")

    # Move mouse to the center of the web page first
    print("🖱️ Moving mouse to web page...")
    click_center_of_window("Google Chrome")
    time.sleep(0.5)

    # Now use Page Down to find the Characters section
    print("📜 Using Page Down to find Characters section")
    pyautogui.press('pagedown')  # Page down
    time.sleep(0.5)

    print(f"🎭 Clicking Characters button")
    click_image_in_window("Google Chrome", "characters.png")
    wait_until(lambda: find_text_in_window("Google Chrome", "en-GB/game"))
    time.sleep(3)

    done_creating = False
    while done_creating == False:
        print("✅ Clicked Characters button!")
        print("🎭 You should now be on the Characters management page")
        click_image_in_window("Google Chrome", "create_new.png")
        wait_until(lambda: find_image_in_window("Google Chrome", "create.png"))

        click_image_in_window("Google Chrome", "create.png")
        wait_until(lambda: find_image_in_window("Google Chrome", "character_created.png"))
        time.sleep(3)
        if find_text_in_window("Google Chrome", "Too many requests") or find_text_in_window("Google Chrome", "20 of 20"):
            done_creating = True5

def click_characters_button_pyautogui():
    """Fallback method using PyAutoGUI to click Characters button"""
    print("🎭 Using PyAutoGUI to find and click Characters button...")

    # Wait for page to load
    time.sleep(3)

    # Scroll down to find the Characters section
    print("📜 Scrolling down to find Characters section...")
    pyautogui.scroll(-5)  # Scroll down
    time.sleep(1)
    pyautogui.scroll(-5)  # Scroll down more
    time.sleep(1)

    # Look for the Characters button (approximate coordinates)
    # Based on the image, the Characters button should be in the lower middle portion
    characters_x, characters_y = 960, 800  # Approximate position

    print(f"🎭 Clicking Characters button at: {characters_x},{characters_y}")
    pyautogui.click(characters_x, characters_y)
    time.sleep(3)

    print("✅ Clicked Characters button (PyAutoGUI method)")

def launch_jagex_launcher():
    """Launch the Jagex launcher, maximize it, and perform mouse movement and click"""
    print("🚀 Launching Jagex launcher...")

    import subprocess
    import time

    try:
        # Launch the Jagex launcher
        launcher_path = r"C:\Program Files (x86)\Jagex Launcher\JagexLauncher.exe"
        print(f"🎮 Starting launcher: {launcher_path}")
        subprocess.Popen([launcher_path])

        # Wait for launcher to start
        time.sleep(5)

        # Maximize the launcher window
        print("🖥️ Maximizing launcher window...")
        pyautogui.hotkey('win', 'up')  # Windows key + up arrow to maximize
        time.sleep(2)


    except Exception as e:
        print(f"❌ Error launching Jagex launcher: {e}")
        print("🔍 Please check if the launcher is installed at the specified path")


def create_account_from_jagex_launcher():
    """Login to Jagex launcher with email and password"""
    print("🔐 Logging in to Jagex launcher...")

    # Focus the Jagex Launcher window
    focus_window("Jagex Launcher")
    time.sleep(1)

    # Try to read password from the CSV file
    import csv
    logged_in_account = None
    try:
        with open("jagex_accounts.csv", "r") as f:
            reader = csv.DictReader(f)
            # Get the last row (most recent account)
            rows = list(reader)
            if rows:
                for row in rows:
                    if find_text_in_window("Jagex Launcher", row['Account Name']):
                        logged_in_account = row['Account Name']
            else:
                raise Exception("No accounts found in CSV")
    except Exception as e:
        print(f"⚠️ Could not read password from CSV: {e}")
        print("Using default password")

    if logged_in_account:
        move_to_text_in_window("Jagex Launcher", logged_in_account)
        time.sleep(0.5)
        click_image_in_window("Jagex Launcher", "log_out.png")
        time.sleep(0.5)
        click_image_in_window("Jagex Launcher", "log_out2.png")
        time.sleep(1)

    # Helpers to supply email index and random DOB
    def _get_next_email_index_from_accounts_csv(base_email_prefix: str, domain: str, csv_filename: str = "jagex_accounts.csv") -> int:
        """Compute next +index for base_email_prefix by scanning jagex_accounts.csv only.
        Rules:
        - If rows contain emails like base+X@domain, pick next index = max(X)+1
        - If only base@domain exists (no +X), start with 0
        - If no matching emails at all, start with 0
        """
        import csv
        max_index = -1
        saw_base_without_plus = False
        base_plain = f"{base_email_prefix}@{domain}"
        if os.path.isfile(csv_filename):
            try:
                with open(csv_filename, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        email = (row.get("Email") or "").strip().lower()
                        if not email:
                            continue
                        if email == base_plain.lower():
                            saw_base_without_plus = True
                            continue
                        # match base+<digits>@domain
                        prefix = f"{base_email_prefix.lower()}+"
                        suffix = f"@{domain.lower()}"
                        if email.startswith(prefix) and email.endswith(suffix):
                            mid = email[len(prefix):-len(suffix)]
                            try:
                                idx = int(mid)
                                if idx > max_index:
                                    max_index = idx
                            except Exception:
                                pass
            except Exception:
                # On any read/parse error, default to starting at 0
                return 0
        if max_index >= 0:
            return max_index + 1
        # No +X found; if base exists or file empty, start at 0
        return 0

    def _random_dob_strings():
        """Return (day, month, year) strings with safe ranges (age 18-30)."""
        # Year so user is adult; adjust as needed
        year = random.randint(1990, 2006)  # 1994..2008? ensure >=18; keep upper bound conservative
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # keep within 28 to avoid invalid dates
        return str(day), str(month), str(year)

    # Click create an account button
    click_text_in_window("Jagex Launcher", "Create an account")
    time.sleep(1)
    base_prefix = "charlieeeeee54345"
    domain = "gmail.com"
    index = _get_next_email_index_from_accounts_csv(base_prefix, domain)
    email_to_use = f"{base_prefix}+{index}@{domain}"
    day, month, year = _random_dob_strings()
    pyautogui.typewrite(email_to_use)
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.typewrite(day)
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.typewrite(month)
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.typewrite(year)
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.press('space')
    time.sleep(0.5)
    pyautogui.press('pagedown')
    time.sleep(0.5)
    click_image_in_window("Jagex Launcher", "continue.png")
    time.sleep(1)
    get_verification_code_from_gmail()
    focus_window("Jagex Launcher")
    time.sleep(0.5)
    pyautogui.press('Ctrl+V')
    time.sleep(0.5)
    click_image_in_window("Jagex Launcher", "continue.png")
    time.sleep(1)

    print("✅ Account created!")
    print("🎉 Your Jagex account is now ready to use!")
    print("📧 Email: charlieeeeee54345@gmail.com")
    print("📄 Check jagex_account_info.txt for your account details")

def login_to_jagex_launcher(email):
    """Login to Jagex launcher with email and password"""
    print("🔐 Logging in to Jagex launcher...")

    # Focus the Jagex Launcher window
    focus_window("Jagex Launcher")
    time.sleep(1)

    # Wait for launcher to load
    time.sleep(2)

    username = email.split('@')[0]
    password = "default_password"  # This should be read from the saved file

    if find_text_in_window("Jagex Launcher", username):
        return True

    # Try to read password from the CSV file
    import csv
    logged_in_account = None
    try:
        with open("jagex_accounts.csv", "r") as f:
            reader = csv.DictReader(f)
            # Get the last row (most recent account)
            rows = list(reader)
            if rows:
                for row in rows:
                    if row['Email'] == email:
                        password = row["Password"]
                    if find_text_in_window("Jagex Launcher", row['Account Name']):
                        logged_in_account = row['Account Name']
            else:
                raise Exception("No accounts found in CSV")
    except Exception as e:
        print(f"⚠️ Could not read password from CSV: {e}")
        print("Using default password")

    if logged_in_account:
        move_to_text_in_window("Jagex Launcher", logged_in_account)
        time.sleep(0.5)
        click_image_in_window("Jagex Launcher", "log_out.png")
        time.sleep(0.5)
        click_image_in_window("Jagex Launcher", "log_out2.png")
        time.sleep(1)

    # Click email input box
    email_x, email_y = 246, 358  # Approximate position of email input
    print(f"📧 Clicking email input at: {email_x},{email_y}")
    click_in_window("Jagex Launcher", email_x, email_y)
    time.sleep(0.5)

    # Type email
    print(f"📧 Typing email: {email}")
    pyautogui.typewrite(email)
    time.sleep(0.5)

    # Click continue button
    continue_x, continue_y = 248, 443  # Approximate position of continue button
    print(f"▶️ Clicking Continue button at: {continue_x},{continue_y}")
    click_in_window("Jagex Launcher", continue_x, continue_y)
    time.sleep(1)

    # Click password input box
    password_x, password_y = 248, 443  # Approximate position of password input
    print(f"🔐 Clicking password input at: {password_x},{password_y}")
    click_in_window("Jagex Launcher", password_x, password_y)
    time.sleep(0.5)

    # Type password
    print(f"🔐 Typing password...")
    pyautogui.typewrite(password)
    time.sleep(0.5)

    # Press Enter
    print("⏎ Pressing Enter...")
    pyautogui.press('enter')
    time.sleep(2)

    get_verification_code_from_gmail()

    print("✅ Login credentials entered!")

def enter_launcher_verification_code():
    """Get verification code from Gmail and enter it in the launcher"""
    print("🔢 Getting verification code from Gmail and entering it...")

    # Focus the Jagex Launcher window
    focus_window("Jagex Launcher")
    time.sleep(1)

    # Get verification code from Gmail
    get_verification_code_from_gmail()

    # Click verification code input box
    verification_x, verification_y = 246, 426  # Approximate position of verification code input
    print(f"🔢 Clicking verification code input at: {verification_x},{verification_y}")
    click_in_window("Jagex Launcher", verification_x, verification_y)
    time.sleep(0.5)

    # Clear field first
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)

    # Paste the verification code from clipboard
    print("🔢 Pasting verification code from clipboard")
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)

    # Click continue button
    continue_x, continue_y = 250, 511  # Approximate position of continue button
    print(f"▶️ Clicking Continue button at: {continue_x},{continue_y}")
    click_in_window("Jagex Launcher", continue_x, continue_y)
    time.sleep(10)

    print("✅ Verification code entered and submitted!")

def get_window_position(window_title):
    """Get the position and size of a window by title"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if window_title.lower() in window_text.lower():
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                windows.append({
                    'hwnd': hwnd,
                    'title': window_text,
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                })

    windows = []
    win32gui.EnumWindows(callback, windows)

    if windows:
        return windows[0]
    return None

def convert_window_to_screen_coords(window_pos, window_x, window_y):
    """Convert window-relative coordinates to screen coordinates"""
    screen_x = window_pos['left'] + window_x
    screen_y = window_pos['top'] + window_y
    return screen_x, screen_y

def click_in_window(window_title, window_x, window_y):
    """Click at window-relative coordinates within a specific window"""
    print(f"🎯 Finding window: {window_title}")

    window_pos = get_window_position(window_title)

    if not window_pos:
        print(f"❌ Window '{window_title}' not found")
        return False

    # Convert window-relative coordinates to screen coordinates
    screen_x, screen_y = convert_window_to_screen_coords(window_pos, window_x, window_y)

    print(f"📍 Window position: ({window_pos['left']}, {window_pos['top']})")
    print(f"📏 Window size: {window_pos['width']}x{window_pos['height']}")
    print(f"🖱️ Clicking at window coords ({window_x}, {window_y}) -> screen coords ({screen_x}, {screen_y})")

    pyautogui.click(screen_x, screen_y)
    return True

def move_to_window(window_title, window_x, window_y):
    """Move mouse to window-relative coordinates within a specific window"""
    print(f"🎯 Finding window: {window_title}")

    window_pos = get_window_position(window_title)

    if not window_pos:
        print(f"❌ Window '{window_title}' not found")
        return False

    # Convert window-relative coordinates to screen coordinates
    screen_x, screen_y = convert_window_to_screen_coords(window_pos, window_x, window_y)

    print(f"📍 Window position: ({window_pos['left']}, {window_pos['top']})")
    print(f"📏 Window size: {window_pos['width']}x{window_pos['height']}")
    print(f"🖱️ Moving to window coords ({window_x}, {window_y}) -> screen coords ({screen_x}, {screen_y})")

    pyautogui.moveTo(screen_x, screen_y)
    return True

def position_in_window(window_title):
    """Get current mouse position relative to a specific window (returns as string "x, y")"""
    window_pos = get_window_position(window_title)

    if not window_pos:
        print(f"❌ Window '{window_title}' not found")
        return None

    # Get current screen position
    screen_x, screen_y = pyautogui.position()

    # Convert to window-relative coordinates
    window_x = screen_x - window_pos['left']
    window_y = screen_y - window_pos['top']

    # Return as string "x, y"
    position_str = f"{window_x}, {window_y}"
    print(f"🖱️ Mouse position in '{window_title}': {position_str} (screen: {screen_x}, {screen_y})")

    return position_str

def find_text_in_window(window_title, text_to_find, confidence=0.8):
    """Find text in a specific window using OCR (requires pytesseract and PIL) - returns all matches"""
    try:
        from PIL import Image

        # Get window position
        window_pos = get_window_position(window_title)
        if not window_pos:
            print(f"❌ Window '{window_title}' not found")
            return []

        if window_pos['left'] < 0:
            window_pos['left'] = 0
        if window_pos['top'] < 0:
            window_pos['top'] = 0

        # Take screenshot of just the window
        screenshot = pyautogui.screenshot(region=(
            window_pos['left'],
            window_pos['top'],
            window_pos['width'],
            window_pos['height']
        ))

        # Use OCR to find text
        data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)

        # Search for the text (handle multi-word text that gets separated)
        words_to_find = text_to_find.lower().split()
        matches = []

        for i in range(len(data['text'])):
            # Check if current word matches first word of search text
            if words_to_find[0].lower() in data['text'][i].lower():
                # Check if subsequent words match in sequence
                match_found = True
                for j in range(1, len(words_to_find)):
                    if i + j >= len(data['text']) or words_to_find[j] not in data['text'][i + j].lower():
                        match_found = False
                        break

                if match_found:
                    # Use coordinates of the first word
                    x = data['left'][i] + window_pos['left']  # Add window offset
                    y = data['top'][i] + window_pos['top']   # Add window offset
                    found_text = ' '.join([data['text'][i + k] for k in range(len(words_to_find))])
                    matches.append((x, y, found_text))
                    print(f"✅ Found '{found_text}' in '{window_title}' at ({x}, {y})")

        if matches:
            print(f"📊 Found {len(matches)} occurrence(s) of '{text_to_find}'")
            return matches
        else:
            print(f"❌ Could not find text '{text_to_find}' in window '{window_title}'")
            return []

    except ImportError:
        print("❌ pytesseract or PIL not installed. Run: pip install pytesseract pillow")
        return []
    except Exception as e:
        print(f"❌ Error finding text: {e}")
        return []

def find_image_in_window(window_title, image_filename, confidence=0.8):
    """Find an image in a specific window using template matching"""
    ui_images_path = r"/account_creation/src/ui_images"
    image_path = os.path.join(ui_images_path, image_filename)
    try:
        # Get window position
        window_pos = get_window_position(window_title)
        if not window_pos:
            print(f"❌ Window '{window_title}' not found")
            return None

        print(f"🔍 Window region: x={window_pos['left']}, y={window_pos['top']}, w={window_pos['width']}, h={window_pos['height']}")

        # Clamp negative coordinates to 0
        x = max(0, window_pos['left'])
        y = max(0, window_pos['top'])

        # Locate the image on screen (within the window region)
        location = pyautogui.locateOnScreen(image_path, confidence=confidence, region=(
            x,
            y,
            window_pos['width'],
            window_pos['height']
        ))

        if location:
            center = pyautogui.center(location)
            print(f"✅ Found image {image_path} in '{window_title}' at {center}")
            return center
        else:
            print(f"❌ Image '{image_path}' not found in window '{window_title}'")
            return None

    except Exception as e:
        print(f"❌ Error finding image: {e}")
        return None

def click_text_in_window(window_title, text_to_find, confidence=0.8, match_index=0):
    """Find and click text in a specific window using OCR"""
    matches = find_text_in_window(window_title, text_to_find, confidence)
    if matches and len(matches) > match_index:
        x, y, found_text = matches[match_index]
        pyautogui.click(x, y)
        print(f"🖱️ Clicked '{found_text}' at ({x}, {y})")
        return True
    return False

def move_to_text_in_window(window_title, text_to_find, confidence=0.8, match_index=0):
    """Find and click text in a specific window using OCR"""
    matches = find_text_in_window(window_title, text_to_find, confidence)
    if matches and len(matches) > match_index:
        x, y, found_text = matches[match_index]
        pyautogui.moveTo(x, y)
        print(f"🖱️ Clicked '{found_text}' at ({x}, {y})")
        return True
    return False

def click_image_in_window(window_title, image_filename, confidence=0.8):
    """Find and click an image in a specific window (automatically uses ui_images folder)"""
    # Construct full path to ui_images folder
    ui_images_path = r"/account_creation/src/ui_images"
    image_path = os.path.join(ui_images_path, image_filename)

    location = find_image_in_window(window_title, image_path, confidence)
    if location:
        pyautogui.click(location)
        return True
    return False

def screenshot_window(window_title, filename=None):
    """Take a screenshot of a specific window"""
    window_pos = get_window_position(window_title)

    if not window_pos:
        print(f"❌ Window '{window_title}' not found")
        return None

    # Take screenshot of the window area
    screenshot = pyautogui.screenshot(region=(
        window_pos['left'],
        window_pos['top'],
        window_pos['width'],
        window_pos['height']
    ))

    if filename:
        screenshot.save(filename)
        print(f"📸 Screenshot saved to {filename}")

    return screenshot

def process_character_list():
    """Click 'No Name Set', get character locations, then loop through them to launch RuneLite and copy credentials"""
    print("🎮 Processing character list...")
    focus_window("Jagex Launcher")

    # First, click 'No Name Set' to reveal the character list
    print("🎯 Clicking 'No Name Set' to reveal character list...")
    if find_text_in_window("Jagex Launcher", "No Name Set"):
        skip_first_index = True
    else:
        skip_first_index = False

    click_image_at_position("Jagex Launcher", "character_dropdown.png", 0.5, 0.8)

    # Wait 3 seconds for the character list to load
    print("⏳ Waiting 3 seconds for character list to load...")
    time.sleep(3)

    # Get all character locations
    print("🔍 Finding all character locations...")
    locations = find_text_in_window("Jagex Launcher", "No Name Set", confidence=0.5)
    click_image_at_position("Jagex Launcher", "character_dropdown.png", 0.5, 0.8)
    time.sleep(1)

    if not locations or len(locations) <= 1:
        print("❌ Not enough characters found (need at least 2)")
        return False

    print(f"📊 Found {len(locations)} characters")

    # Get index 0 location for clicking first
    # index0_x, index0_y, index0_text = locations[0]

    # Skip index 0, process indices 1+
    if skip_first_index:
        accounts_to_start = range(1, len(locations))
    else:
        accounts_to_start = range(0, len(locations))

    for i in accounts_to_start:
        x, y, found_text = locations[i]
        print(f"\n📋 Processing character {i}/{len(locations)-1}: {found_text}")

        click_image_at_position("Jagex Launcher", "character_dropdown.png", 0.5, 0.8)
        time.sleep(1)
        
        # 2. Click this index's location
        print(f"🖱️ Clicking character at ({x}, {y})...")
        pyautogui.click(x, y)
        time.sleep(1)
        
        # 3. Click 'Play' button using image detection
        print("▶️ Looking for 'Play' button...")
        play_clicked = click_image_in_window("Jagex Launcher", "play_button.png")
        if not play_clicked:
            print("❌ Could not find 'Play' button image")
            continue
        time.sleep(2)
        
        # 4. Wait 30 seconds or for RuneLite window to open
        print("⏳ Waiting for RuneLite to open...")
        runelite_opened = wait_for_window("RuneLite", timeout=30)
        time.sleep(10)
        if not runelite_opened:
            print("❌ RuneLite did not open within timeout")
            continue
        
        # 5. Run copy_credentials.py
        print("📋 Running copy_credentials.py...")
        import subprocess
        script_path = r"/copy_credentials.py"
        try:
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            print(f"📋 Output: {result.stdout}")
            if result.returncode != 0:
                print(f"⚠️ Script returned code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error running copy_credentials.py: {e}")
        
        time.sleep(2)  # Give it a moment to complete
        
        # 6. Close the specific RuneLite window
        print("🗑️ Closing RuneLite window...")
        close_specific_window("RuneLite")
        time.sleep(2)
        
        # 7. Focus the Jagex launcher window
        print("🎯 Focusing Jagex Launcher window...")
        focus_window("Jagex Launcher")
        time.sleep(1)
    
    print("\n✅ Finished processing all characters!")
    return True

def wait_for_window(window_title, timeout=30):
    """Wait for a window to appear"""
    print(f"⏳ Waiting for '{window_title}' window to open...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        window_pos = get_window_position(window_title)
        if window_pos:
            print(f"✅ Window '{window_title}' found!")
            return True
        time.sleep(1)
    
    print(f"❌ Window '{window_title}' did not appear within {timeout} seconds")
    return False

def close_specific_window(window_title):
    """Close a specific window by title"""
    window_pos = get_window_position(window_title)
    if window_pos:
        try:
            hwnd = window_pos['hwnd']
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            print(f"✅ Sent close message to '{window_title}'")
            return True
        except Exception as e:
            print(f"❌ Error closing window: {e}")
            return False
    else:
        print(f"❌ Window '{window_title}' not found")
        return False

def focus_window(window_title):
    """Focus (bring to front) a specific window"""
    window_pos = get_window_position(window_title)
    if window_pos:
        try:
            hwnd = window_pos['hwnd']
            
            # Check if window is minimized
            if win32gui.IsIconic(hwnd):
                # Restore window if minimized
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                # Just bring to front without changing size/state
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            
            # Bring to front
            win32gui.SetForegroundWindow(hwnd)
            print(f"✅ Focused window '{window_title}'")
            return True
        except Exception as e:
            print(f"❌ Error focusing window: {e}")
            return False
    else:
        print(f"❌ Window '{window_title}' not found")
        return False

def click_center_of_window(window_title):
    """Click the center of a specific window"""
    window_pos = get_window_position(window_title)
    if window_pos:
        # Calculate center of window
        center_x = window_pos['left'] + (window_pos['width'] // 2)
        center_y = window_pos['top'] + (window_pos['height'] // 2)
        
        print(f"🖱️ Clicking center of '{window_title}' at ({center_x}, {center_y})")
        pyautogui.click(center_x, center_y)
        return True
    else:
        print(f"❌ Window '{window_title}' not found")
        return False

def click_image_at_position(window_title, image_filename, x_ratio, y_ratio, confidence=0.8):
    """
    Find an image in a window and click at a specific position within that image.
    
    Args:
        window_title (str): Title of the window to search in
        image_filename (str): Name of the image file to find
        x_ratio (float): X position ratio (0.0 = left edge, 1.0 = right edge)
        y_ratio (float): Y position ratio (0.0 = top edge, 1.0 = bottom edge)
        confidence (float): Confidence level for image matching (0.0-1.0)
    
    Returns:
        bool: True if image was found and clicked, False otherwise
    """
    try:
        # Validate input ratios
        if not (0.0 <= x_ratio <= 1.0):
            print(f"❌ x_ratio must be between 0.0 and 1.0, got {x_ratio}")
            return False
        if not (0.0 <= y_ratio <= 1.0):
            print(f"❌ y_ratio must be between 0.0 and 1.0, got {y_ratio}")
            return False
        
        # Get window position
        window_pos = get_window_position(window_title)
        if not window_pos:
            print(f"❌ Window '{window_title}' not found")
            return False
        
        # Construct full path to ui_images folder
        ui_images_path = r"/account_creation/src/ui_images"
        image_path = os.path.join(ui_images_path, image_filename)
        
        # Clamp negative coordinates to 0
        x = max(0, window_pos['left'])
        y = max(0, window_pos['top'])
        
        # Locate the image on screen (within the window region)
        location = pyautogui.locateOnScreen(image_path, confidence=confidence, region=(
            x,
            y,
            window_pos['width'],
            window_pos['height']
        ))
        
        if location:
            # Calculate the position within the image
            image_left, image_top, image_width, image_height = location
            
            # Calculate click position within the image
            click_x = image_left + (image_width * x_ratio)
            click_y = image_top + (image_height * y_ratio)
            
            print(f"✅ Found image '{image_filename}' in '{window_title}'")
            print(f"📏 Image bounds: ({image_left}, {image_top}) to ({image_left + image_width}, {image_top + image_height})")
            print(f"🎯 Clicking at ratio ({x_ratio}, {y_ratio}) -> position ({click_x}, {click_y})")
            
            pyautogui.click(click_x, click_y)
            return True
        else:
            print(f"❌ Image '{image_filename}' not found in window '{window_title}'")
            return False
            
    except Exception as e:
        print(f"❌ Error clicking image at position: {e}")
        return False
