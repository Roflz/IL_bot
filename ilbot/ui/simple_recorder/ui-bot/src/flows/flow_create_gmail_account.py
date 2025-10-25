from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from ..ui_helpers import safe_click, safe_type, wait_visible, wait_clickable
import time
import random
import string
import pyautogui
import subprocess
import os
from ..jagex_automation import focus_window, find_text_in_window, click_text_in_window, click_image_in_window, \
    get_window_position, click_center_of_window, find_image_in_window


class GmailSignupSelectors:
    # Step 1: Name entry
    FIRST_NAME = (By.NAME, "firstName")
    LAST_NAME = (By.NAME, "lastName") 
    NEXT_BUTTON = (By.ID, "collectNameNext")
    
    # Step 2: Birthday and gender (using more flexible selectors)
    BIRTH_MONTH = (By.CSS_SELECTOR, "[aria-label*='month'], #month, [data-value*='month']")
    BIRTH_DAY = (By.CSS_SELECTOR, "[aria-label*='day'], #day, input[placeholder*='day']")
    BIRTH_YEAR = (By.CSS_SELECTOR, "[aria-label*='year'], #year, input[placeholder*='year']")
    GENDER = (By.CSS_SELECTOR, "[aria-label*='gender'], #gender, [data-value*='gender']")
    NEXT_BUTTON_BIRTHDAY = (By.ID, "birthdaygenderNext")
    
    # Step 3: Username creation
    USERNAME_INPUT = (By.ID, "username")
    NEXT_BUTTON_USERNAME = (By.ID, "accountDetailsNext")
    
    # Step 4: Password creation
    PASSWORD_INPUT = (By.NAME, "Passwd")
    CONFIRM_PASSWORD_INPUT = (By.NAME, "ConfirmPasswd")
    SHOW_PASSWORD_CHECKBOX = (By.CSS_SELECTOR, "input[type='checkbox'][aria-label='Show password']")
    NEXT_BUTTON_PASSWORD = (By.ID, "createpasswordNext")
    
    # Step 5: Recovery email (optional)
    RECOVERY_EMAIL_INPUT = (By.ID, "recoveryEmailId")
    SKIP_BUTTON = (By.ID, "skipRecoveryEmail")
    NEXT_BUTTON_RECOVERY = (By.ID, "recoveryEmailNext")
    
    # Step 6: Account confirmation
    NEXT_BUTTON_CONFIRM = (By.ID, "confirmNext")
    
    # Step 7: Privacy and Terms
    I_AGREE_BUTTON = (By.CSS_SELECTOR, "button[data-action='accept']")
    
    # Step 8: Personalization
    CONFIRM_PERSONALIZATION = (By.CSS_SELECTOR, "button[data-action='confirm']")
    
    # OAuth Authorization (after account creation)
    CONTINUE_OAUTH = (By.CSS_SELECTOR, "button[data-action='continue']")
    ADVANCED_LINK = (By.CSS_SELECTOR, "a[href*='advanced']")
    GO_TO_APP_LINK = (By.CSS_SELECTOR, "a[href*='unsafe']")

def generate_random_username(base_name: str, length: int = 8) -> str:
    """Generate a completely random username"""
    # Generate random letters and numbers
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{base_name}{random_chars}"

def create_gmail_account(
    driver: WebDriver,
    first_name: str,
    last_name: str,
    base_username: str,
    password: str,
    recovery_email: str = None,
    birth_month: str = "February",
    birth_day: str = "12", 
    birth_year: str = "1989",
    gender: str = "Male"
) -> str:
    """
    Automates Gmail account creation and returns the created email address
    """
    print(f"🚀 Starting Gmail account creation for {first_name} {last_name}")
    
    # Navigate to Gmail signup
    driver.get("https://accounts.google.com/signup")
    time.sleep(2)
    
    # Step 1: Enter name
    print("📝 Step 1: Entering name...")
    wait_visible(driver, GmailSignupSelectors.FIRST_NAME)
    safe_type(driver, GmailSignupSelectors.FIRST_NAME, first_name)
    safe_type(driver, GmailSignupSelectors.LAST_NAME, last_name)
    safe_click(driver, GmailSignupSelectors.NEXT_BUTTON)
    time.sleep(2)
    
    # Step 2: Birthday and gender
    print("🎂 Step 2: Entering birthday and gender...")
    wait_visible(driver, GmailSignupSelectors.BIRTH_MONTH)
    
    # Click and select birth month (custom dropdown)
    print(f"   Selecting month: {birth_month}")
    safe_click(driver, GmailSignupSelectors.BIRTH_MONTH)
    
    # Just click February (index 1) directly
    month_selected = False
    try:
        all_options = driver.find_elements(By.XPATH, "//div[@role='option'] | //li | //option | //div[contains(@class, 'option')]")
        if len(all_options) > 1:  # Make sure we have at least 2 options (January=0, February=1)
            print(f"   Clicking February (index 1)")
            all_options[1].click()  # February is index 1
            month_selected = True
            print(f"   ✅ Selected February")
        else:
            print(f"   ⚠️ Not enough options found, trying first option")
            all_options[0].click()
            month_selected = True
    except Exception as e:
        print(f"   ❌ Month selection failed: {e}")
    
    if not month_selected:
        print(f"   ⚠️ Could not select month, trying to type it...")
        safe_type(driver, GmailSignupSelectors.BIRTH_MONTH, "February")

    time.sleep(0.5)
    # Enter birth day and year
    print(f"   Entering day: {birth_day}, year: {birth_year}")
    safe_type(driver, GmailSignupSelectors.BIRTH_DAY, birth_day)
    safe_type(driver, GmailSignupSelectors.BIRTH_YEAR, birth_year)
    
    # Click and select gender (custom dropdown)
    print(f"   Selecting gender...")
    safe_click(driver, GmailSignupSelectors.GENDER)
    
    # Just click the first available gender option (index 0)
    gender_selected = False
    try:
        # Wait a moment for the gender dropdown to fully load
        time.sleep(0.5)
        
        # Debug: Print all available options
        print("   🔍 Debugging gender options:")
        all_options = driver.find_elements(By.XPATH, "//div[@role='option'] | //li | //option | //div[contains(@class, 'option')]")
        print(f"   Found {len(all_options)} total options")
        
        # Print first 10 options to see what's available
        for i, option in enumerate(all_options[:10]):
            try:
                text = option.text.strip()
                if text:
                    print(f"      Option {i}: '{text}'")
            except:
                pass
        
        # Try multiple approaches to find gender options
        gender_options = []
        
        # Approach 3: Just try clicking any option that's clickable
        print("   ⚠️ No specific gender options found, trying any clickable option...")
        for option in all_options:
            try:
                if option.is_displayed() and option.is_enabled():
                    gender_options.append(option)
                    break  # Just take the first clickable one
            except:
                pass
        
        if gender_options:
            print(f"   Found {len(gender_options)} gender options, clicking first one")
            gender_options[0].click()  # Just click the first gender option
            gender_selected = True
            print(f"   ✅ Selected gender option")
        else:
            print("   ⚠️ No gender options found")
    except Exception as e:
        print(f"   ❌ Gender selection failed: {e}")
    
    if not gender_selected:
        print(f"   ⚠️ Could not select gender, trying to type it...")
        safe_type(driver, GmailSignupSelectors.GENDER, "Male")
    
    time.sleep(1)
    
    safe_click(driver, GmailSignupSelectors.NEXT_BUTTON_BIRTHDAY)
    time.sleep(2)
    
    # Step 3: Choose username
    print("📧 Step 3: Creating username...")
    
    # Check if we're on the "Choose your Gmail address" page first
    try:
        page_source = driver.page_source.lower()
        if "choose your gmail address" in page_source or "pick a gmail address" in page_source:
            print("   📧 Found Gmail address selection page, choosing 'Create your own'...")
            
            # Look for "Create your own Gmail address" radio button
            create_own_selectors = [
                "//label[contains(text(), 'Create your own Gmail address')]",
                "//input[@type='radio' and contains(@value, 'custom')]",
                "//input[@type='radio' and contains(@aria-label, 'Create your own')]",
                "//div[contains(text(), 'Create your own Gmail address')]"
            ]
            
            create_own_clicked = False
            for selector in create_own_selectors:
                try:
                    element = driver.find_element(By.XPATH, selector)
                    element.click()
                    print("   ✅ Clicked 'Create your own Gmail address'")
                    create_own_clicked = True
                    break
                except:
                    continue
            
            if not create_own_clicked:
                print("   ⚠️ Could not find 'Create your own' option, trying to click any radio button...")
                # Try to click any unselected radio button
                radio_buttons = driver.find_elements(By.XPATH, "//input[@type='radio']")
                for radio in radio_buttons:
                    try:
                        if not radio.is_selected():
                            radio.click()
                            print("   ✅ Clicked alternative radio button")
                            break
                    except:
                        continue
            
            # Click Next button
            try:
                next_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')] | //button[contains(text(), 'next')]")
                next_button.click()
                print("   ✅ Clicked Next button")
                time.sleep(2)
            except:
                print("   ⚠️ Could not find Next button")
                
    except Exception as e:
        print(f"   ⚠️ Could not handle Gmail address selection: {e}")
    
    # Now try to find the username input field
    username_input = None
    try:
        username_input = driver.find_element(*GmailSignupSelectors.USERNAME_INPUT)
        print("   Found username input with original selector")
    except:
        print("   Original username selector failed, trying alternatives...")
        username_selectors = [
            "input[type='text']",
            "input[placeholder*='username']",
            "input[placeholder*='email']", 
            "input[name*='username']",
            "input[name*='email']",
            "input[id*='username']",
            "input[id*='email']",
            "input[aria-label*='username']",
            "input[aria-label*='email']"
        ]
        
        for selector in username_selectors:
            try:
                username_input = driver.find_element(By.CSS_SELECTOR, selector)
                print(f"   Found username input with: {selector}")
                break
            except:
                continue
    
    if not username_input:
        print("   ⚠️ Could not find username input, waiting for any input field...")
        time.sleep(3)  # Wait a bit more
        try:
            username_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
            print("   Found username input after waiting")
        except:
            print("   ❌ Still no username input found")
            return None
    
    # Generate completely random usernames
    max_attempts = 5
    
    for attempt in range(max_attempts):
        # Generate a completely random username
        username = generate_random_username("user", 8)  # Just use "user" as base, will add random numbers
        print(f"   Trying username: {username}")
        
        # Clear and type the username
        username_input.clear()
        username_input.send_keys(username)
        time.sleep(0.5)


        try:
            all_buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in all_buttons:
                try:
                    if button.is_displayed() and button.is_enabled():
                        button_text = button.text.strip().lower()
                        if any(word in button_text for word in ['next', 'continue', 'proceed', 'submit']):
                            button.click()
                            print(f"   ✅ Clicked button with text: '{button.text.strip()}'")
                            next_clicked = True
                            break
                except:
                    continue
        except:
            pass
        
        if not next_clicked:
            print("   ❌ Could not find any next button")
        
        time.sleep(2)
        
        # Check if username was accepted (no error message)
        try:
            # Look for error message indicating username is taken
            error_element = driver.find_element(By.CSS_SELECTOR, "[data-error-message*='taken']")
            if error_element.is_displayed():
                print(f"❌ Username '{username}' is taken, trying another...")
                username = generate_random_username(base_username)
                continue
        except:
            # No error found, username was accepted
            break
    
    if attempt == max_attempts - 1:
        raise Exception("Could not find available username after multiple attempts")
    
    created_email = f"{username}@gmail.com"
    print(f"✅ Username created: {created_email}")
    
    # Step 4: Create password
    print("🔒 Step 4: Creating password...")
    
    # Find password input with multiple approaches
    password_input = None
    try:
        password_input = driver.find_element(*GmailSignupSelectors.PASSWORD_INPUT)
        print("   Found password input with original selector")
    except:
        print("   Original password selector failed, trying alternatives...")
        password_selectors = [
            "input[type='password']",
            "input[name*='password']",
            "input[name*='Passwd']",
            "input[id*='password']",
            "input[placeholder*='password']"
        ]
        
        for selector in password_selectors:
            try:
                password_input = driver.find_element(By.CSS_SELECTOR, selector)
                print(f"   Found password input with: {selector}")
                break
            except:
                continue
    
    if password_input:
        password_input.clear()
        password_input.send_keys(password)
        print("   ✅ Entered password")
    else:
        print("   ❌ Could not find password input")
        return None
    
    # Find confirm password input with multiple approaches
    confirm_password_input = None
    try:
        confirm_password_input = driver.find_element(*GmailSignupSelectors.CONFIRM_PASSWORD_INPUT)
        print("   Found confirm password input with original selector")
    except:
        print("   Original confirm password selector failed, trying alternatives...")
        
        # Try to find all password inputs and use the second one
        all_password_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        if len(all_password_inputs) >= 2:
            confirm_password_input = all_password_inputs[1]  # Use the second password input
            print("   Found confirm password input (second password field)")
        else:
            # Try other selectors
            confirm_selectors = [
                "input[name*='confirm']",
                "input[name*='ConfirmPasswd']",
                "input[id*='confirm']",
                "input[placeholder*='confirm']"
            ]
            
            for selector in confirm_selectors:
                try:
                    confirm_password_input = driver.find_element(By.CSS_SELECTOR, selector)
                    print(f"   Found confirm password input with: {selector}")
                    break
                except:
                    continue
    
    if confirm_password_input:
        confirm_password_input.clear()
        confirm_password_input.send_keys(password)
        print("   ✅ Entered confirm password")
    else:
        print("   ❌ Could not find confirm password input")
        return None
    
    # Try to find and click the next button
    try:
        next_button = driver.find_element(*GmailSignupSelectors.NEXT_BUTTON_PASSWORD)
        next_button.click()
        print("   ✅ Clicked password next button")
    except:
        print("   ⚠️ Could not find password next button, trying alternatives...")
        try:
            next_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')] | //button[contains(text(), 'next')] | //input[@type='submit']")
            next_button.click()
            print("   ✅ Clicked alternative next button")
        except:
            print("   ❌ Could not find any next button")
    
    time.sleep(2)
    
    # Check if we hit the QR code verification screen
    print("🔍 Checking for QR code verification...")
    try:
        # Look for QR code related text on the page
        page_source = driver.page_source.lower()
        qr_indicators = [
            "qr code",
            "qr verification", 
            "scan the qr code",
            "use your phone",
            "verify your identity",
            "security check",
            "phone verification"
        ]
        
        qr_detected = any(indicator in page_source for indicator in qr_indicators)
        
        if qr_detected:
            print("🚨 QR CODE DETECTED!")
            print("📱 Please complete the QR code verification manually in the browser")
            print("⏳ Waiting for you to complete verification...")
            print("   Press ENTER in the terminal when you're done...")
            
            # Wait for user input
            input("   (Press ENTER to continue after completing QR verification)")
            print("✅ Continuing with automation...")
        else:
            print("✅ No QR code detected, continuing normally...")
            
    except Exception as e:
        print(f"⚠️ Could not check for QR code: {e}")
        print("   Continuing anyway...")
    
    # Step 5: Recovery email (optional)
    print("📧 Step 5: Recovery email...")
    try:
        wait_visible(driver, GmailSignupSelectors.RECOVERY_EMAIL_INPUT, timeout=5)
        if recovery_email:
            safe_type(driver, GmailSignupSelectors.RECOVERY_EMAIL_INPUT, recovery_email)
            safe_click(driver, GmailSignupSelectors.NEXT_BUTTON_RECOVERY)
        else:
            safe_click(driver, GmailSignupSelectors.SKIP_BUTTON)
        time.sleep(2)
    except:
        print("⏭️ Recovery email step skipped")
    
    # Step 6: Account confirmation
    print("✅ Step 6: Confirming account...")
    wait_visible(driver, GmailSignupSelectors.NEXT_BUTTON_CONFIRM)
    safe_click(driver, GmailSignupSelectors.NEXT_BUTTON_CONFIRM)
    time.sleep(2)
    
    # Step 7: Privacy and Terms
    print("📋 Step 7: Accepting Privacy and Terms...")
    wait_visible(driver, GmailSignupSelectors.I_AGREE_BUTTON)
    safe_click(driver, GmailSignupSelectors.I_AGREE_BUTTON)
    time.sleep(2)
    
    # Step 8: Personalization (if appears)
    try:
        print("🎯 Step 8: Handling personalization...")
        wait_visible(driver, GmailSignupSelectors.CONFIRM_PERSONALIZATION, timeout=5)
        safe_click(driver, GmailSignupSelectors.CONFIRM_PERSONALIZATION)
        time.sleep(2)
    except:
        print("⏭️ Personalization step skipped")
    
    print(f"🎉 Gmail account creation completed: {created_email}")
    return created_email

def handle_oauth_authorization(driver: WebDriver) -> bool:
    """
    Handles the OAuth authorization flow after account creation
    Returns True if authorization was successful
    """
    print("🔐 Handling OAuth authorization...")
    
    try:
        # Wait for OAuth warning page
        wait_visible(driver, GmailSignupSelectors.ADVANCED_LINK, timeout=10)
        
        # Click "Advanced" to bypass warning
        safe_click(driver, GmailSignupSelectors.ADVANCED_LINK)
        time.sleep(2)
        
        # Click "Go to UI Bot (unsafe)" to continue
        wait_visible(driver, GmailSignupSelectors.GO_TO_APP_LINK)
        safe_click(driver, GmailSignupSelectors.GO_TO_APP_LINK)
        time.sleep(3)
        
        # Check if we reached the success page
        success_text = "The authentication flow has completed"
        if success_text in driver.page_source:
            print("✅ OAuth authorization completed successfully!")
            return True
        else:
            print("❌ OAuth authorization may have failed")
            return False
            
    except Exception as e:
        print(f"❌ OAuth authorization failed: {e}")
        return False

def open_gmail_signup_in_browser():
    """Open Gmail signup page in the default browser"""
    print("🌐 Opening Gmail signup page in browser...")
    
    try:
        # Open Gmail signup URL in default browser
        url = "https://accounts.google.com/signup"
        subprocess.run(["start", url], shell=True)
        time.sleep(3)
        
        print("✅ Gmail signup page opened in browser")
        return True
    except Exception as e:
        print(f"❌ Failed to open Gmail signup page: {e}")
        return False

def create_gmail_account_pyautogui(
    first_name: str,
    last_name: str,
    base_username: str,
    password: str,
    recovery_email: str = None,
    birth_month: str = "February",
    birth_day: str = "12", 
    birth_year: str = "1989",
    gender: str = "Male"
) -> str:
    """
    Automates Gmail account creation using PyAutoGUI and returns the created email address
    """
    print(f"🚀 Starting Gmail account creation with PyAutoGUI for {first_name} {last_name}")
    
    # Wait a moment and find the browser window
    time.sleep(2)
    
    # Use the specific window title for Gmail signup
    browser_window = "Google Chrome"
    
    # Focus the browser window
    focus_window(browser_window)
    time.sleep(2)
    
    # # Step 1: Enter name
    # print("📝 Step 1: Entering name...")
    #
    # # Click Use another account
    # click_text_in_window(browser_window, "Use another account")
    # time.sleep(0.5)
    #
    # if find_image_in_window(browser_window, "robot.png"):
    #     print('captcha appeared')
    #     click_image_in_window(browser_window, "robot.png")
    #     time.sleep(1)
    #
    # click_center_of_window(browser_window)
    # time.sleep(0.5)
    #
    # click_image_in_window(browser_window, "create_account.png")
    # time.sleep(1)

    if find_image_in_window(browser_window, "robot.png"):
        print('captcha appeared')
        click_image_in_window(browser_window, "robot.png")
        time.sleep(1)

    # browser_window = "Create your Google Account - Google Chrome"
    click_text_in_window(browser_window, "First name")
    time.sleep(0.5)

    pyautogui.typewrite(first_name)
    time.sleep(0.5)
    
    # Click Next button
    click_image_in_window(browser_window, "next.png")
    time.sleep(1)

    if find_image_in_window(browser_window, "robot.png"):
        print('captcha appeared')
        click_image_in_window(browser_window, "robot.png")
        time.sleep(1)

    click_image_in_window(browser_window, "month.png")
    time.sleep(0.5)

    click_image_in_window(browser_window, "february.png")
    time.sleep(0.5)

    click_image_in_window(browser_window, "day.png")
    time.sleep(0.5)

    pyautogui.typewrite(birth_day)
    time.sleep(0.5)

    click_image_in_window(browser_window, "year.png")
    time.sleep(0.5)

    pyautogui.typewrite(birth_year)
    time.sleep(0.5)

    click_image_in_window(browser_window, "gender.png")
    time.sleep(0.5)

    click_image_in_window(browser_window, "male.png")
    time.sleep(0.5)

    click_image_in_window(browser_window, "next.png")
    time.sleep(1)

    if find_image_in_window(browser_window, "robot.png"):
        print('captcha appeared')
        click_image_in_window(browser_window, "robot.png")
        time.sleep(1)

    # browser_window = "Choose your Gmail address Pick a Gmail address or create your own - Google Chrome"
    if find_image_in_window(browser_window, "gmail.png"):
        click_image_in_window(browser_window, "gmail.png")
    else:
        click_image_in_window(browser_window, "username.png")
    time.sleep(0.5)

    pyautogui.typewrite(base_username)
    time.sleep(0.5)

    click_image_in_window(browser_window, "next.png")
    time.sleep(1)

    if find_image_in_window(browser_window, "robot.png"):
        print('captcha appeared')
        click_image_in_window(browser_window, "robot.png")
        time.sleep(1)

    click_image_in_window(browser_window, "password.png")
    time.sleep(0.5)

    pyautogui.typewrite(password)
    time.sleep(0.5)

    click_image_in_window(browser_window, "confirm.png")
    time.sleep(0.5)

    pyautogui.typewrite(password)
    time.sleep(0.5)

    click_image_in_window(browser_window, "next.png")
    time.sleep(1)

    if find_image_in_window(browser_window, "robot.png"):
        print('captcha appeared')
        click_image_in_window(browser_window, "robot.png")
        time.sleep(1)

    if find_image_in_window(browser_window, "verification.png"):
        print("🚨 QR CODE VERIFICATION DETECTED!")
        print("📱 Please complete the QR code verification manually in the browser")
        print("⏳ Waiting for you to complete verification...")
        print("   Press ENTER in the terminal when you're done...")
        
        # Wait for user input
        input("   (Press ENTER to continue after completing QR verification)")
        print("✅ Continuing with automation...")
    
    created_email = f"{base_username}@gmail.com"
    print(f"🎉 Gmail account creation completed: {created_email}")
    return created_email

def handle_oauth_authorization_pyautogui() -> bool:
    """
    Handles the OAuth authorization flow after account creation using PyAutoGUI
    Returns True if authorization was successful
    """
    print("🔐 Handling OAuth authorization with PyAutoGUI...")
    
    # Use the specific window title for Gmail signup
    browser_window = "Sign in - Google Accounts - Google Chrome"
    
    # Focus the browser window
    focus_window(browser_window)
    time.sleep(1)
    
    try:
        # Wait for OAuth warning page
        time.sleep(3)
        
        # Try to find Advanced link
        advanced_found = False
        advanced_indicators = ["Advanced", "Show more", "More options"]
        for indicator in advanced_indicators:
            success = click_text_in_window(browser_window, indicator)
            if success:
                advanced_found = True
                print(f"✅ Found Advanced link using '{indicator}'")
                break
        
        if not advanced_found:
            print("⚠️ Advanced link not found, trying fallback...")
            # Fallback
            advanced_x, advanced_y = 500, 400
            pyautogui.click(advanced_x, advanced_y)
        
        time.sleep(2)
        
        # Try to find "Go to app" link
        go_to_app_found = False
        go_to_app_indicators = ["Go to", "Continue to", "Proceed to", "unsafe"]
        for indicator in go_to_app_indicators:
            success = click_text_in_window(browser_window, indicator)
            if success:
                go_to_app_found = True
                print(f"✅ Found Go to app link using '{indicator}'")
                break
        
        if not go_to_app_found:
            print("⚠️ Go to app link not found, trying fallback...")
            # Fallback
            go_to_app_x, go_to_app_y = 500, 450
            pyautogui.click(go_to_app_x, go_to_app_y)
        
        time.sleep(3)
        
        # Check if we reached the success page by looking for success text
        print("🔍 Checking for success page...")
        success_indicators = ["authentication flow has completed", "success", "authorized", "complete"]
        success_found = False
        
        for indicator in success_indicators:
            locations = find_text_in_window(browser_window, indicator)
            if locations:
                success_found = True
                print(f"✅ Found success indicator: '{indicator}'")
                break
        
        if success_found:
            print("✅ OAuth authorization completed successfully!")
            return True
        else:
            print("❌ OAuth authorization may have failed")
            return False
            
    except Exception as e:
        print(f"❌ OAuth authorization failed: {e}")
        return False
