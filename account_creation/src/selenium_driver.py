from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from config import Config

def make_driver() -> webdriver.Chrome:
    opts = Options()
    if Config.HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-notifications")
    
    # Use a unique profile directory with timestamp
    import os
    import time
    timestamp = int(time.time())
    selenium_profile = f"C:\\Users\\mahnr\\AppData\\Local\\Google\\Chrome\\User Data\\Selenium_{timestamp}"
    
    # Create unique profile directory
    try:
        if os.path.exists(selenium_profile):
            import shutil
            shutil.rmtree(selenium_profile)
        os.makedirs(selenium_profile, exist_ok=True)
        print(f"✅ Created unique Chrome profile: {selenium_profile}")
    except Exception as e:
        print(f"⚠️ Could not create profile: {e}")
    
    opts.add_argument(f"--user-data-dir={selenium_profile}")
    opts.add_argument("--profile-directory=Default")
    
    # Add arguments to make it more like a regular browser session
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)

    if Config.BROWSER_EXECUTABLE:
        opts.binary_location = Config.BROWSER_EXECUTABLE

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(Config.PAGELOAD_TIMEOUT)
    driver.implicitly_wait(Config.IMPLICIT_WAIT)
    return driver
