from selenium.webdriver.remote.webdriver import WebDriver
import time

def navigate_to_jagex_account_creation(driver: WebDriver) -> None:
    """Navigate to Jagex account creation page"""
    print("🚀 Navigating to Jagex account creation page...")
    driver.get("https://account.jagex.com/")
    time.sleep(2)
    print("✅ Browser opened to https://account.jagex.com/")
    print("📝 You can now manually create the RuneScape account.")




