from selenium.webdriver.remote.webdriver import WebDriver
from ..ui_helpers import safe_click, safe_type, wait_visible
from ..ui_selectors import example_selectors as S

def navigate_to_jagex_account_creation(driver: WebDriver) -> None:
    """Navigate to Jagex account creation page"""
    driver.get("https://account.jagex.com/")

def perform_login(driver: WebDriver, login_url: str, username: str, password: str) -> None:
    driver.get(login_url)
    wait_visible(driver, S.LOGIN_USERNAME)
    safe_type(driver, S.LOGIN_USERNAME, username)
    safe_type(driver, S.LOGIN_PASSWORD, password)
    safe_click(driver, S.LOGIN_SUBMIT)

def submit_otp(driver: WebDriver, code: str) -> None:
    safe_type(driver, S.OTP_INPUT, code)
    safe_click(driver, S.OTP_SUBMIT)

def login_with_email_otp(
    driver: WebDriver,
    login_url: str,
    username: str,
    password: str,
    fetch_code_fn,
    fetch_kwargs: dict
) -> None:
    perform_login(driver, login_url, username, password)
    code = fetch_code_fn(**fetch_kwargs)
    if not code:
        raise RuntimeError("Timed out waiting for verification code email.")
    submit_otp(driver, code)
