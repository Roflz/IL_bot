from typing import Callable, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from .config import Config

def wait_visible(driver: WebDriver, locator: tuple[By, str], timeout: Optional[int]=None) -> WebElement:
    t = timeout or Config.EXPLICIT_WAIT
    return WebDriverWait(driver, t).until(EC.visibility_of_element_located(locator))

def wait_clickable(driver: WebDriver, locator: tuple[By, str], timeout: Optional[int]=None) -> WebElement:
    t = timeout or Config.EXPLICIT_WAIT
    return WebDriverWait(driver, t).until(EC.element_to_be_clickable(locator))

def safe_click(driver: WebDriver, locator: tuple[By, str], timeout: Optional[int]=None) -> None:
    el = wait_clickable(driver, locator, timeout)
    el.click()

def safe_type(driver: WebDriver, locator: tuple[By, str], text: str, clear: bool=True, timeout: Optional[int]=None) -> None:
    el = wait_visible(driver, locator, timeout)
    if clear:
        el.clear()
    el.send_keys(text)

def exists(driver: WebDriver, locator: tuple[By, str], timeout: int=3) -> bool:
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located(locator))
        return True
    except TimeoutException:
        return False

def retry(fn: Callable, attempts: int=3, on_error: Optional[Callable]=None):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if on_error:
                on_error(e)
    if last:
        raise last
