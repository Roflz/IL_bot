from selenium.webdriver.remote.webdriver import WebDriver
from ..ui_helpers import safe_click, safe_type, wait_visible
from ..selectors import example_selectors as S

def create_character(driver: WebDriver, name: str) -> None:
    wait_visible(driver, S.CREATE_CHAR_BTN)
    safe_click(driver, S.CREATE_CHAR_BTN)
    wait_visible(driver, S.CHAR_NAME_INPUT)
    safe_type(driver, S.CHAR_NAME_INPUT, name)
    safe_click(driver, S.CHAR_CONFIRM)

def bulk_create_characters(driver: WebDriver, names: list[str]) -> None:
    for n in names:
        create_character(driver, n)
