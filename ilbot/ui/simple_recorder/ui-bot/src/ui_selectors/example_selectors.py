from selenium.webdriver.common.by import By

LOGIN_USERNAME = (By.CSS_SELECTOR, "input[name='username'], input#username")
LOGIN_PASSWORD = (By.CSS_SELECTOR, "input[name='password'], input#password")
LOGIN_SUBMIT   = (By.CSS_SELECTOR, "button[type='submit'], button#login, .login-button")

OTP_INPUT      = (By.CSS_SELECTOR, "input[name='otp'], input[name='code'], input.verification-code")
OTP_SUBMIT     = (By.CSS_SELECTOR, "button[type='submit'], button.verify, .otp-submit")

CREATE_CHAR_BTN = (By.CSS_SELECTOR, "button#create-character, .create-char-btn")
CHAR_NAME_INPUT = (By.CSS_SELECTOR, "input[name='characterName']")
CHAR_CONFIRM    = (By.CSS_SELECTOR, "button.confirm, button[type='submit']")
