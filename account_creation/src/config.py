import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
    BROWSER_EXECUTABLE = os.getenv("BROWSER_EXECUTABLE", "")
    PAGELOAD_TIMEOUT = int(os.getenv("PAGELOAD_TIMEOUT", "40"))
    IMPLICIT_WAIT = int(os.getenv("IMPLICIT_WAIT", "0"))
    EXPLICIT_WAIT = int(os.getenv("EXPLICIT_WAIT", "25"))

    IMAP_HOST = os.getenv("IMAP_HOST", "")
    IMAP_USERNAME = os.getenv("IMAP_USERNAME", "")
    IMAP_PASSWORD = os.getenv("IMAP_PASSWORD", "")
    IMAP_SSL = os.getenv("IMAP_SSL", "true").lower() == "true"

    GMAIL_CREDENTIALS = os.getenv("GMAIL_CREDENTIALS", "/account_creation/auth/gmail_oauth/credentials.json")
    GMAIL_TOKEN = os.getenv("GMAIL_TOKEN", "/account_creation/auth/gmail_oauth/token.json")

    OTP_CODE_REGEX = os.getenv("OTP_CODE_REGEX", r"(\b\d{5,8}\b)")
