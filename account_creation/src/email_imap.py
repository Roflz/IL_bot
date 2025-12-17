import re, time
from typing import Optional
from imapclient import IMAPClient
from email import message_from_bytes
from bs4 import BeautifulSoup
from config import Config

def _extract_text_from_email(msg_bytes: bytes) -> str:
    msg = message_from_bytes(msg_bytes)
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                return part.get_payload(decode=True).decode(errors="ignore")
            if ctype == "text/html":
                html = part.get_payload(decode=True).decode(errors="ignore")
                return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")
    return ""

def poll_for_code_via_imap(subject_contains: str, from_contains: str="", timeout_s: int=120, poll_interval_s: int=5) -> Optional[str]:
    end = time.time() + timeout_s
    pattern = re.compile(Config.OTP_CODE_REGEX)

    with IMAPClient(Config.IMAP_HOST, ssl=Config.IMAP_SSL) as client:
        client.login(Config.IMAP_USERNAME, Config.IMAP_PASSWORD)
        client.select_folder("INBOX", readonly=True)

        while time.time() < end:
            criteria = ["UNSEEN"]
            if subject_contains:
                criteria += ["SUBJECT", subject_contains]
            if from_contains:
                criteria += ["FROM", from_contains]
            uids = client.search(criteria)
            if uids:
                for uid in reversed(uids):
                    raw = client.fetch([uid], ["RFC822"])[uid][b"RFC822"]
                    text = _extract_text_from_email(raw)
                    m = pattern.search(text)
                    if m:
                        return m.group(1)
            time.sleep(poll_interval_s)
    return None
