import re, time, os, base64
from typing import Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from config import Config

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def _gmail_service(token_file: str = None):
    """Get Gmail service for a specific account"""
    token_path = token_file or "D:/repos/bot_runelite_IL/ilbot/ui/simple_recorder/account_creation/auth/gmail_oauth/token.json"
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            pass
        else:
            flow = InstalledAppFlow.from_client_secrets_file("/account_creation/auth/gmail_oauth/credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_path, "w") as f:
                f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def poll_for_code_gmail_api(q: str, timeout_s: int=120, poll_interval_s: int=5, code_regex: str|None=None, token_file: str=None) -> Optional[str]:
    end = time.time() + timeout_s
    pattern = re.compile(code_regex or Config.OTP_CODE_REGEX)
    svc = _gmail_service(token_file)

    while time.time() < end:
        resp = svc.users().messages().list(userId="me", q=q, maxResults=5).execute()
        msgs = resp.get("messages", [])
        for m in msgs:
            full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
            snippet = full.get("snippet", "")
            parts = full.get("payload", {}).get("parts", []) or []
            text_blobs = [snippet]
            for p in parts:
                if p.get("mimeType") == "text/plain":
                    data = p.get("body", {}).get("data", "")
                    if data:
                        text_blobs.append(base64.urlsafe_b64decode(data.encode()).decode(errors="ignore"))
            text = "\n".join(text_blobs)
            mcode = pattern.search(text)
            if mcode:
                return mcode.group(1)
        time.sleep(poll_interval_s)
    return None
