"""
Supabase auth config. Reads from web/config.js (single source of truth).
No custom project override allowed.
"""
import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_JS = _REPO_ROOT / "web" / "config.js"


def _read_config_js() -> tuple[str, str, str, str]:
    """Parse web/config.js. Returns (supabase_url, anon_key, signup_url, reset_url)."""
    url, anon_key, signup_url, reset_url = "", "", "", ""
    if not _CONFIG_JS.is_file():
        return url, anon_key, signup_url, reset_url
    try:
        text = _CONFIG_JS.read_text(encoding="utf-8")
        m = re.search(r'FLEZ_BOT_SUPABASE_URL\s*=\s*"([^"]*)"', text)
        if m:
            url = m.group(1).strip().rstrip("/")
        m = re.search(r'FLEZ_BOT_SUPABASE_ANON_KEY\s*=\s*"([^"]*)"', text)
        if m:
            anon_key = m.group(1).strip()
        m = re.search(r'FLEZ_BOT_SITE_URL\s*=\s*"([^"]*)"', text)
        if m:
            base = m.group(1).strip().rstrip("/")
            signup_url = base + "/signup.html"
            reset_url = base + "/reset-password.html"
    except Exception:
        pass
    return url, anon_key, signup_url, reset_url


_url, _key, _signup, _reset = _read_config_js()
SUPABASE_URL = _url
SUPABASE_ANON_KEY = _key
SIGNUP_URL = os.environ.get("FLEZ_BOT_SIGNUP_URL", _signup) or ""
PASSWORD_RESET_URL = (os.environ.get("FLEZ_BOT_PASSWORD_RESET_URL", _reset) or "").strip()


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)
