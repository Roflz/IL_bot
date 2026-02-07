"""
Supabase auth config. Set via environment variables so you don't commit secrets.
"""
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# URLs opened in browser for sign up / password reset (your site or Supabase hosted)
SIGNUP_URL = os.environ.get("FLEZ_BOT_SIGNUP_URL", "")
PASSWORD_RESET_URL = os.environ.get("FLEZ_BOT_PASSWORD_RESET_URL", "")


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)
