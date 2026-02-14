"""
Supabase Auth and profile API (REST). Sign in returns JWT; GET profiles with Bearer token returns "me" via RLS.
"""
import base64
import json
from typing import Any

from auth.config import SUPABASE_URL, SUPABASE_ANON_KEY, is_configured


def profile_from_jwt(access_token: str) -> dict[str, Any] | None:
    """
    Decode JWT payload to get minimal profile (no network). Use when get_profile/get_user_from_token fail.
    Returns {"user_id", "email", "display_name", "subscription_tier"} or None if decode fails.
    """
    if not access_token or "." not in access_token:
        return None
    try:
        payload_b64 = access_token.split(".")[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        data = json.loads(base64.urlsafe_b64decode(payload_b64))
        return {
            "user_id": data.get("sub"),
            "email": data.get("email") or "",
            "display_name": (data.get("user_metadata") or {}).get("full_name") or "",
            "subscription_tier": "free",
        }
    except Exception:
        return None

try:
    import requests
except ImportError:
    requests = None


def refresh_session(refresh_token: str) -> dict[str, Any] | None:
    """
    Exchange a refresh token for a new access token (and new refresh token).
    Returns {"access_token", "refresh_token", "user"} on success, None on failure.
    Use this on startup when the stored access token is expired but refresh token is still valid.
    """
    if not is_configured() or not requests:
        return None
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    payload = {"refresh_token": refresh_token}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "access_token": data.get("access_token"),
            "refresh_token": data.get("refresh_token"),
            "user": data.get("user"),
        }
    except Exception:
        return None


def sign_in(email: str, password: str) -> dict[str, Any] | None:
    """
    Sign in with email/password. Returns {"access_token", "refresh_token", "user"} on success, None on failure.
    """
    if not is_configured() or not requests:
        return None
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    payload = {"email": email, "password": password}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "access_token": data.get("access_token"),
            "refresh_token": data.get("refresh_token"),
            "user": data.get("user"),
        }
    except Exception:
        return None


def get_profile(access_token: str) -> dict[str, Any] | None:
    """
    GET current user profile (profiles table, RLS). Returns {"user_id", "email", "display_name", "subscription_tier"} or None.
    """
    if not is_configured() or not requests:
        return None
    url = f"{SUPABASE_URL}/rest/v1/profiles?select=user_id,email,display_name,subscription_tier"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        rows = r.json()
        if not rows or len(rows) == 0:
            return None
        row = rows[0]
        return {
            "user_id": row.get("user_id"),
            "email": row.get("email") or "",
            "display_name": row.get("display_name") or "",
            "subscription_tier": row.get("subscription_tier") or "free",
        }
    except Exception:
        return None


def recover_password(email: str, redirect_to: str | None = None) -> tuple[bool, str]:
    """
    Send a password recovery email. Returns (True, "") on success, (False, error_message) on failure.
    If redirect_to is provided, it must be in Supabase Auth → URL Configuration → Redirect URLs.
    """
    if not is_configured() or not requests:
        return False, "Auth not configured."
    api_url = f"{SUPABASE_URL}/auth/v1/recover"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    payload = {"email": email}
    if redirect_to:
        redirect_to = (redirect_to or "").strip()
        if redirect_to:
            payload["redirect_to"] = redirect_to
    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            return True, ""
        try:
            err = r.json()
            msg = err.get("msg") or err.get("error_description") or err.get("message") or r.text or f"HTTP {r.status_code}"
        except Exception:
            msg = r.text or f"HTTP {r.status_code}"
        return False, msg
    except requests.exceptions.RequestException as e:
        return False, str(e) or "Network error."
    except Exception as e:
        return False, str(e)


def update_profile(access_token: str, display_name: str) -> bool:
    """
    Update the current user's profile (display_name) in the profiles table. RLS allows update own row.
    """
    uid = _user_id_from_token(access_token)
    if not is_configured() or not requests or not uid:
        return False
    url = f"{SUPABASE_URL}/rest/v1/profiles?user_id=eq.{uid}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {"display_name": display_name}
    try:
        r = requests.patch(url, json=payload, headers=headers, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


def _user_id_from_token(access_token: str) -> str:
    """Extract user id (sub) from JWT for use in profile update URL."""
    if not access_token or "." not in access_token:
        return ""
    try:
        payload_b64 = access_token.split(".")[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        data = json.loads(base64.urlsafe_b64decode(payload_b64))
        return data.get("sub") or ""
    except Exception:
        return ""


def get_user_from_token(access_token: str) -> dict[str, Any] | None:
    """
    Validate token and get user info from Supabase Auth (GET /auth/v1/user). Fallback if profiles table not set up yet.
    """
    if not is_configured() or not requests:
        return None
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "user_id": data.get("id"),
            "email": data.get("email") or "",
            "display_name": "",
            "subscription_tier": "free",
        }
    except Exception:
        return None
