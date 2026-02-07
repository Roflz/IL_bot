"""
In-memory session after login: current_user profile and token for this run.
"""
from typing import Any

# Set by app after successful login; cleared on sign out.
current_user: dict[str, Any] | None = None   # { user_id, email, display_name, subscription_tier }
current_token: str | None = None              # access_token (from storage or session-only)


def set_session(profile: dict[str, Any], token: str) -> None:
    global current_user, current_token
    current_user = profile
    current_token = token


def clear_session() -> None:
    global current_user, current_token
    current_user = None
    current_token = None


def is_paid() -> bool:
    return current_user is not None and (current_user.get("subscription_tier") or "free") == "paid"
