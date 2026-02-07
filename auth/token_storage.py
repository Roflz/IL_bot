"""
Store and retrieve auth tokens (no plain passwords).
Uses Windows Credential Manager via keyring when available; falls back to a
user-only token file so "Stay logged in" works even if keyring fails (e.g. size limits).
"""
import json
import os

STORAGE_SERVICE = "flez-bot"
TOKEN_KEY = "access_token"
REFRESH_KEY = "refresh_token"


def _token_file_path():
    """Path to fallback token file (user-only, not shared)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~")
    dir_path = os.path.join(base, ".flez-bot")
    try:
        os.makedirs(dir_path, mode=0o700, exist_ok=True)
    except OSError:
        pass
    return os.path.join(dir_path, "tokens.json")


def _read_file_tokens() -> tuple[str | None, str | None]:
    """Read access_token and refresh_token from file. Returns (access, refresh) or (None, None)."""
    path = _token_file_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("access_token") or None,
            data.get("refresh_token") or None,
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None


def _write_file_tokens(access_token: str, refresh_token: str | None) -> None:
    """Write tokens to file. File is created with user-only permissions."""
    path = _token_file_path()
    try:
        data = {"access_token": access_token, "refresh_token": refresh_token or ""}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass


def _clear_file_tokens() -> None:
    try:
        path = _token_file_path()
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _get_keyring():
    try:
        import keyring
        return keyring
    except ImportError:
        return None


def get_token() -> str | None:
    kr = _get_keyring()
    if kr:
        try:
            t = kr.get_password(STORAGE_SERVICE, TOKEN_KEY)
            if t:
                return t
        except Exception:
            pass
    access, _ = _read_file_tokens()
    return access


def get_refresh_token() -> str | None:
    kr = _get_keyring()
    if kr:
        try:
            t = kr.get_password(STORAGE_SERVICE, REFRESH_KEY)
            if t:
                return t
        except Exception:
            pass
    _, refresh = _read_file_tokens()
    return refresh


def set_tokens(access_token: str, refresh_token: str | None = None) -> None:
    # File fallback so "Stay logged in" works even when keyring fails (e.g. Windows size limit).
    _write_file_tokens(access_token, refresh_token)
    kr = _get_keyring()
    if not kr:
        return
    try:
        kr.set_password(STORAGE_SERVICE, TOKEN_KEY, access_token)
        if refresh_token:
            kr.set_password(STORAGE_SERVICE, REFRESH_KEY, refresh_token)
    except Exception:
        pass


def clear_tokens() -> None:
    _clear_file_tokens()
    kr = _get_keyring()
    if not kr:
        return
    try:
        kr.delete_password(STORAGE_SERVICE, TOKEN_KEY)
        kr.delete_password(STORAGE_SERVICE, REFRESH_KEY)
    except Exception:
        pass
