#!/usr/bin/env python3
"""
Simple Recorder GUI (PySide6)
==============================

Entry point for the PySide6 GUI. Normally run via the flez-bot root launcher
(launcher.py), which handles updates and readiness before starting this.
For development you can run this file directly from bot_runelite_IL.
"""

import sys
from pathlib import Path

# Load .env from repo root FIRST so auth.config sees SUPABASE_URL/SUPABASE_ANON_KEY
# (when run via "python gui_pyside.py" or from IDE, launcher never runs so .env wasn't loaded)
_bot_root = Path(__file__).resolve().parent
_repo_root = _bot_root.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass

# Ensure bot_runelite_IL is on the path so "from gui.main_window_pyside" works
sys.path.insert(0, str(_bot_root))

from PySide6.QtWidgets import QApplication, QStyleFactory, QDialog
from PySide6.QtGui import QPalette, QColor

from gui.main_window_pyside import SimpleRecorderGUI
from gui.login_dialog_pyside import LoginDialog
from auth.config import is_configured
from auth.token_storage import get_token, get_refresh_token, set_tokens, clear_tokens
from auth.api import get_profile, get_user_from_token, profile_from_jwt, refresh_session
from auth.session import set_session, clear_session


def _profile_from_auth_user(u: dict) -> dict:
    """Build profile dict from Supabase auth user (sign-in response)."""
    meta = u.get("user_metadata") or {}
    return {
        "user_id": u.get("id"),
        "email": u.get("email") or "",
        "display_name": meta.get("full_name") or meta.get("display_name") or "",
        "subscription_tier": "free",
    }


def apply_dark_theme(app: QApplication):
    """Apply a dark theme to the application using QPalette."""
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(130, 42, 218))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))  # light gray for placeholders
    app.setPalette(palette)


def _fallback_profile() -> dict:
    return {"user_id": None, "email": "", "display_name": "Signed in", "subscription_tier": "free"}


def _ensure_logged_in() -> bool:
    """Show login if needed; set auth.session.current_user and current_token. Returns True to continue, False to exit."""
    if not is_configured():
        return True
    token = get_token()
    if token:
        # Prefer profile from API (validates token). JWT decode works even when token is expired.
        p1, p2 = get_profile(token), get_user_from_token(token)
        if p1 or p2:
            profile = p1 or p2
            set_session(profile, token)
            return True
        # Token may be expired; try refresh (Stay logged in stores refresh_token).
        refresh_tok = get_refresh_token()
        if refresh_tok:
            result = refresh_session(refresh_tok)
            if result and result.get("access_token"):
                new_access = result["access_token"]
                set_tokens(new_access, result.get("refresh_token"))
                p1, p2, p3 = get_profile(new_access), get_user_from_token(new_access), profile_from_jwt(new_access)
                profile = p1 or p2 or p3
                if result.get("user"):
                    profile = profile or _profile_from_auth_user(result["user"])
                if not profile:
                    profile = _fallback_profile()
                set_session(profile, new_access)
                return True
        clear_tokens()
    dlg = LoginDialog()
    dlg.exec()
    if dlg.result() != QDialog.DialogCode.Accepted:
        return False
    token = dlg.get_session_token() or get_token()
    if not token:
        return True
    p1, p2, p3, p4 = get_profile(token), get_user_from_token(token), (dlg.get_sign_in_user() and _profile_from_auth_user(dlg.get_sign_in_user())), profile_from_jwt(token)
    profile = p1 or p2 or p3 or p4 or _fallback_profile()
    set_session(profile, token)
    return True


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    app.setApplicationName("Simple Recorder")
    app.setOrganizationName("Simple Recorder")
    if not _ensure_logged_in():
        sys.exit(0)
    window = SimpleRecorderGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
