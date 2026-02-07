"""
Home tab for the main window: welcome, latest updates, getting started, help links.
"""

import json
import threading
import urllib.request
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGroupBox, QFrame, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QFont, QPixmap

# Configure these URLs when you have a site (help and changelog can be the same or different)
HOME_HELP_URL = "https://github.com/Roflz/flez-bot#readme"
HOME_CHANGELOG_URL = "https://github.com/Roflz/flez-bot/releases"
HOME_UPDATES_JSON_URL = ""  # e.g. "https://yoursite.com/api/updates.json" when you have one


def _open_url(url: str) -> None:
    if not url:
        return
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def _mask_email(email: str) -> str:
    if not email or "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked = "*" * len(local)
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


class HomeTabWidget(QWidget):
    """Home tab: hero, latest updates, getting started, help/changelog buttons."""

    # Index in content layout where the profile card is inserted (after hero + icon)
    PROFILE_CARD_LAYOUT_INDEX = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updates_label = None
        self._content_layout = None  # Set in _build_ui so we can add profile card on show
        self._build_ui()
        # Fetch updates in background if URL is set
        if HOME_UPDATES_JSON_URL:
            QTimer.singleShot(500, self._check_updates)

    def showEvent(self, event):
        """When tab is shown, add profile card if we're logged in but didn't have one at build time."""
        super().showEvent(event)
        self._ensure_profile_card()

    def _build_profile_card(self) -> QGroupBox | None:
        try:
            import auth.session as session
            current_user = session.current_user
        except ImportError:
            return None
        if not current_user:
            return None
        group = QGroupBox("Profile")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #b0b0b0; }
        """)
        layout = QVBoxLayout(group)
        label_style = "color: #c0c0c0; padding: 2px 0;"
        name = current_user.get("display_name") or current_user.get("email") or "Signed in"
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(label_style)
        layout.addWidget(name_lbl)
        email = current_user.get("email")
        if email:
            email_lbl = QLabel(_mask_email(email))
            email_lbl.setStyleSheet(label_style)
            layout.addWidget(email_lbl)
        tier = (current_user.get("subscription_tier") or "free").capitalize()
        tier_lbl = QLabel(f"Plan: {tier}")
        tier_lbl.setStyleSheet(label_style)
        layout.addWidget(tier_lbl)
        btn_style = "QPushButton { background-color: #4a4a4a; color: #fff; padding: 6px 14px; border-radius: 4px; } QPushButton:hover { background-color: #5a5a5a; }"
        try:
            import auth.session as session
            is_paid = session.is_paid()
        except ImportError:
            is_paid = False
        manage_btn = QPushButton("Manage subscription" if is_paid else "Upgrade")
        manage_btn.setStyleSheet(btn_style)
        manage_btn.clicked.connect(lambda: _open_url(HOME_HELP_URL))
        layout.addWidget(manage_btn)
        edit_btn = QPushButton("Edit profile")
        edit_btn.setStyleSheet(btn_style)
        edit_btn.clicked.connect(self._on_edit_profile)
        layout.addWidget(edit_btn)
        signout_btn = QPushButton("Sign out")
        signout_btn.setStyleSheet(btn_style)
        signout_btn.clicked.connect(self._on_sign_out)
        layout.addWidget(signout_btn)
        return group

    def _ensure_profile_card(self) -> None:
        """If we have a session but no profile card yet, build and insert it."""
        if self._profile_group is not None:
            return
        try:
            import auth.session as session
            if not session.current_user:
                return
        except ImportError:
            return
        if self._content_layout is None:
            return
        self._profile_group = self._build_profile_card()
        if self._profile_group:
            self._content_layout.insertWidget(self.PROFILE_CARD_LAYOUT_INDEX, self._profile_group)

    def refresh_profile_card(self) -> None:
        """Remove existing profile card and rebuild if still logged in (e.g. after sign out or edit)."""
        if self._profile_group and self._content_layout:
            self._profile_group.setParent(None)
            self._profile_group.deleteLater()
            self._profile_group = None
        self._ensure_profile_card()

    def _on_edit_profile(self) -> None:
        try:
            import auth.session as session
            from auth.api import update_profile, get_profile
            current = session.current_user
            token = session.current_token
            if not current or not token:
                return
            name, ok = QInputDialog.getText(
                self, "Edit profile", "Display name:",
                text=current.get("display_name") or current.get("email") or ""
            )
            if not ok or name is None:
                return
            name = name.strip()
            if not name:
                return
            if update_profile(token, name):
                profile = get_profile(token) or {**current, "display_name": name}
                session.set_session(profile, token)
                self.refresh_profile_card()
            else:
                QMessageBox.warning(self, "Edit profile", "Could not update profile. Try again later.")
        except ImportError:
            pass

    def _on_sign_out(self) -> None:
        main_win = self.window()
        if main_win and hasattr(main_win, "show_login_dialog_and_refresh"):
            main_win.show_login_dialog_and_refresh()
        else:
            try:
                from auth.session import clear_session
                from auth.token_storage import clear_tokens
                clear_session()
                clear_tokens()
            except ImportError:
                pass
            from PySide6.QtWidgets import QApplication
            QApplication.quit()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        self._content_layout = layout
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # Hero / title
        hero = QLabel("Welcome to flez-bot")
        hero.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        hero.setStyleSheet("color: #e0e0e0; padding: 8px 0;")
        layout.addWidget(hero)

        # Optional icon next to title (reuse app icon)
        icon_path = Path(__file__).resolve().parent / "icon.png"
        if icon_path.is_file():
            icon_label = QLabel()
            pix = QPixmap(str(icon_path)).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon_label.setPixmap(pix)
            icon_label.setStyleSheet("background: transparent;")
            layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Card: Profile (when signed in)
        self._profile_group = self._build_profile_card()
        if self._profile_group:
            layout.addWidget(self._profile_group)

        # Card: Latest updates
        updates_group = QGroupBox("Latest updates")
        updates_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #b0b0b0; }
        """)
        updates_layout = QVBoxLayout(updates_group)
        self._updates_label = QLabel("Loading updates…" if HOME_UPDATES_JSON_URL else "No update feed configured. Set HOME_UPDATES_JSON_URL when you have a site.")
        self._updates_label.setWordWrap(True)
        self._updates_label.setStyleSheet("color: #c0c0c0; padding: 4px 0;")
        updates_layout.addWidget(self._updates_label)
        check_btn = QPushButton("Check for updates")
        check_btn.setStyleSheet("""
            QPushButton { background-color: #4a4a4a; color: #fff; padding: 6px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        check_btn.clicked.connect(self._check_updates)
        updates_layout.addWidget(check_btn)
        layout.addWidget(updates_group)

        # Card: Getting started
        getting_group = QGroupBox("Getting started")
        getting_group.setStyleSheet(updates_group.styleSheet())
        getting_layout = QVBoxLayout(getting_group)
        steps = [
            "1. Add credentials in Client → Setup & Configuration, or place .properties files in credentials/.",
            "2. Use Client → Launcher to launch one or more RuneLite instances.",
            "3. Switch to the Instances tab and pick a plan for each instance.",
            "4. Start the plan from the instance tab to run the bot.",
            "5. Use View → Statistics and the plan editor as needed.",
        ]
        for step in steps:
            lbl = QLabel(step)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #c0c0c0; padding: 2px 0;")
            getting_layout.addWidget(lbl)
        layout.addWidget(getting_group)

        # Help / Changelog buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        help_btn = QPushButton("Open help (docs)")
        help_btn.setStyleSheet(check_btn.styleSheet())
        help_btn.clicked.connect(lambda: _open_url(HOME_HELP_URL))
        help_btn.setToolTip(HOME_HELP_URL)
        changelog_btn = QPushButton("Changelog / releases")
        changelog_btn.setStyleSheet(check_btn.styleSheet())
        changelog_btn.clicked.connect(lambda: _open_url(HOME_CHANGELOG_URL))
        changelog_btn.setToolTip(HOME_CHANGELOG_URL)
        btn_layout.addWidget(help_btn)
        btn_layout.addWidget(changelog_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        scroll.setWidget(content)
        # Match dark theme
        self.setStyleSheet("background-color: #353535;")
        content.setStyleSheet("background-color: #353535;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def set_updates_text(self, text: str) -> None:
        if self._updates_label:
            self._updates_label.setText(text)

    def _check_updates(self) -> None:
        if not HOME_UPDATES_JSON_URL:
            self.set_updates_text("No update feed configured. Set HOME_UPDATES_JSON_URL in gui/home_tab_pyside.py when you have a site.")
            return

        self.set_updates_text("Checking for updates…")

        def fetch():
            try:
                req = urllib.request.urlopen(HOME_UPDATES_JSON_URL, timeout=8)
                data = json.loads(req.read().decode())
                # Support simple formats: {"latest": "v1.0", "notes": "..."} or {"releases": [{"version": "...", "notes": "..."}]}
                text = ""
                if isinstance(data, dict):
                    if "latest" in data and "notes" in data:
                        text = f"Latest: {data['latest']} — {data['notes']}"
                    elif "releases" in data and isinstance(data["releases"], list) and len(data["releases"]) > 0:
                        r = data["releases"][0]
                        text = f"Latest: {r.get('version', '?')} — {r.get('notes', '')}"
                    else:
                        text = "No release info in feed."
                else:
                    text = "Unexpected update feed format."
                QTimer.singleShot(0, lambda: self.set_updates_text(text))
            except Exception as e:
                err = str(e)
                QTimer.singleShot(0, lambda: self.set_updates_text(f"Could not load updates: {err}"))

        threading.Thread(target=fetch, daemon=True).start()
