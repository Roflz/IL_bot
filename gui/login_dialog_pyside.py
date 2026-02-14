"""
Login dialog for Supabase auth. Email, password, Sign in / Sign up / Forgot password.
"""
import webbrowser

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QMessageBox, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from auth.config import is_configured, SIGNUP_URL, PASSWORD_RESET_URL
from auth.api import sign_in, recover_password
from auth.token_storage import set_tokens, clear_tokens


class LoginDialog(QDialog):
    """Modal login dialog. On success, stores token and accept()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign in to flez-bot")
        self.setMinimumWidth(360)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        if not is_configured():
            layout.addWidget(QLabel("Supabase is not configured."))
            skip_btn = QPushButton("Continue without signing in")
            skip_btn.clicked.connect(self.accept)
            layout.addWidget(skip_btn)
            return

        line_edit_style = "QLineEdit::placeholder-text { color: #a0a0a0; }"
        layout.addWidget(QLabel("Email"))
        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("you@example.com")
        self.email_edit.setStyleSheet(line_edit_style)
        self.email_edit.setClearButtonEnabled(True)
        layout.addWidget(self.email_edit)

        layout.addWidget(QLabel("Password"))
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("••••••••")
        self.password_edit.setStyleSheet(line_edit_style)
        layout.addWidget(self.password_edit)

        self.remember_cb = QCheckBox("Stay logged in")
        self.remember_cb.setChecked(True)
        layout.addWidget(self.remember_cb)

        btn_layout = QHBoxLayout()
        signin_btn = QPushButton("Sign in")
        signin_btn.setDefault(True)
        signin_btn.clicked.connect(self._on_sign_in)
        signup_btn = QPushButton("Sign up")
        signup_btn.clicked.connect(self._on_sign_up)
        forgot_btn = QPushButton("Forgot password")
        forgot_btn.clicked.connect(self._on_forgot_password)
        btn_layout.addWidget(signin_btn)
        btn_layout.addWidget(signup_btn)
        btn_layout.addWidget(forgot_btn)
        layout.addLayout(btn_layout)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: #f88;")
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

    def _on_sign_in(self):
        self._error_label.setText("")
        email = self.email_edit.text().strip()
        password = self.password_edit.text()
        if not email or not password:
            self._error_label.setText("Enter email and password.")
            return
        result = sign_in(email, password)
        if not result or not result.get("access_token"):
            self._error_label.setText("Invalid email or password, or network error.")
            return
        # Always keep token in memory for this run so caller gets it even if keyring fails
        self._session_token = result["access_token"]
        if self.remember_cb.isChecked():
            set_tokens(
                result["access_token"],
                result.get("refresh_token"),
            )
        self._sign_in_user = result.get("user")  # So caller can build profile if get_profile fails
        self.accept()

    def _on_sign_up(self):
        """Open sign-up on the website (not in-app)."""
        if SIGNUP_URL:
            webbrowser.open(SIGNUP_URL)
            self._error_label.setText("Opened sign-up page in your browser.")
        else:
            self._error_label.setText("Sign-up URL is not configured. Set FLEZ_BOT_SIGNUP_URL in .env for a custom sign-up page.")

    def _on_forgot_password(self):
        """Send password reset email in-app via Supabase recover."""
        self._error_label.setText("")
        email = self.email_edit.text().strip()
        if not email:
            self._error_label.setText("Enter your email above, then click Forgot password.")
            return
        redirect_to = (PASSWORD_RESET_URL or "").strip() or None
        ok, err_msg = recover_password(email, redirect_to=redirect_to)
        if ok:
            self._error_label.setStyleSheet("color: #8c8;")
            self._error_label.setText("If an account exists for that email, we've sent a reset link. Check your inbox.")
        else:
            self._error_label.setStyleSheet("color: #f88;")
            msg = err_msg or "Could not send reset email."
            if msg and ("redirect" in msg.lower() or "url" in msg.lower() or "allowed" in msg.lower()):
                msg = msg + " Add your reset page to Supabase Auth → URL Configuration → Redirect URLs."
            self._error_label.setText(msg)

    def get_session_token(self) -> str | None:
        """If user chose 'Stay logged in' = false, token is only in memory for this run."""
        return getattr(self, "_session_token", None)

    def get_sign_in_user(self) -> dict | None:
        """Auth user from last successful sign-in (id, email, user_metadata, ...). Use if get_profile returns None."""
        return getattr(self, "_sign_in_user", None)
