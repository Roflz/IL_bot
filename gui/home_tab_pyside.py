"""
Home tab for the main window: welcome, latest updates, getting started, help links.
"""

import json
import threading
import urllib.request
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGroupBox, QFrame
)
from PySide6.QtCore import Qt, QTimer
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


class HomeTabWidget(QWidget):
    """Home tab: hero, latest updates, getting started, help/changelog buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updates_label = None
        self._build_ui()
        # Fetch updates in background if URL is set
        if HOME_UPDATES_JSON_URL:
            QTimer.singleShot(500, self._check_updates)

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
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
