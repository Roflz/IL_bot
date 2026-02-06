#!/usr/bin/env python3
"""
Simple Recorder GUI (PySide6)
==============================

Entry point for PySide6 version of the GUI.
Runs a readiness check on startup and can auto-fix PySide6 and submodules.
"""

import sys
import subprocess
from pathlib import Path

# Add the parent directory to the path for imports
_bot_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_bot_root.parent))

# Ensure PySide6 is available before importing Qt (auto-fix if missing)
def _ensure_pyside6():
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        pass
    print("PySide6 not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "PySide6", "-q"],
            check=True,
            timeout=120,
        )
        import PySide6  # noqa: F401
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ImportError) as e:
        print(f"Could not install PySide6: {e}")
        print("Run manually: python -m pip install PySide6")
        return False

if not _ensure_pyside6():
    sys.exit(1)

from PySide6.QtWidgets import QApplication, QStyleFactory, QMessageBox
from PySide6.QtGui import QPalette, QColor

# Import the main GUI class from the PySide6 version
from gui.main_window_pyside import SimpleRecorderGUI


def _readiness_check(apply_auto_fix: bool = True):
    """
    Check that the environment is ready (runelite path, gradlew). Optionally run
    git submodule update --init --recursive if we're in the flez-bot repo and submodules
    are missing. Returns (ok: bool, missing: list[str], flez_bot_root: Path, runelite_path: Path).
    """
    # bot_runelite_IL root (where gui_pyside.py lives) and flez-bot root (parent)
    bot_root = Path(__file__).resolve().parent
    flez_bot_root = bot_root.parent
    runelite_path = flez_bot_root / "runelite"
    missing = []

    # Check runelite directory exists
    if not runelite_path.is_dir():
        # Auto-fix: try submodule init if we're in a git repo
        if apply_auto_fix and (flez_bot_root / ".git").exists():
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(flez_bot_root),
                    check=True,
                    timeout=300,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                pass
        if not runelite_path.is_dir():
            missing.append("runelite directory not found. Run setup.ps1 from the flez-bot directory.")
            return False, missing, flez_bot_root, runelite_path

    # Check gradlew exists
    gradlew = runelite_path / "gradlew.bat" if sys.platform == "win32" else runelite_path / "gradlew"
    if not gradlew.is_file():
        missing.append("runelite/gradlew (or gradlew.bat) not found. Run setup.ps1 to initialize submodules.")

    ok = len(missing) == 0
    return ok, missing, flez_bot_root, runelite_path


def _show_readiness_dialog(missing: list, flez_bot_root: Path):
    """Show a dialog listing what's missing and offer to run setup.ps1."""
    app = QApplication.instance()
    setup_ps1 = flez_bot_root / "setup.ps1"
    msg = "The following are missing or incomplete:\n\n" + "\n".join("• " + m for m in missing)
    msg += "\n\nRun setup.ps1 from the flez-bot directory to fix this (one-time)."
    if not setup_ps1.is_file():
        msg += "\n\n(setup.ps1 not found at {})".format(setup_ps1)
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Setup required")
    box.setText(msg)
    if setup_ps1.is_file() and sys.platform == "win32":
        run_btn = box.addButton("Run setup.ps1", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() == run_btn:
            try:
                subprocess.Popen(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(setup_ps1)],
                    cwd=str(flez_bot_root),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
                )
                QMessageBox.information(
                    None,
                    "Setup started",
                    "Setup.ps1 has been started in a new window. When it finishes, close this message and restart the application.",
                )
            except Exception as e:
                QMessageBox.critical(None, "Error", "Could not start setup.ps1: {}".format(e))
    else:
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()


def apply_dark_theme(app: QApplication):
    """Apply a dark theme to the application using QPalette."""
    # Use Fusion style (cross-platform, supports custom palettes well)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Create dark palette
    palette = QPalette()
    
    # Window (background)
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    
    # Base (input fields background)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    
    # Text
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    
    # Button
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    
    # Highlight (selected items)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Tooltip
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    
    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(127, 127, 127))
    
    # Link
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(130, 42, 218))
    
    # Apply the palette
    app.setPalette(palette)


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Set application properties
    app.setApplicationName("Simple Recorder")
    app.setOrganizationName("Simple Recorder")
    
    # Readiness check (auto-fix submodules if possible)
    ok, missing, flez_bot_root, _ = _readiness_check(apply_auto_fix=True)
    if not ok:
        _show_readiness_dialog(missing, flez_bot_root)
        sys.exit(0)
    
    # Create and show main window
    window = SimpleRecorderGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
