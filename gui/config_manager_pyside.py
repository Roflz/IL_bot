"""
Configuration Manager Module (PySide6)
=======================================

Handles loading/saving the single launcher preference (auto-detect paths)
via QSettings. No launch-config.json; all paths are derived at runtime.
"""

from pathlib import Path
from typing import Dict, Optional
import logging
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QSettings


class ConfigManager:
    """Manages launcher preference (auto-detect on startup). Uses QSettings only."""
    
    _SETTINGS_ORG = "Simple Recorder"
    _SETTINGS_APP = "Simple Recorder"
    _KEY_AUTO_DETECT = "launcher/autoDetectPaths"
    
    def __init__(self, config_file: Optional[Path], config_vars: Dict[str, str], path_entries: Dict[str, object]):
        """
        Args:
            config_file: Unused (kept for API compat); no JSON config file.
            config_vars: Dictionary including "autoDetect" (bool).
            path_entries: Path widgets (may be empty).
        """
        self.config_file = config_file
        self.config_vars = config_vars
        self.path_entries = path_entries
    
    def load_config(self, base_port_var=None, launch_delay_var=None, log_callback=None):
        """Load auto-detect preference from QSettings."""
        try:
            settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
            enabled = settings.value(self._KEY_AUTO_DETECT, True)
            if isinstance(enabled, str):
                enabled = enabled.lower() in ("true", "1", "yes")
            self.config_vars["autoDetect"] = bool(enabled)
            self.toggle_auto_detect()
            if log_callback:
                log_callback("Configuration loaded", 'success')
        except Exception as e:
            if log_callback:
                log_callback(f"Error loading config: {str(e)}", 'error')
            else:
                logging.error(f"Error loading config: {str(e)}")
    
    def save_config(self, base_port_var=None, launch_delay_var=None, log_callback=None):
        """Save auto-detect preference to QSettings."""
        try:
            settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
            settings.setValue(self._KEY_AUTO_DETECT, self.config_vars["autoDetect"])
            if log_callback:
                log_callback("Configuration saved", 'success')
        except Exception as e:
            if log_callback:
                log_callback(f"Error saving config: {str(e)}", 'error')
            QMessageBox.critical(None, "Config Error", f"Failed to save configuration: {str(e)}")
    
    def auto_detect_paths(self, log_callback=None):
        """Auto-detect paths using PowerShell script logic."""
        import os
        # TODO: Implement auto-detection
        if log_callback:
            log_callback("Auto-detection not yet implemented", 'info')
    
    def toggle_auto_detect(self):
        """Toggle auto-detect mode and enable/disable path entries."""
        state = not self.config_vars["autoDetect"]
        for entry in self.path_entries.values():
            entry.setEnabled(state)
    
    def browse_path(self, key: str, is_file: bool = False):
        """Open file/directory browser for a configuration path."""
        current_path = self.path_entries.get(key, "").text() if key in self.path_entries else self.config_vars.get(key, "")
        initial_dir = current_path if current_path and Path(current_path).exists() else str(Path.home())
        
        if is_file:
            path, _ = QFileDialog.getOpenFileName(
                None,
                f"Select {key}",
                initial_dir
            )
        else:
            path = QFileDialog.getExistingDirectory(
                None,
                f"Select {key}",
                initial_dir
            )
        
        if path:
            self.config_vars[key] = path
            if key in self.path_entries:
                self.path_entries[key].setText(path)
