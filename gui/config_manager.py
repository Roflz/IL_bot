"""
Configuration Manager Module
============================

Handles loading, saving, and auto-detection of RuneLite configuration paths.
"""

import tkinter as tk
from pathlib import Path
from typing import Dict
import json
import logging


class ConfigManager:
    """Manages RuneLite launcher configuration."""
    
    def __init__(self, config_file: Path, config_vars: Dict[str, tk.Variable], path_entries: Dict[str, tk.Widget]):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to the configuration JSON file
            config_vars: Dictionary of tkinter variables for config values
            path_entries: Dictionary of tkinter entry widgets for paths
        """
        self.config_file = config_file
        self.config_vars = config_vars
        self.path_entries = path_entries
    
    def load_config(self, base_port_var=None, launch_delay_var=None, 
                   build_maven_var=None, log_callback=None):
        """
        Load configuration from JSON file.
        
        Args:
            base_port_var: Optional IntVar for base port
            launch_delay_var: Optional IntVar for launch delay
            build_maven_var: Optional BooleanVar for build Maven
            log_callback: Optional callback for logging messages
        """
        try:
            if not self.config_file.exists():
                if log_callback:
                    log_callback("Config file not found, using defaults", 'info')
                self.auto_detect_paths(log_callback)
                return
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Load paths
            if 'paths' in config:
                for key, value in config['paths'].items():
                    if key in self.config_vars:
                        self.config_vars[key].set(value or "")
            
            # Load auto-detect setting
            if 'autoDetect' in config and 'enabled' in config['autoDetect']:
                self.config_vars["autoDetect"].set(config['autoDetect']['enabled'])
            
            # Load launch settings (only if widgets exist)
            if base_port_var and 'launch' in config:
                if 'basePort' in config['launch']:
                    base_port_var.set(config['launch']['basePort'])
                if launch_delay_var and 'delaySeconds' in config['launch']:
                    launch_delay_var.set(config['launch']['delaySeconds'])
                if build_maven_var and 'buildMaven' in config['launch']:
                    build_maven_var.set(config['launch']['buildMaven'])
            
            self.toggle_auto_detect()
            if log_callback:
                log_callback("Configuration loaded successfully", 'success')
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error loading config: {str(e)}", 'error')
            else:
                logging.error(f"Error loading config: {str(e)}")
    
    def save_config(self, base_port_var=None, launch_delay_var=None, 
                   build_maven_var=None, log_callback=None):
        """
        Save current configuration to JSON file.
        
        Args:
            base_port_var: Optional IntVar for base port
            launch_delay_var: Optional IntVar for launch delay
            build_maven_var: Optional BooleanVar for build Maven
            log_callback: Optional callback for logging messages
        """
        from tkinter import messagebox
        
        try:
            config = {
                "version": "1.0",
                "autoDetect": {
                    "enabled": self.config_vars["autoDetect"].get(),
                    "java": True,
                    "maven": True,
                    "projectDir": True,
                    "credentialsDir": True,
                    "baseDir": True,
                    "exportsBase": True,
                    "classPathFile": True
                },
                "paths": {
                    "projectDir": self.config_vars["projectDir"].get(),
                    "classPathFile": self.config_vars["classPathFile"].get(),
                    "javaExe": self.config_vars["javaExe"].get(),
                    "baseDir": self.config_vars["baseDir"].get(),
                    "exportsBase": self.config_vars["exportsBase"].get(),
                    "credentialsDir": self.config_vars["credentialsDir"].get()
                },
                "launch": {
                    "basePort": base_port_var.get() if base_port_var else 17000,
                    "delaySeconds": launch_delay_var.get() if launch_delay_var else 0,
                    "defaultWorld": 0,
                    "buildMaven": build_maven_var.get() if build_maven_var else True
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            if log_callback:
                log_callback("Configuration saved successfully", 'success')
            messagebox.showinfo("Config Saved", "Configuration saved successfully!")
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error saving config: {str(e)}", 'error')
            messagebox.showerror("Config Error", f"Failed to save configuration: {str(e)}")
    
    def auto_detect_paths(self, log_callback=None):
        """
        Auto-detect paths using PowerShell script logic.
        
        Args:
            log_callback: Optional callback for logging messages
        """
        import os
        
        try:
            script_dir = Path(__file__).resolve().parent.parent
            
            # Find Java
            java_path = None
            if os.environ.get('JAVA_HOME'):
                java_exe = Path(os.environ['JAVA_HOME']) / "bin" / "java.exe"
                if java_exe.exists():
                    java_path = str(java_exe)
            
            if not java_path:
                # Check common paths
                common_paths = [
                    Path("C:/Program Files/Java/jdk-11*"),
                    Path("C:/Program Files (x86)/Java/jdk-11*"),
                ]
                for pattern in common_paths:
                    matches = list(pattern.parent.glob(pattern.name))
                    if matches:
                        java_exe = matches[0] / "bin" / "java.exe"
                        if java_exe.exists():
                            java_path = str(java_exe)
                            break
            
            if java_path:
                self.config_vars["javaExe"].set(java_path)
            
            # Find RuneLite project
            possible_paths = [
                script_dir.parent / "IdeaProjects" / "runelite",
                script_dir / "runelite",
                Path.home() / "IdeaProjects" / "runelite",
                Path("D:/IdeaProjects/runelite"),
                Path("C:/IdeaProjects/runelite")
            ]
            for path in possible_paths:
                if path.exists() and (path / "pom.xml").exists():
                    self.config_vars["projectDir"].set(str(path))
                    break
            
            # Set relative paths
            cred_dir = script_dir / "credentials"
            if cred_dir.exists():
                self.config_vars["credentialsDir"].set(str(cred_dir))
            
            self.config_vars["baseDir"].set(str(script_dir / "instances"))
            self.config_vars["exportsBase"].set(str(script_dir / "exports"))
            self.config_vars["classPathFile"].set(str(script_dir / "rl-classpath.txt"))
            
            if log_callback:
                log_callback("Auto-detection completed", 'success')
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error during auto-detection: {str(e)}", 'error')
            else:
                logging.error(f"Error during auto-detection: {str(e)}")
    
    def toggle_auto_detect(self):
        """Toggle auto-detect mode and enable/disable path entries."""
        state = 'disabled' if self.config_vars["autoDetect"].get() else 'normal'
        for entry in self.path_entries.values():
            entry.config(state=state)
    
    def browse_path(self, key: str, is_file: bool = False):
        """
        Open file/directory browser for a configuration path.
        
        Args:
            key: Configuration key (e.g., "projectDir", "javaExe")
            is_file: If True, open file dialog; if False, open directory dialog
        """
        from tkinter import filedialog
        
        current_path = self.config_vars[key].get()
        initial_dir = current_path if current_path and Path(current_path).exists() else str(Path.home())
        
        if is_file:
            path = filedialog.askopenfilename(
                title=f"Select {key}",
                initialdir=initial_dir
            )
        else:
            path = filedialog.askdirectory(
                title=f"Select {key}",
                initialdir=initial_dir
            )
        
        if path:
            self.config_vars[key].set(path)
