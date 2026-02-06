"""
RuneLite Launcher Module
========================

Handles launching and managing RuneLite instances.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Callable
import subprocess
import threading
import os
import logging
from pathlib import Path


class RuneLiteLauncher:
    """Manages RuneLite instance launching and credential management."""
    
    def __init__(self, root, config_vars: dict, base_port_var: tk.IntVar, 
                 launch_delay_var: tk.IntVar, build_maven_var: tk.BooleanVar,
                 credentials_listbox: tk.Listbox, selected_credentials_listbox: tk.Listbox,
                 selected_credentials: List[str], log_callback: Optional[Callable] = None,
                 instance_count_label: Optional[ttk.Label] = None,
                 launch_button: Optional[ttk.Button] = None,
                 create_instance_tab_callback: Optional[Callable] = None):
        """
        Initialize RuneLite launcher.
        
        Args:
            root: Root tkinter window
            config_vars: Configuration variables dictionary
            base_port_var: Variable for base port number
            launch_delay_var: Variable for launch delay
            build_maven_var: Variable for build Maven checkbox
            credentials_listbox: Listbox for available credentials
            selected_credentials_listbox: Listbox for selected credentials
            selected_credentials: List to store selected credential names
            log_callback: Optional callback for logging messages
            instance_count_label: Optional label to update instance count
            launch_button: Optional launch button to disable during launch
            create_instance_tab_callback: Optional callback to create instance tabs
        """
        self.root = root
        self.config_vars = config_vars
        self.base_port_var = base_port_var
        self.launch_delay_var = launch_delay_var
        self.build_maven_var = build_maven_var
        self.credentials_listbox = credentials_listbox
        self.selected_credentials_listbox = selected_credentials_listbox
        self.selected_credentials = selected_credentials
        self.log_callback = log_callback or (lambda msg, level='info': None)
        self.instance_count_label = instance_count_label
        self.launch_button = launch_button
        self.create_instance_tab_callback = create_instance_tab_callback
        self.runelite_process = None
        self.monitor_thread = None
    
    def setup_dependencies(self, save_config_callback: Optional[Callable] = None):
        """Run setup-dependencies.ps1 script."""
        try:
            script_path = Path(__file__).resolve().parent.parent / "setup-dependencies.ps1"
            if not script_path.exists():
                messagebox.showerror("Script Not Found", f"Setup script not found at {script_path}")
                return
            
            self.log_callback("Running dependency setup...", 'info')
            
            ps_cmd = [
                "powershell.exe",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"& '{script_path}'"
            ]
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                ps_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            def monitor_setup():
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.log_callback(f"[Setup] {output.strip()}", 'info')
                        self.root.update_idletasks()
                
                return_code = process.wait()
                if return_code == 0:
                    self.log_callback("Dependency setup completed successfully", 'success')
                    messagebox.showinfo("Setup Complete", "Dependency setup completed successfully!")
                else:
                    self.log_callback(f"Dependency setup failed with return code {return_code}", 'error')
                    messagebox.showerror("Setup Failed", f"Dependency setup failed. Check logs for details.")
            
            threading.Thread(target=monitor_setup, daemon=True).start()
            
        except Exception as e:
            self.log_callback(f"Error running setup script: {str(e)}", 'error')
            messagebox.showerror("Setup Error", f"Failed to run setup script: {str(e)}")
    
    def populate_credentials(self):
        """Populate credentials listbox from credentials directory."""
        credentials_dir = Path(__file__).resolve().parent.parent / "credentials"
        logging.info(f"[GUI] Credentials directory: {credentials_dir}")
        if credentials_dir.exists():
            for cred_file in sorted(credentials_dir.glob("*.properties")):
                self.credentials_listbox.insert(tk.END, cred_file.name)
        else:
            self.log_callback(f"Credentials directory not found: {credentials_dir}", 'error')
    
    def refresh_credentials(self):
        """Refresh the credentials list."""
        try:
            # Best-effort: remember selection by filename
            selected_names = set()
            try:
                for idx in self.credentials_listbox.curselection():
                    selected_names.add(self.credentials_listbox.get(idx))
            except Exception:
                selected_names = set()
            
            self.credentials_listbox.delete(0, tk.END)
            self.populate_credentials()
            
            # Restore selection
            if selected_names:
                for i in range(self.credentials_listbox.size()):
                    name = self.credentials_listbox.get(i)
                    if name in selected_names:
                        self.credentials_listbox.selection_set(i)
            
            self.log_callback(f"Refreshed credentials list", 'info')
        except Exception as e:
            self.log_callback(f"Failed to refresh credentials: {e}", 'error')
    
    def add_credential(self):
        """Add selected credential to launch list."""
        selection = self.credentials_listbox.curselection()
        for index in selection:
            cred_name = self.credentials_listbox.get(index)
            if cred_name not in self.selected_credentials:
                self.selected_credentials.append(cred_name)
        self.update_selected_credentials_display()
    
    def remove_credential(self):
        """Remove selected credential from launch list."""
        selection = self.selected_credentials_listbox.curselection()
        for index in reversed(selection):  # Reverse to maintain indices
            self.selected_credentials.pop(index)
        self.update_selected_credentials_display()
    
    def clear_credentials(self):
        """Clear all selected credentials."""
        self.selected_credentials = []
        self.update_selected_credentials_display()
    
    def move_credential_up(self):
        """Move selected credential up in the list."""
        selection = self.selected_credentials_listbox.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            item = self.selected_credentials.pop(index)
            self.selected_credentials.insert(index - 1, item)
            self.update_selected_credentials_display()
            self.selected_credentials_listbox.selection_set(index - 1)
    
    def move_credential_down(self):
        """Move selected credential down in the list."""
        selection = self.selected_credentials_listbox.curselection()
        if selection and selection[0] < len(self.selected_credentials) - 1:
            index = selection[0]
            item = self.selected_credentials.pop(index)
            self.selected_credentials.insert(index + 1, item)
            self.update_selected_credentials_display()
            self.selected_credentials_listbox.selection_set(index + 1)
    
    def update_selected_credentials_display(self):
        """Update the selected credentials display."""
        self.selected_credentials_listbox.delete(0, tk.END)
        for i, cred_name in enumerate(self.selected_credentials):
            self.selected_credentials_listbox.insert(tk.END, f"{i+1}. {cred_name}")
        
        # Update instance count display
        if self.instance_count_label:
            self.instance_count_label.config(text=str(len(self.selected_credentials)))
    
    def launch_runelite(self, save_config_callback: Optional[Callable] = None):
        """Launch RuneLite instances for selected credentials."""
        if not self.selected_credentials:
            messagebox.showwarning("No Credentials Selected", "Please select at least one credential file.")
            return
        
        instance_count = len(self.selected_credentials)
        if instance_count <= 0:
            messagebox.showerror("Invalid Configuration", "Instance count must be greater than 0.")
            return
        
        # Save config before launching
        if save_config_callback:
            save_config_callback()
        
        try:
            # Build PowerShell command
            script_path = Path(__file__).resolve().parent.parent / "launch-runelite.ps1"
            config_path = Path(__file__).resolve().parent.parent / "launch-config.json"
            logging.info(f"[GUI] RuneLite launcher script path: {script_path}")
            if not script_path.exists():
                messagebox.showerror("Script Not Found", f"RuneLite launcher script not found at {script_path}")
                return
            
            # Create credential files array for PowerShell
            cred_files = "', '".join(self.selected_credentials)
            cred_files = f"@('{cred_files}')"
            
            # Build PowerShell command with config file
            build_maven_flag = "-BuildMaven" if self.build_maven_var.get() else ""
            ps_cmd = [
                "powershell.exe",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"& '{script_path}' -Count {instance_count} -BasePort {self.base_port_var.get()} -DelaySeconds {self.launch_delay_var.get()} -CredentialFiles {cred_files} -ConfigFile '{config_path}' {build_maven_flag}"
            ]
            
            self.log_callback(f"Launching RuneLite instances...", 'info')
            self.log_callback(f"Command: {' '.join(ps_cmd)}", 'info')
            
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Launch PowerShell script
            self.runelite_process = subprocess.Popen(
                ps_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_runelite_launch, daemon=True)
            self.monitor_thread.start()
            
            # Create instance tabs immediately after successful launch start
            if self.create_instance_tab_callback:
                self.root.after(2000, self.create_instance_tabs)  # Wait 2 seconds for instances to start
            
            if self.launch_button:
                self.launch_button.config(state=tk.DISABLED)
            self.log_callback("RuneLite launcher started", 'success')
            
        except Exception as e:
            self.log_callback(f"Error launching RuneLite: {str(e)}", 'error')
            messagebox.showerror("Launch Error", f"Failed to launch RuneLite instances: {str(e)}")
    
    def monitor_runelite_launch(self):
        """Monitor the RuneLite launcher process."""
        try:
            while True:
                if not self.runelite_process:
                    break
                
                output = self.runelite_process.stdout.readline()
                if output == '' and self.runelite_process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    if line:
                        self.log_callback(f"[RuneLite] {line}", 'info')
                        self.root.update_idletasks()
            
            # Process completed
            return_code = self.runelite_process.wait()
            self.runelite_process = None
            
            if return_code == 0:
                self.log_callback("RuneLite instances launched successfully", 'success')
            else:
                self.log_callback(f"RuneLite launcher failed with return code {return_code}", 'error')
            
        except Exception as e:
            self.log_callback(f"Error monitoring RuneLite launcher: {str(e)}", 'error')
        finally:
            if self.launch_button:
                self.launch_button.config(state=tk.NORMAL)
    
    def create_instance_tabs(self):
        """Create tabs for each launched RuneLite instance."""
        if not self.create_instance_tab_callback:
            return
        
        try:
            self.log_callback("Starting to create instance tabs...", 'info')
            base_port = self.base_port_var.get()
            self.log_callback(f"Base port: {base_port}, Selected credentials: {self.selected_credentials}", 'info')
            
            for i, cred_name in enumerate(self.selected_credentials):
                # Extract username from credential filename (remove .properties)
                username = cred_name.replace('.properties', '')
                port = base_port + i
                
                self.log_callback(f"Creating tab for {username} on port {port}", 'info')
                
                # Create the instance tab via callback
                self.create_instance_tab_callback(username, port)
                
                self.log_callback(f"Created tab for {username} on port {port}", 'info')
                
        except Exception as e:
            self.log_callback(f"Error creating instance tabs: {str(e)}", 'error')
            import traceback
            self.log_callback(f"Traceback: {traceback.format_exc()}", 'error')
    
    def stop_runelite(self, stop_all_instances_callback: Optional[Callable] = None):
        """Stop all running RuneLite instances."""
        try:
            # First stop all running plans
            self.log_callback("Stopping all running plans...", 'info')
            if stop_all_instances_callback:
                stop_all_instances_callback()
            
            # Then stop RuneLite instances using the PID file
            pid_file = Path("D:/bots/instances/runelite-pids.txt")
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 1:
                            try:
                                pid = int(parts[0])
                                # Try to terminate the process
                                import psutil
                                process = psutil.Process(pid)
                                process.terminate()
                                self.log_callback(f"Stopped RuneLite instance PID {pid}", 'info')
                            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                                pass
                
                # Remove PID file
                pid_file.unlink()
                self.log_callback("All RuneLite instances stopped", 'success')
            else:
                self.log_callback("No RuneLite instances found to stop", 'warning')
                
        except Exception as e:
            self.log_callback(f"Error stopping RuneLite instances: {str(e)}", 'error')
            messagebox.showerror("Stop Error", f"Failed to stop RuneLite instances: {str(e)}")
