"""
Client Detector Module (PySide6)
================================

Automatic detection of running RuneLite clients.
"""

import psutil
from pathlib import Path
from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import QTimer
from typing import Dict, Optional, Callable


class ClientDetector:
    """Automatic detection of running RuneLite clients."""
    
    def __init__(self, root: QWidget, detected_clients: Dict, 
                 client_detection_running: bool, detection_status_label: Optional[QLabel] = None,
                 project_dir: Optional[Path] = None, base_dir: Optional[Path] = None, credentials_dir: Optional[Path] = None):
        """Initialize client detector."""
        self.root = root
        self.detected_clients = detected_clients
        self.client_detection_running = client_detection_running
        self.detection_status_label = detection_status_label
        self.create_instance_tab_callback = None
        self.remove_instance_tab_callback = None
        self.detection_timer = QTimer()
        self.project_dir = project_dir or Path.cwd()
        self.base_dir = base_dir or (self.project_dir / "instances")
        self.credentials_dir = credentials_dir or (self.project_dir / "credentials")
        # Store port -> credential name mapping
        self.port_to_credential: Dict[int, str] = {}
    
    def start_client_detection(self):
        """Start automatic client detection."""
        # TODO: Implement
        pass
    
    def stop_client_detection(self):
        """Stop automatic client detection."""
        # TODO: Implement
        pass
    
    def detect_running_clients(self, create_instance_tab_callback=None, 
                              remove_instance_tab_callback=None, log_callback=None):
        """
        Detect currently running RuneLite clients.
        
        Args:
            create_instance_tab_callback: Callback to create instance tabs (instance_name, port)
            remove_instance_tab_callback: Callback to remove instance tabs (instance_name)
            log_callback: Callback for logging messages
        """
        if not log_callback:
            log_callback = getattr(self, 'log_callback', lambda msg, level='info': None)
        if not create_instance_tab_callback:
            create_instance_tab_callback = getattr(self, 'create_instance_tab_callback', None)
        if not remove_instance_tab_callback:
            remove_instance_tab_callback = getattr(self, 'remove_instance_tab_callback', None)
        
        try:
            # Find RuneLite processes
            runelite_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('runelite' in arg.lower() for arg in cmdline):
                            runelite_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for IPC ports in the range we use (17000-17099)
            # Map port -> (process, instance_dir)
            detected_ports_info: Dict[int, tuple] = {}
            for proc in runelite_processes:
                try:
                    # Check if process has network connections on our port range
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            port = conn.laddr.port
                            # Try to find instance directory from process working directory or PID file
                            instance_dir = self._find_instance_directory_for_port(port, proc, log_callback)
                            detected_ports_info[port] = (proc, instance_dir)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            detected_ports = set(detected_ports_info.keys())
            
            # Create instance tabs for detected ports
            for port in detected_ports:
                instance_name = f"detected_{port}"
                if instance_name not in self.detected_clients:
                    proc, instance_dir = detected_ports_info[port]
                    
                    # Try to get credential name from instance directory
                    credential_name = self._get_credential_name_from_instance(instance_dir, port, log_callback)
                    self.port_to_credential[port] = credential_name
                    
                    log_callback(f"Detected RuneLite client on port {port} (credential: {credential_name})", 'info')
                    if create_instance_tab_callback:
                        create_instance_tab_callback(instance_name, port)
                    self.detected_clients[instance_name] = port
            
            # Remove instance tabs for clients that are no longer running
            clients_to_remove = []
            for instance_name, port in list(self.detected_clients.items()):
                if port not in detected_ports:
                    clients_to_remove.append(instance_name)
            
            for instance_name in clients_to_remove:
                port = self.detected_clients[instance_name]
                log_callback(f"RuneLite client on port {port} no longer running", 'info')
                if remove_instance_tab_callback:
                    remove_instance_tab_callback(instance_name)
                del self.detected_clients[instance_name]
                # Clean up credential mapping
                if port in self.port_to_credential:
                    del self.port_to_credential[port]
                
        except Exception as e:
            log_callback(f"Error detecting clients: {e}", 'error')
    
    def _find_instance_directory_for_port(self, port: int, proc: psutil.Process, log_callback) -> Optional[Path]:
        """Find the instance directory for a given port by checking PID file or process working directory."""
        try:
            # First, try reading PID file to find instance directory
            # PID file is stored in base_dir (where instances are stored)
            pid_file = self.base_dir / "runelite-pids.txt"
            
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            try:
                                pid = int(parts[0])
                                inst_index = int(parts[1])
                                inst_port = int(parts[2])
                                inst_dir = Path(parts[3])
                                
                                if pid == proc.pid and inst_port == port:
                                    return inst_dir
                            except (ValueError, IndexError):
                                continue
            
            # Fallback: check process working directory
            try:
                cwd = Path(proc.cwd())
                if 'inst_' in cwd.name and self.base_dir in cwd.parents:
                    return cwd
            except (psutil.AccessDenied, AttributeError):
                pass
        except Exception as e:
            if log_callback:
                log_callback(f"Error finding instance directory for port {port}: {e}", 'warning')
        return None
    
    def _get_credential_name_from_instance(self, instance_dir: Optional[Path], port: int, log_callback) -> str:
        """Extract credential name from instance directory credentials.properties file."""
        if not instance_dir:
            return "Unknown"
        
        try:
            cred_file = instance_dir / ".runelite" / "credentials.properties"
            if not cred_file.exists():
                return "Unknown"
            
            # Read credentials.properties to get JX_DISPLAY_NAME
            username = None
            with open(cred_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('JX_DISPLAY_NAME='):
                        username = line.split('=', 1)[1].strip()
                        break
            
            if not username:
                return "Unknown"
            
            # Try to match with available credential files
            if self.credentials_dir.exists():
                for cred_file_path in self.credentials_dir.glob("*.properties"):
                    # Check if the credential file contains this username
                    # Or if the filename matches the username
                    if cred_file_path.stem.lower() == username.lower():
                        return cred_file_path.stem
                    
                    # Also check if the credential file has matching JX_DISPLAY_NAME
                    try:
                        with open(cred_file_path, 'r') as cf:
                            for line in cf:
                                if line.startswith('JX_DISPLAY_NAME='):
                                    cred_username = line.split('=', 1)[1].strip()
                                    if cred_username.lower() == username.lower():
                                        return cred_file_path.stem
                                    break
                    except Exception:
                        continue
            
            # Return the username if we can't match to a credential file
            return username
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error reading credential for port {port}: {e}", 'warning')
            return "Unknown"
    
    def get_credential_for_port(self, port: int) -> str:
        """Get credential name for a given port."""
        credential = self.port_to_credential.get(port, "Unknown")
        # Log for debugging (but this might be called frequently, so maybe not always)
        return credential
    
    def test_client_detection(self, log_callback=None):
        """Test client detection functionality."""
        if not log_callback:
            log_callback = getattr(self, 'log_callback', lambda msg, level='info': None)
        
        try:
            log_callback("Testing client detection...", 'info')
            
            # Find all Java processes
            java_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline:
                            cmdline_str = ' '.join(cmdline)
                            if 'runelite' in cmdline_str.lower():
                                java_processes.append(proc)
                                log_callback(f"Found RuneLite Java process: PID {proc.info['pid']}", 'info')
                                log_callback(f"  Command line: {cmdline_str[:200]}...", 'info')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for IPC ports
            detected_ports = set()
            for proc in java_processes:
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            detected_ports.add(conn.laddr.port)
                            log_callback(f"Found IPC port: {conn.laddr.port}", 'info')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if detected_ports:
                log_callback(f"Detected {len(detected_ports)} RuneLite client(s) on ports: {sorted(detected_ports)}", 'success')
            else:
                log_callback("No RuneLite clients detected on IPC ports (17000-17099)", 'warning')
                
        except Exception as e:
            log_callback(f"Error testing client detection: {e}", 'error')
