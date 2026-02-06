"""
Client Detector Module
======================

Handles automatic detection of running RuneLite clients.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional
import psutil
import logging


class ClientDetector:
    """Detects and manages running RuneLite client instances."""
    
    def __init__(self, root, notebook: ttk.Notebook, instance_tabs: Dict, instance_ports: Dict, 
                 detected_clients: Dict, detection_status_label: ttk.Label, log_callback=None):
        """
        Initialize client detector.
        
        Args:
            root: Root tkinter window
            notebook: Notebook widget containing instance tabs
            instance_tabs: Dictionary of instance tabs
            instance_ports: Dictionary mapping instance names to ports
            detected_clients: Dictionary tracking detected clients
            detection_status_label: Label for detection status
            log_callback: Optional callback for logging messages
        """
        self.root = root
        self.notebook = notebook
        self.instance_tabs = instance_tabs
        self.instance_ports = instance_ports
        self.detected_clients = detected_clients
        self.detection_status_label = detection_status_label
        self.detection_running = False
        self.log_callback = log_callback
        self.create_instance_tab_callback = None
        self.remove_instance_tab_callback = None
    
    def start_client_detection(self):
        """Start automatic client detection."""
        if not self.detection_running:
            self.detection_running = True
            self.detection_status_label.config(text="Auto-detection: Running", style='Success.TLabel')
            if hasattr(self, 'log_callback'):
                self.log_callback("Started automatic client detection", 'info')
            self.detect_running_clients()
            # Schedule next detection in 5 seconds
            self.root.after(5000, self._client_detection_loop)
    
    def stop_client_detection(self):
        """Stop automatic client detection."""
        self.detection_running = False
        self.detection_status_label.config(text="Auto-detection: Stopped", style='Info.TLabel')
        if hasattr(self, 'log_callback'):
            self.log_callback("Stopped automatic client detection", 'info')
    
    def detect_running_clients(self, create_instance_tab_callback=None, 
                               remove_instance_tab_callback=None, log_callback=None):
        """
        Detect currently running RuneLite clients.
        
        Args:
            create_instance_tab_callback: Callback to create instance tabs (username, port)
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
            detected_ports = set()
            for proc in runelite_processes:
                try:
                    # Check if process has network connections on our port range
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            detected_ports.add(conn.laddr.port)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Create instance tabs for detected ports
            for port in detected_ports:
                instance_name = f"detected_{port}"
                if instance_name not in self.instance_tabs:
                    log_callback(f"Detected RuneLite client on port {port}", 'info')
                    if create_instance_tab_callback:
                        create_instance_tab_callback(instance_name, port)
                    self.detected_clients[instance_name] = port
            
            # Remove instance tabs for clients that are no longer running
            clients_to_remove = []
            for instance_name, port in self.detected_clients.items():
                if port not in detected_ports:
                    clients_to_remove.append(instance_name)
            
            for instance_name in clients_to_remove:
                log_callback(f"RuneLite client on port {self.detected_clients[instance_name]} no longer running", 'info')
                if remove_instance_tab_callback:
                    remove_instance_tab_callback(instance_name)
                del self.detected_clients[instance_name]
                
        except Exception as e:
            log_callback(f"Error detecting clients: {e}", 'error')
    
    def _client_detection_loop(self):
        """Internal method for client detection loop."""
        if self.detection_running:
            self.detect_running_clients()
            # Schedule next detection in 5 seconds
            self.root.after(5000, self._client_detection_loop)
    
    def test_client_detection(self, log_callback=None):
        """
        Test method to debug client detection.
        
        Args:
            log_callback: Callback for logging messages
        """
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
                            java_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': ' '.join(cmdline[:3]) + '...' if len(cmdline) > 3 else ' '.join(cmdline)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            log_callback(f"Found {len(java_processes)} Java processes:", 'info')
            for proc in java_processes:
                log_callback(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline']}", 'info')
            
            # Check for RuneLite processes specifically
            runelite_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'java' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('runelite' in arg.lower() for arg in cmdline):
                            runelite_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            log_callback(f"Found {len(runelite_processes)} RuneLite processes", 'info')
            
            # Check for IPC ports
            detected_ports = set()
            for proc in runelite_processes:
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr and 17000 <= conn.laddr.port <= 17099:
                            detected_ports.add(conn.laddr.port)
                            log_callback(f"  Found IPC port {conn.laddr.port} on PID {proc.pid}", 'info')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if detected_ports:
                log_callback(f"Detected IPC ports: {sorted(detected_ports)}", 'info')
            else:
                log_callback("No IPC ports detected in range 17000-17099", 'info')
                
        except Exception as e:
            log_callback(f"Error testing client detection: {e}", 'error')
    
    def remove_instance_tab(self, instance_name: str, stop_plans_callback=None, 
                           stop_stats_monitor_callback=None):
        """
        Remove an instance tab and clean up references.
        
        Args:
            instance_name: Name of the instance to remove
            stop_plans_callback: Optional callback to stop plans (instance_name)
            stop_stats_monitor_callback: Optional callback to stop stats monitor (instance_name)
        """
        if instance_name in self.instance_tabs:
            # Stop any running plans for this instance
            instance_tab = self.instance_tabs[instance_name]
            if hasattr(instance_tab, 'is_running') and instance_tab.is_running:
                if stop_plans_callback:
                    stop_plans_callback(instance_name)
            
            # Remove the tab from the notebook
            self.notebook.forget(instance_tab)
            
            # Clean up references
            del self.instance_tabs[instance_name]
            if instance_name in self.instance_ports:
                del self.instance_ports[instance_name]
            
            # Stop stats monitor for this instance
            if stop_stats_monitor_callback:
                stop_stats_monitor_callback(instance_name)
