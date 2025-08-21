#!/usr/bin/env python3
"""Logs View - displays bot logs and status messages"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional
import time
from ..styles import create_dark_booleanvar, create_dark_text


class LogsView(ttk.Frame):
    """View for displaying bot logs and status messages"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # UI state
        self.auto_scroll = True
        self.max_lines = 1000
        
        self._setup_ui()
        self._bind_events()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Log area gets most space
        
        # Header
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Bot Logs", 
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        
        # Controls frame
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Left controls
        ttk.Button(controls_frame, text="📋 Copy Logs", 
                  command=self._copy_logs).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls_frame, text="💾 Save Logs", 
                  command=self._save_logs).grid(row=0, column=1, padx=(0, 12))
        ttk.Button(controls_frame, text="🗑️ Clear Logs", 
                  command=self._clear_logs).grid(row=0, column=2, padx=(0, 12))
        
        # Right controls
        self.auto_scroll_var = create_dark_booleanvar(self, value=True)
        ttk.Checkbutton(controls_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).grid(row=0, column=3)
        
        # Log area
        self.log_text = scrolledtext.ScrolledText(
            self, 
            wrap=tk.WORD, 
            height=20,
            font=("Consolas", 9),
            bg="#202225",
            fg="#e6e6e6"
        )
        self.log_text.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        
        # Configure tags for different log levels - optimized for dark theme
        self.log_text.tag_configure("info", foreground="#4e8ef7")      # Blue/cyan for info
        self.log_text.tag_configure("warning", foreground="#ffa500")   # Orange for warnings
        self.log_text.tag_configure("error", foreground="#ff6b6b")     # Bright red for errors
        self.log_text.tag_configure("success", foreground="#51cf66")   # Bright green for success
        self.log_text.tag_configure("debug", foreground="#868e96")     # Muted gray for debug
        self.log_text.tag_configure("timestamp", foreground="#74c0fc") # Light blue for timestamps
        
        # Initial log message
        self.log("Bot Controller GUI initialized", level="info")
    
    def _bind_events(self):
        """Bind UI events"""
        # Bind mouse wheel to scroll
        self.log_text.bind('<MouseWheel>', self._on_mousewheel)
        
        # Bind key events
        self.log_text.bind('<Control-a>', self._select_all)
        self.log_text.bind('<Control-c>', self._copy_selection)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.log_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _select_all(self, event):
        """Select all text (Ctrl+A)"""
        self.log_text.tag_add(tk.SEL, "1.0", tk.END)
        return "break"
    
    def _copy_selection(self, event):
        """Copy selected text (Ctrl+C)"""
        try:
            selected_text = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except tk.TclError:
            # No selection
            pass
        return "break"
    
    def log(self, message: str, level: str = "info"):
        """
        Add a log message to the view.
        
        Args:
            message: The log message
            level: Log level (info, warning, error, success, debug)
        """
        # Get current timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Format the log line
        log_line = f"[{timestamp}] {message}\n"
        
        # Insert at the end
        self.log_text.insert(tk.END, log_line, (level, "timestamp"))
        
        # Apply level-specific formatting
        start_pos = f"{self.log_text.index(tk.END).split('.')[0]}.0"
        end_pos = f"{self.log_text.index(tk.END).split('.')[0]}.1"
        self.log_text.tag_add("timestamp", start_pos, end_pos)
        
        # Limit the number of lines
        self._limit_lines()
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            self.log_text.see(tk.END)
    
    def _limit_lines(self):
        """Limit the number of lines in the log"""
        lines = int(self.log_text.index(tk.END).split('.')[0])
        if lines > self.max_lines:
            # Remove excess lines from the beginning
            excess = lines - self.max_lines
            self.log_text.delete("1.0", f"{excess + 1}.0")
    
    def _copy_logs(self):
        """Copy all logs to clipboard"""
        try:
            all_text = self.log_text.get("1.0", tk.END)
            self.clipboard_clear()
            self.clipboard_append(all_text)
            self.log("Logs copied to clipboard", level="success")
        except Exception as e:
            self.log(f"Failed to copy logs: {e}", level="error")
    
    def _save_logs(self):
        """Save logs to a file"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            all_text = self.log_text.get("1.0", tk.END)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(all_text)
            
            self.log(f"Logs saved to {filename}", level="success")
            
        except Exception as e:
            self.log(f"Failed to save logs: {e}", level="error")
    
    def _clear_logs(self):
        """Clear all logs"""
        self.log_text.delete("1.0", tk.END)
        self.log("Logs cleared", level="info")
    
    def log_info(self, message: str):
        """Log an info message"""
        self.log(message, level="info")
    
    def log_warning(self, message: str):
        """Log a warning message"""
        self.log(message, level="warning")
    
    def log_error(self, message: str):
        """Log an error message"""
        self.log(message, level="error")
    
    def log_success(self, message: str):
        """Log a success message"""
        self.log(message, level="success")
    
    def log_debug(self, message: str):
        """Log a debug message"""
        self.log(message, level="debug")
    
    def clear(self):
        """Clear all logs"""
        self._clear_logs()
    
    def add_log_message(self, message: str, level: str = "info"):
        """
        Add a log message (alias for log method for compatibility)
        
        Args:
            message: The log message
            level: Log level (info, warning, error, success, debug)
        """
        self.log(message, level)
