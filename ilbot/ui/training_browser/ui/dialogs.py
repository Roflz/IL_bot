"""
Dialog Boxes

Common dialog boxes for the training browser.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable


class ChangeDataRootDialog:
    """Dialog for changing the data root directory."""
    
    def __init__(self, parent, current_root: str):
        """
        Initialize the dialog.
        
        Args:
            parent: Parent window
            current_root: Current data root path
        """
        self.parent = parent
        self.current_root = current_root
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Change Data Root")
        self.dialog.geometry("500x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Title label
        self.title_label = ttk.Label(
            self.dialog, 
            text="Select new data root directory:",
            font=("Tahoma", 10, "bold")
        )
        
        # Current path display
        self.current_label = ttk.Label(
            self.dialog,
            text=f"Current: {self.current_root}",
            font=("Tahoma", 9)
        )
        
        # Path entry
        self.path_var = tk.StringVar(value=self.current_root)
        self.path_entry = ttk.Entry(
            self.dialog,
            textvariable=self.path_var,
            width=60,
            state="readonly"
        )
        
        # Browse button
        self.browse_button = ttk.Button(
            self.dialog,
            text="Browse...",
            command=self._browse_directory
        )
        
        # Buttons frame
        self.buttons_frame = ttk.Frame(self.dialog)
        
        self.ok_button = ttk.Button(
            self.buttons_frame,
            text="OK",
            command=self._on_ok
        )
        
        self.cancel_button = ttk.Button(
            self.buttons_frame,
            text="Cancel",
            command=self._on_cancel
        )
    
    def _setup_layout(self):
        """Setup dialog layout."""
        # Title
        self.title_label.pack(pady=(20, 10))
        
        # Current path
        self.current_label.pack(pady=(0, 10))
        
        # Path selection frame
        path_frame = ttk.Frame(self.dialog)
        path_frame.pack(pady=(0, 20), padx=20, fill="x")
        
        self.path_entry.pack(side="left", fill="x", expand=True)
        self.browse_button.pack(side="right", padx=(10, 0))
        
        # Buttons
        self.buttons_frame.pack(pady=(0, 20))
        self.ok_button.pack(side="left", padx=(0, 10))
        self.cancel_button.pack(side="left")
        
        # Bind Enter/Escape keys
        self.dialog.bind("<Return>", lambda e: self._on_ok())
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
        
        # Focus on browse button
        self.browse_button.focus_set()
    
    def _browse_directory(self):
        """Open directory browser."""
        directory = filedialog.askdirectory(
            title="Select Data Root Directory",
            initialdir=self.current_root
        )
        if directory:
            self.path_var.set(directory)
    
    def _on_ok(self):
        """Handle OK button click."""
        new_path = self.path_var.get()
        if new_path and new_path != self.current_root:
            self.result = new_path
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.dialog.destroy()
    
    def show(self) -> Optional[str]:
        """
        Show the dialog and wait for result.
        
        Returns:
            New data root path if changed, None if cancelled
        """
        self.dialog.wait_window()
        return self.result


class SearchDialog:
    """Dialog for searching features."""
    
    def __init__(self, parent):
        """
        Initialize the dialog.
        
        Args:
            parent: Parent window
        """
        self.parent = parent
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Search Features")
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 100,
            parent.winfo_rooty() + 100
        ))
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Search label
        self.search_label = ttk.Label(
            self.dialog,
            text="Search for feature:",
            font=("Tahoma", 10)
        )
        
        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(
            self.dialog,
            textvariable=self.search_var,
            width=40
        )
        
        # Options frame
        self.options_frame = ttk.Frame(self.dialog)
        
        self.case_sensitive_var = tk.BooleanVar(value=False)
        self.case_sensitive_check = ttk.Checkbutton(
            self.options_frame,
            text="Case sensitive",
            variable=self.case_sensitive_var
        )
        
        self.whole_word_var = tk.BooleanVar(value=False)
        self.whole_word_check = ttk.Checkbutton(
            self.options_frame,
            text="Whole word",
            variable=self.whole_word_var
        )
        
        # Buttons frame
        self.buttons_frame = ttk.Frame(self.dialog)
        
        self.search_button = ttk.Button(
            self.buttons_frame,
            text="Search",
            command=self._on_search
        )
        
        self.cancel_button = ttk.Button(
            self.buttons_frame,
            text="Cancel",
            command=self._on_cancel
        )
    
    def _setup_layout(self):
        """Setup dialog layout."""
        # Search label
        self.search_label.pack(pady=(20, 10))
        
        # Search entry
        self.search_entry.pack(pady=(0, 15))
        
        # Options
        self.options_frame.pack(pady=(0, 15))
        self.case_sensitive_check.pack(side="left", padx=(0, 20))
        self.whole_word_check.pack(side="left")
        
        # Buttons
        self.buttons_frame.pack()
        self.search_button.pack(side="left", padx=(0, 10))
        self.cancel_button.pack(side="left")
        
        # Bind Enter/Escape keys
        self.dialog.bind("<Return>", lambda e: self._on_search())
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
        
        # Focus on search entry
        self.search_entry.focus_set()
    
    def _on_search(self):
        """Handle search button click."""
        search_text = self.search_var.get().strip()
        if search_text:
            self.result = {
                'text': search_text,
                'case_sensitive': self.case_sensitive_var.get(),
                'whole_word': self.whole_word_var.get()
            }
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.dialog.destroy()
    
    def show(self) -> Optional[dict]:
        """
        Show the dialog and wait for result.
        
        Returns:
            Search parameters if search initiated, None if cancelled
        """
        self.dialog.wait_window()
        return self.result


class ExportDialog:
    """Dialog for exporting data."""
    
    def __init__(self, parent, default_filename: str = "export.csv"):
        """
        Initialize the dialog.
        
        Args:
            parent: Parent window
            default_filename: Default filename for export
        """
        self.parent = parent
        self.default_filename = default_filename
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Export Data")
        self.dialog.geometry("500x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Title label
        self.title_label = ttk.Label(
            self.dialog,
            text="Export data to file:",
            font=("Tahoma", 10, "bold")
        )
        
        # Filename entry
        self.filename_var = tk.StringVar(value=self.default_filename)
        self.filename_entry = ttk.Entry(
            self.dialog,
            textvariable=self.filename_var,
            width=50
        )
        
        # Browse button
        self.browse_button = ttk.Button(
            self.dialog,
            text="Browse...",
            command=self._browse_file
        )
        
        # Format selection
        self.format_label = ttk.Label(
            self.dialog,
            text="Export format:",
            font=("Tahoma", 9)
        )
        
        self.format_var = tk.StringVar(value="CSV")
        self.format_combo = ttk.Combobox(
            self.dialog,
            textvariable=self.format_var,
            values=["CSV", "JSON", "TXT"],
            state="readonly",
            width=10
        )
        
        # Buttons frame
        self.buttons_frame = ttk.Frame(self.dialog)
        
        self.export_button = ttk.Button(
            self.buttons_frame,
            text="Export",
            command=self._on_export
        )
        
        self.cancel_button = ttk.Button(
            self.buttons_frame,
            text="Cancel",
            command=self._on_cancel
        )
    
    def _setup_layout(self):
        """Setup dialog layout."""
        # Title
        self.title_label.pack(pady=(20, 15))
        
        # Filename selection frame
        filename_frame = ttk.Frame(self.dialog)
        filename_frame.pack(pady=(0, 15), padx=20, fill="x")
        
        self.filename_entry.pack(side="left", fill="x", expand=True)
        self.browse_button.pack(side="right", padx=(10, 0))
        
        # Format selection frame
        format_frame = ttk.Frame(self.dialog)
        format_frame.pack(pady=(0, 20), padx=20)
        
        self.format_label.pack(side="left")
        self.format_combo.pack(side="right")
        
        # Buttons
        self.buttons_frame.pack()
        self.export_button.pack(side="left", padx=(0, 10))
        self.cancel_button.pack(side="left")
        
        # Bind Enter/Escape keys
        self.dialog.bind("<Return>", lambda e: self._on_export())
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
        
        # Focus on filename entry
        self.filename_entry.focus_set()
    
    def _browse_file(self):
        """Open file save dialog."""
        filetypes = [
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Export File",
            filetypes=filetypes,
            defaultextension=".csv"
        )
        if filename:
            self.filename_var.set(filename)
    
    def _on_export(self):
        """Handle export button click."""
        filename = self.filename_var.get().strip()
        if filename:
            self.result = {
                'filename': filename,
                'format': self.format_var.get()
            }
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.dialog.destroy()
    
    def show(self) -> Optional[dict]:
        """
        Show the dialog and wait for result.
        
        Returns:
            Export parameters if export initiated, None if cancelled
        """
        self.dialog.wait_window()
        return self.result


def show_info_dialog(parent, title: str, message: str):
    """Show an information dialog."""
    messagebox.showinfo(title, message, parent=parent)


def show_error_dialog(parent, title: str, message: str):
    """Show an error dialog."""
    messagebox.showerror(title, message, parent=parent)


def show_warning_dialog(parent, title: str, message: str):
    """Show a warning dialog."""
    messagebox.showwarning(title, message, parent=parent)


def show_yes_no_dialog(parent, title: str, message: str) -> bool:
    """Show a yes/no dialog."""
    return messagebox.askyesno(title, message, parent=parent)
