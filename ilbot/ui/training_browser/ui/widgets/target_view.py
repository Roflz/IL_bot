"""
Target View

Text/pretty view for flattened target frames.
"""

import tkinter as tk
from tkinter import ttk


class TargetView(ttk.Frame):
    """Target view for displaying action targets."""
    
    def __init__(self, parent, controller):
        """
        Initialize the target view.
        
        Args:
            parent: Parent widget
            controller: Application controller
        """
        super().__init__(parent)
        self.controller = controller
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create the target view widgets."""
        # Title label
        self.title_label = ttk.Label(
            self, 
            text="Target Actions",
            font=("Tahoma", 12, "bold")
        )
        
        # Target display text
        self.target_text = tk.Text(
            self,
            height=20,
            width=80,
            wrap="word",
            font=("Consolas", 9)
        )
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.target_text.yview)
        self.target_text.configure(yscrollcommand=self.scrollbar.set)
    
    def _setup_layout(self):
        """Setup the widget layout."""
        self.title_label.pack(pady=(10, 5))
        
        # Text widget and scrollbar
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.target_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def refresh(self):
        """Refresh the target view display."""
        self._update_display()
    
    def update(self):
        """Update the target view display."""
        self._update_display()
    
    def _update_display(self):
        """Update the target display."""
        if not self.controller.is_data_loaded():
            self.target_text.delete(1.0, tk.END)
            self.target_text.insert(tk.END, "No data loaded")
            return
        
        # Get current sequence data
        sequence_data = self.controller.get_current_sequence_data()
        if not sequence_data:
            self.target_text.delete(1.0, tk.END)
            self.target_text.insert(tk.END, "No sequence data available")
            return
        
        target_sequence = sequence_data.get('target_sequence')
        if not target_sequence:
            self.target_text.delete(1.0, tk.END)
            self.target_text.insert(tk.END, "No target sequence available")
            return
        
        # Clear text widget first
        self.target_text.delete(1.0, tk.END)
        
        # Display target sequence
        self.target_text.insert(tk.END, f"Target Sequence for Sequence {sequence_data['sequence_idx']}\n")
        self.target_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Format and display target data
        action_decoder = self.controller.get_action_decoder()
        if action_decoder:
            try:
                formatted = action_decoder.format_action_summary(target_sequence)
                self.target_text.insert(tk.END, formatted)
            except Exception as e:
                self.target_text.insert(tk.END, f"Error formatting actions: {e}\n\n")
                self.target_text.insert(tk.END, f"Raw target data: {target_sequence}")
        else:
            # Fallback display
            self.target_text.insert(tk.END, f"Raw target data: {target_sequence}")
        
        # Make text read-only
        self.target_text.configure(state="disabled")
