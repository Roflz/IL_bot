"""
Normalization View

View for normalization analysis and comparison.
"""

import tkinter as tk
from tkinter import ttk


class NormalizationView(ttk.Frame):
    """Normalization view for analyzing normalization."""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        self.title_label = ttk.Label(self, text="Normalization Analysis", font=("Tahoma", 12, "bold"))
        self.content_text = tk.Text(self, height=20, width=80, wrap="word", font=("Consolas", 9))
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=self.scrollbar.set)
    
    def _setup_layout(self):
        self.title_label.pack(pady=(10, 5))
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.content_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def refresh(self):
        self._update_display()
    
    def update(self):
        self._update_display()
    
    def _update_display(self):
        self.content_text.delete(1.0, tk.END)
        if not self.controller.is_data_loaded():
            self.content_text.insert(tk.END, "No data loaded")
            return
        
        # Get current sequence data
        sequence_data = self.controller.get_current_sequence_data()
        if not sequence_data:
            self.content_text.insert(tk.END, "No sequence data available")
            return
        
        self.content_text.insert(tk.END, f"Normalization Analysis for Sequence {sequence_data['sequence_idx']}\n")
        self.content_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Compare raw vs normalized data
        raw_features = sequence_data['input_sequence']
        normalized_features = sequence_data.get('normalized_sequence')
        
        self.content_text.insert(tk.END, "Data Comparison:\n")
        self.content_text.insert(tk.END, f"• Raw features shape: {raw_features.shape}\n")
        
        if normalized_features is not None:
            self.content_text.insert(tk.END, f"• Normalized features shape: {normalized_features.shape}\n")
            
            # Show normalization statistics
            self.content_text.insert(tk.END, f"\nNormalization Statistics:\n")
            
            # Compare first few features
            for i in range(min(5, raw_features.shape[1])):
                if len(raw_features.shape) > 1:
                    raw_values = raw_features[:, i]
                    norm_values = normalized_features[:, i]
                    
                    raw_range = f"{raw_values.min():.4f} to {raw_values.max():.4f}"
                    norm_range = f"{norm_values.min():.4f} to {norm_values.max():.4f}"
                    
                    self.content_text.insert(tk.END, f"  Feature {i}: Raw {raw_range} → Normalized {norm_range}\n")
        else:
            self.content_text.insert(tk.END, "• Normalized features: Not available\n")
            self.content_text.insert(tk.END, "\nNote: Use the 'Show Normalized Data' checkbox in the Gamestate Features tab to view normalized data.\n")
        
        self.content_text.configure(state="disabled")
