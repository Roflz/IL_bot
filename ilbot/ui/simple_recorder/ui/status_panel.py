# ilbot/ui/simple_recorder/ui/status_panel.py
import os
import tkinter as tk
from tkinter import ttk, filedialog

class InstanceStatusPanel(ttk.LabelFrame):
    """
    Left-side status panel showing:
      - Selected Window title
      - Gamestate directory (pickable)
      - IPC Port (editable)
    NOTE: Pure move from main_window; no behavior changes.
    """
    def __init__(self, master, *, on_choose_dir, window_title_var, session_dir_var, ipc_port_var):
        super().__init__(master, text="Instance")
        self.grid_columnconfigure(1, weight=1)

        # Window
        ttk.Label(self, text="Window:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        self._window_lbl = ttk.Label(self, textvariable=window_title_var, width=36)
        self._window_lbl.grid(row=0, column=1, sticky="ew", padx=6, pady=(6, 2))

        # Gamestate dir
        ttk.Label(self, text="Gamestate Dir:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        gs_frame = ttk.Frame(self)
        gs_frame.grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        gs_frame.grid_columnconfigure(0, weight=1)
        self._gs_entry = ttk.Entry(gs_frame, textvariable=session_dir_var)
        self._gs_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(gs_frame, text="Browse", command=on_choose_dir).grid(row=0, column=1, padx=(6, 0))

        # IPC port (free text)
        ttk.Label(self, text="IPC Port:").grid(row=2, column=0, sticky="w", padx=6, pady=(2, 8))
        self._port_entry = ttk.Entry(self, textvariable=ipc_port_var, width=12)
        self._port_entry.grid(row=2, column=1, sticky="w", padx=6, pady=(2, 8))
