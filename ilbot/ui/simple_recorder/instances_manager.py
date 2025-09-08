import tkinter as tk
from tkinter import ttk

try:
    # package import
    from .main_window import SimpleRecorderWindow
except ImportError:
    # fallback when running as a script
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from ilbot.ui.simple_recorder.main_window import SimpleRecorderWindow


class MultiInstanceHost(ttk.Frame):
    """
    Container that provides:
      - A top row with an 'Add Instance' button
      - A ttk.Notebook where each tab contains a full SimpleRecorderWindow
    """
    def __init__(self, master):
        super().__init__(master)
        self.grid(sticky="nsew")

        # make this frame stretch
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Controls row (top)
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        controls.grid_columnconfigure(0, weight=1)

        self.add_btn = ttk.Button(controls, text="Add Instance", command=self.add_instance)
        self.add_btn.grid(row=0, column=1, sticky="e")

        # Notebook (tabs)
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky="nsew")

        self._instances = []   # list[SimpleRecorderWindow]
        self._counter = 0

        # create the first instance by default
        self.add_instance()

    def add_instance(self):
        self._counter += 1
        idx = self._counter - 1  # 0-based
        tab = ttk.Frame(self.nb)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        label = f"Instance {self._counter:02d}"
        self.nb.add(tab, text=label)

        # Build your full UI inside this tab
        instance = SimpleRecorderWindow(tab, instance_index=idx)
        instance.grid(row=0, column=0, sticky="nsew")
        self._instances.append(instance)

        # Focus the new tab
        self.nb.select(len(self._instances) - 1)
