#!/usr/bin/env python3
"""
GUI for manual background removal on images.
Simple tool to clean images by removing backgrounds with manual masking.
"""

import os
import numpy as np
import colorsys
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Hardcoded configuration
BG_COLOR = "#222222"
FG_COLOR = "#ffffff"
ENTRY_BG = "#333333"
ENTRY_FG = "#ffffff"


class ToolTip:
    """Simple tooltip for Tkinter widgets."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            tw,
            text=self.text,
            background="#222222",
            foreground="#ffffff",
            relief="solid",
            borderwidth=1,
            padding=(5, 2)
        )
        label.pack()

    def hide_tip(self, _event=None):
        tw = self.tipwindow
        if tw:
            tw.destroy()
            self.tipwindow = None


class IconCleaner:
    def __init__(self):
        # State
        self.index = 0
        self.files = []
        self.src_images_dir = ""
        self.output_dir = ""
        self.mask_list = []
        self.zoom_by_image = []
        self.threshold_value = 0.0
        self.single_mode = False
        self.hue_value = 0.0
        self.zoom_extra = 0
        self.win_geometry = None
        self.padding = [0, 0, 0, 0]  # top, bottom, left, right

        # Populated in _load_image_data
        self.orig_arr = None
        self.mask_arr = None
        self.current_crop_box = None

        # Open window
        self._open_window()

    def _load_images(self):
        """Load image list from selected directory."""
        if not self.src_images_dir or not os.path.isdir(self.src_images_dir):
            self.files = []
            self.index = 0
            self.mask_list = []
            self.zoom_by_image = []
            self._update_image()
            return

        self.files = sorted(f for f in os.listdir(self.src_images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        self.index = 0
        self.mask_list = [None] * len(self.files)
        self.zoom_by_image = [0] * len(self.files)
        
        if self.files:
            self._load_image_data()
        else:
            self.orig_arr = None
            self.mask_arr = None
            self._update_image()

    def _load_image_data(self):
        """Load image at self.index into numpy arrays."""
        if not self.files or self.index < 0 or self.index >= len(self.files):
            self.orig_arr = None
            self.mask_arr = None
            return

        fname = self.files[self.index]
        img_path = os.path.join(self.src_images_dir, fname)
        try:
            img = Image.open(img_path).convert("RGBA")
            self.orig_arr = np.array(img)
            self.current_crop_box = None

            # Restore zoom for this image
            self.zoom_extra = self.zoom_by_image[self.index]

            # Restore or init mask
            if self.mask_list[self.index] is not None:
                self.mask_arr = self.mask_list[self.index]
            else:
                self.mask_arr = np.zeros(self.orig_arr.shape[:2], dtype=bool)

            self.zoom_var.set(self.zoom_extra)
            self.filename_label.config(text=f"File: {fname} ({self.index + 1}/{len(self.files)})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            self.orig_arr = None
            self.mask_arr = None

    def _open_window(self):
        """Build the main window."""
        if hasattr(self, 'root') and self.root:
            self.root.destroy()

        self.root = tk.Tk()
        # Apply basic dark theme styling
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', background=BG_COLOR, foreground=FG_COLOR)
        style.map('TEntry', foreground=[('disabled', '#888888')])
        style.configure('Error.TEntry', fieldbackground='#550000')

        self.status_var = tk.StringVar(self.root, value='')
        self.status_color = '#00cc44'

        if self.win_geometry:
            self.root.geometry(self.win_geometry)
        self.root.title("Image Background Cleaner")
        self.root.bind('<Configure>', self._save_geometry)

        # UI vars
        self.threshold_var = tk.DoubleVar(self.root, value=self.threshold_value)
        self.single_var = tk.BooleanVar(self.root, value=self.single_mode)
        self.hue_var = tk.DoubleVar(self.root, value=self.hue_value)
        self.zoom_var = tk.IntVar(self.root, value=self.zoom_extra)
        self.preview_var = tk.BooleanVar(self.root, value=False)
        self.pad_top_var = tk.IntVar(self.root, value=0)
        self.pad_bottom_var = tk.IntVar(self.root, value=0)
        self.pad_left_var = tk.IntVar(self.root, value=0)
        self.pad_right_var = tk.IntVar(self.root, value=0)
        self.thr_label_var = tk.StringVar(self.root, value=f"{self.threshold_var.get():.0f}")
        self.hue_label_var = tk.StringVar(self.root, value=f"{self.hue_var.get():.0f}")
        
        # Directory selection vars
        self.src_dir_var = tk.StringVar(self.root, value="")
        self.output_dir_var = tk.StringVar(self.root, value="")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Top: Directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Directories", padding=5)
        dir_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Images:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        src_entry = ttk.Entry(dir_frame, textvariable=self.src_dir_var, width=50)
        src_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_src_dir).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(dir_frame, text="Output:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        out_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=50)
        out_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_output_dir).grid(row=1, column=2, padx=5, pady=2)

        ttk.Button(dir_frame, text="Load Images", command=self._load_images).grid(row=2, column=0, columnspan=3, pady=5)

        # Left column: controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)

        # Controls
        ctrl_frame = ttk.LabelFrame(left_frame, text="Controls", padding=5)
        ctrl_frame.pack(fill='x', pady=(0, 10))
        ctrl_frame.columnconfigure(1, weight=1)

        # Navigation buttons
        nav_frame = ttk.Frame(ctrl_frame)
        nav_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        btn_prev = ttk.Button(nav_frame, text='◀ Prev', command=self._prev)
        btn_save = ttk.Button(nav_frame, text='Save', command=self._save)
        btn_next = ttk.Button(nav_frame, text='Next ▶', command=self._next)
        btn_clear = ttk.Button(nav_frame, text='Clear', command=self._clear)
        btn_prev.pack(side='left', padx=2)
        btn_save.pack(side='left', padx=2)
        btn_next.pack(side='left', padx=2)
        btn_clear.pack(side='left', padx=2)

        # Single Pixel + Preview
        chk_frame = ttk.Frame(ctrl_frame)
        chk_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=2)
        chk_single = ttk.Checkbutton(chk_frame, text='Single Pixel', variable=self.single_var, command=self._on_single_toggle)
        chk_preview = ttk.Checkbutton(chk_frame, text='Preview', variable=self.preview_var, command=self._update_image)
        chk_single.pack(side='left', padx=5)
        chk_preview.pack(side='left', padx=5)

        # Zoom
        zoom_frame = ttk.Frame(ctrl_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Label(zoom_frame, text='Zoom:').pack(side='left', padx=5)
        vcmd = (self.root.register(self._validate_zoom), '%P')
        spin_zoom = tk.Spinbox(zoom_frame, from_=-9999, to=9999, width=5, textvariable=self.zoom_var,
                              command=self._on_zoom_change, validate='focusout', validatecommand=vcmd)
        spin_zoom.pack(side='left', padx=2)
        spin_zoom.bind('<Return>', lambda e: self._on_zoom_change())
        ttk.Button(zoom_frame, text='Reset', command=self._reset_zoom).pack(side='left', padx=2)

        # Threshold slider
        ttk.Label(ctrl_frame, text='Threshold:').grid(row=3, column=0, sticky='w', padx=5, pady=2)
        slider_thr = ttk.Scale(ctrl_frame, from_=0, to=100, orient='horizontal',
                               variable=self.threshold_var, command=self._on_threshold_change)
        slider_thr.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        lbl_thr_val = ttk.Label(ctrl_frame, textvariable=self.thr_label_var)
        lbl_thr_val.grid(row=3, column=2, padx=5, pady=2)

        # Hue slider
        ttk.Label(ctrl_frame, text='Mask Hue:').grid(row=4, column=0, sticky='w', padx=5, pady=2)
        slider_hue = ttk.Scale(ctrl_frame, from_=0, to=360, orient='horizontal',
                               variable=self.hue_var, command=self._on_hue_change)
        slider_hue.grid(row=4, column=1, sticky='ew', padx=5, pady=2)
        lbl_hue_val = ttk.Label(ctrl_frame, textvariable=self.hue_label_var)
        lbl_hue_val.grid(row=4, column=2, padx=5, pady=2)

        # Padding controls
        pad_frame = ttk.LabelFrame(ctrl_frame, text="Padding", padding=5)
        pad_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Label(pad_frame, text='↑').grid(row=0, column=1)
        spin_up = tk.Spinbox(pad_frame, from_=-64, to=64, width=4, textvariable=self.pad_top_var, command=self._on_pad_change)
        spin_up.grid(row=1, column=1, padx=2)
        ttk.Label(pad_frame, text='←').grid(row=2, column=0)
        spin_l = tk.Spinbox(pad_frame, from_=-64, to=64, width=4, textvariable=self.pad_left_var, command=self._on_pad_change)
        spin_l.grid(row=2, column=1, padx=2)
        ttk.Label(pad_frame, text='→').grid(row=2, column=2)
        spin_r = tk.Spinbox(pad_frame, from_=-64, to=64, width=4, textvariable=self.pad_right_var, command=self._on_pad_change)
        spin_r.grid(row=2, column=3, padx=2)
        ttk.Label(pad_frame, text='↓').grid(row=3, column=1)
        spin_dn = tk.Spinbox(pad_frame, from_=-64, to=64, width=4, textvariable=self.pad_bottom_var, command=self._on_pad_change)
        spin_dn.grid(row=4, column=1, padx=2)
        ttk.Button(pad_frame, text='Reset', command=self._reset_padding).grid(row=2, column=4, padx=5)

        # Right column: canvas + filename
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky='nsew')
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Canvas
        self.canvas = tk.Canvas(right_frame, bg='black')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', lambda e: self._update_image())
        self.canvas.bind('<Button-1>', self._on_click)

        # Filename label below canvas
        self.filename_label = ttk.Label(right_frame, text="No images loaded", font=("Helvetica", 10, "italic"),
                                       foreground=FG_COLOR, background=BG_COLOR)
        self.filename_label.grid(row=1, column=0, sticky='ew', pady=(5, 0))

        self.status_label = ttk.Label(right_frame, textvariable=self.status_var, font=("Helvetica", 9, "italic"),
                                      foreground=self.status_color, background=BG_COLOR)
        self.status_label.grid(row=2, column=0, sticky='ew')

        # Initial display
        self.current_crop_box = None
        self.zoom_var.set(0)
        self._update_image()
        self.root.mainloop()

    def _browse_src_dir(self):
        """Browse for source images directory."""
        dir_path = filedialog.askdirectory(title="Select Images Directory", initialdir=self.src_dir_var.get())
        if dir_path:
            self.src_dir_var.set(dir_path)
            self.src_images_dir = dir_path

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir_var.get())
        if dir_path:
            self.output_dir_var.set(dir_path)
            self.output_dir = dir_path
            os.makedirs(self.output_dir, exist_ok=True)

    def _update_image(self):
        """Redraw canvas."""
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if c_w < 2 or c_h < 2:
            return

        if self.orig_arr is None:
            self.canvas.delete('all')
            self.canvas.create_text(c_w//2, c_h//2, text='Load images to start',
                                   fill=FG_COLOR)
            return

        arr = self.orig_arr
        mask_slice = self.mask_arr

        x1, y1, x2, y2 = self._get_view_box()

        arr = self.orig_arr[y1:y2, x1:x2]
        mask_slice = self.mask_arr[y1:y2, x1:x2]
        display_arr = arr
        if self.preview_var.get():
            display_arr = arr.copy()
            display_arr[mask_slice, 3] = 0

        h, w = arr.shape[:2]
        scale = min(c_w / w, c_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = Image.fromarray(display_arr).resize((new_w, new_h), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img, master=self.canvas)

        self.canvas.delete('all')
        x0, y0 = (c_w - new_w) // 2, (c_h - new_h) // 2
        self.canvas.create_image(x0, y0, anchor='nw', image=self.photo)
        self.canvas.image = self.photo

        # Overlay mask pixels unless previewing transparency
        if mask_slice is not None and not self.preview_var.get():
            hue = self.hue_value / 360.0
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            hex_color = f'#{int(r_f*255):02x}{int(g_f*255):02x}{int(b_f*255):02x}'
            length = max(2, int(scale))

            ys, xs = np.where(mask_slice)
            for oy, ox in zip(ys, xs):
                cx = x0 + int(ox * scale)
                cy = y0 + int(oy * scale)
                self.canvas.create_line(cx, cy, cx+length, cy+length, fill=hex_color)
                self.canvas.create_line(cx+length, cy, cx, cy+length, fill=hex_color)

    def _on_click(self, event):
        """Toggle mask on click."""
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        x1, y1, x2, y2 = self._get_view_box()
        display_w, display_h = x2 - x1, y2 - y1

        scale = min(c_w / display_w, c_h / display_h)
        new_w, new_h = int(display_w * scale), int(display_h * scale)
        x0, y0 = (c_w - new_w)//2, (c_h - new_h)//2

        ox = (event.x - x0) * display_w // new_w
        oy = (event.y - y0) * display_h // new_h
        if not (0 <= ox < display_w and 0 <= oy < display_h):
            return
        ox += x1
        oy += y1

        if self.single_mode:
            if self.mask_arr[oy, ox]:
                self.mask_arr[oy, ox] = False
            else:
                self.mask_arr[oy, ox] = True
        else:
            bgr = self.orig_arr[oy, ox, :3].astype(float)
            diff = np.linalg.norm(self.orig_arr[:, :, :3].astype(float) - bgr, axis=2)
            region = diff <= self.threshold_value
            if self.mask_arr[oy, ox]:
                self.mask_arr[region] = False
            else:
                self.mask_arr[region] = True

        self._update_image()

    def _clear(self):
        self.mask_arr[:] = False
        self._update_image()

    def _get_view_box(self):
        """Return the bounding box of the region currently displayed."""
        if self.current_crop_box is None:
            x1 = y1 = 0
            x2 = self.orig_arr.shape[1]
            y2 = self.orig_arr.shape[0]
        else:
            x1, y1, x2, y2 = self.current_crop_box
            x1 += self.zoom_extra
            y1 += self.zoom_extra
            x2 -= self.zoom_extra
            y2 -= self.zoom_extra
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.orig_arr.shape[1], x2)
            y2 = min(self.orig_arr.shape[0], y2)
            if x2 <= x1:
                x1 = max(0, x2 - 1)
            if y2 <= y1:
                y1 = max(0, y2 - 1)

        x1 = max(0, x1 - self.pad_left_var.get())
        y1 = max(0, y1 - self.pad_top_var.get())
        x2 = min(self.orig_arr.shape[1], x2 + self.pad_right_var.get())
        y2 = min(self.orig_arr.shape[0], y2 + self.pad_bottom_var.get())
        return x1, y1, x2, y2

    def _save_state(self):
        if 0 <= self.index < len(self.mask_list):
            self.mask_list[self.index] = self.mask_arr.copy()
            self.zoom_by_image[self.index] = self.zoom_extra

    def _save(self):
        if not self.files or self.index < 0 or self.index >= len(self.files):
            return

        if not self.output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return

        self._save_state()
        self._clear_status()

        x1, y1, x2, y2 = self._get_view_box()
        arr = self.orig_arr[y1:y2, x1:x2].copy()
        mask_slice = self.mask_arr[y1:y2, x1:x2]
        arr[mask_slice, 3] = 0

        img = Image.fromarray(arr)

        # Save with original filename
        fname = self.files[self.index]
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(self.output_dir, f"{base_name}_cleaned.png")

        overwrite = os.path.exists(out_path)
        img.save(out_path)

        rel = os.path.relpath(out_path, self.output_dir)
        msg = f"Saved {rel}" + (" (overwrote)" if overwrite else "")
        self._set_status(msg, '#00cc44')

        # Reset zoom and padding after saving
        self.zoom_extra = 0
        self.zoom_var.set(0)
        if 0 <= self.index < len(self.zoom_by_image):
            self.zoom_by_image[self.index] = 0
        for var in (self.pad_top_var, self.pad_bottom_var, self.pad_left_var, self.pad_right_var):
            var.set(0)
        self.padding = [0, 0, 0, 0]
        self._update_image()

        self._next(preserve_status=True)

    def _prev(self):
        if not self.files:
            return
        self._save_state()
        self.index = (self.index - 1) % len(self.files)
        self._load_image_data()
        self._update_image()

    def _next(self, preserve_status=False):
        if not self.files:
            return
        self._save_state()
        self.index = (self.index + 1) % len(self.files)
        self._load_image_data()
        if not preserve_status:
            self._clear_status()
        self._update_image()

    def _skip(self):
        self._next()

    def _save_geometry(self, event):
        if event.widget == self.root:
            self.win_geometry = self.root.geometry()

    def _on_threshold_change(self, val):
        self.threshold_value = float(val)
        self.thr_label_var.set(f"{self.threshold_value:.0f}")
        self._update_image()

    def _on_hue_change(self, val):
        self.hue_value = float(val)
        self.hue_label_var.set(f"{self.hue_value:.0f}")
        self._update_image()

    def _on_zoom_change(self):
        try:
            v = int(self.zoom_var.get())
        except tk.TclError:
            v = 0
        if self.orig_arr is None:
            return
        if self.current_crop_box is None:
            x1, y1, x2, y2 = 0, 0, self.orig_arr.shape[1], self.orig_arr.shape[0]
        else:
            x1, y1, x2, y2 = self.current_crop_box
        max_in = min((x2 - x1 - 1) // 2, (y2 - y1 - 1) // 2) if (x2 > x1 and y2 > y1) else 0
        max_out = min(x1, y1, self.orig_arr.shape[1] - x2, self.orig_arr.shape[0] - y2)
        v = max(-max_out, min(max_in, v))
        self.zoom_extra = v
        self.zoom_var.set(v)
        self._update_image()

    def _on_pad_change(self):
        self.padding = [
            self.pad_top_var.get(),
            self.pad_bottom_var.get(),
            self.pad_left_var.get(),
            self.pad_right_var.get(),
        ]
        self._update_image()

    def _reset_zoom(self):
        self.zoom_extra = 0
        self.zoom_var.set(0)
        self._update_image()
        self._clear_status()

    def _reset_padding(self):
        for var in (self.pad_top_var, self.pad_bottom_var, self.pad_left_var, self.pad_right_var):
            var.set(0)
        self.padding = [0, 0, 0, 0]
        self._update_image()
        self._clear_status()

    def _on_single_toggle(self):
        self.single_mode = self.single_var.get()

    def _validate_zoom(self, value: str) -> bool:
        try:
            v = int(value)
        except ValueError:
            return False
        self.zoom_var.set(v)
        self.zoom_extra = v
        if hasattr(self, "canvas"):
            self._on_zoom_change()
        return True

    def _set_status(self, text: str, color: str = '#00cc44'):
        """Display a status message."""
        self.status_label.config(foreground=color)
        self.status_var.set(text)

    def _clear_status(self):
        self.status_var.set('')
        self.status_label.config(foreground='#00cc44')


if __name__ == '__main__':
    IconCleaner()
