#!/usr/bin/env python3
"""UI package for Bot Controller GUI"""

from .main_window import MainWindow
from .styles import (
    apply_dark_theme, 
    create_dark_text, 
    create_dark_canvas, 
    create_dark_stringvar, 
    create_dark_booleanvar, 
    create_dark_intvar
)

__all__ = [
    'MainWindow',
    'apply_dark_theme',
    'create_dark_text',
    'create_dark_canvas',
    'create_dark_stringvar',
    'create_dark_booleanvar',
    'create_dark_intvar'
]
