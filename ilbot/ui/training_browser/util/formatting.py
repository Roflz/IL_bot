"""
Formatting Utilities

Value formatting, CSV export, and text wrapping helpers.
"""

import csv
import numpy as np
from typing import Any, List, Dict
import pyperclip


def format_value_for_display(value: Any, feature_idx: int, show_translations: bool, 
                           translate_func=None) -> str:
    """
    Format a value for display in the GUI.
    
    Args:
        value: Raw value to format
        feature_idx: Feature index for context
        show_translations: Whether to show translations
        translate_func: Optional translation function
        
    Returns:
        Formatted string
    """
    if value is None:
        return "None"
    
    # Handle boolean values
    if isinstance(value, bool):
        return str(value)
    
    # Handle translations if requested - attempt this BEFORE numeric formatting
    if show_translations and translate_func:
        try:
            # Pass int(value) if value is numeric and has no fractional part
            translate_value = value
            if isinstance(value, (int, float)) and float(value).is_integer():
                translate_value = int(value)
            
            translated = translate_func(feature_idx, translate_value)
            if translated != str(value) and translated != str(translate_value):
                return translated
        except Exception:
            pass  # Fall back to numeric formatting
    
    # Handle integers (avoid .0)
    if isinstance(value, (int, float)) and value == int(value):
        return str(int(value))
    
    # Handle floats
    if isinstance(value, float):
        return f"{value:.6f}"
    
    # Default to string representation
    return str(value)


def wrap_text(text: str, width: int = 80) -> str:
    """
    Wrap text to specified width.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        
    Returns:
        Wrapped text
    """
    if not text:
        return ""
    
    # If text fits within width, return as-is
    if len(text) <= width:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        
        # If adding this word would exceed width, start new line
        if current_length + word_length + 1 > width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)
    
    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def export_to_csv(data: List[List[Any]], headers: List[str], filepath: str) -> bool:
    """
    Export data to CSV file.
    
    Args:
        data: 2D list of data values
        headers: List of column headers
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow(headers)
            
            # Write data
            for row in data:
                writer.writerow(row)
        
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to clipboard.
    
    Args:
        text: Text to copy
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pyperclip.copy(text)
        return True
    except Exception as e:
        print(f"Error copying to clipboard: {e}")
        return False


def format_sequence_data(sequence: np.ndarray, feature_names: Dict[str, str], 
                        feature_groups: Dict[str, str], show_translations: bool = False,
                        translate_func=None) -> List[List[str]]:
    """
    Format sequence data for CSV export.
    
    Args:
        sequence: Sequence array (10, 128)
        feature_names: Feature names mapping
        feature_groups: Feature groups mapping
        show_translations: Whether to show translations
        translate_func: Optional translation function
        
    Returns:
        List of formatted rows for CSV
    """
    rows = []
    
    # Add header row
    header = ["Feature", "Index", "Group"]
    for i in range(10):
        header.append(f"Timestep_{i}")
    rows.append(header)
    
    # Add data rows
    for feature_idx in range(128):
        feature_name = feature_names.get(str(feature_idx), f'feature_{feature_idx}')
        feature_group = feature_groups.get(str(feature_idx), 'other')
        
        row = [feature_name, feature_idx, feature_group]
        
        # Add values for each timestep
        for timestep in range(10):
            value = sequence[timestep, feature_idx]
            formatted_value = format_value_for_display(
                value, feature_idx, show_translations, translate_func
            )
            row.append(formatted_value)
        
        rows.append(row)
    
    return rows


def format_target_data(target: List[float], action_decoder=None) -> List[List[str]]:
    """
    Format target data for CSV export.
    
    Args:
        target: Target action sequence
        action_decoder: Optional ActionDecoder for formatting
        
    Returns:
        List of formatted rows for CSV
    """
    rows = []
    
    if not target:
        return [["No actions"]]
    
    # Add header
    action_count = int(target[0]) if target else 0
    if action_count > 0:
        headers = ["Action", "Timestamp", "Type", "X", "Y", "Button", "Key", "Scroll DX", "Scroll DY"]
        rows.append(headers)
        
        # Add action data
        for i in range(action_count):
            base_idx = 1 + i * 8
            if base_idx + 7 < len(target):
                action_data = target[base_idx:base_idx + 8]
                
                # Format with action decoder if available
                if action_decoder:
                    row = [f"Action_{i+1}"]
                    row.extend([str(val) for val in action_data])
                else:
                    row = [f"Action_{i+1}"]
                    row.extend([str(val) for val in action_data])
                
                rows.append(row)
    else:
        rows.append(["No actions"])
    
    return rows
