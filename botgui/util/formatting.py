#!/usr/bin/env python3
"""Formatting utilities for displaying values"""

from typing import Any, Optional, Callable
import numpy as np


def format_value_for_display(
    value: Any, 
    feature_idx: int, 
    show_translations: bool, 
    translate_func: Optional[Callable[[int, Any], Optional[str]]] = None
) -> str:
    """
    Format a value for display in the GUI.
    
    Args:
        value: The raw value to format
        feature_idx: Feature index for translation lookup
        show_translations: Whether to attempt translation
        translate_func: Function to translate values (feature_idx, raw_value) -> str
        
    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"
    
    # Try translation first if enabled
    if show_translations and translate_func:
        try:
            translated = translate_func(feature_idx, value)
            if translated and translated != str(value):
                return translated
        except Exception:
            pass  # Fall back to normal formatting
    
    # Format based on type
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            # Format floats with appropriate precision
            if abs(value) < 0.01 and value != 0:
                return f"{value:.6f}"
            elif abs(value) < 1:
                return f"{value:.4f}"
            elif abs(value) < 100:
                return f"{value:.2f}"
            else:
                return f"{value:.1f}"
        else:
            return str(value)
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return format_value_for_display(value.item(), feature_idx, show_translations, translate_func)
        else:
            return f"Array({value.shape})"
    else:
        return str(value)


def format_timestamp(timestamp: float) -> str:
    """Format a timestamp for display"""
    if timestamp is None:
        return "—"
    
    try:
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return f"{timestamp:.1f}s"


def format_buffer_status(status: dict) -> str:
    """Format buffer status for display"""
    features_count = status.get('features_count', 0)
    actions_count = status.get('actions_count', 0)
    source_mode = status.get('source_mode', 'unknown')
    
    if status.get('is_warm', False):
        status_text = f"Active | {source_mode} | {features_count}/10 features, {actions_count}/10 actions"
    else:
        status_text = f"Warming | {source_mode} | {features_count}/10 features, {actions_count}/10 actions"
    
    # Add age if available
    age = status.get('age_seconds')
    if age is not None:
        if age < 60:
            status_text += f" | {age:.1f}s ago"
        else:
            status_text += f" | {age/60:.1f}m ago"
    
    return status_text


def format_prediction_summary(prediction: np.ndarray) -> str:
    """Format a prediction summary for display"""
    if prediction is None or len(prediction) == 0:
        return "No prediction"
    
    try:
        count = int(prediction[0])
        if count == 0:
            return "No actions predicted"
        
        return f"{count} actions predicted"
    except Exception:
        return "Invalid prediction format"


def format_feature_name(feature_info: dict) -> str:
    """Format a feature name for display"""
    if not feature_info:
        return "Unknown"
    
    name = feature_info.get('feature_name', 'Unknown')
    group = feature_info.get('feature_group', 'Unknown')
    
    return f"{name} ({group})"


def format_action_type(action_type: int, action_encoder=None) -> str:
    """Format an action type for display"""
    if action_encoder:
        try:
            return action_encoder.get_action_type_name(action_type)
        except Exception:
            pass
    
    # Fallback formatting
    return f"Type {action_type}"


def format_button_type(button_type: int, action_encoder=None) -> str:
    """Format a button type for display"""
    if action_encoder:
        try:
            return action_encoder.get_button_name(button_type)
        except Exception:
            pass
    
    # Fallback formatting
    return f"Button {button_type}"


def format_key_value(key_value: int, action_encoder=None) -> str:
    """Format a key value for display"""
    if action_encoder:
        try:
            return action_encoder.get_key_name(key_value)
        except Exception:
            pass
    
    # Fallback formatting
    if key_value == 0:
        return "None"
    else:
        return f"Key {key_value}"
