# widgets.py (actions)

from __future__ import annotations
from typing import Optional
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.widgets import widget_exists, get_widget_info

def click_widget(widget_id: int, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Click on a widget by its ID.
    
    Args:
        widget_id: The widget ID to click
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Check if widget exists and is visible
    if not widget_exists(widget_id, payload):
        return None
    
    # Get widget info to get coordinates
    widget_info = get_widget_info(widget_id, payload)
    if not widget_info:
        return None
    
    widget_data = widget_info.get("data", {})
    bounds = widget_data.get("bounds")
    
    if not bounds:
        return None
    
    # Calculate center coordinates
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    # Click on the widget
    step = emit({
        "action": "widget-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": f"widget_{widget_id}"},
    })
    return ui.dispatch(step)

def click_widget_if_visible(widget_id: int, payload: Optional[dict] = None, ui=None) -> bool:
    """
    Click on a widget by its ID if it's visible.
    
    Args:
        widget_id: The widget ID to click
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        True if clicked successfully, False otherwise
    """
    result = click_widget(widget_id, payload, ui)
    return result is not None

def click_widget_by_name(widget_name: str, payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Click on a widget by its name (for character design widgets).
    
    Args:
        widget_name: The widget name to click (e.g., "HEAD_LEFT", "HAIR_RIGHT")
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get character design widgets
    from ..helpers.widgets import get_all_character_design_buttons
    design_buttons = get_all_character_design_buttons()
    
    if widget_name not in design_buttons:
        return None
    
    widget_data = design_buttons[widget_name]
    bounds = widget_data.get("bounds")
    
    if not bounds:
        return None
    
    # Calculate center coordinates
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    # Click on the widget
    step = emit({
        "action": "widget-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": widget_name},
    })
    return ui.dispatch(step)
