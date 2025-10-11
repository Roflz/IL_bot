# widgets.py (actions)

from __future__ import annotations
from typing import Optional, List, Dict, Any
import time
import json
from datetime import datetime
from ..helpers.runtime_utils import ui, dispatch, ipc
from ..helpers.widgets import widget_exists, get_widget_info

# Timing instrumentation
_TIMING_ENABLED = True
_TIMING_FILE = "widgets.timing.jsonl"

def _mark_timing(label: str) -> int:
    """Mark a timing point and return nanoseconds timestamp."""
    return time.perf_counter_ns()

def _emit_timing(data: dict):
    """Emit timing data as JSONL."""
    if not _TIMING_ENABLED:
        return
    
    data["ts"] = datetime.now().isoformat()
    print(f"[WIDGET_TIMING] {json.dumps(data)}")
    
    # Also write to file
    try:
        with open(_TIMING_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass  # Don't fail if we can't write to file

def click_widget(widget_id: int) -> Optional[dict]:
    """
    Click on a widget by its ID.
    
    Args:
        widget_id: The widget ID to click
    
    Returns:
        UI dispatch result or None if failed
    """
    # Timing instrumentation
    t0_start = _mark_timing("start")
    timing_data = {
        "phase": "widget_click_timing",
        "widget_id": widget_id,
        "ok": True,
        "error": None,
        "dur_ms": {},
        "counts": {"checks": 0}
    }
    
    try:
        # Check if widget exists and is visible
        t1_check_exists = _mark_timing("check_exists")
        timing_data["dur_ms"]["check_exists"] = (t1_check_exists - t0_start) / 1_000_000
        timing_data["counts"]["checks"] += 1
        
        if not widget_exists(widget_id):
            timing_data["ok"] = False
            timing_data["error"] = "Widget does not exist"
            _emit_timing(timing_data)
            return None
        
        # Get widget info to get coordinates
        t2_get_info = _mark_timing("get_info")
        timing_data["dur_ms"]["get_info"] = (t2_get_info - t1_check_exists) / 1_000_000
        
        widget_info = get_widget_info(widget_id)
        if not widget_info:
            timing_data["ok"] = False
            timing_data["error"] = "Could not get widget info"
            _emit_timing(timing_data)
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds:
            timing_data["ok"] = False
            timing_data["error"] = "No bounds found"
            _emit_timing(timing_data)
            return None
        
        # Calculate center coordinates
        t3_calculate = _mark_timing("calculate")
        timing_data["dur_ms"]["calculate"] = (t3_calculate - t2_get_info) / 1_000_000
        
        x = bounds.get("x", 0) + bounds.get("width", 0) // 2
        y = bounds.get("y", 0) + bounds.get("height", 0) // 2
        
        # Click on the widget
        t4_dispatch = _mark_timing("dispatch")
        timing_data["dur_ms"]["dispatch"] = (t4_dispatch - t3_calculate) / 1_000_000
        
        step = {
            "action": "widget-click",
            "click": {"type": "point", "x": int(x), "y": int(y)},
            "target": {"domain": "widget", "name": f"widget_{widget_id}"},
        }
        result = dispatch(step)
        
        t5_complete = _mark_timing("complete")
        timing_data["dur_ms"]["total"] = (t5_complete - t0_start) / 1_000_000
        _emit_timing(timing_data)
        
        return result
    
    except Exception as e:
        t6_error = _mark_timing("error")
        timing_data["ok"] = False
        timing_data["error"] = str(e)
        timing_data["dur_ms"]["total"] = (t6_error - t0_start) / 1_000_000
        _emit_timing(timing_data)
        return None

def click_widget_by_text(widget_id: int, text: str) -> Optional[dict]:
    """
    Click on a widget by its ID and text content.
    
    Args:
        widget_id: The widget ID to search within
        text: The text content to find and click
    
    Returns:
        UI dispatch result or None if failed
    """
    widgets_data = ipc.get_ge_widgets()
    if not widgets_data or not widgets_data.get("ok"):
        return None
    
    widgets = widgets_data.get("widgets", [])
    for widget in widgets:
        if widget.get("id") == widget_id and text.lower() in widget.get("text", "").lower():
            bounds = widget.get("bounds")
            if bounds:
                x = bounds.get("x", 0) + bounds.get("width", 0) // 2
                y = bounds.get("y", 0) + bounds.get("height", 0) // 2
                
                step = {
                    "action": "widget-click",
                    "click": {"type": "point", "x": int(x), "y": int(y)},
                    "target": {"domain": "widget", "name": f"widget_{widget_id}_text_{text}"},
                }
                return dispatch(step)
    return None

def get_widget_text(widget_id: int) -> Optional[str]:
    """
    Get the text content of a widget.
    
    Args:
        widget_id: The widget ID to get text from
    
    Returns:
        Widget text content or None if not found
    """
    widget_info = get_widget_info(widget_id)
    if not widget_info:
        return None
    
    widget_data = widget_info.get("data", {})
    return widget_data.get("text", "")

def get_widget_bounds(widget_id: int) -> Optional[Dict[str, int]]:
    """
    Get the bounds of a widget.
    
    Args:
        widget_id: The widget ID to get bounds from
    
    Returns:
        Widget bounds dict with x, y, width, height or None if not found
    """
    widget_info = get_widget_info(widget_id)
    if not widget_info:
        return None
    
    widget_data = widget_info.get("data", {})
    return widget_data.get("bounds")

def is_widget_visible(widget_id: int) -> bool:
    """
    Check if a widget is visible.
    
    Args:
        widget_id: The widget ID to check
    
    Returns:
        True if widget is visible, False otherwise
    """
    widget_info = get_widget_info(widget_id)
    if not widget_info:
        return False
    
    widget_data = widget_info.get("data", {})
    return widget_data.get("visible", False)

def has_widget_listener(widget_id: int) -> bool:
    """
    Check if a widget has a click listener.
    
    Args:
        widget_id: The widget ID to check
    
    Returns:
        True if widget has a listener, False otherwise
    """
    widget_info = get_widget_info(widget_id)
    if not widget_info:
        return False
    
    widget_data = widget_info.get("data", {})
    return widget_data.get("hasListener", False)

def find_widgets_by_text(text: str, parent_widget_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Find widgets containing specific text.
    
    Args:
        text: Text to search for
        parent_widget_id: Optional parent widget ID to search within
    
    Returns:
        List of widgets containing the text
    """
    widgets_data = ipc.get_ge_widgets()
    if not widgets_data or not widgets_data.get("ok"):
        return []
    
    widgets = widgets_data.get("widgets", [])
    matching_widgets = []
    
    for widget in widgets:
        widget_text = widget.get("text", "")
        if text.lower() in widget_text.lower():
            if parent_widget_id is None or widget.get("id") == parent_widget_id:
                matching_widgets.append(widget)
    
    return matching_widgets

def find_widgets_by_sprite(sprite_id: int, parent_widget_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Find widgets with a specific sprite ID.
    
    Args:
        sprite_id: Sprite ID to search for
        parent_widget_id: Optional parent widget ID to search within
    
    Returns:
        List of widgets with the sprite ID
    """
    widgets_data = ipc.get_ge_widgets()
    if not widgets_data or not widgets_data.get("ok"):
        return []
    
    widgets = widgets_data.get("widgets", [])
    matching_widgets = []
    
    for widget in widgets:
        if widget.get("spriteId") == sprite_id:
            if parent_widget_id is None or widget.get("id") == parent_widget_id:
                matching_widgets.append(widget)
    
    return matching_widgets

def get_widget_center(widget_id: int) -> Optional[tuple[int, int]]:
    """
    Get the center coordinates of a widget.
    
    Args:
        widget_id: The widget ID to get center coordinates from
    
    Returns:
        Tuple of (x, y) center coordinates or None if not found
    """
    bounds = get_widget_bounds(widget_id)
    if not bounds:
        return None
    
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    return (int(x), int(y))

def click_widget_relative(widget_id: int, offset_x: int = 0, offset_y: int = 0) -> Optional[dict]:
    """
    Click on a widget with a relative offset from its center.
    
    Args:
        widget_id: The widget ID to click
        offset_x: X offset from center (positive = right, negative = left)
        offset_y: Y offset from center (positive = down, negative = up)
    
    Returns:
        UI dispatch result or None if failed
    """
    center = get_widget_center(widget_id)
    if not center:
        return None
    
    x, y = center
    x += offset_x
    y += offset_y
    
    step = {
        "action": "widget-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": f"widget_{widget_id}_offset_{offset_x}_{offset_y}"},
    }
    return dispatch(step)

def right_click_widget(widget_id: int) -> Optional[dict]:
    """
    Right-click on a widget by its ID.
    
    Args:
        widget_id: The widget ID to right-click
    
    Returns:
        UI dispatch result or None if failed
    """
    center = get_widget_center(widget_id)
    if not center:
        return None
    
    x, y = center
    
    step = {
        "action": "widget-right-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": f"widget_{widget_id}_right"},
    }
    return dispatch(step)

def hover_widget(widget_id: int) -> Optional[dict]:
    """
    Hover over a widget by its ID.
    
    Args:
        widget_id: The widget ID to hover over
    
    Returns:
        UI dispatch result or None if failed
    """
    center = get_widget_center(widget_id)
    if not center:
        return None
    
    x, y = center
    
    step = {
        "action": "widget-hover",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": f"widget_{widget_id}_hover"},
    }
    return dispatch(step)

def get_widget_children(parent_widget_id: int) -> List[Dict[str, Any]]:
    """
    Get all child widgets of a parent widget using IPC command.
    
    Args:
        parent_widget_id: The parent widget ID
    
    Returns:
        List of child widgets
    """
    # Use the IPC method to get widget children
    children_data = ipc.get_widget_children(parent_widget_id)
    if not children_data:
        return []
    
    return children_data

def get_widget_child_with_text(parent_widget_id: int, search_text: str, case_sensitive: bool = False) -> Optional[Dict[str, Any]]:
    """
    Find a child widget that contains the specified text in its text field.
    
    Args:
        parent_widget_id: The parent widget ID to search within
        search_text: The text to search for in child widgets
        case_sensitive: Whether the search should be case sensitive (default: False)
    
    Returns:
        The first child widget that contains the text, or None if not found
    """
    children = get_widget_children(parent_widget_id)
    if children.get('ok') == False:
        return None
    if not children or not children.get('children'):
        return None
    
    # Prepare search text based on case sensitivity
    if not case_sensitive:
        search_text = search_text.lower()
    
    for child in children.get('children'):
        child_text = child.get("text", "")
        if not child_text:
            continue
            
        # Prepare child text based on case sensitivity
        if not case_sensitive:
            child_text = child_text.lower()
        
        # Check if search text is contained in child text
        if search_text in child_text:
            return child
    
    return None

def get_widget_children_with_text(parent_widget_id: int, search_text: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Find all child widgets that contain the specified text in their text field.
    
    Args:
        parent_widget_id: The parent widget ID to search within
        search_text: The text to search for in child widgets
        case_sensitive: Whether the search should be case sensitive (default: False)
    
    Returns:
        List of child widgets that contain the text
    """
    children = get_widget_children(parent_widget_id).get('children')
    if not children:
        return []
    
    # Prepare search text based on case sensitivity
    if not case_sensitive:
        search_text = search_text.lower()
    
    matching_children = []
    for child in children:
        child_text = child.get("text", "")
        if not child_text:
            continue
            
        # Prepare child text based on case sensitivity
        if not case_sensitive:
            child_text = child_text.lower()
        
        # Check if search text is contained in child text
        if search_text in child_text:
            matching_children.append(child)
    
    return matching_children

def get_widget_children_ge(parent_widget_id: int) -> List[Dict[str, Any]]:
    """
    Get all child widgets of a parent widget from GE widgets data.
    
    Args:
        parent_widget_id: The parent widget ID
    
    Returns:
        List of child widgets
    """
    widgets_data = ipc.get_ge_widgets()
    if not widgets_data or not widgets_data.get("ok"):
        return []
    
    widgets = widgets_data.get("widgets", [])
    children = []
    
    for widget in widgets:
        # This is a simplified approach - in practice you'd need to check parent-child relationships
        # based on the widget hierarchy structure
        if widget.get("parent_id") == parent_widget_id:
            children.append(widget)
    
    return children

def wait_for_widget(widget_id: int, timeout: float = 5.0) -> bool:
    """
    Wait for a widget to become visible.
    
    Args:
        widget_id: The widget ID to wait for
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if widget becomes visible within timeout, False otherwise
    """
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if widget_exists(widget_id) and is_widget_visible(widget_id):
            return True
        time.sleep(0.1)
    
    return False

def wait_for_widget_text(widget_id: int, expected_text: str, timeout: float = 5.0) -> bool:
    """
    Wait for a widget to contain specific text.
    
    Args:
        widget_id: The widget ID to wait for
        expected_text: The text to wait for
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if widget contains expected text within timeout, False otherwise
    """
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        widget_text = get_widget_text(widget_id)
        if widget_text and expected_text.lower() in widget_text.lower():
            return True
        time.sleep(0.1)
    
    return False

def get_widget_item_info(widget_id: int) -> Optional[Dict[str, Any]]:
    """
    Get item information from a widget (if it contains an item).
    
    Args:
        widget_id: The widget ID to get item info from
    
    Returns:
        Dict with item information or None if not found
    """
    widget_info = get_widget_info(widget_id)
    if not widget_info:
        return None
    
    widget_data = widget_info.get("data", {})
    item_id = widget_data.get("itemId", -1)
    item_quantity = widget_data.get("itemQuantity", 0)
    
    if item_id == -1:
        return None
    
    return {
        "itemId": item_id,
        "itemQuantity": item_quantity,
        "text": widget_data.get("text", ""),
        "spriteId": widget_data.get("spriteId", -1)
    }

def click_widget_if_visible(widget_id: int) -> Optional[dict]:
    """
    Click on a widget only if it's visible.
    
    Args:
        widget_id: The widget ID to click
    
    Returns:
        UI dispatch result or None if widget not visible or failed
    """
    if not is_widget_visible(widget_id):
        return None
    
    return click_widget(widget_id)

def get_widget_at_position(x: int, y: int) -> Optional[Dict[str, Any]]:
    """
    Get the widget at a specific screen position.
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        Widget data or None if no widget at position
    """
    widgets_data = ipc.get_ge_widgets()
    if not widgets_data or not widgets_data.get("ok"):
        return None
    
    widgets = widgets_data.get("widgets", [])
    
    for widget in widgets:
        bounds = widget.get("bounds")
        if bounds:
            widget_x = bounds.get("x", 0)
            widget_y = bounds.get("y", 0)
            widget_width = bounds.get("width", 0)
            widget_height = bounds.get("height", 0)
            
            if (widget_x <= x <= widget_x + widget_width and 
                widget_y <= y <= widget_y + widget_height):
                return widget
    
    return None

def smithing_interface_open() -> bool:
    """
    Check if the smithing interface is open and visible.
    
    Returns:
        True if smithing interface is open and visible, False otherwise
    """
    return widget_exists(20447232)

def bank_interface_open() -> bool:
    """
    Check if the bank interface is open and visible.
    
    Returns:
        True if bank interface is open and visible, False otherwise
    """
    return widget_exists(786445)  # Bank items container

def ge_interface_open() -> bool:
    """
    Check if the Grand Exchange interface is open and visible.
    
    Returns:
        True if GE interface is open and visible, False otherwise
    """
    return widget_exists(30474241)  # Main GE widget

def inventory_interface_open() -> bool:
    """
    Check if the inventory interface is open and visible.
    
    Returns:
        True if inventory interface is open and visible, False otherwise
    """
    return widget_exists(9764864)  # Inventory container

def equipment_interface_open() -> bool:
    """
    Check if the equipment interface is open and visible.
    
    Returns:
        True if equipment interface is open and visible, False otherwise
    """
    return widget_exists(9764865)  # Equipment container

def prayer_interface_open() -> bool:
    """
    Check if the prayer interface is open and visible.
    
    Returns:
        True if prayer interface is open and visible, False otherwise
    """
    return widget_exists(9764866)  # Prayer container

def spellbook_interface_open() -> bool:
    """
    Check if the spellbook interface is open and visible.
    
    Returns:
        True if spellbook interface is open and visible, False otherwise
    """
    return widget_exists(9764867)  # Spellbook container

def chat_interface_open() -> bool:
    """
    Check if the chat interface is open and visible.
    
    Returns:
        True if chat interface is open and visible, False otherwise
    """
    return widget_exists(162)  # Chat container

def minimap_visible() -> bool:
    """
    Check if the minimap is visible.
    
    Returns:
        True if minimap is visible, False otherwise
    """
    return widget_exists(160)  # Minimap container

def find_chat_continue_widget() -> Optional[Dict[str, Any]]:
    """
    Find the "Click here to continue" widget under Chatbox.CHATMODAL (ID 10617398).
    
    Returns:
        The continue widget if found, or None if not found
    """
    # Chatbox.CHATMODAL widget ID
    chatmodal_widget_id = 10617398
    
    # Search for "Click here to continue" text in the children
    continue_widget = get_widget_child_with_text(
        chatmodal_widget_id, 
        "Click here to continue", 
        case_sensitive=False
    )
    
    return continue_widget

def click_chat_continue() -> Optional[dict]:
    """
    Click the "Click here to continue" widget if it exists.
    
    Returns:
        UI dispatch result or None if continue widget not found
    """
    continue_widget = find_chat_continue_widget()
    if not continue_widget:
        return None
    
    # Get the widget ID and bounds
    widget_id = continue_widget.get("id")
    bounds = continue_widget.get("bounds")
    
    if not widget_id or not bounds:
        return None
    
    # Calculate center coordinates
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    # Click on the continue widget
    step = {
        "action": "widget-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "widget", "name": f"chat_continue_{widget_id}"},
    }
    return dispatch(step)
