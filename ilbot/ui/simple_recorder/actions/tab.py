# tab.py (actions)

from __future__ import annotations
from typing import Optional
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.tab import is_inventory_tab_open

def open_inventory_tab(payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Open the inventory tab if it's not already open.
    
    Args:
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        UI dispatch result or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Check if inventory tab is already open
    if is_inventory_tab_open(payload):
        return None  # Already open
    
    # Get tab coordinates from IPC
    from ..helpers.tab import get_current_tab
    tab_info = get_current_tab(payload)
    
    if not tab_info or not tab_info.get("ok"):
        return None
    
    # Find the inventory tab in the tabs list
    tabs = tab_info.get("tabs", [])
    inventory_tab = None
    for tab in tabs:
        if tab.get("name") == "INVENTORY":
            inventory_tab = tab
            break
    
    if not inventory_tab:
        return None
    
    # Get coordinates from the tab info
    canvas = inventory_tab.get("canvas", {})
    x = canvas.get("x")
    y = canvas.get("y")
    
    if x is None or y is None:
        return None
    
    # Click on the inventory tab using dynamic coordinates
    step = emit({
        "action": "tab-click",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "tab", "name": "INVENTORY"},
    })
    return ui.dispatch(step)

def ensure_inventory_tab_open(payload: Optional[dict] = None, ui=None) -> bool:
    """
    Ensure the inventory tab is open, opening it if necessary.
    
    Args:
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance, will get if None
    
    Returns:
        True if inventory tab is open (or was opened), False if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Check if already open
    if is_inventory_tab_open(payload):
        return True
    
    # Try to open it
    result = open_inventory_tab(payload, ui)
    if result is not None:
        # Wait a moment for the tab to open
        import time
        time.sleep(0.2)
        # Check again
        return is_inventory_tab_open(payload)
    
    return False
