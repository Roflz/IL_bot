# tab.py (actions)

from __future__ import annotations
import random
from typing import Optional
from helpers.tab import is_tab_open, find_tab_by_name, get_available_tab_names, get_current_tab_name
from helpers.runtime_utils import ipc
from helpers.utils import sleep_exponential, rect_beta_xy

def open_tab(tab_name: str) -> Optional[dict]:
    """
    Open a specific tab if it's not already open.
    
    Args:
        tab_name: Name of the tab to open (e.g., "INVENTORY", "COMBAT", "SKILLS", etc.)
                  Can also be "RANDOM" to open a random tab.
    
    Returns:
        Result dict or None if failed
    """
    # Handle RANDOM tab selection
    if tab_name.upper() == "RANDOM":
        available_tabs = get_available_tab_names()
        if not available_tabs:
            return None
        # Get current tab name to avoid selecting it
        current_tab = get_current_tab_name()
        # Filter out current tab if it exists
        selectable_tabs = [tab for tab in available_tabs if tab != current_tab]
        if not selectable_tabs:
            # If only current tab is available, use all tabs
            selectable_tabs = available_tabs
        tab_name = random.choice(selectable_tabs)
    
    # Check if tab is already open
    if is_tab_open(tab_name):
        return None  # Already open
    
    # Find the tab by name
    tab = find_tab_by_name(tab_name)
    if not tab:
        return None
    
    # Get coordinates from bounds first, then fallback to canvas
    bounds = tab.get("bounds", {})
    if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
        x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                             bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    else:
        # Fallback to canvas coordinates
        canvas = tab.get("canvas", {})
        x = canvas.get("x")
        y = canvas.get("y")
        
        if x is None or y is None:
            return None

    try:
        
        tab_data = ipc.get_tab() or {}
        click_resp = ipc._send({
            "cmd": "click",
            "x": int(x),
            "y": int(y)
        })
        
        if click_resp and click_resp.get("ok"):
            return {"ok": True, "action": "tab-click", "tab": tab_name}
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to click tab {tab_name}: {e}")
        return None

def open_inventory_tab() -> Optional[dict]:
    """
    Open the inventory tab if it's not already open.
    
    Returns:
        Result dict or None if failed
    """
    return open_tab("INVENTORY")

def ensure_tab_open(tab_name: str) -> bool:
    """
    Ensure a specific tab is open, opening it if necessary.
    
    Args:
        tab_name: Name of the tab to ensure is open (e.g., "INVENTORY", "COMBAT", "SKILLS", etc.)
    
    Returns:
        True if tab is open (or was opened), False if failed
    """
    # Check if already open
    if is_tab_open(tab_name):
        return True
    
    # Try to open it
    result = open_tab(tab_name)
    if result is not None:
        # Wait a moment for the tab to open
        sleep_exponential(0.1, 0.3, 1.5)
        # Check again
        return is_tab_open(tab_name)
    
    return False


