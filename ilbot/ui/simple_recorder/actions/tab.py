# tab.py (actions)

from __future__ import annotations
from typing import Optional
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.tab import is_inventory_tab_open, is_tab_open, find_tab_by_name

def open_tab(tab_name: str) -> Optional[dict]:
    """
    Open a specific tab if it's not already open.
    
    Args:
        tab_name: Name of the tab to open (e.g., "INVENTORY", "COMBAT", "SKILLS", etc.)
    
    Returns:
        Result dict or None if failed
    """
    # Check if tab is already open
    if is_tab_open(tab_name):
        return None  # Already open
    
    # Find the tab by name
    tab = find_tab_by_name(tab_name)
    if not tab:
        return None
    
    # Get coordinates from the tab info
    canvas = tab.get("canvas", {})
    x = canvas.get("x")
    y = canvas.get("y")
    
    if x is None or y is None:
        return None
    
    # Use direct IPC click instead of UI dispatch
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        ipc = RuneLiteIPC()
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
        import time
        time.sleep(0.2)
        # Check again
        return is_tab_open(tab_name)
    
    return False

def ensure_inventory_tab_open() -> bool:
    """
    Ensure the inventory tab is open, opening it if necessary.
    
    Returns:
        True if inventory tab is open (or was opened), False if failed
    """
    return ensure_tab_open("INVENTORY")

