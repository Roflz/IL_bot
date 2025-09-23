# tab.py (helpers)

from __future__ import annotations
from typing import Optional, Dict, Any
from .ipc import ipc_send
from .context import get_payload

def get_current_tab(payload: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """
    Get the currently open tab information.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        Dictionary with tab information or None if failed
        {
            "ok": True,
            "tab": 3,  # tab index (0-11)
            "tabName": "INVENTORY",  # tab name
            "tabs": [...]  # list of all available tabs
        }
    """
    if payload is None:
        payload = get_payload() or {}
    
    resp = ipc_send({"cmd": "tab"}, payload)
    if resp and resp.get("ok"):
        return resp
    return None

def is_inventory_tab_open(payload: Optional[dict] = None) -> bool:
    """
    Check if the inventory tab is currently open.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        True if inventory tab is open, False otherwise
    """
    tab_info = get_current_tab(payload)
    if tab_info:
        return tab_info.get("tabName") == "INVENTORY"
    return False

def get_tab_name(payload: Optional[dict] = None) -> Optional[str]:
    """
    Get the name of the currently open tab.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        Tab name (e.g., "INVENTORY", "COMBAT", etc.) or None if failed
    """
    tab_info = get_current_tab(payload)
    if tab_info:
        return tab_info.get("tabName")
    return None

def get_tab_index(payload: Optional[dict] = None) -> Optional[int]:
    """
    Get the index of the currently open tab.
    
    Args:
        payload: Optional payload, will get fresh if None
    
    Returns:
        Tab index (0-11) or None if failed
    """
    tab_info = get_current_tab(payload)
    if tab_info:
        return tab_info.get("tab")
    return None
