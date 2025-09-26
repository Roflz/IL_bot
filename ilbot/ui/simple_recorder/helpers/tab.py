# tab.py (helpers)

from __future__ import annotations
from typing import Optional, Dict, Any
from .ipc import ipc_send
from .context import get_payload

def _get_tab_data() -> Optional[Dict[str, Any]]:
    """Internal function to get tab data from IPC."""
    try:
        from ..services.ipc_client import RuneLiteIPC
        
        # Create IPC client directly
        ipc = RuneLiteIPC()
        
        # Send tab command
        resp = ipc._send({"cmd": "tab"})
        if resp and resp.get("ok"):
            return resp
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to get tab data: {e}")
        return None

def get_current_tab_name() -> Optional[str]:
    """Get the name of the currently open tab."""
    tab_data = _get_tab_data()
    return tab_data.get("tabName") if tab_data else None

def get_current_tab_index() -> Optional[int]:
    """Get the index of the currently open tab."""
    tab_data = _get_tab_data()
    return tab_data.get("tab") if tab_data else None

def is_tab_open(tab_name: str) -> bool:
    """Check if a specific tab is currently open."""
    current_name = get_current_tab_name()
    return current_name == tab_name.upper() if current_name else False

def get_all_tabs() -> list:
    """Get a list of all available tabs."""
    tab_data = _get_tab_data()
    return tab_data.get("tabs", []) if tab_data else []

def find_tab_by_name(tab_name: str) -> Optional[Dict[str, Any]]:
    """Find a specific tab by name."""
    tabs = get_all_tabs()
    tab_name_upper = tab_name.upper()
    
    for tab in tabs:
        if tab.get("name") == tab_name_upper:
            return tab
    return None

# Convenience functions
def is_inventory_tab_open() -> bool:
    """Check if the inventory tab is currently open."""
    return is_tab_open("INVENTORY")

def get_available_tab_names() -> list:
    """Get a list of all available tab names."""
    tabs = get_all_tabs()
    return [tab.get("name") for tab in tabs if tab.get("name")]

# Legacy aliases for backward compatibility
def get_current_tab() -> Optional[Dict[str, Any]]:
    """Legacy function - use get_current_tab_name() and get_current_tab_index() instead."""
    return _get_tab_data()

def get_tab_name() -> Optional[str]:
    """Legacy function - use get_current_tab_name() instead."""
    return get_current_tab_name()

def get_tab_index() -> Optional[int]:
    """Legacy function - use get_current_tab_index() instead."""
    return get_current_tab_index()
