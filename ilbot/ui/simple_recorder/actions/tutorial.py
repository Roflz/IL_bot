# tutorial.py (actions)

from __future__ import annotations
from typing import Optional
import time

from ..helpers.runtime_utils import dispatch, ipc
from ..helpers.widgets import get_widget_text, rect_center_from_widget
from ..helpers.utils import sleep_exponential


def type_tutorial_name(name: str) -> bool:
    """Type a character name in the tutorial name input field."""
    # Get the name input widget from tutorial data via IPC
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        print("[TUTORIAL] Tutorial interface not open")
        return False
    
    name_input = tutorial_data.get("nameInput")
    if not name_input or not name_input.get("visible", False):
        print("[TUTORIAL] Name input widget not found or not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(name_input)
    if x is None or y is None:
        print("[TUTORIAL] Could not get name input coordinates")
        return False
    
    # Click on the name input field
    step = {
        "action": "click-name-input",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "name_input"}
    }
    
    dispatch(step)
    
    # Wait a moment for the field to be focused
    sleep_exponential(0.3, 0.8, 1.2)
    
    # Type the character name
    step = {
        "action": "type-character-name",
        "click": {"type": "type", "text": name, "per_char_ms": 50},
        "target": {"domain": "tutorial", "name": "name_input"}
    }
    
    dispatch(step)
    
    # Wait a moment then press enter
    sleep_exponential(0.8, 1.5, 1.0)
    
    # Press enter to confirm
    step = {
        "action": "press-enter",
        "click": {"type": "key", "key": "ENTER"},
        "target": {"domain": "tutorial", "name": "confirm_name"}
    }
    
    dispatch(step)
    
    print(f"[TUTORIAL] Successfully entered character name: {name}")
    return True


def click_tutorial_set_name() -> bool:
    """Click the tutorial SET_NAME button to confirm the character name."""
    # Get the set name widget via IPC
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        print("[TUTORIAL] Tutorial interface not open")
        return False
    
    set_name_widget = tutorial_data.get("setName")
    if not set_name_widget or not set_name_widget.get("visible", False):
        print("[TUTORIAL] SET_NAME widget not found or not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(set_name_widget)
    if x is None or y is None:
        print("[TUTORIAL] Could not get SET_NAME coordinates")
        return False
    
    # Click on the SET_NAME button
    step = {
        "action": "click-set-name",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "set_name"}
    }
    
    result = dispatch(step)
    if result is None:
        print("[TUTORIAL] Failed to click SET_NAME button")
        return False
    
    print("[TUTORIAL] Successfully clicked SET_NAME button")
    return True


def click_tutorial_lookup_name() -> bool:
    """Click the tutorial LOOK_UP_NAME button to check name availability."""
    # Get the lookup name widget via IPC
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        print("[TUTORIAL] Tutorial interface not open")
        return False
    
    lookup_widget = tutorial_data.get("lookupName")
    if not lookup_widget or not lookup_widget.get("visible", False):
        print("[TUTORIAL] LOOK_UP_NAME widget not found or not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(lookup_widget)
    if x is None or y is None:
        print("[TUTORIAL] Could not get LOOK_UP_NAME coordinates")
        return False
    
    # Click on the LOOK_UP_NAME button
    step = {
        "action": "click-lookup-name",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "lookup_name"}
    }
    
    result = dispatch(step)
    if result is None:
        print("[TUTORIAL] Failed to click LOOK_UP_NAME button")
        return False
    
    print("[TUTORIAL] Successfully clicked LOOK_UP_NAME button")
    return True


def get_tutorial_name_text() -> Optional[str]:
    """Get the current text from the tutorial name input field."""
    # Get the name text widget via IPC
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        return None
    
    name_text_widget = tutorial_data.get("nameText")
    if not name_text_widget:
        return None
    
    return name_text_widget.get("text", "")


def get_tutorial_status_text() -> Optional[str]:
    """Get the current status text from the tutorial status widget."""
    # Get the status widget via IPC
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        return None
    
    status_widget = tutorial_data.get("status")
    if not status_widget:
        return None
    
    return status_widget.get("text", "")


def is_tutorial_open() -> bool:
    """Check if the tutorial interface is currently open."""
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    return tutorial_data.get("open", False)


def get_tutorial_name_input_bounds() -> Optional[dict]:
    """Get the bounds of the tutorial name input widget."""
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        return None
    
    name_input = tutorial_data.get("nameInput")
    if not name_input:
        return None
    
    return name_input.get("bounds")


def get_tutorial_set_name_bounds() -> Optional[dict]:
    """Get the bounds of the tutorial SET_NAME button."""
    tutorial_data = ipc.get_tutorial().get('tutorial') or {}
    if not tutorial_data.get("open", False):
        return None
    
    set_name_widget = tutorial_data.get("setName")
    if not set_name_widget:
        return None
    
    return set_name_widget.get("bounds")
