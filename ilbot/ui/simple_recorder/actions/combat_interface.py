# combat_interface.py
from __future__ import annotations
from typing import Optional, List, Dict, Any

from . import tab
from .timing import wait_until
from ..helpers.runtime_utils import ipc, ui, dispatch
from ..helpers.utils import rect_beta_xy


def get_combat_styles() -> List[Dict[str, Any]]:
    """
    Get all available combat styles from the combat interface.
        
    Returns:
        List of combat style widgets with their data
    """
    
    # Combat style widget IDs
    combat_style_widgets = [
        {"id": 38862853, "index": 0, "name": "CombatStyle_0"},
        {"id": 38862857, "index": 1, "name": "CombatStyle_1"}, 
        {"id": 38862861, "index": 2, "name": "CombatStyle_2"},
        {"id": 38862865, "index": 3, "name": "CombatStyle_3"}
    ]
    
    styles = []
    
    for style in combat_style_widgets:
        # Get widget data
        resp = ipc.get_widget_info(style["id"])
        
        if resp and resp.get("ok") and resp.get("widget"):
            widget_data = resp["widget"]
            style_data = {
                "index": style["index"],
                "id": style["id"],
                "name": style["name"],
                "visible": widget_data.get("visible", False),
                "bounds": widget_data.get("bounds"),
                "text": widget_data.get("text", ""),
                "spriteId": widget_data.get("spriteId", -1)
            }
            styles.append(style_data)
    
    print(f"[COMBAT] Retrieved {len(styles)} combat styles")
    return styles


def select_combat_style(style_index: int) -> Optional[dict]:
    """
    Select a combat style by clicking on it.
    
    Args:
        style_index: Index of combat style to select (0-3)
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    if style_index < 0 or style_index > 3:
        print(f"[COMBAT] Invalid combat style index: {style_index}")
        return None

    if not tab.is_tab_open("COMBAT"):
        tab.open_tab("COMBAT")
        wait_until(lambda: tab.is_tab_open("COMBAT"), max_wait_ms=3000)
    
    # Combat style widget IDs
    widget_ids = [38862853, 38862857, 38862861, 38862865]
    widget_id = widget_ids[style_index]
    
    # Get widget data to get bounds
    resp = ipc.get_widget_info(widget_id)
    
    if not resp or not resp.get("ok") or not resp.get("widget"):
        print(f"[COMBAT] Failed to get combat style widget {style_index}")
        return None
    
    widget_data = resp["widget"]
    if not widget_data.get("visible", False):
        print(f"[COMBAT] Combat style widget {style_index} not visible")
        return None
    
    bounds = widget_data.get("bounds")
    if not bounds:
        print(f"[COMBAT] No bounds for combat style widget {style_index}")
        return None
    
    # Calculate center coordinates
    x, y = rect_beta_xy((bounds["x"], bounds["x"] + bounds["width"],
                         bounds["y"], bounds["y"] + bounds["height"]), alpha=2.0, beta=2.0)
    
    print(f"[COMBAT] Selecting combat style {style_index} at ({x}, {y})")
    
    # Create click step
    step = {
        "action": "combat-style-select",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "combat", "name": f"style_{style_index}"}
    }
    
    # Execute the click
    result = dispatch(step)
    
    if result:
        print(f"[COMBAT] Successfully selected combat style {style_index}")
    else:
        print(f"[COMBAT] Failed to select combat style {style_index}")
    
    return result


def current_combat_style() -> Optional[int]:
    """
    Get the currently selected combat style using RuneLite's COM_MODE VarPlayer.
        
    Returns:
        Index of currently selected combat style (0-3), or None if unknown
    """
    
    # Use RuneLite's COM_MODE VarPlayer to get current combat style
    resp = ipc.get_varp(43)  # VarPlayerID.COM_MODE
    
    if not resp or not resp.get("ok"):
        print(f"[COMBAT] Failed to get current combat style from RuneLite API")
        return None
    
    combat_style_value = resp.get("value", -1)
    
    # Map RuneLite's COM_MODE values to our widget indices
    # Based on testing: 0 = Attack, 1 = Strength, 2 = Defence, 3 = Ranged/Magic
    # We need to map these to our widget indices: 0=Attack, 1=Strength, 3=Defence
    
    if combat_style_value == 0:  # Attack
        style_index = 0
    elif combat_style_value == 1:  # Strength  
        style_index = 1
    elif combat_style_value == 3:  # Defence
        style_index = 3
    else:
        print(f"[COMBAT] Unknown combat style value: {combat_style_value}")
        return None
    
    print(f"[COMBAT] Current combat style: {style_index} (COM_MODE value: {combat_style_value})")
    return style_index


def select_auto_retaliate() -> Optional[dict]:
    """
    Toggle auto retaliate by clicking on it.
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    # Auto retaliate widget ID
    widget_id = 38862882
    
    # Get widget data to get bounds
    resp = ipc.get_widget_info(widget_id)
    
    if not resp or not resp.get("ok") or not resp.get("widget"):
        print(f"[COMBAT] Failed to get auto retaliate widget")
        return None
    
    widget_data = resp["widget"]
    if not widget_data.get("visible", False):
        print(f"[COMBAT] Auto retaliate widget not visible")
        return None
    
    bounds = widget_data.get("bounds")
    if not bounds:
        print(f"[COMBAT] No bounds for auto retaliate widget")
        return None
    
    # Calculate center coordinates
    x, y = rect_beta_xy((bounds["x"], bounds["x"] + bounds["width"],
                         bounds["y"], bounds["y"] + bounds["height"]), alpha=2.0, beta=2.0)
    
    print(f"[COMBAT] Toggling auto retaliate at ({x}, {y})")
    
    # Create click step
    step = {
        "action": "auto-retaliate-toggle",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "combat", "name": "auto_retaliate"}
    }
    
    # Execute the click
    result = dispatch(step)
    
    if result:
        print(f"[COMBAT] Successfully toggled auto retaliate")
    else:
        print(f"[COMBAT] Failed to toggle auto retaliate")
    
    return result


def auto_retaliate_on() -> bool:
    """
    Check if auto retaliate is currently on.
        
    Returns:
        True if auto retaliate is on, False otherwise
    """
    
    # Auto retaliate widget ID
    widget_id = 38862882
    
    # Get widget data
    resp = ipc.get_widget_info(widget_id)
    
    if not resp or not resp.get("ok") or not resp.get("widget"):
        print(f"[COMBAT] Failed to get auto retaliate widget")
        return False
    
    widget_data = resp["widget"]
    if not widget_data.get("visible", False):
        print(f"[COMBAT] Auto retaliate widget not visible")
        return False
    
    # Check sprite ID to determine if auto retaliate is on
    sprite_id = widget_data.get("spriteId", -1)
    
    # According to the user: spriteId 1749 = on, 1748 = off
    is_on = (sprite_id == 1749)
    
    print(f"[COMBAT] Auto retaliate is {'ON' if is_on else 'OFF'} (spriteId: {sprite_id})")
    return is_on
