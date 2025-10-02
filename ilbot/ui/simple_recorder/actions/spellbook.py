"""
Spellbook action methods for interacting with the magic spellbook interface.
"""

from typing import List, Dict, Optional, Any

from .npc import click_npc_simple
from ..helpers.ipc import ipc_send
from ..helpers.utils import get_payload, get_ui
from .runtime import emit
from ..services.camera_integration import dispatch_with_camera


def get_spells(payload: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Get all available spells from the spellbook.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        List of spell dictionaries with name, bounds, canvas, etc.
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_spellbook"}, payload)
    if resp and resp.get("ok"):
        return resp.get("spells", [])
    return []


def select_spell(spell_name: str, payload: Optional[Dict] = None, ui=None) -> Optional[Dict]:
    """
    Select a spell by name from the spellbook.
    
    Args:
        spell_name: Name of the spell to select (partial match supported)
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance
        
    Returns:
        Result dictionary if successful, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None
    
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get all available spells
    spells = get_spells(payload)
    if not spells:
        return None
    
    # Find matching spell (case-insensitive partial match)
    target_spell = None
    spell_name_lower = str(spell_name).strip().lower()
    
    for spell in spells:
        spell_name_actual = (spell.get("name") or "").lower()
        if spell_name_lower in spell_name_actual:
            target_spell = spell
            break
    
    if not target_spell:
        return None
    
    # Check if spell is visible and clickable
    if not target_spell.get("visible", False) or not target_spell.get("hasListener", False):
        return None
    
    # Get click coordinates
    canvas = target_spell.get("canvas", {})
    if not canvas.get("x") or not canvas.get("y"):
        return None
    
    # Create click action
    step = emit({
        "action": "click-spell",
        "click": {
            "type": "point",
            "x": int(canvas["x"]),
            "y": int(canvas["y"])
        },
        "target": {
            "domain": "spell",
            "name": target_spell.get("name", spell_name)
        }
    })
    
    return ui.dispatch(step)


def is_spell_selected(spell_name: str, payload: Optional[Dict] = None) -> bool:
    """
    Check if a specific spell is currently selected.
    
    Args:
        spell_name: Name of the spell to check
        payload: Optional payload, will get fresh if None
        
    Returns:
        True if the spell is selected, False otherwise
    """
    selected = selected_spell(payload)
    if not selected:
        return False
    
    selected_name = (selected.get("name") or "").lower()
    spell_name_lower = str(spell_name).strip().lower()
    
    return spell_name_lower in selected_name


def selected_spell(payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Get the currently selected spell.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        Dictionary with selected spell info, or None if no spell selected
    """
    if payload is None:
        payload = get_payload()
    
    # Get all spells and check which one appears to be selected
    # This is a heuristic - we'll look for visual indicators of selection
    spells = get_spells(payload)
    if not spells:
        return None
    
    # For now, we'll return the first visible spell as a placeholder
    # In a real implementation, you'd need to check for visual selection indicators
    # like different sprite states, highlighted borders, etc.
    for spell in spells:
        if spell.get("visible", False) and spell.get("hasListener", False):
            # This is a simplified check - in practice you'd need to detect
            # actual selection state through visual indicators
            return spell
    
    return None


def cast_spell(spell_name: str, target_name: str = None, target_type: str = "npc",
               payload: Optional[Dict] = None, ui=None) -> Optional[Dict]:
    """
    Select a spell and cast it on a target (object, item, or NPC).
    
    Args:
        spell_name: Name of the spell to cast
        target_name: Name of the target to cast on (optional)
        target_type: Type of target ("object", "item", "npc")
        payload: Optional payload, will get fresh if None
        ui: Optional UI instance
        
    Returns:
        Result dictionary if successful, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None
    
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # First, select the spell
    spell_result = select_spell(spell_name, payload, ui)
    if not spell_result:
        return None
    
    # If no target specified, just return the spell selection result
    if not target_name:
        return spell_result
    
    # Wait a moment for spell selection to register
    import time
    time.sleep(0.1)
    
    # Now find and click the target
    if target_type == "object":
        from .objects import click
        return click(target_name, payload=payload, ui=ui)
    elif target_type == "npc":
        return click_npc_simple(target_name, payload=payload, ui=ui)
    elif target_type == "item":
        from .inventory import interact
        return interact(target_name, "Cast", payload=payload, ui=ui)
    else:
        return None


def get_spell_by_name(spell_name: str, payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Get a specific spell by name.
    
    Args:
        spell_name: Name of the spell to find
        payload: Optional payload, will get fresh if None
        
    Returns:
        Spell dictionary if found, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None
    
    spells = get_spells(payload)
    if not spells:
        return None
    
    spell_name_lower = str(spell_name).strip().lower()
    
    for spell in spells:
        spell_name_actual = (spell.get("name") or "").lower()
        if spell_name_lower in spell_name_actual:
            return spell
    
    return None


def is_spellbook_open(payload: Optional[Dict] = None) -> bool:
    """
    Check if the spellbook interface is open.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        True if spellbook is open, False otherwise
    """
    spells = get_spells(payload)
    return len(spells) > 0


def get_available_spell_names(payload: Optional[Dict] = None) -> List[str]:
    """
    Get a list of all available spell names.
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        List of spell names
    """
    spells = get_spells(payload)
    return [spell.get("name", "") for spell in spells if spell.get("name")]


def spell_exists(spell_name: str, payload: Optional[Dict] = None) -> bool:
    """
    Check if a spell exists in the spellbook.
    
    Args:
        spell_name: Name of the spell to check
        payload: Optional payload, will get fresh if None
        
    Returns:
        True if spell exists, False otherwise
    """
    return get_spell_by_name(spell_name, payload) is not None
