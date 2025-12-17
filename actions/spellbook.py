"""
Spellbook action methods for interacting with the magic spellbook interface.
"""

from typing import List, Dict, Optional, Any

from .npc import click_npc_simple
from helpers.runtime_utils import ipc, dispatch
from helpers.utils import sleep_exponential, rect_beta_xy

def get_spells() -> List[Dict[str, Any]]:
    """
    Get all available spells from the spellbook.
        
    Returns:
        List of spell dictionaries with name, bounds, canvas, etc.
    """
    
    resp = ipc.get_spellbook()
    if resp and resp.get("ok"):
        return resp.get("spells", [])
    return []


def select_spell(spell_name: str) -> Optional[Dict]:
    """
    Select a spell by name from the spellbook.
    
    Args:
        spell_name: Name of the spell to select (partial match supported)
        
    Returns:
        Result dictionary if successful, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None

    # Get all available spells
    spells = get_spells()
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
    
    # Get click coordinates from bounds first, then fallback to canvas
    bounds = target_spell.get("bounds", {})
    if bounds and bounds.get("width", 0) > 0 and bounds.get("height", 0) > 0:
        x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                             bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    else:
        # Fallback to canvas coordinates
        canvas = target_spell.get("canvas", {})
        if not canvas.get("x") or not canvas.get("y"):
            return None
        x, y = canvas.get("x"), canvas.get("y")
    
    # Create click action
    step = {
        "action": "click-spell",
        "click": {
            "type": "point",
            "x": x,
            "y": y
        },
        "target": {
            "domain": "spell",
            "name": target_spell.get("name", spell_name)
        }
    }
    
    return dispatch(step)


def is_spell_selected(spell_name: str) -> bool:
    """
    Check if a specific spell is currently selected.
    
    Args:
        spell_name: Name of the spell to check
        
    Returns:
        True if the spell is selected, False otherwise
    """
    selected = selected_spell()
    if not selected:
        return False
    
    selected_name = (selected.get("name") or "").lower()
    spell_name_lower = str(spell_name).strip().lower()
    
    return spell_name_lower in selected_name


def selected_spell() -> Optional[Dict[str, Any]]:
    """
    Get the currently selected spell.
    
    Args:
        
    Returns:
        Dictionary with selected spell info, or None if no spell selected
    """
    
    # Get all spells and check which one appears to be selected
    # This is a heuristic - we'll look for visual indicators of selection
    spells = get_spells()
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


def cast_spell(spell_name: str, target_name: str = None, target_type: str = "npc") -> Optional[Dict]:
    """
    Select a spell and cast it on a target (object, item, or NPC).
    
    Args:
        spell_name: Name of the spell to cast
        target_name: Name of the target to cast on (optional)
        target_type: Type of target ("object", "item", "npc")
        
    Returns:
        Result dictionary if successful, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None

    # First, select the spell
    spell_result = select_spell(spell_name)
    if not spell_result:
        return None
    
    # If no target specified, just return the spell selection result
    if not target_name:
        return spell_result
    
    # Wait a moment for spell selection to register
    sleep_exponential(0.05, 0.15, 1.5)
    
    # Now find and click the target
    if target_type == "object":
        from .objects import click
        return click(target_name)
    elif target_type == "npc":
        return click_npc_simple(target_name, "Cast")
    elif target_type == "item":
        from .inventory import interact
        return interact(target_name, "Cast")
    else:
        return None


def get_spell_by_name(spell_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific spell by name.
    
    Args:
        spell_name: Name of the spell to find
        
    Returns:
        Spell dictionary if found, None otherwise
    """
    if not spell_name or not str(spell_name).strip():
        return None
    
    spells = get_spells()
    if not spells:
        return None
    
    spell_name_lower = str(spell_name).strip().lower()
    
    for spell in spells:
        spell_name_actual = (spell.get("name") or "").lower()
        if spell_name_lower in spell_name_actual:
            return spell
    
    return None

def get_available_spell_names() -> List[str]:
    """
    Get a list of all available spell names.
    
    Args:
        
    Returns:
        List of spell names
    """
    spells = get_spells()
    return [spell.get("name", "") for spell in spells if spell.get("name")]


def spell_exists(spell_name: str) -> bool:
    """
    Check if a spell exists in the spellbook.
    
    Args:
        spell_name: Name of the spell to check
        
    Returns:
        True if spell exists, False otherwise
    """
    return get_spell_by_name(spell_name) is not None
