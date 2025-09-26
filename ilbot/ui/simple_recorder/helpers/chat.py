# chat.py (helpers)

from __future__ import annotations
from typing import Optional, List, Dict, Any

from ..helpers.context import get_payload

# Expected payload structure (only visible widgets exported):
# payload["chat_dialogue"] = {
#   "open": bool,
#   "name": {"text": "...", "exists": bool} | None,
#   "text": {"text": "...", "exists": bool} | None,
#   "continue": {"text": "...", "exists": bool} | None
# }
# payload["chat_menu"] = {
#   "open": bool,
#   "parentId": 14352385,
#   "options": [
#       {"index": 0, "text": "Option text 1"},
#       {"index": 1, "text": "Option text 2"},
#       ...
#   ]
# }

def _dlg_left(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatLeft") or {}) if isinstance(payload, dict) else {}

def _dlg_right(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatRight") or {}) if isinstance(payload, dict) else {}

def _dlg_objectbox(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("objectbox") or {}) if isinstance(payload, dict) else {}

def _menu(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatMenu") or {}) if isinstance(payload, dict) else {}

def dialogue_is_open(payload: Optional[dict] = None) -> bool:
    L = _dlg_left(payload)
    R = _dlg_right(payload)

    # Check for Messagebox.CONTINUE widget (ID 15007748) - this indicates actual dialogue
    from ..helpers.widgets import widget_exists
    if widget_exists(15007748):
        return True

    # explicit flag wins if present
    if bool(L.get("open")) or bool(R.get("open")):
        return True

    def any_visible(d: dict) -> bool:
        name = d.get("name") or {}
        text = d.get("text") or {}
        cont = d.get("continue") or {}
        return bool(name.get("exists") or text.get("exists") or cont.get("exists"))

    return any_visible(L) or any_visible(R)

def can_continue(payload: Optional[dict] = None) -> bool:
    # Check for Messagebox.CONTINUE widget (ID 15007748) first - must be visible and text contains "continue"
    from ..helpers.widgets import widget_exists, get_widget_info
    if widget_exists(15007748):
        widget_info = get_widget_info(15007748)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    L = _dlg_left(payload)
    R = _dlg_right(payload)
    OB = _dlg_objectbox(payload)

    l = (L.get("continue") or {}).get("exists")
    r = (R.get("continue") or {}).get("exists")

    # Objectbox.UNIVERSE (your new "continue" line)
    uni = (OB.get("universe") or {})
    # Prefer explicit visibility; fall back to non-empty text as a heuristic.
    o_exists = uni.get("exists")
    o_text   = (uni.get("textStripped") or uni.get("text") or "").strip()
    o = bool(o_exists) or bool(o_text)

    return bool(l or r or o)

def can_choose_option(payload: Optional[dict] = None) -> bool:
    m = _menu(payload)
    if bool(m.get("open")):
        return True
    opts = m.get("options") or []
    return opts['exists']

def get_options(payload: Optional[dict] = None) -> List[str]:
    m = _menu(payload)
    opts = ((m.get("options") or {}).get("texts") or [])
    out: List[str] = []
    for o in opts:
        t = o.strip()
        if t:
            out.append(t)
    return out

def get_option(index: int, payload: Optional[dict] = None) -> Optional[str]:
    """
    index may be 0-based or 1-based. If 1..N, we treat as human index.
    """
    opts = get_options(payload)
    if index >= 1:
        idx = index - 1
    else:
        idx = index
    if 0 <= idx < len(opts):
        return opts[idx]
    return None

def get_dialogue_text(payload: Optional[dict] = None) -> Dict[str, str]:
    """
    Returns { <speaker_name>: <dialogue_text>, ... }.
    If both left and right are visible, both are included.
    """
    out: Dict[str, str] = {}
    for d in (_dlg_left(payload), _dlg_right(payload)):
        if not d:
            continue
        name = ((d.get("name") or {}).get("text") or "").strip()
        text = ((d.get("text") or {}).get("text") or "").strip()
        if name or text:
            out[name] = text
    return out

def has_informational_text(payload: Optional[dict] = None) -> bool:
    """
    Check if there's informational text overlay (like tutorial messages) that can be read.
    This is different from dialogue_is_open() which checks for interactive dialogue.
    
    Returns:
        True if informational text is present, False otherwise
    """
    if payload is None:
        payload = get_payload() or {}
    
    # Check for informational text widgets
    informational_widgets = [
        17235969,  # Mesoverlay.TEXT (tutorial dialogue)
        15007747,  # Messagebox.TEXT (general dialogue)
    ]
    
    from ..helpers.widgets import widget_exists
    for widget_id in informational_widgets:
        if widget_exists(widget_id):
            return True
    
    return False

def get_dialogue_text_raw(payload: Optional[dict] = None) -> Optional[str]:
    """
    Extract raw dialogue text from any available chat widget.
    Checks multiple widget types in order of preference.
    
    Returns:
        Raw dialogue text or None if no dialogue found
    """
    if payload is None:
        payload = get_payload() or {}
    
    # List of dialogue widgets to check (in order of preference)
    dialogue_widgets = [
        17235969,  # Mesoverlay.TEXT (tutorial dialogue)
        15007747,  # Messagebox.TEXT (general dialogue)
        15007748,  # Messagebox.CONTINUE (continue dialogue)
    ]
    
    # Check each widget for text
    for widget_id in dialogue_widgets:
        try:
            from ..helpers.widgets import get_widget_info
            widget_info = get_widget_info(widget_id, payload)
            
            if widget_info and widget_info.get("data"):
                widget_data = widget_info.get("data", {})
                text = widget_data.get("text", "")
                
                if text and text.strip():
                    return text.strip()
        except Exception as e:
            print(f"[DIALOGUE] Error checking widget {widget_id}: {e}")
            continue
    
    # Fallback: check payload-based dialogue
    try:
        # Check left dialogue
        left_dlg = _dlg_left(payload)
        if left_dlg.get("text", {}).get("exists"):
            text = left_dlg["text"].get("text", "")
            if text and text.strip():
                return text.strip()
        
        # Check right dialogue
        right_dlg = _dlg_right(payload)
        if right_dlg.get("text", {}).get("exists"):
            text = right_dlg["text"].get("text", "")
            if text and text.strip():
                return text.strip()
        
        # Check objectbox dialogue
        obj_dlg = _dlg_objectbox(payload)
        if obj_dlg.get("universe", {}).get("exists"):
            text = obj_dlg["universe"].get("text", "")
            if text and text.strip():
                return text.strip()
    except Exception as e:
        print(f"[DIALOGUE] Error checking payload dialogue: {e}")
    
    return None

def get_clean_dialogue_text(payload: Optional[dict] = None) -> Optional[str]:
    """
    Get dialogue text with HTML tags and color codes stripped.
    
    Returns:
        Clean dialogue text or None if no dialogue found
    """
    raw_text = get_dialogue_text_raw(payload)
    if not raw_text:
        return None
    
    import re
    
    # Remove HTML color tags like <col=0000ff>
    clean_text = re.sub(r'<col=[0-9a-fA-F]+>', '', raw_text)
    
    # Remove HTML line breaks
    clean_text = clean_text.replace('<br>', ' ')
    clean_text = clean_text.replace('<br/>', ' ')
    
    # Remove other HTML tags
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    
    # Clean up whitespace
    clean_text = ' '.join(clean_text.split())
    
    return clean_text.strip()

def dialogue_contains_phrase(phrase: str, payload: Optional[dict] = None, case_sensitive: bool = False) -> bool:
    """
    Check if dialogue contains a specific phrase.
    
    Args:
        phrase: Phrase to search for
        payload: Optional payload, will get fresh if None
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        True if phrase found in dialogue, False otherwise
    """
    dialogue_text = get_clean_dialogue_text(payload)
    if not dialogue_text:
        return False
    
    if not case_sensitive:
        return phrase.lower() in dialogue_text.lower()
    else:
        return phrase in dialogue_text

def dialogue_contains_any_phrase(phrases: List[str], payload: Optional[dict] = None, case_sensitive: bool = False) -> bool:
    """
    Check if dialogue contains any of the given phrases.
    
    Args:
        phrases: List of phrases to search for
        payload: Optional payload, will get fresh if None
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        True if any phrase found in dialogue, False otherwise
    """
    dialogue_text = get_clean_dialogue_text(payload)
    if not dialogue_text:
        return False
    
    for phrase in phrases:
        if not case_sensitive:
            if phrase.lower() in dialogue_text.lower():
                return True
        else:
            if phrase in dialogue_text:
                return True
    
    return False

def get_dialogue_info(payload: Optional[dict] = None) -> Dict[str, Any]:
    """
    Get comprehensive dialogue information.
    
    Returns:
        Dictionary with dialogue status and text information
    """
    info = {
        "has_dialogue": False,
        "raw_text": None,
        "clean_text": None,
        "widget_source": None,
        "can_continue": False
    }
    
    # Check if there's any text to read (either dialogue or informational text)
    if not dialogue_is_open(payload) and not has_informational_text(payload):
        return info
    
    info["has_dialogue"] = True
    info["can_continue"] = can_continue(payload)
    
    # Get raw text
    raw_text = get_dialogue_text_raw(payload)
    if raw_text:
        info["raw_text"] = raw_text
        info["clean_text"] = get_clean_dialogue_text(payload)
        
        # Try to identify which widget provided the text
        dialogue_widgets = [
            (17235969, "Mesoverlay.TEXT"),
            (15007747, "Messagebox.TEXT"),
            (15007748, "Messagebox.CONTINUE"),
        ]
        
        for widget_id, widget_name in dialogue_widgets:
            try:
                from ..helpers.widgets import get_widget_info
                widget_info = get_widget_info(widget_id, payload)
                
                if widget_info and widget_info.get("data"):
                    widget_data = widget_info.get("data", {})
                    text = widget_data.get("text", "")
                    
                    if text and text.strip() and text.strip() == raw_text.strip():
                        info["widget_source"] = f"{widget_name} (ID: {widget_id})"
                        break
            except Exception:
                continue
    
    return info
