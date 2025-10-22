# chat.py (helpers)

from __future__ import annotations
from typing import Optional, List, Dict, Any

from ..actions.widgets import get_widget_children
from ..helpers.runtime_utils import ipc

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

def _dlg_left() -> dict:
    chat_data = ipc.get_chat()
    return chat_data.get("chatLeft") or {}

def _dlg_right() -> dict:
    chat_data = ipc.get_chat()
    return chat_data.get("chatRight") or {}

def _dlg_objectbox() -> dict:
    chat_data = ipc.get_chat()
    return chat_data.get("objectbox") or {}

def _menu() -> dict:
    chat_data = ipc.get_chat()
    return chat_data.get("chatMenu") or {}

def dialogue_is_open() -> bool:
    L = _dlg_left()
    R = _dlg_right()

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

def can_continue() -> bool:
    # Check for Messagebox.CONTINUE widget (ID 15007748) first - must be visible and text contains "continue"
    from ..helpers.widgets import widget_exists, get_widget_info
    if widget_exists(15007748):
        widget_info = get_widget_info(15007748)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    # Check for Objectbox.TEXT widget (ID 12648450) - poll booth dialogue
    if widget_exists(12648450):
        widget_info = get_widget_info(12648450)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    # Check for Chatbox.MES_TEXT2 widget (ID 10616875) - another "Click here to continue" widget
    if widget_exists(10616875):
        widget_info = get_widget_info(10616875)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    # Check for ObjectboxDouble.PAUSEBUTTON widget (ID 720900) - "Click here to continue"
    if widget_exists(720900):
        widget_info = get_widget_info(720900)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    # Check for LevelupDisplay.CONTINUE widget (ID 15269891) - "Click here to continue"
    if widget_exists(15269891):
        widget_info = get_widget_info(15269891)
        if widget_info:
            widget_data = widget_info.get("data", {})
            text = widget_data.get("text", "").lower()
            if "continue" in text:
                return True

    L = _dlg_left()
    R = _dlg_right()
    OB = _dlg_objectbox()

    l = (L.get("continue") or {}).get("exists")
    r = (R.get("continue") or {}).get("exists")

    # Objectbox.UNIVERSE (your new "continue" line)
    uni = (OB.get("universe") or {})
    # Prefer explicit visibility; fall back to non-empty text as a heuristic.
    o_exists = uni.get("exists")
    o_text   = (uni.get("textStripped") or uni.get("text") or "").strip()
    o = bool(o_exists) or bool(o_text)

    return bool(l or r or o)

def can_choose_option() -> bool:
    m = _menu()
    if bool(m.get("open")):
        return True
    opts = m.get("options") or []
    if not opts:
        return False
    return opts.get('exists', [])

def get_options() -> List[str]:
    m = _menu()
    opts = ((m.get("options") or {}).get("texts") or [])
    out: List[str] = []
    for o in opts:
        t = o.strip()
        if t:
            out.append(t)
    return out

def get_option(index: int) -> Optional[str]:
    """
    index may be 0-based or 1-based. If 1..N, we treat as human index.
    """
    opts = get_options()
    if index >= 1:
        idx = index - 1
    else:
        idx = index
    if 0 <= idx < len(opts):
        return opts[idx]
    return None

def get_dialogue_text() -> Dict[str, str]:
    """
    Returns { <speaker_name>: <dialogue_text>, ... }.
    If both left and right are visible, both are included.
    """
    out: Dict[str, str] = {}
    for d in (_dlg_left(), _dlg_right()):
        if not d:
            continue
        name = ((d.get("name") or {}).get("text") or "").strip()
        text = ((d.get("text") or {}).get("text") or "").strip()
        if name or text:
            out[name] = text
    return out

def has_informational_text() -> bool:
    """
    Check if there's informational text overlay (like tutorial messages) that can be read.
    This is different from dialogue_is_open() which checks for interactive dialogue.
    
    Returns:
        True if informational text is present, False otherwise
    """
    
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

def get_dialogue_text_raw() -> list[str]:
    """
    Extract raw dialogue text from chat widget children.
    Gets all children from widget 10616866 and returns a list of all text content.
    
    Returns:
        List of dialogue text strings (empty list if no dialogue found)
    """
    try:
        # Get all children from the main chat widget
        widgets = get_widget_children(10616866)
        
        if not widgets or not widgets.get("ok"):
            return []
        
        children = widgets.get("children", [])
        dialogue_texts = []
        
        # Loop through all children and extract text
        for child in children:
            text = child.get("text", "")
            if text and text.strip():
                dialogue_texts.append(text.strip())
        
        return dialogue_texts
        
    except Exception as e:
        print(f"[DIALOGUE] Error getting dialogue text: {e}")
        return []


def get_clean_dialogue_text() -> str:
    """
    Get dialogue text with HTML tags and color codes stripped.
    
    Returns:
        Clean dialogue text or empty string if no dialogue found
    """
    raw_text = get_dialogue_text_raw()
    if not raw_text:
        return ""
    
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

def dialogue_contains_phrase(phrase: str, case_sensitive: bool = False) -> bool:
    """
    Check if dialogue contains a specific phrase.
    
    Args:
        phrase: Phrase to search for
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        True if phrase found in dialogue, False otherwise
    """
    dialogue_text = get_clean_dialogue_text()
    if not dialogue_text:
        return False
    
    if not case_sensitive:
        return phrase.lower() in dialogue_text.lower()
    else:
        return phrase in dialogue_text

def dialogue_contains_any_phrase(phrases: List[str], case_sensitive: bool = False) -> bool:
    """
    Check if dialogue contains any of the given phrases.
    
    Args:
        phrases: List of phrases to search for
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        True if any phrase found in dialogue, False otherwise
    """
    dialogue_text = get_clean_dialogue_text()
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

def get_dialogue_info() -> Dict[str, Any]:
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
    if not dialogue_is_open() and not has_informational_text():
        return info
    
    info["has_dialogue"] = True
    info["can_continue"] = can_continue()
    
    # Get raw text
    raw_text = get_dialogue_text_raw()
    if raw_text:
        info["raw_text"] = raw_text
        info["clean_text"] = get_clean_dialogue_text()
        
        # Try to identify which widget provided the text
        dialogue_widgets = [
            (17235969, "Mesoverlay.TEXT"),
            (15007747, "Messagebox.TEXT"),
            (15007748, "Messagebox.CONTINUE"),
            (12648450, "Objectbox.TEXT"),
            (10616875, "Chatbox.MES_TEXT2"),
        ]
        
        for widget_id, widget_name in dialogue_widgets:
            try:
                from ..helpers.widgets import get_widget_info
                widget_info = get_widget_info(widget_id)
                
                if widget_info and widget_info.get("data"):
                    widget_data = widget_info.get("data", {})
                    text = widget_data.get("text", "")
                    
                    if text and text.strip() and text.strip() == raw_text.strip():
                        info["widget_source"] = f"{widget_name} (ID: {widget_id})"
                        break
            except Exception:
                continue
    
    return info
