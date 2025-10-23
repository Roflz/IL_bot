# chat.py (actions)

from __future__ import annotations
from typing import Optional, Union

from .timing import wait_until
from .widgets import click_chat_continue, find_chat_continue_widget, get_widget_children
from ..helpers.runtime_utils import ui, dispatch
from ..helpers.chat import (
    can_continue as _can_continue,
    can_choose_option as _can_choose_option,
    get_options as _get_options,
    get_option as _get_option,
    dialogue_is_open as _dialogue_is_open,
    get_dialogue_text_raw as _get_dialogue_text_raw,
)
from ..helpers.utils import press_spacebar, clean_rs


def dialogue_is_open() -> bool:
    return _dialogue_is_open()

def can_continue() -> bool:
    """
    Check if we can continue dialogue by looking for "Click here to continue" widget.
    """
    from .widgets import find_chat_continue_widget
    
    # Check for the continue widget under Chatbox.CHATMODAL
    continue_widget = find_chat_continue_widget()
    if continue_widget:
        return True
    
    # Fallback to original method
    return _can_continue()

def can_choose_option() -> bool:
    return _can_choose_option()

def get_options():
    options = _get_options()
    if options and len(options) > 0:
        return options
    else:
        return None

def get_option(index: int):
    return _get_option(index)

def dialogue_contains(substr: str) -> bool:
    """
    True if the current dialogue (left or right) contains `substr` in its *text* block.
    Only checks the TEXT portion (not names or 'continue').
    """
    if not substr:
        return False
    
    # Get chat data directly from IPC
    from ..helpers.runtime_utils import ipc
    chat_data = ipc.get_chat() or {}

    def _side_text(side_key: str) -> str:
        side = (chat_data.get(side_key) or {}) if isinstance(chat_data, dict) else {}
        text_block = side.get("text") or {}
        # Prefer stripped if present; otherwise raw text; else empty string
        return (text_block.get("textStripped")
                or text_block.get("text")
                or "").strip()

    needle = substr.strip().lower()
    if not needle:
        return False

    left_txt  = _side_text("chatLeft").lower()
    right_txt = _side_text("chatRight").lower()

    return (needle in left_txt) or (needle in right_txt)

def option_exists(text: str) -> bool:
    """
    Return True if a chat menu option containing `text` (case-insensitive) exists.
    """
    # Get chat data directly from IPC
    from ..helpers.runtime_utils import ipc
    chat_data = ipc.get_chat() or {}

    opts = (((chat_data.get("chatMenu") or {}).get("options") or {}).get("texts") or [])
    want = (text or "").strip().lower()
    if not want:
        return False

    for s in opts:
        if isinstance(s, str) and want in s.lower():
            return True
    return False

def any_chat_active() -> bool:
    """
    Return True if any kind of chat/dialogue is currently active.
    This includes:
    - Regular dialogue (can continue)
    - Option menus (can choose option)
    - Any dialogue text present (left or right)
    - Any chat menu options available
    """
    # Get chat data directly from IPC
    from ..helpers.runtime_utils import ipc
    chat_data = ipc.get_chat() or {}
    
    # Check if dialogue is open
    if _dialogue_is_open():
        return True
    
    # Check if we can continue dialogue
    if _can_continue():
        return True
    
    # Check if we can choose options
    if _can_choose_option():
        return True
    
    # Check if there's any dialogue text present
    def _has_text(side_key: str) -> bool:
        side = (chat_data.get(side_key) or {}) if isinstance(chat_data, dict) else {}
        text_block = side.get("text") or {}
        text = (text_block.get("textStripped") or text_block.get("text") or "").strip()
        return bool(text)
    
    if _has_text("chatLeft") or _has_text("chatRight"):
        return True
    
    # Check if there are any chat menu options
    opts = (((chat_data.get("chatMenu") or {}).get("options") or {}).get("texts") or [])
    if opts and any(isinstance(s, str) and s.strip() for s in opts):
        return True
    
    return False


def continue_dialogue() -> Optional[dict]:
    """
    Action: Press Space to advance dialogue or click continue widget as fallback.
    """
    if find_chat_continue_widget():
        press_spacebar()
        return True

    elif can_continue():
        press_spacebar()
        return True
    
    # Fallback to clicking the "Click here to continue" widget
    result = click_chat_continue()
    if result:
        return result
    
    return None

def can_click_continue_widget() -> bool:
    """
    Check if any "Click here to continue" widget is visible and clickable.
    """
    from ..helpers.widgets import widget_exists, get_widget_info
    
    # Check Messagebox.CONTINUE widget (ID 15007748)
    if widget_exists(15007748):
        widget_info = get_widget_info(15007748)
        if widget_info and widget_info.get("ok"):
            widget_data = widget_info.get("widget", {})
            if widget_data.get("visible", False) and widget_data.get("hasListener", False):
                return True
    
    # Check LevelupDisplay.CONTINUE widget (ID 15269891)
    if widget_exists(15269891):
        widget_info = get_widget_info(15269891)
        if widget_info and widget_info.get("ok"):
            widget_data = widget_info.get("widget", {})
            if widget_data.get("visible", False) and widget_data.get("hasListener", False):
                return True
    
    return False

def click_continue_widget() -> Optional[dict]:
    """
    Action: Click any "Click here to continue" widget.
    """
    from ..helpers.widgets import widget_exists, get_widget_info
    
    # Try Messagebox.CONTINUE widget (ID 15007748) first
    if widget_exists(15007748):
        widget_info = get_widget_info(15007748)
        if widget_info and widget_info.get("ok"):
            widget_data = widget_info.get("widget", {})
            if widget_data.get("visible", False):
                bounds = widget_data.get("bounds")
                if bounds:
                    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
                    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
                    
                    step = {
                        "action": "click-continue-widget",
                        "click": {"type": "point", "x": x, "y": y},
                        "target": {"domain": "chat", "name": "continue_widget"},
                    }
                    return dispatch(step)
    
    # Try LevelupDisplay.CONTINUE widget (ID 15269891)
    if widget_exists(15269891):
        widget_info = get_widget_info(15269891)
        if widget_info and widget_info.get("ok"):
            widget_data = widget_info.get("widget", {})
            if widget_data.get("visible", False):
                bounds = widget_data.get("bounds")
                if bounds:
                    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
                    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
                    
                    step = {
                        "action": "click-continue-widget",
                        "click": {"type": "point", "x": x, "y": y},
                        "target": {"domain": "chat", "name": "continue_widget"},
                    }
                    return dispatch(step)
    
    return None

def choose_option(choice: Union[int, str]) -> Optional[dict]:
    """
    Action: choose an option by index (1..N) or by exact text.
    Uses number input for index: '1' selects the first, etc.
    """
    if not _can_choose_option():
        return None

    # Resolve to an index (human 1..N) if text is given
    if isinstance(choice, str):
        choice = choice.strip()
        options = _get_options()
        try:
            idx = next((i for i, s in enumerate(options) if choice.casefold() in (s or "").casefold()), 0)
        except ValueError:
            return None
    else:
        idx = int(choice)

    if idx < 1 or idx > 9:
        # Only 1..9 supported via single-key press
        return None

    step = {
        "action": "chat-choose-option",
        "click": {"type": "key", "key": str(idx)},
        "target": {"domain": "chat", "name": f"option-{idx}"},
    }
    return dispatch(step)



def get_dialogue_text_raw() -> Optional[str]:
    """Get raw dialogue text from any available chat widget."""
    return _get_dialogue_text_raw()

def dialogue_contains_text(phrase: str, case_sensitive: bool = False) -> bool:
    """
    Check if any dialogue text contains a specific phrase.
    
    Args:
        phrase: The text phrase to search for
        case_sensitive: Whether the search should be case sensitive (default: False)
    
    Returns:
        True if the phrase is found in any dialogue text, False otherwise
    """
    try:
        dialogue_texts = _get_dialogue_text_raw()
        
        if not dialogue_texts:
            return False
        
        # Prepare search phrase based on case sensitivity
        if not case_sensitive:
            phrase = phrase.lower()
        
        # Check each dialogue text for the phrase
        for text in dialogue_texts:
            # Prepare text based on case sensitivity
            search_text = text if case_sensitive else text.lower()
            
            if phrase in search_text:
                return True
        
        return False
        
    except Exception as e:
        print(f"[DIALOGUE] Error checking dialogue for phrase '{phrase}': {e}")
        return False


def get_chatbox_scroll_areas() -> list:
    """
    Retrieves all scroll area widgets within the main chatbox scroll area (162.57)
    that contain actual chat messages.
    
    The main chatbox scroll area container is widget group 162, child 57.
    Its runtime ID is 10616889, and its children also report this as their ParentId.
    The line widgets (e.g., S 162.58 Chatbox.LINE@) are also children of this
    container but typically do not have text content.
    
    Returns:
        A list of dictionaries, where each dictionary represents a chat message
        scroll area widget. Returns an empty list if no widgets are found or
        an error occurs.
    """
    
    CHATBOX_SCROLLAREA_GROUP_ID = 10616889
    
    try:
        children_widgets = get_widget_children(CHATBOX_SCROLLAREA_GROUP_ID)
        
        if not children_widgets:
            print(f"[CHAT] No children widgets found for {CHATBOX_SCROLLAREA_GROUP_ID}")
            return []


        filtered = []
        for widget in children_widgets.get("children"):
            if widget.get('id') == 10616889:
                filtered.append(widget)

        messages = []
        waiting_for_body = False

        for m in filtered:
            # work with a shallow copy so we retain metadata but don't mutate the original list
            msg = dict(m)
            text = msg.get("text", "")
            text_stripped = text.rstrip()

            # speaker/header line like "AllTheTime:"
            if text_stripped.endswith(":"):
                author = text_stripped[:-1].strip()
                msg["author"] = author
                msg["full_text"] = text_stripped  # we'll append the body next
                msg["parts"] = [text_stripped]  # [header, body?]
                messages.append(msg)
                waiting_for_body = True

            elif waiting_for_body:
                body = clean_rs(text)
                # append body to the LAST aggregated message
                messages[-1]["full_text"] += " " + body
                messages[-1]["parts"].append(body)
                waiting_for_body = False

            else:
                # stand-alone/system/broadcast message; keep metadata + cleaned text
                msg["author"] = None
                msg["full_text"] = clean_rs(text)
                msg["parts"] = [msg["full_text"]]
                messages.append(msg)

        return messages
        
    except Exception as e:
        print(f"[CHAT] Error retrieving chatbox scroll areas: {e}")
        return []

def click_chat_message(search_text: str, case_sensitive: bool = False) -> Optional[dict]:
    """
    Click on a chat message that matches the search text.
    
    Args:
        search_text: The text to search for in chat messages
        case_sensitive: Whether the search should be case sensitive (default: False)
    
    Returns:
        Dictionary with click result and interaction data if successful, None if no match found
    """
    try:
        # Get all chat message widgets
        messages = get_chatbox_scroll_areas()
        
        if not messages:
            print(f"[CHAT] No chat messages found")
            return None
        
        # Prepare search text based on case sensitivity
        if not case_sensitive:
            search_text = search_text.lower()
        
        # Find matching message
        matching_message = None
        for message in messages:
            # Check both full_text and text fields
            full_text = message.get("full_text", "")
            text = message.get("text", "")
            
            # Prepare text for comparison based on case sensitivity
            if not case_sensitive:
                full_text = full_text.lower()
                text = text.lower()
            
            # Check if search text matches either field
            if search_text in full_text or search_text in text:
                # Check if message is visible and clickable
                if message.get("visible", False):
                    matching_message = message
                    break
        
        if not matching_message:
            print(f"[CHAT] No matching visible chat message found for: '{search_text}'")
            return None
        
        # Get click coordinates from bounds
        bounds = matching_message.get("bounds", {})
        if not bounds:
            print(f"[CHAT] No bounds found for matching message")
            return None
        
        # Calculate center coordinates
        x = bounds.get("x", 0) + bounds.get("width", 0) // 2
        y = bounds.get("y", 0) + bounds.get("height", 0) // 2
        
        print(f"[CHAT] Clicking chat message at ({x}, {y}): {matching_message.get('full_text', '')[:50]}...")
        
        # Create click step
        step = {
            "action": "click-chat-message",
            "click": {"type": "point", "x": x, "y": y},
            "target": {"domain": "chat", "name": "message", "text": search_text},
        }
        
        return dispatch(step)
        
    except Exception as e:
        print(f"[CHAT] Error clicking chat message '{search_text}': {e}")
        return None


def find_chat_message(search_text: str, case_sensitive: bool = False) -> Optional[dict]:
    """
    Find a chat message that matches the search text without clicking it.
    
    Args:
        search_text: The text to search for in chat messages
        case_sensitive: Whether the search should be case sensitive (default: False)
    
    Returns:
        Dictionary with message data if found, None if no match found
    """
    try:
        # Get all chat message widgets
        messages = get_chatbox_scroll_areas()[:8]
        
        if not messages:
            return None
        
        # Prepare search text based on case sensitivity
        if not case_sensitive:
            search_text = search_text.lower()
        
        # Find matching message
        for message in messages:
            # Check both full_text and text fields
            full_text = message.get("full_text", "")
            text = message.get("text", "")
            
            # Prepare text for comparison based on case sensitivity
            if not case_sensitive:
                full_text = full_text.lower()
                text = text.lower()
            
            # Check if search text matches either field
            if search_text in full_text or search_text in text:
                return message
        
        return None
        
    except Exception as e:
        print(f"[CHAT] Error finding chat message '{search_text}': {e}")
        return None
