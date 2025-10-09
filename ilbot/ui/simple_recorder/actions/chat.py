# chat.py (actions)

from __future__ import annotations
from typing import Optional, Union

from .timing import wait_until
from .widgets import click_chat_continue, find_chat_continue_widget
from ..helpers.runtime_utils import ui, dispatch
from ..helpers.chat import (
    can_continue as _can_continue,
    can_choose_option as _can_choose_option,
    get_options as _get_options,
    get_option as _get_option,
    dialogue_is_open as _dialogue_is_open,
    get_dialogue_text_raw as _get_dialogue_text_raw,
)
from ..helpers.utils import press_spacebar


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
        if wait_until(lambda: not can_continue(), max_wait_ms=2000):
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


def type_tutorial_name(name: str) -> bool:
    """Type a character name in the tutorial name input field."""
    from ..helpers.widgets import get_widget_text, rect_center_from_widget

    # Get the name input widget from tutorial data
    from ..helpers.runtime_utils import ipc
    tutorial_data = ipc.get_tutorial() or {}
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
    import time
    time.sleep(0.5)
    
    # Type the character name
    step = {
        "action": "type-character-name",
        "click": {"type": "type", "text": name, "per_char_ms": 50},
        "target": {"domain": "tutorial", "name": "name_input"}
    }
    
    dispatch(step)
    
    # Wait a moment then press enter
    time.sleep(1.0)
    
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
    from ..helpers.widgets import get_tutorial_set_name, rect_center_from_widget

    # Get the set name widget
    set_name_widget = get_tutorial_set_name()
    if not set_name_widget:
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
    from ..helpers.widgets import get_tutorial_lookup_name, rect_center_from_widget

    # Get the lookup name widget
    lookup_widget = get_tutorial_lookup_name()
    if not lookup_widget:
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

def get_dialogue_text_raw() -> Optional[str]:
    """Get raw dialogue text from any available chat widget."""
    return _get_dialogue_text_raw()
