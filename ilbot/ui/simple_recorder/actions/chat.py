# chat.py (actions)

from __future__ import annotations
from typing import Optional, Union

from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.chat import (
    can_continue as _can_continue,
    can_choose_option as _can_choose_option,
    get_options as _get_options,
    get_option as _get_option,
    dialogue_is_open as _dialogue_is_open,
    get_dialogue_text_raw as _get_dialogue_text_raw,
    get_clean_dialogue_text as _get_clean_dialogue_text,
    dialogue_contains_phrase as _dialogue_contains_phrase,
    dialogue_contains_any_phrase as _dialogue_contains_any_phrase,
    get_dialogue_info as _get_dialogue_info,
    has_informational_text as _has_informational_text,
)

def dialogue_is_open(payload: Optional[dict] = None) -> bool:
    if payload is None:
        payload = get_payload()
    return _dialogue_is_open(payload)

def can_continue(payload: Optional[dict] = None) -> bool:
    if payload is None:
        payload = get_payload()
    return _can_continue(payload)

def can_choose_option(payload: Optional[dict] = None) -> bool:
    if payload is None:
        payload = get_payload()
    return _can_choose_option(payload)

def get_options(payload: Optional[dict] = None):
    if payload is None:
        payload = get_payload()
    if  not _get_options(payload) == []:
        return _get_options(payload)
    else:
        return None


def get_option(index: int, payload: Optional[dict] = None):
    if payload is None:
        payload = get_payload()
    return _get_option(index, payload)

def dialogue_contains(substr: str, payload: Optional[dict] = None) -> bool:
    """
    True if the current dialogue (left or right) contains `substr` in its *text* block.
    Only checks the TEXT portion (not names or 'continue').
    """
    if not substr:
        return False
    if payload is None:
        payload = get_payload() or {}

    def _side_text(side_key: str) -> str:
        side = (payload.get(side_key) or {}) if isinstance(payload, dict) else {}
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

def option_exists(text: str, payload: dict | None = None) -> bool:
    """
    Return True if a chat menu option containing `text` (case-insensitive) exists.
    Uses payload['chatMenu']['options']['texts'].
    """
    if payload is None:
        payload = get_payload() or {}

    opts = (((payload.get("chatMenu") or {}).get("options") or {}).get("texts") or [])
    want = (text or "").strip().lower()
    if not want:
        return False

    for s in opts:
        if isinstance(s, str) and want in s.lower():
            return True
    return False

def any_chat_active(payload: Optional[dict] = None) -> bool:
    """
    Return True if any kind of chat/dialogue is currently active.
    This includes:
    - Regular dialogue (can continue)
    - Option menus (can choose option)
    - Any dialogue text present (left or right)
    - Any chat menu options available
    """
    if payload is None:
        payload = get_payload() or {}
    
    # Check if dialogue is open
    if _dialogue_is_open(payload):
        return True
    
    # Check if we can continue dialogue
    if _can_continue(payload):
        return True
    
    # Check if we can choose options
    if _can_choose_option(payload):
        return True
    
    # Check if there's any dialogue text present
    def _has_text(side_key: str) -> bool:
        side = (payload.get(side_key) or {}) if isinstance(payload, dict) else {}
        text_block = side.get("text") or {}
        text = (text_block.get("textStripped") or text_block.get("text") or "").strip()
        return bool(text)
    
    if _has_text("chatLeft") or _has_text("chatRight"):
        return True
    
    # Check if there are any chat menu options
    opts = (((payload.get("chatMenu") or {}).get("options") or {}).get("texts") or [])
    if opts and any(isinstance(s, str) and s.strip() for s in opts):
        return True
    
    return False


def continue_dialogue(payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Action: press Space to advance dialogue or click continue widget.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Try traditional space key method first
    if _can_continue(payload):
        step = emit({
            "action": "chat-continue",
            "click": {"type": "key", "key": "space"},
            "target": {"domain": "chat", "name": "continue"},
        })
        return ui.dispatch(step)
    
    # Try clicking the "Click here to continue" widget
    if can_click_continue_widget(payload):
        return click_continue_widget(payload, ui)
    
    return None

def can_click_continue_widget(payload: Optional[dict] = None) -> bool:
    """
    Check if the "Click here to continue" widget (ID 15007748) is visible and clickable.
    """
    from ..helpers.widgets import widget_exists, get_widget_info
    
    if not widget_exists(15007748):  # Messagebox.CONTINUE widget
        return False
    
    widget_info = get_widget_info(15007748)
    if not widget_info or not widget_info.get("ok"):
        return False
    
    widget_data = widget_info.get("widget", {})
    return widget_data.get("visible", False) and widget_data.get("hasListener", False)

def click_continue_widget(payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Action: Click the "Click here to continue" widget (ID 15007748).
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Check if the continue widget exists and is visible
    from ..helpers.widgets import widget_exists, get_widget_info
    
    if not widget_exists(15007748):  # Messagebox.CONTINUE widget
        return None
    
    # Get widget info to check if it's visible and get coordinates
    widget_info = get_widget_info(15007748)
    if not widget_info or not widget_info.get("ok"):
        return None
    
    widget_data = widget_info.get("widget", {})
    if not widget_data.get("visible", False):
        return None
    
    # Get click coordinates from bounds
    bounds = widget_data.get("bounds")
    if not bounds:
        return None
    
    x = bounds.get("x", 0) + bounds.get("width", 0) // 2
    y = bounds.get("y", 0) + bounds.get("height", 0) // 2
    
    step = emit({
        "action": "click-continue-widget",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "chat", "name": "continue_widget"},
    })
    return ui.dispatch(step)

def choose_option(choice: Union[int, str], payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Action: choose an option by index (1..N) or by exact text.
    Uses number input for index: '1' selects the first, etc.
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    if not _can_choose_option(payload):
        return None

    # Resolve to an index (human 1..N) if text is given
    if isinstance(choice, str):
        choice = choice.strip()
        options = _get_options(payload)
        try:
            idx = next((i for i, s in enumerate(options) if choice.casefold() in (s or "").casefold()), 0)
        except ValueError:
            return None
    else:
        idx = int(choice)

    if idx < 1 or idx > 9:
        # Only 1..9 supported via single-key press
        return None

    step = emit({
        "action": "chat-choose-option",
        "click": {"type": "key", "key": str(idx)},
        "target": {"domain": "chat", "name": f"option-{idx}"},
    })
    return ui.dispatch(step)


def type_tutorial_name(name: str, payload: Optional[dict] = None, ui=None) -> bool:
    """Type a character name in the tutorial name input field."""
    from ..helpers.widgets import get_widget_text, rect_center_from_widget
    
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get the name input widget from tutorial data
    tutorial_data = payload.get("tutorial", {})
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
    step = emit({
        "action": "click-name-input",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "name_input"}
    })
    
    ui.dispatch(step)
    
    # Wait a moment for the field to be focused
    import time
    time.sleep(0.2)
    
    # Type the character name
    step = emit({
        "action": "type-character-name",
        "click": {"type": "type", "text": name, "per_char_ms": 50},
        "target": {"domain": "tutorial", "name": "name_input"}
    })
    
    ui.dispatch(step)
    
    # Wait a moment then press enter
    time.sleep(0.2)
    
    # Press enter to confirm
    step = emit({
        "action": "press-enter",
        "click": {"type": "key", "key": "ENTER"},
        "target": {"domain": "tutorial", "name": "confirm_name"}
    })
    
    ui.dispatch(step)
    
    print(f"[TUTORIAL] Successfully entered character name: {name}")
    return True


def click_tutorial_set_name(payload: Optional[dict] = None, ui=None) -> bool:
    """Click the tutorial SET_NAME button to confirm the character name."""
    from ..helpers.widgets import get_tutorial_set_name, rect_center_from_widget
    
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get the set name widget
    set_name_widget = get_tutorial_set_name(payload)
    if not set_name_widget:
        print("[TUTORIAL] SET_NAME widget not found or not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(set_name_widget)
    if x is None or y is None:
        print("[TUTORIAL] Could not get SET_NAME coordinates")
        return False
    
    # Click on the SET_NAME button
    step = emit({
        "action": "click-set-name",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "set_name"}
    })
    
    result = ui.dispatch(step)
    if result is None:
        print("[TUTORIAL] Failed to click SET_NAME button")
        return False
    
    print("[TUTORIAL] Successfully clicked SET_NAME button")
    return True


def click_tutorial_lookup_name(payload: Optional[dict] = None, ui=None) -> bool:
    """Click the tutorial LOOK_UP_NAME button to check name availability."""
    from ..helpers.widgets import get_tutorial_lookup_name, rect_center_from_widget
    
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Get the lookup name widget
    lookup_widget = get_tutorial_lookup_name(payload)
    if not lookup_widget:
        print("[TUTORIAL] LOOK_UP_NAME widget not found or not visible")
        return False
    
    # Get click coordinates
    x, y = rect_center_from_widget(lookup_widget)
    if x is None or y is None:
        print("[TUTORIAL] Could not get LOOK_UP_NAME coordinates")
        return False
    
    # Click on the LOOK_UP_NAME button
    step = emit({
        "action": "click-lookup-name",
        "click": {"type": "point", "x": x, "y": y},
        "target": {"domain": "tutorial", "name": "lookup_name"}
    })
    
    result = ui.dispatch(step)
    if result is None:
        print("[TUTORIAL] Failed to click LOOK_UP_NAME button")
        return False
    
    print("[TUTORIAL] Successfully clicked LOOK_UP_NAME button")
    return True

def get_dialogue_text_raw(payload: Optional[dict] = None) -> Optional[str]:
    """Get raw dialogue text from any available chat widget."""
    if payload is None:
        payload = get_payload()
    return _get_dialogue_text_raw(payload)

def get_clean_dialogue_text(payload: Optional[dict] = None) -> Optional[str]:
    """Get dialogue text with HTML tags and color codes stripped."""
    if payload is None:
        payload = get_payload()
    return _get_clean_dialogue_text(payload)

def dialogue_contains_phrase(phrase: str, payload: Optional[dict] = None, case_sensitive: bool = False) -> bool:
    """Check if dialogue contains a specific phrase."""
    if payload is None:
        payload = get_payload()
    return _dialogue_contains_phrase(phrase, payload, case_sensitive)

def dialogue_contains_any_phrase(phrases: list, payload: Optional[dict] = None, case_sensitive: bool = False) -> bool:
    """Check if dialogue contains any of the given phrases."""
    if payload is None:
        payload = get_payload()
    return _dialogue_contains_any_phrase(phrases, payload, case_sensitive)

def get_dialogue_info(payload: Optional[dict] = None) -> dict:
    """Get comprehensive dialogue information."""
    if payload is None:
        payload = get_payload()
    return _get_dialogue_info(payload)

def has_informational_text(payload: Optional[dict] = None) -> bool:
    """Check if there's informational text overlay (like tutorial messages) that can be read."""
    if payload is None:
        payload = get_payload()
    return _has_informational_text(payload)
