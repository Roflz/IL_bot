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


def continue_dialogue(payload: Optional[dict] = None, ui=None) -> Optional[dict]:
    """
    Action: press Space to advance dialogue. (Works for ChatLeft.CONTINUE.)
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    if not _can_continue(payload):
        return None

    step = emit({
        "action": "chat-continue",
        "click": {"type": "key", "key": "space"},
        "target": {"domain": "chat", "name": "continue"},
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
