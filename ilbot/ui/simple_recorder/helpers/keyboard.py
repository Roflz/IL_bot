from ilbot.ui.simple_recorder.helpers.runtime_utils import dispatch


def press_enter() -> dict | None:
    step = {
        "id": "key-enter",
        "action": "key",
        "description": "Press Enter",
        "click": {"type": "key", "key": "ENTER"},
        "preconditions": [], "postconditions": []
    }
    return dispatch(step)


def press_esc() -> dict | None:
    step = {
        "id": "key-esc",
        "action": "key",
        "description": "Press Escape",
        "click": {"type": "key", "key": "ESC"},
        "preconditions": [], "postconditions": []
    }
    return dispatch(step)


def press_backspace() -> dict | None:
    step = {
        "id": "key-backspace",
        "action": "key",
        "description": "Press Backspace",
        "click": {"type": "key", "key": "BACKSPACE"},
        "preconditions": [], "postconditions": []
    }
    return dispatch(step)


def press_spacebar() -> dict | None:
    step = {
        "id": "key-spacebar",
        "action": "key",
        "description": "Press Spacebar",
        "click": {"type": "key", "key": "SPACE"},
        "preconditions": [], "postconditions": []
    }
    return dispatch(step)


def type_text(text: str) -> dict | None:
    step = {
        "id": "type-text",
        "action": "type",
        "description": f"Type text: {text}",
        "click": {"type": "type", "text": text, "per_char_ms": 20},
    }
    return dispatch(step)
