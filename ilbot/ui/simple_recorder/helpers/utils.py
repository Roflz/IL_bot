import re, time
from ilbot.ui.simple_recorder.helpers.runtime_utils import dispatch

_STEP_HITS: dict[str, int] = {}
_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')

def clean_rs(s: str | None) -> str:
    if not s:
        return ""
    return _RS_TAG_RE.sub('', s)

def norm_name(s: str | None) -> str:
    return clean_rs(s or "").strip().lower()

def now_ms() -> int:
    return int(time.time() * 1000)

def closest_object_by_names(names: list[str]) -> dict | None:
    from .runtime_utils import ipc
    objects_data = ipc.get_closest_objects() or {}
    wanted = [n.lower() for n in names]

    # Fallback to generic nearby objects
    for obj in (objects_data.get("objects") or []):
        nm = norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj

    return None

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