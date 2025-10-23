import re, time, csv
from pathlib import Path
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

def type_text(text: str) -> dict | None:
    step = {
        "id": "type-text",
        "action": "type",
        "description": f"Type text: {text}",
        "click": {"type": "type", "text": text, "per_char_ms": 20},
    }
    return dispatch(step)


def get_world_from_csv(username: str) -> int | None:
    """
    Get the world number for a specific character from the character_stats.csv file.
    
    Args:
        username: Character username to look up
        
    Returns:
        - World number (int) if character found
        - None if character not found or error occurred
    """
    try:
        csv_file = Path("D:/repos/bot_runelite_IL/ilbot/ui/simple_recorder/character_data/character_stats.csv")
        
        if not csv_file.exists():
            return None
            
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('username') == username:
                    world_str = row.get('world_number')
                    if world_str:
                        return int(world_str)
                    return None
                    
        return None
        
    except Exception as e:
        print(f"[get_world_from_csv] Error reading CSV: {e}")
        return None