from helpers.runtime_utils import dispatch
import random
import time


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


def type_text(
    text: str,
    *,
    enter: bool = True,
    per_char_ms_range: tuple[int, int] = (45, 110),
    chunk_pause_ms_range: tuple[int, int] = (40, 180),
) -> dict | None:
    """
    Type text in a more human-like way:
    - variable per-character timing (via per_char_ms)
    - small pauses between random chunks
    """
    s = "" if text is None else str(text)
    if s == "":
        return None

    lo, hi = per_char_ms_range
    lo = max(5, int(lo))
    hi = max(lo, int(hi))

    # Split into a few chunks so speed can vary during the string.
    # Keep chunk count small to avoid excessive IPC calls for long strings.
    n = len(s)
    if n <= 2:
        chunks = [s]
    else:
        max_chunks = 3 if n <= 8 else 4
        k = random.randint(2, max_chunks)
        # random cut points
        cuts = sorted(random.sample(range(1, n), k - 1))
        cuts = [0] + cuts + [n]
        chunks = [s[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]

    last_result = None
    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        step = {
            "id": f"type-text-{i}",
            "action": "type",
            "description": f"Type text: {chunk}",
            "click": {
                "type": "type",
                "text": chunk,
                "enter": bool(enter and is_last),
                "per_char_ms": random.randint(lo, hi),
            },
        }
        last_result = dispatch(step)
        if not is_last:
            # pause a bit between chunks (humans do this)
            pause_ms = random.randint(int(chunk_pause_ms_range[0]), int(chunk_pause_ms_range[1]))
            time.sleep(max(0.0, pause_ms / 1000.0))

    return last_result
