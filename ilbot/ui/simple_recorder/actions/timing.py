# actions/timing.py
import time
from typing import Callable, Optional, Union

from ilbot.ui.simple_recorder.helpers.context import get_ui, set_payload, get_payload

BoolOrCallable = Union[bool, Callable[[], bool]]

def wait_until(
    condition: Callable,           # must be callable
    min_wait_ms: int = 0,
    max_wait_ms: Optional[int] = 10000,
    poll_ms: int = 50,
) -> bool:
    """
    Block until `condition` returns True.

    - Always wait at least `min_wait_ms` before you are allowed to return True.
    - After the min wait, poll up to `max_wait_ms` total (if provided).
    - Before every poll, refresh the payload from UI (if available) and set it globally.
    """
    if not callable(condition):
        raise TypeError("wait_until(condition): condition must be callable")

    min_wait = max(0, int(min_wait_ms))
    max_wait = None if max_wait_ms is None else max(0, int(max_wait_ms))
    poll = max(1, int(poll_ms))

    t0 = time.monotonic()

    # minimum wait
    if min_wait > 0:
        time.sleep(min_wait / 1000.0)

    def _now_ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    def _refresh_payload() -> dict:
        ui = get_ui()
        if ui and hasattr(ui, "latest_payload"):
            try:
                p = ui.latest_payload() or {}
                set_payload(p)
                return p
            except Exception:
                pass
        # fallback to whatever is currently stored
        return get_payload() or {}

    def _check() -> bool:
        # refresh before every check
        payload = _refresh_payload()
        try:
            # prefer calling with payload; fall back to zero-arg if incompatible
            try:
                return bool(condition(payload))
            except TypeError:
                return bool(condition())
        except Exception:
            return False

    # immediate check after min wait
    if _check():
        return True

    if max_wait is not None and _now_ms() >= max_wait:
        return False

    while True:
        time.sleep(poll / 1000.0)
        if _check():
            return True
        if max_wait is not None and _now_ms() >= max_wait:
            return False