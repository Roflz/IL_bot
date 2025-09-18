# actions/context.py
from contextvars import ContextVar
from typing import Any, Optional

_current_payload: ContextVar[Optional[dict[str, Any]]] = ContextVar("current_payload", default=None)
_current_ui: ContextVar[Optional[dict[str, Any]]] = ContextVar("current_ui", default=None)

def set_payload(payload: dict | None) -> None:
    _current_payload.set(payload)

def get_payload() -> dict:
    """
    Always try to return the freshest payload by asking the UI shim
    (which reads the newest JSON). Fall back to the last cached payload.
    """
    # Prefer live read if UI is available
    ui = _current_ui.get()
    if ui and hasattr(ui, "latest_payload"):
        try:
            latest = ui.latest_payload() or {}
            if isinstance(latest, dict):
                _current_payload.set(latest)  # keep cache in sync
                return latest
        except Exception:
            pass

    # Fallback to whatever is currently cached
    p = _current_payload.get()
    return p if isinstance(p, dict) else {}

def set_ui(ui: dict | None) -> None:
    _current_ui.set(ui)

def get_ui() -> dict:
    p = _current_ui.get()
    return p