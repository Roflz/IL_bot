# actions/context.py
from contextvars import ContextVar
from typing import Any, Optional

_current_payload: ContextVar[Optional[dict[str, Any]]] = ContextVar("current_payload", default=None)
_current_ui: ContextVar[Optional[dict[str, Any]]] = ContextVar("current_ui", default=None)

def set_payload(payload: dict | None) -> None:
    _current_payload.set(payload)

def get_payload() -> dict:
    p = _current_payload.get()
    return p if isinstance(p, dict) else {}

def set_ui(ui: dict | None) -> None:
    _current_ui.set(ui)

def get_ui() -> dict:
    p = _current_ui.get()
    return p