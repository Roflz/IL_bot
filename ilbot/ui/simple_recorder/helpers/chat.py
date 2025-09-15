# chat.py (helpers)

from __future__ import annotations
from typing import Optional, List, Dict, Any

from ..helpers.context import get_payload

# Expected payload structure (only visible widgets exported):
# payload["chat_dialogue"] = {
#   "open": bool,
#   "name": {"text": "...", "exists": bool} | None,
#   "text": {"text": "...", "exists": bool} | None,
#   "continue": {"text": "...", "exists": bool} | None
# }
# payload["chat_menu"] = {
#   "open": bool,
#   "parentId": 14352385,
#   "options": [
#       {"index": 0, "text": "Option text 1"},
#       {"index": 1, "text": "Option text 2"},
#       ...
#   ]
# }

def _dlg_left(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatLeft") or {}) if isinstance(payload, dict) else {}

def _dlg_right(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatRight") or {}) if isinstance(payload, dict) else {}

def _menu(payload: Optional[dict] = None) -> dict:
    if payload is None:
        payload = get_payload() or {}
    return (payload.get("chatMenu") or {}) if isinstance(payload, dict) else {}

def dialogue_is_open(payload: Optional[dict] = None) -> bool:
    L = _dlg_left(payload)
    R = _dlg_right(payload)

    # explicit flag wins if present
    if bool(L.get("open")) or bool(R.get("open")):
        return True

    def any_visible(d: dict) -> bool:
        name = d.get("name") or {}
        text = d.get("text") or {}
        cont = d.get("continue") or {}
        return bool(name.get("exists") or text.get("exists") or cont.get("exists"))

    return any_visible(L) or any_visible(R)

def can_continue(payload: Optional[dict] = None) -> bool:
    L = _dlg_left(payload)
    R = _dlg_right(payload)
    l = (L.get("continue") or {}).get("exists")
    r = (R.get("continue") or {}).get("exists")
    return bool(l or r)

def can_choose_option(payload: Optional[dict] = None) -> bool:
    m = _menu(payload)
    if bool(m.get("open")):
        return True
    opts = m.get("options") or []
    return opts['exists']

def get_options(payload: Optional[dict] = None) -> List[str]:
    m = _menu(payload)
    opts = ((m.get("options") or {}).get("texts") or [])
    out: List[str] = []
    for o in opts:
        t = o.strip()
        if t:
            out.append(t)
    return out

def get_option(index: int, payload: Optional[dict] = None) -> Optional[str]:
    """
    index may be 0-based or 1-based. If 1..N, we treat as human index.
    """
    opts = get_options(payload)
    if index >= 1:
        idx = index - 1
    else:
        idx = index
    if 0 <= idx < len(opts):
        return opts[idx]
    return None

def get_dialogue_text(payload: Optional[dict] = None) -> Dict[str, str]:
    """
    Returns { <speaker_name>: <dialogue_text>, ... }.
    If both left and right are visible, both are included.
    """
    out: Dict[str, str] = {}
    for d in (_dlg_left(payload), _dlg_right(payload)):
        if not d:
            continue
        name = ((d.get("name") or {}).get("text") or "").strip()
        text = ((d.get("text") or {}).get("text") or "").strip()
        if name or text:
            out[name] = text
    return out
