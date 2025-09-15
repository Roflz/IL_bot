# helpers/quests.py

from ilbot.ui.simple_recorder.helpers.context import get_payload

def _norm(s: str) -> str:
    """Lightweight normalization so 'Romeo & Juliet' == 'romeo and juliet'."""
    return "".join(ch for ch in s.lower().replace("&", "and") if ch.isalnum())

def _find_quest_key(name: str, quests: dict) -> str | None:
    """Exact key if present; else case/format-insensitive match."""
    if not isinstance(quests, dict) or not name:
        return None
    if name in quests:  # fast path exact
        return name
    n = _norm(name)
    for k in quests.keys():
        if _norm(k) == n:
            return k
    return None

def quest_state(name: str, payload: dict | None = None) -> str | None:
    """
    Return the quest's state string from payload['quests'] or None if unknown.
    States are the exact strings exported by RuneLite (e.g., NOT_STARTED, IN_PROGRESS, FINISHED).
    """
    if payload is None:
        payload = get_payload()
    quests = (payload or {}).get("quests") or {}
    key = _find_quest_key(name, quests)
    return quests.get(key) if key else None

def quest_is(name: str, state: str, payload: dict | None = None) -> bool:
    """True if quest_state(name) equals state (case-insensitive)."""
    s = quest_state(name, payload)
    return isinstance(s, str) and s.upper() == (state or "").upper()

def quest_in_progress(name: str, payload: dict | None = None) -> bool:
    return quest_is(name, "IN_PROGRESS", payload)

def quest_finished(name: str, payload: dict | None = None) -> bool:
    return quest_is(name, "FINISHED", payload)

def quest_not_started(name: str, payload: dict | None = None) -> bool:
    return quest_is(name, "NOT_STARTED", payload)
