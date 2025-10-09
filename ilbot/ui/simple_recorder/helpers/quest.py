# helpers/quests.py

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

def quest_state(name: str) -> str | None:
    """
    Return the quest's state string from IPC or None if unknown.
    States are the exact strings exported by RuneLite (e.g., NOT_STARTED, IN_PROGRESS, FINISHED).
    """
    from .runtime_utils import ipc
    quests_data = ipc.get_quests() or {}
    quests = quests_data.get("quests") or {}
    key = _find_quest_key(name, quests)
    return quests.get(key) if key else None

def quest_is(name: str, state: str) -> bool:
    """True if quest_state(name) equals state (case-insensitive)."""
    s = quest_state(name)
    return isinstance(s, str) and s.upper() == (state or "").upper()

def quest_in_progress(name: str) -> bool:
    return quest_is(name, "IN_PROGRESS")

def quest_finished(name: str) -> bool:
    return quest_is(name, "FINISHED")

def quest_not_started(name: str) -> bool:
    return quest_is(name, "NOT_STARTED")
