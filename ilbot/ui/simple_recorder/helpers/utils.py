import re, time

from ilbot.ui.simple_recorder.actions.runtime import emit
from ilbot.ui.simple_recorder.helpers.context import get_payload, get_ui

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

def mark_step_done(step_id: str):
    """Record that we just executed this step (used to advance UI flows)."""
    if step_id:
        _STEP_HITS[step_id] = now_ms()

def step_recent(step_id: str, max_ms: int = 1800) -> bool:
    """True if step_id was executed in the last max_ms milliseconds."""
    t = _STEP_HITS.get(step_id)
    return isinstance(t, int) and (now_ms() - t) <= max_ms

def fmt_age_ms(ms_ago: int) -> str:
    if ms_ago < 1000:
        return f"{ms_ago} ms ago"
    s = ms_ago / 1000.0
    if s < 60:
        return f"{s:.1f}s ago"
    m = int(s // 60)
    s = int(s % 60)
    return f"{m}m {s}s ago"

def closest_object_by_names(payload: dict, names: list[str]) -> dict | None:
    wanted = [n.lower() for n in names]

    # Fallback to generic nearby objects
    for obj in (payload.get("closestGameObjects") or []):
        nm = norm_name(obj.get("name"))
        if any(w in nm for w in wanted):
            return obj

    return None

def list_plans_for_ui() -> list[tuple[str, str]]:
    from ilbot.ui.simple_recorder.plans import PLAN_REGISTRY  # lazy: avoid circular on import
    preferred = ["GE_SELL_BUY", "RING_CRAFT", "GO_TO_RECT"]
    seen = set()
    out = []
    for pid in preferred:
        if pid in PLAN_REGISTRY:
            out.append((pid, PLAN_REGISTRY[pid].label)); seen.add(pid)
    for pid, plan in PLAN_REGISTRY.items():
        if pid not in seen:
            out.append((pid, plan.label))
    return out

def get_plan(plan_id: str):
    from ilbot.ui.simple_recorder.plans import PLAN_REGISTRY  # lazy
    return PLAN_REGISTRY.get(plan_id)

_CRAFT_ANIMS = {899}
def is_crafting_anim(anim_id: int) -> bool:
    return anim_id in _CRAFT_ANIMS


def press_enter(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    step = emit({
        "id": "key-enter",
        "action": "key",
        "description": "Press Enter",
        "click": {"type": "key", "key": "ENTER"},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)

def press_esc(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    step = emit({
        "id": "key-esc",
        "action": "key",
        "description": "Press Escape",
        "click": {"type": "key", "key": "ESC"},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)