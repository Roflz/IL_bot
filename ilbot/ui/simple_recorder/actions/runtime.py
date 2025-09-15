# ilbot/ui/simple_recorder/actions/runtime.py
from contextvars import ContextVar
from typing import Optional

class _Emitter:
    __slots__ = ("steps",)
    def __init__(self):
        self.steps: list[dict] = []

    def add(self, step: dict) -> None:
        if isinstance(step, dict):
            self.steps.append(step)

_current: ContextVar[Optional[_Emitter]] = ContextVar("actions_emitter", default=None)

def begin():
    """
    Start capturing steps for the current plan tick.
    Returns (emitter, token). Call end(token) in a finally block.
    """
    em = _Emitter()
    token = _current.set(em)
    return em, token

def end(token):
    _current.reset(token)

def emit(step: Optional[dict]) -> list[dict]:
    """
    Append a step to the current emitter if present, and return a LIST suitable
    for plan["steps"]. This lets plans do:
        plan["steps"] = emit(bank.close_bank())
    Or accumulate:
        steps += emit(bank.deposit_inventory(...))
    """
    em = _current.get()
    if isinstance(step, dict):
        if em is not None:
            em.add(step)
        return [step]  # <-- critical: return list, not dict
    return []  # nothing emitted -> empty list

# (Optional) tiny helper to merge multiple emit returns
def merge(*parts: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in parts:
        if isinstance(p, list):
            out.extend(p)
    return out
