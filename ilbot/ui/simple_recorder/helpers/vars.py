from threading import Lock
from typing import Optional
from ilbot.ui.simple_recorder.helpers.ipc import ipc_send


def get_var(var_id: int, payload: dict | None = None, timeout: float = 0.35) -> int | None:
    """
    Ask the IPC server for a varbit value by ID.
    Returns the integer value, or None on error.
    """
    resp = ipc_send({"cmd": "get-var", "id": int(var_id)}, payload=payload, timeout=timeout)
    if not resp or not resp.get("ok"):
        return None
    return resp.get("value")
