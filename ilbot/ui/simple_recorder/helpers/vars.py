from ilbot.ui.simple_recorder.helpers.runtime_utils import ipc


def get_var(var_id: int, timeout: float = 0.35) -> int | None:
    """
    Ask the IPC server for a varbit value by ID.
    Returns the integer value, or None on error.
    """
    resp = ipc.get_var(int(var_id), timeout=timeout)
    if not resp or not resp.get("ok"):
        return None
    return resp.get("value")
