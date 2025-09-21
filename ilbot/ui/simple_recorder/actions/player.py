from typing import Optional

from ilbot.ui.simple_recorder.helpers.context import get_payload
from ilbot.ui.simple_recorder.helpers.vars import get_var


def get_player_plane(payload: dict | None = None, default=None):
    """
    Return the player's plane (0/1/2/3) from a payload.
    Works if payload is the full game payload or the `player` sub-dict.
    """
    if payload is None:
        payload = get_payload()
    if not isinstance(payload, dict):
        return default
    # If it's the full payload, expect a "player" key; otherwise treat it as the player dict.
    player = payload.get("player", payload)
    plane = player.get("plane")
    return int(plane) if isinstance(plane, (int, float)) else default

def in_cutscene(payload: dict | None = None, timeout: float = 0.35) -> bool:
    """
    Check if the player is in a cutscene.
    Returns True if varbit 542 == 1, False otherwise.
    """
    val = get_var(542, payload=payload, timeout=timeout)
    return val == 1

