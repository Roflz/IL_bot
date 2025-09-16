from ilbot.ui.simple_recorder.helpers.context import get_payload


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
