# session_ports.py
import re
import socket
import json
from pathlib import Path

from .constants import SESSIONS_DIR  # e.g. r"D:\repos\bot_runelite_IL\data\recording_sessions"

_USERNAME_PAT = re.compile(r"^rune?lite\s*-\s*(.+)$", re.I)

def _username_from_title(title: str) -> str | None:
    m = _USERNAME_PAT.match(title or "")
    return m.group(1).strip() if m else None

def _session_dir_for_username(user: str) -> Path:
    p = Path(SESSIONS_DIR) / user / "gamestates"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _probe_port_player_name(port: int, timeout_s: float = 0.5) -> str | None:
    """
    Ask IPC on `port` for {"cmd":"info"} → {"ok":true,"player":"<name>"}.
    If unsupported or unreachable, returns None.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as s:
            s.sendall((json.dumps({"cmd": "info"}) + "\n").encode("utf-8"))
            s.shutdown(socket.SHUT_WR)
            line = s.makefile("r", encoding="utf-8").readline().strip()
            if not line:
                return None
            resp = json.loads(line)
            if isinstance(resp, dict) and resp.get("ok"):
                name = resp.get("player") or resp.get("username")
                return str(name).strip() if name else None
    except Exception:
        return None
    return None

def _autofill_port_for_username(username: str, start: int = 17000, end: int = 17020) -> int | None:
    """
    Scan ports and return the one whose IPC 'info' player matches `username`.
    Requires the IPC plugin to implement {"cmd":"info"}.
    """
    for p in range(start, end + 1):
        pn = _probe_port_player_name(p)
        if pn and pn.lower() == username.lower():
            return p
    return None
