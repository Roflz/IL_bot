# utils/common.py
import re
import time

_RS_TAG_RE = re.compile(r'</?col(?:=[0-9a-fA-F]+)?>')

def _clean_rs(s: str | None) -> str:
    if not s:
        return ""
    return _RS_TAG_RE.sub('', s)

def _fmt_age_ms(ms_ago: int) -> str:
    if ms_ago < 1000:
        return f"{ms_ago} ms ago"
    s = ms_ago / 1000.0
    if s < 60:
        return f"{s:.1f}s ago"
    m = int(s // 60)
    s = int(s % 60)
    return f"{m}m {s}s ago"

def _norm_name(s: str | None) -> str:
    return _clean_rs(s or "").strip().lower()

def _now_ms() -> int:
    return int(time.time() * 1000)
