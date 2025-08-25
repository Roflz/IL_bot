from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
import time
import pandas as pd

ABS_THRESHOLD = 10_000_000_000  # 1e10: anything smaller is assumed 'relative' (ms)
REQUIRED_ACTION_COLS = ["event_type", "x", "y", "btn", "key", "dx", "dy"]

def now_abs_ms() -> int:
    return int(time.time() * 1000)

@dataclass
class SessionClock:
    session_start_abs_ms: int

    def to_session(self, abs_or_rel_ms: pd.Series | Iterable[int]) -> pd.Series:
        s = pd.to_numeric(abs_or_rel_ms, errors="coerce")
        if s.notna().any() and s.quantile(0.95) > ABS_THRESHOLD:
            return (s - self.session_start_abs_ms).astype("int64")
        return s.fillna(0).astype("int64")

    def to_absolute(self, abs_or_rel_ms: pd.Series | Iterable[int]) -> pd.Series:
        s = pd.to_numeric(abs_or_rel_ms, errors="coerce")
        if s.notna().any() and s.quantile(0.95) < ABS_THRESHOLD:
            return (s + self.session_start_abs_ms).astype("int64")
        return s.fillna(0).astype("int64")


def coerce_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "x_in_window" in df.columns: rename["x_in_window"] = "x"
    if "y_in_window" in df.columns: rename["y_in_window"] = "y"
    if "scroll_dx"   in df.columns: rename["scroll_dx"]   = "dx"
    if "scroll_dy"   in df.columns: rename["scroll_dy"]   = "dy"
    if "timestamp_abs" in df.columns: rename["timestamp_abs"] = "timestamp_abs_ms"
    df = df.rename(columns=rename).copy()

    for c in ["timestamp_abs_ms", "t_session_ms", "timestamp", *REQUIRED_ACTION_COLS]:
        if c not in df.columns:
            if c in ["event_type", "key"]:
                df[c] = ""
            else:
                df[c] = 0
    return df


def infer_session_start_abs_ms(
    gamestates: Iterable[Dict[str, Any]] | None = None,
    meta: Optional[Dict[str, Any]] = None,
    features_df: Optional[pd.DataFrame] = None
) -> Optional[int]:
    if meta and "session_start_time" in meta:
        try:
            v = int(meta["session_start_time"])
            if v > ABS_THRESHOLD:
                return v
        except Exception:
            pass

    gs_ts = []
    for gs in gamestates or []:
        try:
            ts = int(gs.get("timestamp", 0))
            if ts > ABS_THRESHOLD:
                gs_ts.append(ts)
        except Exception:
            pass
    if gs_ts:
        return min(gs_ts)

    if features_df is not None:
        for col in ["timestamp_abs_ms", "timestamp"]:
            if col in features_df.columns:
                s = pd.to_numeric(features_df[col], errors="coerce").dropna()
                if len(s) and s.quantile(0.95) > ABS_THRESHOLD:
                    return int(s.min())

    return None
