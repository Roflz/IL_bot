from ilbot.ui.simple_recorder.helpers.rects import unwrap_rect, rect_center_xy


def craft_widget_rect(payload: dict, key: str) -> dict | None:
    w = (payload.get("crafting_widgets", {}) or {}).get(key)
    return unwrap_rect((w or {}).get("bounds") if isinstance(w, dict) else None)

def bank_widget_rect(payload: dict, key: str) -> dict | None:
    """Return screen-rect for a bank widget exported under data.bank_widgets[key]."""
    w = ((payload.get("bank_widgets") or {}).get(key) or {})
    b = (w.get("bounds") if isinstance(w, dict) else None)
    if isinstance(b, dict) and all(k in b for k in ("x","y","width","height")):
        return b
    return None

def rect_center_from_widget(w: dict | None) -> tuple[int | None, int | None]:
    rect = unwrap_rect((w or {}).get("bounds"))
    return rect_center_xy(rect)