from .utils import norm_name

def bank_slots_matching(payload: dict, names: list[str]) -> list[dict]:
    """Return bank slots whose itemName matches (case-insensitive) any of names."""
    want = { (n or "").strip().lower() for n in names if n }
    out = []
    for s in (payload.get("bank", {}).get("slots") or []):
        nm = (s.get("itemName") or "").strip().lower()
        qty = int(s.get("quantity") or 0)
        if nm in want and qty > 0:
            out.append(s)
    return out

def first_bank_slot(payload: dict, name: str) -> dict | None:
    n = norm_name(name)
    best = None
    for s in bank_slots(payload):
        if norm_name(s.get("itemName")) == n:
            if best is None:
                best = s
            else:
                q1 = int(s.get("quantity") or 0)
                q2 = int(best.get("quantity") or 0)
                if q1 > q2 or (q1 == q2 and int(s.get("slotId") or 9_999) < int(best.get("slotId") or 9_999)):
                    best = s
    return best

def bank_slots(payload: dict) -> list[dict]:
    return (payload.get("bank", {}) or {}).get("slots", []) or []

def bank_note_selected(payload: dict) -> bool:
    bw = payload.get("bank_widgets") or {}
    node = bw.get("withdraw_note_toggle") or {}
    # exporter now provides: {"bounds": {...}, "selected": bool}
    sel = node.get("selected")
    if isinstance(sel, bool):
        return sel
    # fallback (older payloads): treat missing as not-selected
    return False

def bank_qty_all_selected(payload: dict) -> bool:
    bw = (payload.get("bank_widgets") or {})
    qall = (bw.get("withdraw_quantity_all") or {})
    return bool(qall.get("selected"))

def deposit_all_button_bounds(payload: dict) -> dict | None:
    bw = payload.get("bank_widgets") or {}
    node = bw.get("deposit_inventory") or {}
    b = node.get("bounds") or node  # support old shape that stored bounds directly
    return b if int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0 else None

def nearest_banker(payload: dict) -> dict | None:
    me = payload.get("player") or {}
    mx, my, mp = int(me.get("worldX") or 0), int(me.get("worldY") or 0), int(me.get("plane") or 0)
    best, best_d2 = None, 1e18
    for npc in (payload.get("closestNPCs") or []) + (payload.get("npcs") or []):
        nm = (npc.get("name") or "").lower()
        if "banker" not in nm:
            continue
        if int(npc.get("plane") or 0) != mp:
            continue
        nx, ny = int(npc.get("worldX") or 0), int(npc.get("worldY") or 0)
        dx, dy = nx - mx, ny - my
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = npc, d2
    return best