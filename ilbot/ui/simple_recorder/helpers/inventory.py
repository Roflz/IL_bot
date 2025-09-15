from .utils import norm_name

from .rects import unwrap_rect


def inv_slots(payload: dict) -> list[dict]:
    return (payload.get("inventory", {}) or {}).get("slots", []) or []

def inv_has(payload: dict, name: str) -> bool:
    n = norm_name(name)
    return any(norm_name(s.get("itemName")) == n for s in inv_slots(payload))

def inv_count(payload: dict, name: str) -> int:
    n = norm_name(name)
    return sum(int(s.get("quantity") or 0) for s in inv_slots(payload)
               if norm_name(s.get("itemName")) == n)

def first_inv_slot(payload: dict, name: str) -> dict | None:
    n = norm_name(name)
    for s in inv_slots(payload):
        if norm_name(s.get("itemName")) == n:
            return s
    return None

def inventory_has_foreign_items(payload: dict) -> bool:
    allowed = {"Ring mould", "Gold bar", "Sapphire", "Emerald"}
    for slot in (payload.get("inventory") or {}).get("slots") or []:
        if int(slot.get("quantity") or 0) > 0 and (slot.get("itemName") or "") not in allowed:
            return True
    return False

def inventory_ring_slots(payload: dict) -> list[dict]:
    out = []
    for s in inv_slots(payload):
        nm = norm_name(s.get("itemName"))
        if "ring" in nm and "mould" not in nm and int(s.get("quantity") or 0) > 0:
            out.append(s)
    return out

def inv_slot_bounds(payload: dict, slot_id: int) -> dict | None:
    iw = (payload.get("inventory_widgets") or {}).get(str(slot_id)) or {}
    return unwrap_rect(iw.get("bounds") if isinstance(iw, dict) else None)

def coins(payload: dict) -> int:
    return inv_count(payload, "Coins")

def inv_has_any(payload: dict) -> bool:
    return int((payload.get("inventory") or {}).get("totalItems") or 0) > 0