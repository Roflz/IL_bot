from ..helpers.inventory import inv_has, inv_has_any
from ..helpers.context import get_payload  # for optional payload

def has_item(name: str, min_qty: int = 1, payload: dict | None = None) -> bool:
    """
    True if inventory contains item `name` with quantity >= min_qty.
    Falls back to simple presence when min_qty <= 1.
    Tries inv_count(...) if available; otherwise scans payload["inventory"]["items"].
    """
    if payload is None:
        payload = get_payload()
    if min_qty <= 1:
        return inv_has(payload, name)

    # Prefer an existing counter helper if your codebase provides it
    try:
        inv_count_fn = globals().get("inv_count") or globals().get("inventory_count")
        if callable(inv_count_fn):
            return (int(inv_count_fn(payload, name)) >= int(min_qty))
    except Exception:
        pass

    # Fallback: manual scan of the payload
    try:
        items = (((payload or {}).get("inventory") or {}).get("items") or [])
        want = (name or "").strip().lower()
        total = 0
        for it in items:
            nm = (it.get("name") or "").strip().lower()
            if nm == want:
                q = it.get("quantity", it.get("qty", 1))
                try:
                    q = int(q)
                except Exception:
                    q = 1
                total += max(1, q)
        return total >= int(min_qty)
    except Exception:
        return False



def is_empty(payload: dict | None = None) -> bool:
    """
    True if the player's inventory is empty.
    Accepts an optional payload; if omitted, uses the current global payload.
    """
    if payload is None:
        payload = get_payload()
    try:
        # preferred helper
        return not inv_has_any(payload)
    except Exception:
        # conservative fallback if helper/payload structure changes
        inv = (payload or {}).get("inventory") or []
        return len(inv) == 0 or all(
            (not item) or int(item.get("quantity", 0)) <= 0
            for item in inv
        )
