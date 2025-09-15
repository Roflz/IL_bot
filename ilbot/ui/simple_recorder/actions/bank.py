# ilbot/ui/simple_recorder/actions/banking.py

from .runtime import emit
from .travel import in_area
from ..constants import BANK_REGIONS
from ..helpers.context import get_payload, get_ui
from ..helpers.inventory import inv_has, inv_has_any
from ..helpers.bank import first_bank_slot, deposit_all_button_bounds
from ..helpers.rects import unwrap_rect, rect_center_xy
from ..helpers.utils import closest_object_by_names


def is_open(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return bool((payload.get("bank") or {}).get("bankOpen", False))

def is_closed(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return not is_open(payload)

def has_item(name: str, payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return first_bank_slot(payload, name) is not None

def open_bank(prefer: str | None = None, payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    prefer = (prefer or "").strip().lower()
    want_booth = prefer in ("bank booth", "grand exchange booth")
    want_banker = prefer == "banker"

    # --- NEW: if we're in the GE area, automatically prefer banker ---
    try:
        if in_area(BANK_REGIONS["GE"], payload):
            want_banker = True
            want_booth = False
    except Exception:
        pass
    # -----------------------------------------------------------------

    # ---------- gather candidates ----------
    objs = (payload.get("closestGameObjects") or []) + (payload.get("gameObjects") or []) + (payload.get("ge_booths") or [])
    objs = [o for o in objs if (o.get("name") or "").lower() not in ("", "null")]
    npcs = (payload.get("closestNPCs") or []) + (payload.get("npcs") or [])
    npcs = [n for n in npcs if (n.get("name") or "").lower() not in ("", "null")]

    def bank_index(actions) -> int | None:
        try:
            acts = [a.lower() for a in (actions or []) if a]
            return acts.index("bank") if "bank" in acts else None
        except Exception:
            return None

    # ---------- pick target ----------
    target = None
    target_domain = None

    if not want_banker:
        booth_names = ("bank booth", "grand exchange booth")
        for o in objs:
            nm = (o.get("name") or "").lower()
            idx = bank_index(o.get("actions"))
            if idx is not None and any(bn in nm for bn in booth_names):
                target, target_domain = o, "object"
                break
        if target is None and not want_booth and prefer != "banker":
            for o in objs:
                idx = bank_index(o.get("actions"))
                if idx is not None:
                    target, target_domain = o, "object"
                    break

    if target is None and not want_booth:
        for n in npcs:
            nm = (n.get("name") or "").lower()
            if ("banker" in nm) or want_banker:
                idx = bank_index(n.get("actions"))
                if idx is not None:
                    target, target_domain = n, "npc"
                    break

    if target is None:
        return None

    # ---------- build step ----------
    idx = bank_index(target.get("actions"))
    rect = unwrap_rect(target.get("clickbox"))
    name = target.get("name") or "Bank"

    if rect:
        cx, cy = rect_center_xy(rect)
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
    elif isinstance(target.get("canvasX"), (int, float)) and isinstance(target.get("canvasY"), (int, float)):
        cx, cy = int(target["canvasX"]), int(target["canvasY"])
        anchor = {}
        point = {"x": cx, "y": cy}
    else:
        return None

    if idx == 0:
        step = emit({
            "action": "open-bank",
            "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
            "target": {"domain": target_domain, "name": name, **anchor},
            "postconditions": ["bankOpen == true"],
        })
    else:
        # NOTE: Unchanged per your request
        step = emit({
            "action": "open-bank-context",
            "click": {
                "type": "context-select",
                "index": int(idx),
                "x": cx,
                "y": cy,
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120
            },
            "target": {"domain": target_domain, "name": name, **anchor} if rect else {"domain": target_domain, "name": name},
            "anchor": point
        })

    return ui.dispatch(step)


def deposit_inventory(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    if not inv_has_any(payload):
        return None
    rect = deposit_all_button_bounds(payload)
    if not rect:
        return None
    step = emit({
        "action": "bank-deposit-inv",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": "Deposit Inventory", "bounds": rect},
        "postconditions": [],
    })
    return ui.dispatch(step)


def withdraw_item(
    name: str,
    withdraw_x: int | None = None,
    withdraw_all: bool = False,
    ui=None,
    payload: dict | None = None
) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    slot = first_bank_slot(payload, name)
    if not slot:
        return None
    rect = unwrap_rect((slot or {}).get("bounds"))
    if not rect:
        return None

    cx, cy = rect_center_xy(rect)

    # Withdraw-All via context menu (no typing)
    if withdraw_all:
        all_idx = 4  # assumed 0-based index for "Withdraw-All" (adjust if needed)
        step = emit({
            "action": "withdraw-item-all",
            "click": {
                "type": "context-select",
                "index": int(all_idx),
                "x": cx,
                "y": cy,
                "row_height": 16,
                "start_dy": 18,
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": name, "bounds": rect},
        })
        return ui.dispatch(step)

    # Withdraw-X via context menu then type amount
    if withdraw_x is not None:
        x_idx = 3  # assumed 0-based index for "Withdraw-X" (adjust if needed)
        steps = [
            emit({
                "action": "withdraw-item-x-menu",
                "click": {
                    "type": "context-select",
                    "index": int(x_idx),
                    "x": cx,
                    "y": cy,
                    "row_height": 16,
                    "start_dy": 18,
                    "open_delay_ms": 120
                },
                "target": {"domain": "bank-slot", "name": name, "bounds": rect},
            }),
            emit({"action": "wait-after-context", "click": {"type": "wait", "ms": 300}}),
            emit({
                "action": "type-withdraw-x",
                "click": {"type": "type", "text": str(int(withdraw_x)), "enter": False, "per_char_ms": 15, "focus": True}
            }),
            emit({"action": "wait-before-enter", "click": {"type": "wait", "ms": 150}}),
            emit({"action": "confirm-withdraw-x", "click": {"type": "key", "key": "enter"}}),
        ]
        return ui.dispatch(steps)

    # Default: simple left-click withdraw
    step = emit({
        "action": "withdraw-item",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-slot", "name": name, "bounds": rect},
        "postconditions": [f"inventory contains '{name}'"],
    })
    return ui.dispatch(step)


def close_bank(ui=None) -> dict | None:
    if ui is None:
        ui = get_ui()

    step = emit({
        "action": "close-bank",
        "click": {"type": "key", "key": "esc"},
        "target": {"domain": "bank", "name": "Close"},
        "postconditions": ["bankOpen == false"],
    })
    return ui.dispatch(step)