# ilbot/ui/simple_recorder/actions/ge.py
# Tiny, reusable GE action builders (return single-click/keypress steps).
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.rects import rect_center_xy
from ..helpers.utils import closest_object_by_names
from ..helpers.widgets import rect_center_from_widget
from ..helpers.ge import (
    ge_open, ge_offer_open, ge_selected_item_is, ge_first_buy_slot_btn,
    widget_by_id_text_contains, ge_qty_button, ge_qty_matches, chat_qty_prompt_active, find_ge_plus5_bounds
)

# ---------- simple reads ----------
def is_open(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return ge_open(payload)

def is_closed(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return not is_open(payload)

def offer_open(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return ge_offer_open(payload)

def selected_item_is(name: str, payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return ge_selected_item_is(payload, name)

def qty_is(qty: int, payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return ge_qty_matches(payload, qty)

def can_confirm(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return bool(widget_by_id_text_contains(payload, 30474266, "confirm"))

def has_collect(payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    return bool(widget_by_id_text_contains(payload, 30474246, "collect"))

def buy_chatbox_text_input_contains(substr: str, payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    cb = (payload.get("ge_buy_chatbox") or {})
    p = (cb.get("prompt") or {})
    text = (p.get("textStripped") or p.get("text") or "")
    s = (substr or "").strip()
    return bool(s) and (s.lower() in text.lower())

def buy_chatbox_first_item_is(name: str, payload: dict | None = None) -> bool:
    if payload is None:
        payload = get_payload()
    cb = (payload.get("ge_buy_chatbox") or {})
    items = (cb.get("items") or [])
    if not items:
        return False
    first = items[0] or {}
    nm = first.get("nameStripped") or first.get("name") or ""
    want = (name or "").strip()
    return bool(want) and (want.lower() in nm.lower())



# ---------- one-step actions ----------
def open_ge(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # --- Prefer the Clerk; use live context menu to pick "Exchange" ---
    npc = None
    for cand in (payload.get("closestNPCs") or []) + (payload.get("npcs") or []):
        nm = (cand.get("name") or "").lower()
        nid = int(cand.get("id") or -1)
        if "grand exchange clerk" in nm or (2148 <= nid <= 2151):
            npc = cand
            break

    if npc and isinstance(npc.get("canvasX"), (int, float)) and isinstance(npc.get("canvasY"), (int, float)):
        cx = int(npc["canvasX"])
        cy = int(npc["canvasY"]) - 8  # keep your slight lift
        step = emit({
            "id": "ge-open-clerk",
            "action": "open-ge-context",
            "description": "Open GE (clerk via context menu: Exchange)",
            "click": {
                "type": "context-select",   # uses IPC 'menu' to match rows by text + rect
                "option": "exchange",       # case-insensitive
                "target": "grand exchange clerk",  # substring match against target text
                "x": cx,                    # right-click anchor (canvas)
                "y": cy,
                "open_delay_ms": 120
            },
            "target": {"domain": "npc", "name": npc.get("name"), "id": npc.get("id")},
        })
        return ui.dispatch(step)

    # --- Fallback: booth simple left-click (unchanged) ---
    booth = closest_object_by_names(payload, ["grand exchange booth"])
    if booth and isinstance(booth.get("canvasX"), (int, float)) and isinstance(booth.get("canvasY"), (int, float)):
        step = emit({
            "id": "ge-open-booth",
            "action": "click",
            "description": "Open GE (booth)",
            "click": {"type": "point", "x": int(booth["canvasX"]), "y": int(booth["canvasY"]) - 12},
            "target": {"domain": "object", "name": booth.get("name"), "id": booth.get("id")},
        })
        return ui.dispatch(step)

    return None
def close_ge(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None: payload = get_payload()
    if ui is None: ui = get_ui()

    step = emit({
        "id": "ge-close",
        "action": "click",
        "description": "Close GE",
        "click": {"type": "key", "key": "escape"},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)

def begin_buy_offer(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    w = ge_first_buy_slot_btn(payload)
    if not w:
        return None
    cx, cy = rect_center_from_widget(w)
    step = emit({
        "id": "ge-begin-buy",
        "action": "click",
        "description": "Open buy offer",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "widget", "name": "ge_first_buy_slot", "bounds": w.get("bounds")},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)


def type_item_name(name: str, payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    if selected_item_is(name):
        return None

    step = emit({
        "id": "ge-type-item",
        "action": "type",
        "description": f"Type item: {name}",
        "click": {"type": "type", "text": name, "per_char_ms": 20},
    })
    return ui.dispatch(step)


def set_quantity(payload: dict | None = None, qty: int = 0, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    if qty_is(payload, qty):
        return None

    if not chat_qty_prompt_active(payload):
        dot = ge_qty_button(payload)
        if dot:
            cx, cy = rect_center_from_widget(dot)
            step = emit({
                "id": "ge-qty-dot",
                "action": "click",
                "description": "Open quantity prompt",
                "click": {"type": "point", "x": cx, "y": cy},
                "target": {"domain": "widget", "name": "qty_button", "bounds": dot.get("bounds")},
                "preconditions": [], "postconditions": []
            })
            return ui.dispatch(step)
        return None

    step = emit({
        "id": "ge-qty-type",
        "action": "type",
        "description": f"Type buy quantity: {qty}",
        "click": {"type": "type", "text": str(int(qty)), "enter": True, "per_char_ms": 10, "focus": True},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)



def click_plus5(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None: payload = get_payload()
    if ui is None: ui = get_ui()

    rect = find_ge_plus5_bounds(payload)
    if not rect:
        return None
    cx, cy = rect_center_xy(rect)
    step = emit({
        "id": "ge-plus5",
        "action": "click",
        "description": "+5% price",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "widget", "name": "+5%", "bounds": rect},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)

def confirm_buy(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    w = widget_by_id_text_contains(payload, 30474266, "confirm")
    if not w:
        return None
    cx, cy = rect_center_from_widget(w)
    step = emit({
        "id": "ge-confirm",
        "action": "click",
        "description": "Confirm buy offer",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "widget", "name": "confirm", "bounds": w.get("bounds")},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)


def collect_to_inventory(payload: dict | None = None, ui=None) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    w = widget_by_id_text_contains(payload, 30474246, "collect")
    if not w:
        return None
    cx, cy = rect_center_from_widget(w)
    step = emit({
        "id": "ge-collect",
        "action": "click",
        "description": "Collect items to inventory",
        "click": {"type": "point", "x": cx, "y": cy},
        "target": {"domain": "widget", "name": "collect", "bounds": w.get("bounds")},
        "preconditions": [], "postconditions": []
    })
    return ui.dispatch(step)