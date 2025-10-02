# ilbot/ui/simple_recorder/actions/ge.py
# Tiny, reusable GE action builders (return single-click/keypress steps).
from .runtime import emit
from ..helpers.context import get_payload, get_ui
from ..helpers.rects import rect_center_xy
from ..helpers.utils import closest_object_by_names
from ..helpers.widgets import rect_center_from_widget
from ..helpers.ge import (
    ge_open, ge_offer_open, ge_selected_item_is, ge_first_buy_slot_btn,
    widget_by_id_text_contains, ge_qty_button, ge_qty_matches, chat_qty_prompt_active, find_ge_plus5_bounds,
    widget_by_id_text, ge_buy_confirm_widget
)
from ..helpers.utils import press_enter
from ..actions import travel as trav, bank, inventory as inv
from ..actions.timing import wait_until
from ..constants import BANK_REGIONS
import time

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
            "option": "exchange",  # case-insensitive
            "click": {
                "type": "context-select",   # uses IPC 'menu' to match rows by text + rect
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

    if qty_is(qty):
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

    if not buy_chatbox_text_input_contains(str(qty)):
        step = emit({
            "id": "ge-qty-type",
            "action": "type",
            "description": f"Type buy quantity: {qty}",
            "click": {"type": "type", "text": str(qty), "per_char_ms": 20},
            "preconditions": [], "postconditions": []
        })
        return ui.dispatch(step)

    else:
        return press_enter(payload, ui)


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


def buy_item_from_ge(item, ui) -> bool | None:
    """
    Returns True when all items are in inventory.
    Otherwise performs exactly one small step toward buying them and returns None.
    Safe to call every tick.
    
    Args:
        item: Name of the item to buy (str) or dict of items to buy {item_name: (quantity, price_bumps)}
        ui: UI instance for dispatching actions
    """
    payload = get_payload()
    
    # Normalize input to dict
    if isinstance(item, str):
        items_to_buy = {item: (1, 5)}  # (quantity, price_bumps)
    else:
        items_to_buy = item
    
    # Check if we already have all items
    if all(inv.has_item(item_name) for item_name in items_to_buy.keys()):
        close_ge()
        if is_closed():
            return True
        return None

    # Go to GE
    if not trav.in_area(BANK_REGIONS["GE"]):
        trav.go_to_ge()
        return None

    # Ensure coins: if we don't have any, grab from bank quickly (GE has a bank close by)
    if not inv.has_item("coins"):
        if bank.is_closed():
            bank.open_bank()
            wait_until(bank.is_open, max_wait_ms=6000)
            return None
        if bank.is_open():
            # make space then withdraw all coins
            if not inv.is_empty():
                bank.deposit_inventory()
                wait_until(inv.is_empty, max_wait_ms=4000, min_wait_ms=150)
                return None
            bank.withdraw_item("coins", withdraw_all=True)
            wait_until(lambda: inv.has_item("coins"), max_wait_ms=3000)
            return None
    elif bank.is_open():
        bank.close_bank()
        return None

    # Open GE
    if is_closed():
        open_ge()
        wait_until(is_open, max_wait_ms=5000)
        return None

    # Buy each item one by one
    for item_name, (item_quantity, item_price_bumps) in items_to_buy.items():
        if inv.has_item(item_name):
            continue  # Skip if we already have this item
            
        # Open buy offer panel
        if not offer_open():
            if has_collect():
                collect_to_inventory()
                return None
            begin_buy_offer()
            if not wait_until(offer_open, max_wait_ms=5000):
                return None
            return None

        # Type item name and select first result
        if not buy_chatbox_first_item_is(item_name) and not selected_item_is(item_name):
            type_item_name(item_name)
            if not wait_until(lambda: buy_chatbox_first_item_is(item_name), min_wait_ms=600, max_wait_ms=3000):
                return None
            return None

        elif not selected_item_is(item_name):
            press_enter(payload, ui)
            if not wait_until(lambda: selected_item_is(item_name), min_wait_ms=600, max_wait_ms=3000):
                return None
            return None

        if selected_item_is(item_name):
            # Set quantity if needed
            if item_quantity > 1:
                if not qty_is(item_quantity):
                    set_quantity(qty=item_quantity)
                    return None

            # Nudge price a few times (+5%)
            for _ in range(max(0, int(item_price_bumps))):
                plus = widget_by_id_text(30474266, "+5%")
                if not plus:
                    break
                cx, cy = rect_center_from_widget(plus)
                ui.dispatch({
                    "id": "ge-plus5",
                    "action": "click",
                    "description": "+5% price",
                    "target": {"name": "+5%", "bounds": plus.get("bounds")},
                    "click": {"type": "point", "x": cx, "y": cy},
                })
                time.sleep(0.25)

            # Confirm buy
            confirm_buy()
            wait_until(lambda: ge_buy_confirm_widget() is None, min_wait_ms=600, max_wait_ms=3000)
            wait_until(has_collect, max_wait_ms=60000)
            if has_collect():
                collect_to_inventory()
                return None
            return None

    # Close GE after all items are bought
    close_ge()
    return True