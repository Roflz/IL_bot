# ilbot/ui/simple_recorder/actions/banking.py
from . import chat, ge
from .runtime import emit
from .timing import wait_until
from .travel import in_area
from ..constants import BANK_REGIONS
from ..helpers.context import get_payload, get_ui
from ..helpers.inventory import inv_has, inv_has_any
from ..helpers.bank import first_bank_slot, deposit_all_button_bounds
from ..helpers.rects import unwrap_rect, rect_center_xy
from ..helpers.utils import closest_object_by_names
from ..services.camera_integration import dispatch_with_camera


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
    """
    Open a bank using direct IPC detection with pathing and door handling.
    If a CLOSED door lies on the path to the bank, click the earliest blocking door first.
    """
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

    max_retries = 3
    
    for attempt in range(max_retries):
        # Get fresh payload and bank data on each retry
        fresh_payload = get_payload()
        
        # Use direct IPC to get bank objects and NPCs - OPTIMIZED VERSION
        from ..helpers.ipc import ipc_send
        
        # Try to find bank booth first
        booth_resp = ipc_send({"cmd": "find_object", "name": "bank booth", "types": ["GAME"]}, fresh_payload)
        booth_found = booth_resp and booth_resp.get("ok") and booth_resp.get("found")
        
        # Try to find banker NPC
        banker_resp = ipc_send({"cmd": "find_npc", "name": "banker"}, fresh_payload)
        banker_found = banker_resp and banker_resp.get("ok") and banker_resp.get("found")
        
        # Convert to old format for compatibility
        objs = [booth_resp.get("object")] if booth_found else []
        npcs = [banker_resp.get("npc")] if banker_found else []

        def bank_index(actions) -> int | None:
            try:
                acts = [a.lower() for a in (actions or []) if a]
                if "bank" in acts:
                    return acts.index("bank")
                elif "use" in acts:
                    return acts.index("use")
                else:
                    return None
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
            continue  # Try next attempt

        # 1) Check for doors on the path to the bank
        gx, gy = target["world"].get("x"), target["world"].get("y")
        if isinstance(gx, int) and isinstance(gy, int):
            from ..helpers.ipc import ipc_path
            from ..helpers.navigation import _first_blocking_door_from_waypoints
            from .travel import _handle_door_opening
            
            wps, dbg_path = ipc_path(fresh_payload, goal=(gx, gy))
            door_plan = _first_blocking_door_from_waypoints(wps)
            if door_plan:
                # Handle door opening with retry logic and recently traversed door tracking
                if not _handle_door_opening(door_plan, fresh_payload, ui):
                    # Door opening failed after retries, continue to next attempt
                    continue

        # 2) Click the bank
        idx = bank_index(target.get("actions"))
        rect = unwrap_rect(target.get("clickbox"))
        name = target.get("name") or "Bank"
        world_coords = {"x": gx, "y": gy, "p": 0}  # Use the world coordinates from the target

        if rect:
            cx, cy = rect_center_xy(rect)
            anchor = {"bounds": rect}
            point = {"x": cx, "y": cy}
        elif isinstance(target["canvas"].get("x"), (int, float)) and isinstance(target["canvas"].get("y"), (int, float)):
            cx, cy = int(target["canvas"]["x"]), int(target["canvas"]["y"])
            anchor = {}
            point = {"x": cx, "y": cy}
        else:
            continue  # Try next attempt

        if idx == 0:
            step = emit({
                "action": "open-bank",
                "click": ({"type": "rect-center"} if rect else {"type": "point", **point}),
                "target": {"domain": target_domain, "name": name, **anchor, "world": world_coords},
                "postconditions": ["bankOpen == true"],
            })
        else:
            step = emit({
                "action": "open-bank-context",
                "click": {
                    "type": "context-select",
                    "x": point["x"],  # canvas coords where to open the menu
                    "y": point["y"],
                    "option": "bank",  # match by text
                    "target": "banker",
                    "open_delay_ms": 120
                },
                "target": {"domain": "npc", "name": "Banker", "world": world_coords}
            })

        # Use centralized click with camera function
        from ..services.click_with_camera import click_object_with_camera, click_npc_with_camera
        
        if target_domain == "object":
            result = click_object_with_camera(
                object_name=name,
                action="bank" if idx != 0 else None,
                action_index=idx,
                world_coords=world_coords,
                ui=ui,
                payload=fresh_payload,
                aim_ms=420
            )
        else:  # target_domain == "npc"
            result = click_npc_with_camera(
                npc_name=name,
                action="bank" if idx != 0 else None,
                action_index=idx,
                world_coords=world_coords,
                ui=ui,
                payload=fresh_payload,
                aim_ms=420
            )
        
        if result:
            # Wait for bank interface to actually open
            if wait_until(is_open, max_wait_ms=5000, min_wait_ms=200):
                return result
            else:
                # Bank didn't open, continue to next attempt
                continue

    return None


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
    return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)


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

    rect = unwrap_rect(slot.get("bounds"))
    if not rect:
        return None

    cx, cy = rect_center_xy(rect)
    item_name = str(name).strip()

    # --- Withdraw-All via live context menu ---
    if withdraw_all:
        step = emit({
            "action": "withdraw-item-all",
            "click": {
                "type": "context-select",        # uses live menu geometry via IPC "menu"
                "option": "withdraw-all",        # case-insensitive match
                "target": item_name.lower(),     # substring match against menu target
                "x": int(cx),                    # right-click anchor (canvas)
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        })
        return ui.dispatch(step)

    # --- Withdraw-X via live context menu + type amount ---
    if withdraw_x is not None:
        # Step 1: Right-click and select "Withdraw-X"
        step1 = emit({
            "action": "withdraw-item-x-menu",
            "click": {
                "type": "context-select",
                "option": "withdraw-x",
                "target": item_name.lower(),
                "x": int(cx),
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        })
        dispatch_with_camera(step1, ui=ui, payload=payload, aim_ms=420)
        if not wait_until(ge.chat_qty_prompt_active, min_wait_ms=300, max_wait_ms=3000):
            return None
        
        # Step 3: Type the quantity
        step3 = emit({
            "action": "type-withdraw-x",
            "click": {"type": "type", "text": str(int(withdraw_x)), "enter": False, "per_char_ms": 15, "focus": True}
        })
        ui.dispatch(step3)
        if not wait_until(lambda: ge.buy_chatbox_text_input_contains(str(withdraw_x)), min_wait_ms=300, max_wait_ms=3000):
            return None

        
        # Step 4: Press Enter to confirm
        step4 = emit({"action": "confirm-withdraw-x", "click": {"type": "key", "key": "enter"}})
        ui.dispatch(step4)
        
        # Wait until inventory contains the expected amount
        from .inventory import inventory_has_amount
        if not wait_until(lambda: inventory_has_amount(item_name, int(withdraw_x)), min_wait_ms=500, max_wait_ms=5000):
            return None
        
        return True

    # --- Default: simple left-click withdraw (Withdraw-1 / custom shift-click config) ---
    step = emit({
        "action": "withdraw-item",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        "postconditions": [f"inventory contains '{item_name}'"],
    })
    return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)

def close_bank(ui=None) -> dict | None:
    if ui is None:
        ui = get_ui()

    step = emit({
        "action": "close-bank",
        "click": {"type": "key", "key": "esc"},
        "target": {"domain": "bank", "name": "Close"},
        "postconditions": ["bankOpen == false"],
    })
    return dispatch_with_camera(step, ui=ui, payload=get_payload(), aim_ms=420)

def toggle_note_mode(payload: dict | None = None, ui=None) -> dict | None:
    """Toggle bank note mode (withdraw as note/withdraw as item)"""
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    from ..helpers.bank import bank_note_selected
    bw = payload.get("bank_widgets") or {}
    node = bw.get("withdraw_note_toggle") or {}
    b = node.get("bounds") or {}
    
    if int(b.get("width") or 0) > 0 and int(b.get("height") or 0) > 0:
        step = emit({
            "action": "bank-note-toggle",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Withdraw as Note", "bounds": b},
            "postconditions": [],
        })
        return ui.dispatch(step)
    return None

def ensure_note_mode_disabled(payload: dict | None = None, ui=None) -> dict | None:
    """Ensure bank note mode is disabled (withdraw as items, not notes)"""
    if payload is None:
        payload = get_payload()
    
    from ..helpers.bank import bank_note_selected
    if bank_note_selected(payload):
        return toggle_note_mode(payload, ui)
    return None