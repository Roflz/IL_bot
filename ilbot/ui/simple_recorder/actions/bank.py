# ilbot/ui/simple_recorder/actions/banking.py
from typing import Optional, List

from . import chat, ge, widgets
from .runtime import emit
from .timing import wait_until
from ..constants import BANK_REGIONS
from ..helpers.context import get_payload, get_ui
from ..helpers.inventory import inv_has, inv_has_any
from ..helpers.bank import first_bank_slot, deposit_all_button_bounds
from ..helpers.ipc import ipc_send
from ..helpers.rects import unwrap_rect, rect_center_xy
from ..helpers.utils import closest_object_by_names
from ..plans.ge_trade import in_area
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
                    "option": "Bank",  # match by text
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
                action="Bank" if idx != 0 else None,
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

    if not inv_has_any():
        return None
    rect = deposit_all_button_bounds(payload)
    if not rect:
        return None
    step = {
        "action": "bank-deposit-inv",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-widget", "name": "Deposit Inventory", "bounds": rect},
        "postconditions": [],
    }
    return ui.dispatch(step)


def withdraw_item(
    name: str | list,
    withdraw_x: int | None = None,
    withdraw_all: bool = False,
    ui=None,
    payload: dict | None = None
) -> dict | None:
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()

    # Handle list of items
    if isinstance(name, list):
        return withdraw_items(name, withdraw_x, withdraw_all, ui, payload)
    
    # Handle single item
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
            "option": "withdraw-all",
            "click": {
                "type": "context-select",        # uses live menu geometry via IPC "menu"        # case-insensitive match
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
            "option": "withdraw-x",
            "click": {
                "type": "context-select",
                "target": item_name.lower(),
                "x": int(cx),
                "y": int(cy),
                "open_delay_ms": 120
            },
            "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        })
        ui.dispatch(step1)
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
        
        return True

    # --- Default: simple left-click withdraw (Withdraw-1 / custom shift-click config) ---
    step = emit({
        "action": "withdraw-item",
        "click": {"type": "rect-center"},
        "target": {"domain": "bank-slot", "name": item_name, "bounds": rect},
        "postconditions": [f"inventory contains '{item_name}'"],
    })
    return dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)


def withdraw_items(
    items: list,
    withdraw_x: int | None = None,
    withdraw_all: bool = False,
    ui=None,
    payload: dict | None = None
) -> dict | None:
    """
    Withdraw multiple items from the bank.
    
    Args:
        items: List of item names to withdraw
        withdraw_x: Amount to withdraw for each item (if None, uses default behavior)
        withdraw_all: If True, withdraw all of each item
        ui: UI instance
        payload: Game state payload
        
    Returns:
        Result of the last successful withdrawal, or None if all failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    if not isinstance(items, list) or not items:
        print("[BANK] withdraw_items: Invalid items list provided")
        return None
    
    print(f"[BANK] Withdrawing {len(items)} items: {items}")
    
    last_result = None
    successful_withdrawals = 0
    
    for item in items:
        if not isinstance(item, str):
            print(f"[BANK] Skipping invalid item: {item}")
            continue
            
        print(f"[BANK] Withdrawing item: {item}")
        
        # Try to withdraw this item
        result = withdraw_item(
            name=item,
            withdraw_x=withdraw_x,
            withdraw_all=withdraw_all,
            ui=ui,
            payload=payload
        )
        
        if result:
            last_result = result
            successful_withdrawals += 1
            print(f"[BANK] Successfully withdrew: {item}")
        else:
            print(f"[BANK] Failed to withdraw: {item}")
        
        # Small delay between withdrawals to avoid overwhelming the interface
        import time
        time.sleep(0.3)
    
    print(f"[BANK] Withdrawal complete: {successful_withdrawals}/{len(items)} items successful")
    return last_result


def close_bank(ui=None) -> dict | None:
    if ui is None:
        ui = get_ui()

    step = {
        "action": "close-bank",
        "click": {"type": "key", "key": "esc"},
        "target": {"domain": "bank", "name": "Close"},
        "postconditions": ["bankOpen == false"],
    }
    return ui.dispatch(step)

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


def deposit_equipment(payload: dict | None = None, ui=None) -> dict | None:
    """Deposit all equipped items into the bank."""
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
 
    widgets.click_widget(786478)


def interact(item_name: str | list, action: str | list, payload: dict | None = None, ui=None) -> dict | None:
    """
    Interact with bank inventory items using context clicks.
    
    Args:
        item_name: Name of the item(s) to interact with (string or list)
        action: Action(s) to perform (string or list of strings)
                If list, will try each action until one is available
        payload: Game state payload (optional)
        ui: UI instance (optional)
        
    Returns:
        Result of the interaction, or None if failed
    """
    if payload is None:
        payload = get_payload()
    if ui is None:
        ui = get_ui()
    
    # Handle list of items
    if isinstance(item_name, list):
        return _interact_multiple_items(item_name, action, payload, ui)
    
    # Handle single item
    return _interact_single_item(item_name, action, payload, ui)


def _interact_single_item(item_name: str, action: str | list, payload: dict, ui) -> dict | None:
    """Interact with a single bank inventory item."""
    from ..helpers.bank_inventory import find_bank_inventory_item
    
    # Find the item in bank inventory
    item = find_bank_inventory_item(item_name, payload)
    if not item:
        print(f"[BANK] Item '{item_name}' not found in bank inventory")
        return None
    
    # Get item bounds for clicking
    bounds = item.get("bounds")
    if not bounds:
        print(f"[BANK] No bounds found for item '{item_name}'")
        return None
    
    # Get canvas coordinates for context click
    canvas = item.get("canvas")
    if not canvas:
        print(f"[BANK] No canvas coordinates found for item '{item_name}'")
        return None
    
    # Handle single action
    if isinstance(action, str):
        return _try_action(item_name, action, bounds, canvas, payload, ui)
    
    # Handle list of actions - try each one until one works
    if isinstance(action, list):
        for action_option in action:
            if not isinstance(action_option, str):
                print(f"[BANK] Skipping invalid action: {action_option}")
                continue
            
            print(f"[BANK] Trying action '{action_option}' for item '{item_name}'")
            result = _try_action(item_name, action_option, bounds, canvas, payload, ui)
            if result:
                print(f"[BANK] Successfully used action '{action_option}' for item '{item_name}'")
                return result
            else:
                print(f"[BANK] Action '{action_option}' not available for item '{item_name}'")
        
        print(f"[BANK] No available actions found for item '{item_name}' from list: {action}")
        return None
    
    print(f"[BANK] Invalid action type: {type(action)}")
    return None


def _try_action(item_name: str, action: str, bounds: dict, canvas: dict, payload: dict, ui) -> dict | None:
    """Try a specific action on an item."""
    print(f"[BANK] Attempting action '{action}' on item '{item_name}'")
    
    # Create context click step
    step = emit({
        "action": "bank-inventory-interact",
        "option": action,
        "click": {
            "type": "context-select",
            "target": item_name.lower(),
            "x": int(canvas["x"]),
            "y": int(canvas["y"]),
            "open_delay_ms": 120
        },
        "target": {"domain": "bank-inventory", "name": item_name, "bounds": bounds},
        "postconditions": [],
    })
    
    try:
        result = dispatch_with_camera(step, ui=ui, payload=payload, aim_ms=420)
        return result
    except Exception as e:
        print(f"[BANK] Action '{action}' failed for item '{item_name}': {e}")
        return None


def _interact_multiple_items(item_names: list, action: str | list, payload: dict, ui) -> dict | None:
    """
    Interact with multiple bank inventory items.
    
    Args:
        item_names: List of item names to interact with
        action: Action(s) to perform on each item (string or list)
        payload: Game state payload
        ui: UI instance
        
    Returns:
        Result of the last successful interaction, or None if all failed
    """
    if not isinstance(item_names, list) or not item_names:
        print("[BANK] Invalid item list provided")
        return None
    
    print(f"[BANK] Interacting with {len(item_names)} items: {item_names}")
    print(f"[BANK] Using action(s): {action}")
    
    last_result = None
    successful_interactions = 0
    
    for item_name in item_names:
        if not isinstance(item_name, str):
            print(f"[BANK] Skipping invalid item: {item_name}")
            continue
        
        print(f"[BANK] Interacting with item: {item_name}")
        
        # Try to interact with this item
        result = _interact_single_item(item_name, action, payload, ui)
        
        if result:
            last_result = result
            successful_interactions += 1
            print(f"[BANK] Successfully interacted with: {item_name}")
        else:
            print(f"[BANK] Failed to interact with: {item_name}")
        
        # Small delay between interactions to avoid overwhelming the interface
        import time
        time.sleep(0.2)
    
    print(f"[BANK] Interaction complete: {successful_interactions}/{len(item_names)} items successful")
    return last_result


def get_bank_inventory(payload: Optional[dict] = None) -> List[dict]:
    """
    Get all items in the bank (actual bank contents, not interface).
    
    Args:
        payload: Optional payload, will get fresh if None
        
    Returns:
        List of bank items with name, quantity, and other details
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_bank"}, payload)
    if not resp or not resp.get("ok"):
        print(f"[BANK] Failed to get bank contents: {resp.get('err', 'Unknown error')}")
        return []
    
    bank_items = resp.get("items", [])
    print(f"[BANK] Retrieved {len(bank_items)} bank items")
    return bank_items


def get_item_count(item_name: str, payload: dict | None = None) -> int:
    """
    Get the count of a specific item in the bank.
    
    Args:
        item_name: Name of the item to count
        payload: Game state payload (optional)
        
    Returns:
        Number of the item in the bank, or 0 if not found
    """
    if payload is None:
        payload = get_payload()
    
    bank_items = get_bank_contents(payload)
    if not bank_items:
        return 0
    
    total_count = 0
    for item in bank_items:
        if item.get("name") == item_name:
            total_count += item.get("quantity", 0)
    
    return total_count


def get_bank_contents(payload: dict | None = None) -> list:
    """
    Get all items in the bank.
    
    Args:
        payload: Game state payload (optional)
        
    Returns:
        List of bank items, or empty list if failed
    """
    if payload is None:
        payload = get_payload()
    
    resp = ipc_send({"cmd": "get_bank"}, payload)
    if not resp or not resp.get("ok"):
        print(f"[BANK] Failed to get bank contents: {resp.get('err', 'Unknown error')}")
        return []
    
    bank_items = resp.get("items", [])
    print(f"[BANK] Retrieved {len(bank_items)} bank items")
    return bank_items