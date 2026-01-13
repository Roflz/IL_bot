import logging
from .utils import norm_name, rect_beta_xy
from .widgets import click_listener_on, get_widget_info
from actions.travel import in_area
from constants import BANK_WIDGETS
from .runtime_utils import ipc, dispatch


def bank_slots_matching(names: list[str]) -> list[dict]:
    """Return bank slots whose itemName matches (case-insensitive) any of names."""
    try:
        bank_data = ipc.get_bank()
        if not bank_data or not bank_data.get("ok"):
            logging.error("[bank_slots_matching] helpers/bank.py: Failed to get bank data from IPC")
            return []
        
        want = { (n or "").strip().lower() for n in names if n }
        out = []
        for s in bank_data.get("slots", []):
            # Clean item name by removing item ID (e.g., "Leather|14" -> "Leather")
            item_name = s.get("itemName") or ""
            if "|" in item_name:
                item_name = item_name.split("|")[0]
            
            nm = item_name.strip().lower()
            qty = int(s.get("quantity") or 0)
            if nm in want and qty > 0:
                out.append(s)
        return out
    except Exception as e:
        logging.error(f"[bank_slots_matching] helpers/bank.py: {e}")
        return []

def first_bank_slot(name: str) -> dict | None:
    try:
        n = norm_name(name)
        slots = bank_slots()
        for s in slots:
            # Clean item name by removing item ID (e.g., "Leather|14" -> "Leather")
            item_name = s.get("name") or ""
            if "|" in item_name:
                item_name = item_name.split("|")[0]
            
            if norm_name(item_name) == n:
                return s
        return {'quantity': 0}
    except Exception as e:
        logging.error(f"[first_bank_slot] helpers/bank.py: {e}")
        return None

def bank_slots() -> list[dict]:
    try:
        bank_data = ipc.get_bank()
        if not bank_data or not bank_data.get("ok"):
            logging.error("[bank_slots] helpers/bank.py: Failed to get bank data from IPC")
            return []
        return bank_data.get("items", [])
    except Exception as e:
        logging.error(f"[bank_slots] helpers/bank.py: {e}")
        return []

def bank_note_selected() -> bool:
    """Check if bank note mode is selected (withdraw as notes)."""
    try:
        # When it has a click listener, note mode is enabled
        # When it has no click listener, note mode is disabled
        return click_listener_on(BANK_WIDGETS["NOTE"])
    except Exception as e:
        logging.error(f"[bank_note_selected] helpers/bank.py: {e}")
        return False

def bank_qty_all_selected() -> bool:
    """Check if 'All' quantity button is selected for withdrawals."""
    try:
        # Prefer varbit-based mode detection (consistent with actions/bank.py)
        return is_quantityall_selected()
    except Exception as e:
        logging.error(f"[bank_qty_all_selected] helpers/bank.py: {e}")
        return False

def deposit_all_button_bounds() -> dict | None:
    """Get bounds for the deposit inventory button."""
    try:
        deposit_buttons_data = ipc.get_bank_deposit_buttons()
        if not deposit_buttons_data or not deposit_buttons_data.get("ok"):
            logging.error("[deposit_all_button_bounds] helpers/bank.py: Failed to get bank deposit buttons from IPC")
            return None
        
        deposit_buttons = deposit_buttons_data.get("deposit_buttons", [])
        for button in deposit_buttons:
            if button.get("name") == "deposit_inventory":
                bounds = button.get("bounds")
                if bounds and int(bounds.get("width", 0)) > 0 and int(bounds.get("height", 0)) > 0:
                    return bounds
        return None
    except Exception as e:
        logging.error(f"[deposit_all_button_bounds] helpers/bank.py: {e}")
        return None

def nearest_banker() -> dict | None:
    try:
        player_data = ipc.get_player()
        if not player_data or not player_data.get("ok"):
            logging.error("[nearest_banker] helpers/bank.py: Failed to get player data from IPC")
            return None
        
        me = player_data.get("player") or {}
        mx, my, mp = int(me.get("worldX") or 0), int(me.get("worldY") or 0), int(me.get("plane") or 0)
        
        # Get all NPCs and filter for bankers
        all_npcs_resp = ipc.get_npcs("banker")
        if not all_npcs_resp or not all_npcs_resp.get("ok"):
            logging.error("[nearest_banker] helpers/bank.py: Failed to get NPCs from IPC")
            return None
        
        all_npcs = all_npcs_resp.get("npcs", [])
        
        best, best_d2 = None, 1e18
        for npc in all_npcs:
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
    except Exception as e:
        logging.error(f"[nearest_banker] helpers/bank.py: {e}")
        return None

def near_any_bank(destination_area: str) -> bool:
    """
    Check if there is a bank booth or banker within 20 tiles of the player.
    Uses find_object_in_area() for bank booths and npc_in_area() for bankers when destination_area is provided.
    
    Args:
        destination_area: Area name to check for bank facilities (e.g., "FALADOR_BANK", "CLOSEST_BANK")
                          If None, uses regular find_object and get_npcs_in_radius
                          If "CLOSEST_BANK", finds the closest bank area and checks there
    
    Returns:
        - True if bank booth or banker found within 20 tiles
        - False if no bank facilities found nearby
    """
    try:
        # Handle special case for CLOSEST_BANK
        if destination_area == "CLOSEST_BANK" or not destination_area:
            from helpers.navigation import closest_bank_key
            actual_bank_area = closest_bank_key()
            destination_area = actual_bank_area

        if in_area(destination_area):
            return True

        # Check for bank booths in the specific area
        from actions.objects import _find_closest_object_in_area
        bank_booth = _find_closest_object_in_area("bank booth", destination_area)
        if bank_booth:
            distance = bank_booth.get("distance", 999)
            if distance <= 20:
                return True

        # Check for bankers in the specific area
        from actions.npc import _find_closest_npc_in_area
        banker = _find_closest_npc_in_area("banker", destination_area)
        if banker:
            distance = banker.get("distance", 999)
            if distance <= 20:
                return True

        return False
        
    except Exception as e:
        logging.error(f"[near_any_bank] helpers/bank.py: {e}")
        return False

def _has_interactable_bank_in_area(bank_area: str, max_distance: int = 20) -> bool:
    """
    Check if there's an interactable bank (booth, chest, or banker) within the target area
    that is within interaction distance of the player, regardless of whether the player
    is currently in that area.
    
    Args:
        bank_area: Area name to check for bank facilities (e.g., "FALADOR_BANK", "CLOSEST_BANK")
        max_distance: Maximum distance in tiles from player to consider interactable (default: 20)
    
    Returns:
        True if there's an interactable bank in the target area within max_distance, False otherwise
    """
    try:
        # Handle special case for CLOSEST_BANK
        if bank_area == "CLOSEST_BANK" or not bank_area:
            from helpers.navigation import closest_bank_key
            actual_bank_area = closest_bank_key()
            bank_area = actual_bank_area
        
        # Get area bounds
        from helpers.navigation import get_nav_rect
        area_rect = get_nav_rect(bank_area)
        if not area_rect or len(area_rect) != 4:
            return False
        
        min_x, max_x, min_y, max_y = area_rect
        
        # Get player position for distance checking
        player_data = ipc.get_player()
        if not player_data or not player_data.get("ok"):
            return False
        
        player = player_data.get("player", {})
        player_x = player.get("worldX")
        player_y = player.get("worldY")
        player_plane = player.get("plane", 0)
        
        if not isinstance(player_x, int) or not isinstance(player_y, int):
            return False
        
        # Helper function to find bank action index
        def find_bank_action(actions):
            if not actions:
                return None
            for i, action in enumerate(actions):
                if action and ("bank" in action.lower() or "use" in action.lower()):
                    return i
            return None
        
        # Get all banks (booths, chests, bankers)
        booth_resp = ipc.get_objects("bank booth", ["GAME"])
        chest_resp = ipc.get_objects("bank chest", ["GAME"])
        banker_resp = ipc.get_npcs("banker")
        
        booths = booth_resp.get("objects", []) if booth_resp and booth_resp.get("ok") else []
        chests = chest_resp.get("objects", []) if chest_resp and chest_resp.get("ok") else []
        bankers = banker_resp.get("npcs", []) if banker_resp and banker_resp.get("ok") else []
        
        # Check all banks to see if any are in the target area and within interaction distance
        all_banks = []
        all_banks.extend(booths)
        all_banks.extend(chests)
        all_banks.extend(bankers)
        
        for bank in all_banks:
            # Get bank world coordinates
            world = bank.get("world", {})
            bank_x = world.get("x")
            bank_y = world.get("y")
            bank_plane = world.get("p", world.get("plane", 0))
            
            if not isinstance(bank_x, int) or not isinstance(bank_y, int):
                continue
            
            # Check if bank is in the target area
            if bank_x < min_x or bank_x > max_x or bank_y < min_y or bank_y > max_y:
                continue
            
            # Check if bank is on the same plane
            if int(bank_plane) != int(player_plane):
                continue
            
            # Check if bank has a valid bank action
            actions = bank.get("actions", [])
            if find_bank_action(actions) is None:
                continue
            
            # Check distance from player
            distance = bank.get("distance", float('inf'))
            if isinstance(distance, (int, float)) and distance <= max_distance:
                return True
        
        return False
        
    except Exception as e:
        logging.error(f"[_has_interactable_bank_in_area] helpers/bank.py: {e}")
        return False

def get_bank_tabs() -> list[dict]:
    """Get all bank organization tabs."""
    try:
        tabs_data = ipc.get_bank_tabs()
        if not tabs_data or not tabs_data.get("ok"):
            logging.error("[get_bank_tabs] helpers/bank.py: Failed to get bank tabs from IPC")
            return []
        return tabs_data.get("tabs", [])
    except Exception as e:
        logging.error(f"[get_bank_tabs] helpers/bank.py: {e}")
        return []

def get_bank_quantity_buttons() -> list[dict]:
    """Get all bank quantity buttons (1, 5, 10, X, All)."""
    try:
        quantity_data = ipc.get_bank_quantity_buttons()
        if not quantity_data or not quantity_data.get("ok"):
            logging.error("[get_bank_quantity_buttons] helpers/bank.py: Failed to get bank quantity buttons from IPC")
            return []
        return quantity_data.get("quantity_buttons", [])
    except Exception as e:
        logging.error(f"[get_bank_quantity_buttons] helpers/bank.py: {e}")
        return []

def get_bank_deposit_buttons() -> list[dict]:
    """Get bank deposit buttons (inventory, equipment)."""
    try:
        deposit_data = ipc.get_bank_deposit_buttons()
        if not deposit_data or not deposit_data.get("ok"):
            logging.error("[get_bank_deposit_buttons] helpers/bank.py: Failed to get bank deposit buttons from IPC")
            return []
        return deposit_data.get("deposit_buttons", [])
    except Exception as e:
        logging.error(f"[get_bank_deposit_buttons] helpers/bank.py: {e}")
        return []

def get_bank_search_widgets() -> list[dict]:
    """Get bank search interface widgets."""
    try:
        search_data = ipc.get_bank_search()
        if not search_data or not search_data.get("ok"):
            logging.error("[get_bank_search_widgets] helpers/bank.py: Failed to get bank search widgets from IPC")
            return []
        return search_data.get("search_widgets", [])
    except Exception as e:
        logging.error(f"[get_bank_search_widgets] helpers/bank.py: {e}")
        return []

def get_bank_quantity_mode() -> dict:
    """Get current bank withdraw mode and X value using RuneLite Varbits."""
    try:
        # Check if bank is open first
        bank_data = ipc.get_bank()
        if not bank_data or not bank_data.get("ok"):
            logging.error("[get_bank_quantity_mode] helpers/bank.py: Bank is not open")
            return {"ok": False, "mode": None, "x": None, "err": "Bank is not open"}
        
        xvalue_data = ipc.get_bank_xvalue()
        
        if not xvalue_data or not xvalue_data.get("ok"):
            error_msg = xvalue_data.get("err", "Unknown error") if xvalue_data else "No response"
            logging.error(f"[get_bank_quantity_mode] helpers/bank.py: Failed to get bank quantity mode from IPC: {error_msg}")
            return {"ok": False, "mode": None, "x": None, "err": f"IPC error: {error_msg}"}
        
        mode = xvalue_data.get("mode")
        x_value = xvalue_data.get("x")
        
        # Mode meanings:
        # 0 = "1" (withdraw 1)
        # 1 = "5" (withdraw 5) 
        # 2 = "10" (withdraw 10)
        # 3 = "X" (withdraw custom amount)
        # 4 = "All" (withdraw all)
        
        return {
            "ok": True,
            "mode": mode,
            "x": x_value,
            "mode_name": ["1", "5", "10", "X", "All"][mode] if mode is not None and 0 <= mode <= 4 else "Unknown"
        }
    except Exception as e:
        logging.error(f"[get_bank_quantity_mode] helpers/bank.py: {e}")
        return {"ok": False, "mode": None, "x": None, "err": str(e)}

def get_bank_items_widgets() -> list[dict]:
    """Get all bank item slot widgets."""
    try:
        items_data = ipc.get_bank_items()
        if not items_data or not items_data.get("ok"):
            logging.error("[get_bank_items_widgets] helpers/bank.py: Failed to get bank items widgets from IPC")
            return []
        return items_data.get("items", [])
    except Exception as e:
        logging.error(f"[get_bank_items_widgets] helpers/bank.py: {e}")
        return []

def find_bank_item_widget(item_name: str) -> dict | None:
    """Find a specific bank item widget by name."""
    try:
        items = get_bank_items_widgets()
        search_name = norm_name(item_name)
        
        for item in items:
            item_text = item.get("text", "")
            if norm_name(item_text) == search_name:
                return item
        return None
    except Exception as e:
        logging.error(f"[find_bank_item_widget] helpers/bank.py: {e}")
        return None

def get_deposit_equipment_button() -> dict | None:
    """Get the deposit equipment button widget."""
    try:
        deposit_buttons = get_bank_deposit_buttons()
        for button in deposit_buttons:
            if button.get("name") == "deposit_equipment":
                return button
        return None
    except Exception as e:
        logging.error(f"[get_deposit_equipment_button] helpers/bank.py: {e}")
        return None

# Bank button selection methods using widget IDs
def is_swap_selected() -> bool:
    """Check if SWAP mode is selected."""
    try:
        from .widgets import click_listener_on
        from constants import BANK_WIDGETS
        return click_listener_on(BANK_WIDGETS["SWAP"])
    except Exception as e:
        logging.error(f"[is_swap_selected] helpers/bank.py: {e}")
        return False

def is_insert_selected() -> bool:
    """Check if INSERT mode is selected."""
    try:
        from .widgets import click_listener_on
        from constants import BANK_WIDGETS
        return click_listener_on(BANK_WIDGETS["INSERT"])
    except Exception as e:
        logging.error(f"[is_insert_selected] helpers/bank.py: {e}")
        return False

def is_item_selected() -> bool:
    """Check if ITEM mode is selected."""
    try:
        from .widgets import click_listener_on
        from constants import BANK_WIDGETS
        return click_listener_on(BANK_WIDGETS["ITEM"])
    except Exception as e:
        logging.error(f"[is_item_selected] helpers/bank.py: {e}")
        return False

def is_note_selected() -> bool:
    """Check if NOTE mode is selected."""
    try:
        from .widgets import click_listener_on
        from constants import BANK_WIDGETS
        return click_listener_on(BANK_WIDGETS["NOTE"])
    except Exception as e:
        logging.error(f"[is_note_selected] helpers/bank.py: {e}")
        return False

def is_quantity1_selected() -> bool:
    """Check if QUANTITY 1 is selected."""
    try:
        mode_data = get_bank_quantity_mode()
        if not mode_data.get("ok"):
            logging.error("[is_quantity1_selected] helpers/bank.py: Failed to get bank quantity mode")
            return False
        return mode_data.get("mode") == 0
    except Exception as e:
        logging.error(f"[is_quantity1_selected] helpers/bank.py: {e}")
        return False

def is_quantity5_selected() -> bool:
    """Check if QUANTITY 5 is selected."""
    try:
        mode_data = get_bank_quantity_mode()
        if not mode_data.get("ok"):
            logging.error("[is_quantity5_selected] helpers/bank.py: Failed to get bank quantity mode")
            return False
        return mode_data.get("mode") == 1
    except Exception as e:
        logging.error(f"[is_quantity5_selected] helpers/bank.py: {e}")
        return False

def is_quantity10_selected() -> bool:
    """Check if QUANTITY 10 is selected."""
    try:
        mode_data = get_bank_quantity_mode()
        if not mode_data.get("ok"):
            logging.error("[is_quantity10_selected] helpers/bank.py: Failed to get bank quantity mode")
            return False
        return mode_data.get("mode") == 2
    except Exception as e:
        logging.error(f"[is_quantity10_selected] helpers/bank.py: {e}")
        return False

def is_quantityx_selected() -> bool:
    """Check if QUANTITY X is selected."""
    try:
        mode_data = get_bank_quantity_mode()
        if not mode_data.get("ok"):
            logging.error("[is_quantityx_selected] helpers/bank.py: Failed to get bank quantity mode")
            return False
        return mode_data.get("mode") == 3
    except Exception as e:
        logging.error(f"[is_quantityx_selected] helpers/bank.py: {e}")
        return False

def is_quantityall_selected() -> bool:
    """Check if QUANTITY ALL is selected."""
    try:
        mode_data = get_bank_quantity_mode()
        if not mode_data.get("ok"):
            logging.error("[is_quantityall_selected] helpers/bank.py: Failed to get bank quantity mode")
            return False
        return mode_data.get("mode") == 4
    except Exception as e:
        logging.error(f"[is_quantityall_selected] helpers/bank.py: {e}")
        return False

# Bank button selection methods
def select_swap() -> dict | None:
    """Select SWAP mode."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["SWAP"])
        if not widget_info:
            logging.error("[select_swap] helpers/bank.py: Failed to get widget info for SWAP button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_swap] helpers/bank.py: SWAP button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-swap",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Swap", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_swap] helpers/bank.py: {e}")
        return None

def select_insert() -> dict | None:
    """Select INSERT mode."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["INSERT"])
        if not widget_info:
            logging.error("[select_insert] helpers/bank.py: Failed to get widget info for INSERT button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_insert] helpers/bank.py: INSERT button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-insert",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Insert", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_insert] helpers/bank.py: {e}")
        return None

def select_item() -> dict | None:
    """Select ITEM mode."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["ITEM"])
        if not widget_info:
            logging.error("[select_item] helpers/bank.py: Failed to get widget info for ITEM button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_item] helpers/bank.py: ITEM button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-item",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Item", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_item] helpers/bank.py: {e}")
        return None

def select_note() -> dict | None:
    """Select NOTE mode."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["NOTE"])
        if not widget_info:
            logging.error("[select_note] helpers/bank.py: Failed to get widget info for NOTE button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_note] helpers/bank.py: NOTE button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-note",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Note", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_note] helpers/bank.py: {e}")
        return None

def select_quantity1() -> dict | None:
    """Select QUANTITY 1."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITY1"])
        if not widget_info:
            logging.error("[select_quantity1] helpers/bank.py: Failed to get widget info for QUANTITY1 button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantity1] helpers/bank.py: QUANTITY1 button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-quantity1",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Quantity 1", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantity1] helpers/bank.py: {e}")
        return None

def select_quantity5() -> dict | None:
    """Select QUANTITY 5."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITY5"])
        if not widget_info:
            logging.error("[select_quantity5] helpers/bank.py: Failed to get widget info for QUANTITY5 button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantity5] helpers/bank.py: QUANTITY5 button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-quantity5",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Quantity 5", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantity5] helpers/bank.py: {e}")
        return None

def select_quantity10() -> dict | None:
    """Select QUANTITY 10."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITY10"])
        if not widget_info:
            logging.error("[select_quantity10] helpers/bank.py: Failed to get widget info for QUANTITY10 button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantity10] helpers/bank.py: QUANTITY10 button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-quantity10",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Quantity 10", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantity10] helpers/bank.py: {e}")
        return None

def select_quantityx() -> dict | None:
    """Select QUANTITY X."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITYX"])
        if not widget_info:
            logging.error("[select_quantityx] helpers/bank.py: Failed to get widget info for QUANTITYX button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantityx] helpers/bank.py: QUANTITYX button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-quantityx",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Quantity X", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantityx] helpers/bank.py: {e}")
        return None

def select_quantityx_custom() -> dict | None:
    """Select QUANTITY X using context-select for 'Custom Quantity' option."""
    try:
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITYX"])
        if not widget_info:
            logging.error("[select_quantityx_context] helpers/bank.py: Failed to get widget info for QUANTITYX button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantityx_context] helpers/bank.py: QUANTITYX button has invalid bounds")
            return None
        
        # Calculate randomized point for context menu
        center_x, center_y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                                          bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        
        step = {
            "action": "bank-select-quantityx-context",
            "option": "Set custom quantity",
            "click": {
                "type": "context-select",
                "x": center_x,
                "y": center_y,
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
            },
            "target": {"domain": "bank-widget", "name": "", "bounds": bounds},
            "anchor": {"x": center_x, "y": center_y},
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantityx_context] helpers/bank.py: {e}")
        return None

def select_quantityall() -> dict | None:
    """Select QUANTITY ALL."""
    try:
        
        widget_info = get_widget_info(BANK_WIDGETS["QUANTITYALL"])
        if not widget_info:
            logging.error("[select_quantityall] helpers/bank.py: Failed to get widget info for QUANTITYALL button")
            return None
        
        widget_data = widget_info.get("data", {})
        bounds = widget_data.get("bounds")
        
        if not bounds or int(bounds.get("width", 0)) <= 0 or int(bounds.get("height", 0)) <= 0:
            logging.error("[select_quantityall] helpers/bank.py: QUANTITYALL button has invalid bounds")
            return None
        
        step = {
            "action": "bank-select-quantityall",
            "click": {"type": "rect-center"},
            "target": {"domain": "bank-widget", "name": "Quantity All", "bounds": bounds},
            "postconditions": [],
        }
        return dispatch(step)
    except Exception as e:
        logging.error(f"[select_quantityall] helpers/bank.py: {e}")
        return None


def randomize_item_order(items: list, reverse_probability_range: tuple[float, float] = (0.2834, 0.4167)) -> list:
    """
    Randomly reverse the order of items with weighted probability.
    
    Creates a copy of the items list and randomly decides whether to reverse it,
    using a random-looking percentage within the specified range for variance.
    
    Args:
        items: List of items to potentially reorder
        reverse_probability_range: Tuple of (min, max) probability to reverse order (default: ~28-42%)
        
    Returns:
        List of items (potentially reversed)
    """
    import random
    
    if not items or len(items) <= 1:
        return list(items)
    
    # Create a copy
    items_copy = list(items)
    
    # Randomly decide whether to reverse the order
    reverse_probability = random.uniform(reverse_probability_range[0], reverse_probability_range[1])
    
    if random.random() < reverse_probability:
        items_copy.reverse()
    
    return items_copy
