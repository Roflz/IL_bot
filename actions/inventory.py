from .tab import open_tab
from helpers.inventory import inv_has, inv_has_any, inv_count, first_inv_slot
from helpers.inventory import has_only_items as _has_only_items
from helpers.runtime_utils import ipc, dispatch
from helpers.tab import is_inventory_tab_open
from helpers.utils import sleep_exponential, rect_beta_xy, clean_rs, random_number, normal_number

from typing import Optional, List

def has_item(name: str, min_qty: int = 1) -> bool:
    """
    True if inventory contains item `name` with quantity >= min_qty.
    """
    return inv_has(name, min_qty)

def has_items(items: dict, noted: bool = False) -> bool:
    """
    Check whether the inventory contains all specified items in the required quantities.

    Args:
        items (dict or set or list or tuple): Mapping of item name -> required quantity,
                                              or iterable of item names for ANY quantity.
        noted (bool): If True, check for noted items; otherwise check for unnoted items.

    Returns:
        bool: True if all items are present in the required quantities, False otherwise.
    """
    if not items:
        print("[INV] inventory_has_items: Invalid items provided")
        return False

    # Allow set/list/tuple as "require ANY amount of each item"
    if isinstance(items, (set, list, tuple)):
        items = {name: "any" for name in items}

    if not isinstance(items, dict):
        print("[INV] inventory_has_items: Invalid items type")
        return False

    for item, qty in items.items():
        # allow "any" or None as "at least 1 present"
        if isinstance(qty, str) and qty.lower() == "any":
            needed_any = True
        elif qty is None:
            needed_any = True
        else:
            needed_any = False

        if not isinstance(item, str):
            print(f"[INV] Skipping invalid entry: {item} -> {qty}")
            continue

        if noted:
            count = inv_count(item)
        else:
            count = count_unnoted_item(item)

        if needed_any:
            if count <= 0:
                print(f"[INV] Missing {item}: need ANY, have 0")
                return False
        else:
            if not isinstance(qty, int) or qty <= 0:
                print(f"[INV] Skipping invalid qty entry: {item} -> {qty}")
                continue
            if count < qty:
                print(f"[INV] Missing {item}: need {qty}, have {count}")
                return False

    return True

def has_any_items(item_names: List[str], min_qty: int = 1) -> bool:
    """
    Check if inventory contains ANY item from the list with quantity >= min_qty.
    
    Args:
        item_names: List of item names to check for
        min_qty: Minimum quantity required for the item (default: 1)
        
    Returns:
        True if ANY item is found with sufficient quantity, False otherwise
    """
    if not item_names:
        return False  # Empty list means no items to find
    
    # Check each item individually - return True as soon as we find one
    for item_name in item_names:
        if has_item(item_name, min_qty):
            return True
    
    return False


def has_only_items(allowed_items: List[str]) -> bool:
    """
    True if the inventory contains only the allowed items (and empty slots).
    """
    return _has_only_items(allowed_items)


def item_has_action(item_name: str, action: str) -> bool:
    """
    True if the inventory item exists and its slot exposes the given action.
    (Case-insensitive exact match against slot['actions'] entries.)
    """
    item = first_inv_slot(item_name)
    if not item:
        return False
    want = (action or "").strip().lower()
    if not want:
        return False
    for a in (item.get("actions") or []):
        if not a:
            continue
        if str(a).strip().lower() == want:
            return True
    return False


def has_noted_item(name: str) -> bool:
    """
    True if inventory contains noted version of item `name`.
    """
    resp = ipc.get_inventory()
    
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    target_name = (name or "").strip().lower()
    
    for slot in slots:
        item_name = (slot.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Check for noted items
        if target_name in item_name and slot.get("noted"):
            return True
    
    return False

def count_unnoted_item(name: str) -> int:
    """
    Count the total quantity of unnoted version of item `name`.
    This counts the normal item, not the noted version.
    """
    resp = ipc.get_inventory()
    
    if not resp or not resp.get("ok"):
        return 0
    
    slots = resp.get("slots", [])
    target_name = (name or "").strip().lower()
    total_count = 0
    
    for slot in slots:
        item_name = (slot.get("itemName") or "").strip().lower()
        if not item_name:
            continue
            
        # Exact match and not noted
        if item_name == target_name and not slot.get("noted"):
            quantity = int(slot.get("quantity") or 0)
            total_count += quantity
    
    return total_count


def has_unnoted_item(name: str, min_qty: int = 1) -> bool:
    """
    True if inventory contains at least min_qty unnoted version of item `name`.
    This is the normal item, not the noted version.
    """
    return count_unnoted_item(name) >= min_qty



def is_empty(excepted_items: list[str] = None) -> bool:
    """
    True if the player's inventory is empty.
    """
    return not inv_has_any(excepted_items=excepted_items)


def get_empty_slots_count() -> int:
    """
    Returns the number of empty inventory slots.
    
    Returns:
        Number of empty slots (0-28, where 28 is completely empty)
    """
    try:
        # Use IPC command to get inventory data
        resp = ipc.get_inventory()
        
        if not resp or not resp.get("ok"):
            print(f"[INVENTORY] Failed to get inventory data: {resp.get('err', 'Unknown error')}")
            return 28  # Assume all slots empty if can't get data
        
        slots = resp.get("slots", [])
        
        # Count empty slots
        empty_count = 0
        for slot in slots:
            # Check if slot is empty
            item_name = slot.get("itemName", "").strip()
            quantity = int(slot.get("quantity", 0))
            
            # Slot is empty if no item name or quantity is 0
            if not item_name or quantity <= 0:
                empty_count += 1
        
        return empty_count
        
    except Exception as e:
        print(f"[INVENTORY] Error counting empty slots: {e}")
        # Fallback: assume all slots are empty if we can't read the data
        return 28

def use_item_on_item(item1_name: str, item2_name: str, max_retries: int = 3) -> Optional[dict]:
    """
    Use an item in the inventory on another item in the inventory.
    
    Args:
        item1_name: Name of the item to use (the "using" item)
        item2_name: Name of the item to use on (the "target" item)
        max_retries: Maximum number of retry attempts if interaction fails
    
    Returns:
        UI dispatch result or None if failed
    """
    # Check if both items exist in inventory
    if not has_item(item1_name):
        return None
    if not has_item(item2_name):
        return None

    result1 = interact(item1_name, "Use")

    if not result1:
        return None

    sleep_exponential(0.3, 0.8)

    result2 = interact(item2_name, "Use", exact_match=False)

    return result2

def use_item_on_object(item_name: str, object_name: str, max_retries: int = 3) -> Optional[dict]:
    """
    Use an item in the inventory on a game object.
    
    Args:
        item_name: Name of the item to use from inventory
        object_name: Name of the game object to use the item on
        max_retries: Maximum number of retry attempts if interaction fails
    
    Returns:
        UI dispatch result or None if failed
    """
    # Check if item exists in inventory
    if not has_item(item_name):
        return None

    result1 = interact(item_name, "Use")

    if not result1:
        return None

    sleep_exponential(0.3, 0.8)

    from actions import objects
    result2 = objects.click_object_closest_by_distance_simple(object_name, f"Use")
    
    return result2


def inventory_has_amount(item_name: str, expected_amount: int) -> bool:
    """
    Helper function for use with wait_until to check if inventory contains 
    the expected amount of an item.
    
    Args:
        item_name: Name of the item to check
        expected_amount: Expected quantity in inventory
    
    Returns:
        True if inventory contains at least the expected amount
    """
    current_count = inv_count(item_name)
    return current_count >= expected_amount


def interact(item_name: str, menu_option: str, exact_match: bool = False) -> Optional[dict]:
    """
    Context-click an inventory item and select a specific menu option.
    
    Args:
        item_name: Name of the item to interact with
        menu_option: Menu option to select (e.g., "Use", "Drop", "Examine")
        exact_match: If True, only matches exact target and action names; if False, uses substring matching
    
    Returns:
        UI dispatch result or None if failed
    """
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Opening inventory tab before interacting with {item_name}")
        open_tab("INVENTORY")
        # Wait a moment for the tab to open
        sleep_exponential(0.5, 2.0, 1.5)
    
    # Ensure we're still on inventory tab before proceeding
    if not is_inventory_tab_open():
        print(f"[INVENTORY] Failed to open inventory tab, aborting interaction")
        return None
    
    # Inner attempt loop with fresh coordinate recalculation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Fresh coordinate recalculation
        item = first_inv_slot(item_name)
        if not item:
            continue
        
        bounds = item.get("bounds")
        if not bounds:
            continue
        
        # Calculate center coordinates
        x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                             bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        
        # Use context-select so we can reliably choose non-default inventory actions like "Empty".
        step = {
            "action": "inventory-interact",
            "option": menu_option,
            "click": {
                "type": "context-select",
                "x": int(x),
                "y": int(y),
                "target": item_name.lower(),
                "open_delay_ms": 120,
                # ActionExecutor uses this to decide exact vs partial matching on menu text.
                "exact_match": bool(exact_match),
            },
            "target": {"domain": "inventory", "name": item_name, "bounds": bounds},
        }
        
        result = dispatch(step)
        
        if result:
            # Check if the correct interaction was performed
            from helpers.ipc import get_last_interaction
            last_interaction = get_last_interaction()
            
            expected_target = item_name
            expected_action = menu_option
            
            # Use exact match or contains based on exact_match parameter
            target_match = (clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()) if exact_match else (expected_target.lower() in clean_rs(last_interaction.get("target", "")).lower())
            action_match = (clean_rs(last_interaction.get("action", "")).lower() == expected_action.lower()) if exact_match else (expected_action.lower() in clean_rs(last_interaction.get("action", "")).lower())
            
            if last_interaction and target_match and action_match:
                print(f"[CLICK] {expected_target} ({menu_option}) - interaction verified")
                return result
            else:
                print(f"[CLICK] {expected_target} ({menu_option}) - incorrect interaction, retrying...")
                continue
    
    return None


def shift_click_item(item_name: str, max_retries: int = 3) -> Optional[dict]:
    """
    Shift-click an inventory item.

    Notes:
    - What shift-click does depends on your in-game shift-click configuration (e.g. Drop).
    - This function holds SHIFT down, performs a left click on the item, then releases SHIFT.
    """
    if not item_name or not str(item_name).strip():
        return None

    if not is_inventory_tab_open():
        print(f"[INVENTORY] Opening inventory tab before shift-clicking {item_name}")
        open_tab("INVENTORY")
        sleep_exponential(0.1, 0.3, 1.5)

    if not is_inventory_tab_open():
        print("[INVENTORY] Failed to open inventory tab, aborting shift-click")
        return None

    for attempt in range(max_retries):
        item = first_inv_slot(item_name)
        if not item:
            continue

        bounds = item.get("bounds")
        if not bounds:
            continue

        x, y = rect_beta_xy(
            (
                bounds.get("x", 0),
                bounds.get("x", 0) + bounds.get("width", 0),
                bounds.get("y", 0),
                bounds.get("y", 0) + bounds.get("height", 0),
            ),
            alpha=2.0,
            beta=2.0,
        )

        step = {
            "action": "inventory-shift-click",
            "click": {"type": "point", "x": int(x), "y": int(y)},
            "target": {"domain": "inventory", "name": item_name, "mod": "SHIFT"},
        }

        try:
            ipc.focus()
            ipc.key_press("SHIFT")
            result = dispatch(step)
        finally:
            try:
                ipc.key_release("SHIFT")
            except Exception:
                pass

        if result:
            return result

        sleep_exponential(0.05, 0.15, 1.5)

    return None


def drop_all(items: str | List[str], max_clicks: int = 250) -> int:
    """
    Drop all occurrences of the specified inventory items using shift-click.

    - Holds SHIFT down for most of the operation.
    - Accepts a single string or a list of strings.
    - Uses a mostly horizontal or vertical sweep pattern across the 4x7 inventory grid,
      with occasional out-of-order clicks and rare SHIFT re-toggles.

    Returns:
        Number of successful click attempts performed (best-effort proxy for drops).
    """
    # =====================
    # Config (tune these)
    # =====================
    # **ORIENTATION_WEIGHTS**: Controls the *single* orientation pick for this whole run.
    # Implementation: we draw one uniform random \(u \in [0,1)\) and compare against:
    #   cut = horizontal / (horizontal + vertical)
    # - "horizontal" = row-major sweep (left→right across a row, then next row)
    # - "vertical"   = column-major sweep (top→bottom down a column, then next column)
    ORIENTATION_WEIGHTS = {"horizontal": 0.70, "vertical": 0.30}

    # **OUT_OF_ORDER_PROB_RANGE**: Per-call range for the out-of-order probability.
    # Implementation:
    # - At the START of drop_all(), we sample:
    #     OUT_OF_ORDER_PROB ~ Uniform(OUT_OF_ORDER_PROB_RANGE[0], OUT_OF_ORDER_PROB_RANGE[1])
    # - For each drop, we draw uniform \(u \in [0,1)\); if u < OUT_OF_ORDER_PROB, we pick a random candidate slot.
    OUT_OF_ORDER_PROB_RANGE = (0.01, 0.03)

    # NOTE: We intentionally do NOT re-toggle SHIFT. We hold SHIFT for the entire run.

    # **CLICK_SLEEP**: (min,max) seconds between consecutive drop clicks.
    # Implementation: uses sleep_exponential(min,max, CLICK_SLEEP_LAMBDA).
    # If too low, RuneLite/IPC may not keep up and drops can be missed.
    CLICK_SLEEP = (0.05, 0.8)
    # **CLICK_SLEEP_LAMBDA**: Shape for sleep_exponential between clicks.
    CLICK_SLEEP_LAMBDA = 2.0

    # **JITTER_ALPHA_RANGE / JITTER_BETA_RANGE**: Ranges for the alpha/beta parameters fed into `rect_beta_xy(...)`.
    # `rect_beta_xy` picks the click point inside the slot using a Beta distribution:
    # - Higher alpha/beta => clicks cluster nearer the center
    # - Lower alpha/beta  => clicks spread more across the slot
    #
    # We *choose* alpha/beta each click using normal_number(min,max, JITTER_PARAM_CENTER_BIAS),
    # which is then clamped to [min,max]. So you control both the allowed range and how centered
    # the chosen parameter tends to be.
    JITTER_ALPHA_RANGE = (1.6, 2.6)
    JITTER_BETA_RANGE = (1.6, 2.6)
    # **JITTER_PARAM_CENTER_BIAS**: Controls how strongly alpha/beta selection is biased toward the middle of the range.
    # 0.0 ~ more uniform-ish; 1.0 ~ strongly centered.
    JITTER_PARAM_CENTER_BIAS = 0.7

    # Inventory geometry: 4 columns x 7 rows, slot index assumed row-major (0..27)
    COLS = 4
    ROWS = 7

    # =====================
    # Normalize inputs
    # =====================
    if isinstance(items, list):
        names = [str(x).strip() for x in items if x and str(x).strip()]
    else:
        names = [str(items).strip()] if items and str(items).strip() else []

    # Normalize for matching: use clean_rs + lowercase
    want = [clean_rs(n).lower() for n in names if n]
    if not want:
        return 0

    # =====================
    # Ensure inventory open
    # =====================
    if not is_inventory_tab_open():
        open_tab("INVENTORY")
        sleep_exponential(0.10, 0.25, 1.5)
    if not is_inventory_tab_open():
        return 0

    def rand01() -> float:
        return float(random_number(0.0, 1.0, output_type="float"))

    def rand_int(lo: int, hi: int) -> int:
        return int(random_number(float(lo), float(hi), output_type="int"))

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    # Pick an overall sweep orientation (biased) using our math utilities.
    # (One draw for the entire run.)
    wh = float(ORIENTATION_WEIGHTS["horizontal"])
    wv = float(ORIENTATION_WEIGHTS["vertical"])
    cut = wh / (wh + wv) if (wh + wv) > 0 else 0.5
    orientation = "horizontal" if rand01() < cut else "vertical"

    # Sample per-run probabilities from the configured ranges
    OUT_OF_ORDER_PROB = clamp01(float(random_number(OUT_OF_ORDER_PROB_RANGE[0], OUT_OF_ORDER_PROB_RANGE[1], output_type="float")))
    # No shift-retoggle probability (SHIFT is held throughout).

    def slot_rank(idx: int) -> int:
        # idx -> (row,col)
        row = idx // COLS
        col = idx % COLS
        if orientation == "vertical":
            return col * ROWS + row  # column-major
        return row * COLS + col      # row-major

    # =====================
    # New behavior knobs
    # =====================
    # **DOUBLE_CLICK_PROB_RANGE**: per-call range for probability to click the *same* slot twice.
    DOUBLE_CLICK_PROB_RANGE = (0.01, 0.04)
    # **MISS_CLICK_PROB_RANGE**: per-call range for probability to "accidentally" skip a slot that should be clicked.
    MISS_CLICK_PROB_RANGE = (0.01, 0.04)
    # **DOUBLE_CLICK_SLEEP**: short pause between the two clicks when double-clicking.
    DOUBLE_CLICK_SLEEP = (0.02, 0.10)
    DOUBLE_CLICK_SLEEP_LAMBDA = 2.0

    DOUBLE_CLICK_PROB = clamp01(
        float(random_number(DOUBLE_CLICK_PROB_RANGE[0], DOUBLE_CLICK_PROB_RANGE[1], output_type="float"))
    )
    MISS_CLICK_PROB = clamp01(
        float(random_number(MISS_CLICK_PROB_RANGE[0], MISS_CLICK_PROB_RANGE[1], output_type="float"))
    )

    # Build the one-time sweep order (each slot visited at most once in the main pass)
    ordered_slots = sorted(list(range(COLS * ROWS)), key=slot_rank)
    # When we do an "out of order" click, constrain it to a small neighborhood
    # around the intended slot in sweep-order (prevents huge jumps across inventory).
    OUT_OF_ORDER_NEIGHBOR_WINDOW_RANGE = (1, 3)  # +/- N positions in `ordered_slots`

    def _slot_matches(slot: dict) -> bool:
        try:
            qty = int(slot.get("quantity", 0))
        except Exception:
            qty = 0
        if qty <= 0:
            return False
        item_name = clean_rs(slot.get("itemName", "")).lower()
        if not item_name:
            return False
        return any((w == item_name) or (w in item_name) for w in want)

    def _click_slot_once(slot: dict) -> bool:
        nonlocal dropped_clicks, shift_down

        bounds = slot.get("bounds") or {}
        if not isinstance(bounds, dict) or not bounds:
            return False

        alpha = float(
            normal_number(
                JITTER_ALPHA_RANGE[0],
                JITTER_ALPHA_RANGE[1],
                center_bias=JITTER_PARAM_CENTER_BIAS,
                output_type="float",
            )
        )
        beta = float(
            normal_number(
                JITTER_BETA_RANGE[0],
                JITTER_BETA_RANGE[1],
                center_bias=JITTER_PARAM_CENTER_BIAS,
                output_type="float",
            )
        )
        x, y = rect_beta_xy(
            (
                bounds.get("x", 0),
                bounds.get("x", 0) + bounds.get("width", 0),
                bounds.get("y", 0),
                bounds.get("y", 0) + bounds.get("height", 0),
            ),
            alpha=alpha,
            beta=beta,
        )

        # Best-effort: ensure SHIFT is held. We can't read key state, so we periodically
        # re-assert keyPress as a safety measure (idempotent in practice).
        if not shift_down:
            ipc.key_press("SHIFT")
            shift_down = True
            sleep_exponential(0.2, 0.5, 1)
        elif (dropped_clicks % 12) == 0:
            # every ~12 successful clicks, re-press SHIFT to guard against missed keyDown
            try:
                ipc.key_press("SHIFT")
            except Exception:
                pass

        step = {
            "action": "inventory-drop-all-shift-click",
            "click": {"type": "point", "x": int(x), "y": int(y)},
            "target": {"domain": "inventory", "name": slot.get("itemName", ""), "mod": "SHIFT"},
        }

        result = dispatch(step)
        if result:
            dropped_clicks += 1
            return True
        return False

    dropped_clicks = 0
    clicked_slot_idxs: set[int] = set()      # slots we actually clicked at least once
    skipped_slot_idxs: set[int] = set()      # slots we intentionally "missed" in main pass

    # Hold SHIFT for most of the operation
    ipc.focus()
    ipc.key_press("SHIFT")
    shift_down = True

    try:
        # ---------------------
        # Main sweep pass
        # ---------------------
        resp = ipc.get_inventory() or {}
        if not resp.get("ok"):
            return 0
        slots = resp.get("slots", []) or []

        # Sample a per-run neighbor window size (how "close" out-of-order clicks stay)
        neighbor_window = max(
            0,
            int(
                random_number(
                    float(OUT_OF_ORDER_NEIGHBOR_WINDOW_RANGE[0]),
                    float(OUT_OF_ORDER_NEIGHBOR_WINDOW_RANGE[1]),
                    output_type="int",
                )
            ),
        )

        for pos, idx in enumerate(ordered_slots):
            if dropped_clicks >= int(max_clicks):
                break
            if idx >= len(slots):
                continue
            if idx in clicked_slot_idxs:
                continue

            slot = slots[idx]
            if not _slot_matches(slot):
                continue

            # "Out of order" but LOCAL: pick a nearby slot in the sweep order.
            # This avoids jumping to the far end of the inventory.
            if neighbor_window > 0 and rand01() < OUT_OF_ORDER_PROB:
                lo = max(0, pos - neighbor_window)
                hi = min(len(ordered_slots) - 1, pos + neighbor_window)
                neighborhood = []
                for p in range(lo, hi + 1):
                    j = ordered_slots[p]
                    if j == idx:
                        continue
                    if j in clicked_slot_idxs:
                        continue
                    if j >= len(slots):
                        continue
                    s2 = slots[j]
                    if _slot_matches(s2):
                        neighborhood.append(j)
                if neighborhood:
                    # Choose a nearby candidate index
                    idx = neighborhood[rand_int(0, len(neighborhood) - 1)]
                    slot = slots[idx]

            # Chance to "miss" the click (skip this slot for now)
            if rand01() < MISS_CLICK_PROB:
                skipped_slot_idxs.add(idx)
                continue

            if _click_slot_once(slot):
                clicked_slot_idxs.add(idx)

                # Chance to double-click this same slot (only allowed extra click)
                if dropped_clicks < int(max_clicks) and rand01() < DOUBLE_CLICK_PROB:
                    sleep_exponential(DOUBLE_CLICK_SLEEP[0], DOUBLE_CLICK_SLEEP[1], DOUBLE_CLICK_SLEEP_LAMBDA)
                    _click_slot_once(slot)

            sleep_exponential(CLICK_SLEEP[0], CLICK_SLEEP[1], CLICK_SLEEP_LAMBDA)

        # ---------------------
        # Cleanup pass (click missed slots once)
        # ---------------------
        if dropped_clicks < int(max_clicks):
            resp2 = ipc.get_inventory() or {}
            if resp2.get("ok"):
                slots2 = resp2.get("slots", []) or []

                # Only click slots we intentionally skipped AND still contain a wanted item.
                for idx in sorted(skipped_slot_idxs, key=slot_rank):
                    if dropped_clicks >= int(max_clicks):
                        break
                    if idx in clicked_slot_idxs:
                        continue
                    if idx >= len(slots2):
                        continue
                    slot = slots2[idx]
                    if not _slot_matches(slot):
                        continue

                    if _click_slot_once(slot):
                        clicked_slot_idxs.add(idx)
                        # Allow double-click here too (still first time we click this slot)
                        if dropped_clicks < int(max_clicks) and rand01() < DOUBLE_CLICK_PROB:
                            sleep_exponential(DOUBLE_CLICK_SLEEP[0], DOUBLE_CLICK_SLEEP[1], DOUBLE_CLICK_SLEEP_LAMBDA)
                            _click_slot_once(slot)

                    sleep_exponential(CLICK_SLEEP[0], CLICK_SLEEP[1], CLICK_SLEEP_LAMBDA)

    finally:
        if shift_down:
            try:
                ipc.key_release("SHIFT")
            except Exception:
                pass

    return dropped_clicks


def is_full() -> bool:
    """
    Check if inventory is full (no empty slots).
    
    Returns:
        True if inventory is full (0 empty slots), False otherwise
    """
    empty_slots = get_empty_slots_count()
    return empty_slots == 0
