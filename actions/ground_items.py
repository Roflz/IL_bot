from __future__ import annotations
from typing import Optional

from helpers.runtime_utils import ipc, dispatch


def loot(item_name: str, radius: int = 10, *, max_attempts: int = 3, verify_wait_ms: int = 3500) -> Optional[dict]:
    """
    Search for a ground item around the player and pick it up.
    
    Args:
        item_name: Name of the item to look for (partial match allowed)
        radius: Search radius around player (default: 10 tiles)
        max_attempts: Maximum number of attempts. Each attempt re-queries IPC for a fresh item snapshot.
        verify_wait_ms: How long to wait for inventory to reflect the pickup before retrying.
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    from actions.timing import wait_until
    from helpers.utils import norm_name

    want = (item_name or "").strip()
    if not want:
        return None

    try:
        radius_i = int(radius)
    except Exception:
        radius_i = 10
    radius_i = max(1, radius_i)

    try:
        attempts = int(max_attempts)
    except Exception:
        attempts = 3
    attempts = max(1, attempts)

    try:
        verify_ms = int(verify_wait_ms)
    except Exception:
        verify_ms = 3500
    verify_ms = max(250, verify_ms)

    def _inv_count(needle: str) -> int:
        """Count total quantity of inventory items whose name matches (exact or contains) `needle`."""
        inv = ipc.get_inventory() or {}
        if not inv.get("ok"):
            return 0
        slots = inv.get("slots", []) or []
        n = norm_name(needle)
        if not n:
            return 0

        # Prefer exact-match counting if possible, fallback to contains for partial-name workflows.
        exact_total = 0
        contains_total = 0
        for s in slots:
            nm = norm_name(s.get("itemName"))
            if not nm:
                continue
            qty = int(s.get("quantity") or 0)
            if nm == n:
                exact_total += qty
            if n in nm:
                contains_total += qty
        return exact_total if exact_total > 0 else contains_total

    before_qty = _inv_count(want)

    for attempt in range(attempts):
        # Re-query each attempt so we always click using fresh canvas/clickbox coordinates.
        ground_items_resp = ipc.get_ground_items(want, radius_i) or {}
        if not ground_items_resp.get("ok"):
            print(f"[LOOT] Failed to search for ground items: {want}")
            continue

        ground_items = ground_items_resp.get("items", []) or []
        if not ground_items:
            return None

        # Find closest by server-provided distance.
        closest_item = None
        closest_distance = float("inf")
        for it in ground_items:
            d = it.get("distance", float("inf"))
            if d < closest_distance:
                closest_distance = d
                closest_item = it

        if not closest_item:
            return None

        print(f"[LOOT] Found ground item: {closest_item.get('name')} at distance {closest_distance} (attempt {attempt+1}/{attempts})")

        click_res = _pickup_ground_item_once(closest_item)
        if not click_res:
            continue

        # Wait for the pickup to actually register in inventory; otherwise retry with a fresh item snapshot.
        ok = wait_until(lambda: _inv_count(want) > before_qty, max_wait_ms=verify_ms)
        if ok:
            print(f"[LOOT] Successfully picked up {closest_item.get('name')}")
            return click_res

    return None


def _pickup_ground_item_once(item: dict) -> Optional[dict]:
    """
    Pick up a specific ground item.
    
    Args:
        item: Ground item data dictionary
        
    Returns:
        UI dispatch result if successful, None if failed
    """
    from helpers.rects import unwrap_rect
    from helpers.utils import rect_beta_xy, clean_rs

    rect = unwrap_rect(item.get("clickbox"))

    # Determine click coordinates
    if rect:
        cx, cy = rect_beta_xy(
            (rect.get("x", 0), rect.get("x", 0) + rect.get("width", 0), rect.get("y", 0), rect.get("y", 0) + rect.get("height", 0)),
            alpha=2.0,
            beta=2.0,
        )
        anchor = {"bounds": rect}
        point = {"x": cx, "y": cy}
        print(f"[LOOT] Using rect coordinates: ({cx}, {cy})")
    elif isinstance(item.get("canvas", {}).get("x"), (int, float)) and isinstance(item.get("canvas", {}).get("y"), (int, float)):
        cx, cy = int(item.get("canvas", {}).get("x")), int(item.get("canvas", {}).get("y"))
        anchor = {}
        point = {"x": cx, "y": cy}
        print(f"[LOOT] Using canvas coordinates: ({cx}, {cy})")
    else:
        print(f"[LOOT] No valid coordinates found for ground item")
        return None

    # Ground items always use "Take" action - use context menu with specific item name
    item_name = item.get("name", "Unknown")
    world_coords = {
        "x": item.get("world", {}).get("x"),
        "y": item.get("world", {}).get("y"),
        "p": item.get("world", {}).get("p", 0),
    }
    print(f"[LOOT] Using context menu with 'Take {item_name}' for ground item pickup")

    # Create the pickup step using context menu with "Take [ItemName]" action
    step = {
        "action": "ground-item-pickup-context",
        "option": "Take",
        "click": {
            "type": "context-select",
            "target": f"{item_name}",
            "x": cx,
            "y": cy,
            "row_height": 16,
            "start_dy": 18,
            "open_delay_ms": 120,
        },
        "target": {"domain": "ground-item", "name": item_name, "world": world_coords, **anchor} if rect else {"domain": "ground-item", "name": item_name, "world": world_coords},
        "anchor": point,
    }

    result = dispatch(step)
    if not result:
        return None

    # Check if the correct interaction was performed (fast fail if wrong menu row).
    from helpers.ipc import get_last_interaction

    last_interaction = get_last_interaction()
    expected_action = "Take"
    expected_target = item_name

    if (
        last_interaction
        and last_interaction.get("action") == expected_action
        and clean_rs(last_interaction.get("target", "")).lower() == expected_target.lower()
    ):
        print(f"[CLICK] {expected_target} - interaction verified")
        return result

    print(f"[CLICK] {expected_target} - incorrect interaction")
    return None
