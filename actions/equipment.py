                                                                   # equipment.py (actions)

from __future__ import annotations
from typing import Optional, Union

from helpers.runtime_utils import ipc, dispatch
from helpers.utils import clean_rs, rect_beta_xy
from helpers.widgets import widget_exists


def interact(item_name: str, menu_option: str) -> Optional[dict]:
    """
    Context-click an equipment inventory item and select a specific menu option.
    This interacts with unequipped items in the equipment interface, not equipped items.
    
    Args:
        item_name: Name of the equipment inventory item to interact with
        menu_option: Menu option to select (e.g., "Wear", "Examine", "Drop")
    
    Returns:
        UI dispatch result or None if failed
    """
    # Check if equipment interface is open
    if not equipment_interface_open():
        return None
    
    # Find the item in equipment inventory
    item = find_equipment_inventory_item(item_name)
    if not item:
        return None
    
    # Get item bounds
    bounds = item.get('bounds')
    if not bounds:
        return None
    
    # Calculate center coordinates
    x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                         bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
    
    # Context-click the item
    step = {
        "action": "equipment-interact",
        "click": {"type": "point", "x": int(x), "y": int(y)},
        "target": {"domain": "equipment", "name": item_name, "menu_option": menu_option},
    }
    return dispatch(step)


def equipment_interface_open() -> bool:
    """
    Check if the equipment interface is open and visible.
    
    Returns:
        True if equipment interface is open and visible, False otherwise
    """
    
    # Check if the equipment interface widget exists and is visible
    return widget_exists(5570560)


def find_equipment_item(item_name: str):
    """
    Find an equipped item by name using direct IPC detection.
    This looks for currently equipped items, not equipment inventory items.
    
    Args:
        item_name: Name of the equipped item to find
    
    Returns:
        Equipped item data dict if found, None otherwise
    """
    
    # Get equipment data using the get_equipment command (equipped items)
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return None
    
    equipment = resp.get("equipment", {})
    slots = resp.get("slots", [])
    
    # Search through equipment slots
    for slot_data in slots:
        slot_name = clean_rs(slot_data.get("name", ""))
        if slot_name and slot_name.lower() == item_name.lower():
            return slot_data
    
    return None


def find_equipment_inventory_item(item_name: str):
    """
    Find an equipment inventory item by name using direct IPC detection.
    This looks for unequipped items in the equipment interface.
    
    Args:
        item_name: Name of the equipment inventory item to find
    
    Returns:
        Equipment inventory item data dict if found, None otherwise
    """
    
    # Get equipment inventory data using the new get_equipment_inventory command
    resp = ipc.get_equipment_inventory()
    if not resp or not resp.get("ok"):
        return None
    
    items = resp.get("items", [])
    
    # Search through equipment inventory items
    for item_data in items:
        item_name_clean = clean_rs(item_data.get("name", ""))
        if item_name_clean and item_name_clean.lower() == item_name.lower():
            return item_data
    
    return None


def get_equipment_item_bounds(slot_index: int) -> Optional[dict]:
    """
    Get the bounds of an equipment item by slot index using direct IPC detection.
    
    Args:
        slot_index: Equipment slot index (0-27)
    
    Returns:
        Dictionary with bounds (x, y, width, height) or None if not found
    """
    
    # Get equipment widget children using IPC
    resp = ipc.get_widget_children(5570560)
    if not resp or not resp.get("ok"):
        return None
    
    children = resp.get("children", [])
    
    if slot_index >= len(children):
        return None
    
    child = children[slot_index]
    child_id = child.get("id")
    if not child_id:
        return None
    
    # Get detailed info for this child widget
    child_resp = ipc.get_widget_info(child_id)
    if not child_resp or not child_resp.get("ok"):
        return None
    
    child_data = child_resp.get("data", {})
    bounds = child_data.get("bounds")
    
    if not bounds:
        return None
    
    return {
        "x": bounds.get("x", 0),
        "y": bounds.get("y", 0),
        "width": bounds.get("width", 0),
        "height": bounds.get("height", 0)
    }


def has_equipment_item(item_name: str) -> bool:
    """
    Check if a specific equipment item is equipped using direct IPC detection.
    
    Args:
        item_name: Name of the equipment item to check
    
    Returns:
        True if item is equipped, False otherwise
    """
    return find_equipment_item(item_name) is not None


def get_equipment_data() -> Optional[dict]:
    """
    Get all equipment data using direct IPC detection.
    
    Returns:
        Equipment data dict with all slots, or None if failed
    """
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return None
    
    return resp.get("equipment", {})


def get_equipment_slot(slot_name: str) -> Optional[dict]:
    """
    Get equipment data for a specific slot (e.g., "HEAD", "WEAPON", "BODY").
    
    Args:
        slot_name: Name of the equipment slot (HEAD, WEAPON, BODY, etc.)
    
    Returns:
        Equipment slot data dict, or None if not found
    """
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return None
    
    equipment = resp.get("equipment", {})
    return equipment.get(slot_name.lower())


def list_equipped_items() -> list[dict]:
    """
    Get a list of all currently equipped items.
    
    Returns:
        List of equipped item data dicts
    """
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return []
    
    slots = resp.get("slots", [])
    equipped = []
    
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            equipped.append(slot_data)
    
    return equipped


def has_equipped(item_names: Union[str, list[str]]) -> bool:
    """
    Check if you have all the specified items equipped.
    
    Args:
        item_names: Single item name or list of item names to check for 
                   (e.g., "Bronze sword" or ["Bronze sword", "Wooden shield"])
    
    Returns:
        True if all items are equipped, False otherwise
    """
    # Convert single string to list for consistent handling
    if isinstance(item_names, str):
        item_names = [item_names]
    
    if not item_names:
        return True
    
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    # Get all equipped item names
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name.lower())
    
    # Check if all requested items are equipped
    for item_name in item_names:
        if clean_rs(item_name).lower() not in equipped_names:
            return False
    
    return True


def ensure_only_equipped(
    allowed_items: list[str],
    *,
    bank_prefer: str = "bank chest",
    keep_bank_open: bool = True,
) -> bool:
    """
    Ensure that ONLY the allowed items are equipped (and that they are equipped).

    Intended usage (Blast Furnace):
        ensure_only_equipped(["Ice gloves"], bank_prefer="bank chest")

    Notes:
    - This is an equipment helper: it does NOT open/close the bank.
    - If the bank is open, it will use bank helpers (deposit_equipment / withdraw_item / bank.interact).
    - If the bank is not open, it will use inventory helpers (inventory.interact) and will NOT attempt
      to unequip extra items (since that's bank/UI-dependent in our current codebase).
    - Uses lazy imports to avoid import cycles.
    """
    allowed_items = [clean_rs(x).strip() for x in (allowed_items or []) if clean_rs(x).strip()]
    if not allowed_items:
        return True

    allowed_norm = {clean_rs(x).lower() for x in allowed_items}
    required_items = allowed_items

    def _equipped_names() -> list[str]:
        eq = list_equipped_items() or []
        out = []
        for it in eq:
            nm = clean_rs(it.get("name", "")).strip()
            if nm:
                out.append(nm.lower())
        return out

    # Fast path: already exactly what we want (no extras, and all required equipped)
    eqn = _equipped_names()
    if eqn and all(n in allowed_norm for n in eqn) and all(clean_rs(x).lower() in eqn for x in required_items):
        return True

    # Lazy imports (avoid cycles)
    from actions import bank, inventory, wait_until

    # 1) If we have extra equipment, only the bank-open path can cleanly remove it in current codebase.
    eqn = _equipped_names()
    has_extras = any(n not in allowed_norm for n in eqn)
    if has_extras:
        if not bank.is_open():
            return False
        bank.deposit_equipment()
        wait_until(lambda: all(n in allowed_norm for n in _equipped_names()), max_wait_ms=3000)

    # 2) Ensure each required item is equipped.
    for item in required_items:
        if has_equipped(item):
            continue

        # If bank is open, prefer bank helpers (withdraw if needed, then bank.interact).
        if bank.is_open():
            if not inventory.has_item(item):
                bank.withdraw_item(item, withdraw_x=1)
                wait_until(lambda: inventory.has_item(item), max_wait_ms=3000)
                if not inventory.has_item(item):
                    return False

            # Try to equip via bank inventory interactions (no slot knowledge needed).
            bank.interact(item, ["wear", "wield"])
            wait_until(lambda it=item: has_equipped(it), max_wait_ms=3000)
            if not has_equipped(item):
                return False
            continue

        # Bank not open: must be able to equip from inventory only.
        if not inventory.has_item(item):
            return False
        inventory.interact(item, "Wear", exact_match=False) or inventory.interact(item, "Wield", exact_match=False)
        wait_until(lambda it=item: has_equipped(it), max_wait_ms=3000)
        if not has_equipped(item):
            return False

    # Final verification: no extras, and all required equipped.
    eqn2 = _equipped_names()
    return bool(eqn2) and all(n in allowed_norm for n in eqn2) and all(has_equipped(x) for x in required_items)


def has_any_equipped(item_names: list[str]) -> bool:
    """
    Check if you have any of the specified items equipped.
    
    Args:
        item_names: List of item names to check for (e.g., ["Bronze sword", "Iron sword"])
    
    Returns:
        True if any item is equipped, False otherwise
    """
    if not item_names:
        return False
    
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return False
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    # Get all equipped item names
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name.lower())
    
    # Check if any requested item is equipped
    for item_name in item_names:
        if clean_rs(item_name).lower() in equipped_names:
            return True
    
    return False


def get_equipped_item_names() -> list[str]:
    """
    Get a list of all currently equipped item names.
    
    Returns:
        List of equipped item names
    """
    
    resp = ipc.get_equipment()
    if not resp or not resp.get("ok"):
        return []
    
    slots = resp.get("slots", [])
    equipped_names = []
    
    for slot_data in slots:
        if slot_data.get("id", -1) != -1:  # Item is equipped
            name = clean_rs(slot_data.get("name", ""))
            if name:
                equipped_names.append(name)
    
    return equipped_names


def get_best_weapon_for_level_in_bank(weapon_tiers: list, plan_id: str = "EQUIPMENT") -> Optional[dict]:
    """Get the best weapon available based on attack level and bank contents."""
    from .bank import get_bank_inventory, is_open
    
    # Get the best weapon for our level (ignoring bank)
    target_weapon = get_best_weapon_for_level(weapon_tiers, plan_id)
    if not target_weapon:
        print(f"[{plan_id}] No suitable weapons found for current attack level!")
        return None
    
    # Ensure bank is open before getting inventory
    if not is_open():
        print(f"[{plan_id}] Bank not open, cannot check if weapon is available")
        return None
    
    # Get available items in bank
    bank_items = get_bank_inventory()
    available_items = [item.get("name", "") for item in bank_items if item.get("name")]
    print(f"[{plan_id}] Available bank items: {available_items[:10]}...")  # Show first 10 items
    
    # Check if the target weapon is available in bank
    if target_weapon["name"] in available_items:
        print(f"[{plan_id}] Target weapon {target_weapon['name']} found in bank!")
        return target_weapon
    else:
        print(f"[{plan_id}] Target weapon {target_weapon['name']} not found in bank! Available items: {available_items}")
        return None


def get_best_weapon_for_level(weapon_tiers: list, plan_id: str = "EQUIPMENT") -> Optional[dict]:
    """Get the best weapon available based on attack level only (ignores bank contents)."""
    from .player import get_skill_level
    
    # Get current attack level
    attack_level = get_skill_level("attack")
    print(f"[{plan_id}] Getting best weapon for attack level {attack_level} (ignoring bank contents)")
    
    # Find the best weapon we can use based on level only
    for weapon in reversed(weapon_tiers):  # Start from highest tier
        print(f"[{plan_id}] Checking weapon: {weapon['name']} (req: {weapon['attack_req']})")
        if weapon["attack_req"] <= attack_level:
            print(f"[{plan_id}] Selected weapon: {weapon['name']} (req: {weapon['attack_req']}, level: {attack_level})")
            return weapon
    
    print(f"[{plan_id}] No suitable weapons found for attack level {attack_level}!")
    return None


def get_best_armor_for_level_in_bank(armor_tiers: dict, plan_id: str = "EQUIPMENT") -> Optional[dict]:
    """Get the best armor available based on defence level and bank contents."""
    from .bank import get_bank_inventory, is_open
    
    # Get the best armor for our level (ignoring bank)
    target_armor = get_best_armor_for_level(armor_tiers, plan_id)
    if not target_armor:
        print(f"[{plan_id}] No suitable armor found for current defence level!")
        return None
    
    # Ensure bank is open before getting inventory
    if not is_open():
        print(f"[{plan_id}] Bank not open, cannot check if armor is available")
        return None
    
    # Get available items in bank
    bank_items = get_bank_inventory()
    available_items = [item.get("name", "") for item in bank_items if item.get("name")]
    print(f"[{plan_id}] Available bank items: {available_items[:10]}...")  # Show first 10 items
    
    # Check which target armor pieces are available in bank
    available_armor = {}
    for armour_type, armor_item in target_armor.items():
        if armor_item["name"] in available_items:
            print(f"[{plan_id}] Target {armour_type} {armor_item['name']} found in bank!")
            available_armor[armour_type] = armor_item
        else:
            print(f"[{plan_id}] Target {armour_type} {armor_item['name']} not found in bank!")
    
    if not available_armor:
        print(f"[{plan_id}] No target armor pieces found in bank! Available items: {available_items}")
        return None
    
    print(f"[{plan_id}] Available armor: {list(available_armor.keys())}")
    return available_armor


def get_best_armor_for_level(armor_tiers: dict, plan_id: str = "EQUIPMENT") -> Optional[dict]:
    """Get the best armor available based on defence level only (ignores bank contents)."""
    from .player import get_skill_level
    
    # Get current defence level
    defence_level = get_skill_level("defence")
    print(f"[{plan_id}] Getting best armor for defence level {defence_level} (ignoring bank contents)")
    
    # Find the best armor we can use for each type based on level only
    best_armor = {}
    for armour_type, armor_list in armor_tiers.items():
        print(f"[{plan_id}] Checking {armour_type} armor...")
        for armor in reversed(armor_list):  # Start from highest tier
            print(f"[{plan_id}] Checking armor: {armor['name']} (req: {armor['defence_req']})")
            if armor["defence_req"] <= defence_level:
                print(f"[{plan_id}] Selected {armour_type}: {armor['name']} (req: {armor['defence_req']}, level: {defence_level})")
                best_armor[armour_type] = armor
                break
    
    if not best_armor:
        print(f"[{plan_id}] No suitable armor found for defence level {defence_level}!")
        return None
    
    print(f"[{plan_id}] Selected armor: {list(best_armor.keys())}")
    return best_armor


def needs_equipment_change(target_weapon: Optional[dict], target_armor_dict: Optional[dict], 
                          weapon_tiers: list, armor_tiers: dict, plan_id: str = "EQUIPMENT") -> bool:
    """Check if we need to change equipment based on what's currently equipped vs what should be equipped."""
    from .player import get_skill_level
    
    # Get current skill levels
    attack_level = get_skill_level("attack")
    defence_level = get_skill_level("defence")
    
    # Check if we have a weapon equipped
    current_weapon = None
    for weapon in weapon_tiers:
        if has_equipped(weapon["name"]):
            current_weapon = weapon
            break
    
    # Check if we have armor equipped
    current_armor = {}
    for armor_type, armor_list in armor_tiers.items():
        for armor in armor_list:
            if has_equipped(armor["name"]):
                current_armor[armor_type] = armor
                break
    
    print(f"[{plan_id}] Current equipment - Weapon: {current_weapon['name'] if current_weapon else 'None'}")
    armor_list = [f"{k}: {v['name']}" for k, v in current_armor.items()]
    print(f"[{plan_id}] Current armor: {armor_list}")
    
    # Check if we need to change weapon
    needs_weapon_change = False
    if target_weapon and current_weapon:
        if current_weapon["attack_req"] < target_weapon["attack_req"]:
            print(f"[{plan_id}] Need weapon upgrade: {current_weapon['name']} -> {target_weapon['name']}")
            needs_weapon_change = True
        elif current_weapon["name"] != target_weapon["name"]:
            print(f"[{plan_id}] Need weapon change: {current_weapon['name']} -> {target_weapon['name']}")
            needs_weapon_change = True
    elif target_weapon and not current_weapon:
        print(f"[{plan_id}] Need to equip weapon: {target_weapon['name']}")
        needs_weapon_change = True
    
    # Check if we need to change armor
    needs_armor_change = False
    if target_armor_dict:
        for armor_type, target_armor in target_armor_dict.items():
            current_armor_item = current_armor.get(armor_type)
            if current_armor_item:
                if current_armor_item["defence_req"] < target_armor["defence_req"]:
                    print(f"[{plan_id}] Need {armor_type} upgrade: {current_armor_item['name']} -> {target_armor['name']}")
                    needs_armor_change = True
                elif current_armor_item["name"] != target_armor["name"]:
                    print(f"[{plan_id}] Need {armor_type} change: {current_armor_item['name']} -> {target_armor['name']}")
                    needs_armor_change = True
            else:
                print(f"[{plan_id}] Need to equip {armor_type}: {target_armor['name']}")
                needs_armor_change = True
    
    needs_change = needs_weapon_change or needs_armor_change
    print(f"[{plan_id}] Equipment change needed: {needs_change} (weapon: {needs_weapon_change}, armor: {needs_armor_change})")
    return needs_change


def get_best_tool_for_level(tool_options: list, skill_name: str, plan_id: str = "EQUIPMENT") -> Optional[tuple]:
    """Get the best tool the player can use based on skill level only (ignores availability)."""
    from .player import get_skill_level
    
    try:
        # Get current skill level
        skill_level = get_skill_level(skill_name)
        if skill_level is None:
            skill_level = 1  # Fallback
    except:
        skill_level = 1  # Fallback
    
    # Find the best tool we can use based on skill level only
    for tool_name, skill_req, att_req, def_req in tool_options:
        if skill_level >= skill_req:
            return tool_name, skill_req, att_req, def_req
    
    raise Exception(f"No suitable tool found for {skill_name} level {skill_level}.")


def can_equip_item(item_name: str, required_attack: int = 0, required_defence: int = 0) -> bool:
    """Check if player has high enough attack or defence level to equip an item."""
    from .player import get_skill_level
    
    try:
        # Get current attack and defence levels using IPC
        attack_level = get_skill_level("attack")
        defence_level = get_skill_level("defence")
        
        if attack_level is None:
            attack_level = 1
        if defence_level is None:
            defence_level = 1
        
        # Check if we meet the requirements
        has_attack = attack_level >= required_attack
        has_defence = defence_level >= required_defence
        
        # Return True if we have both attack AND defence level requirements met
        can_equip = has_attack and has_defence
        
        print(f"[EQUIPMENT] Can equip {item_name}? Attack: {attack_level}/{required_attack}, Defence: {defence_level}/{required_defence} -> {can_equip}")
        
        return can_equip
        
    except Exception as e:
        print(f"[EQUIPMENT] Error checking equip requirements for {item_name}: {e}")
        return True  # Fallback to allowing equip if we can't check
