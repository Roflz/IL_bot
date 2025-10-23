"""
Simple rules system for plan execution stopping conditions.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta
from ..helpers.runtime_utils import ipc
from ..actions.player import get_skill_level, check_total_level


def check_time_rule(start_time: datetime, max_minutes: int) -> bool:
    """Check if max time has been reached."""
    if max_minutes <= 0:
        return False
    
    elapsed = datetime.now() - start_time
    return elapsed >= timedelta(minutes=max_minutes)


def check_skill_rule(skill_name: str, target_level: int) -> bool:
    """Check if skill level has been reached."""
    try:
        current_level = get_skill_level(skill_name)
        return current_level and current_level >= target_level
    except Exception:
        return False


def check_item_rule(item_name: str, target_quantity: int) -> bool:
    """Check if inventory + bank has target quantity of item."""
    try:
        # Check inventory
        inventory_resp = ipc.get_inventory()
        inventory_quantity = 0
        if inventory_resp and inventory_resp.get("ok"):
            for slot in inventory_resp.get("slots", []):
                if slot.get("itemName", "").lower() == item_name.lower():
                    inventory_quantity += int(slot.get("quantity", 0))
        
        # Check bank
        bank_resp = ipc.get_bank_items()
        bank_quantity = 0
        if bank_resp and bank_resp.get("ok"):
            for slot in bank_resp.get("slots", []):
                if slot.get("itemName", "").lower() == item_name.lower():
                    bank_quantity += int(slot.get("quantity", 0))
        
        total_quantity = inventory_quantity + bank_quantity
        return total_quantity >= target_quantity
    except Exception:
        return False


def check_rules(start_time: datetime, max_minutes: int = 0, skill_name: str = "", skill_level: int = 0, 
                total_level: int = 0, item_name: str = "", item_quantity: int = 0) -> Optional[str]:
    """
    Check all rules and return description of triggered rule, or None if none triggered.
    """
    # Time rule
    if max_minutes > 0 and check_time_rule(start_time, max_minutes):
        return f"Time limit reached: {max_minutes} minutes"
    
    # Skill rule
    if skill_name and skill_level > 0 and check_skill_rule(skill_name, skill_level):
        return f"Skill level reached: {skill_name} level {skill_level}"
    
    # Total level rule
    if total_level > 0 and check_total_level(total_level):
        return f"Total level reached: {total_level}"
    
    # Item rule
    if item_name and item_quantity > 0 and check_item_rule(item_name, item_quantity):
        return f"Item quantity reached: {item_quantity} {item_name}"
    
    return None