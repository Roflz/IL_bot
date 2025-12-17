"""
Simple rules system for plan execution stopping conditions.
Reads from CSV/stats data instead of making IPC calls.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


def check_time_rule(start_time: datetime, max_minutes: int) -> bool:
    """Check if max time has been reached."""
    if max_minutes <= 0:
        return False
    
    elapsed = datetime.now() - start_time
    return elapsed >= timedelta(minutes=max_minutes)


def check_skill_rule_from_stats(stats: Dict[str, Any], skill_name: str, target_level: int) -> bool:
    """Check if skill level has been reached using stats dict."""
    if not stats:
        return False
    
    try:
        # Stats dict uses lowercase skill names with _level suffix
        skill_key = f"{skill_name.lower()}_level"
        current_level = stats.get(skill_key, 0)
        # Convert to int if it's a string
        if isinstance(current_level, str):
            current_level = int(current_level) if current_level else 0
        else:
            current_level = int(current_level) if current_level else 0
        return current_level >= target_level
    except (ValueError, TypeError):
        return False


def check_total_level_from_stats(stats: Dict[str, Any], target_total_level: int) -> bool:
    """Check if total level has been reached using stats dict."""
    if not stats or target_total_level <= 0:
        return False
    
    try:
        # Calculate total level from all skill levels in stats
        skills = [
            'attack', 'strength', 'defence', 'ranged', 'prayer', 'magic',
            'runecraft', 'construction', 'hitpoints', 'agility', 'herblore',
            'thieving', 'crafting', 'fletching', 'slayer', 'hunter',
            'mining', 'smithing', 'fishing', 'cooking', 'firemaking',
            'woodcutting', 'farming'
        ]
        
        total = 0
        for skill in skills:
            skill_key = f"{skill}_level"
            level = stats.get(skill_key, 0)
            if isinstance(level, str):
                level = int(level) if level else 0
            else:
                level = int(level) if level else 0
            total += level
        
        return total >= target_total_level
    except (ValueError, TypeError):
        return False


def check_item_rule_from_stats(stats: Dict[str, Any], item_name: str, target_quantity: int) -> bool:
    """Check if inventory + bank has target quantity of item using stats dict."""
    if not stats:
        return False
    
    try:
        # Stats dict uses item keys like 'coins', 'logs', etc.
        # And has _bank and _inventory suffixes for breakdowns
        # The main key (e.g., 'coins') is the total (bank + inventory)
        item_key = item_name.lower().replace(' ', '_')
        
        # Try to get total from main key first
        total = stats.get(item_key, 0)
        if isinstance(total, str):
            total = int(total) if total else 0
        else:
            total = int(total) if total else 0
        
        # If total is 0, try calculating from bank + inventory
        if total == 0:
            bank_key = f"{item_key}_bank"
            inv_key = f"{item_key}_inventory"
            bank_count = stats.get(bank_key, 0)
            inv_count = stats.get(inv_key, 0)
            if isinstance(bank_count, str):
                bank_count = int(bank_count) if bank_count else 0
            else:
                bank_count = int(bank_count) if bank_count else 0
            if isinstance(inv_count, str):
                inv_count = int(inv_count) if inv_count else 0
            else:
                inv_count = int(inv_count) if inv_count else 0
            total = bank_count + inv_count
        
        return total >= target_quantity
    except (ValueError, TypeError, AttributeError):
        return False


def check_rules_from_stats(stats: Dict[str, Any], start_time: datetime, max_minutes: int = 0, 
                           skill_name: str = "", skill_level: int = 0, 
                           total_level: int = 0, item_name: str = "", item_quantity: int = 0) -> Optional[str]:
    """
    Check all rules using stats dict and return description of triggered rule, or None if none triggered.
    """
    if not stats:
        return None
    
    # Time rule
    if max_minutes > 0 and check_time_rule(start_time, max_minutes):
        return f"Time limit reached: {max_minutes} minutes"
    
    # Skill rule
    if skill_name and skill_level > 0 and check_skill_rule_from_stats(stats, skill_name, skill_level):
        return f"Skill level reached: {skill_name} level {skill_level}"
    
    # Total level rule
    if total_level > 0 and check_total_level_from_stats(stats, total_level):
        return f"Total level reached: {total_level}"
    
    # Item rule
    if item_name and item_quantity > 0 and check_item_rule_from_stats(stats, item_name, item_quantity):
        return f"Item quantity reached: {item_quantity} {item_name}"
    
    return None