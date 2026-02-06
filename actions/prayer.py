#!/usr/bin/env python3
"""
Prayer action methods for interacting with the prayer interface.
"""

import logging
import time
from typing import Optional

from actions import tab
from actions import widgets
from actions import wait_until
from helpers.tab import is_tab_open
from helpers.utils import sleep_exponential
from constants import PRAYER_WIDGETS


def ensure_prayer_tab_open() -> bool:
    """
    Ensure the prayer tab is open. Opens it if not already open.
    
    Returns:
        True if prayer tab is open (or was successfully opened), False otherwise
    """
    if not is_tab_open("PRAYER"):
        tab.open_tab("PRAYER")
        if not wait_until(lambda: is_tab_open("PRAYER"), min_wait_ms=200, max_wait_ms=2000):
            logging.warning("Failed to open prayer tab")
            return False
        sleep_exponential(0.2, 0.5, 1.2)
    return True


def flick_prayer(prayer_name: str, delay_between_clicks: tuple = (0.1, 5, 4)) -> bool:
    """
    Flick a prayer by clicking it twice (on then off).
    
    Args:
        prayer_name: Name of the prayer to flick (must be in PRAYER_WIDGETS)
        delay_between_clicks: Tuple of (min, max, beta) for sleep_exponential between clicks
    
    Returns:
        True if both clicks succeeded, False otherwise
    """
    widget_id = PRAYER_WIDGETS.get(prayer_name)
    if not widget_id:
        logging.warning(f"Prayer '{prayer_name}' not found in PRAYER_WIDGETS")
        return False
    
    # Ensure prayer tab is open before clicking
    if not ensure_prayer_tab_open():
        return False
    
    # First click (turn on)
    while not widgets.is_prayer_active(prayer_name):
        result1 = widgets.click_widget(widget_id)
        if not result1:
            logging.warning(f"Failed to click {prayer_name} prayer widget (first click)")
            return False
        if wait_until(lambda: widgets.is_prayer_active(prayer_name), max_wait_ms=1000):
            break
    
    # Sleep between clicks
    sleep_exponential(delay_between_clicks[0], delay_between_clicks[1], delay_between_clicks[2])
    
    # Ensure prayer tab is still open before second click
    if not ensure_prayer_tab_open():
        return False
    
    # Second click (turn off)
    while widgets.is_prayer_active(prayer_name):
        result2 = widgets.click_widget(widget_id)
        if not result2:
            logging.warning(f"Failed to click {prayer_name} prayer widget (second click)")
            return False
        if wait_until(lambda: not widgets.is_prayer_active(prayer_name), max_wait_ms=1000):
            break
    
    logging.info(f"Flicked {prayer_name} prayer (two clicks)")
    return True


def ensure_prayer_inactive(prayer_name: str, max_attempts: int = 10) -> bool:
    """
    Ensure a prayer is inactive by clicking it until it's off.
    
    Args:
        prayer_name: Name of the prayer to ensure is inactive
        max_attempts: Maximum number of attempts to turn off the prayer
    
    Returns:
        True if prayer is inactive (or was successfully turned off), False otherwise
    """
    widget_id = PRAYER_WIDGETS.get(prayer_name)
    if not widget_id:
        logging.warning(f"Prayer '{prayer_name}' not found in PRAYER_WIDGETS")
        return False
    
    attempt = 0
    while attempt < max_attempts:
        # Ensure prayer tab is open before checking/clicking
        if not ensure_prayer_tab_open():
            return False
        
        prayer_active = widgets.is_prayer_active(prayer_name)
        
        if prayer_active is False:
            # Prayer is off - we're done
            return True
        elif prayer_active is True:
            # Prayer is still active - click to turn it off
            logging.warning(f"{prayer_name} is still active (attempt {attempt + 1}/{max_attempts}), clicking to turn off")
            
            # Ensure prayer tab is open before clicking
            if not ensure_prayer_tab_open():
                return False
            
            result = widgets.click_widget(widget_id)
            if not result:
                logging.warning(f"Failed to click {prayer_name} prayer widget")
            time.sleep(0.1)  # Small delay between attempts
        else:
            # Could not determine prayer state
            logging.warning(f"Could not determine if {prayer_name} is active (attempt {attempt + 1}/{max_attempts})")
        
        attempt += 1
    
    # Check one more time after max attempts
    if not ensure_prayer_tab_open():
        return False
    
    prayer_active = widgets.is_prayer_active(prayer_name)
    if prayer_active is False:
        return True
    
    logging.error(f"Failed to turn off {prayer_name} after {max_attempts} attempts")
    return False


def flick_prayer_with_verification(
    prayer_name: str,
    delay_between_clicks: tuple = (0.3, 10, 4),
    verify_inactive: bool = True
) -> bool:
    """
    Flick a prayer and optionally verify it's inactive afterwards.
    
    Args:
        prayer_name: Name of the prayer to flick
        delay_between_clicks: Tuple of (min, max, beta) for sleep_exponential between clicks
        verify_inactive: If True, verify the prayer is inactive after flicking
    
    Returns:
        True if flick succeeded (and verification passed if enabled), False otherwise
    """
    if not flick_prayer(prayer_name, delay_between_clicks):
        return False
    
    if verify_inactive:
        if not wait_until(lambda: ensure_prayer_inactive(prayer_name), min_wait_ms=0, max_wait_ms=1000):
            return False
        return True
    
    return True
