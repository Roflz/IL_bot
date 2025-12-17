# retry.py (helpers)

from __future__ import annotations
from typing import Optional, Callable, Any, Dict
import time

def _clean_item_name(name: str) -> str:
    """Remove color codes and clean item name for comparison."""
    import re
    # Remove <col=...> tags
    cleaned = re.sub(r'<col=[^>]*>', '', name)
    # Remove </col> tags
    cleaned = cleaned.replace('</col>', '')
    return cleaned.strip()

def verify_interaction(expected_action: str, expected_target: str) -> bool:
    """
    Verify that the last interaction matches the expected action and target.
    
    Args:
        expected_action: Expected action type (e.g., "Use", "Talk-to", etc.)
        expected_target: Expected target name (cleaned of color codes)
    
    Returns:
        True if the interaction matches expectations
    """
    from .runtime_utils import ipc
    interaction_data = ipc.get_last_interaction() or {}
    last_interaction = interaction_data.get("interaction", {})
    if not last_interaction:
        return False
    
    # Check action
    actual_action = last_interaction.get("action", "")
    if actual_action != expected_action:
        return False
    
    # Check target (clean color codes for comparison)
    actual_target = last_interaction.get("target_name", "")
    expected_target_clean = _clean_item_name(expected_target)
    actual_target_clean = _clean_item_name(actual_target)
    
    return expected_target_clean.lower() in actual_target_clean.lower()

def retry_interaction(
    interaction_func: Callable[[], Any],
    expected_action: str,
    expected_target: str,
    max_retries: int = 3,
    retry_delay: float = 0.2,
    verification_delay: float = 0.1
) -> Any:
    """
    Retry an interaction until it succeeds or max_retries is reached.
    
    Args:
        interaction_func: Function that performs the interaction (should return UI dispatch result)
        expected_action: Expected action type for verification
        expected_target: Expected target name for verification
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retry attempts
        verification_delay: Delay after interaction before verification
    
    Returns:
        Result of successful interaction or None if all retries failed
    """
    for attempt in range(max_retries):
        # Perform the interaction
        result = interaction_func()
        
        if result is None:
            continue
        
        # Wait for the game to register the interaction
        from .utils import sleep_exponential
        sleep_exponential(verification_delay * 0.8, verification_delay * 1.2, 1.0)
        
        # Verify the interaction was successful
        if verify_interaction(expected_action, expected_target):
            return result
        
        # If verification failed, wait before retrying
        if attempt < max_retries - 1:  # Don't wait after the last attempt
            from .utils import sleep_exponential
            sleep_exponential(retry_delay * 0.8, retry_delay * 1.2, 1.0)
    
    return None

def retry_click_interaction(
    click_func: Callable[[], Any],
    expected_action: str,
    expected_target: str,
    max_retries: int = 3,
    retry_delay: float = 0.2,
    verification_delay: float = 0.1
) -> Any:
    """
    Convenience wrapper for retry_interaction specifically for click interactions.
    """
    return retry_interaction(
        click_func,
        expected_action,
        expected_target,
        max_retries,
        retry_delay,
        verification_delay
    )
